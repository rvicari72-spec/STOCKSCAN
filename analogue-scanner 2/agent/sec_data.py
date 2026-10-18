"""SEC EDGAR access. Free, no API key, but SEC requires a descriptive User-Agent
with a contact email and asks for <= 10 requests/second."""
import html
import json
import os
import re
import time
import zipfile

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
BULK_FACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# Ordered fallbacks: the first tag that has a value for a year wins.
DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
}
INSTANT_TAGS = {
    "equity": ["StockholdersEquity"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}
SHARE_TAGS = {"shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"]}

_ANNUAL = re.compile(r"^CY(\d{4})$")
_YEAR_END = re.compile(r"^CY(\d{4})Q4I$")


class SEC:
    def __init__(self, user_agent):
        if not user_agent or "example.com" in user_agent:
            raise SystemExit("Set SEC_USER_AGENT (e.g. 'Jane Doe jane@company.com') - SEC blocks anonymous clients.")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0

    def _get(self, url, stream=False):
        for attempt in range(4):
            wait = 0.12 - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.s.get(url, timeout=60, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            time.sleep(2 ** attempt)
        return None

    def get_json(self, url):
        r = self._get(url)
        return r.json() if r is not None else None

    def tickers(self):
        """{cik: {"ticker":..., "name":...}} for every currently listed company."""
        raw = self.get_json(TICKERS_URL) or {}
        out = {}
        for row in raw.values():
            out.setdefault(int(row["cik_str"]), {"ticker": row["ticker"], "name": row["title"]})
        return out

    def facts(self, cik):
        return self.get_json(FACTS_URL.format(cik=cik))

    def iter_bulk_facts(self, path="companyfacts.zip"):
        """Downloads the nightly bulk archive (~1+ GB) and yields each company's facts JSON."""
        if not os.path.exists(path):
            print("Downloading SEC bulk companyfacts archive...")
            r = self._get(BULK_FACTS_URL, stream=True)
            if r is None:
                raise SystemExit("Could not download companyfacts.zip - check the SEC bulk data page for its current URL.")
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    try:
                        yield json.loads(z.read(name))
                    except (ValueError, KeyError):
                        continue

    def latest_10k_text(self, cik, max_chars=90000):
        """Plain text of the most recent 10-K, starting at Item 1 (Business)."""
        subs = self.get_json(SUBMISSIONS_URL.format(cik=cik))
        if not subs:
            return None, None
        recent = subs.get("filings", {}).get("recent", {})
        for i, form in enumerate(recent.get("form", [])):
            if form == "10-K":
                acc = recent["accessionNumber"][i].replace("-", "")
                url = FILING_URL.format(cik=cik, acc=acc, doc=recent["primaryDocument"][i])
                r = self._get(url)
                if r is None:
                    return None, url
                return _extract_business_section(r.text, max_chars), url
        return None, None


def _extract_business_section(raw_html, max_chars):
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", html.unescape(text))
    # Skip the table of contents: take the first "Item 1. Business" not followed closely by "Item 2".
    for m in re.finditer(r"(?i)item\s*1\.?\s*business", text):
        if not re.search(r"(?i)item\s*2\.?\s*properties", text[m.start():m.start() + 4000]):
            return text[m.start():m.start() + max_chars]
    return text[:max_chars]


def annual_series(facts):
    """Collapse companyfacts JSON to {field: {year: value}} using SEC calendar frames."""
    gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    dei = (facts or {}).get("facts", {}).get("dei", {})
    out = {}

    def collect(tag_map, unit, pattern):
        for field, tags in tag_map.items():
            series = {}
            for tag in tags:
                for item in gaap.get(tag, {}).get("units", {}).get(unit, []):
                    m = pattern.match(item.get("frame", ""))
                    if m:
                        series.setdefault(int(m.group(1)), float(item["val"]))
            out[field] = series

    collect(DURATION_TAGS, "USD", _ANNUAL)
    collect(INSTANT_TAGS, "USD", _YEAR_END)
    collect(SHARE_TAGS, "shares", _ANNUAL)

    # Public float is reported on the 10-K cover page (measured mid-year), not split-sensitive.
    pf = {}
    for item in dei.get("EntityPublicFloat", {}).get("units", {}).get("USD", []):
        if item.get("form", "").startswith("10-K") and item.get("end"):
            pf[int(item["end"][:4])] = float(item["val"])
    out["public_float"] = pf

    gp = out["gross_profit"]
    for y, rev in out["revenue"].items():
        if y not in gp and y in out["cost_of_revenue"]:
            gp[y] = rev - out["cost_of_revenue"][y]
    return out
