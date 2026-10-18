"""Reads the 10-K business section and scores the dependencies that financial
statements can't show: customer and supplier concentration, regulation, platforms."""
import json
import os
import re

import requests

API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = (
    "You are a buy-side analyst screening small and mid-cap companies for multibagger potential. "
    "You read the filing text provided and answer only from it. When the filing does not say, use null. "
    "Respond with a single JSON object and nothing else."
)

PROMPT = """Company: {name} ({ticker})
Closest historical analogue by financial fingerprint: {analogue} in {year} ({archetype}: {story}).

Score each from 0 to 10 using only the filing text below.
Return exactly this JSON shape:
{{
  "customer_concentration": 0-10 (10 = one or two customers dominate revenue),
  "top_customer_share_pct": number or null,
  "supplier_dependence": 0-10 (10 = single-source critical inputs),
  "regulatory_dependence": 0-10 (10 = economics hinge on a permit, subsidy, or rule),
  "platform_dependence": 0-10 (10 = relies on another company's platform or channel to reach customers),
  "secular_tailwind": 0-10 (10 = strong, durable demand growth in its end market),
  "business_model": "one sentence",
  "moat": "one sentence on what is hard to copy, or 'none evident'",
  "key_dependencies": ["up to 4 short items"],
  "red_flags": ["up to 4 short items"],
  "analogue_comment": "two sentences: how it resembles and differs from the analogue",
  "thesis": "two or three sentences: what would have to go right for a very large return"
}}

FILING TEXT:
{text}"""


def review(sec, cand, seed, model, max_chars):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    text, url = sec.latest_10k_text(cand["cik"], max_chars)
    if not text:
        return None
    body = {
        "model": model,
        "max_tokens": 1500,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": PROMPT.format(
            name=cand["name"], ticker=cand["ticker"], analogue=seed["ticker"], year=seed["start_year"],
            archetype=seed["archetype"], story=seed["story"], text=text)}],
    }
    r = requests.post(API_URL, json=body, timeout=180, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    if r.status_code != 200:
        print(f"  Claude review failed for {cand['ticker']}: {r.status_code} {r.text[:200]}")
        return None
    out = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", out, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not data:
        return None
    data["filing_url"] = url
    return data


def filing_score(rev):
    """0-100: low dependency risk and a strong tailwind score higher."""
    keys = ["customer_concentration", "supplier_dependence", "regulatory_dependence", "platform_dependence"]
    vals = [rev[k] for k in keys if isinstance(rev.get(k), (int, float))]
    if not vals:
        return None, None
    risk = 10 * sum(vals) / len(vals)
    tail = rev.get("secular_tailwind")
    tail = 10 * tail if isinstance(tail, (int, float)) else 50
    return round(0.6 * (100 - risk) + 0.4 * tail, 1), round(risk, 1)
