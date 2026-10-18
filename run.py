"""Multibagger analogue scan.

Funnel:
  1. Rebuild each seed multibagger's fingerprint from SEC filings for its pre-run year.
  2. Compute the same fingerprint for every listed US filer's latest year.
  3. Score similarity across economics, structure, dependencies, size and KPIs.
  4. Pull market 'intercept' signals for the top N.
  5. Optionally have Claude read the 10-K of the top K for dependencies and thesis.
  6. Write data/scan.json for the dashboard.

Usage:
  python agent/run.py                 # full run (GitHub Actions)
  python agent/run.py --mode list --tickers CELH,ONON,HIMS --no-llm   # quick local test
  python agent/run.py --demo          # synthetic data, no network
"""
import argparse
import datetime as dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import ALL_FEATURES, FEATURE_GROUPS, LABELS, Scaler, compare, compute, inflections  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, "config", name)) as f:
        return json.load(f)


def present(feats):
    return sum(1 for k in ALL_FEATURES if feats.get(k) is not None)


def build_fingerprints(sec, seeds, ticker_to_cik):
    from sec_data import annual_series
    out = []
    for s in seeds:
        cik = ticker_to_cik.get(s["ticker"])
        fp = compute(annual_series(sec.facts(cik)), s["start_year"]) if cik else None
        if not fp:
            print(f"  skipped seed {s['ticker']} {s['start_year']}: not enough SEC data")
            continue
        out.append({**s, "id": f"{s['ticker']}-{s['start_year']}", "cik": cik, **fp})
        print(f"  seed {s['ticker']} {s['start_year']}: {present(fp['features'])}/{len(ALL_FEATURES)} features")
    return out


def scan_universe(sec, tickers, settings, mode, only):
    from sec_data import annual_series
    u, this_year = settings["universe"], dt.date.today().year
    rows = []
    source = (sec.facts(c) for c in only) if mode == "list" else sec.iter_bulk_facts()
    for i, facts in enumerate(source):
        if not facts:
            continue
        cik = int(facts.get("cik", 0))
        if cik not in tickers:
            continue
        ann = annual_series(facts)
        if not ann["revenue"]:
            continue
        latest = max(ann["revenue"])
        if latest < this_year - u["max_staleness_years"]:
            continue
        fp = compute(ann, latest)
        if not fp or present(fp["features"]) < u["min_features_present"]:
            continue
        rev, pf = fp["raw"]["revenue"], fp["raw"]["public_float"]
        if not (u["min_revenue_usd"] <= rev <= u["max_revenue_usd"]):
            continue
        if pf and pf > u["max_public_float_usd"]:
            continue
        rows.append({"cik": cik, **tickers[cik], "year": latest, **fp, "events": inflections(fp)})
        if i % 1000 == 0:
            print(f"  scanned {i} filers, {len(rows)} in range")
    return rows


def score(cands, fps, weights):
    scaler = Scaler(rows=[c["features"] for c in cands])
    for fp in fps:
        fp["z"] = scaler.z(fp["features"])
    for c in cands:
        c["z"] = scaler.z(c["features"])
        matches = compare(c["z"], fps, weights)
        c["analogues"] = matches[:3]
        c["similarity"] = matches[0]["overall"] if matches else 0
        c["groups"] = matches[0]["groups"] if matches else {}
    return scaler


def composite(c, w):
    comps = dict(c["groups"])
    comps["intercept"] = (c.get("intercept") or {}).get("score")
    comps["filing_review"] = (c.get("filing") or {}).get("score")
    num = sum(w[k] * v for k, v in comps.items() if v is not None and k in w)
    den = sum(w[k] for k, v in comps.items() if v is not None and k in w)
    return round(num / den, 1) if den else 0


def write(payload, demo):
    path = os.path.join(ROOT, "data", "demo" if demo else "", "scan.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), default=lambda o: None)
    print(f"Wrote {path}")


def run(args):
    settings, seeds = load("settings.json"), load("seeds.json")["seeds"]
    weights = settings["weights"]
    from sec_data import SEC
    import signals
    import llm

    sec = SEC(os.environ.get("SEC_USER_AGENT") or settings["sec_user_agent"])
    print("Loading ticker map...")
    tickers = sec.tickers()
    t2c = {v["ticker"]: k for k, v in tickers.items()}

    print("Building seed fingerprints...")
    fps = build_fingerprints(sec, seeds, t2c)
    if not fps:
        raise SystemExit("No seed fingerprints could be built.")
    seed_ciks = {fp["cik"] for fp in fps}

    print("Scanning universe...")
    only = [t2c[t] for t in (args.tickers or "").split(",") if t in t2c]
    cands = [c for c in scan_universe(sec, tickers, settings, args.mode, only) if c["cik"] not in seed_ciks]
    print(f"{len(cands)} companies with complete fingerprints")
    scaler = score(cands, fps, weights)

    m = settings["market"]
    cands.sort(key=lambda c: c["similarity"], reverse=True)
    shortlist = cands[:m["top_n_for_market_data"]]
    print(f"Pulling market signals for {len(shortlist)}...")
    bench = signals.benchmark_return()
    kept = []
    for c in shortlist:
        md = signals.market_data(c["ticker"])
        cap, adv = md.get("market_cap"), md.get("avg_dollar_volume")
        if cap and not (m["min_market_cap_usd"] <= cap <= m["max_market_cap_usd"]):
            continue
        if adv is not None and adv < m["min_avg_dollar_volume"]:
            continue
        s, hits = signals.intercept_score(md, c["events"], bench)
        c["market"], c["intercept"] = md, {"score": s, "signals": hits}
        c["score"] = composite(c, weights)
        kept.append(c)
    kept.sort(key=lambda c: c["score"], reverse=True)

    cfg = settings["llm"]
    if cfg["enabled"] and not args.no_llm and os.environ.get("ANTHROPIC_API_KEY"):
        seed_by_id = {fp["id"]: fp for fp in fps}
        print(f"Claude reviewing 10-Ks for top {cfg['top_k']}...")
        for c in kept[:cfg["top_k"]]:
            rev = llm.review(sec, c, seed_by_id[c["analogues"][0]["seed"]], cfg["model"], cfg["max_chars"])
            if rev:
                rev["score"], rev["risk"] = llm.filing_score(rev)
                c["filing"] = rev
                c["score"] = composite(c, weights)
        kept.sort(key=lambda c: c["score"], reverse=True)

    write(payload(kept[:settings["output_count"]], fps, scaler, weights, len(cands), False), False)


def payload(cands, fps, scaler, weights, universe, demo):
    keep_fp = ["id", "ticker", "start_year", "archetype", "story", "features", "z"]
    keep_c = ["ticker", "name", "cik", "year", "features", "z", "raw", "events", "analogues",
              "similarity", "groups", "market", "intercept", "filing", "score"]
    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "demo": demo,
        "universe_scored": universe,
        "weights": weights,
        "feature_groups": FEATURE_GROUPS,
        "labels": LABELS,
        "scaler": scaler.params,
        "fingerprints": [{k: fp.get(k) for k in keep_fp} for fp in fps],
        "candidates": [{k: c.get(k) for k in keep_c} for c in cands],
    }


# ---------- demo mode: synthetic, clearly fictional data to preview the dashboard ----------

def demo():
    rnd = random.Random(7)
    seeds = load("seeds.json")["seeds"][:8]
    weights = load("settings.json")["weights"]

    def fake():
        g = rnd.uniform(-0.05, 0.6)
        gp = rnd.uniform(-0.05, 0.5)
        fcf = rnd.uniform(-0.2, 0.25)
        return {
            "gross_margin": rnd.uniform(0.2, 0.8), "gm_delta": rnd.uniform(-0.05, 0.08),
            "op_margin": rnd.uniform(-0.25, 0.3), "op_margin_delta": rnd.uniform(-0.08, 0.12),
            "fcf_margin": fcf, "roic": rnd.uniform(-0.3, 0.5), "share_change": rnd.uniform(-0.04, 0.15),
            "net_cash_to_rev": rnd.uniform(-0.5, 1.0), "debt_to_equity": rnd.uniform(0, 1.5),
            "capex_intensity": rnd.uniform(0.01, 0.15), "rnd_intensity": rnd.uniform(0, 0.3),
            "sga_intensity": rnd.uniform(0.1, 0.5), "log_revenue": rnd.uniform(7.5, 9.3),
            "log_float": rnd.uniform(8, 10), "rev_growth": g, "rev_growth_prev": gp, "rev_accel": g - gp,
            "rule_of_40": g + fcf, "incremental_margin": rnd.uniform(-0.3, 0.6),
        }

    fps = [{**s, "id": f"{s['ticker']}-{s['start_year']}", "features": fake()} for s in seeds]
    words = ["Harbour", "Lumen", "Quarry", "Tessel", "Brightwater", "Kestrel", "Orbit", "Granite", "Vela",
             "Saltmarsh", "Copperline", "Halcyon", "Meridian", "Fernhill", "Arbor", "Pinecrest", "Riverbend",
             "Northgate", "Solace", "Ironwood", "Bluefin", "Larkspur", "Cobalt", "Tidewater"]
    kinds = ["Robotics", "Diagnostics", "Payments", "Foods", "Networks", "Energy", "Software", "Logistics"]
    cands = []
    for i, w in enumerate(words):
        f = fake()
        fp = {"features": f, "raw": {"revenue": 10 ** f["log_revenue"], "public_float": 10 ** f["log_float"],
                                     "op_income": f["op_margin"] * 10 ** f["log_revenue"],
                                     "op_income_prev": (f["op_margin"] - f["op_margin_delta"]) * 10 ** f["log_revenue"],
                                     "fcf_margin_prev": f["fcf_margin"] - rnd.uniform(-0.1, 0.2), "year": 2025}}
        cands.append({"ticker": f"DEMO{i + 1}", "name": f"{w} {rnd.choice(kinds)} (fictional)", "cik": 0,
                      "year": 2025, **fp, "events": inflections(fp)})
    scaler = Scaler(rows=[c["features"] for c in cands] * 2)
    for fp in fps:
        fp["z"] = scaler.z(fp["features"])
    for c in cands:
        c["z"] = scaler.z(c["features"])
        mt = compare(c["z"], fps, weights)
        c["analogues"], c["similarity"], c["groups"] = mt[:3], mt[0]["overall"], mt[0]["groups"]
        import signals
        md = {"market_cap": c["raw"]["public_float"] * rnd.uniform(1.1, 2), "insider_own": rnd.uniform(0, 0.35),
              "institutional_own": rnd.uniform(0.2, 0.95), "analysts": rnd.randint(0, 14),
              "ret_12m": rnd.uniform(-0.4, 1.2), "pct_from_high": rnd.uniform(-0.6, 0),
              "insider_net_shares_6m": rnd.choice([-50000, 0, 120000]), "short_float": rnd.uniform(0, 0.2),
              "sector": "Demo sector", "industry": "Demo industry"}
        s, hits = signals.intercept_score(md, c["events"], 0.15)
        c["market"], c["intercept"] = md, {"score": s, "signals": hits}
        c["score"] = composite(c, weights)
    cands.sort(key=lambda c: c["score"], reverse=True)
    import llm
    for c in cands[:8]:
        rev = {"customer_concentration": rnd.randint(1, 8), "top_customer_share_pct": rnd.choice([None, 12, 31]),
               "supplier_dependence": rnd.randint(1, 7), "regulatory_dependence": rnd.randint(0, 6),
               "platform_dependence": rnd.randint(0, 8), "secular_tailwind": rnd.randint(4, 9),
               "business_model": "Demo text: in a real run Claude summarises the business model from the 10-K.",
               "moat": "Demo text: what the filing suggests is hard to copy.",
               "key_dependencies": ["Demo: largest customer", "Demo: single-source component"],
               "red_flags": ["Demo: receivables growing faster than revenue"],
               "analogue_comment": "Demo text comparing this company with its closest historical analogue.",
               "thesis": "Demo text: what would have to go right for a very large return.", "filing_url": None}
        rev["score"], rev["risk"] = llm.filing_score(rev)
        c["filing"] = rev
        c["score"] = composite(c, weights)
    cands.sort(key=lambda c: c["score"], reverse=True)
    write(payload(cands, fps, scaler, weights, len(cands), True), True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--mode", choices=["bulk", "list"], default=None)
    p.add_argument("--tickers", help="comma-separated tickers for --mode list")
    p.add_argument("--no-llm", action="store_true")
    a = p.parse_args()
    if a.demo:
        demo()
    else:
        a.mode = a.mode or load("settings.json")["universe"]["mode"]
        run(a)
