"""Turns annual financials into a fingerprint and compares it to past multibaggers."""
import math

import numpy as np

# The five dimensions the screen matches on. Every feature is computable from SEC
# filings for any year since ~2009, so seeds and live candidates are measured identically.
FEATURE_GROUPS = {
    "economics": ["gross_margin", "gm_delta", "op_margin", "op_margin_delta", "fcf_margin", "roic"],
    "structure": ["share_change", "net_cash_to_rev", "debt_to_equity"],
    "dependencies": ["capex_intensity", "rnd_intensity", "sga_intensity"],
    "size": ["log_revenue", "log_float"],
    "kpis": ["rev_growth", "rev_growth_prev", "rev_accel", "rule_of_40", "incremental_margin"],
}
ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]

LABELS = {
    "gross_margin": "Gross margin", "gm_delta": "Gross margin change", "op_margin": "Operating margin",
    "op_margin_delta": "Operating margin change", "fcf_margin": "Free cash flow margin", "roic": "Return on invested capital",
    "share_change": "Share count change", "net_cash_to_rev": "Net cash / revenue", "debt_to_equity": "Debt / equity",
    "capex_intensity": "Capex / revenue", "rnd_intensity": "R&D / revenue", "sga_intensity": "SG&A / revenue",
    "log_revenue": "Revenue", "log_float": "Public float",
    "rev_growth": "Revenue growth", "rev_growth_prev": "Prior-year revenue growth", "rev_accel": "Growth acceleration",
    "rule_of_40": "Growth + FCF margin", "incremental_margin": "Incremental operating margin",
}


def _ratio(a, b):
    if a is None or b is None or b == 0 or (isinstance(b, float) and math.isnan(b)):
        return None
    return a / b


def _sub(a, b):
    return None if a is None or b is None else a - b


def compute(ann, year):
    """Feature dict for `year`, or None if three years of revenue are not available."""
    g = lambda field, y: ann.get(field, {}).get(y)
    r0, r1, r2 = g("revenue", year), g("revenue", year - 1), g("revenue", year - 2)
    if not r0 or not r1 or not r2 or r0 <= 0 or r1 <= 0 or r2 <= 0:
        return None

    gm0, gm1 = _ratio(g("gross_profit", year), r0), _ratio(g("gross_profit", year - 1), r1)
    om0, om1 = _ratio(g("op_income", year), r0), _ratio(g("op_income", year - 1), r1)
    cfo, capex = g("cfo", year), g("capex", year) or 0.0
    fcf_m = _ratio(None if cfo is None else cfo - capex, r0)
    cfo1, capex1 = g("cfo", year - 1), g("capex", year - 1) or 0.0
    fcf_m1 = _ratio(None if cfo1 is None else cfo1 - capex1, r1)
    eq, debt, cash = g("equity", year), g("lt_debt", year) or 0.0, g("cash", year)
    invested = None if eq is None or cash is None else eq + debt - cash
    oi0, oi1 = g("op_income", year), g("op_income", year - 1)
    growth, growth_prev = r0 / r1 - 1, r1 / r2 - 1
    pf = g("public_float", year) or g("public_float", year + 1)

    f = {
        "gross_margin": gm0,
        "gm_delta": _sub(gm0, gm1),
        "op_margin": om0,
        "op_margin_delta": _sub(om0, om1),
        "fcf_margin": fcf_m,
        "roic": _ratio(oi0, invested) if invested and invested > 0 else None,
        "share_change": g("shares", year) / g("shares", year - 1) - 1
        if g("shares", year) and g("shares", year - 1) else None,
        "net_cash_to_rev": _ratio(None if cash is None else cash - debt, r0),
        "debt_to_equity": _ratio(debt, eq) if eq and eq > 0 else None,
        "capex_intensity": _ratio(capex, r0) if g("capex", year) is not None else None,
        "rnd_intensity": _ratio(g("rnd", year), r0),
        "sga_intensity": _ratio(g("sga", year), r0),
        "log_revenue": math.log10(r0),
        "log_float": math.log10(pf) if pf and pf > 0 else None,
        "rev_growth": growth,
        "rev_growth_prev": growth_prev,
        "rev_accel": growth - growth_prev,
        "rule_of_40": None if fcf_m is None else growth + fcf_m,
        "incremental_margin": _ratio(_sub(oi0, oi1), r0 - r1) if oi0 is not None and oi1 is not None and r0 > r1 else None,
    }
    raw = {"revenue": r0, "op_income": oi0, "op_income_prev": oi1, "fcf_margin_prev": fcf_m1,
           "public_float": pf, "year": year}
    return {"features": f, "raw": raw}


def inflections(fp):
    """Fundamental 'intercept' events: the moments multibaggers often show before re-rating."""
    f, raw, out = fp["features"], fp["raw"], []
    if raw["op_income"] is not None and raw["op_income_prev"] is not None and raw["op_income"] > 0 >= raw["op_income_prev"]:
        out.append("Turned operating-profitable this year")
    if (f["fcf_margin"] or 0) > 0 and raw["fcf_margin_prev"] is not None and raw["fcf_margin_prev"] <= 0:
        out.append("Became free-cash-flow positive")
    if (f["rev_accel"] or 0) > 0.05 and f["rev_growth"] > 0.15:
        out.append("Revenue growth is accelerating")
    if (f["gm_delta"] or 0) > 0.02 and (f["op_margin_delta"] or 0) > 0.03:
        out.append("Gross and operating margins both expanding")
    if (f["incremental_margin"] or 0) > 0.25 and f["rev_growth"] > 0.10:
        out.append("Strong operating leverage on new revenue")
    if f["share_change"] is not None and f["share_change"] < -0.01:
        out.append("Shrinking share count")
    return out


class Scaler:
    """Robust z-scores (median / IQR) fitted on the live universe, clipped to +/-3."""

    def __init__(self, rows=None, params=None):
        if params:
            self.params = params
            return
        self.params = {}
        for feat in ALL_FEATURES:
            vals = np.array([r[feat] for r in rows if r.get(feat) is not None], dtype=float)
            if len(vals) < 20:
                self.params[feat] = [0.0, 1.0]
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            self.params[feat] = [float(med), float(q3 - q1) or 1.0]

    def z(self, feats):
        out = {}
        for feat in ALL_FEATURES:
            v = feats.get(feat)
            if v is None:
                out[feat] = None
            else:
                med, iqr = self.params[feat]
                out[feat] = float(np.clip((v - med) / iqr, -3, 3))
        return out


def group_similarity(zc, zf, feats, min_present=1):
    diffs = [(zc[x] - zf[x]) ** 2 for x in feats if zc.get(x) is not None and zf.get(x) is not None]
    if len(diffs) < min_present:
        return None
    d2 = sum(diffs) / len(diffs)
    return round(100 * math.exp(-0.5 * d2), 1)


def compare(zc, fingerprints, weights):
    """Similarity of one candidate to every fingerprint, best first."""
    results = []
    for fp in fingerprints:
        sims = {g: group_similarity(zc, fp["z"], feats) for g, feats in FEATURE_GROUPS.items()}
        num = sum(weights.get(g, 1) * s for g, s in sims.items() if s is not None)
        den = sum(weights.get(g, 1) for g, s in sims.items() if s is not None)
        if den:
            results.append({"seed": fp["id"], "overall": round(num / den, 1), "groups": sims})
    return sorted(results, key=lambda r: r["overall"], reverse=True)
