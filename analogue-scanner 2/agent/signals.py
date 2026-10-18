"""Market-side 'intercept' signals: ways to catch a future multibagger early that
don't show up in the financial statements - ownership, coverage, price behaviour."""
import math

try:
    import yfinance as yf
except ImportError:  # demo mode works without it
    yf = None


def _num(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def benchmark_return(symbol="SPY"):
    try:
        h = yf.Ticker(symbol).history(period="1y")["Close"]
        return float(h.iloc[-1] / h.iloc[0] - 1)
    except Exception:
        return 0.0


def market_data(ticker):
    t = yf.Ticker(ticker.replace(".", "-"))
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        pass
    m = {
        "market_cap": _num(info.get("marketCap")),
        "price": _num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "insider_own": _num(info.get("heldPercentInsiders")),
        "institutional_own": _num(info.get("heldPercentInstitutions")),
        "analysts": _num(info.get("numberOfAnalystOpinions")),
        "short_float": _num(info.get("shortPercentOfFloat")),
        "avg_volume": _num(info.get("averageVolume")),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "ret_12m": None, "pct_from_high": None, "insider_net_shares_6m": None,
    }
    try:
        close = t.history(period="1y")["Close"]
        if len(close) > 50:
            m["ret_12m"] = float(close.iloc[-1] / close.iloc[0] - 1)
            m["pct_from_high"] = float(close.iloc[-1] / close.max() - 1)
            m["price"] = m["price"] or float(close.iloc[-1])
    except Exception:
        pass
    try:
        ip = t.insider_purchases
        col = ip.columns[0]
        row = ip[ip[col].astype(str).str.contains("Net Shares", case=False)]
        if len(row):
            m["insider_net_shares_6m"] = _num(row["Shares"].iloc[0])
    except Exception:
        pass
    m["avg_dollar_volume"] = (m["avg_volume"] or 0) * (m["price"] or 0) or None
    return m


def intercept_score(m, fundamental_events, bench_ret):
    """0-100 score plus the list of signals that fired."""
    hits = []
    if (m.get("insider_own") or 0) >= 0.10:
        hits.append(("Insiders own 10%+ (founder or owner-operator alignment)", 15))
    io = m.get("institutional_own")
    if io is not None and io < 0.50:
        hits.append(("Institutions own under half (under-owned)", 10))
    an = m.get("analysts")
    if an is not None and an <= 3:
        hits.append(("Three or fewer covering analysts (under-followed)", 15))
    if m.get("ret_12m") is not None and m["ret_12m"] - bench_ret > 0.20:
        hits.append(("Beating the S&P 500 by 20+ points over 12 months", 15))
    if m.get("pct_from_high") is not None and m["pct_from_high"] > -0.15:
        hits.append(("Within 15% of its 52-week high", 10))
    if (m.get("insider_net_shares_6m") or 0) > 0:
        hits.append(("Net insider buying in the last 6 months", 15))
    for ev in fundamental_events[:4]:
        hits.append((ev, 5))
    return min(100, sum(p for _, p in hits)), [h for h, _ in hits]
