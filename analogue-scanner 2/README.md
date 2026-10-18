# Analogue Scanner

A weekly agent that finds listed US companies whose financial fingerprint today resembles past multibaggers in the year **before** their run, then ranks them with the early "intercept" signals that let you catch such companies before the market does. Results are published as a static dashboard on GitHub Pages.

Research tool only, not investment advice.

## How it works

```
GitHub Actions (weekly)                                   GitHub Pages
┌──────────────────────────────────────────────────┐     ┌──────────────┐
│ 1 Seed fingerprints  SEC XBRL, pre-run year       │     │  index.html  │
│ 2 Universe scan      every US filer, latest FY    │ ──► │  reads       │
│ 3 Similarity         5 dimensions vs each seed    │     │  data/       │
│ 4 Intercept signals  top 150, Yahoo market data   │     │  scan.json   │
│ 5 10-K review        top 25, Claude (optional)    │     └──────────────┘
│ 6 Commit data/scan.json + dated history copy      │
└──────────────────────────────────────────────────┘
```

**Fingerprint dimensions** (all computed from SEC filings, so seeds from 2012 and companies today are measured the same way):

| Dimension | Features |
|---|---|
| Economics | gross margin and its change, operating margin and its change, FCF margin, ROIC |
| Structure | share count change, net cash / revenue, debt / equity |
| Dependencies | capex, R&D and SG&A intensity (plus the 10-K review below) |
| Size | revenue, public float |
| KPIs | revenue growth, prior-year growth, acceleration, growth + FCF margin, incremental operating margin |

Each feature is converted to a robust z-score against the current universe (median and interquartile range, clipped at ±3), so similarity means "sits in the same place relative to other listed companies".

**Intercept signals** — the other ways to catch these companies early:
- Fundamental inflections: first operating profit, first positive FCF, revenue acceleration, simultaneous gross and operating margin expansion, strong incremental margins, buybacks starting
- Ownership: insiders own 10%+, institutions own less than half, net insider buying
- Discovery: three or fewer analysts covering
- Price: beating the S&P 500 by 20+ points, within 15% of 52-week high

**10-K dependency review** (optional): Claude reads Item 1 and risk factors from the latest 10-K and scores customer concentration, supplier dependence, regulatory and platform dependence, and secular tailwind, then writes a comparison with the analogue and a short thesis.

## Setup

1. Create a **public** repository (Pages and scheduled Actions on private repos need a paid plan) and push these files.
2. **Settings → Secrets and variables → Actions**, add:
   - `SEC_USER_AGENT` — your name and email, e.g. `Ric Example ric@example.com` (SEC rejects anonymous traffic)
   - `ANTHROPIC_API_KEY` — optional, enables the 10-K review
3. **Settings → Pages** → deploy from branch `main`, folder `/ (root)`.
4. **Actions → Weekly analogue scan → Run workflow.** The first full run downloads SEC's bulk archive (~1 GB) and takes roughly 30–60 minutes. After that it runs every Saturday.

Until the first run finishes, the dashboard shows synthetic demo data with fictional companies.

## Run locally

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"
python agent/run.py --mode list --tickers ONON,HIMS,CAVA,RDDT --no-llm   # quick test on a few names
python agent/run.py                                                     # full universe
python agent/run.py --demo                                              # regenerate demo data
python -m http.server   # then open http://localhost:8000
```

A small list run scores against very few companies, so its z-scores are less meaningful than a full run.

## Tuning

- `config/seeds.json` — the analogue library. Add companies and the year before their run; the agent rebuilds the fingerprint from filings. Seeds need ~2009+ XBRL data and three years of revenue.
- `config/settings.json` — universe size limits, market-cap and liquidity gates, how many names get market data and filing reviews, default weights, Claude model.
- Dashboard weights can be changed live and are remembered in your browser.

## Known limitations

- **Survivorship bias.** Seeds are chosen because they won. Plenty of companies with the same fingerprint went nowhere. Treat matches as a reading list, not a signal.
- **US filers only.** SEC XBRL covers 10-K filers; foreign private issuers filing 20-F use IFRS tags and are mostly skipped.
- **Tag gaps.** Companies use different XBRL tags; the code tries common fallbacks, and features that can't be computed are excluded from that comparison.
- **Yahoo data is unofficial.** yfinance can rate-limit or change shape; market signals degrade gracefully when fields are missing.
- **Not yet backtested.** The obvious next step is running the screen as of past years and measuring forward returns of the top decile against the universe.
