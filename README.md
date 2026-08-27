# Cross-Asset Systematic Macro Trend Strategy

A vol-targeted, risk-weighted trend-following portfolio spanning US rate
curves, FX and commodities/equities — built as an extension of my
[2s10s Treasury curve momentum strategy](https://github.com/SachinMS206/2s10s-momentum-strategy)
into a diversified, portfolio-level systematic macro book.

The core question this project is trying to answer honestly: **does a trend
signal add anything once you account for costs, and does the answer hold up
out-of-sample?** — rather than presenting a single flattering backtest.

## Why this exists

A single-instrument trend backtest shows you can code a signal. It doesn't
show you can run a **book**. The gap between the two is:

- **Position sizing** that accounts for each instrument's own volatility, so
  the portfolio isn't accidentally dominated by whichever instrument moves
  the most in price terms (a rates spread in bps and gold in dollars are not
  directly comparable without normalizing).
- **Portfolio-level risk allocation** across instruments, rather than
  treating each strategy leg as independent.
- **Validation discipline** — checking whether the edge is consistent across
  time (walk-forward), not just present in one static backtest window.

This project is structured around Rob Carver's *Systematic Trading* /
[`pysystemtrade`](https://github.com/robcarver17/pysystemtrade) framework:
volatility-scaled forecasts (roughly in `[-20, +20]`, averaging `~10` in
absolute value), instrument-level vol targeting, and portfolio-level risk
weighting — the same conceptual building blocks used in CTA/systematic
macro books.

## Universe

| Instrument | Description | Live data source |
|---|---|---|
| `US2S10S` | 10Y–2Y Treasury yield spread (bps) | FRED (`DGS10` − `DGS2`) |
| `US5S30S` | 30Y–5Y Treasury yield spread (bps) | FRED (`DGS30` − `DGS5`) |
| `EURUSD`  | EUR/USD spot | Yahoo Finance (`EURUSD=X`) |
| `GOLD`    | Gold spot (USD/oz) | Yahoo Finance (`GC=F`) |
| `SPX`     | S&P 500 index level | Yahoo Finance (`^GSPC`) |

## Methodology

1. **Signal — volatility-normalized EWMAC** (`src/signals.py`)
   An EMA(fast) − EMA(slow) crossover, divided by the instrument's own
   rolling price volatility and rescaled so the historical average absolute
   forecast is `~10`. This makes forecast strength comparable across very
   differently-scaled instruments (a rates spread vs. an FX rate vs. an
   equity index) before they're combined.

2. **Instrument-level vol targeting** (`src/risk.py`)
   Each forecast is converted into a position sized to a constant annualized
   volatility contribution (`target_ann_vol`, default 10%):
   `position = (forecast / 10) × (target_vol / realized_vol)`.
   A fast, choppy instrument and a slow, low-vol instrument end up
   contributing similar risk for the same forecast strength.

3. **Portfolio-level weighting** (`src/risk.py`)
   Instruments are combined either inverse-vol weighted (default,
   dependency-free) or via full risk parity (`scipy.optimize`, equal risk
   contribution) — with an optional `hrp_weights()` wrapper around
   Riskfolio-Lib's Hierarchical Risk Parity if that package is installed.

4. **Costs** — every position change is charged in bps of notional traded
   (`cost_bps`, default 1bp), so reported returns are net of a simple
   transaction cost model, not gross.

5. **No look-ahead** — every position is lagged one day relative to the
   signal that generated it; portfolio weights are similarly lagged.

6. **Walk-forward validation** (`src/backtest.py::walk_forward_splits`) —
   Sharpe ratio is checked fold-by-fold on an expanding out-of-sample
   window, rather than reporting a single backtest number, to see whether
   performance is regime-dependent.

7. **Benchmark** — a vol-targeted **constant long** position (forecast fixed
   at the max, `+10`) run through the *same* vol-targeting and cost
   machinery. This isolates the value of the trend **signal** specifically,
   rather than conflating it with the benefit of vol-targeting itself.

## Results

Two ways to run this:

```bash
python scripts/run_demo.py            # synthetic data, no internet required
python scripts/run_demo.py --live      # real FRED + Yahoo Finance data
```

The results below are refreshed automatically by
[`.github/workflows/refresh-live-data.yml`](.github/workflows/refresh-live-data.yml),
which runs `scripts/run_demo.py --live` against **real** FRED/Yahoo Finance
data on GitHub's own servers — either on a schedule or on demand via the
**Actions tab → "Refresh live backtest results" → Run workflow** button —
and commits the output straight back to this README. No number below was
typed in by hand.

<!-- LIVE_RESULTS_START -->
*Not yet run — click **Actions → Refresh live backtest results → Run
workflow** on GitHub to populate this section with real results, or run
`python scripts/run_demo.py --live` locally and `python
scripts/update_readme.py`.*
<!-- LIVE_RESULTS_END -->

Walk-forward out-of-sample Sharpe by fold is checked in
`CRITIQUE.md` — on the synthetic dataset it was inconsistent across regimes
(a useful, honest result showing the signal is regime-dependent rather than
a reliably positive edge); re-run the same check against live data before
relying on that conclusion in an interview.

## Repo structure

```
src/
  data_loaders.py   # real FRED/Yahoo loaders + labeled synthetic generator
  signals.py        # dual-MA and vol-normalized EWMAC signal construction
  risk.py            # vol targeting, inverse-vol & risk-parity weighting
  backtest.py        # backtest engine, cost model, walk-forward splits
  metrics.py          # Sharpe, Sortino, drawdown, Calmar, tail ratio, etc.
  report.py            # PNG tearsheet + markdown table + optional quantstats HTML
scripts/
  run_demo.py           # end-to-end pipeline runner
tests/
  test_signals.py, test_risk.py, test_backtest.py   # 17 unit tests
WALKTHROUGH.md            # worked-example explanation of the core formulas
CRITIQUE.md                # quantified sensitivity testing + honest weaknesses
```

## Setup

```bash
pip install -r requirements.txt
python scripts/run_demo.py
python -m unittest discover -s tests -v
```

## Known limitations

See `CRITIQUE.md` for a quantified breakdown (parameter sensitivity, cost
sensitivity, leave-one-out instrument analysis, and random-seed robustness
testing), plus `WALKTHROUGH.md` for a worked-example explanation of the
vol-targeting and forecast-normalization formulas.

Headline findings from stress-testing the pipeline: the specific EWMAC
speed (16/64) was not optimized and nearby choices swing Sharpe
substantially; the strategy's edge turns cost-negative above ~8-9bps
round-trip; and across different random data draws it's inconclusive
whether the trend signal reliably beats a simple vol-targeted long-only
benchmark. That last point is the most important one to be upfront about —
see `CRITIQUE.md` §5.

## Acknowledgements

Signal/position-sizing conventions informed by Rob Carver's *Systematic
Trading* and the open-source
[`pysystemtrade`](https://github.com/robcarver17/pysystemtrade) framework.
Built with tools from
[awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading).
