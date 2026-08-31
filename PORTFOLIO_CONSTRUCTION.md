# Does Combining the Trend Book With a Mean-Reversion Book Help?

A specific, well-founded portfolio-construction question, tested honestly
rather than assumed: this repo's trend-following book follows moves; the
mean-reversion book in
[cross-asset-rv-monitor](https://github.com/SachinMS206/cross-asset-rv-monitor)
fades statistical extremes on largely different instruments. Standard
portfolio theory says combining two genuinely different, lowly-correlated
return streams should improve risk-adjusted returns even if *neither*
individually beats a simple benchmark. This tests whether that actually
holds here, on real data, rather than assuming it.

## Method

Both books' real daily `portfolio_returns` are exported directly from
their own tested pipelines (`scripts/run_demo.py --export-returns` here,
`scripts/run_monitor.py --export-returns` in cross-asset-rv-monitor) --
this analysis does not re-derive either strategy's logic, it combines
their actual live output. The two series are aligned on their overlapping
dates, then blended with an **equal-risk-contribution** weighting: each
stream is rescaled to the same target volatility using its own trailing
(lagged, no look-ahead) realized vol before averaging 50/50, so neither
book's naturally higher or lower realized vol dominates the blend by
default. See `scripts/combine_portfolios.py::equal_risk_blend`.

The result is refreshed automatically by
[`.github/workflows/combined-portfolio.yml`](.github/workflows/combined-portfolio.yml),
which checks out both repos, runs both live pipelines, and commits the
output straight back here — via the **Actions tab → "Refresh combined
portfolio" → Run workflow** button or its schedule.

<!-- LIVE_RESULTS_START -->
*Not yet run — click **Actions → Refresh combined portfolio → Run
workflow** on GitHub to populate this section with real results.*
<!-- LIVE_RESULTS_END -->

## Reading this honestly

- **The correlation number above is the whole ballgame.** If it isn't
  meaningfully close to zero, the diversification argument doesn't hold
  here regardless of what the blended Sharpe shows, and the block above
  says so explicitly rather than letting a good-looking blended number
  speak for itself.
- **A higher blended Sharpe than either book alone is the expected,
  mechanical result of combining two lowly-correlated streams** — that's
  what diversification does by construction, not evidence that either
  underlying strategy suddenly has more edge than `CRITIQUE.md` in either
  repo already claims for it.
- **This still doesn't mean the blend reliably beats a passive benchmark.**
  Check the benchmark row in the table above before drawing that
  conclusion either way.
- Sample size here is bounded by data availability in both repos (FRED
  history typically starts around 2015 for the relevant series) — not a
  multi-decade, multi-cycle test.
