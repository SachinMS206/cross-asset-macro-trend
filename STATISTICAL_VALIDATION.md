# Is the Multi-Speed EWMAC Choice Statistically Real, or Selection Bias?

CRITIQUE.md already documents that five nearby, equally-defensible EWMAC
speed choices produce Sharpe ratios ranging from -0.07 to 0.44, and that
the multi-speed blend was adopted specifically to reduce that fragility.
That's an honest description of the problem. This is the honest,
*quantitative* answer to the question it raises: **given that six
configurations were actually tried on this data, is the one we settled on
genuinely better than chance, or does it just look that way because it
was the best of several attempts?**

## Method

Two standard techniques from Bailey & Lopez de Prado's work on backtest
statistics (`src/stats_validation.py`), applied to the real trial set —
the five speeds in CRITIQUE.md's own sensitivity table plus the
multi-speed blend, run on the same live data, not a synthetic or
cherry-picked set:

- **Deflated Sharpe Ratio (DSR)**: the probability that the chosen
  strategy's true Sharpe exceeds zero, benchmarked against the *expected
  maximum* Sharpe you'd see across six independent trials by chance alone
  — not against zero directly. More trials raises the bar. A Sharpe that
  looks significant against zero can fail to clear this higher, correct
  bar.
- **Probability of Backtest Overfitting (PBO)** via Combinatorially
  Symmetric Cross-Validation: repeatedly split the live sample in half,
  check whether whichever speed looked best in one half still looks
  above-median in the other half, and report how often it doesn't. PBO
  near 50% is indistinguishable from picking the "best" of several coin
  flips.

Refreshed automatically by
[`.github/workflows/refresh-significance.yml`](.github/workflows/refresh-significance.yml)
via **Actions → "Refresh statistical validation" → Run workflow**.

<!-- LIVE_RESULTS_START -->
*Last refreshed automatically from live data: 2026-08-31 21:00 UTC. See `.github/workflows/refresh-significance.yml`.*

**Trials considered** (the five speeds in CRITIQUE.md's own sensitivity table, plus the multi-speed blend):

| Variant     |   Sharpe |
|:------------|---------:|
| 4/16        |   -0.302 |
| 8/32        |    0.102 |
| 16/64       |    0.348 |
| 32/128      |    0.548 |
| 64/256      |    0.845 |
| multi-speed |    0.377 |

**Chosen strategy: multi-speed** (observed Sharpe 0.377, n=2847 days)

| Metric | Value |
|---|---|
| Expected max Sharpe by chance alone (6 trials) | 0.509 |
| PSR vs. zero (no selection-bias correction) | 100.0% |
| **Deflated Sharpe Ratio** | **0.0%** |
| **Probability of Backtest Overfitting (PBO)** | **0.0%** |

**Reading this honestly:** once deflated for the number of trials, this Sharpe is NOT distinguishable from what chance alone would produce -- the honest reading is that speed selection here is not statistically validated. Separately, the PBO of 0.0% is low, consistent with the in-sample-best configuration generalizing out-of-sample.
<!-- LIVE_RESULTS_END -->

## Reading this honestly

- **A low Deflated Sharpe Ratio or a PBO near 50% is not a failure of
  this analysis — it's the analysis doing its job.** If that's what the
  live numbers above show, the honest conclusion is that the multi-speed
  choice, however sensible the underlying reasoning in CRITIQUE.md, is
  not statistically distinguishable from having picked well by chance
  among six options — and that conclusion should stand exactly as
  reported here, not be reframed or re-run until it looks better.
- **Six trials is a small number** for either technique to be highly
  powered — this is a real, honest count of what was actually tried, not
  an argument for treating the result as definitive either way.
- This validates the **speed-selection decision specifically**. It says
  nothing about the deeper question CRITIQUE.md §5 already raises (in the
  cross-asset-macro-trend project) — whether the trend signal has any
  edge over a vol-targeted benchmark at all across different random
  seeds/data draws.
