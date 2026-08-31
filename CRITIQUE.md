# Critique: what I'd say if asked "what are the weaknesses of this?"

Everything below is from actually stress-testing the pipeline against the
synthetic dataset (`src/data_loaders.py::generate_synthetic_macro_universe`,
seed=42 unless stated). Re-run these against `--live` data before quoting
specific numbers in an interview — the *pattern* of findings (parameter
sensitivity, cost sensitivity, seed sensitivity) is the important, durable
part; the exact numbers below will change with real data.

Reproduce any of this yourself:
```python
from src import backtest, data_loaders, metrics
universe = data_loaders.generate_synthetic_macro_universe(n_days=1500, seed=42)
r = backtest.run_portfolio_backtest(universe, target_ann_vol=0.10)['portfolio_returns']
metrics.sharpe_ratio(r)
```

---

## 1. Sharpe is invariant to `target_ann_vol` (sanity check, not a weakness)

```
target_vol=0.05  Sharpe=0.117  MaxDD=-6.38%   AnnRet=0.26%
target_vol=0.10  Sharpe=0.117  MaxDD=-12.44%  AnnRet=0.45%
target_vol=0.15  Sharpe=0.117  MaxDD=-18.19%  AnnRet=0.59%
target_vol=0.20  Sharpe=0.117  MaxDD=-23.64%  AnnRet=0.67%
```
Sharpe is flat by construction (scaling every position by a constant scales
return and vol proportionally, leaving their ratio unchanged) — this is
confirming the code does what it's supposed to, not a finding. Worth
mentioning briefly if asked "how did you validate the code was doing what
you think," because it's a clean way to say you checked for bugs
mathematically rather than just eyeballing a chart.

## 2. Signal speed (fast/slow EWMAC spans) — real weakness: non-monotonic, unstable

```
fast= 4  slow= 16   Sharpe= 0.389
fast= 8  slow= 32   Sharpe= 0.445
fast=16  slow= 64   Sharpe= 0.117   <- the "default" used in the README
fast=32  slow=128   Sharpe= 0.254
fast=64  slow=256   Sharpe=-0.070
```
There's no clean pattern (faster isn't monotonically better or worse), and
the specific default I picked (16/64) is actually one of the *worse*
options in this sample. **This is the single most important thing to say
if asked about weaknesses**: I didn't optimize this parameter — I picked a
textbook default — and the fact that nearby parameter choices swing the
result this much is exactly the kind of instability that should make you
suspicious of any single reported Sharpe ratio. A more rigorous version
would average forecasts across several speeds (a "multi-speed" EWMAC, which
`pysystemtrade` does) specifically to avoid this fragility, rather than
committing to one speed.

**Update, tested against live data and real statistics, not just prose:**
the multi-speed blend described above was actually implemented
(`signals.py::multi_speed_ewmac_signal`) and tested rigorously —
[`STATISTICAL_VALIDATION.md`](STATISTICAL_VALIDATION.md) applies the
Deflated Sharpe Ratio and Probability of Backtest Overfitting (Bailey &
Lopez de Prado) to the same six configurations shown above, run on live
FRED/Yahoo data. Two honest findings came out of it, and neither is the
"multi-speed fixes it" story this section originally hoped for:

```
Live Sharpe, six trials:
  4/16          -0.30
  8/32           0.10
  16/64          0.35
  32/128         0.55
  64/256         0.85   <- best live performer, by a wide margin
  multi-speed    0.38   <- the "fix" adopted above
```

1. **The multi-speed blend is not the best performer on live data — the
   single slowest speed (64/256) is, by more than double.** The blend was
   adopted to reduce fragility across nearby choices, not because it was
   expected to top the table, but that's still worth stating plainly
   rather than only reporting the blend's own flattering number.
2. **Deflated Sharpe Ratio for the chosen multi-speed strategy: 0.0%.**
   Once correctly benchmarked against the expected best-of-six-trials
   Sharpe under pure chance (not against zero), the multi-speed choice is
   statistically indistinguishable from having picked well by luck. This
   is the rigorous, quantitative version of the fragility this section
   already suspected — it's no longer just prose.

The honest conclusion: the multi-speed blend is a reasonable, principled
idea, but on the live data actually tested, it is neither the best
performer nor statistically validated as the right choice over the
alternatives. See `STATISTICAL_VALIDATION.md` for the full methodology,
including why PBO and DSR can legitimately tell different-sounding
stories (they answer different questions) rather than treating that as a
contradiction.

## 3. Transaction costs — breakeven around 8-9bps

```
cost_bps= 0   Sharpe= 0.130
cost_bps= 1   Sharpe= 0.117
cost_bps= 2   Sharpe= 0.103
cost_bps= 5   Sharpe= 0.063
cost_bps=10   Sharpe=-0.003
cost_bps=20   Sharpe=-0.136
```
The strategy's edge is thin enough that it's cost-negative somewhere around
8-9bps of round-trip cost. That's a genuinely useful, specific number to
have ready: it tells you this signal, as built, is not robust to
realistic-but-elevated transaction costs (e.g. trading less liquid
instruments, wider spreads in stressed markets), and any live version would
need either a lower-turnover signal or a more careful cost model before it
means anything.

## 4. Leave-one-out — result is sensitive to which instruments are included

```
Full universe    Sharpe=0.117
Drop US2S10S     Sharpe=0.558
Drop US5S30S     Sharpe=0.004
Drop EURUSD      Sharpe=0.218
Drop GOLD        Sharpe=-0.268
Drop SPX         Sharpe=0.028
```
Removing `US2S10S` alone nearly 5x's the Sharpe; removing `GOLD` flips it
negative. With only 5 instruments, single-instrument effects dominate the
portfolio result rather than washing out — which is expected with a small
universe, but it means the specific "0.117 Sharpe" headline number is not a
stable, diversified conclusion; it's meaningfully driven by one or two
legs. Worth saying explicitly: **a 5-instrument portfolio isn't enough to
make strong diversification claims** — real CTA books run 50-100+
instruments partly for this reason.

## 5. Random-seed sensitivity — the biggest one

```
seed=1  Strategy Sharpe=+0.429   Benchmark Sharpe=+0.423
seed=2  Strategy Sharpe=+0.703   Benchmark Sharpe=+0.351
seed=3  Strategy Sharpe=+0.863   Benchmark Sharpe=+0.784
seed=4  Strategy Sharpe=+0.344   Benchmark Sharpe=+0.463
seed=5  Strategy Sharpe=+0.686   Benchmark Sharpe=+1.323
```
Across five different synthetic draws, the trend strategy beats the
vol-targeted buy-and-hold benchmark in some draws (seeds 1-3) and loses to
it, sometimes clearly, in others (seeds 4-5). **This is the headline
honest finding of the whole exercise**: on this synthetic data, there is no
reliable evidence that the trend *signal* adds value over simply being
vol-targeted long — the result depends heavily on which random draw you
happen to look at. If a real `--live` run shows something similar, that's
not a reason to hide the project — it's a much stronger interview answer to
say "I tested whether the signal specifically added value beyond
vol-targeting, and the honest answer was 'inconclusive, needs a longer
sample and non-synthetic data.'" That's a research-mindset answer, not a
"my backtest made money" answer — and it's the difference recruiters at
this level are actually screening for.

---

## Other known structural weaknesses (not from sensitivity testing, but real)

- **Rates legs are sized in bps-of-spread terms, not DV01/duration-adjusted
  notional.** A real book trading `2s10s` would size the position in terms
  of dollar-duration risk, not raw bps moves — this repo's vol-targeting
  compensates for this only approximately.
- **Default portfolio weighting (`inverse_vol`) ignores correlation
  entirely.** Two highly correlated instruments would each get full weight
  as if independent, understating concentration risk. `risk_parity_weights`
  and `hrp_weights` fix this but are more sensitive to noisy covariance
  estimates over short lookback windows — a real tradeoff, not a free
  upgrade.
- **Vol targeting is mechanically pro-cyclical over short windows**: because
  `realized_ann_vol` is a rolling estimate, a sudden regime change means
  positions are sized on stale (too-low) vol for several days before the
  estimate catches up — a known, standard critique of vol-targeting systems
  generally, not specific to this implementation.
- **Flat bps transaction cost model** doesn't capture market impact, wider
  spreads during stress, or the fact that FX/rates/commodities have very
  different real-world liquidity profiles.
