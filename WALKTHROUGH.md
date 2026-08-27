# Walkthrough: the two formulas you need to be able to explain from memory

If an interviewer only asks about two things in this repo, it'll be the
forecast normalization and the vol-targeting formula. Below is each one
worked through with actual numbers, not just the code. Read this, then close
it and try to redo the worked examples from scratch — if you can't, that's
the part to go back and sit with.

---

## 1. Why normalize the forecast at all?

`src/signals.py::ewmac_signal`

**The problem:** `EMA_fast(price) - EMA_slow(price)` is in the *instrument's
own units*. For `US2S10S` (quoted in bps) that difference might be `+8`. For
`GOLD` (quoted in dollars) it might be `+45`. For `EURUSD` (quoted to 4
decimal places) it might be `+0.006`. These three numbers are not
comparable — `+45` doesn't mean "stronger trend" than `+8`, it just means
gold's price scale is bigger.

**The fix, step by step:**

```
raw          = EMA_fast(price) - EMA_slow(price)              # instrument's own units
price_vol    = rolling std of daily price changes (EWM, span=36)
vol_adj      = raw / price_vol                                 # now in "std-devs of daily move" — comparable across instruments
scalar       = 10 / (expanding mean of |vol_adj|)               # rescale so avg |forecast| ≈ 10 historically
forecast     = clip(vol_adj * scalar, -20, +20)
```

**Worked example** (made-up numbers, just to see the mechanics):

Say for `GOLD` on a given day: `raw = 45`, and the rolling std of gold's
daily price changes is `price_vol = 15`. Then:

```
vol_adj = 45 / 15 = 3.0
```

Say historically `|vol_adj|` for gold has averaged `0.3`. Then:

```
scalar = 10 / 0.3 = 33.3
forecast = 3.0 * 33.3 = 100  ->  clipped to +20
```

That's a maxed-out forecast: today's trend is unusually strong relative to
gold's own typical daily move. Do the same for `US2S10S` with `raw = 8`,
`price_vol = 4` → `vol_adj = 2.0`, and if its historical `|vol_adj|` average
is `0.2` → `scalar = 50` → `forecast = 100` → also clipped to `+20`.

**The point:** both instruments now say "forecast = +20" despite completely
different raw numbers (`45` vs `8`), because both represent an equally
*unusual* move relative to that instrument's own normal behavior. That's
what makes it valid to feed both into the same position-sizing formula next.

**If asked "why divide by price_vol AND rescale by a historical average" —**
the vol division makes it scale-free; the historical rescaling is just
cosmetic, so the *number itself* (e.g. "forecast of 15") means roughly the
same thing across every instrument and every backtest run, which makes
forecasts easier to interpret and compare (this is Carver's convention, not
a mathematical necessity — you could skip the rescaling and it would still
work, just with less interpretable numbers).

---

## 2. Vol-targeted position sizing

`src/risk.py::vol_target_position_size`

```
normalized_forecast = clip(forecast, -20, 20) / 10
position = normalized_forecast * (target_ann_vol / realized_ann_vol)
```

**Worked example:** Say `target_ann_vol = 0.10` (10%).

- **Instrument A** (calm, e.g. a rate spread): `realized_ann_vol = 0.04` (4%
  annualized). Forecast = `+10` (average strength).
  ```
  normalized_forecast = 10/10 = 1.0
  position = 1.0 * (0.10 / 0.04) = 2.5
  ```
- **Instrument B** (choppy, e.g. FX): `realized_ann_vol = 0.20` (20%
  annualized). Same forecast = `+10`.
  ```
  normalized_forecast = 1.0
  position = 1.0 * (0.10 / 0.20) = 0.5
  ```

**The point:** the calm instrument gets a *much bigger* position (2.5x) than
the choppy one (0.5x) for the *identical* forecast strength. Multiply
position × realized vol for both: `2.5 × 4% = 10%` and `0.5 × 20% = 10%` —
identical risk contribution. That's the entire idea of vol targeting in one
sentence: **size positions so that risk contributed, not capital or
contracts, is what's being controlled.**

**If asked "what happens if realized_vol is estimated badly" —** this is a
real weakness, and you should say so: the formula divides by realized vol,
so if a quiet instrument suddenly has one huge move, `realized_ann_vol` will
lag (because it's a rolling estimate) and the position sizing will be too
big for a few days until the estimate catches up. This is a known,
real failure mode of vol targeting (it's slightly pro-cyclical / can
oversize right before a vol spike) — see `CRITIQUE.md`.

---

## Quiz yourself

Before this goes on your CV, you should be able to answer these without
looking at the code:

1. Why can't you just compare raw `EMA_fast - EMA_slow` values across
   instruments directly?
2. If `target_ann_vol` is doubled, what happens to Sharpe ratio? (Run the
   sensitivity test in `CRITIQUE.md` §1 if you're not sure — then explain
   *why* that result makes mathematical sense.)
3. Two instruments have the same forecast of `+15` but different realized
   vols of 6% and 18%. What's the ratio of their position sizes?
4. Why is portfolio weight *also* lagged by one day (`.shift(1)` in
   `backtest.py::run_portfolio_backtest`), separately from the position lag
   inside each instrument's own backtest?
