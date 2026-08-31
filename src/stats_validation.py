"""
stats_validation.py

Statistical machinery for asking, rigorously, the question CRITIQUE.md
already raises in prose: given that several strategy variants were tried
(five EWMAC speeds plus a multi-speed blend), how much of the reported
Sharpe ratio is real skill versus an artifact of picking the best result
out of several trials?

Two standard, published techniques (Bailey & Lopez de Prado):

1. Deflated Sharpe Ratio (DSR) -- the Probabilistic Sharpe Ratio (PSR)
   of the chosen strategy, benchmarked not against zero but against the
   *expected maximum* Sharpe ratio you'd see by chance alone after N
   independent trials. More trials -> higher bar to clear -> a Sharpe
   that looked significant against zero can become insignificant once
   deflated for how many configurations were actually tried.

2. Probability of Backtest Overfitting (PBO) via Combinatorially
   Symmetric Cross-Validation (CSCV) -- repeatedly split history into two
   halves, ask "does the strategy that looked best in the first half
   still look above-median in the second half?", and report how often it
   doesn't. A PBO near 0.5 means the in-sample winner is no better than a
   coin flip out-of-sample -- a hallmark of overfitting.

References: Bailey, D. and Lopez de Prado, M., "The Sharpe Ratio
Efficient Frontier" (2012) and "The Probability of Backtest Overfitting"
(2016).
"""

from __future__ import annotations

import itertools
from math import erf, sqrt

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation (no scipy dependency)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def expected_max_sharpe_iid(sharpe_variance: float, n_trials: int) -> float:
    """
    Closed-form approximation (Bailey & Lopez de Prado 2012) for the
    expected value of the MAXIMUM Sharpe ratio observed across n_trials
    independent strategies, each with Sharpe ~ N(0, sharpe_variance) under
    the null of no real skill. This is the benchmark DSR compares the
    chosen strategy's Sharpe against -- not zero.

    Requires n_trials >= 2 (the correction is undefined for a single
    trial, since there is no selection effect to correct for).
    """
    if n_trials < 2:
        raise ValueError("expected_max_sharpe_iid requires at least 2 trials")

    term1 = (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * np.e))
    return sqrt(sharpe_variance) * (term1 + term2)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    P(true Sharpe > benchmark_sharpe | observed_sharpe, n_obs, return
    distribution's skew/kurtosis). Accounts for the fact that Sharpe
    ratio estimation error is worse for skewed, fat-tailed return series
    than the textbook i.i.d.-normal assumption -- both trend-following
    and mean-reversion returns are typically not normal.
    """
    denom = sqrt(1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe ** 2)
    z = (observed_sharpe - benchmark_sharpe) * sqrt(n_obs - 1) / denom
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: list[float],
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    Deflated Sharpe Ratio for a strategy chosen as the best (or simply
    reported) among len(trial_sharpes) trials. Returns a dict with the
    benchmark (expected max Sharpe under the null given this many trials),
    the resulting DSR (a probability), and the raw PSR against a zero
    benchmark for comparison.
    """
    n_trials = len(trial_sharpes)
    trial_variance = float(np.var(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0

    benchmark = expected_max_sharpe_iid(trial_variance, n_trials) if n_trials >= 2 else 0.0

    psr_vs_zero = probabilistic_sharpe_ratio(observed_sharpe, 0.0, n_obs, skew, kurtosis)
    dsr = probabilistic_sharpe_ratio(observed_sharpe, benchmark, n_obs, skew, kurtosis)

    return {
        "n_trials": n_trials,
        "trial_sharpe_std": sqrt(trial_variance),
        "benchmark_sharpe": benchmark,
        "psr_vs_zero": psr_vs_zero,
        "deflated_sharpe_ratio": dsr,
    }


def pbo_cscv(returns: pd.DataFrame, n_splits: int = 10) -> dict:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu 2016).

    `returns`: DataFrame, one column per strategy variant, one row per
    period, all on the same aligned index (no NaNs -- align/dropna first).

    Splits the sample into n_splits contiguous blocks. For every way of
    picking exactly half the blocks as the "training" set (its complement
    is "testing"), ranks strategies by Sharpe in training, finds the
    training-best strategy's relative rank in testing, and converts that
    rank to a logit. PBO is the fraction of splits where the logit is <=
    0, i.e. the training-best strategy performed at or below the test-set
    median -- exactly what you'd expect half the time from a strategy
    with no real, generalizing edge.
    """
    if returns.isna().any().any():
        raise ValueError("pbo_cscv requires a fully aligned, NaN-free returns DataFrame")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")

    n_strategies = returns.shape[1]
    blocks = np.array_split(returns.index, n_splits)
    block_indices = list(range(n_splits))

    logits = []
    for train_blocks in itertools.combinations(block_indices, n_splits // 2):
        train_blocks = set(train_blocks)
        test_blocks = set(block_indices) - train_blocks

        train_idx = blocks[0][:0].append([blocks[i] for i in sorted(train_blocks)])
        test_idx = blocks[0][:0].append([blocks[i] for i in sorted(test_blocks)])

        train_sharpe = returns.loc[train_idx].mean() / returns.loc[train_idx].std()
        test_sharpe = returns.loc[test_idx].mean() / returns.loc[test_idx].std()

        best_in_train = train_sharpe.idxmax()
        # relative rank of the training-best strategy within the test-set
        # ranking, in (0, 1); 1.0 = best in test too, 0.0 = worst in test
        test_rank = test_sharpe.rank(pct=True)[best_in_train]
        # avoid +/-inf at the boundary
        test_rank = min(max(test_rank, 1.0 / (n_strategies + 1)), n_strategies / (n_strategies + 1))
        logit = np.log(test_rank / (1 - test_rank))
        logits.append(logit)

    logits = np.array(logits)
    pbo = float((logits <= 0).mean())

    return {
        "n_combinations": len(logits),
        "pbo": pbo,
        "mean_logit": float(logits.mean()),
    }
