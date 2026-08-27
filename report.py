"""
report.py

Generates the artifacts a reviewer actually looks at:
  - a cumulative return / drawdown chart (PNG, via matplotlib, no extra deps)
  - a metrics summary table (Markdown, embeddable straight into the README)
  - optionally, a full quantstats HTML tearsheet if quantstats is installed
    (pip install quantstats) -- this is the file worth linking from the CV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import metrics as metrics_mod


def plot_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    out_path: str = "reports/performance.png",
    title: str = "Cross-Asset Macro Trend Strategy",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cum_strategy = (1 + portfolio_returns.fillna(0)).cumprod()
    drawdown = cum_strategy / cum_strategy.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    axes[0].plot(cum_strategy.index, cum_strategy.values, label="Strategy (net of costs)", linewidth=1.6)
    if benchmark_returns is not None:
        cum_bm = (1 + benchmark_returns.reindex(cum_strategy.index).fillna(0)).cumprod()
        axes[0].plot(cum_bm.index, cum_bm.values, label="Equal-weight benchmark", linewidth=1.2, linestyle="--")
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def metrics_markdown_table(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> str:
    cols = [metrics_mod.summary(portfolio_returns, "Strategy")]
    if benchmark_returns is not None:
        cols.append(metrics_mod.summary(benchmark_returns, "Benchmark"))
    table = pd.concat(cols, axis=1)

    pct_rows = {"Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Rate"}

    def fmt(row_name, value):
        if pd.isna(value):
            return "n/a"
        return f"{value:.2%}" if row_name in pct_rows else f"{value:.2f}"

    formatted = pd.DataFrame(
        {c: [fmt(idx, table.loc[idx, c]) for idx in table.index] for c in table.columns},
        index=table.index,
    )
    return formatted.to_markdown()


def try_quantstats_html(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    out_path: str = "reports/tearsheet.html",
) -> str | None:
    """
    If quantstats is installed, generate the full HTML tearsheet (the file
    worth linking from a CV/GitHub README). Returns the output path, or None
    if quantstats isn't available -- callers should fall back to
    plot_performance() + metrics_markdown_table() in that case.
    """
    try:
        import quantstats as qs
    except ImportError:
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    qs.extend_pandas()
    qs.reports.html(
        portfolio_returns.fillna(0),
        benchmark=benchmark_returns.fillna(0) if benchmark_returns is not None else None,
        output=out_path,
        title="Cross-Asset Macro Trend Strategy",
    )
    return out_path
