"""Run backtests and compare strategies with metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aivestor.backtest.engine import BacktestConfig, vectorized_backtest
from aivestor.metrics import summarize_backtest_result


def run_strategy(
    prices: pd.DataFrame,
    weights: np.ndarray,
    config: BacktestConfig | None = None,
    *,
    risk_free_daily: float = 0.0,
) -> dict:
    raw = vectorized_backtest(prices, weights, config)
    metrics = summarize_backtest_result(raw, risk_free_daily=risk_free_daily)
    return {"raw": raw, "metrics": metrics}


def compare_weights(
    prices: pd.DataFrame,
    strategies: dict[str, np.ndarray] | dict[str, object],
    config: BacktestConfig | None = None,
    *,
    risk_free_daily: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for name, w in strategies.items():
        out = run_strategy(prices, w, config, risk_free_daily=risk_free_daily)
        m = out["metrics"]
        m = {"strategy": name, **m}
        rows.append(m)
    return pd.DataFrame(rows)


def format_metrics_table(df: pd.DataFrame) -> str:
    cols = [
        "strategy",
        "cumulative_return",
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "total_cost_fraction",
    ]
    present = [c for c in cols if c in df.columns]
    sub = df[present].copy()
    for c in sub.columns:
        if c == "strategy":
            continue
        sub[c] = sub[c].map(lambda x: f"{float(x):.4f}" if np.isfinite(x) else str(x))
    return sub.to_string(index=False)
