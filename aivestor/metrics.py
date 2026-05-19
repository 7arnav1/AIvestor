"""Risk and return metrics from a return or equity series (daily bars assumed)."""

from __future__ import annotations

import numpy as np


def _annualization_factor(trading_days_per_year: int) -> float:
    return float(np.sqrt(trading_days_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough decline as a positive fraction (e.g. 0.2 = 20% drawdown)."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.clip(peak, 1e-12, None)
    return float(np.max(dd))


def cumulative_return(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    return float(eq[-1] / eq[0] - 1.0)


def annualized_return(daily_returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r) - 1.0)
    n = r.size
    years = n / trading_days_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * _annualization_factor(trading_days_per_year))


def sharpe_ratio(
    daily_returns: np.ndarray,
    *,
    risk_free_daily: float = 0.0,
    trading_days_per_year: int = 252,
) -> float:
    excess = np.asarray(daily_returns, dtype=np.float64) - risk_free_daily
    if excess.size < 2:
        return 0.0
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * _annualization_factor(trading_days_per_year))


def sortino_ratio(
    daily_returns: np.ndarray,
    *,
    risk_free_daily: float = 0.0,
    trading_days_per_year: int = 252,
) -> float:
    excess = np.asarray(daily_returns, dtype=np.float64) - risk_free_daily
    if excess.size < 2:
        return 0.0
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    dstd = np.std(downside, ddof=1)
    if dstd < 1e-12:
        return 0.0
    return float(np.mean(excess) / dstd * _annualization_factor(trading_days_per_year))


def calmar_ratio(
    daily_returns: np.ndarray,
    equity: np.ndarray,
    *,
    trading_days_per_year: int = 252,
) -> float:
    ann_ret = annualized_return(daily_returns, trading_days_per_year)
    mdd = max_drawdown(equity)
    if mdd < 1e-12:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / mdd)


def summarize_backtest_result(
    result: dict,
    *,
    risk_free_daily: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Augment vectorized_backtest output with standard metrics."""
    eq = np.asarray(result["equity"], dtype=np.float64)
    net = np.asarray(result["step_returns"], dtype=np.float64)
    return {
        "cumulative_return": cumulative_return(eq),
        "annualized_return": annualized_return(net, trading_days_per_year),
        "annualized_vol": annualized_volatility(net, trading_days_per_year),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe_ratio(net, risk_free_daily=risk_free_daily, trading_days_per_year=trading_days_per_year),
        "sortino": sortino_ratio(net, risk_free_daily=risk_free_daily, trading_days_per_year=trading_days_per_year),
        "calmar": calmar_ratio(net, eq, trading_days_per_year=trading_days_per_year),
        "total_cost_fraction": float(result.get("total_cost_fraction", 0.0)),
    }
