from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aivestor.backtest.costs import CostModel


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    costs: CostModel = CostModel(commission_rate=0.0005, slippage_rate=0.0005)


def vectorized_backtest(
    prices: pd.DataFrame,
    target_weights: np.ndarray,
    config: BacktestConfig | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Long-only, daily rebalance at close.

    `prices`: T x N asset close prices, no NaNs.
    `target_weights`: T x N, rows sum to ~1 (cash implicitly 1 - sum if < 1).
    Returns equity curve, per-step returns, turnover, total cost drag.
    """
    cfg = config or BacktestConfig()
    p = prices.to_numpy(dtype=np.float64)
    w = np.asarray(target_weights, dtype=np.float64)
    if p.shape != w.shape:
        raise ValueError(f"prices {p.shape} vs weights {w.shape}")

    rets = np.empty_like(p)
    rets[0, :] = 0.0
    rets[1:, :] = (p[1:] - p[:-1]) / np.clip(p[:-1], 1e-12, None)

    port_ret = np.sum(w[:-1] * rets[1:], axis=1)
    turnover = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    costs = np.array([cfg.costs.total_on_turnover(t) for t in turnover])
    net = port_ret - costs

    equity = cfg.initial_cash * np.cumprod(1.0 + np.concatenate([[0.0], net]))
    total_cost = float(np.sum(costs * cfg.initial_cash))

    return {
        "equity": equity,
        "step_returns": net,
        "turnover": turnover,
        "total_cost_fraction": total_cost / cfg.initial_cash,
    }


def equal_weight_weights(n_assets: int, n_steps: int) -> np.ndarray:
    w = np.full((n_steps, n_assets), 1.0 / n_assets, dtype=np.float64)
    return w
