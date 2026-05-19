"""Shared knobs for backtest and RL env so scripts stay in sync."""

from __future__ import annotations

from dataclasses import dataclass

from aivestor.backtest.costs import CostModel
from aivestor.backtest.engine import BacktestConfig


@dataclass
class PortfolioEnvConfig:
    lookback: int = 20
    risk_free_daily: float = 0.0
    costs: CostModel = CostModel(commission_rate=0.0005, slippage_rate=0.0005)
    initial_cash: float = 100_000.0


def make_run_configs(
    *,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    lookback: int = 20,
    initial_cash: float = 100_000.0,
) -> tuple[BacktestConfig, PortfolioEnvConfig]:
    costs = CostModel(commission_rate=commission, slippage_rate=slippage)
    bt = BacktestConfig(initial_cash=initial_cash, costs=costs)
    env_cfg = PortfolioEnvConfig(
        lookback=lookback,
        initial_cash=initial_cash,
        costs=costs,
    )
    return bt, env_cfg
