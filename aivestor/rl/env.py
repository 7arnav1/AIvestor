from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from aivestor.backtest.engine import BacktestConfig, vectorized_backtest
from aivestor.config import PortfolioEnvConfig


class PortfolioEnv(gym.Env):
    """
    Multi-asset allocation: observe stacked recent returns, output target weights.
    Step reward is portfolio return minus CostModel on turnover; optional turnover_penalty
  on top for training regularization (defaults to 0 so it lines up with the backtest engine).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        cfg: PortfolioEnvConfig | None = None,
        *,
        turnover_penalty: float = 0.0,
    ):
        super().__init__()
        self.prices = prices.sort_index().ffill().dropna()
        self.returns = self.prices.pct_change().iloc[1:].to_numpy(dtype=np.float64)
        self.dates = self.prices.index[1:]
        self.cfg = cfg or PortfolioEnvConfig()
        self.turnover_penalty = turnover_penalty
        self.n_assets = self.returns.shape[1]
        self._t0 = self.cfg.lookback
        self._t = self._t0
        self._w = np.ones(self.n_assets, dtype=np.float64) / self.n_assets

        obs_dim = self.cfg.lookback * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        window = self.returns[self._t - self.cfg.lookback : self._t].ravel()
        pad = self.cfg.lookback * self.n_assets - len(window)
        if pad > 0:
            window = np.pad(window, (pad, 0), mode="constant")
        return np.concatenate([window, self._w.astype(np.float64)]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = self._t0
        self._w = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        exp = np.exp(a - np.max(a))
        w_new = exp / np.sum(exp)

        r = self.returns[self._t]
        port_r = float(np.dot(self._w, r))
        turnover = float(np.sum(np.abs(w_new - self._w)))
        cost = self.cfg.costs.total_on_turnover(turnover)
        reward = port_r - self.cfg.risk_free_daily - self.turnover_penalty * turnover - cost

        self._w = w_new
        self._t += 1
        terminated = self._t >= len(self.returns) - 1
        truncated = False
        return self._obs(), reward, terminated, truncated, {}

    def evaluate_weights_matrix(self, weights: np.ndarray) -> dict:
        """Run full-sample backtest for logging / benchmarking."""
        cfg = BacktestConfig(initial_cash=self.cfg.initial_cash, costs=self.cfg.costs)
        return vectorized_backtest(self.prices, weights, cfg)
