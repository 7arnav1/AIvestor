"""Roll out a trained policy into a weight matrix aligned with `prices` rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aivestor.config import PortfolioEnvConfig
from aivestor.rl.env import PortfolioEnv


def weights_from_ppo(
    model,
    prices: pd.DataFrame,
    cfg: PortfolioEnvConfig,
    *,
    deterministic: bool = True,
) -> np.ndarray:
    """
    Build T x N weights: equal-weight prefix until `lookback`, then policy weights.
    Rows align with `vectorized_backtest` / engine indexing for the same `prices`.
    """
    prices = prices.sort_index().ffill().dropna()
    t, n = len(prices), prices.shape[1]
    lb = cfg.lookback
    wmat = np.zeros((t, n), dtype=np.float64)
    wmat[:] = 1.0 / n

    env = PortfolioEnv(prices, cfg)
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        idx = env._t
        if idx < t:
            wmat[idx] = env._w
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, _, _ = env.step(action)

    return wmat
