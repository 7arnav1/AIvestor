import numpy as np
import pandas as pd

from aivestor.backtest.engine import BacktestConfig, vectorized_backtest
from aivestor.config import make_run_configs
import pytest

gymnasium = pytest.importorskip("gymnasium")

from aivestor.rl.env import PortfolioEnv
from aivestor.rl.rollout import weights_from_ppo


def _prices(rows: int = 80, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2018-01-01", periods=rows, freq="B")
    data = 100 * np.cumprod(1 + 0.002 * rng.standard_normal((rows, cols)), axis=0)
    return pd.DataFrame(data, index=idx, columns=["A", "B"])


def test_env_step_shape():
    p = _prices()
    _, env_cfg = make_run_configs(lookback=10)
    env = PortfolioEnv(p, env_cfg)
    obs, _ = env.reset()
    assert obs.shape == (10 * 2 + 2,)
    action = np.zeros(2, dtype=np.float32)
    obs2, reward, term, trunc, _ = env.step(action)
    assert obs2.shape == obs.shape
    assert np.isfinite(reward)


def test_rollout_weights_match_backtest_indexing():
    """Weights from a fixed path should backtest without shape errors."""
    p = _prices()
    n = p.shape[1]
    t = len(p)
    w = np.full((t, n), 1.0 / n)
    _, env_cfg = make_run_configs(lookback=10)
    bt_cfg = BacktestConfig(initial_cash=env_cfg.initial_cash, costs=env_cfg.costs)
    out = vectorized_backtest(p, w, bt_cfg)
    assert len(out["equity"]) == t


def test_random_policy_rollout_smoke():
    """weights_from_ppo needs a model; use a tiny mock with predict."""
    p = _prices(rows=40)

    class _MockModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(2, dtype=np.float32), None

    _, env_cfg = make_run_configs(lookback=5)
    w = weights_from_ppo(_MockModel(), p, env_cfg)
    assert w.shape == (len(p), 2)
