import numpy as np
import pandas as pd

from aivestor.baselines import from_strategy_name


def _prices(n_rows: int = 60, n_cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2019-01-01", periods=n_rows, freq="B")
    data = 100 * np.cumprod(1 + 0.001 * rng.standard_normal((n_rows, n_cols)), axis=0)
    return pd.DataFrame(data, index=idx, columns=["SPY", "AGG", "GLD"])


def test_equal_weight_sums_to_one():
    p = _prices()
    w = from_strategy_name("equal", p)
    assert w.shape == (len(p), 3)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_risk_parity():
    p = _prices()
    w = from_strategy_name("risk_parity", p)
    assert w.shape == (len(p), 3)
    np.testing.assert_allclose(w[0].sum(), 1.0)
