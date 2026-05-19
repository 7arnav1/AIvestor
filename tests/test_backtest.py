import numpy as np
import pandas as pd

from aivestor.backtest.engine import BacktestConfig, vectorized_backtest
from aivestor.baselines import equal_weight
from aivestor.metrics import summarize_backtest_result


def test_vectorized_equal_weight_smoke():
    rng = np.random.default_rng(0)
    t, n = 50, 3
    p0 = 100.0 + rng.standard_normal((t, n))
    prices = pd.DataFrame(p0, columns=list("ABC"))
    w = equal_weight(n, t)
    out = vectorized_backtest(prices, w, BacktestConfig(initial_cash=10_000.0))
    m = summarize_backtest_result(out)
    assert "sharpe" in m
    assert len(out["equity"]) == t
