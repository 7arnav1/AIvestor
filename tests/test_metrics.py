import numpy as np

from aivestor.metrics import max_drawdown, sharpe_ratio


def test_max_drawdown_simple():
    eq = np.array([100.0, 110.0, 90.0, 100.0])
    assert max_drawdown(eq) > 0


def test_sharpe_zero_vol():
    r = np.zeros(100)
    assert sharpe_ratio(r) == 0.0
