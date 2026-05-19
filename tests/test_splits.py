import numpy as np
import pandas as pd

from aivestor.splits import split_by_fraction


def test_split_by_fraction():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame(np.arange(100.0), index=idx)
    train, test = split_by_fraction(df, 0.7)
    assert len(train) == 70
    assert len(test) == 30
