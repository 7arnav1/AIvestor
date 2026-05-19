"""Time-based splits for walk-forward style evaluation."""

from __future__ import annotations

import pandas as pd


def split_by_date(
    prices: pd.DataFrame,
    train_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows strictly before train_end (train) and from train_end onward (test)."""
    te = pd.Timestamp(train_end)
    train = prices[prices.index < te]
    test = prices[prices.index >= te]
    if train.empty or test.empty:
        raise ValueError("train and test must both be non-empty; adjust train_end.")
    return train, test


def split_by_fraction(prices: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    n = len(prices)
    cut = max(1, int(n * train_fraction))
    if cut >= n:
        raise ValueError("train split too large for series length")
    train = prices.iloc[:cut]
    test = prices.iloc[cut:]
    return train, test
