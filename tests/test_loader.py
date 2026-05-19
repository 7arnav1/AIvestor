import pandas as pd
import pytest

from aivestor.data.loader import load_prices


def test_load_from_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    closes = pd.DataFrame({"SPY": 100.0, "AGG": 50.0}, index=idx)
    closes.to_csv(cache / "closes.csv")

    out = load_prices(["SPY", "AGG"], start="2020-01-01", source="cache", cache_dir=cache)
    assert list(out.columns) == ["SPY", "AGG"]
    assert len(out) == 30


def test_cache_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(["SPY"], source="cache", cache_dir=tmp_path / "nope")
