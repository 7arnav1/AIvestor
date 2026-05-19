"""Load aligned close prices from cache, database, or Yahoo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aivestor.data.yahoo import align_close_prices, fetch_panel
from aivestor.db.session import get_session_factory, init_db
from aivestor.db.repository import load_close_panel


def _load_from_cache(
    tickers: list[str],
    cache_dir: Path,
    start: str | None,
) -> pd.DataFrame | None:
    closes_path = cache_dir / "closes.csv"
    if not closes_path.exists():
        return None
    df = pd.read_csv(closes_path, index_col=0, parse_dates=True)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        return None
    out = df[tickers].sort_index()
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    return out.dropna(how="any")


def _load_from_db(
    tickers: list[str],
    database_url: str | None,
    start: str | None,
) -> pd.DataFrame | None:
    init_db(database_url)
    sf = get_session_factory(database_url)
    with sf() as session:
        panel = load_close_panel(session, tickers)
    if panel.empty:
        return None
    panel.index = pd.to_datetime(panel.index)
    if start:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    return panel.dropna(how="any")


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    source: str = "auto",
    cache_dir: Path | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """
    Return T x N aligned close prices.

    source:
      - yahoo: always download
      - cache: data/cache/closes.csv (per-ticker CSVs not used here)
      - db: ohlc_bars table
      - auto: cache, then db, then yahoo
    """
    cache_dir = cache_dir or Path("data/cache")
    tickers = list(tickers)

    if source == "yahoo":
        panel = fetch_panel(tickers, start=start, end=end)
        return align_close_prices(panel)

    if source == "cache":
        cached = _load_from_cache(tickers, cache_dir, start)
        if cached is None or cached.empty:
            raise FileNotFoundError(
                f"No usable cache at {cache_dir / 'closes.csv'} for {tickers}. "
                "Run fetch_data first or use --data-source yahoo."
            )
        return cached

    if source == "db":
        loaded = _load_from_db(tickers, database_url, start)
        if loaded is None or loaded.empty:
            raise FileNotFoundError(
                "No rows in database for these tickers. "
                "Run fetch_data --persist-db or use --data-source yahoo."
            )
        return loaded

    if source == "auto":
        cached = _load_from_cache(tickers, cache_dir, start)
        if cached is not None and not cached.empty:
            return cached
        loaded = _load_from_db(tickers, database_url, start)
        if loaded is not None and not loaded.empty:
            return loaded
        panel = fetch_panel(tickers, start=start, end=end)
        return align_close_prices(panel)

    raise ValueError(f"Unknown data source: {source}")
