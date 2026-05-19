from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from aivestor.data.schema import validate_ohlcv


def fetch_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    interval: str = "1d",
    validate: bool = True,
) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    if validate:
        return validate_ohlcv(raw)
    return raw


def fetch_panel(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    return {t: fetch_history(t, start=start, end=end, interval=interval) for t in tickers}


def align_close_prices(
    panel: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join on calendar index; forward-fill missing (stale quote) then drop leading NaNs."""
    closes = {sym: df["Close"].rename(sym) for sym, df in panel.items()}
    out = pd.DataFrame(closes).sort_index()
    out = out.ffill().dropna()
    return out


def cache_panel_csv(panel: dict[str, pd.DataFrame], dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for sym, df in panel.items():
        df.to_csv(dir_path / f"{sym}.csv")
