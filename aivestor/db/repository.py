from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from aivestor.db.models import OhlcBar


def _row_from_series(ticker: str, idx, row: pd.Series) -> OhlcBar:
    d = idx.date() if hasattr(idx, "date") else idx
    if not isinstance(d, date):
        d = pd.Timestamp(idx).date()
    return OhlcBar(
        ticker=ticker,
        bar_date=d,
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
)


def upsert_ohlcv_merge(session: Session, ticker: str, df: pd.DataFrame) -> int:
    """Delete overlapping dates for ticker, then insert (works on SQLite and Postgres)."""
    if df.empty:
        return 0
    dates = [pd.Timestamp(ix).date() for ix in df.index]
    session.execute(delete(OhlcBar).where(OhlcBar.ticker == ticker, OhlcBar.bar_date.in_(dates)))
    rows = [_row_from_series(ticker, idx, row) for idx, row in df.iterrows()]
    session.add_all(rows)
    session.commit()
    return len(rows)


def load_close_panel(session: Session, tickers: list[str]) -> pd.DataFrame:
    stmt = (
        select(OhlcBar.ticker, OhlcBar.bar_date, OhlcBar.close)
        .where(OhlcBar.ticker.in_(tickers))
        .order_by(OhlcBar.bar_date)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    pivot = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return pivot
