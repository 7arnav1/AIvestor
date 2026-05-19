"""Concurrent Yahoo Finance downloads (thread pool; yfinance is synchronous)."""

from __future__ import annotations

import asyncio

import pandas as pd

from aivestor.data.yahoo import fetch_history


async def fetch_history_async(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    interval: str = "1d",
    validate: bool = True,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[str, pd.DataFrame]:
    sem = semaphore or asyncio.Semaphore(10)

    async def _run() -> pd.DataFrame:
        async with sem:
            return await asyncio.to_thread(
                fetch_history, ticker, start, end, interval=interval, validate=validate
            )

    df = await _run()
    return ticker, df


async def fetch_panel_async(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    interval: str = "1d",
    max_concurrent: int = 8,
) -> dict[str, pd.DataFrame]:
    sem = asyncio.Semaphore(max_concurrent)
    pairs = await asyncio.gather(
        *[
            fetch_history_async(t, start, end, interval=interval, semaphore=sem)
            for t in tickers
        ]
    )
    return dict(pairs)


def fetch_panel_async_run(*args, **kwargs) -> dict[str, pd.DataFrame]:
    return asyncio.run(fetch_panel_async(*args, **kwargs))
