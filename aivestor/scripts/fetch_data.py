"""Download Yahoo OHLCV, validate, cache CSV, optional DB persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

from aivestor.data.yahoo import align_close_prices, cache_panel_csv, fetch_panel
from aivestor.db import init_db, upsert_ohlcv_merge
from aivestor.db.session import get_session_factory


def main() -> None:
    p = argparse.ArgumentParser(description="AIvestor — fetch Yahoo Finance data")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument(
        "--async-fetch",
        action="store_true",
        help="Download tickers concurrently (thread pool)",
    )
    p.add_argument(
        "--persist-db",
        action="store_true",
        help="Upsert OHLCV into DATABASE_URL (SQLite file by default)",
    )
    p.add_argument("--database-url", default=None, help="Override SQLAlchemy URL for this run")
    args = p.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    if args.async_fetch:
        from aivestor.data.async_yahoo import fetch_panel_async_run

        panel = fetch_panel_async_run(args.tickers, start=args.start, end=args.end)
    else:
        panel = fetch_panel(args.tickers, start=args.start, end=args.end)

    closes = align_close_prices(panel)
    cache_panel_csv(panel, args.cache_dir)
    out = args.cache_dir / "closes.csv"
    closes.to_csv(out)
    print(f"Saved per-ticker OHLCV under {args.cache_dir}/")
    print(f"Aligned closes: {closes.shape} -> {out}")

    if args.persist_db:
        init_db(args.database_url)
        sf = get_session_factory(args.database_url)
        with sf() as session:
            for sym, df in panel.items():
                n = upsert_ohlcv_merge(session, sym, df)
                print(f"DB upsert {sym}: {n} rows")


if __name__ == "__main__":
    main()
