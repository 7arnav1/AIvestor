"""Create database tables (SQLite by default, or DATABASE_URL)."""

from __future__ import annotations

import argparse

from aivestor.db.session import get_database_url, init_db


def main() -> None:
    p = argparse.ArgumentParser(description="AIvestor — initialize SQL schema")
    p.add_argument("--url", default=None, help="SQLAlchemy URL (default: env DATABASE_URL or SQLite file)")
    args = p.parse_args()
    init_db(args.url)
    print(f"Initialized schema at {args.url or get_database_url()}")


if __name__ == "__main__":
    main()
