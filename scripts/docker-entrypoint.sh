#!/bin/sh
set -e

# First boot in a fresh container: pull market data if the cache is empty.
if [ ! -f /app/data/cache/closes.csv ]; then
  echo "No price cache found — downloading SPY AGG GLD (this runs once)..."
  python -m aivestor.scripts.fetch_data \
    --tickers SPY AGG GLD \
    --start "${DATA_START:-2015-01-01}"
fi

exec "$@"
