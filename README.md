# AIvestor

Quantitative backtesting and reinforcement learning for multi-asset portfolios. Pull market data, simulate strategies with transaction costs, compare risk-adjusted metrics, and train a PPO allocator on a train window—then report results on a held-out test slice.

## Install

```bash
cd AIvestor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[rl,dev,api]"
```

Optional Postgres: `pip install -e ".[postgres]"` and set `DATABASE_URL` (see `.env.example`). Start Postgres with `docker compose up -d` if you use the bundled compose file.

## Workflow

Typical offline run (no repeated Yahoo calls after the first fetch):

```bash
# 1. Schema (SQLite default: data/aivestor.db)
python -m aivestor.scripts.init_db

# 2. Download and cache
python -m aivestor.scripts.fetch_data \
  --tickers SPY AGG GLD --start 2015-01-01

# Optional: store OHLCV in the database
python -m aivestor.scripts.fetch_data \
  --tickers SPY AGG GLD --start 2015-01-01 --persist-db

# 3. Train PPO on the train fraction (default 70%)
python -m aivestor.scripts.train_rl \
  --timesteps 50000 \
  --data-source cache

# 4. Evaluate baselines + PPO on the test slice
python -m aivestor.scripts.evaluate \
  --model data/models/ppo_portfolio \
  --data-source cache \
  --baselines equal bh_0 risk_parity \
  --output data/runs
```

Use `--data-source yahoo` to force a live download. `auto` tries cache, then the database, then Yahoo.

## Dashboard (REST API + TypeScript UI)

After you have cached data and a trained model:

```bash
# Terminal 1 — API
source .venv/bin/activate
python -m aivestor.api

# Terminal 2 — UI (dev, proxies /api to port 8000)
cd web && npm install && npm run dev
```

Open http://localhost:5173, click **Run evaluation**. You get an out-of-sample metrics table, equity curves (baselines vs PPO), and the latest PPO allocation weights.

Production-style (single server serves built UI):

```bash
cd web && npm run build
python -m aivestor.api   # http://127.0.0.1:8000
```

Stack for demos: Python backtest/RL core, FastAPI REST layer, SQL-backed OHLCV (optional), TypeScript frontend.

## Commands

| Command | Purpose |
|--------|---------|
| `python -m aivestor.scripts.init_db` | Create tables |
| `python -m aivestor.scripts.fetch_data --tickers SPY AGG GLD --start 2018-01-01` | Yahoo OHLCV to CSV cache |
| `python -m aivestor.scripts.fetch_data ... --async-fetch --persist-db` | Concurrent fetch + DB upsert |
| `python -m aivestor.scripts.train_rl --timesteps 50000 --data-source cache` | Train PPO on train split |
| `python -m aivestor.scripts.evaluate --model data/models/ppo_portfolio` | Test-split metrics |

### Useful flags

- `--data-source` — `auto`, `cache`, `db`, or `yahoo`
- `--train-end YYYY-MM-DD` — fixed train/test cut instead of `--train-fraction`
- `--commission`, `--slippage` — shared between env and backtest
- `--turnover-penalty` — extra training regularization (default 0)
- `--output data/runs` — write `metrics.csv`, `metrics.json`, and `run_meta.json`

## What it does

- **Data** — Yahoo Finance OHLCV, Pydantic validation, aligned closes, CSV cache, optional async fetch, SQLAlchemy storage (SQLite or Postgres).
- **Backtest** — Vectorized daily rebalance with commission and slippage on turnover.
- **Metrics** — Cumulative and annualized return, volatility, max drawdown, Sharpe, Sortino, Calmar.
- **Baselines** — Equal weight, buy-and-hold (`bh_0`, `bh_SPY`, etc.), 60/40 (two assets), inverse-vol risk parity.
- **Splits** — By date or fraction for train / test.
- **RL** — `PortfolioEnv` (Gymnasium): lookback returns plus current weights in the observation; softmax actions; PPO via Stable-Baselines3; rollout to a weight matrix for backtest comparison.

## Layout

```
aivestor/
  data/          # Yahoo fetch, validation, load_prices
  db/            # Schema, upsert, load closes
  backtest/      # Engine and cost model
  config.py      # Shared backtest / env settings
  metrics.py
  baselines.py
  splits.py
  evaluation.py
  rl/            # Env and rollout
  scripts/       # CLI entrypoints
  api/           # FastAPI REST
web/             # TypeScript dashboard (Vite)
tests/
```

## Docker

Build and run the API plus dashboard on port 8000:

```bash
# Train locally first so PPO shows up in the UI (optional)
python -m aivestor.scripts.fetch_data --tickers SPY AGG GLD --start 2015-01-01
python -m aivestor.scripts.train_rl --timesteps 50000 --data-source cache

docker compose build api
docker compose up api
# http://127.0.0.1:8000
```

`data/cache` and `data/models` are mounted from your machine. On a fresh image with no volumes, the entrypoint downloads prices from Yahoo on first start.

Postgres only (optional): `docker compose up -d postgres`

## Azure (Container Apps)

```bash
az login
az extension add --name containerapp --upgrade
./scripts/deploy-azure.sh
```

Details: [infra/azure-setup.md](infra/azure-setup.md)

GitHub Actions: run the **Deploy to Azure** workflow (needs `AZURE_CREDENTIALS` secret). CI runs `pytest` on every push via [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Design notes

- Training uses only the train window; evaluation scripts always score the test window so you are not tuning on holdout data.
- Step rewards in the env use the same `CostModel` as the backtest engine by default (`--turnover-penalty` is optional).
- Simulation only—not live trading. Past backtests do not predict future returns.

## Tests

```bash
pytest
```

## Disclaimer

For research and education only. Not financial advice.
