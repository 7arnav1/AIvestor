# AIvestor

**Cloud-ready portfolio simulation with reinforcement learning**

AIvestor downloads historical market data, simulates multi-asset portfolios with transaction costs, trains a **PPO** (Proximal Policy Optimization) allocation policy on a train window, and evaluates baselines versus the learned policy on a **held-out test** period. Results are available via CLI, REST API, and a TypeScript dashboard.

> **Research and education only.** Past simulation does not guarantee future results. Not financial advice.

---

## Demo

<!-- Replace the link below with your recorded walkthrough (YouTube, Loom, or GitHub upload). -->

**[Watch demo video](https://github.com/7arnav1/AIvestor#demo)** — *add your link after recording*

Quick local demo (about 2 minutes after setup):

1. Start the API: `python -m aivestor.api`
2. Build and open the UI: `cd web && npm run build && npm run dev` → http://localhost:5173
3. Click **Run evaluation** to see out-of-sample metrics, equity curves, and PPO weights.

Step-by-step recording script: [docs/DEMO.md](docs/DEMO.md)

---

## Features

- **Data pipeline** — Yahoo Finance OHLCV, Pydantic validation, CSV cache, optional SQLAlchemy storage (SQLite / PostgreSQL)
- **Backtesting** — Vectorized daily rebalance with commission and slippage on turnover
- **Baselines** — Equal weight, buy-and-hold, 60/40, inverse-volatility risk parity
- **RL agent** — Gymnasium `PortfolioEnv` + Stable-Baselines3 PPO; softmax portfolio weights
- **Honest evaluation** — Train on first 70% of dates (configurable); report metrics only on the test slice
- **REST API** — FastAPI (`/api/health`, `/api/model`, `/api/evaluate`)
- **Dashboard** — TypeScript + Vite UI (metrics table, equity chart, latest allocation)
- **Deployment** — Docker / docker-compose; Azure Container Apps script included

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Yahoo/cache │ ──► │ Train (PPO)  │ ──► │ Model (.zip)    │
│  + optional │     │  70% dates   │     └────────┬────────┘
│     DB      │     └──────────────┘              │
└──────┬──────┘                                   ▼
       │                              ┌──────────────────────┐
       └────────────────────────────► │ Evaluate (test 30%)  │
                                      │ baselines + PPO      │
                                      └──────────┬───────────┘
                                                 ▼
                                      ┌──────────────────────┐
                                      │ FastAPI + TS dashboard │
                                      └──────────────────────┘
```

---

## Quick start

### Requirements

- Python 3.10+
- Node.js 18+ (for the dashboard)

### Install

```bash
git clone https://github.com/7arnav1/AIvestor.git
cd AIvestor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[rl,api,dev]"
```

### Run the full pipeline

```bash
# 1. Download and cache prices
python -m aivestor.scripts.fetch_data \
  --tickers SPY AGG GLD --start 2015-01-01

# 2. Train PPO on the train split (~few min CPU)
python -m aivestor.scripts.train_rl \
  --timesteps 50000 --data-source cache

# 3. Evaluate on the test split
python -m aivestor.scripts.evaluate \
  --model data/models/ppo_portfolio \
  --data-source cache \
  --baselines equal bh_0 risk_parity \
  --output data/runs
```

### Dashboard

```bash
# Terminal 1
python -m aivestor.api

# Terminal 2
cd web && npm install && npm run dev
```

Open http://localhost:5173 (dev) or build UI and use http://127.0.0.1:8000 (single server).

### Tests

```bash
pytest
```

---

## Docker

```bash
python -m aivestor.scripts.fetch_data --tickers SPY AGG GLD --start 2015-01-01
python -m aivestor.scripts.train_rl --timesteps 50000 --data-source cache

docker compose build api
docker compose up api
```

Visit http://127.0.0.1:8000. Local `data/cache` and `data/models` are mounted as volumes.

---

## Azure

```bash
az login
az extension add --name containerapp --upgrade
./scripts/deploy-azure.sh
```

See [infra/azure-setup.md](infra/azure-setup.md) for details and model/data mounting.

---

## Project structure

| Path | Description |
|------|-------------|
| `aivestor/data/` | Fetch, validate, `load_prices` (cache / DB / Yahoo) |
| `aivestor/backtest/` | Simulation engine and cost model |
| `aivestor/rl/` | Gymnasium environment and PPO rollout |
| `aivestor/api/` | FastAPI application |
| `aivestor/scripts/` | CLI: `fetch_data`, `train_rl`, `evaluate` |
| `web/` | TypeScript dashboard (Vite) |
| `tests/` | pytest suite |
| `docs/DEMO.md` | Demo recording guide |

---

## Configuration

| Variable / flag | Purpose |
|-----------------|--------|
| `DATABASE_URL` | SQLite default (`data/aivestor.db`) or PostgreSQL |
| `--data-source` | `auto`, `cache`, `db`, or `yahoo` |
| `--train-fraction` | Default `0.7` train / `0.3` test |
| `--train-end` | Fixed date split instead of fraction |
| `--commission`, `--slippage` | Shared between RL env and backtest |

Copy `.env.example` to `.env` for database overrides.

---

## Tech stack

Python · NumPy · pandas · Pydantic · yfinance · SQLAlchemy · Gymnasium · Stable-Baselines3 · FastAPI · Uvicorn · TypeScript · Vite · Docker · Azure Container Apps

---

## License

Use for portfolio and learning. Add a license file if you open-source formally.

---

## Disclaimer

For research and education only. Not financial advice.
