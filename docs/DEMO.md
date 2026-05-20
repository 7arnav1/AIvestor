# AIvestor — Demo recording guide

Use this when filming a walkthrough for GitHub, LinkedIn, or interviews.

## Before you record

```bash
cd AIvestor
source .venv/bin/activate
pip install -e ".[rl,api]"

# Ensure data + model exist
python -m aivestor.scripts.fetch_data --tickers SPY AGG GLD --start 2015-01-01
python -m aivestor.scripts.train_rl --timesteps 50000 --data-source cache

cd web && npm install && npm run build && cd ..
```

Run evaluate once so the UI is fast:

```bash
python -m aivestor.scripts.evaluate --model data/models/ppo_portfolio --data-source cache
```

## Recording setup

- Terminal font size: large
- Close unrelated windows
- Two terminals ready (API + UI) or single server after `npm run build`

## Suggested script (~5 min)

| Time | What to show | What to say |
|------|----------------|-------------|
| 0:00 | GitHub repo README | "AIvestor simulates multi-asset portfolios on historical data." |
| 0:30 | Folder tree | "Python core for data, backtest, and PPO; FastAPI; TypeScript UI." |
| 1:00 | `data/cache/closes.csv` | "Prices are cached for reproducible runs." |
| 1:30 | `ppo_portfolio.zip` | "PPO trains on the first 70% of dates only." |
| 2:00 | Start API + UI | `python -m aivestor.api` and `cd web && npm run dev` |
| 2:30 | Dashboard → Run evaluation | "This scores everything on the held-out test window." |
| 3:30 | Metrics table | "Compare equal weight, buy-and-hold, risk parity, and PPO." |
| 4:00 | Equity chart | "Out-of-sample equity curves after costs." |
| 4:20 | PPO weights | "Latest learned allocation across SPY, AGG, GLD." |
| 4:40 | Closing | "Simulation only—not live trading. Stack: Python, RL, FastAPI, TypeScript, Docker, Azure-ready." |

## After recording

1. Upload video (YouTube unlisted, Loom, or GitHub asset).
2. Update the demo link in `README.md`:

   ```markdown
   **[Watch demo](https://your-video-url-here)**
   ```

3. Commit and push only `README.md` (demo link change) if you want the link public.

## Single-server shortcut (one URL)

```bash
cd web && npm run build && cd ..
python -m aivestor.api
# Open http://127.0.0.1:8000
```

Good for a single-browser screen recording.
