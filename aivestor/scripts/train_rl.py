"""Train PPO on PortfolioEnv using a train-only slice (test reserved for evaluate.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from aivestor.config import make_run_configs
from aivestor.data.loader import load_prices
from aivestor.rl.env import PortfolioEnv
from aivestor.splits import split_by_date, split_by_fraction


def main() -> None:
    ap = argparse.ArgumentParser(description="AIvestor — PPO portfolio allocation")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "AGG", "GLD"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--train-end", default=None, help="Train on rows before this date (exclusive)")
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--turnover-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=Path, default=Path("data/models/ppo_portfolio"))
    ap.add_argument(
        "--data-source",
        choices=["auto", "yahoo", "cache", "db"],
        default="auto",
        help="Where to load prices (default: cache/db then Yahoo)",
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    prices = load_prices(
        list(args.tickers),
        start=args.start,
        source=args.data_source,
        cache_dir=args.cache_dir,
        database_url=args.database_url,
    )
    if prices.shape[1] < 2:
        raise SystemExit("Need at least 2 tickers for a portfolio env.")

    if args.train_end:
        train, _ = split_by_date(prices, args.train_end)
    else:
        train, _ = split_by_fraction(prices, args.train_fraction)

    if len(train) < args.lookback + 5:
        raise SystemExit("Train window too short for lookback; widen dates or lower --lookback.")

    _, env_cfg = make_run_configs(
        commission=args.commission,
        slippage=args.slippage,
        lookback=args.lookback,
    )

    def make_env():
        return PortfolioEnv(train, env_cfg, turnover_penalty=args.turnover_penalty)

    venv = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    base = args.save
    if base.suffix == ".zip":
        base = base.with_suffix("")
    model.save(str(base))
    print(f"Saved model to {base}.zip")
    print("Evaluate on the test split: python -m aivestor.scripts.evaluate --model", base)


if __name__ == "__main__":
    main()
