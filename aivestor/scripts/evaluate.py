"""Compare baselines (and optional trained PPO) on a held-out test window."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from aivestor.baselines import from_strategy_name
from aivestor.config import make_run_configs
from aivestor.data.loader import load_prices
from aivestor.evaluation import compare_weights, format_metrics_table
from aivestor.rl.rollout import weights_from_ppo
from aivestor.splits import split_by_date, split_by_fraction


def main() -> None:
    ap = argparse.ArgumentParser(description="AIvestor — evaluate strategies on test data")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "AGG", "GLD"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--train-end", default=None, help="If set, train is before this date (YYYY-MM-DD)")
    ap.add_argument("--train-fraction", type=float, default=0.7, help="Used when --train-end omitted")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--model", type=Path, default=None, help="PPO save path (Stable-Baselines3 .zip)")
    ap.add_argument("--baselines", nargs="+", default=["equal", "bh_0", "risk_parity"])
    ap.add_argument(
        "--data-source",
        choices=["auto", "yahoo", "cache", "db"],
        default="auto",
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--database-url", default=None)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write metrics.csv and metrics.json",
    )
    args = ap.parse_args()

    prices = load_prices(
        list(args.tickers),
        start=args.start,
        source=args.data_source,
        cache_dir=args.cache_dir,
        database_url=args.database_url,
    )

    if args.train_end:
        _, test = split_by_date(prices, args.train_end)
    else:
        _, test = split_by_fraction(prices, args.train_fraction)

    cfg_bt, cfg_env = make_run_configs(
        commission=args.commission,
        slippage=args.slippage,
        lookback=args.lookback,
    )

    weights: dict[str, object] = {}
    for b in args.baselines:
        try:
            weights[f"baseline_{b}"] = from_strategy_name(b, test)
        except ValueError as err:
            print(f"[skip baseline {b}] {err}")

    if args.model is not None:
        load_path = args.model
        if not load_path.exists() and load_path.with_suffix(".zip").exists():
            load_path = load_path.with_suffix(".zip")
        if load_path.exists():
            from stable_baselines3 import PPO

            load_stem = load_path.with_suffix("") if load_path.suffix == ".zip" else load_path
            model = PPO.load(str(load_stem))
            weights["ppo"] = weights_from_ppo(model, test, cfg_env)
        else:
            print(f"[skip ppo] file not found: {args.model}")

    if not weights:
        raise SystemExit("No strategies to evaluate.")

    df = compare_weights(test, weights, cfg_bt)
    table = format_metrics_table(df)
    print(table)

    if args.output is not None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = args.output / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "metrics.csv", index=False)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
        meta = {
            "tickers": list(args.tickers),
            "start": args.start,
            "train_fraction": args.train_fraction,
            "train_end": args.train_end,
            "test_rows": len(test),
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
        }
        with open(run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metrics to {run_dir}")


if __name__ == "__main__":
    main()
