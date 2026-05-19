"""Business logic for the HTTP API (wraps existing backtest / eval code)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aivestor.baselines import from_strategy_name
from aivestor.config import make_run_configs
from aivestor.data.loader import load_prices
from aivestor.evaluation import compare_weights, run_strategy
from aivestor.rl.rollout import weights_from_ppo
from aivestor.splits import split_by_date, split_by_fraction

DEFAULT_TICKERS = ["SPY", "AGG", "GLD"]
DEFAULT_MODEL = Path("data/models/ppo_portfolio")


def model_info(model_path: Path = DEFAULT_MODEL) -> dict:
    zip_path = model_path.with_suffix(".zip")
    path = zip_path if zip_path.exists() else model_path
    if not path.exists():
        return {"trained": False, "path": str(model_path), "message": "No saved model yet. Run train_rl."}
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return {
        "trained": True,
        "path": str(path),
        "size_kb": round(stat.st_size / 1024, 1),
        "modified_utc": mtime.isoformat(),
    }


def run_evaluation(
    *,
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    train_fraction: float = 0.7,
    train_end: str | None = None,
    baselines: list[str] | None = None,
    model_path: Path = DEFAULT_MODEL,
    data_source: str = "cache",
    cache_dir: Path = Path("data/cache"),
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    baselines = baselines or ["equal", "bh_0", "risk_parity"]

    prices = load_prices(tickers, start=start, source=data_source, cache_dir=cache_dir)
    if train_end:
        _, test = split_by_date(prices, train_end)
    else:
        _, test = split_by_fraction(prices, train_fraction)

    cfg_bt, cfg_env = make_run_configs()
    strategies: dict[str, np.ndarray] = {}
    for b in baselines:
        try:
            strategies[f"baseline_{b}"] = from_strategy_name(b, test)
        except ValueError:
            pass

    info = model_info(model_path)
    if info["trained"]:
        from stable_baselines3 import PPO

        load_stem = Path(info["path"]).with_suffix("")
        model = PPO.load(str(load_stem))
        strategies["ppo"] = weights_from_ppo(model, test, cfg_env)

    if not strategies:
        raise ValueError("No strategies available to evaluate.")

    metrics_df = compare_weights(test, strategies, cfg_bt)
    metrics = metrics_df.replace({np.nan: None}).to_dict(orient="records")

    curves = []
    dates = [d.strftime("%Y-%m-%d") for d in test.index]
    for name, w in strategies.items():
        out = run_strategy(test, w, cfg_bt)
        eq = out["raw"]["equity"]
        curves.append(
            {
                "strategy": name,
                "dates": dates,
                "equity": [float(x) for x in eq],
            }
        )

    last_ppo = None
    if "ppo" in strategies:
        w = strategies["ppo"][-1]
        last_ppo = {t: float(wi) for t, wi in zip(test.columns, w)}

    return {
        "tickers": tickers,
        "test_start": dates[0],
        "test_end": dates[-1],
        "test_days": len(dates),
        "train_fraction": train_fraction,
        "model": info,
        "metrics": metrics,
        "equity_curves": curves,
        "latest_ppo_weights": last_ppo,
    }
