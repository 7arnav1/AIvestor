"""Static allocation strategies as weight matrices aligned with price rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(n_assets: int, n_rows: int) -> np.ndarray:
    w = 1.0 / n_assets
    return np.full((n_rows, n_assets), w, dtype=np.float64)


def buy_and_hold_asset(asset_index: int, n_assets: int, n_rows: int) -> np.ndarray:
    w = np.zeros((n_rows, n_assets), dtype=np.float64)
    w[:, asset_index] = 1.0
    return w


def static_weights(weights: list[float] | np.ndarray, n_rows: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w))
    if s < 1e-12:
        raise ValueError("weights must sum to a positive value")
    w = w / s
    n = w.size
    return np.tile(w, (n_rows, 1))


def risk_parity_approx(volatilities: np.ndarray, n_rows: int) -> np.ndarray:
    """Inverse-vol weights from per-asset vol estimates (e.g. trailing std of returns)."""
    v = np.asarray(volatilities, dtype=np.float64)
    v = np.clip(v, 1e-12, None)
    inv = 1.0 / v
    w = inv / np.sum(inv)
    return np.tile(w, (n_rows, 1))


def from_strategy_name(
    name: str,
    prices: pd.DataFrame,
    *,
    asset_order: list[str] | None = None,
) -> np.ndarray:
    """
    Named baselines: 'equal', 'bh_0' (first column), 'bh_SPY' if column exists,
    '60_40' if exactly two columns (first equity, second bonds).
    """
    cols = list(prices.columns) if asset_order is None else asset_order
    n = len(cols)
    t = len(prices)
    if name == "equal":
        return equal_weight(n, t)
    if name == "risk_parity":
        rets = prices.pct_change().iloc[1:]
        vols = rets.std().to_numpy()
        if np.any(~np.isfinite(vols)) or np.all(vols < 1e-12):
            raise ValueError("need return history to estimate vol for risk_parity")
        return risk_parity_approx(vols, t)
    if name == "60_40" and n == 2:
        return static_weights([0.6, 0.4], t)
    if name in cols:
        return buy_and_hold_asset(cols.index(name), n, t)
    if name.startswith("bh_"):
        rest = name[3:]
        if rest.isdigit():
            return buy_and_hold_asset(int(rest), n, t)
    raise ValueError(f"Unknown baseline strategy: {name}")
