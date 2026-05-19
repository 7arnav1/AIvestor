from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class OHLCVBar(BaseModel):
    """Single bar schema for downstream pipelines."""

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def ohlc_consistent(self) -> OHLCVBar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


REQUIRED_COLS = ("Open", "High", "Low", "Close", "Volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaNs, enforce column names, run row-wise schema checks."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[list(REQUIRED_COLS)].copy()
    out = out.dropna()
    for _, row in out.iterrows():
        OHLCVBar(
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )
    return out
