"""FastAPI app for AIvestor dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aivestor.api.service import model_info, run_evaluation

app = FastAPI(title="AIvestor API", version="0.1.0")

_cors = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _web_dist_dir() -> Path | None:
    env = os.getenv("WEB_DIST")
    candidates = [
        Path(env) if env else None,
        Path(__file__).resolve().parents[2] / "web" / "dist",
        Path("/app/web/dist"),
    ]
    for p in candidates:
        if p is not None and p.is_dir():
            return p
    return None


class EvaluateRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "AGG", "GLD"])
    start: str = "2015-01-01"
    train_fraction: float = 0.7
    baselines: list[str] = Field(default_factory=lambda: ["equal", "bh_0", "risk_parity"])
    data_source: str = "cache"


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/model")
def get_model():
    return model_info()


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest):
    try:
        return run_evaluation(
            tickers=body.tickers,
            start=body.start,
            train_fraction=body.train_fraction,
            baselines=body.baselines,
            data_source=body.data_source,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


_web_dist = _web_dist_dir()
if _web_dist is not None:
    app.mount("/", StaticFiles(directory=str(_web_dist), html=True), name="web")
