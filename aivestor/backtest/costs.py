from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Proportional transaction costs as fractions (not bps)."""

    commission_rate: float = 0.0
    slippage_rate: float = 0.0

    def total_on_turnover(self, turnover: float) -> float:
        return turnover * (self.commission_rate + self.slippage_rate)


def linear_slippage_bps(bps: float) -> float:
    return bps / 10_000.0
