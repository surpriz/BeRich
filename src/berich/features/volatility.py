"""Causal one-step-ahead volatility forecast for adaptive stop-loss / take-profit.

The triple-barrier serving code historically set barriers at a fixed ATR multiple. A vol
forecast lets the barrier width track *expected* volatility instead of trailing realized
range: wider stops/targets when the model expects turbulence, tighter when it expects calm.

EWMA (RiskMetrics, λ=0.94) is the default — three lines of math, runs in microseconds, and
is causal by construction (it only ever sees returns up to ``t``). A GARCH(1,1) path is
available when the optional ``arch`` package is installed; it falls back to EWMA otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_EWMA_LAMBDA = 0.94


@dataclass
class VolForecast:
    """One-step-ahead daily vol and its horizon-scaled counterpart."""

    sigma_daily: float  # forecast daily log-return std (causal, uses data <= t)
    horizon_sigma: float  # sigma_daily * sqrt(horizon_days)
    method: str  # "ewma" | "garch"


def _ewma_sigma(returns: np.ndarray, lam: float) -> float:
    """EWMA one-step-ahead daily vol: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_t^2."""
    var = float(np.var(returns))
    for r in returns:
        var = lam * var + (1.0 - lam) * r * r
    return math.sqrt(max(var, 0.0))


def _garch_sigma(returns: np.ndarray) -> float | None:
    """GARCH(1,1) one-step-ahead daily vol, or None if ``arch`` is unavailable/fails."""
    try:
        from arch import arch_model  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError:
        return None
    try:
        scaled = returns * 100.0  # arch is better-conditioned on percent returns
        res = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero").fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        return float(np.sqrt(fc.variance.to_numpy()[-1, 0])) / 100.0
    except Exception:  # noqa: BLE001 — any arch failure degrades to the EWMA fallback
        logger.debug("GARCH fit failed; falling back to EWMA", exc_info=True)
        return None


def forecast_vol(
    close: pd.Series,
    *,
    horizon_days: int,
    method: str = "ewma",
    lookback: int = 252,
    lam: float = _EWMA_LAMBDA,
) -> VolForecast:
    """Forecast next-step daily vol from log returns up to the last bar (causal).

    Returns a degenerate zero forecast when there is too little history to estimate vol.
    """
    log_close = np.log(close.astype(float).to_numpy())
    rets = np.diff(log_close)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:  # noqa: PLR2004
        return VolForecast(0.0, 0.0, method)
    rets = rets[-lookback:]

    sigma = _garch_sigma(rets) if method == "garch" else None
    used = "garch"
    if sigma is None:
        sigma = _ewma_sigma(rets, lam)
        used = "ewma"

    return VolForecast(
        sigma_daily=sigma,
        horizon_sigma=sigma * math.sqrt(max(horizon_days, 1)),
        method=used,
    )


__all__ = ["VolForecast", "forecast_vol"]
