"""Triple-barrier labeling.

For each bar ``t`` we open a hypothetical long and watch the next ``horizon`` bars.
Two horizontal barriers sit at ``close[t] ± k * ATR[t]`` (k from config) and a
vertical barrier sits ``horizon`` bars ahead. The label is:

* ``1``  — the upper (take-profit) barrier is hit first  → an up-trend materialized;
* ``-1`` — the lower (stop-loss) barrier is hit first;
* ``0``  — neither is hit within the horizon (time barrier wins).

The model is trained to predict ``P(label == 1)`` — the probability that a swing
long would have reached its target before its stop. This is the trend-probability
target chosen during design. Labels look *forward*, which is correct for a target;
features never do. The last ``horizon`` bars get NaN labels (incomplete) and must be
dropped before training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from berich.features.indicators import atr

if TYPE_CHECKING:
    from berich.features.volatility import VolForecast


class LabelConfig(BaseModel):
    """Triple-barrier parameters (mirrors the ``labeling`` block in YAML)."""

    horizon_days: int = 10
    atr_window: int = 14
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 1.0
    direction: Literal["long", "short"] = "long"


# Adaptive barrier scaling is clipped to this band so a single noisy vol estimate can't
# blow the stop out (or collapse it) relative to the configured ATR width.
_ADAPTIVE_SCALE_MIN = 0.5
_ADAPTIVE_SCALE_MAX = 2.5


def adaptive_barriers(
    entry: float,
    atr_t: float,
    vol_forecast: VolForecast,
    config: LabelConfig,
    *,
    quantiles: tuple[float, float] | None = None,
    direction: Literal["long", "short"] = "long",
) -> tuple[float, float, dict[str, float | str]]:
    """Derive (stop, target) from a vol forecast or predicted return quantiles.

    - When ``quantiles`` (q_low_ret, q_high_ret) are supplied (a distributional model's
      forward-return band), barriers are placed directly at those return levels.
    - Otherwise the configured ATR multipliers are scaled by the ratio of forecasted
      daily vol to the ATR-implied daily range, clipped to a sane band.

    For a short the profit target sits below entry and the stop above, so the ATR
    barriers are mirrored about ``entry`` and the favorable/adverse quantile sides swap.

    Returns ``(stop, target, rationale)`` where ``rationale`` explains the choice.
    """
    if quantiles is not None:
        q_low, q_high = quantiles
        if direction == "short":
            target = entry * (1.0 + q_low)
            stop = entry * (1.0 + q_high)
        else:
            target = entry * (1.0 + q_high)
            stop = entry * (1.0 + q_low)
        return (
            stop,
            target,
            {"method": "quantile", "q_low": q_low, "q_high": q_high, "direction": direction},
        )

    atr_pct = atr_t / entry if entry > 0 else 0.0
    if atr_pct > 0 and vol_forecast.sigma_daily > 0:
        scale = float(
            np.clip(vol_forecast.sigma_daily / atr_pct, _ADAPTIVE_SCALE_MIN, _ADAPTIVE_SCALE_MAX)
        )
    else:
        scale = 1.0
    if direction == "short":
        target = entry - config.take_profit_atr * scale * atr_t
        stop = entry + config.stop_loss_atr * scale * atr_t
    else:
        target = entry + config.take_profit_atr * scale * atr_t
        stop = entry - config.stop_loss_atr * scale * atr_t
    return (
        stop,
        target,
        {
            "method": "vol_scaled",
            "scale": scale,
            "sigma_daily": vol_forecast.sigma_daily,
            "horizon_sigma": vol_forecast.horizon_sigma,
            "direction": direction,
        },
    )


def triple_barrier_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Compute triple-barrier outcomes for every bar of an OHLCV frame.

    Returns a frame indexed like ``df`` with columns:
    ``label`` (-1/0/1), ``ret`` (realized return at the touch/time barrier),
    ``bars_held`` (bars until the barrier), and ``sample_weight`` (|ret|, for
    emphasizing decisive moves). Rows without a full forward horizon are NaN.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_vals = atr(df["high"], df["low"], df["close"], config.atr_window).to_numpy(dtype=float)

    n = len(df)
    horizon = config.horizon_days
    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    held = np.full(n, np.nan)

    for t in range(n):
        if t + horizon >= n or np.isnan(atr_vals[t]):
            continue  # incomplete forward window or ATR not warmed up
        entry = close[t]
        if config.direction == "short":
            tp_barrier = entry - config.take_profit_atr * atr_vals[t]
            sl_barrier = entry + config.stop_loss_atr * atr_vals[t]
        else:
            tp_barrier = entry + config.take_profit_atr * atr_vals[t]
            sl_barrier = entry - config.stop_loss_atr * atr_vals[t]

        label, ret, bars = _first_touch(
            high[t + 1 : t + horizon + 1],
            low[t + 1 : t + horizon + 1],
            close[t + 1 : t + horizon + 1],
            entry=entry,
            tp_barrier=tp_barrier,
            sl_barrier=sl_barrier,
            direction=config.direction,
        )
        labels[t], rets[t], held[t] = label, ret, bars

    out = pd.DataFrame(
        {"label": labels, "ret": rets, "bars_held": held},
        index=df.index,
    )
    out["sample_weight"] = out["ret"].abs()
    return out


def _first_touch(
    fwd_high: np.ndarray,
    fwd_low: np.ndarray,
    fwd_close: np.ndarray,
    *,
    entry: float,
    tp_barrier: float,
    sl_barrier: float,
    direction: Literal["long", "short"] = "long",
) -> tuple[int, float, int]:
    """Return (label, realized_return, bars_held) for one entry's forward window.

    For a long the take-profit barrier sits above entry (touched on the forward
    high) and the stop below (touched on the forward low). For a short these sides
    swap, and ``ret`` is measured as ``entry / exit - 1`` so a winning short is
    positive. Label is ``1`` when the take-profit is hit first, ``-1`` for the stop.
    """
    for i in range(len(fwd_high)):
        if direction == "short":
            hit_tp = fwd_low[i] <= tp_barrier
            hit_sl = fwd_high[i] >= sl_barrier
            tp_ret = entry / tp_barrier - 1.0
            sl_ret = entry / sl_barrier - 1.0
        else:
            hit_tp = fwd_high[i] >= tp_barrier
            hit_sl = fwd_low[i] <= sl_barrier
            tp_ret = tp_barrier / entry - 1.0
            sl_ret = sl_barrier / entry - 1.0
        if hit_tp and hit_sl:
            # Both barriers inside one bar — resolve conservatively as a stop.
            return -1, sl_ret, i + 1
        if hit_tp:
            return 1, tp_ret, i + 1
        if hit_sl:
            return -1, sl_ret, i + 1
    # Time barrier: label by the sign of the realized return at horizon end.
    exit_close = fwd_close[-1]
    final_ret = entry / exit_close - 1.0 if direction == "short" else exit_close / entry - 1.0
    return 0, final_ret, len(fwd_high)
