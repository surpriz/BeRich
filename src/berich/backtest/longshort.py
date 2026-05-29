"""Dollar-neutral long/short cross-sectional backtest.

Consumes the out-of-sample cross-sectional scores (:class:`CrossSectionalOof`): on each
rebalance date it ranks names, goes long the top decile and short the bottom decile, sizes
each leg (equal or inverse-vol), and holds until the next rebalance. Daily portfolio
returns are net of turnover cost (fees + slippage on L1 weight change) and a daily short
borrow charge. An optional vol-target overlay scales the whole book to a target annualized
vol. There is **no buy-&-hold benchmark** — success is a positive, statistically
significant Sharpe (see :mod:`berich.backtest.significance`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.backtest.significance import SharpeSignificance, assess_sharpe
from berich.risk.sizing import inverse_vol_size, vol_target_size

if TYPE_CHECKING:
    from berich.training.cross_sectional import CrossSectionalOof

TRADING_DAYS = 252


@dataclass
class LongShortConfig:
    """Parameters for the dollar-neutral long/short backtest."""

    top_decile: float = 0.1
    bottom_decile: float = 0.1
    weighting: str = "inverse_vol"  # "equal" | "inverse_vol"
    rebalance_days: int = 5
    gross_leverage: float = 1.0
    target_vol: float = 0.10  # annualized; 0 disables the overlay
    vol_lookback: int = 20
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    min_names: int = 10


@dataclass
class LongShortResult:
    """Outcome of a long/short backtest."""

    metrics: PerfMetrics
    returns: pd.Series
    significance: SharpeSignificance
    avg_gross_exposure: float
    n_rebalances: int


def _leg_weights(
    names: list[str],
    vols: pd.Series,
    *,
    sign: float,
    budget: float,
    weighting: str,
) -> dict[str, float]:
    """Weights for one leg summing (in absolute value) to ``budget``."""
    if not names:
        return {}
    if weighting == "inverse_vol":
        ref = float(vols.loc[names].median())
        raw = {t: inverse_vol_size(float(vols.get(t, np.nan)), ref, ceiling=5.0) for t in names}
        total = sum(raw.values()) or float(len(names))
        return {t: sign * budget * (w / total) for t, w in raw.items()}
    w = budget / len(names)
    return dict.fromkeys(names, sign * w)


def build_baskets(
    oof: CrossSectionalOof,
    ret: pd.DataFrame,
    config: LongShortConfig,
) -> pd.DataFrame:
    """Target-weight matrix (rebalance_date x ticker), long top / short bottom decile."""
    vol_wide = ret.rolling(config.vol_lookback, min_periods=config.vol_lookback // 2).std()
    dates = sorted(oof.frame.index.unique())
    rebalance_dates = dates[:: config.rebalance_days]
    budget = 0.5 * config.gross_leverage

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for rd in rebalance_dates:
        day = oof.frame.loc[[rd]]
        if len(day) < config.min_names:
            continue
        ranked = day.sort_values("score")
        n_short = max(1, int(len(ranked) * config.bottom_decile))
        n_long = max(1, int(len(ranked) * config.top_decile))
        short_names = ranked["ticker"].iloc[:n_short].tolist()
        long_names = ranked["ticker"].iloc[-n_long:].tolist()
        vols = vol_wide.loc[rd] if rd in vol_wide.index else pd.Series(dtype=float)
        weights = {
            **_leg_weights(long_names, vols, sign=1.0, budget=budget, weighting=config.weighting),
            **_leg_weights(short_names, vols, sign=-1.0, budget=budget, weighting=config.weighting),
        }
        rows[rd] = weights

    if not rows:
        return pd.DataFrame(columns=ret.columns)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=ret.columns).fillna(0.0)


def returns_from_weights(
    baskets: pd.DataFrame,
    ret: pd.DataFrame,
    config: LongShortConfig,
) -> tuple[pd.Series, float]:
    """Daily net returns from a (rebalance_date x ticker) weight matrix + a returns matrix.

    Holds weights between rebalances, charges turnover + borrow, applies the vol-target
    overlay. Shared by the backtester and the paper-book equity replay. Returns
    ``(net_returns, avg_gross_exposure)``.
    """
    if baskets.empty:
        return pd.Series(dtype=float), 0.0
    held = baskets.reindex(ret.index).ffill().fillna(0.0)
    effective = held.shift(1).fillna(0.0)

    gross_ret = (effective * ret).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = turnover * (config.fee_bps + config.slippage_bps) / 1e4
    short_gross = effective.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_gross * config.borrow_bps_annual / TRADING_DAYS / 1e4
    net_ret = (gross_ret - cost - borrow).loc[baskets.index[0] :].dropna()

    if config.target_vol and config.target_vol > 0:
        roll_vol = net_ret.rolling(config.vol_lookback).std() * np.sqrt(TRADING_DAYS)
        scale = roll_vol.apply(
            lambda v: vol_target_size(float(v), target_vol=config.target_vol, ceiling=3.0)
        )
        net_ret = net_ret * scale.shift(1).fillna(1.0)

    net_ret = net_ret.dropna()
    gross_series = effective.abs().sum(axis=1).loc[net_ret.index]
    avg_gross = float(gross_series.mean()) if len(net_ret) else 0.0
    return net_ret, avg_gross


def run_longshort_backtest(
    prices_by_ticker: dict[str, pd.DataFrame],
    oof: CrossSectionalOof,
    config: LongShortConfig,
    *,
    n_trials: int = 1,
) -> LongShortResult:
    """Simulate the dollar-neutral long/short book and assess Sharpe significance."""
    tickers = sorted(oof.frame["ticker"].unique())
    close = pd.DataFrame(
        {t: prices_by_ticker[t]["close"] for t in tickers if t in prices_by_ticker}
    ).sort_index()
    ret = close.pct_change(fill_method=None)

    baskets = build_baskets(oof, ret, config)
    if baskets.empty:
        empty = pd.Series(dtype=float)
        return LongShortResult(compute_metrics(empty), empty, assess_sharpe(empty), 0.0, 0)

    net_ret, avg_gross = returns_from_weights(baskets, ret, config)
    return LongShortResult(
        metrics=compute_metrics(net_ret),
        returns=net_ret,
        significance=assess_sharpe(net_ret, n_trials=n_trials),
        avg_gross_exposure=avg_gross,
        n_rebalances=len(baskets),
    )


__all__ = [
    "LongShortConfig",
    "LongShortResult",
    "build_baskets",
    "returns_from_weights",
    "run_longshort_backtest",
]
