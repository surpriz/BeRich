"""Risk-aware PEAD backtest (Phase 8).

The plain :func:`berich.training.pead.run_pead_backtest` gives every
predicted-positive event a unit position. This module re-uses the same
signal (the OOF probas) but layers a :class:`~berich.risk.RiskOverlay` on
top: each event runs through the gates first (skipped if any blocks), then
gets a fractional position from the sizer. The result is a per-event
record + a daily equity curve so we can compare Sharpe / max-DD / total
return against the unfiltered baseline honestly.

The benchmark is identical to the daily-PEAD backtest's benchmark — "long
every (entry, exit) window with no signal filter" — so a passing Sharpe
gate here means risk management actually pulled real Sharpe out of the
PEAD signal, not just out of a more favorable benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.features.build import MARKET_TICKER
from berich.features.indicators import realized_vol
from berich.risk import RiskOverlay, RiskOverlayConfig
from berich.risk.sizing import annualize_daily_vol

if TYPE_CHECKING:
    from berich.data.store import OhlcvStore
    from berich.datasets.pead import PeadDataset
    from berich.training.pead import PeadOofResult

DEFAULT_HOLD_DAYS = 5
DEFAULT_ENTRY_SLIPPAGE_BPS = 10.0
DEFAULT_EXIT_SLIPPAGE_BPS = 5.0
DEFAULT_FEE_BPS = 1.0


@dataclass
class RiskAwarePeadResult:
    """Backtest outcome with explicit position-size + gate annotations per trade."""

    strategy: PerfMetrics
    benchmark: PerfMetrics
    trades: pd.DataFrame
    n_events_considered: int
    n_events_gated: int
    n_events_zero_sized: int

    @property
    def beats_buy_hold(self) -> bool:
        return self.strategy.sharpe > self.benchmark.sharpe


def _spy_rvol_series(store: OhlcvStore, window: int = 20) -> pd.Series:
    """SPY 20-day realized vol — annualized — used by the regime gate."""
    spy = store.load(MARKET_TICKER)
    if spy is None or spy.empty:
        return pd.Series(dtype=float)
    rvol = realized_vol(spy["close"], window) * np.sqrt(252.0)
    return rvol.dropna()


def _asset_vol_at(ohlcv: pd.DataFrame, as_of: pd.Timestamp, window: int = 20) -> float | None:
    """Annualized 20d rvol for ``ohlcv`` strictly at or before ``as_of``."""
    rvol = realized_vol(ohlcv["close"], window) * np.sqrt(252.0)
    prior = rvol.loc[rvol.index <= as_of].dropna()
    if prior.empty:
        return None
    return float(prior.iloc[-1])


def run_risk_aware_pead_backtest(  # noqa: PLR0915 — orchestrates many overlay branches
    dataset: PeadDataset,
    oof: PeadOofResult,
    store: OhlcvStore,
    *,
    config: RiskOverlayConfig | None = None,
    threshold: float = 0.5,
    hold_days: int = DEFAULT_HOLD_DAYS,
    entry_slippage_bps: float = DEFAULT_ENTRY_SLIPPAGE_BPS,
    exit_slippage_bps: float = DEFAULT_EXIT_SLIPPAGE_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> RiskAwarePeadResult:
    """Run the PEAD signal through the configured risk overlay event-by-event.

    Each candidate event goes through three checks in order:
      1. **Threshold**: proba must clear ``threshold``.
      2. **Gates** (regime, drawdown): every active gate must allow.
      3. **Sizer** (Kelly + inverse vol + vol target): final size > 0.

    Trades are simulated chronologically so the drawdown gate and the vol
    target both see only the already-realized strategy equity at the time of
    each decision. The benchmark is the same "blind long every event window"
    as the plain backtest, sized to the same per-event position so the
    Sharpe comparison isolates "did the gate + size help" rather than "did
    we skip the bad ones".
    """
    overlay = RiskOverlay(config)
    cfg = overlay.config
    fee = fee_bps / 1e4
    entry_slip = entry_slippage_bps / 1e4
    exit_slip = exit_slippage_bps / 1e4

    spy_rvol = _spy_rvol_series(store) if cfg.use_regime_gate else pd.Series(dtype=float)

    # Re-join OOF probas onto the original event frame.
    events = dataset.events.copy()
    events = events.merge(
        oof.frame[["ticker", "entry_date", "proba"]],
        on=["ticker", "entry_date"],
        how="left",
    )
    events = events.dropna(subset=["proba"]).copy()
    candidates = events[events["proba"] >= threshold].sort_values("entry_date")

    n_considered = len(candidates)
    n_gated = 0
    n_zero = 0

    trades: list[dict] = []
    strat_returns: list[tuple[pd.Timestamp, float]] = []
    bench_returns: list[tuple[pd.Timestamp, float]] = []

    # Running equity for the drawdown gate + vol target. Both are computed on
    # the realized strategy returns up to (but not including) the current
    # event, so no leak from the test window.
    strategy_equity: list[tuple[pd.Timestamp, float]] = []
    cumulative = 1.0

    for _, ev in candidates.iterrows():
        ticker = str(ev["ticker"])
        raw_entry_date = pd.Timestamp(ev["entry_date"])
        if pd.isna(raw_entry_date):
            continue
        entry_date: pd.Timestamp = raw_entry_date  # ty: ignore[invalid-assignment]
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty or entry_date not in ohlcv.index:
            continue
        entry_idx = int(ohlcv.index.get_loc(entry_date))
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(ohlcv):
            continue

        equity_series = (
            pd.Series([v for _, v in strategy_equity], index=[d for d, _ in strategy_equity])
            if strategy_equity
            else None
        )
        if not overlay.gates_pass(
            as_of=entry_date,
            spy_rvol_series=spy_rvol if cfg.use_regime_gate else None,
            strategy_equity=equity_series if cfg.use_drawdown_gate else None,
        ):
            n_gated += 1
            continue

        # Per-asset vol used by inverse-vol sizing + as the reference point for
        # vol-target (we use SPY's recent vol as the ref).
        asset_vol = _asset_vol_at(ohlcv, entry_date)
        ref_vol = (
            float(spy_rvol.loc[spy_rvol.index <= entry_date].iloc[-1])
            if not spy_rvol.empty and (spy_rvol.index <= entry_date).any()
            else None
        )
        if equity_series is not None and len(equity_series) > 21:  # noqa: PLR2004
            recent_returns = equity_series.pct_change().tail(20).dropna()
            portfolio_vol = annualize_daily_vol(float(recent_returns.std()))
        else:
            portfolio_vol = None

        size = overlay.position_size(
            proba=float(ev["proba"]),
            asset_vol_20d=asset_vol,
            ref_vol=ref_vol,
            portfolio_vol_recent=portfolio_vol,
        )
        if size <= 0:
            n_zero += 1
            continue

        entry_open = float(ohlcv.iloc[entry_idx]["open"])
        if entry_open <= 0:
            continue
        entry_fill = entry_open * (1.0 + entry_slip)
        exit_close = float(ohlcv.iloc[exit_idx]["close"])
        exit_fill = exit_close * (1.0 - exit_slip)
        gross = exit_fill / entry_fill - 1.0
        net = (gross - 2.0 * fee) * size
        bench_gross = exit_close / entry_open - 1.0
        bench_net = (bench_gross - 2.0 * fee) * size

        trades.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": ohlcv.index[exit_idx],
                "entry_price": entry_open,
                "exit_price": exit_close,
                "proba": float(ev["proba"]),
                "size": size,
                "net_return": net,
                "bench_net_return": bench_net,
                "label_drift_5d": int(ev["label_drift_5d"]),
            }
        )
        strat_returns.append((entry_date, net))
        bench_returns.append((entry_date, bench_net))
        cumulative *= 1.0 + net
        strategy_equity.append((entry_date, cumulative))

    if not trades:
        empty = pd.Series(dtype=float)
        return RiskAwarePeadResult(
            strategy=compute_metrics(empty),
            benchmark=compute_metrics(empty),
            trades=pd.DataFrame(),
            n_events_considered=n_considered,
            n_events_gated=n_gated,
            n_events_zero_sized=n_zero,
        )

    strat_daily = _aggregate_returns(strat_returns)
    bench_daily = _aggregate_returns(bench_returns)
    return RiskAwarePeadResult(
        strategy=compute_metrics(strat_daily, trade_returns=[t["net_return"] for t in trades]),
        benchmark=compute_metrics(bench_daily),
        trades=pd.DataFrame(trades),
        n_events_considered=n_considered,
        n_events_gated=n_gated,
        n_events_zero_sized=n_zero,
    )


def _aggregate_returns(events: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    """Average per-event returns onto a daily series indexed by entry date."""
    if not events:
        return pd.Series(dtype=float)
    df = pd.DataFrame(events, columns=pd.Index(["date", "ret"]))
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["ret"].mean()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    return daily.reindex(full_index).fillna(0.0)
