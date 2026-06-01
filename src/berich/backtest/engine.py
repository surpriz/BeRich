"""Event-based walk-forward backtest with ATR stop-loss / take-profit.

Consumes out-of-sample probabilities (from :func:`berich.training.oof_predict`) and
the OHLCV cache, simulates long-only swing trades per ticker, and marks the book to
market daily so :mod:`berich.backtest.metrics` can compute Sharpe and drawdown. Every
fill pays a fee and slippage, and the result is benchmarked against an equal-weight
buy & hold of the same universe over the same dates — the design's go/no-go test.

Trade rule (mirrors the triple-barrier label so signals and labels agree):
enter long at the close when P(win) >= threshold; exit at the ATR stop, the ATR
target, or after ``horizon`` bars, whichever comes first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.features.indicators import atr

if TYPE_CHECKING:
    from berich.training.walk_forward import OofResult


class BacktestConfig(BaseModel):
    """Trading and cost assumptions for the simulation.

    Slippage can be either a flat ``slippage_bps`` (Phase 2 default — fine
    for mega-caps with deep books) or volume-proportional via
    ``volume_proportional_slippage=True``. The latter scales each ticker's
    slippage as ``slippage_bps * sqrt(volume_ref / ticker_median_volume)``
    so a small-cap with 1/100th of SPY's volume pays ~10x the per-side
    slippage of a mega-cap. The result is capped at ``slippage_cap_bps``
    to keep extreme tickers from dominating the bill.
    """

    entry_threshold: float = 0.5
    horizon_days: int = 10
    atr_window: int = 14
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 1.0
    fee_bps: float = 1.0  # per-side commission, basis points of notional
    slippage_bps: float = 5.0  # per-side slippage, basis points
    volume_proportional_slippage: bool = False
    # Reference volume = SPY's long-run median (~80M shares/day historically).
    # When the per-ticker median is below this, slippage scales up.
    volume_ref: float = 80_000_000.0
    slippage_cap_bps: float = 100.0  # safety cap to keep micro-caps sane
    borrow_bps_annual: float = 0.0  # short borrow fee, bps/yr; only charged on shorts
    direction: str = "long"


@dataclass
class Trade:
    """A single completed round-trip swing trade (long or short)."""

    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    reason: str  # "target" | "stop" | "time"
    direction: str = "long"

    @property
    def gross_return(self) -> float:
        # A short profits when it buys back below the (filled) entry price.
        if self.direction == "short":
            return self.entry_price / self.exit_price - 1.0
        return self.exit_price / self.entry_price - 1.0


@dataclass
class BacktestResult:
    """Strategy vs benchmark outcome over the out-of-sample period."""

    strategy: PerfMetrics
    benchmark: PerfMetrics
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    trades: list[Trade] = field(default_factory=list)

    @property
    def beats_buy_hold(self) -> bool:
        """True if the strategy's Sharpe exceeds buy & hold's."""
        return self.strategy.sharpe > self.benchmark.sharpe


def run_backtest(
    prices_by_ticker: dict[str, pd.DataFrame],
    signals: OofResult,
    config: BacktestConfig,
) -> BacktestResult:
    """Simulate the strategy across tickers and compare to equal-weight buy & hold."""
    fee = config.fee_bps / 1e4

    strat_returns: dict[str, pd.Series] = {}
    bench_returns: dict[str, pd.Series] = {}
    all_trades: list[Trade] = []

    for ticker, df in prices_by_ticker.items():
        sig = signals.frame[signals.frame["ticker"] == ticker]["proba"]
        if sig.empty:
            continue
        slip = _slippage_for_ticker(df, config)
        r_strat, trades = _simulate_ticker(ticker, df, sig, config, fee=fee, slip=slip)
        # Benchmark and strategy share the same date window for a fair comparison.
        bench = df["close"].pct_change().reindex(r_strat.index).fillna(0.0)
        strat_returns[ticker] = r_strat
        bench_returns[ticker] = bench
        all_trades.extend(trades)

    strat_daily = _equal_weight(strat_returns)
    bench_daily = _equal_weight(bench_returns)
    trade_rets = [t.gross_return - 2 * (fee + slip) for t in all_trades]

    return BacktestResult(
        strategy=compute_metrics(strat_daily, trade_returns=trade_rets),
        benchmark=compute_metrics(bench_daily),
        strategy_returns=strat_daily,
        benchmark_returns=bench_daily,
        trades=all_trades,
    )


def _equal_weight(per_ticker: dict[str, pd.Series]) -> pd.Series:
    """Average daily returns across tickers on the union of their dates."""
    if not per_ticker:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(per_ticker).sort_index()
    return frame.mean(axis=1)


def _slippage_for_ticker(df: pd.DataFrame, config: BacktestConfig) -> float:
    """Per-side slippage (fraction, not bps) for one ticker.

    Constant when ``volume_proportional_slippage`` is off (Phase 2 behavior).
    Otherwise scaled by sqrt(volume_ref / median_volume) so a small-cap with
    1/100th of SPY's daily volume pays ~10x the base bps. The result is
    capped at ``slippage_cap_bps`` so a truly illiquid micro-cap doesn't
    distort the aggregate result with an unrealistic fill cost.
    """
    base_bps = config.slippage_bps
    if not config.volume_proportional_slippage:
        return base_bps / 1e4
    median_volume = float(df["volume"].median())
    if median_volume <= 0 or pd.isna(median_volume):
        scale = 1.0
    else:
        scale = (config.volume_ref / median_volume) ** 0.5
    effective_bps = min(base_bps * scale, config.slippage_cap_bps)
    return effective_bps / 1e4


def _simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    proba: pd.Series,
    config: BacktestConfig,
    *,
    fee: float,
    slip: float,
) -> tuple[pd.Series, list[Trade]]:
    """Simulate one ticker; return its daily strategy returns and completed trades."""
    df = df.sort_index()
    atr_vals = atr(df["high"], df["low"], df["close"], config.atr_window)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    proba = proba.reindex(df.index)

    direction = config.direction
    borrow_per_day = config.borrow_bps_annual / 1e4 / 252.0 if direction == "short" else 0.0

    dates = df.index
    daily = pd.Series(0.0, index=dates)
    trades: list[Trade] = []

    i = 0
    n = len(dates)
    while i < n - 1:
        p = proba.iloc[i]
        a = atr_vals.iloc[i]
        if np.isnan(p) or p < config.entry_threshold or np.isnan(a):
            i += 1
            continue

        if direction == "short":
            entry_price = close.iloc[i] * (1 - slip)  # sell on entry, fill below close
            stop = close.iloc[i] + config.stop_loss_atr * a
            target = close.iloc[i] - config.take_profit_atr * a
        else:
            entry_price = close.iloc[i] * (1 + slip)
            stop = close.iloc[i] - config.stop_loss_atr * a
            target = close.iloc[i] + config.take_profit_atr * a
        time_exit = min(i + config.horizon_days, n - 1)
        daily.iloc[i] -= fee  # entry commission, charged on the entry bar

        exit_idx, exit_price, reason = _resolve_exit(
            high,
            low,
            close,
            start=i + 1,
            time_exit=time_exit,
            stop=stop,
            target=target,
            direction=direction,
        )

        # Mark to market across the holding period. A short earns the negated price
        # return each day and pays the borrow fee for every day the position is held.
        if direction == "short":
            exit_fill = exit_price * (1 + slip)  # buy back on exit, fill above price
            daily.iloc[i] += -(close.iloc[i] / entry_price - 1.0) - borrow_per_day
            for j in range(i + 1, exit_idx):
                daily.iloc[j] += -(close.iloc[j] / close.iloc[j - 1] - 1.0) - borrow_per_day
            daily.iloc[exit_idx] += -(exit_fill / close.iloc[exit_idx - 1] - 1.0) - borrow_per_day
        else:
            exit_fill = exit_price * (1 - slip)
            daily.iloc[i] += close.iloc[i] / entry_price - 1.0
            for j in range(i + 1, exit_idx):
                daily.iloc[j] += close.iloc[j] / close.iloc[j - 1] - 1.0
            daily.iloc[exit_idx] += exit_fill / close.iloc[exit_idx - 1] - 1.0
        daily.iloc[exit_idx] -= fee  # exit commission

        trades.append(
            Trade(
                ticker=ticker,
                entry_date=dates[i],
                exit_date=dates[exit_idx],
                entry_price=entry_price,
                exit_price=exit_fill,
                reason=reason,
                direction=direction,
            )
        )
        i = exit_idx + 1  # no overlapping positions

    return daily, trades


def _resolve_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    start: int,
    time_exit: int,
    stop: float,
    target: float,
    direction: str = "long",
) -> tuple[int, float, str]:
    """Find the first bar in [start, time_exit] that hits stop/target, else time-exit.

    For a short the stop sits above entry (touched on the bar high) and the target
    below (touched on the bar low), mirroring the long case.
    """
    for j in range(start, time_exit + 1):
        if direction == "short":
            hit_stop = high.iloc[j] >= stop
            hit_target = low.iloc[j] <= target
        else:
            hit_stop = low.iloc[j] <= stop
            hit_target = high.iloc[j] >= target
        if hit_stop and hit_target:
            return j, stop, "stop"  # conservative: assume stop first
        if hit_stop:
            return j, stop, "stop"
        if hit_target:
            return j, target, "target"
    return time_exit, close.iloc[time_exit], "time"
