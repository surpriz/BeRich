"""Walk-forward training + event-driven backtest for the PEAD model.

The training loop is chronological at the **event** level (not the bar
level): each fold trains on every event with ``entry_date`` strictly before
the fold's start, then scores the events in the fold. Returns out-of-sample
probas aligned to the dataset's row order.

The backtest is a simple, honest event-driven simulator: when a
proba clears ``threshold`` for a given event we open a long at that event's
``entry_date`` open price (with a configurable slippage) and exit at close
``hold_days`` later. There's no pyramiding, no ATR stops — PEAD is a
4-per-ticker-per-year trade rule, not a daily oscillator. The benchmark
buy-and-hold uses the same set of (entry, exit) date ranges on each
ticker so the Sharpe comparison stays apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.datasets.pead import PeadDataset, split_walk_forward

if TYPE_CHECKING:
    from berich.data.store import OhlcvStore
    from berich.models.base import Model

ModelFactory = Callable[[], "Model"]

DEFAULT_HOLD_DAYS = 5
DEFAULT_ENTRY_SLIPPAGE_BPS = 10.0  # higher than daily: opening at the AM after a print
DEFAULT_EXIT_SLIPPAGE_BPS = 5.0
DEFAULT_FEE_BPS = 1.0


@dataclass
class PeadOofResult:
    """Out-of-sample event-level predictions."""

    frame: pd.DataFrame  # columns: ticker, entry_date, proba, y_true

    @property
    def auc(self) -> float:
        y = self.frame["y_true"]
        if y.nunique() < 2:  # noqa: PLR2004
            return float("nan")
        return float(roc_auc_score(y, self.frame["proba"]))


@dataclass
class PeadBacktestResult:
    """Event-driven PEAD backtest outcome."""

    strategy: PerfMetrics
    benchmark: PerfMetrics
    trades: pd.DataFrame  # per-event records with realized P&L

    @property
    def beats_buy_hold(self) -> bool:
        return self.strategy.sharpe > self.benchmark.sharpe


def oof_predict_pead(
    dataset: PeadDataset,
    model_factory: ModelFactory,
    *,
    n_folds: int = 5,
    min_train: int = 500,
) -> PeadOofResult:
    """Walk-forward OOS for the event-level PEAD dataset."""
    folds = split_walk_forward(dataset, n_folds=n_folds, min_train=min_train)
    if not folds:
        msg = (
            f"PEAD dataset has {len(dataset)} events but needs > {min_train}"
            " to produce at least one walk-forward fold"
        )
        raise ValueError(msg)

    rows: list[pd.DataFrame] = []
    for train_idx, test_idx in folds:
        model = model_factory()
        x_tr = dataset.x.iloc[train_idx]
        y_tr = dataset.y.iloc[train_idx]
        model.fit(x_tr, y_tr)
        x_te = dataset.x.iloc[test_idx]
        proba = model.predict_proba(x_te)
        rows.append(
            pd.DataFrame(
                {
                    "ticker": dataset.tickers.iloc[test_idx].to_numpy(),
                    "entry_date": dataset.entry_dates[test_idx],
                    "proba": proba,
                    "y_true": dataset.y.iloc[test_idx].to_numpy(),
                }
            )
        )
    frame = pd.concat(rows).sort_values("entry_date").reset_index(drop=True)
    return PeadOofResult(frame=frame)


def run_pead_backtest(
    dataset: PeadDataset,
    oof: PeadOofResult,
    store: OhlcvStore,
    *,
    threshold: float = 0.5,
    hold_days: int = DEFAULT_HOLD_DAYS,
    entry_slippage_bps: float = DEFAULT_ENTRY_SLIPPAGE_BPS,
    exit_slippage_bps: float = DEFAULT_EXIT_SLIPPAGE_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> PeadBacktestResult:
    """Long the predicted-positive events at the entry_date open; exit ``hold_days`` later.

    The benchmark is an equal-weight buy-and-hold of the same (ticker,
    entry_date → exit_date) windows — so if the strategy fires on N events
    its B&H is also N positions held over the same days. This makes the
    Sharpe comparison honest event-by-event rather than against a
    time-averaged S&P proxy.
    """
    fee = fee_bps / 1e4
    entry_slip = entry_slippage_bps / 1e4
    exit_slip = exit_slippage_bps / 1e4

    # Join OOF probas back onto the events frame to recover entry context.
    events = dataset.events.copy()
    events["proba"] = np.nan
    events = events.set_index(["ticker", "entry_date"])
    for _, row in oof.frame.iterrows():
        key = (row["ticker"], pd.Timestamp(row["entry_date"]))
        if key in events.index:
            events.loc[key, "proba"] = float(row["proba"])
    events = events.reset_index()
    in_sample = events.dropna(subset=["proba"]).copy()
    picks = in_sample[in_sample["proba"] >= threshold].copy()

    trades: list[dict] = []
    strat_returns: list[tuple[pd.Timestamp, float]] = []
    bench_returns: list[tuple[pd.Timestamp, float]] = []

    for _, ev in picks.iterrows():
        ticker = str(ev["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            continue
        raw_entry_date = pd.Timestamp(ev["entry_date"])
        if pd.isna(raw_entry_date):
            continue
        entry_date: pd.Timestamp = raw_entry_date  # ty: ignore[invalid-assignment]
        if entry_date not in ohlcv.index:
            continue
        entry_idx = int(ohlcv.index.get_loc(entry_date))
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(ohlcv):
            continue
        entry_open = float(ohlcv.iloc[entry_idx]["open"])
        if entry_open <= 0:
            continue
        # Realistic fills: pay the spread + a commission on each side.
        entry_fill = entry_open * (1.0 + entry_slip)
        exit_close = float(ohlcv.iloc[exit_idx]["close"])
        exit_fill = exit_close * (1.0 - exit_slip)
        gross = exit_fill / entry_fill - 1.0
        net = gross - 2.0 * fee  # entry + exit commissions
        # Benchmark: hold from same entry-day open to same exit-day close,
        # paying the same costs (so the comparison is "did the proba help
        # at all" not "did long-only earnings windows beat cash").
        exit_close_no_slip = float(ohlcv.iloc[exit_idx]["close"])
        bench_gross = exit_close_no_slip / entry_open - 1.0
        bench_net = bench_gross - 2.0 * fee
        trades.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": ohlcv.index[exit_idx],
                "entry_price": entry_open,
                "exit_price": exit_close,
                "proba": float(ev["proba"]),
                "net_return": net,
                "bench_net_return": bench_net,
                "label_drift_5d": int(ev["label_drift_5d"]),
            }
        )
        strat_returns.append((entry_date, net))
        bench_returns.append((entry_date, bench_net))

    if not trades:
        empty = pd.Series(dtype=float)
        return PeadBacktestResult(
            strategy=compute_metrics(empty),
            benchmark=compute_metrics(empty),
            trades=pd.DataFrame(),
        )

    strat_series = _aggregate_returns(strat_returns)
    bench_series = _aggregate_returns(bench_returns)
    strat_metrics = compute_metrics(strat_series, trade_returns=[t["net_return"] for t in trades])
    bench_metrics = compute_metrics(bench_series)
    return PeadBacktestResult(
        strategy=strat_metrics,
        benchmark=bench_metrics,
        trades=pd.DataFrame(trades),
    )


def _aggregate_returns(events: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    """Average per-event returns onto a daily series indexed by entry date.

    Two events on the same day average; days with no event return 0. This is
    a deliberately simple aggregator — PEAD's trade frequency is event-driven,
    so a daily Sharpe interpretation requires us to assume cash sits on the
    sidelines on non-event days.
    """
    if not events:
        return pd.Series(dtype=float)
    df = pd.DataFrame(events, columns=pd.Index(["date", "ret"]))
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["ret"].mean()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    return daily.reindex(full_index).fillna(0.0)
