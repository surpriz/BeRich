"""Calibration of the paper-trade hit rate against the model's predicted proba.

For each closed paper trade we have a predicted ``proba`` (from the original
signal) and a realized outcome (``pnl_eur > 0``). Calibration bins the trades
into probability buckets and reports, per bucket, the empirical win rate vs
the bucket's predicted-proba midpoint — the classic reliability diagram.

Perfect calibration: predicted == empirical along the diagonal. Over-confident
models sit below the diagonal at the high end; under-confident above. The
table here doesn't *judge* the model — it just exposes the discrepancy so the
user can decide whether to trust the proba threshold they're using.

The paper store's ``signal``/``proba`` columns aren't in ``paper_trades``
directly — we recover them by joining each closed trade back to its original
signal record via ``(date_open, ticker)``. Trades whose signal row has been
purged are silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from berich.signals.paper import CLOSED_STATUSES, PaperStore
from berich.signals.store import SignalStore

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

CALIBRATOR_FILE = "calibrator.joblib"


@dataclass
class ProbaCalibrator:
    """A fitted transform mapping raw model probabilities to calibrated ones.

    Fit on out-of-fold ``(proba, y_true)`` so it is leak-free, then applied to the live
    proba before it feeds SL/TP sizing and the BUY/SELL threshold.
    """

    method: str  # "isotonic" | "sigmoid"
    model: Any

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = np.asarray(proba, dtype=float).ravel()
        if self.method == "sigmoid":
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            out = self.model.predict(p)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def optimal_decision_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    reward: float = 2.0,
    risk: float = 1.0,
    min_count: int = 20,
    grid: np.ndarray | None = None,
) -> float | None:
    """Decision threshold on (calibrated) win prob maximizing OOS risk-adjusted expectancy.

    For each candidate threshold τ, take the trades with ``proba >= τ``, estimate their win rate,
    and score the triple-barrier expectancy ``wr*reward - (1-wr)*risk`` weighted by ``sqrt(n)`` so
    a thin, lucky bucket can't win on edge alone. Returns the best τ, or ``None`` when no threshold
    clears ``min_count`` (caller then keeps the global threshold). ``reward``/``risk`` are the
    barrier ATR multiples, so the objective reflects this asset's actual payoff ratio.
    """
    p = np.asarray(proba, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    if grid is None:
        grid = np.round(np.arange(0.30, 0.71, 0.01), 2)
    best_tau: float | None = None
    best_score = -np.inf
    for tau in grid:
        mask = p >= tau
        n = int(mask.sum())
        if n < min_count:
            continue
        wr = float(y[mask].mean())
        score = (wr * reward - (1.0 - wr) * risk) * np.sqrt(n)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def fit_calibrator(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str = "isotonic",
) -> ProbaCalibrator:
    """Fit a :class:`ProbaCalibrator` on out-of-fold predictions (leak-free)."""
    p = np.asarray(proba, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        model = LogisticRegression().fit(p.reshape(-1, 1), y.astype(int))
    else:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return ProbaCalibrator(method=method, model=model)


def save_calibrator(calibrator: ProbaCalibrator, *, artifact_dir: Path) -> Path:
    """Persist a calibrator alongside its model artifact."""
    path = artifact_dir / CALIBRATOR_FILE
    joblib.dump(calibrator, path)
    return path


def load_calibrator(artifact_dir: Path) -> ProbaCalibrator | None:
    """Load a calibrator if one was saved next to the artifact, else ``None``."""
    path = artifact_dir / CALIBRATOR_FILE
    if not path.exists():
        return None
    return joblib.load(path)


DEFAULT_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.00, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 1.00),
)


@dataclass
class CalibrationBucket:
    """One row of the reliability table."""

    low: float
    high: float
    midpoint: float  # bucket midpoint, the "predicted" proba reported in the diagram
    mean_predicted: float  # actual mean of predicted probas in this bucket
    win_rate: float  # empirical hit rate (pnl_eur > 0)
    n_trades: int

    def as_row(self) -> dict[str, object]:
        return {
            "bucket": f"[{self.low:.2f}, {self.high:.2f})",
            "low": self.low,
            "high": self.high,
            "midpoint": self.midpoint,
            "mean_predicted": self.mean_predicted,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
        }


@dataclass
class CalibrationReport:
    """Full calibration result — bucket table + a scalar summary."""

    buckets: list[CalibrationBucket]
    n_trades_total: int
    n_with_proba: int

    @property
    def is_well_calibrated(self) -> bool:
        """Cheap heuristic: average |win_rate - midpoint| < 0.10 over populated buckets."""
        populated = [b for b in self.buckets if b.n_trades >= 5]  # noqa: PLR2004
        if not populated:
            return False
        avg_gap = float(np.mean([abs(b.win_rate - b.midpoint) for b in populated]))
        return avg_gap < 0.10  # noqa: PLR2004


def _join_trades_with_signals(
    paper_trades: pd.DataFrame,
    signal_store: SignalStore,
) -> pd.DataFrame:
    """Annotate closed paper trades with the predicted proba from the signals table."""
    if paper_trades.empty:
        return paper_trades.assign(proba=pd.Series(dtype=float))
    # Pull the union of signal histories for each ticker present in the closed
    # paper-trade set. Cheaper than reading the whole signals table and lets
    # us scope the join purely on (date, ticker).
    tickers = paper_trades["ticker"].unique().tolist()
    if not tickers:
        return paper_trades.assign(proba=pd.Series(dtype=float))
    histories = []
    for ticker in tickers:
        hist = signal_store.history(ticker)
        if hist.empty:
            continue
        histories.append(hist[["date", "ticker", "proba"]])
    if not histories:
        return paper_trades.assign(proba=pd.Series(dtype=float))
    signals = pd.concat(histories, ignore_index=True)
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    paper_trades = paper_trades.copy()
    paper_trades["date_open"] = pd.to_datetime(paper_trades["date_open"]).dt.date
    return paper_trades.merge(
        signals.rename(columns={"date": "date_open"}),
        on=["date_open", "ticker"],
        how="left",
    )


def compute_calibration(
    config: Config,
    *,
    buckets: tuple[tuple[float, float], ...] = DEFAULT_BUCKETS,
) -> CalibrationReport:
    """Compute the reliability table from the closed paper trades + signal history."""
    paper = PaperStore(config.db_path)
    signal_store = SignalStore(config.db_path)
    trades = paper.closed_trades()
    if trades.empty:
        return CalibrationReport(
            buckets=[
                CalibrationBucket(
                    low=lo,
                    high=hi,
                    midpoint=(lo + hi) / 2,
                    mean_predicted=0.0,
                    win_rate=0.0,
                    n_trades=0,
                )
                for lo, hi in buckets
            ],
            n_trades_total=0,
            n_with_proba=0,
        )
    # Only ``closed_*`` rows belong here; the store already filters but we keep
    # the predicate for defensive clarity.
    trades = trades[trades["status"].isin(CLOSED_STATUSES)]
    joined = _join_trades_with_signals(trades, signal_store)
    joined = joined.dropna(subset=["proba"])

    bucket_rows: list[CalibrationBucket] = []
    for lo, hi in buckets:
        mask = (joined["proba"] >= lo) & (joined["proba"] < hi)
        subset = joined[mask]
        n = len(subset)
        if n == 0:
            bucket_rows.append(
                CalibrationBucket(
                    low=lo,
                    high=hi,
                    midpoint=(lo + hi) / 2,
                    mean_predicted=0.0,
                    win_rate=0.0,
                    n_trades=0,
                )
            )
            continue
        win_rate = float((subset["pnl_eur"] > 0).mean())
        mean_pred = float(subset["proba"].mean())
        bucket_rows.append(
            CalibrationBucket(
                low=lo,
                high=hi,
                midpoint=(lo + hi) / 2,
                mean_predicted=mean_pred,
                win_rate=win_rate,
                n_trades=n,
            )
        )
    return CalibrationReport(
        buckets=bucket_rows,
        n_trades_total=len(trades),
        n_with_proba=len(joined),
    )


__all__ = [
    "CalibrationBucket",
    "CalibrationReport",
    "ProbaCalibrator",
    "compute_calibration",
    "fit_calibrator",
    "load_calibrator",
    "optimal_decision_threshold",
    "save_calibrator",
]
