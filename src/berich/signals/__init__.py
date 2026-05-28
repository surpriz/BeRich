"""Daily signal generation, position sizing, persistence, paper trading, calibration."""

from berich.signals.calibration import (
    CalibrationBucket,
    CalibrationReport,
    compute_calibration,
)
from berich.signals.paper import (
    OpenPosition,
    PaperStore,
    get_equity_curve,
    get_open_positions,
    get_paper_metrics,
    open_new_trades,
    update_open_trades,
)
from berich.signals.service import Signal, generate_signals
from berich.signals.store import SignalStore

__all__ = [
    "CalibrationBucket",
    "CalibrationReport",
    "OpenPosition",
    "PaperStore",
    "Signal",
    "SignalStore",
    "compute_calibration",
    "generate_signals",
    "get_equity_curve",
    "get_open_positions",
    "get_paper_metrics",
    "open_new_trades",
    "update_open_trades",
]
