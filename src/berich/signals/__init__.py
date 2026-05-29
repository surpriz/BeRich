"""Daily signal generation, position sizing, persistence, paper trading, calibration."""

from berich.signals.calibration import (
    CalibrationBucket,
    CalibrationReport,
    compute_calibration,
)
from berich.signals.longshort_service import (
    LongShortBook,
    LongShortPosition,
    LongShortStore,
    generate_longshort_book,
    longshort_equity,
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
from berich.signals.service import (
    Signal,
    explain_signal,
    generate_multi_asset_signals,
    generate_signals,
)
from berich.signals.store import SignalStore

__all__ = [
    "CalibrationBucket",
    "CalibrationReport",
    "LongShortBook",
    "LongShortPosition",
    "LongShortStore",
    "OpenPosition",
    "PaperStore",
    "Signal",
    "SignalStore",
    "compute_calibration",
    "explain_signal",
    "generate_longshort_book",
    "generate_multi_asset_signals",
    "generate_signals",
    "get_equity_curve",
    "get_open_positions",
    "get_paper_metrics",
    "longshort_equity",
    "open_new_trades",
    "update_open_trades",
]
