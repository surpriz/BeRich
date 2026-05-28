"""Daily signal generation, position sizing, persistence, and paper trading."""

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
    "OpenPosition",
    "PaperStore",
    "Signal",
    "SignalStore",
    "generate_signals",
    "get_equity_curve",
    "get_open_positions",
    "get_paper_metrics",
    "open_new_trades",
    "update_open_trades",
]
