"""Risk-management overlay for trading strategies (Phase 8).

This module is the "what you do with a signal once you have one" layer. The
LightGBM / PEAD models live elsewhere and only produce ``P(win)``; the
``risk`` package decides whether to take the trade at all (gating) and how
much to size it (sizing). Everything here is signal-agnostic — the same
overlay applies on top of any model.
"""

from berich.risk.gating import drawdown_gate, regime_gate_spy
from berich.risk.sizing import (
    inverse_vol_size,
    kelly_fraction,
    vol_target_size,
)
from berich.risk.sizing_strategy import RiskOverlay, RiskOverlayConfig

__all__ = [
    "RiskOverlay",
    "RiskOverlayConfig",
    "drawdown_gate",
    "inverse_vol_size",
    "kelly_fraction",
    "regime_gate_spy",
    "vol_target_size",
]
