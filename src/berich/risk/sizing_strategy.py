"""The combined risk overlay — what every signal-driven backtest plugs into.

A :class:`RiskOverlay` holds the configuration (which gates are on, what
Kelly fraction, what vol target, what caps) and exposes a single method
:meth:`position_size` that takes the per-event inputs and returns the
**final position multiplier**, in ``[0, max_position]``. Internally:

1. Run the binary gates. If any is False → return 0.
2. Compute the Kelly fraction (assuming PEAD-style W/L magnitudes).
3. Multiply by fractional Kelly + per-asset inverse-vol + portfolio vol-target.
4. Cap at ``max_position``.

The config is a frozen dataclass so the same overlay applies
deterministically across walk-forward folds — only the rolling statistics
(SPY rvol percentile, equity drawdown) change at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from berich.risk.gating import drawdown_gate, regime_gate_spy
from berich.risk.sizing import inverse_vol_size, kelly_fraction, vol_target_size

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Hyperparameters for the risk overlay; defaults tuned for the PEAD task."""

    # Toggles — let the comparative script flip each gate independently.
    use_regime_gate: bool = False
    use_drawdown_gate: bool = False
    use_kelly: bool = False
    use_vol_target: bool = False
    use_inverse_vol: bool = False

    # Gate parameters.
    regime_quartile_threshold: float = 0.75
    regime_lookback_days: int = 252
    drawdown_max_threshold: float = 0.20

    # Sizing parameters.
    kelly_fraction_multiplier: float = 0.5  # half-Kelly is the standard
    target_annual_vol: float = 0.15
    # The PEAD label fires when fwd_5d > 2 %; the average miss size from the
    # 6 066-event historical dataset is around -2 %. These map directly into
    # the Kelly W/L magnitudes — adjust here if the label or universe changes.
    kelly_win_size: float = 0.04
    kelly_loss_size: float = 0.02

    # Per-trade cap (fraction of capital). Hard ceiling regardless of math.
    max_position: float = 0.05


class RiskOverlay:
    """Stateless wrapper around :class:`RiskOverlayConfig` for clarity."""

    def __init__(self, config: RiskOverlayConfig | None = None) -> None:
        self.config = config or RiskOverlayConfig()

    def gates_pass(
        self,
        *,
        as_of: pd.Timestamp,
        spy_rvol_series: pd.Series | None,
        strategy_equity: pd.Series | None,
    ) -> bool:
        """Run every active gate; return True iff every one allows the trade."""
        cfg = self.config
        if cfg.use_regime_gate:
            if spy_rvol_series is None:
                return False  # can't evaluate the gate; safer to skip than mis-trade
            if not regime_gate_spy(
                spy_rvol_series,
                as_of=as_of,
                quartile_threshold=cfg.regime_quartile_threshold,
                lookback_days=cfg.regime_lookback_days,
            ):
                return False
        if cfg.use_drawdown_gate:
            if strategy_equity is None:
                # No history yet → nothing to block.
                pass
            elif not drawdown_gate(
                strategy_equity,
                max_dd_threshold=cfg.drawdown_max_threshold,
            ):
                return False
        return True

    def position_size(
        self,
        *,
        proba: float,
        asset_vol_20d: float | None,
        ref_vol: float | None,
        portfolio_vol_recent: float | None,
    ) -> float:
        """Combined multiplier for a single trade in ``[0, max_position]``.

        Returns 0 when the Kelly edge is non-positive (model has no edge for
        this proba); ``max_position`` is the hard ceiling regardless of how
        many up-multiplications the components produced.
        """
        cfg = self.config
        position = 1.0
        if cfg.use_kelly:
            full_kelly = kelly_fraction(proba, cfg.kelly_win_size, cfg.kelly_loss_size)
            if full_kelly <= 0:
                return 0.0
            position = cfg.kelly_fraction_multiplier * full_kelly
        if cfg.use_inverse_vol and asset_vol_20d is not None and ref_vol is not None:
            position *= inverse_vol_size(asset_vol_20d, ref_vol, ceiling=2.0)
        if cfg.use_vol_target and portfolio_vol_recent is not None:
            position *= vol_target_size(
                portfolio_vol_recent, target_vol=cfg.target_annual_vol, ceiling=2.0
            )
        return float(max(0.0, min(position, cfg.max_position)))
