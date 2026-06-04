"""Sweep-level promotion reconciliation (multiple-testing control).

Each per-asset tournament promotes independently, with the Deflated Sharpe correcting for *its own*
search. But the nightly/weekend sweep makes hundreds of promotion decisions, and at a 5 % per-test
bar a handful always pass by luck. This pass runs once after the sweep, gathers every currently
promoted (ticker, side, strategy), and applies Benjamini-Hochberg FDR control across the whole
batch — demoting the promotions that don't survive. Demoted models stay on disk and degrade
gracefully to the *observe* tier (paper-only, no capital), so the evidence keeps accruing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from berich.backtest.multiple_testing import benjamini_hochberg
from berich.models.registry import demote, load_active

if TYPE_CHECKING:
    from berich.config import Config

logger = logging.getLogger(__name__)


def _promoted_entries(config: Config) -> list[tuple[str, str, str, float]]:
    """Every currently-promoted (ticker, side, strategy) with its Sharpe p-value.

    The p-value defaults to 1.0 when missing (legacy artifact) so a model with no recorded
    significance can only ever be demoted, never kept, by the FDR pass.
    """
    entries: list[tuple[str, str, str, float]] = []
    for ticker in config.tradeable_tickers():
        for side in config.zoo.ticker_sides:
            for strategy in config.zoo.ticker_exit_strategies:
                registry_dir = config.model_dir_for_ticker(ticker, side, strategy)
                active = load_active(registry_dir)
                if active is None:
                    continue
                _model, meta = active
                pval = float(meta.metrics.get("sharpe_pvalue", 1.0))
                entries.append((ticker, side, strategy, pval))
    return entries


def reconcile_sweep_fdr(config: Config, *, alpha: float = 0.1) -> dict[str, object]:
    """Apply Benjamini-Hochberg FDR control across all promoted per-asset models; demote failures.

    Returns a summary ``{promoted_before, demoted, promoted_after, alpha, demoted_keys}``. A model
    that loses promotion here is not deleted — serving falls back to it as an *observe* candidate.

    Note: longs are gated on beating buy & hold (not significance), but we still hold them to the
    same family-wise significance bar here — an honest, stricter cross-sweep filter, consistent with
    "rigor before all". With no promotions the pass is a no-op.
    """
    entries = _promoted_entries(config)
    if not entries:
        return {
            "promoted_before": 0,
            "demoted": 0,
            "promoted_after": 0,
            "alpha": alpha,
            "demoted_keys": [],
        }

    results = benjamini_hochberg([e[3] for e in entries], alpha=alpha)
    demoted_keys: list[str] = []
    for (ticker, side, strategy, _pval), res in zip(entries, results, strict=True):
        if not res.rejected:
            registry_dir = config.model_dir_for_ticker(ticker, side, strategy)
            if demote(registry_dir):
                demoted_keys.append(f"{ticker}/{side}/{strategy}")

    summary: dict[str, object] = {
        "promoted_before": len(entries),
        "demoted": len(demoted_keys),
        "promoted_after": len(entries) - len(demoted_keys),
        "alpha": alpha,
        "demoted_keys": demoted_keys,
    }
    logger.info(
        "FDR reconciliation: %d promoted -> %d demoted (alpha=%.2f)",
        summary["promoted_before"],
        summary["demoted"],
        alpha,
    )
    return summary


__all__ = ["reconcile_sweep_fdr"]
