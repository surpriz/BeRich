"""Position-sizing primitives — pure math, no I/O.

Three independent sizers that the strategy combiner mixes:

- :func:`kelly_fraction` — classical Kelly criterion for a binary outcome.
  Returns the *full* Kelly; production callers should scale by 0.25 to 0.5
  (half-Kelly) to avoid the well-known compounding blow-ups.
- :func:`vol_target_size` — portfolio-level scale so realized vol of the
  scaled book matches a target (default 15 % annualized).
- :func:`inverse_vol_size` — per-asset scale so high-vol names get
  smaller positions than low-vol names of the same conviction.

Each returns a multiplier in ``[0, ceiling]``; the strategy combiner
multiplies them together and applies a hard per-trade cap.
"""

from __future__ import annotations

import math


def kelly_fraction(
    proba_win: float,
    win_size: float,
    loss_size: float,
) -> float:
    """Classical Kelly criterion for a binary bet.

    Args:
        proba_win: probability of the winning outcome, in [0, 1].
        win_size:  positive magnitude of the win return (e.g. 0.04 for +4%).
        loss_size: positive magnitude of the loss return (e.g. 0.02 for -2%).

    Returns:
        The fraction of capital to risk per Kelly. Negative when the bet
        has no edge (``proba_win * win_size < (1 - proba_win) * loss_size``);
        callers usually clip to 0 in that case. Returns 0 for degenerate
        inputs (zero magnitudes, probabilities outside [0, 1]).
    """
    if not 0.0 <= proba_win <= 1.0:
        return 0.0
    if win_size <= 0 or loss_size <= 0:
        return 0.0
    edge = proba_win * win_size - (1.0 - proba_win) * loss_size
    return edge / (win_size * loss_size)


def vol_target_size(
    portfolio_vol_recent: float,
    *,
    target_vol: float = 0.15,
    ceiling: float = 1.0,
) -> float:
    """Scale the portfolio so realized vol == ``target_vol``.

    ``portfolio_vol_recent`` is the *annualized* realized vol of the strategy
    over a recent window (typically 20 trading days). Returned multiplier is
    clipped at ``ceiling`` so we don't lever up indefinitely when vol is
    tiny. Returns ``ceiling`` when the recent vol is non-positive or NaN.
    """
    if portfolio_vol_recent is None or math.isnan(portfolio_vol_recent):
        return ceiling
    if portfolio_vol_recent <= 0:
        return ceiling
    scale = target_vol / portfolio_vol_recent
    return min(scale, ceiling)


def inverse_vol_size(
    asset_vol_20d: float,
    ref_vol: float,
    *,
    ceiling: float = 1.0,
) -> float:
    """Per-asset scale = ``ref_vol / asset_vol_20d``, clipped to ``ceiling``.

    The reference vol is typically the SPY 20-day rvol — i.e. "size 1.0 for
    an asset as volatile as the market, half as much for an asset twice as
    volatile". Falls back to ``ceiling`` for degenerate vols so a
    zero-vol asset doesn't blow up the multiplier.
    """
    if asset_vol_20d is None or math.isnan(asset_vol_20d):
        return ceiling
    if asset_vol_20d <= 0 or ref_vol is None or ref_vol <= 0:
        return ceiling
    return min(ref_vol / asset_vol_20d, ceiling)


def annualize_daily_vol(daily_std: float, *, bars_per_year: int = 252) -> float:
    """Convenience: turn a per-bar returns std into an annualized vol (sqrt 252 for daily)."""
    if daily_std is None or math.isnan(daily_std) or daily_std <= 0:
        return 0.0
    return float(daily_std * math.sqrt(bars_per_year))
