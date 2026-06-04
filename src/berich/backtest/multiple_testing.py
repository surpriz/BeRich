"""Family-wise / false-discovery control across a sweep of strategy candidates.

The per-model Deflated Sharpe (``significance.assess_sharpe``) corrects each candidate for the
size of *its own* hyperparameter search. It does **not** correct for the fact that the nightly
sweep tries hundreds of (ticker, side, strategy) candidates and we promote whichever clear the
bar — run enough independent coin-flips and some always look significant.

Benjamini-Hochberg (BH) controls the **false-discovery rate** (the expected fraction of promoted
models that are actually noise) across the whole batch of p-values. It is less brutal than
Bonferroni (which controls the probability of *any* false positive and would reject almost
everything on this thin data), which fits an advisory tool: we accept that a known, bounded
fraction of promotions may be luck, rather than promoting nothing.

Pure-stdlib (no scipy) to honor the "avoid fragile deps" house rule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BHResult:
    """Outcome of a Benjamini-Hochberg pass for one candidate."""

    index: int  # position in the input p-value list
    p_value: float
    rejected: bool  # True => survives FDR control (a "discovery" we may promote)
    threshold: float  # the BH critical value this p-value was compared against


def benjamini_hochberg(p_values: list[float], *, alpha: float = 0.1) -> list[BHResult]:
    """Benjamini-Hochberg FDR control at level ``alpha``.

    Returns one :class:`BHResult` per input p-value, in the **input order**. ``rejected=True`` marks
    the discoveries that survive FDR control (i.e. may be promoted). A non-finite or missing p-value
    is treated as 1.0 (never a discovery), so a degenerate candidate can't sneak through.

    BH procedure: sort p-values ascending, find the largest rank ``k`` with
    ``p_(k) <= (k / m) * alpha``, and reject every hypothesis with ``p <= p_(k)``.
    """
    m = len(p_values)
    if m == 0:
        return []
    clean = [
        (i, p if isinstance(p, (int, float)) and 0.0 <= p <= 1.0 else 1.0)
        for i, p in enumerate(p_values)
    ]
    ordered = sorted(clean, key=lambda t: t[1])

    # Largest k (1-based) for which p_(k) <= (k/m)*alpha.
    k_max = 0
    for rank, (_idx, p) in enumerate(ordered, start=1):
        if p <= (rank / m) * alpha:
            k_max = rank
    p_cut = ordered[k_max - 1][1] if k_max > 0 else 0.0

    results = [
        BHResult(
            index=idx,
            p_value=p,
            rejected=(k_max > 0 and p <= p_cut),
            threshold=(k_max / m) * alpha if k_max > 0 else 0.0,
        )
        for idx, p in clean
    ]
    return sorted(results, key=lambda r: r.index)


__all__ = ["BHResult", "benjamini_hochberg"]
