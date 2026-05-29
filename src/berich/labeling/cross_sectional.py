"""Cross-sectional forward-return labels for the market-neutral long/short track.

Unlike the triple-barrier target (binary, per-ticker, event-driven), the long/short
system ranks names *against each other* on each date. The per-ticker piece computed
here is the **beta-residualized H-day forward return**: strip each name's market beta
so the target carries idiosyncratic, market-neutral information by construction. The
final cross-sectional z-score (or rank) standardization happens in the panel assembler
(:func:`berich.datasets.cross_sectional.build_panel_dataset`), which is the only place
that sees every ticker on a date.

Causality: ``beta`` uses only returns up to and including ``t`` (a rolling window ending
at ``t``); the forward return spans ``t .. t+H``. The two never overlap, so there is no
lookahead. The last ``H`` rows have NaN labels (incomplete forward window).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel


class CrossSectionalLabelConfig(BaseModel):
    """Parameters for the residualized cross-sectional forward-return label."""

    horizon_days: int = 5
    beta_window: int = 60
    residualize: bool = True
    standardize: Literal["zscore", "rank"] = "zscore"


def forward_return_labels(
    df: pd.DataFrame,
    config: CrossSectionalLabelConfig,
    *,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-ticker forward-return label pieces, indexed like ``df``.

    Returns a frame with columns:
    ``fwd_ret`` (H-day forward return), ``beta`` (trailing causal market beta),
    ``resid`` (beta-residualized forward return; equals ``fwd_ret`` when
    ``residualize`` is off or no market is supplied), and ``sample_weight`` (``|resid|``).
    The cross-sectional standardization is applied later by the assembler.
    """
    h = config.horizon_days
    close = df["close"].astype(float)
    ret_1 = close.pct_change(fill_method=None)
    fwd_ret = close.shift(-h) / close - 1.0

    beta = pd.Series(0.0, index=df.index)
    resid = fwd_ret.copy()

    if config.residualize and market is not None and not market.empty:
        mkt_close = market["close"].astype(float).reindex(df.index).ffill()
        mkt_ret_1 = mkt_close.pct_change(fill_method=None)
        mkt_fwd = mkt_close.shift(-h) / mkt_close - 1.0
        w = config.beta_window
        cov = ret_1.rolling(w, min_periods=w).cov(mkt_ret_1)
        var = mkt_ret_1.rolling(w, min_periods=w).var()
        beta = (cov / var).replace([float("inf"), float("-inf")], pd.NA)
        resid = fwd_ret - beta * mkt_fwd

    out = pd.DataFrame(
        {"fwd_ret": fwd_ret, "beta": beta, "resid": resid},
        index=df.index,
    )
    out["sample_weight"] = out["resid"].abs()
    return out


__all__ = ["CrossSectionalLabelConfig", "forward_return_labels"]
