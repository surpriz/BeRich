"""Feature and performance drift detection.

A model trained months ago can silently rot when the market regime shifts. We watch
two things:

* **Feature drift** — does the recent feature distribution differ from the training
  distribution? Measured per feature with the Population Stability Index (PSI) and a
  two-sample Kolmogorov-Smirnov test.
* **Performance drift** — handled by re-running the walk-forward backtest; if recent
  out-of-sample AUC decays, retrain.

PSI thresholds follow the common convention: < 0.1 stable, 0.1-0.25 moderate shift,
> 0.25 significant shift. A significant shift on enough features recommends a retrain.
This is a deliberately small, dependency-light alternative to Evidently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass
class FeatureDrift:
    """Drift diagnostics for a single feature."""

    feature: str
    psi: float
    ks_pvalue: float

    @property
    def drifted(self) -> bool:
        """True if PSI signals a significant shift or KS rejects equal distributions."""
        return self.psi >= PSI_SIGNIFICANT or self.ks_pvalue < 0.01  # noqa: PLR2004


@dataclass
class DriftReport:
    """Aggregate drift across all features with a retrain recommendation."""

    features: list[FeatureDrift] = field(default_factory=list)

    @property
    def n_drifted(self) -> int:
        return sum(f.drifted for f in self.features)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / len(self.features) if self.features else 0.0

    @property
    def should_retrain(self) -> bool:
        """Recommend retraining when at least a third of features have drifted."""
        return self.share_drifted >= 1 / 3

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"feature": f.feature, "psi": f.psi, "ks_pvalue": f.ks_pvalue, "drifted": f.drifted}
                for f in self.features
            ]
        ).sort_values("psi", ascending=False, ignore_index=True)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI between a reference and current sample using reference quantiles.

    Bins are defined by the reference distribution's quantiles so each reference bin
    holds ~equal mass; PSI then measures how the current sample's mass redistributes.
    A small epsilon avoids division by / log of zero in empty bins.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:  # noqa: PLR2004 — constant reference, no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    bins: int = 10,
) -> DriftReport:
    """Compute a :class:`DriftReport` over the columns shared by both frames."""
    columns = [c for c in reference.columns if c in current.columns]
    diagnostics: list[FeatureDrift] = []
    for col in columns:
        ref = reference[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur, bins=bins)
        ks = ks_2samp(
            ref[~np.isnan(ref)],
            cur[~np.isnan(cur)],
        )
        diagnostics.append(FeatureDrift(feature=col, psi=psi, ks_pvalue=float(ks.pvalue)))
    return DriftReport(features=diagnostics)
