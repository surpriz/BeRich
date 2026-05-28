"""Diagnostic: LightGBM feature importances on the current 24-feature set.

Fits a LightGBM baseline on the full labeled dataset and prints split-gain
importances ordered high to low. Highlights the top 5 and bottom 5 so we can spot
Phase 3a additions that contribute ~0 (candidates for removal). Side effects: none
— this does not touch the model registry or any persisted artifact.
"""

from __future__ import annotations

import numpy as np

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.features.build import FEATURE_COLUMNS
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel

# Anything below this fraction of the max importance is flagged as "weak".
WEAK_FRACTION = 0.05


def main() -> int:
    config = Config.load(DEFAULT_CONFIG_PATH)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    dataset = build_dataset(store, config.watchlist, label_cfg)
    print(f"Fitting LGBM on {len(dataset)} samples x {len(FEATURE_COLUMNS)} features...")

    model = LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = importances.sum() or 1.0
    pct = 100.0 * importances / total
    order = np.argsort(importances)[::-1]

    print(
        f"\nFeature importances (LGBM split count, n_estimators={model.params['n_estimators']}):\n"
    )
    print(f"  {'rank':>4}  {'feature':<22}{'importance':>12}{'share':>8}")
    for rank, idx in enumerate(order, start=1):
        print(f"  {rank:>4}  {FEATURE_COLUMNS[idx]:<22}{importances[idx]:>12.0f}{pct[idx]:>7.2f}%")

    print("\nTop 5 most useful:")
    for idx in order[:5]:
        print(f"  - {FEATURE_COLUMNS[idx]:<22} {pct[idx]:5.2f}%")

    print("\nBottom 5 least useful:")
    for idx in order[-5:][::-1]:
        print(f"  - {FEATURE_COLUMNS[idx]:<22} {pct[idx]:5.2f}%")

    weak_threshold = WEAK_FRACTION * importances.max()
    weak = [FEATURE_COLUMNS[i] for i in order if importances[i] < weak_threshold]
    if weak:
        print(
            f"\nWeak features (< {WEAK_FRACTION:.0%} of max importance "
            f"= {weak_threshold:.0f} splits):"
        )
        for name in weak:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
