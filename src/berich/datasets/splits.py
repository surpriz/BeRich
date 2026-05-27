"""Walk-forward (time-series) cross-validation splits with an embargo.

Standard k-fold leaks the future into the past. Walk-forward instead trains on a
contiguous past window and tests on the block immediately after it, then rolls
forward. An *embargo* gap (>= the label horizon) is dropped between train and test
so that forward-looking triple-barrier labels in the train set cannot overlap the
test period — the single most common source of inflated backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: integer positions into the (time-sorted) dataset."""

    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    expanding: bool = True,
) -> list[Fold]:
    """Generate walk-forward folds over ``n_samples`` time-ordered rows.

    Args:
        n_samples: total rows, assumed sorted ascending by time.
        train_size: rows in the (initial) training window.
        test_size: rows in each test block.
        embargo: rows skipped between train and test (set to the label horizon).
        expanding: if True the train window grows each fold; if False it rolls
            with a fixed length.

    Returns:
        A list of :class:`Fold`. Empty if the data is too short for one fold.
    """
    if min(train_size, test_size) <= 0:
        msg = "train_size and test_size must be positive"
        raise ValueError(msg)

    folds: list[Fold] = []
    test_start = train_size + embargo
    while test_start + test_size <= n_samples:
        train_lo = 0 if expanding else test_start - embargo - train_size
        train_hi = test_start - embargo  # exclusive
        folds.append(
            Fold(
                train_idx=np.arange(train_lo, train_hi),
                test_idx=np.arange(test_start, test_start + test_size),
            )
        )
        test_start += test_size
    return folds
