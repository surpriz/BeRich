"""Tests for walk-forward splits, sequence windowing, and the scaler."""

from __future__ import annotations

import numpy as np

from berich.datasets.scaling import StandardScaler
from berich.datasets.splits import walk_forward_splits
from berich.datasets.windows import make_sequences


def test_walk_forward_no_overlap_and_chronological():
    folds = walk_forward_splits(100, train_size=40, test_size=20, embargo=0)
    assert len(folds) == 3
    for f in folds:
        # Train strictly precedes test.
        assert f.train_idx.max() < f.test_idx.min()


def test_walk_forward_embargo_gap():
    folds = walk_forward_splits(100, train_size=40, test_size=20, embargo=5)
    f = folds[0]
    # Embargo of 5 rows sits between the last train row and the first test row.
    assert f.test_idx.min() - f.train_idx.max() == 6


def test_walk_forward_expanding_grows():
    folds = walk_forward_splits(120, train_size=40, test_size=20, expanding=True)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_make_sequences_shapes():
    feats = np.arange(20 * 3).reshape(20, 3).astype(float)
    labels = np.arange(20).astype(float)
    x, y = make_sequences(feats, labels, lookback=5)
    assert x.shape == (16, 5, 3)
    assert y.shape == (16,)
    # Each window ends at its labeled bar.
    assert y[0] == labels[4]
    np.testing.assert_array_equal(x[0], feats[0:5])


def test_scaler_uses_fit_stats():
    train = np.array([[0.0], [2.0], [4.0]])
    scaler = StandardScaler().fit(train)
    # Transforming new data reuses train mean/std (no refit).
    out = scaler.transform(np.array([[2.0]]))
    assert abs(out[0, 0]) < 1e-9
