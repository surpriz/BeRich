"""Tests for the leak-free stacking ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.models import AveragingEnsemble, LGBMModel, StackingEnsemble
from berich.models.base import Model


def _panel(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    x = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)], index=idx)
    y = pd.Series((x["f0"] - x["f1"] + rng.normal(0, 0.5, n) > 0).astype(int), index=idx)
    w = pd.Series(1.0, index=idx)
    tickers = pd.Series("AAA", index=idx)
    return x, y, w, tickers


def test_stacking_ensemble_is_model_and_predicts_in_range():
    x, y, w, tickers = _panel()
    ens = StackingEnsemble(
        [lambda: LGBMModel(n_estimators=20), lambda: LGBMModel(n_estimators=20, num_leaves=8)],
        embargo=5,
    )
    ens.fit(x, y, sample_weight=w, tickers=tickers)
    assert isinstance(ens, Model)
    proba = ens.predict_proba(x, tickers=tickers)
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_stacking_ensemble_learns_signal():
    x, y, _w, _tickers = _panel()
    ens = StackingEnsemble([lambda: LGBMModel(n_estimators=40)], embargo=5).fit(x, y)
    # On a learnable target the stacked proba should separate the classes on average.
    proba = ens.predict_proba(x)
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_stacking_requires_factory():
    with pytest.raises(ValueError, match="at least one"):
        StackingEnsemble([])


def test_averaging_ensemble_is_model_and_averages_members():
    x, y, w, tickers = _panel()
    # Two members on DIFFERENT feature subsets: the ensemble must slice the full frame per member.
    ens = AveragingEnsemble(
        [
            (lambda: LGBMModel(n_estimators=30), ["f0", "f1"]),
            (lambda: LGBMModel(n_estimators=30), ["f0", "f2", "f3"]),
        ]
    )
    ens.fit(x, y, sample_weight=w, tickers=tickers)
    assert isinstance(ens, Model)
    proba = ens.predict_proba(x, tickers=tickers)
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0) & (proba <= 1))
    # Equals the mean of the members' own probabilities (soft vote, equal weights).
    m0 = LGBMModel(n_estimators=30).fit(x[["f0", "f1"]], y, sample_weight=w)
    m1 = LGBMModel(n_estimators=30).fit(x[["f0", "f2", "f3"]], y, sample_weight=w)
    p0 = m0.predict_proba(x[["f0", "f1"]])
    p1 = m1.predict_proba(x[["f0", "f2", "f3"]])
    assert np.allclose(proba, 0.5 * p0 + 0.5 * p1, atol=1e-9)


def test_averaging_ensemble_learns_signal_and_requires_member():
    x, y, _w, _t = _panel()
    ens = AveragingEnsemble([(lambda: LGBMModel(n_estimators=40), ["f0", "f1"])]).fit(x, y)
    proba = ens.predict_proba(x)
    assert proba[y == 1].mean() > proba[y == 0].mean()
    with pytest.raises(ValueError, match="at least one"):
        AveragingEnsemble([])
