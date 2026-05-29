"""Tests for the probability calibrator and the meta-labeling dataset/model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.datasets.assemble import SupervisedDataset
from berich.models.base import Model
from berich.models.meta_labeler import PRIMARY_PROBA_COL, MetaLabeler
from berich.signals.calibration import (
    ProbaCalibrator,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
)
from berich.training.meta import build_meta_dataset, train_meta_model
from berich.training.walk_forward import OofResult


def test_calibrator_isotonic_monotone_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    # Realized wins follow a shifted, miscalibrated relationship with raw proba.
    y = (rng.uniform(0, 1, 2000) < (0.2 + 0.6 * raw)).astype(int)
    cal = fit_calibrator(raw, y, method="isotonic")
    out = cal.transform(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= -1e-9)  # isotonic => non-decreasing


def test_calibrator_save_load_roundtrip(tmp_path):
    cal = fit_calibrator(np.linspace(0, 1, 100), (np.linspace(0, 1, 100) > 0.5).astype(int))
    save_calibrator(cal, artifact_dir=tmp_path)
    loaded = load_calibrator(tmp_path)
    assert isinstance(loaded, ProbaCalibrator)
    assert np.allclose(loaded.transform(np.array([0.3, 0.8])), cal.transform(np.array([0.3, 0.8])))
    assert load_calibrator(tmp_path / "nope") is None


def _supervised(n: int = 300, seed: int = 0) -> SupervisedDataset:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    x = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=["f0", "f1", "f2", "f3"], index=idx)
    y = pd.Series((x["f0"] + rng.normal(0, 0.5, n) > 0).astype(int), index=idx)
    return SupervisedDataset(
        x=x,
        y=y,
        weight=pd.Series(1.0, index=idx),
        dates=pd.DatetimeIndex(idx),
        tickers=pd.Series("AAA", index=idx),
    )


def _oof_for(base: SupervisedDataset) -> OofResult:
    # Out-of-fold probas correlated with the true label (stand-in for a real primary OOF).
    rng = np.random.default_rng(1)
    proba = np.clip(0.5 + 0.3 * (base.y.to_numpy() - 0.5) * 2 + rng.normal(0, 0.1, len(base)), 0, 1)
    frame = pd.DataFrame(
        {"proba": proba, "y_true": base.y.to_numpy(), "ticker": base.tickers.to_numpy()},
        index=base.dates,
    )
    frame.index.name = "date"
    return OofResult(frame=frame)


def test_meta_dataset_filters_buys_and_adds_primary_proba():
    base = _supervised()
    oof = _oof_for(base)
    meta = build_meta_dataset(base, oof, buy_threshold=0.55)
    assert PRIMARY_PROBA_COL in meta.x.columns
    assert (meta.x[PRIMARY_PROBA_COL] >= 0.55).all()  # only BUY candidates kept
    assert len(meta) == len(meta.y)
    assert set(meta.y.unique()) <= {0, 1}


def test_meta_model_trains_and_is_model():
    base = _supervised()
    oof = _oof_for(base)
    meta = build_meta_dataset(base, oof, buy_threshold=0.55)
    model, auc = train_meta_model(meta)
    assert isinstance(model, (MetaLabeler, Model))
    p = model.predict_proba(meta.x)
    assert p.shape == (len(meta),)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isnan(auc) or 0.0 <= auc <= 1.0
