"""Tests for the model registry: save/load, guarded promotion, active pointer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.models import (
    LGBMModel,
    ModelMetadata,
    list_models,
    load_active,
    load_model,
    promote,
    save_model,
)


def _trained_model() -> LGBMModel:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
    y = pd.Series((x["a"] + rng.normal(0, 0.1, 200) > 0).astype(int))
    return LGBMModel(n_estimators=10).fit(x, y)


def _meta(name: str, *, beats: bool) -> ModelMetadata:
    return ModelMetadata(
        name=name,
        framework="lightgbm",
        feature_columns=["a", "b", "c"],
        metrics={"auc": 0.6},
        beats_buy_hold=beats,
    )


def test_save_and_load_roundtrip(tmp_path):
    model = _trained_model()
    save_model(model, _meta("m1", beats=True), registry_dir=tmp_path)

    loaded, meta = load_model("m1", registry_dir=tmp_path)
    assert meta.framework == "lightgbm"
    assert meta.feature_columns == ["a", "b", "c"]
    x = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
    assert loaded.predict_proba(x).shape == (5,)


def test_promote_guarded_by_beats_buy_hold(tmp_path):
    save_model(_trained_model(), _meta("weak", beats=False), registry_dir=tmp_path)
    with pytest.raises(ValueError, match="does not beat buy & hold"):
        promote("weak", registry_dir=tmp_path)
    # Force overrides the guard.
    promote("weak", registry_dir=tmp_path, force=True)
    assert load_active(tmp_path) is not None


def test_load_active_returns_promoted(tmp_path):
    assert load_active(tmp_path) is None  # nothing promoted yet
    save_model(_trained_model(), _meta("good", beats=True), registry_dir=tmp_path)
    promote("good", registry_dir=tmp_path)
    active = load_active(tmp_path)
    assert active is not None
    _, meta = active
    assert meta.name == "good"


def test_list_models(tmp_path):
    save_model(_trained_model(), _meta("a", beats=True), registry_dir=tmp_path)
    save_model(_trained_model(), _meta("b", beats=False), registry_dir=tmp_path)
    names = {m.name for m in list_models(tmp_path)}
    assert names == {"a", "b"}
