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


def _mn_meta(name: str, *, sharpe: float, dsr: float, pval: float) -> ModelMetadata:
    return ModelMetadata(
        name=name,
        framework="lightgbm-ranker",
        feature_columns=["a", "b", "c"],
        metrics={"sharpe": sharpe, "deflated_sharpe": dsr, "sharpe_pvalue": pval},
        strategy_type="market_neutral",
    )


def test_market_neutral_promotes_on_significant_sharpe(tmp_path):
    save_model(
        _trained_model(), _mn_meta("ls", sharpe=1.2, dsr=0.98, pval=0.01), registry_dir=tmp_path
    )
    promote("ls", registry_dir=tmp_path)  # beats_buy_hold is False but gate is Sharpe-based
    active = load_active(tmp_path)
    assert active is not None and active[1].name == "ls"


def test_market_neutral_refused_on_weak_dsr(tmp_path):
    save_model(
        _trained_model(), _mn_meta("weak_ls", sharpe=0.4, dsr=0.6, pval=0.2), registry_dir=tmp_path
    )
    with pytest.raises(ValueError, match="deflated Sharpe"):
        promote("weak_ls", registry_dir=tmp_path)


def test_legacy_metadata_without_strategy_type_defaults_long_only(tmp_path):
    # An artifact written before strategy_type existed must still load and gate as long-only.
    save_model(_trained_model(), _meta("legacy", beats=True), registry_dir=tmp_path)
    # Overwrite with a pre-field metadata file (no strategy_type key).
    (tmp_path / "legacy" / "metadata.json").write_text(
        '{"name": "legacy", "framework": "lightgbm", "feature_columns": ["a", "b", "c"], '
        '"metrics": {}, "beats_buy_hold": true}',
        encoding="utf-8",
    )
    meta = load_model("legacy", registry_dir=tmp_path)[1]
    assert meta.strategy_type == "long_only"
    promote("legacy", registry_dir=tmp_path)  # legacy long-only gate still applies
