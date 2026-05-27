"""On-disk model registry — the seam between GPU training and local serving.

A model is trained anywhere (the LightGBM baseline locally, LSTM/TFT on the GPU box)
and saved here as an *artifact*: the pickled model object plus a JSON metadata sidecar
(framework, feature order, train date, backtest metrics). The handoff is then just
"copy the artifact directory over and promote it":

    GPU machine:  train -> save_model(...) -> promote_if_better(...)
    sync:         scp -r data/models/<name>  ->  local data/models/
    local:        load_active() picks it up; signals/backtest use it, no retrain.

Promotion is guarded: a model only becomes active if its metadata says it beats buy &
hold (the design's go/no-go rule), so a worse model can never silently take over.
Artifacts are loaded with joblib; any model implementing the :class:`Model` protocol
works, including future PyTorch wrappers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import joblib
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

    from berich.models.base import Model

MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"
ACTIVE_POINTER = "active.json"


class ModelMetadata(BaseModel):
    """Everything needed to serve and audit a saved model."""

    name: str
    framework: str  # "lightgbm" | "lstm" | "tft" | ...
    feature_columns: list[str]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict[str, float] = Field(default_factory=dict)
    beats_buy_hold: bool = False
    notes: str = ""


def save_model(
    model: Model,
    metadata: ModelMetadata,
    *,
    registry_dir: Path,
) -> Path:
    """Persist ``model`` and its metadata under ``registry_dir/<name>``; return that dir."""
    artifact_dir = registry_dir / metadata.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / MODEL_FILE)
    (artifact_dir / META_FILE).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return artifact_dir


def load_model(name: str, *, registry_dir: Path) -> tuple[Model, ModelMetadata]:
    """Load a named artifact and its metadata."""
    artifact_dir = registry_dir / name
    meta = ModelMetadata.model_validate_json((artifact_dir / META_FILE).read_text(encoding="utf-8"))
    model: Model = joblib.load(artifact_dir / MODEL_FILE)
    return model, meta


def list_models(registry_dir: Path) -> list[ModelMetadata]:
    """Return metadata for every artifact in the registry, newest first."""
    if not registry_dir.exists():
        return []
    metas = [
        ModelMetadata.model_validate_json((d / META_FILE).read_text(encoding="utf-8"))
        for d in registry_dir.iterdir()
        if (d / META_FILE).exists()
    ]
    return sorted(metas, key=lambda m: m.created_at, reverse=True)


def promote(name: str, *, registry_dir: Path, force: bool = False) -> ModelMetadata:
    """Mark a model as the active one used for serving.

    Refuses to promote a model whose metadata reports it does not beat buy & hold,
    unless ``force=True``. This enforces the design's guard rule at the registry level.
    """
    _, meta = load_model(name, registry_dir=registry_dir)
    if not meta.beats_buy_hold and not force:
        msg = (
            f"refusing to promote '{name}': it does not beat buy & hold "
            f"(use force=True to override)"
        )
        raise ValueError(msg)
    (registry_dir / ACTIVE_POINTER).write_text(json.dumps({"name": name}), encoding="utf-8")
    return meta


def load_active(registry_dir: Path) -> tuple[Model, ModelMetadata] | None:
    """Load the promoted model, or ``None`` if nothing has been promoted yet."""
    pointer = registry_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    name = json.loads(pointer.read_text(encoding="utf-8"))["name"]
    if not (registry_dir / name / META_FILE).exists():
        return None
    return load_model(name, registry_dir=registry_dir)
