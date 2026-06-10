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
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import joblib
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

    from berich.models.base import Model

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"
ACTIVE_POINTER = "active.json"

# Market-neutral promotion thresholds (the strategy has no buy-&-hold benchmark, so the
# bar is a positive, statistically significant Sharpe — see backtest/significance.py).
MIN_DEFLATED_SHARPE = 0.95
MAX_SHARPE_PVALUE = 0.05

# A handful of lucky trades can post any Sharpe / beat any benchmark by chance. Below this many
# closed OOS trades the walk-forward verdict is noise, so the gate refuses to promote regardless
# of side — the honest "not enough evidence" outcome. ``n_trades`` is written by the tournament
# from the backtest; legacy artifacts that predate the metric read 0.0 and so fail this floor
# (correct: their evidence was never recorded).
MIN_TRADES = 20


class ModelMetadata(BaseModel):
    """Everything needed to serve and audit a saved model."""

    name: str
    framework: str  # "lightgbm" | "lstm" | "tft" | ...
    feature_columns: list[str]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict[str, float] = Field(default_factory=dict)
    beats_buy_hold: bool = False
    # "long_only" keeps the historical beats-buy-&-hold gate; existing artifacts that
    # predate this field deserialize to "long_only", so the legacy guard is unchanged.
    strategy_type: Literal["long_only", "market_neutral"] = "long_only"
    # Per-ticker tournament bookkeeping. ``side="short"`` switches the promotion gate to the
    # significance bar (a short has no buy-&-hold benchmark — its benchmark is cash/zero).
    # Defaults keep every pre-existing artifact deserializing as a long single-asset/pooled model.
    side: Literal["long", "short"] = "long"
    ticker: str | None = None
    # Triple-barrier horizon (trading days) this model was trained on. The per-asset HPO can
    # search it, so it's recorded here and read back at serve time — the served SL/TP and vol
    # forecast must use the SAME horizon the model learned on. Defaults to 10 (the historical
    # global horizon) so every pre-existing artifact deserializes unchanged.
    horizon_days: int = 10
    # Exit strategy this model was trained, backtested AND must be served under: "fixed" (the
    # historical TP/SL triple barrier), "trailing" (ratcheting stop, no TP), or "trailing_tp"
    # (TP cap + ratcheting stop). Audit-only — the promotion gate is exit-strategy-agnostic (it
    # judges the resulting returns). Defaults to "fixed" so pre-existing artifacts are unchanged.
    exit_strategy: str = "fixed"
    # Bar interval this model was trained/served on: "1d" (daily swing) or "1h" (intraday POC).
    # Audit-only — the registry/guard are interval-agnostic. Defaults to "1d" so every
    # pre-existing artifact deserializes unchanged. For intraday, ``horizon_days`` counts bars.
    interval: str = "1d"
    # Per-asset decision threshold on the CALIBRATED win probability, chosen at training to
    # maximize OOS risk-adjusted expectancy for this (ticker, side). ``None`` => serve falls back
    # to the global ``signals.buy_threshold`` / ``short_threshold``. Defaults to None so every
    # pre-existing artifact keeps using the global threshold.
    decision_threshold: float | None = None
    notes: str = ""


def save_model(
    model: Model,
    metadata: ModelMetadata,
    *,
    registry_dir: Path,
) -> Path:
    """Persist ``model`` and its metadata under ``registry_dir/<name>``; return that dir.

    Written atomically (temp file + ``os.replace``), and the metadata last: a process killed
    mid-save can never leave a half-written ``metadata.json`` that breaks serving — a reader
    sees either no metadata yet (artifact skipped) or the complete previous one.
    """
    artifact_dir = registry_dir / metadata.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Write each file to a temp sibling then atomically rename onto the final name (Path.replace
    # is an atomic rename on POSIX). Metadata is written LAST, so a kill mid-save leaves either no
    # metadata (artifact skipped by list_models) or a complete, valid one — never a partial file.
    model_tmp = artifact_dir / (MODEL_FILE + ".tmp")
    joblib.dump(model, model_tmp)
    model_tmp.replace(artifact_dir / MODEL_FILE)
    meta_tmp = artifact_dir / (META_FILE + ".tmp")
    meta_tmp.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    meta_tmp.replace(artifact_dir / META_FILE)
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
    metas: list[ModelMetadata] = []
    for d in registry_dir.iterdir():
        if not (d / META_FILE).exists():
            continue
        try:
            metas.append(
                ModelMetadata.model_validate_json((d / META_FILE).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            # A single unreadable/invalid artifact must never break serving — skip it. (ValueError
            # covers pydantic's ValidationError, e.g. a legacy metadata with a non-finite metric.)
            logger.warning("skipping unreadable model metadata at %s", d, exc_info=True)
    return sorted(metas, key=lambda m: m.created_at, reverse=True)


def _trade_count_failure(meta: ModelMetadata) -> str | None:
    """Reject a model whose OOS trade count is too thin to trust (applies to every side)."""
    n_trades = meta.metrics.get("n_trades", 0.0)
    if n_trades < MIN_TRADES:
        return f"only {n_trades:.0f} OOS trades (< {MIN_TRADES})"
    return None


def _significance_failure(meta: ModelMetadata) -> str | None:
    """Positive, statistically significant Sharpe gate (no buy-&-hold benchmark)."""
    sharpe = meta.metrics.get("sharpe", 0.0)
    dsr = meta.metrics.get("deflated_sharpe", 0.0)
    pval = meta.metrics.get("sharpe_pvalue", 1.0)
    if sharpe <= 0:
        return f"Sharpe is not positive (sharpe={sharpe:.3f})"
    if dsr < MIN_DEFLATED_SHARPE:
        return f"deflated Sharpe {dsr:.3f} < {MIN_DEFLATED_SHARPE}"
    if pval >= MAX_SHARPE_PVALUE:
        return f"Sharpe p-value {pval:.3f} >= {MAX_SHARPE_PVALUE}"
    return None


def _gate_failure(meta: ModelMetadata) -> str | None:
    """Reason a model fails its promotion gate, or ``None`` if it passes.

    - **every side** first clears a minimum OOS trade count (a few lucky trades prove nothing).
    - ``side="short"``: a directional short has no buy-&-hold benchmark, so its bar is a
      positive, significant Sharpe vs cash (same significance test as market-neutral).
    - ``long_only`` (long side): must beat buy & hold **and** post a positive, significant Sharpe.
      The significance floor was added because on a near-flat benchmark (forex/commodities) "beats
      buy & hold" is a near-trivial bar that promoted edge-less longs; a real long must also clear
      the same anti-luck test as a short.
    - ``market_neutral``: positive, significant Sharpe.
    """
    thin = _trade_count_failure(meta)
    if thin is not None:
        return thin
    if meta.side == "short":
        return _significance_failure(meta)
    if meta.strategy_type == "long_only":
        if not meta.beats_buy_hold:
            return "it does not beat buy & hold"
        return _significance_failure(meta)
    return _significance_failure(meta)


def model_tier(meta: ModelMetadata, *, promoted: bool) -> str:
    """Three-way trust tier used by serving and the paper book.

    - ``"promoted"``: cleared the hardened guard (has an ``active.json``); the real-capital book
      trades it.
    - ``"observe"``: missed the guard but still shows a positive OOS Sharpe on enough trades — a
      near-miss worth tracking live in a paper-only shadow book (no capital), so we keep collecting
      forward evidence instead of discarding it.
    - ``"advisory"``: not enough evidence (thin trades or non-positive Sharpe). Shown for inspection
      only; never paper-traded.

    Keeps the honest design: the real book only ever follows ``promoted`` models.
    """
    if promoted:
        return "promoted"
    if _trade_count_failure(meta) is None and meta.metrics.get("sharpe", 0.0) > 0:
        return "observe"
    return "advisory"


def promote(name: str, *, registry_dir: Path, force: bool = False) -> ModelMetadata:
    """Mark a model as the active one used for serving.

    Refuses to promote a model that fails its strategy-type gate (beats buy & hold for
    long-only; positive, significant Sharpe for market-neutral) unless ``force=True``.
    This enforces the design's guard rule at the registry level.
    """
    _, meta = load_model(name, registry_dir=registry_dir)
    failure = _gate_failure(meta)
    if failure is not None and not force:
        msg = f"refusing to promote '{name}': {failure} (use force=True to override)"
        raise ValueError(msg)
    (registry_dir / ACTIVE_POINTER).write_text(json.dumps({"name": name}), encoding="utf-8")
    return meta


def demote(registry_dir: Path) -> bool:
    """Remove the active pointer so the asset is no longer promoted; return whether one existed.

    The artifact stays on disk, so serving falls back to it as an *observe*/advisory candidate
    (see :func:`load_best` / :func:`model_tier`) — demotion only revokes real-capital trust, it
    never deletes a model. Used by the sweep-level FDR reconciliation to walk back promotions that
    don't survive multiple-testing control.
    """
    pointer = registry_dir / ACTIVE_POINTER
    if not pointer.exists():
        return False
    pointer.unlink()
    return True


def load_active(registry_dir: Path) -> tuple[Model, ModelMetadata] | None:
    """Load the promoted model, or ``None`` if nothing has been promoted yet."""
    pointer = registry_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    name = json.loads(pointer.read_text(encoding="utf-8"))["name"]
    if not (registry_dir / name / META_FILE).exists():
        return None
    return load_model(name, registry_dir=registry_dir)


def served_model_name(registry_dir: Path) -> str | None:
    """Zoo model name ("lgbm" | "lstm" | "patchtst" | "tft") of the served winner.

    Reads only the active pointer (names are ``<model>-<side>``), never the model weights —
    cheap enough for scheduling decisions like the targeted nightly top-up.
    """
    pointer = registry_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    name = str(json.loads(pointer.read_text(encoding="utf-8")).get("name", ""))
    return name.split("-", maxsplit=1)[0] or None


def load_best(registry_dir: Path) -> tuple[Model, ModelMetadata] | None:
    """Load the promoted model if any, else the best saved candidate by AUC (advisory).

    The per-ticker tournament saves its best-AUC candidate even when none clears the guard,
    so an *optimized but advisory* asset can still be served from its own model rather than a
    generic fallback. Returns ``None`` only when the registry holds no artifact at all. Callers
    must consult ``meta`` to know whether it's promoted (``beats_buy_hold`` / the active pointer)
    before treating the signal as anything but advisory.
    """
    active = load_active(registry_dir)
    if active is not None:
        return active
    metas = list_models(registry_dir)
    if not metas:
        return None
    best = max(metas, key=lambda m: m.metrics.get("auc", 0.0))
    return load_model(best.name, registry_dir=registry_dir)
