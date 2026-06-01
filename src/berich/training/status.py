"""Read-only inventory of per-asset training/optimization state for the dashboard.

Scans the per-ticker registry (``data/models/tickers/<TICKER>/<side>/``) and the Optuna RDB
(``data/optuna.db``) to answer: which assets have been trained, with which model, when, are
they promoted or advisory-only, and has a per-asset HPO search actually run. Nothing here
trains or mutates — it's the backing data for the ``/api/training`` endpoint and its tab.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from berich.config import safe_ticker_slug
from berich.models.registry import ACTIVE_POINTER, META_FILE, ModelMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

logger = logging.getLogger(__name__)

_STATUS_FILE = "status.json"
_SIDES = ("long", "short")


def _hpo_trial_counts(optuna_db: Path) -> dict[str, int]:
    """Map per-ticker study name -> completed-trial count (empty if the RDB is absent)."""
    if not optuna_db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{optuna_db}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.warning("could not open optuna db at %s", optuna_db, exc_info=True)
        return {}
    try:
        rows = con.execute(
            "SELECT st.study_name, count(t.trial_id) "
            "FROM studies st LEFT JOIN trials t ON t.study_id = st.study_id "
            "GROUP BY st.study_name"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    return {name: int(n) for name, n in rows}


def _hpo_trials_for(counts: dict[str, int], ticker: str, model: str | None, side: str) -> int:
    """Sum HPO trials across that ticker+side's studies (all frameworks, or one if given)."""
    slug = safe_ticker_slug(ticker)
    total = 0
    for name, n in counts.items():
        # berich-hpo-<SLUG>-<model>-<side>-<metric>
        if not name.startswith(f"berich-hpo-{slug}-"):
            continue
        if f"-{side}-" not in name:
            continue
        if model is not None and f"-{model}-" not in name:
            continue
        total += n
    return total


def _side_entry(
    config: Config, ticker: str, side: str, counts: dict[str, int]
) -> dict[str, object]:
    """Build one (ticker, side) row: status, winner, metrics, candidates, HPO trials, time."""
    reg = config.model_dir_for_ticker(ticker, side)
    entry: dict[str, object] = {
        "ticker": ticker,
        "asset_class": config.asset_class_for(ticker),
        "side": side,
        "status": "never_trained",
        "winner": None,
        "framework": None,
        "trained_at": None,
        "metrics": {},
        "candidates": [],
        "hpo_trials": _hpo_trials_for(counts, ticker, None, side),
    }
    if not reg.exists():
        return entry

    # The tournament summary (full candidate slate + run time), when present.
    status_path = reg / _STATUS_FILE
    if status_path.exists():
        try:
            summary = json.loads(status_path.read_text(encoding="utf-8"))
            entry["trained_at"] = summary.get("trained_at")
            entry["candidates"] = summary.get("candidates", [])
        except (OSError, ValueError):
            logger.warning("unreadable status.json for %s/%s", ticker, side, exc_info=True)

    # The promoted artifact (if any) — authoritative for status + headline metrics.
    if (reg / ACTIVE_POINTER).exists():
        promoted = _promoted_meta(reg, ticker, side)
        if promoted is not None:
            _fill_from_meta(entry, promoted, status="promoted", winner=promoted.name)
            return entry

    # No promoted pointer: advisory-only if any candidate artifact was saved, else never trained.
    saved = [d for d in reg.iterdir() if d.is_dir() and (d / META_FILE).exists()]
    if saved or entry["candidates"]:
        entry["status"] = "advisory_only"
        if saved and not entry["metrics"]:
            meta = _read_meta(saved[0])
            if meta is not None:
                _fill_from_meta(entry, meta, status="advisory_only", winner=None)
    return entry


def _read_meta(artifact_dir: Path) -> ModelMetadata | None:
    """Load one artifact's metadata, or None if unreadable."""
    try:
        return ModelMetadata.model_validate_json(
            (artifact_dir / META_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _promoted_meta(reg: Path, ticker: str, side: str) -> ModelMetadata | None:
    """Metadata of the active (promoted) artifact under ``reg``, or None if unreadable."""
    try:
        name = json.loads((reg / ACTIVE_POINTER).read_text(encoding="utf-8"))["name"]
    except (OSError, ValueError, KeyError):
        logger.warning("unreadable active pointer for %s/%s", ticker, side, exc_info=True)
        return None
    return _read_meta(reg / name)


def _fill_from_meta(
    entry: dict[str, object], meta: ModelMetadata, *, status: str, winner: str | None
) -> None:
    """Populate an entry's status/framework/metrics/trained_at from a model's metadata."""
    entry["status"] = status
    if winner is not None:
        entry["winner"] = winner
    entry["framework"] = meta.framework
    entry["metrics"] = meta.metrics
    if entry["trained_at"] is None:
        entry["trained_at"] = meta.created_at


def training_status(config: Config) -> list[dict[str, object]]:
    """Per-(ticker, side) training inventory across every configured tradeable asset."""
    counts = _hpo_trial_counts(config.optuna_db)
    return [
        _side_entry(config, ticker, side, counts)
        for ticker in config.tradeable_tickers()
        for side in _SIDES
    ]


__all__ = ["training_status"]
