"""Local, rotating backups of the irreplaceable training state.

Everything under ``data/`` is gitignored runtime state, but the per-asset Optuna studies and
promoted models are expensive to recompute (hours of HPO). This makes a timestamped tar.gz of
the studies DB, the per-asset model registry, and the signals/paper DuckDB, keeping the last
``keep`` archives under ``data/backups/``.

Scope: protects against corruption, accidental overwrite, or a bad run clobbering a good
model — NOT against total disk loss (the archives live on the same disk). For off-site
durability, sync ``data/backups/`` elsewhere (rsync/rclone) separately.
"""

from __future__ import annotations

import logging
import tarfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

logger = logging.getLogger(__name__)

_BACKUP_DIRNAME = "backups"
_PREFIX = "berich-data-"
_SUFFIX = ".tar.gz"
DEFAULT_KEEP = 7


def _members(config: Config) -> list[Path]:
    """The paths worth archiving, skipping any that don't exist yet."""
    candidates = [config.optuna_db, config.models_dir, config.db_path]
    return [p for p in candidates if p.exists()]


def _rotate(backup_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` archives; return the ones removed."""
    archives = sorted(
        (p for p in backup_dir.glob(f"{_PREFIX}*{_SUFFIX}") if p.is_file()),
        key=lambda p: p.name,  # name carries the timestamp, so lexical == chronological
    )
    stale = archives[:-keep] if keep > 0 else []
    for p in stale:
        p.unlink(missing_ok=True)
    return stale


def create_backup(config: Config, *, timestamp: str, keep: int = DEFAULT_KEEP) -> dict[str, object]:
    """Archive the training state into ``data/backups/`` and rotate old archives.

    ``timestamp`` is supplied by the caller (the scheduler/CLI stamps it) so this stays
    deterministic and free of wall-clock calls. Returns a summary dict.
    """
    members = _members(config)
    if not members:
        logger.info("backup: nothing to archive yet (no optuna.db / models / duckdb)")
        return {"archived": [], "path": None, "removed": []}

    backup_dir = config.data_dir / _BACKUP_DIRNAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Filesystem-safe stamp: no colons (Windows / some tools choke on them).
    safe_stamp = timestamp.replace(":", "-")
    out_path = backup_dir / f"{_PREFIX}{safe_stamp}{_SUFFIX}"

    with tarfile.open(out_path, "w:gz") as tar:
        for member in members:
            # arcname relative to data_dir so the archive restores cleanly into data/.
            tar.add(member, arcname=member.relative_to(config.data_dir))

    removed = _rotate(backup_dir, keep)
    summary: dict[str, object] = {
        "archived": [m.name for m in members],
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "removed": [p.name for p in removed],
    }
    logger.info(
        "backup: wrote %s (%d members), rotated %d", out_path.name, len(members), len(removed)
    )
    return summary


__all__ = ["DEFAULT_KEEP", "create_backup"]
