"""Tests for the rotating local backup of training state."""

from __future__ import annotations

import tarfile

from berich.backup import create_backup
from berich.config import Config


def _seed(cfg: Config) -> None:
    """Write a fake optuna.db + a model artifact + a signals DB under data/."""
    cfg.optuna_db.parent.mkdir(parents=True, exist_ok=True)
    cfg.optuna_db.write_bytes(b"fake-sqlite")
    art = cfg.models_dir / "tickers" / "AAA" / "long"
    art.mkdir(parents=True, exist_ok=True)
    (art / "model.joblib").write_bytes(b"fake-model")
    cfg.db_path.write_bytes(b"fake-duckdb")


def test_backup_archives_members_and_is_restorable(tmp_path):
    cfg = Config(data_dir=tmp_path)
    _seed(cfg)
    res = create_backup(cfg, timestamp="2026-06-01T03:00:00")
    assert res["path"] is not None
    archive = tmp_path / "backups" / "berich-data-2026-06-01T03-00-00.tar.gz"
    assert archive.exists()
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    # Members are stored relative to data/ so they restore cleanly into data/.
    assert "optuna.db" in names
    assert "berich.duckdb" in names
    assert any(n.endswith("model.joblib") for n in names)


def test_backup_noop_when_nothing_to_archive(tmp_path):
    cfg = Config(data_dir=tmp_path)
    res = create_backup(cfg, timestamp="2026-06-01T03:00:00")
    assert res["path"] is None
    assert res["archived"] == []


def test_backup_rotation_keeps_latest(tmp_path):
    cfg = Config(data_dir=tmp_path)
    _seed(cfg)
    stamps = ["2026-06-01T01:00:00", "2026-06-01T02:00:00", "2026-06-01T03:00:00"]
    for s in stamps:
        create_backup(cfg, timestamp=s, keep=2)
    kept = sorted(p.name for p in (tmp_path / "backups").glob("*.tar.gz"))
    assert kept == [
        "berich-data-2026-06-01T02-00-00.tar.gz",
        "berich-data-2026-06-01T03-00-00.tar.gz",
    ]
