"""Tests for the /ops machine-status collectors (subprocess + scheduler mocked)."""

from __future__ import annotations

from berich.config import Config
from berich.ops import gpus, ops_snapshot, recent_logs, scheduled_jobs


def test_gpus_parses_nvidia_smi(monkeypatch):
    csv = "0, NVIDIA RTX, 72, 18000, 24467, 61\n1, NVIDIA RTX, 0, 2, 24467, 33\n"
    monkeypatch.setattr("berich.ops._run", lambda _cmd: csv)
    out = gpus()
    assert len(out) == 2
    assert out[0] == {
        "index": 0,
        "name": "NVIDIA RTX",
        "util_pct": 72,
        "mem_used_mb": 18000,
        "mem_total_mb": 24467,
        "temp_c": 61,
    }


def test_gpus_empty_when_tool_absent(monkeypatch):
    monkeypatch.setattr("berich.ops._run", lambda _cmd: None)
    assert gpus() == []


def test_recent_logs_parses_message(monkeypatch):
    raw = "2026-06-01T11:31:31+02:00 host uv[123]: INFO apscheduler.scheduler: Scheduler started\n"
    monkeypatch.setattr("berich.ops._run", lambda _cmd: raw)
    logs = recent_logs()
    assert len(logs) == 1
    assert logs[0]["time"] == "2026-06-01T11:31:31+02:00"
    # The host/unit prefix is stripped; the level+logger+message remain (useful for a log view).
    assert logs[0]["message"] == "INFO apscheduler.scheduler: Scheduler started"


def test_scheduled_jobs_have_next_run():
    jobs = scheduled_jobs(Config(watchlist=["AAPL"]))
    ids = {j["id"] for j in jobs}
    assert "ticker_hpo_queue" in ids
    assert "backup" in ids
    # Every job exposes a next_run (cron triggers always have one).
    assert all(j["next_run"] for j in jobs)


def test_ops_snapshot_shape(monkeypatch, tmp_path):
    monkeypatch.setattr("berich.ops._run", lambda _cmd: None)  # no GPU / systemd / journald
    snap = ops_snapshot(Config(data_dir=tmp_path, watchlist=["AAPL"]))
    assert set(snap.keys()) == {"gpus", "scheduler", "jobs", "hpo", "logs"}
    assert snap["gpus"] == []
    assert isinstance(snap["hpo"]["total"], int)
