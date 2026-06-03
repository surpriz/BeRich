"""Tests for the /ops machine-status collectors (subprocess + scheduler mocked)."""

from __future__ import annotations

from berich.config import Config
from berich.ops import (
    _log_level,
    gpus,
    ops_snapshot,
    recent_logs,
    scheduled_jobs,
    sweep_status,
    system_metrics,
)


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


def test_recent_logs_parses_message(monkeypatch, tmp_path):
    raw = "2026-06-01T11:31:31+02:00 host uv[123]: INFO apscheduler.scheduler: Scheduler started\n"

    def _fake_run(cmd):
        return raw if cmd[0] == "journalctl" else None

    monkeypatch.setattr("berich.ops._run", _fake_run)
    logs = recent_logs(Config(data_dir=tmp_path))
    assert len(logs) == 1
    # Timestamp is normalized to YYYY-MM-DDTHH:MM:SS so scheduler + sweep lines sort together.
    assert logs[0]["time"] == "2026-06-01T11:31:31"
    # The host/unit prefix is stripped; the level+logger+message remain (useful for a log view).
    assert logs[0]["message"] == "INFO apscheduler.scheduler: Scheduler started"
    assert logs[0]["level"] == "info"


def test_sweep_status_parses_drainer_log(monkeypatch, tmp_path):
    log = tmp_path / "sweep.log"
    log.write_text(
        "2026-06-02 16:00:00,1 INFO sweep: done NVDA/long/trailing in 530s promoted=False\n"
        "2026-06-02 16:01:00,2 INFO sweep: [9 done, 0 pend] start NVDA/long/trailing_tp\n",
        encoding="utf-8",
    )

    lines = log.read_text(encoding="utf-8").splitlines()

    def _fake_run(cmd):
        if cmd[0] == "pgrep":
            return "12345\n"  # process alive
        if cmd[0] == "grep":  # structured drainer lines
            return "\n".join(line for line in lines if " sweep: " in line)
        if cmd[0] == "tail":  # last line, for liveness
            return lines[-1]
        return None

    monkeypatch.setattr("berich.ops._run", _fake_run)
    st = sweep_status(Config(data_dir=tmp_path))
    assert st["running"] is True
    assert st["current"] == "NVDA/long/trailing_tp"
    assert st["avg_seconds"] == 530
    assert st["gave_up"] == 0
    assert st["last_activity"] == "2026-06-02T16:01:00"


def test_log_level_classifies_by_real_level_not_keywords():
    # An INFO summary that merely contains the dict key 'failed' is NOT an error (the bug).
    assert _log_level("INFO berich.scheduler.jobs: ticker_hpo_queue: {'failed': 0}") == "info"
    assert _log_level("WARNING berich.scheduler.jobs: refresh_signals failed") == "warning"
    assert _log_level("ERROR sweep: FAILED AAA/long/trailing") == "error"
    assert _log_level("Traceback (most recent call last):") == "error"


def test_system_metrics_has_core_fields(tmp_path):
    m = system_metrics(Config(data_dir=tmp_path))
    # Best-effort on Linux: these come from /proc + the data partition.
    assert "cpu_pct" in m
    assert isinstance(m.get("n_cpus"), int)
    assert isinstance(m.get("disk_total_gb"), float)


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
    assert set(snap.keys()) == {
        "gpus",
        "system",
        "sweep",
        "scheduler",
        "jobs",
        "hpo",
        "alerts",
        "logs",
    }
    assert snap["gpus"] == []
    assert isinstance(snap["hpo"]["total"], int)
    assert snap["sweep"]["running"] is False
