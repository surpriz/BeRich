"""Tests for the /ops machine-status collectors (subprocess + scheduler mocked)."""

from __future__ import annotations

from berich.config import AssetUniverses, Config
from berich.ops import (
    _log_level,
    gpus,
    ops_snapshot,
    recent_logs,
    scheduled_jobs,
    sweep_status,
    system_metrics,
    utilization,
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


def test_sweep_status_surfaces_oldest_hpo(monkeypatch, tmp_path):
    log = tmp_path / "sweep.log"
    log.write_text(
        "2026-06-02 16:00:00,1 INFO sweep: done NVDA/long/fixed in 100s promoted=False\n",
        encoding="utf-8",
    )
    lines = log.read_text(encoding="utf-8").splitlines()

    def _fake_run(cmd):
        if cmd[0] == "pgrep":
            return "1\n"
        if cmd[0] == "grep":
            return "\n".join(lines)
        if cmd[0] == "tail":
            return lines[-1]
        return None

    monkeypatch.setattr("berich.ops._run", _fake_run)
    # Per-asset studies for two current tickers + ORPHAN studies from retired subsystems (the
    # global model and the long/short ranker). The orphans are older but must be IGNORED — /ops
    # scopes to currently-swept (ticker, side) combos, so it reconciles with /training.
    monkeypatch.setattr(
        "berich.training.status._hpo_last_trial_times",
        lambda _db: {
            "berich-hpo-AAA-lgbm-long-auc": "2026-06-08T06:00:00+00:00",
            "berich-hpo-BBB-lgbm-long-auc": "2026-06-01T06:00:00+00:00",
            "berich-hpo-tft": "2026-05-01T06:00:00+00:00",  # orphan: retired global model
            "berich-longshort-lgbm": "2026-05-02T06:00:00+00:00",  # orphan: retired ranker
        },
    )
    cfg = Config(data_dir=tmp_path, universes=AssetUniverses(us_stocks=["AAA", "BBB"]))
    st = sweep_status(cfg)
    # Oldest among CURRENT assets (BBB long), not the older retired orphans.
    assert st["oldest_hpo_at"] == "2026-06-01T06:00:00+00:00"
    assert isinstance(st["oldest_hpo_age_seconds"], int)
    assert st["oldest_hpo_age_seconds"] > 0


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
        "utilization",
        "alerts",
        "logs",
    }
    assert snap["gpus"] == []
    assert isinstance(snap["hpo"]["total"], int)
    assert snap["sweep"]["running"] is False
    # No sweep load -> the box is idle, not under-used.
    assert snap["utilization"]["verdict"] == "idle"


def test_utilization_flags_idle_twin_gpu_as_underused():
    # One card pegged, its twin parked, CPU barely loaded, while the sweep runs: throughput on the
    # table -> "under", with the asymmetry called out explicitly.
    gpu_list = [
        {"index": 0, "util_pct": 95, "mem_used_mb": 7000, "mem_total_mb": 24467, "temp_c": 48},
        {"index": 1, "util_pct": 0, "mem_used_mb": 2700, "mem_total_mb": 24467, "temp_c": 27},
    ]
    system = {"load_ratio": 0.1, "n_cpus": 24}
    out = utilization(gpu_list, system, {"running": True})
    assert out["verdict"] == "under"
    assert "gpu_idle_card" in out["reasons"]
    assert "cpu_low" in out["reasons"]
    assert out["idle_gpus"] == 1
    assert out["gpu_avg_pct"] == 48


def test_utilization_balanced_and_saturated():
    both_busy = [
        {"index": 0, "util_pct": 70, "mem_used_mb": 1, "mem_total_mb": 2, "temp_c": 60},
        {"index": 1, "util_pct": 65, "mem_used_mb": 1, "mem_total_mb": 2, "temp_c": 60},
    ]
    assert utilization(both_busy, {"load_ratio": 0.8}, {"running": True})["verdict"] == "balanced"
    maxed = [{"index": 0, "util_pct": 98, "mem_used_mb": 1, "mem_total_mb": 2, "temp_c": 80}]
    out = utilization(maxed, {"load_ratio": 1.5}, {"running": True})
    assert out["verdict"] == "over"
    assert {"gpu_high", "cpu_high"} <= set(out["reasons"])


def test_utilization_idle_when_sweep_stopped():
    busy = [{"index": 0, "util_pct": 5, "mem_used_mb": 1, "mem_total_mb": 2, "temp_c": 30}]
    out = utilization(busy, {"load_ratio": 0.05}, {"running": False})
    assert out["verdict"] == "idle"
    assert out["reasons"] == ["sweep_stopped"]
