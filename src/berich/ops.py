"""Live machine-status collectors for the ``/ops`` dashboard.

The scheduler, API, and frontend run as separate processes, so the API can't read the
scheduler's in-memory state directly. Instead this module reads *observable* signals the box
already exposes: GPU usage via ``nvidia-smi``, service state + recent logs via systemd/journald,
the scheduler's next-fire times by rebuilding the job table, and the per-asset HPO queue
progress from the training-status scan. Every collector is best-effort and degrades to an
empty/neutral payload rather than raising, so one missing tool never breaks the dashboard.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from berich.config import Config

logger = logging.getLogger(__name__)

_SCHEDULER_UNIT = "berich-scheduler"
_TIMEOUT = 5  # seconds — never let a hung subprocess stall the dashboard
_SWEEP_LOG = "sweep.log"  # under config.data_dir — the background training drainer's output
_SWEEP_PROC = "run_full_sweep.py"
_TS_RE = re.compile(r"(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)")


def _run(cmd: list[str]) -> str | None:
    """Run a read-only command, returning stdout or None on any failure/timeout."""
    try:
        out = subprocess.run(  # noqa: S603 — fixed arg lists, no shell, no user input
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _log_level(message: str) -> str:
    """Classify a log line as error / warning / info from its text (best-effort)."""
    upper = message.upper()
    if "ERROR" in upper or "TRACEBACK" in upper or "FAILED" in upper or "EXCEPTION" in upper:
        return "error"
    if "WARNING" in upper or "GIVING UP" in upper or "SKIPPED" in upper:
        return "warning"
    return "info"


def system_metrics(config: Config) -> dict[str, object]:
    """CPU / RAM / disk utilization so the box can be seen as well-used (not idle, not maxed).

    No extra dependency: CPU% from a short /proc/stat sample, load average from os, memory from
    /proc/meminfo, disk from the data partition (where models + the Optuna RDB grow).
    """
    out: dict[str, object] = {}
    try:
        idle0, total0 = _cpu_times()
        time.sleep(0.12)
        idle1, total1 = _cpu_times()
        dt = total1 - total0
        out["cpu_pct"] = round(100.0 * (1.0 - (idle1 - idle0) / dt)) if dt > 0 else 0
    except (OSError, ValueError):
        out["cpu_pct"] = None
    try:
        la = os.getloadavg()
        n = os.cpu_count() or 1
        out["load1"] = round(la[0], 2)
        out["load5"] = round(la[1], 2)
        out["load15"] = round(la[2], 2)
        out["n_cpus"] = n
        out["load_ratio"] = round(la[0] / n, 2)
    except (OSError, ValueError):
        pass
    try:
        total_kb, avail_kb = _meminfo()
        used_kb = total_kb - avail_kb
        out["mem_total_gb"] = round(total_kb / 1e6, 1)
        out["mem_used_gb"] = round(used_kb / 1e6, 1)
        out["mem_used_pct"] = round(100.0 * used_kb / total_kb) if total_kb else None
    except (OSError, ValueError):
        pass
    try:
        du = shutil.disk_usage(str(config.data_dir))
        out["disk_total_gb"] = round(du.total / 1e9, 1)
        out["disk_used_gb"] = round(du.used / 1e9, 1)
        out["disk_used_pct"] = round(100.0 * du.used / du.total) if du.total else None
    except OSError:
        pass
    return out


def _cpu_times() -> tuple[float, float]:
    """Return (idle_jiffies, total_jiffies) from the aggregate ``cpu`` line of /proc/stat."""
    with Path("/proc/stat").open(encoding="utf-8") as fh:
        parts = [float(x) for x in fh.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)  # idle + iowait  # noqa: PLR2004
    return idle, sum(parts)


def _meminfo() -> tuple[float, float]:
    """Return (MemTotal_kB, MemAvailable_kB) from /proc/meminfo."""
    total = avail = 0.0
    with Path("/proc/meminfo").open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1])
    return total, avail


def _sweep_log_path(config: Config) -> Path:
    return config.data_dir / _SWEEP_LOG


def sweep_status(config: Config) -> dict[str, object]:
    """Liveness + progress of the background training sweep (the run_full_sweep.py drainer).

    Reads the drainer's own log so /ops shows what it's actually doing — it runs as a standalone
    process, not under the scheduler, so journald alone can't see it. Returns whether the process
    is alive, the triple in flight, how long since the last log line (staleness), the average
    per-triple duration (for ETA), and how many assets it gave up on.
    """
    running = bool((_run(["pgrep", "-f", _SWEEP_PROC]) or "").strip())
    status: dict[str, object] = {
        "running": running,
        "current": None,
        "last_activity": None,
        "idle_seconds": None,
        "avg_seconds": None,
        "gave_up": 0,
    }
    log = _sweep_log_path(config)
    if not log.exists():
        return status
    # The drainer's structured lines are buried under ~300 Optuna per-trial lines per triple, so
    # grep them out directly (a plain tail would scroll them off the window).
    sweep_lines = (_run(["grep", " sweep: ", str(log)]) or "").splitlines()
    durations: list[float] = []
    gave_up = 0
    for line in sweep_lines[-80:]:
        if "] start " in line:
            status["current"] = line.split("] start ", 1)[1].strip()
        if " done " in line and " in " in line and "promoted=" in line:
            with contextlib.suppress(ValueError, IndexError):
                durations.append(float(line.split(" in ", 1)[1].split("s", 1)[0]))
        if "giving up" in line.lower():
            gave_up += 1
    status["gave_up"] = gave_up
    # Liveness: the timestamp of the very last log line (any kind), so a live Optuna trial counts
    # as activity even mid-triple.
    last_line = (_run(["tail", "-n", "1", str(log)]) or "").strip()
    m = _TS_RE.search(last_line)
    last_ts = f"{m.group(1)}T{m.group(2)}" if m else None
    if durations:
        status["avg_seconds"] = round(sum(durations) / len(durations))
    if last_ts:
        status["last_activity"] = last_ts
        try:
            delta = datetime.now() - datetime.fromisoformat(last_ts)  # noqa: DTZ005 — local log clock
            status["idle_seconds"] = max(0, int(delta.total_seconds()))
        except ValueError:
            pass
    return status


def gpus() -> list[dict[str, object]]:
    """Per-GPU utilization + memory + temperature via nvidia-smi (empty if no GPU/tool)."""
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    raw = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not raw:
        return []
    out: list[dict[str, object]] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:  # noqa: PLR2004 — six queried fields
            continue
        idx, name, util, mem_used, mem_total, temp = parts
        try:
            out.append(
                {
                    "index": int(idx),
                    "name": name,
                    "util_pct": int(util),
                    "mem_used_mb": int(mem_used),
                    "mem_total_mb": int(mem_total),
                    "temp_c": int(temp),
                }
            )
        except ValueError:
            continue
    return out


def scheduler_status() -> dict[str, object]:
    """systemd state + start time of the scheduler unit (best-effort)."""
    raw = _run(["systemctl", "show", _SCHEDULER_UNIT, "-p", "SubState,ActiveEnterTimestamp"])
    state = "unknown"
    since: str | None = None
    if raw:
        for line in raw.strip().splitlines():
            key, _, val = line.partition("=")
            if key == "SubState":
                state = val
            elif key == "ActiveEnterTimestamp":
                since = val or None
    return {"unit": _SCHEDULER_UNIT, "state": state, "active_since": since}


def scheduled_jobs(config: Config) -> list[dict[str, object]]:
    """Each scheduled job with its next fire time, computed from its trigger.

    Rebuilds the job table in-process (the real scheduler is another process) and asks each
    trigger for its next fire time — accurate because triggers are pure cron expressions.
    """
    try:
        from berich.scheduler import build_scheduler  # noqa: PLC0415

        scheduler = build_scheduler(config)
    except Exception:  # noqa: BLE001 — never let dashboard assembly fail on scheduler import
        logger.warning("ops: could not build scheduler for job listing", exc_info=True)
        return []
    now = datetime.now(UTC)
    jobs: list[dict[str, object]] = []
    for job in scheduler.get_jobs():
        try:
            nxt = job.trigger.get_next_fire_time(None, now)
        except Exception:  # noqa: BLE001 — a single bad trigger shouldn't drop the rest
            nxt = None
        jobs.append(
            {
                "id": job.id,
                "next_run": nxt.isoformat() if nxt else None,
            }
        )
    jobs.sort(key=lambda j: (j["next_run"] is None, j["next_run"] or ""))
    return jobs


def hpo_progress(config: Config) -> dict[str, object]:
    """First-HPO queue progress at the (ticker, side, exit strategy) grain: done/pending/promoted.

    Each exit strategy (fixed / trailing / trailing_tp) is its own HPO+tournament unit — the same
    granularity the sweep drains — so the bar reflects the real work-list (tickers x sides x
    strategies), not just the legacy ticker-by-side count.
    """
    try:
        from berich.training.status import (  # noqa: PLC0415
            _hpo_trial_counts,
            _hpo_trials_for,
            training_status,
        )

        rows = training_status(config)
        counts = _hpo_trial_counts(config.optuna_db)
    except Exception:  # noqa: BLE001 — degrade to empty rather than break the dashboard
        logger.warning("ops: training_status failed", exc_info=True)
        return {"total": 0, "hpo_done": 0, "pending": 0, "promoted": 0, "advisory": 0, "recent": []}

    # Per-(ticker, side, strategy) status, from each row's strategy slate.
    status_by: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        for s in cast("list[dict]", r.get("strategies", [])):
            status_by[(str(r["ticker"]), str(r["side"]), str(s["strategy"]))] = s

    units = [
        (ticker, side, strategy)
        for ticker in config.tradeable_tickers()
        for side in config.zoo.ticker_sides
        for strategy in config.zoo.ticker_exit_strategies
    ]
    total = len(units)
    done = [u for u in units if _hpo_trials_for(counts, u[0], None, u[1], u[2]) > 0]
    hpo_done = len(done)
    promoted = sum(1 for u in done if status_by.get(u, {}).get("status") == "promoted")
    advisory = sum(1 for u in done if status_by.get(u, {}).get("status") == "advisory_only")

    # Most-recently-trained few (per strategy), for "last finished" context.
    trained = [
        {
            "ticker": t,
            "side": side,
            "strategy": strategy,
            "status": status_by[(t, side, strategy)].get("status"),
            "trained_at": status_by[(t, side, strategy)].get("trained_at"),
            "hpo_trials": _hpo_trials_for(counts, t, None, side, strategy),
        }
        for (t, side, strategy) in done
        if status_by.get((t, side, strategy), {}).get("trained_at")
    ]
    trained.sort(key=lambda r: str(r.get("trained_at")), reverse=True)
    return {
        "total": total,
        "hpo_done": hpo_done,
        "pending": total - hpo_done,
        "promoted": promoted,
        "advisory": advisory,
        "recent": trained[:5],
    }


def _scheduler_log_lines(lines: int) -> list[dict[str, str]]:
    """Recent scheduler journald lines as {time, message, level} (best-effort)."""
    raw = _run(
        ["journalctl", "-u", _SCHEDULER_UNIT, "-n", str(lines), "-o", "short-iso", "--no-pager"]
    )
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for line in raw.strip().splitlines():
        # Each short-iso line is: <iso-timestamp> <host> <unit[pid]>: <level> <logger>: <message>
        ts, _, rest = line.partition(" ")
        msg = (rest.split(": ", 1)[-1] if ": " in rest else rest).strip()
        # Keep only application log lines (they carry a Python level token); drop systemd
        # lifecycle noise (Started/Stopped/Failed/Consumed) so an intentional stop isn't an "error".
        if not any(f"{lvl}" in msg for lvl in ("INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG")):
            continue
        m = _TS_RE.search(ts)
        norm = f"{m.group(1)}T{m.group(2)}" if m else ts
        out.append({"time": norm, "message": msg, "level": _log_level(msg), "source": "scheduler"})
    return out


def _sweep_log_lines(config: Config, lines: int) -> list[dict[str, str]]:
    """Recent sweep-drainer log lines (its own structured ones, skipping Optuna per-trial spam)."""
    log = _sweep_log_path(config)
    if not log.exists():
        return []
    raw = _run(["tail", "-n", "400", str(log)]) or ""
    out: list[dict[str, str]] = []
    for line in raw.splitlines():
        # Keep the drainer's own "sweep:" lines + any error/traceback; drop Optuna "[I ...] Trial".
        is_sweep = " sweep:" in line
        level = _log_level(line)
        if not is_sweep and level == "info":
            continue
        m = _TS_RE.search(line)
        if not m:
            continue
        msg = line.split(" sweep: ", 1)[-1].strip() if is_sweep else line.strip()
        ts = f"{m.group(1)}T{m.group(2)}"
        out.append(
            {"time": ts, "message": msg, "level": level, "source": "sweep"}
        )
    return out[-lines:]


def recent_logs(config: Config, lines: int = 24) -> list[dict[str, str]]:
    """Merged recent activity from the scheduler and the sweep drainer, oldest→newest by time."""
    merged = _scheduler_log_lines(lines) + _sweep_log_lines(config, lines)
    merged.sort(key=lambda r: r["time"])
    return merged[-lines:]


def ops_snapshot(config: Config) -> dict[str, object]:
    """Full machine-status payload for the /ops dashboard (all collectors, best-effort)."""
    logs = recent_logs(config)
    alerts = [line_log for line_log in logs if line_log["level"] in ("error", "warning")]
    return {
        "gpus": gpus(),
        "system": system_metrics(config),
        "sweep": sweep_status(config),
        "scheduler": scheduler_status(),
        "jobs": scheduled_jobs(config),
        "hpo": hpo_progress(config),
        "alerts": alerts[-6:],
        "logs": logs,
    }


__all__ = ["ops_snapshot"]
