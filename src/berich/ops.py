"""Live machine-status collectors for the ``/ops`` dashboard.

The scheduler, API, and frontend run as separate processes, so the API can't read the
scheduler's in-memory state directly. Instead this module reads *observable* signals the box
already exposes: GPU usage via ``nvidia-smi``, service state + recent logs via systemd/journald,
the scheduler's next-fire times by rebuilding the job table, and the per-asset HPO queue
progress from the training-status scan. Every collector is best-effort and degrades to an
empty/neutral payload rather than raising, so one missing tool never breaks the dashboard.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from berich.config import Config

logger = logging.getLogger(__name__)

_SCHEDULER_UNIT = "berich-scheduler"
_TIMEOUT = 5  # seconds — never let a hung subprocess stall the dashboard


def _run(cmd: list[str]) -> str | None:
    """Run a read-only command, returning stdout or None on any failure/timeout."""
    try:
        out = subprocess.run(  # noqa: S603 — fixed arg lists, no shell, no user input
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


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
    """First-HPO queue progress: done / pending / promoted, derived from training status."""
    try:
        from berich.training.status import training_status  # noqa: PLC0415

        rows = training_status(config)
    except Exception:  # noqa: BLE001 — degrade to empty rather than break the dashboard
        logger.warning("ops: training_status failed", exc_info=True)
        return {"total": 0, "hpo_done": 0, "pending": 0, "promoted": 0, "advisory": 0}
    total = len(rows)
    # The bar is about the per-asset HPO queue, so promoted/advisory are counted *within* the
    # HPO-done set — never mixing in the legacy assets promoted before HPO existed (which would
    # make "promoted" and "done" look contradictory). hpo_done = (ticker, side) pairs searched.
    done_rows = [r for r in rows if cast("int", r.get("hpo_trials", 0)) > 0]
    hpo_done = len(done_rows)
    promoted = sum(1 for r in done_rows if r.get("status") == "promoted")
    advisory = sum(1 for r in done_rows if r.get("status") == "advisory_only")
    # Most-recently-trained few, for "last finished" context.
    trained = [r for r in rows if r.get("trained_at")]
    trained.sort(key=lambda r: str(r.get("trained_at")), reverse=True)
    recent = [
        {
            "ticker": r["ticker"],
            "side": r["side"],
            "status": r["status"],
            "trained_at": r["trained_at"],
            "hpo_trials": r["hpo_trials"],
        }
        for r in trained[:5]
    ]
    return {
        "total": total,
        "hpo_done": hpo_done,
        "pending": total - hpo_done,
        "promoted": promoted,
        "advisory": advisory,
        "recent": recent,
    }


def recent_logs(lines: int = 20) -> list[dict[str, str]]:
    """Recent scheduler log lines (timestamp + message), newest last (best-effort)."""
    raw = _run(
        ["journalctl", "-u", _SCHEDULER_UNIT, "-n", str(lines), "-o", "short-iso", "--no-pager"]
    )
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for line in raw.strip().splitlines():
        # Each short-iso line is: <iso-timestamp> <host> <unit[pid]>: <level> <logger>: <message>
        ts, _, rest = line.partition(" ")
        msg = rest.split(": ", 1)[-1] if ": " in rest else rest
        out.append({"time": ts, "message": msg.strip()})
    return out


def ops_snapshot(config: Config) -> dict[str, object]:
    """Full machine-status payload for the /ops dashboard (all collectors, best-effort)."""
    return {
        "gpus": gpus(),
        "scheduler": scheduler_status(),
        "jobs": scheduled_jobs(config),
        "hpo": hpo_progress(config),
        "logs": recent_logs(),
    }


__all__ = ["ops_snapshot"]
