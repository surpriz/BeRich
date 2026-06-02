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
