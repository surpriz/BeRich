"""Watchdog: verify the 22:30 daily paper run actually happened, email an alert otherwise.

The scheduler is a single process — if it crashes before 22:30, no job runs and (crucially)
no EVENT_JOB_ERROR alert fires either, because there was no job to fail. This script runs from
an independent systemd timer (23:15 Paris, weekdays; see docs/DEPLOY.md) and checks the
heartbeat ``data/last_daily_run.json`` that ``daily_paper_job`` stamps on completion.

Exit code 0 = run found (or non-trading day); 1 = missing run (alert email attempted).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("berich.watchdog")

# The heartbeat must be younger than this when the watchdog fires at 23:15 — the run is
# scheduled at 22:30 the same evening, so anything older means tonight's run never happened.
MAX_AGE_HOURS = 2.0


def main() -> int:
    from berich.config import Config
    from berich.notifications.email import send_alert_email

    config = Config.load()
    heartbeat = config.data_dir / "last_daily_run.json"

    now = datetime.now(UTC)
    if now.weekday() >= 5:  # noqa: PLR2004 — Sat/Sun: no daily run scheduled
        logger.info("weekend — no daily run expected")
        return 0

    age_hours: float | None = None
    detail = "heartbeat file missing"
    if heartbeat.exists():
        try:
            stamp = json.loads(heartbeat.read_text()).get("at")
            ran_at = datetime.fromisoformat(str(stamp))
            age_hours = (now - ran_at).total_seconds() / 3600.0
            detail = f"last run {ran_at.isoformat()} ({age_hours:.1f}h ago)"
        except (ValueError, TypeError, json.JSONDecodeError):
            detail = "heartbeat file unreadable"

    if age_hours is not None and age_hours <= MAX_AGE_HOURS:
        logger.info("daily run OK — %s", detail)
        return 0

    logger.error("daily run MISSING — %s", detail)
    try:
        send_alert_email(
            subject="BeRich ALERTE : le run quotidien de 22:30 n'a pas eu lieu",
            body=(
                f"Le watchdog n'a pas trouvé de run quotidien récent ({detail}).\n"
                "Vérifier : systemctl status berich-scheduler "
                "&& journalctl -u berich-scheduler -n 100\n\n"
                f"The watchdog found no recent daily run ({detail}).\n"
                "Check the berich-scheduler unit and its journal."
            ),
        )
    except Exception:  # the non-zero exit still surfaces in journald
        logger.exception("alert email failed")
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.exit(main())
