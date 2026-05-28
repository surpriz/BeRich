"""APScheduler wiring for local, single-user automation.

`build_scheduler` returns a configured (but unstarted) BlockingScheduler so the CLI
can start it and tests can introspect the registered jobs without blocking. Daily
refresh+signals run after the US market close; the drift check runs weekly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from berich.scheduler.jobs import check_drift_job, daily_paper_job

if TYPE_CHECKING:
    from berich.config import Config


def build_scheduler(config: Config) -> BlockingScheduler:
    """Create a scheduler with the daily-signal and weekly-drift jobs registered."""
    scheduler = BlockingScheduler(timezone="America/New_York")
    # 17:30 ET on weekdays — comfortably after the 16:00 close and data settle.
    # The daily job chains: refresh OHLCV → generate signals → roll paper book.
    scheduler.add_job(
        daily_paper_job,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=30),
        args=[config],
        id="daily_paper",
        replace_existing=True,
    )
    # Weekly drift review, Saturday morning.
    scheduler.add_job(
        check_drift_job,
        CronTrigger(day_of_week="sat", hour=8, minute=0),
        args=[config],
        id="check_drift",
        replace_existing=True,
    )
    return scheduler
