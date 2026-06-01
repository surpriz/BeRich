"""APScheduler wiring for local, single-user automation.

`build_scheduler` returns a configured (but unstarted) BlockingScheduler so the CLI
can start it and tests can introspect the registered jobs without blocking. Daily
refresh+signals run after the US market close; the drift check runs weekly.
"""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from berich.scheduler.jobs import (
    backup_job,
    check_drift_job,
    daily_paper_job,
    longshort_signals_job,
    nightly_hpo_job,
    refresh_universe_job,
    ticker_hpo_queue_job,
    ticker_initial_sweep_job,
    ticker_nightly_refresh_job,
    weekend_hpo_job,
)

if TYPE_CHECKING:
    from apscheduler.events import JobExecutionEvent

    from berich.config import Config

logger = logging.getLogger(__name__)


def _on_job_error(event: JobExecutionEvent) -> None:
    """APScheduler EVENT_JOB_ERROR handler: email an alert when any job raises.

    Fires only on an *uncaught* exception from a job (jobs that handle their own errors and
    return normally never trigger this). Best-effort: a failure to send the alert is logged
    and swallowed so the listener never crashes the scheduler.
    """
    job_id = event.job_id
    exc = event.exception
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else ""
    logger.error("scheduler job '%s' raised: %s", job_id, exc)
    try:
        from berich.notifications import send_alert_email  # noqa: PLC0415 — lazy, optional dep

        send_alert_email(
            subject=f"BeRich — job '{job_id}' failed",
            body=(
                f"The scheduled job '{job_id}' raised an exception at "
                f"{event.scheduled_run_time}.\n\n{type(exc).__name__}: {exc}\n\n{tb}\n"
                "The scheduler keeps running; other jobs are unaffected."
            ),
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the scheduler
        logger.warning("scheduler: could not send failure alert for '%s'", job_id, exc_info=True)


def build_scheduler(config: Config) -> BlockingScheduler:
    """Create a scheduler with the daily-signal and weekly-drift jobs registered."""
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)
    # 22:30 Europe/Paris on weekdays — 30 min after the 16:00 ET close (22:00 Paris
    # year-round, since both zones observe DST), so OHLCV bars are settled daily
    # closes, not intraday snapshots. The trigger timezone is pinned explicitly
    # because the deployed scheduler does not honor the BlockingScheduler default.
    # The daily job chains: refresh OHLCV → generate signals → roll paper book.
    scheduler.add_job(
        daily_paper_job,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone="Europe/Paris"),
        args=[config],
        id="daily_paper",
        replace_existing=True,
    )
    # Wider long/short universe refresh at 22:45 Paris weekdays — after daily_paper (22:30),
    # before the nightly retrain (23:30) so the cross-sectional panel trains on fresh bars.
    scheduler.add_job(
        refresh_universe_job,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=45, timezone="Europe/Paris"),
        args=[config],
        id="refresh_universe",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Light nightly HPO at 23:00 Paris weekdays — a few trials into the shared study, so the
    # best params keep improving and feed the 23:30 retrain (the weekend job does the deep sweep).
    scheduler.add_job(
        nightly_hpo_job,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=0, timezone="Europe/Paris"),
        args=[config],
        id="nightly_hpo",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Per-ticker nightly refresh at 23:30 Paris on weekdays — after daily_paper (22:30) has
    # refreshed OHLCV, so the light HPO top-up + re-tournament run on fresh data. Only tickers
    # that already have a promoted model are touched; the guard rule still gates any promotion.
    # max_instances/coalesce keep a long run from stacking.
    scheduler.add_job(
        ticker_nightly_refresh_job,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=30, timezone="Europe/Paris"),
        args=[config],
        id="ticker_nightly_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Long/short basket generation at 23:45 Paris weekdays — after the nightly retrain,
    # so the freshest promoted ranker drives today's paper basket.
    scheduler.add_job(
        longshort_signals_job,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=45, timezone="Europe/Paris"),
        args=[config],
        id="longshort_signals",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Weekly drift review, Saturday morning.
    scheduler.add_job(
        check_drift_job,
        CronTrigger(day_of_week="sat", hour=8, minute=0),
        args=[config],
        id="check_drift",
        replace_existing=True,
    )
    # Weekend HPO to rentabilize the rented GPUs — long Optuna search, Saturday midday.
    scheduler.add_job(
        weekend_hpo_job,
        CronTrigger(day_of_week="sat", hour=12, minute=0, timezone="Europe/Paris"),
        args=[config],
        id="weekend_hpo",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Heavy per-ticker cold-start sweep, Saturday 14:00 Paris — after the weekend HPO (12:00).
    # Full per-ticker HPO + tournament for every tradeable ticker x side; this is the cold
    # start the weekday nightly_refresh later tops up.
    scheduler.add_job(
        ticker_initial_sweep_job,
        CronTrigger(day_of_week="sat", hour=14, minute=0, timezone="Europe/Paris"),
        args=[config],
        id="ticker_initial_sweep",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Sequential first-HPO queue: every 2h, optimize the NEXT un-searched ticker x side, one at
    # a time. max_instances=1 means a still-running asset simply blocks the next firing, so two
    # deep HPO searches never overlap — the gentle, GPU-friendly path to give every asset its
    # first deep HPO without the Saturday thundering herd. Drops to a no-op once all are done.
    scheduler.add_job(
        ticker_hpo_queue_job,
        CronTrigger(hour="*/2", minute=15, timezone="Europe/Paris"),
        args=[config],
        id="ticker_hpo_queue",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # Daily rotating backup of the training state at 21:00 Paris — before the nightly chain
    # (22:30+) mutates models/signals, so each archive snapshots the last settled state.
    scheduler.add_job(
        backup_job,
        CronTrigger(hour=21, minute=0, timezone="Europe/Paris"),
        args=[config],
        id="backup",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler
