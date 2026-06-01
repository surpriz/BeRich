"""APScheduler wiring for local, single-user automation.

`build_scheduler` returns a configured (but unstarted) BlockingScheduler so the CLI
can start it and tests can introspect the registered jobs without blocking. Daily
refresh+signals run after the US market close; the drift check runs weekly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from berich.scheduler.jobs import (
    check_drift_job,
    daily_paper_job,
    longshort_signals_job,
    nightly_hpo_job,
    refresh_universe_job,
    ticker_initial_sweep_job,
    ticker_nightly_refresh_job,
    weekend_hpo_job,
)

if TYPE_CHECKING:
    from berich.config import Config


def build_scheduler(config: Config) -> BlockingScheduler:
    """Create a scheduler with the daily-signal and weekly-drift jobs registered."""
    scheduler = BlockingScheduler(timezone="America/New_York")
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
    return scheduler
