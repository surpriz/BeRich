"""Local automation: daily data refresh + signals, periodic drift checks."""

from berich.scheduler.jobs import check_drift_job, refresh_and_signal_job
from berich.scheduler.runner import build_scheduler

__all__ = ["build_scheduler", "check_drift_job", "refresh_and_signal_job"]
