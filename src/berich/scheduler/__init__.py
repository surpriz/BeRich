"""Local automation: daily refresh + signals + paper-book update, weekly drift."""

from berich.scheduler.jobs import check_drift_job, daily_paper_job
from berich.scheduler.runner import build_scheduler

__all__ = ["build_scheduler", "check_drift_job", "daily_paper_job"]
