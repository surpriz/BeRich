"""Notification side-effects (email today; potentially Slack / Discord later)."""

from berich.notifications.digest import DailyDigest, build_daily_digest
from berich.notifications.email import (
    EmailConfig,
    send_alert_email,
    send_daily_digest_email,
)

__all__ = [
    "DailyDigest",
    "EmailConfig",
    "build_daily_digest",
    "send_alert_email",
    "send_daily_digest_email",
]
