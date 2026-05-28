"""Notification side-effects (email today; potentially Slack / Discord later)."""

from berich.notifications.email import EmailConfig, send_buy_signals_email

__all__ = ["EmailConfig", "send_buy_signals_email"]
