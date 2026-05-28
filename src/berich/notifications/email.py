"""Plain-text + HTML email notifications via SMTP (STARTTLS).

Config is read from environment variables — same place ``BERICH_API_KEY``
already lives (``/etc/berich/env`` on the production server). All four of
``NOTIFY_EMAIL``, ``SMTP_HOST``, ``SMTP_USER`` and ``SMTP_PASS`` must be
set for the email to be sent; if any is missing the helper returns ``False``
silently so the scheduler keeps running on machines that don't have email
wired up. We never log the credentials.

The only message currently sent is a "new BUY signal" digest: one HTML
table with ticker, proba, entry / stop / target, size. Empty BUY lists
produce no email at all.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from berich.signals.service import Signal

logger = logging.getLogger(__name__)

DEFAULT_SMTP_PORT = 587  # STARTTLS port — Gmail, Outlook, Fastmail all use it
DEFAULT_SUBJECT = "BeRich — new BUY signal(s)"


@dataclass(frozen=True)
class EmailConfig:
    """All the SMTP settings, populated from ``os.environ`` by :meth:`from_env`."""

    notify_email: str
    smtp_host: str
    smtp_user: str
    smtp_pass: str
    smtp_port: int = DEFAULT_SMTP_PORT

    @classmethod
    def from_env(cls) -> EmailConfig | None:
        """Return a config if all four required env vars are present, else None."""
        notify = os.environ.get("NOTIFY_EMAIL")
        host = os.environ.get("SMTP_HOST")
        user = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASS")
        if not all([notify, host, user, password]):
            return None
        port = int(os.environ.get("SMTP_PORT", DEFAULT_SMTP_PORT))
        # mypy/ty narrow above thanks to the all([...]) but they don't pick it
        # up — separate asserts keep callers happy without runtime cost.
        assert notify  # noqa: S101
        assert host  # noqa: S101
        assert user  # noqa: S101
        assert password  # noqa: S101
        return cls(
            notify_email=notify,
            smtp_host=host,
            smtp_user=user,
            smtp_pass=password,
            smtp_port=port,
        )


def _format_html_table(signals: list[Signal]) -> str:
    """Compact HTML table for the email body — one row per BUY signal."""
    rows = "\n".join(
        f"<tr>"
        f"<td>{s.ticker}</td>"
        f"<td>{s.proba:.3f}</td>"
        f"<td>{s.entry:.2f}</td>"
        f"<td>{s.stop_loss:.2f}</td>"
        f"<td>{s.take_profit:.2f}</td>"
        f"<td>{s.size_shares}</td>"
        f"</tr>"
        for s in signals
    )
    return f"""\
<html><body>
<p>BeRich emitted {len(signals)} new BUY signal(s):</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
  <thead><tr>
    <th>Ticker</th><th>P(win)</th><th>Entry</th><th>Stop</th><th>Target</th><th>Shares</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
<p style="color: #666; font-size: 12px; margin-top: 16px;">
  Advisory only — the current model does not beat buy &amp; hold. Not financial advice.
</p>
</body></html>"""


def _format_text_table(signals: list[Signal]) -> str:
    """Plain-text fallback for mail clients that ignore the HTML alternative."""
    header = f"{'Ticker':<8}{'P(win)':>8}{'Entry':>10}{'Stop':>10}{'Target':>10}{'Shares':>8}"
    body = "\n".join(
        f"{s.ticker:<8}{s.proba:>8.3f}{s.entry:>10.2f}"
        f"{s.stop_loss:>10.2f}{s.take_profit:>10.2f}{s.size_shares:>8d}"
        for s in signals
    )
    return f"BeRich — {len(signals)} new BUY signal(s):\n\n{header}\n{body}\n\nAdvisory only."


def send_buy_signals_email(
    signals: Iterable[Signal],
    config: EmailConfig | None = None,
    *,
    subject: str = DEFAULT_SUBJECT,
    smtp_factory=None,  # noqa: ANN001 — small DI hook used only in tests
) -> bool:
    """Send a single email digest for the given BUY signals.

    Returns ``True`` if an email was actually sent, ``False`` otherwise
    (no signals, no config, or SMTP error). All SMTP exceptions are
    caught and logged — the scheduler must keep running even if mail is
    momentarily broken.
    """
    signals = [s for s in signals if s.signal == "BUY"]
    if not signals:
        return False
    cfg = config or EmailConfig.from_env()
    if cfg is None:
        logger.info("email: NOTIFY_EMAIL/SMTP_* not set, skipping notification")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = cfg.smtp_user
    message["To"] = cfg.notify_email
    message.set_content(_format_text_table(signals))
    message.add_alternative(_format_html_table(signals), subtype="html")

    try:
        with (smtp_factory or smtplib.SMTP)(cfg.smtp_host, cfg.smtp_port) as smtp:
            smtp.starttls()
            smtp.login(cfg.smtp_user, cfg.smtp_pass)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("email: SMTP send failed (%s); not blocking the job", exc)
        return False
    logger.info("email: sent BUY-signal digest to %s (%d signals)", cfg.notify_email, len(signals))
    return True
