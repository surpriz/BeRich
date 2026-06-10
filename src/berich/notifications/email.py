"""Plain-text + HTML email notifications via SMTP (STARTTLS).

Config is read from environment variables — same place ``BERICH_API_KEY``
already lives (``/etc/berich/env`` on the production server). All four of
``NOTIFY_EMAIL``, ``SMTP_HOST``, ``SMTP_USER`` and ``SMTP_PASS`` must be
set for the email to be sent; if any is missing the helper returns ``False``
silently so the scheduler keeps running on machines that don't have email
wired up. We never log the credentials.

Two messages are sent: a **daily digest** (:func:`send_daily_digest_email` — the
bilingual FR/EN morning briefing rendered in the "Clarté" palette, fired every
weekday run) and a plain-text **operational alert** (:func:`send_alert_email` —
job crashes, stale data). The digest payload is assembled in ``digest.py``; this
module only formats and sends it.
"""

from __future__ import annotations

import logging
import math
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from berich.notifications.digest import DailyDigest

logger = logging.getLogger(__name__)

DEFAULT_SMTP_PORT = 587  # STARTTLS port — Gmail, Outlook, Fastmail all use it

# "Clarté" palette (kept in sync with frontend/app/globals.css). Web fonts won't load in most
# mail clients, so the renderer falls back to system fonts; the colors are safe inline styles.
_INK = "#14181f"
_MUTED = "#5b6472"
_FAINT = "#97a0b0"
_LINE = "#e6e8ef"
_ACCENT = "#4f46e5"  # indigo brand
_BULL = "#10936b"  # emerald — positive
_BEAR = "#e05252"  # coral — negative
_BG = "#f4f5f8"
_SURFACE = "#ffffff"
_FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"

_FR_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
# closed-trade status -> (FR, EN) human label for the "how it exited" tag.
_EXIT_LABEL = {
    "closed_target": ("cible", "target"),
    "closed_stop": ("stop", "stop"),
    "closed_time": ("échéance", "time"),
    "closed_trail": ("trailing", "trailing"),
}


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


# ---------------------------------------------------------------- formatting helpers ----


def _fr_date(iso: str) -> str:
    """ISO ``YYYY-MM-DD`` -> ``8 juin 2026`` for the header."""
    d = pd.Timestamp(iso)
    return f"{d.day} {_FR_MONTHS[int(d.month) - 1]} {d.year}"


def _fmt_pct(value: float, *, signed: bool = True) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def _fmt_eur(value: float) -> str:
    """Thousands-spaced euros, e.g. ``10 240 €`` (narrow no-break space)."""
    return f"{value:,.0f} €".replace(",", " ")


def _fmt_price(value: float) -> str:
    """Compact price: drop trailing zeros but keep precision for sub-1 FX/crypto rates."""
    return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.2f}"  # noqa: PLR2004


def _pnl_color(value: float) -> str:
    return _BULL if value >= 0 else _BEAR


def _dir_badge(direction: str) -> str:
    is_long = direction == "long"
    label = "Long" if is_long else "Short"
    arrow = "↑" if is_long else "↓"
    color = _BULL if is_long else _BEAR
    return f'<span style="color:{color};font-weight:600;white-space:nowrap;">{label} {arrow}</span>'


# ---------------------------------------------------------------- HTML digest ----


def _section(title_fr: str, title_en: str, inner: str) -> str:
    """A titled block: bilingual small-caps header + body, separated by a hairline rule."""
    return f"""\
<tr><td style="padding:22px 28px 0 28px;">
  <div style="font:600 11px/1.4 {_FONT};letter-spacing:.08em;text-transform:uppercase;color:{_ACCENT};">
    {title_fr} <span style="color:{_FAINT};">· {title_en}</span>
  </div>
  <div style="height:1px;background:{_LINE};margin:8px 0 14px 0;"></div>
  {inner}
</td></tr>"""


def _portfolio_html(d: DailyDigest) -> str:
    ret_c = _pnl_color(d.total_return_paper)
    # SPY return is NaN when the benchmark series isn't available yet (empty/young book).
    spy = "" if math.isnan(d.total_return_spy) else _fmt_pct(d.total_return_spy)
    spy_txt = f' <span style="color:{_FAINT};">· SPY {spy}</span>' if spy else ""
    dd_txt = _fmt_pct(-d.current_drawdown)
    return f"""\
<div style="font:700 30px/1.1 {_MONO};color:{_INK};">{_fmt_eur(d.equity)}</div>
<div style="font:600 15px/1.6 {_FONT};color:{ret_c};">{_fmt_pct(d.total_return_paper)}{spy_txt}</div>
<div style="font:400 13px/1.7 {_FONT};color:{_MUTED};margin-top:6px;">
  Drawdown <span style="color:{_pnl_color(-d.current_drawdown)};font-weight:600;">{dd_txt}</span>
  &nbsp;·&nbsp; {d.n_open} position(s) ouverte(s) / open
  &nbsp;·&nbsp; {d.n_closed_total} clôturée(s) / closed
  &nbsp;·&nbsp; <span title="taux de réussite / win rate">{_fmt_pct(d.win_rate, signed=False)} win</span>
</div>
<div style="font:400 12px/1.6 {_FONT};color:{_FAINT};margin-top:8px;">
  Capital de base / base capital {_fmt_eur(d.capital)} — paper-trading, forward test.
</div>"""


def _orders_html(d: DailyDigest) -> str:
    if not d.opened:
        return (
            f'<div style="font:400 13px/1.6 {_FONT};color:{_MUTED};">'
            "Aucun nouvel ordre ce soir. / No new orders tonight.</div>"
        )
    head = (
        f'<tr style="font:600 11px/1.4 {_FONT};color:{_FAINT};text-align:left;">'
        '<th style="padding:4px 10px 4px 0;">Ticker</th>'
        '<th style="padding:4px 10px;">Sens / Dir</th>'
        f'<th style="padding:4px 10px;text-align:right;">Entrée / Entry</th>'
        '<th style="padding:4px 10px;text-align:right;">Stop</th>'
        '<th style="padding:4px 10px;text-align:right;">Cible / Target</th>'
        '<th style="padding:4px 0 4px 10px;text-align:right;">Taille / Size</th></tr>'
    )
    rows = "".join(
        f'<tr style="font:400 13px/1.5 {_MONO};color:{_INK};border-top:1px solid {_LINE};">'
        f'<td style="padding:7px 10px 7px 0;font-weight:600;">{o["ticker"]}</td>'
        f'<td style="padding:7px 10px;font-family:{_FONT};">{_dir_badge(str(o["direction"]))}</td>'
        f'<td style="padding:7px 10px;text-align:right;">{_fmt_price(float(o["entry"]))}</td>'
        f'<td style="padding:7px 10px;text-align:right;color:{_BEAR};">{_fmt_price(float(o["stop"]))}</td>'
        f'<td style="padding:7px 10px;text-align:right;color:{_BULL};">{_fmt_price(float(o["target"]))}</td>'
        f'<td style="padding:7px 0 7px 10px;text-align:right;">{int(o["size_shares"])}</td></tr>'
        for o in d.opened
    )
    adjust = ""
    if d.adjust:
        items = ", ".join(
            f"{a['ticker']} → {_fmt_price(float(a['effective_stop']))}" for a in d.adjust
        )
        adjust = (
            f'<div style="font:400 12px/1.6 {_FONT};color:{_MUTED};margin-top:10px;">'
            f"Stops trailing à recopier / trailing stops to mirror: "
            f'<span style="font-family:{_MONO};color:{_INK};">{items}</span></div>'
        )
    return (
        f'<table cellspacing="0" cellpadding="0" style="width:100%;border-collapse:collapse;">'
        f"{head}{rows}</table>{adjust}"
    )


def _activity_html(d: DailyDigest) -> str:
    if not d.closed:
        opened_note = (
            f" {len(d.opened)} ouverture(s) / opened."
            if d.opened
            else " Aucun mouvement / no moves."
        )
        return (
            f'<div style="font:400 13px/1.6 {_FONT};color:{_MUTED};">'
            f"Aucune clôture aujourd'hui. / No closes today.{opened_note}</div>"
        )
    lines = ""
    for c in d.closed:
        pnl = c.get("pnl_pct")
        pnl_html = (
            f'<span style="color:{_pnl_color(float(pnl))};font-weight:600;">{_fmt_pct(float(pnl))}</span>'
            if pnl is not None
            else f'<span style="color:{_FAINT};">—</span>'
        )
        fr, en = _EXIT_LABEL.get(str(c.get("status")), ("sortie", "exit"))
        lines += (
            f'<div style="font:400 13px/1.9 {_FONT};color:{_INK};">'
            f'<span style="font-family:{_MONO};font-weight:600;">{c["ticker"]}</span> '
            f"{_dir_badge(str(c['direction']))} &nbsp;{pnl_html} "
            f'<span style="color:{_FAINT};">({fr} / {en})</span></div>'
        )
    opened_note = (
        f'<div style="font:400 12px/1.6 {_FONT};color:{_MUTED};margin-top:6px;">'
        f"+ {len(d.opened)} ouverture(s) ce soir / opened tonight.</div>"
        if d.opened
        else ""
    )
    return lines + opened_note


def _good_to_know_html(d: DailyDigest) -> str:
    blocks: list[str] = []
    if d.open_positions:
        chips = ""
        for p in d.open_positions[:8]:
            mtm = float(p["mtm_pct"])
            chips += (
                f'<span style="display:inline-block;font:400 12px/1.6 {_FONT};color:{_INK};'
                f"background:{_BG};border:1px solid {_LINE};border-radius:8px;"
                f'padding:3px 9px;margin:0 6px 6px 0;">'
                f'<span style="font-family:{_MONO};font-weight:600;">{p["ticker"]}</span> '
                f'<span style="color:{_pnl_color(mtm)};">{_fmt_pct(mtm)}</span> '
                f'<span style="color:{_FAINT};">{int(p["days_held"])}j</span></span>'
            )
        more = (
            f' <span style="color:{_FAINT};font:400 12px {_FONT};">+{len(d.open_positions) - 8}…</span>'
            if len(d.open_positions) > 8  # noqa: PLR2004
            else ""
        )
        blocks.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font:600 12px/1.6 {_FONT};color:{_MUTED};margin-bottom:5px;">'
            f"Positions ouvertes (MTM) / open positions</div>{chips}{more}</div>"
        )
    if d.stale_tickers:
        blocks.append(
            f'<div style="font:600 13px/1.6 {_FONT};color:{_BEAR};margin-bottom:8px;">'
            f"⚠ Données possiblement périmées / possibly stale data: "
            f'<span style="font-family:{_MONO};">{", ".join(d.stale_tickers)}</span></div>'
        )
    if d.concentrated_currencies:
        ccy_txt = ", ".join(
            f"{c['currency']} {float(c['pct_capital']) * 100:.0f}% ({int(c['n_positions'])} pos.)"
            for c in d.concentrated_currencies
        )
        blocks.append(
            f'<div style="font:600 13px/1.6 {_FONT};color:{_BEAR};margin-bottom:8px;">'
            f"⚠ Concentration devise (plusieurs positions = un seul pari) / "
            f"currency concentration (several positions = one bet): "
            f'<span style="font-family:{_MONO};">{ccy_txt}</span></div>'
        )
    blocks.append(
        f'<div style="font:400 13px/1.8 {_FONT};color:{_MUTED};">'
        f'<span style="color:{_INK};font-weight:600;">{d.n_promoted_models}</span> modèles promus '
        f"sur {d.n_promoted_tickers} actifs / promoted models on assets &nbsp;·&nbsp; "
        f'<span style="color:{_INK};font-weight:600;">{d.signals_total}</span> signaux générés '
        f"({d.longs_total} longs / {d.shorts_total} shorts) / signals generated.</div>"
    )
    blocks.append(
        f'<div style="font:400 12px/1.7 {_FONT};color:{_FAINT};margin-top:10px;">'
        "Rappel : seuls les « ordres à copier » ci-dessus ont été réellement exécutés. "
        "Le reste du robot (page Brief) est une prévision recalculée en continu.<br>"
        "Reminder: only the “orders to copy” above were actually executed — the Brief page is a "
        "continuously recomputed forecast.</div>"
    )
    return "".join(blocks)


def _render_digest_html(d: DailyDigest) -> str:
    body = (
        _section("Portefeuille", "Portfolio", _portfolio_html(d))
        + _section(f"Ordres à copier ({len(d.opened)})", "Orders to copy", _orders_html(d))
        + _section("Activité du jour", "Today's activity", _activity_html(d))
        + _section("À savoir", "Good to know", _good_to_know_html(d))
    )
    return f"""\
<!doctype html><html><body style="margin:0;padding:0;background:{_BG};">
<table role="presentation" cellspacing="0" cellpadding="0" width="100%" style="background:{_BG};">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" cellspacing="0" cellpadding="0" width="600" style="max-width:600px;width:100%;
  background:{_SURFACE};border:1px solid {_LINE};border-radius:16px;overflow:hidden;">
<tr><td style="background:{_ACCENT};padding:20px 28px;">
  <div style="font:700 19px/1.2 {_FONT};color:#ffffff;">BeRich</div>
  <div style="font:400 13px/1.4 {_FONT};color:#dcd9fb;">
    Digest du {_fr_date(d.date)} · Daily digest
  </div>
</td></tr>
{body}
<tr><td style="padding:20px 28px 24px 28px;">
  <div style="height:1px;background:{_LINE};margin-bottom:12px;"></div>
  <div style="font:400 11px/1.6 {_FONT};color:{_FAINT};">
    Paper-trading en forward test — aucun capital réel engagé. Ne constitue pas un conseil financier.<br>
    Paper-trading forward test — no real money committed. Not financial advice.
  </div>
</td></tr>
</table></td></tr></table></body></html>"""


# ---------------------------------------------------------------- plain-text digest ----


def _render_digest_text(d: DailyDigest) -> str:
    lines = [
        f"BeRich — digest du {_fr_date(d.date)} / daily digest",
        "",
        "PORTEFEUILLE / PORTFOLIO",
        f"  Équité / equity : {_fmt_eur(d.equity)}  ({_fmt_pct(d.total_return_paper)})",
        f"  Drawdown {_fmt_pct(-d.current_drawdown)} · {d.n_open} ouvertes / open"
        f" · {d.n_closed_total} clôturées / closed · {_fmt_pct(d.win_rate, signed=False)} win",
        "",
        f"ORDRES À COPIER / ORDERS TO COPY ({len(d.opened)})",
    ]
    if d.opened:
        lines += [
            f"  {o['ticker']:<10} {o['direction']:<5} entry {_fmt_price(float(o['entry']))}"
            f"  stop {_fmt_price(float(o['stop']))}  target {_fmt_price(float(o['target']))}"
            f"  x{int(o['size_shares'])}"
            for o in d.opened
        ]
    else:
        lines.append("  Aucun nouvel ordre. / No new orders.")
    if d.adjust:
        lines.append(
            "  Trailing à recopier / mirror: "
            + ", ".join(
                f"{a['ticker']}->{_fmt_price(float(a['effective_stop']))}" for a in d.adjust
            )
        )
    lines += ["", "ACTIVITÉ DU JOUR / TODAY'S ACTIVITY"]
    if d.closed:
        for c in d.closed:
            pnl = c.get("pnl_pct")
            pnl_s = _fmt_pct(float(pnl)) if pnl is not None else "—"
            fr, en = _EXIT_LABEL.get(str(c.get("status")), ("sortie", "exit"))
            lines.append(f"  {c['ticker']:<10} {c['direction']:<5} {pnl_s:>7}  ({fr}/{en})")
    else:
        lines.append("  Aucune clôture. / No closes.")
    lines += [
        "",
        "À SAVOIR / GOOD TO KNOW",
        f"  {d.n_promoted_models} modèles promus / promoted models · {d.n_promoted_tickers} actifs",
        f"  {d.signals_total} signaux ({d.longs_total} longs / {d.shorts_total} shorts)",
    ]
    if d.stale_tickers:
        lines.append(f"  ⚠ données périmées / stale data: {', '.join(d.stale_tickers)}")
    if d.concentrated_currencies:
        lines.append(
            "  ⚠ concentration devise / currency concentration: "
            + ", ".join(
                f"{c['currency']} {float(c['pct_capital']) * 100:.0f}%"
                for c in d.concentrated_currencies
            )
        )
    lines += [
        "",
        "Seuls les ordres ci-dessus ont été exécutés ; le Brief est une prévision.",
        "Only the orders above were executed; the Brief is a forecast.",
        "Paper-trading forward test — no real money. Not financial advice.",
    ]
    return "\n".join(lines)


def _digest_subject(d: DailyDigest) -> str:
    date = _fr_date(d.date)
    if d.opened:
        return f"BeRich — {len(d.opened)} ordre(s) à copier · {date}"
    if d.closed:
        return f"BeRich — digest du {date} ({len(d.closed)} clôture(s))"
    return f"BeRich — digest du {date}"


def send_daily_digest_email(
    digest: DailyDigest,
    config: EmailConfig | None = None,
    *,
    subject: str | None = None,
    smtp_factory=None,  # noqa: ANN001 — small DI hook used only in tests
) -> bool:
    """Send the bilingual daily digest. Returns ``True`` iff an email was actually sent.

    Best-effort like the rest of the module: missing SMTP config or a transient SMTP error returns
    ``False`` and never raises into the scheduler. Unlike the old BUY digest this fires on every
    weekday run (even a quiet day), so the user gets a dependable morning briefing.
    """
    cfg = config or EmailConfig.from_env()
    if cfg is None:
        logger.info("email: NOTIFY_EMAIL/SMTP_* not set, skipping daily digest")
        return False

    message = EmailMessage()
    message["Subject"] = subject or _digest_subject(digest)
    message["From"] = cfg.smtp_user
    message["To"] = cfg.notify_email
    message.set_content(_render_digest_text(digest))
    message.add_alternative(_render_digest_html(digest), subtype="html")

    if not _send(message, cfg, smtp_factory):
        return False
    logger.info(
        "email: sent daily digest to %s (%d opened, %d closed)",
        cfg.notify_email,
        len(digest.opened),
        len(digest.closed),
    )
    return True


def _send(message: EmailMessage, cfg: EmailConfig, smtp_factory) -> bool:  # noqa: ANN001 — test DI hook
    """Send a prepared message over STARTTLS; swallow SMTP/OS errors (never block a job)."""
    try:
        with (smtp_factory or smtplib.SMTP)(cfg.smtp_host, cfg.smtp_port) as smtp:
            smtp.starttls()
            smtp.login(cfg.smtp_user, cfg.smtp_pass)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("email: SMTP send failed (%s); not blocking the caller", exc)
        return False
    return True


def send_alert_email(
    subject: str,
    body: str,
    config: EmailConfig | None = None,
    *,
    smtp_factory=None,  # noqa: ANN001 — small DI hook used only in tests
) -> bool:
    """Send a plain-text operational alert (e.g. a scheduler job crashed).

    Returns ``True`` if sent, ``False`` if email isn't configured or SMTP failed. Best-effort
    by design: an alert failing to send must never itself raise into the scheduler.
    """
    cfg = config or EmailConfig.from_env()
    if cfg is None:
        logger.info("email: NOTIFY_EMAIL/SMTP_* not set, skipping alert %r", subject)
        return False
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = cfg.smtp_user
    message["To"] = cfg.notify_email
    message.set_content(body)
    if not _send(message, cfg, smtp_factory):
        return False
    logger.info("email: sent alert to %s (%s)", cfg.notify_email, subject)
    return True
