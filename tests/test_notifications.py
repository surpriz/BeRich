"""Polish-phase tests: email digest, calibration table, CSV export."""

from __future__ import annotations

import smtplib
from unittest.mock import MagicMock

import pandas as pd
import pytest
from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent

import berich.notifications as notif
from berich.config import Config, SignalConfig
from berich.notifications.email import (
    EmailConfig,
    _format_html_table,
    _format_text_table,
    send_alert_email,
    send_buy_signals_email,
)
from berich.scheduler import runner
from berich.signals import PaperStore, compute_calibration
from berich.signals.calibration import DEFAULT_BUCKETS
from berich.signals.service import BUY, NEUTRAL, Signal
from berich.signals.store import SignalStore

# -------------------------------------------------------------- email config ----


def test_email_config_from_env_returns_none_when_incomplete(monkeypatch):
    """Missing any of the 4 required vars must skip the email path entirely."""
    monkeypatch.delenv("NOTIFY_EMAIL", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASS", "x")
    assert EmailConfig.from_env() is None


def test_email_config_from_env_populates_when_complete(monkeypatch):
    monkeypatch.setenv("NOTIFY_EMAIL", "me@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")
    cfg = EmailConfig.from_env()
    assert cfg is not None
    assert cfg.notify_email == "me@example.com"
    assert cfg.smtp_port == 587  # default


def _signal(ticker: str, *, proba: float = 0.6, signal_type: str = BUY) -> Signal:
    return Signal(
        date=pd.Timestamp("2024-01-05"),
        ticker=ticker,
        signal=signal_type,
        proba=proba,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size_shares=20,
        notional=2000.0,
    )


def test_send_buy_signals_email_skips_when_no_buys():
    """Empty list and SELL/NEUTRAL-only lists must NOT contact SMTP."""
    fake_smtp = MagicMock()
    cfg = EmailConfig(notify_email="me@x.com", smtp_host="h", smtp_user="u", smtp_pass="p")
    assert send_buy_signals_email([], cfg, smtp_factory=fake_smtp) is False
    assert (
        send_buy_signals_email(
            [_signal("AAPL", signal_type=NEUTRAL)],
            cfg,
            smtp_factory=fake_smtp,
        )
        is False
    )
    fake_smtp.assert_not_called()


def test_send_buy_signals_email_sends_when_signals_present():
    """One BUY → one SMTP send_message call with correct From / To headers."""
    fake_conn = MagicMock()
    fake_smtp = MagicMock(return_value=fake_conn)
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.__exit__.return_value = False
    cfg = EmailConfig(notify_email="me@x.com", smtp_host="h", smtp_user="bot@x.com", smtp_pass="p")
    sent = send_buy_signals_email([_signal("AAPL")], cfg, smtp_factory=fake_smtp)
    assert sent is True
    fake_conn.starttls.assert_called_once()
    fake_conn.login.assert_called_once_with("bot@x.com", "p")
    fake_conn.send_message.assert_called_once()
    msg = fake_conn.send_message.call_args[0][0]
    assert msg["To"] == "me@x.com"
    assert msg["From"] == "bot@x.com"
    assert "AAPL" in msg.get_body("plain").get_content()


def test_send_buy_signals_email_returns_false_on_smtp_error():
    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.__exit__.return_value = False
    fake_conn.send_message.side_effect = smtplib.SMTPException("boom")
    fake_smtp = MagicMock(return_value=fake_conn)
    cfg = EmailConfig(notify_email="me@x.com", smtp_host="h", smtp_user="bot@x.com", smtp_pass="p")
    assert send_buy_signals_email([_signal("AAPL")], cfg, smtp_factory=fake_smtp) is False


def test_send_alert_email_sends_plain_text():
    fake_conn = MagicMock()
    fake_smtp = MagicMock(return_value=fake_conn)
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.__exit__.return_value = False
    cfg = EmailConfig(notify_email="me@x.com", smtp_host="h", smtp_user="bot@x.com", smtp_pass="p")
    sent = send_alert_email(
        "BeRich — job 'x' failed", "boom\ntraceback", cfg, smtp_factory=fake_smtp
    )
    assert sent is True
    fake_conn.send_message.assert_called_once()
    msg = fake_conn.send_message.call_args[0][0]
    assert msg["Subject"] == "BeRich — job 'x' failed"
    assert msg["To"] == "me@x.com"
    assert "boom" in msg.get_content()


def test_send_alert_email_skips_when_unconfigured(monkeypatch):
    for var in ("NOTIFY_EMAIL", "SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(var, raising=False)
    fake_smtp = MagicMock()
    assert send_alert_email("subj", "body", smtp_factory=fake_smtp) is False
    fake_smtp.assert_not_called()


def test_scheduler_registers_job_error_listener_and_alerts(monkeypatch):
    """A job that raises triggers EVENT_JOB_ERROR -> send_alert_email is called."""
    calls: list[tuple[str, str]] = []
    # The listener imports send_alert_email lazily from berich.notifications, so patch it there.
    monkeypatch.setattr(
        notif, "send_alert_email", lambda subject, body, **_kw: calls.append((subject, body))
    )

    # The listener must be registered on the scheduler.
    scheduler = runner.build_scheduler(Config(watchlist=["AAPL"]))
    assert scheduler is not None

    event = JobExecutionEvent(
        EVENT_JOB_ERROR,
        "boom_job",
        "default",
        scheduled_run_time=None,
        exception=RuntimeError("kaboom"),
    )
    runner._on_job_error(event)
    assert len(calls) == 1
    subject, body = calls[0]
    assert "boom_job" in subject
    assert "kaboom" in body


def test_email_body_includes_advisory_disclaimer():
    text = _format_text_table([_signal("AAPL")])
    html = _format_html_table([_signal("AAPL")])
    assert "Advisory" in text
    assert "Advisory" in html
    # Sanity: HTML body has a table tag with the ticker symbol.
    assert "<table" in html and "AAPL" in html


# --------------------------------------------------------------- calibration ----


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path,
        watchlist=["AAA"],
        signals=SignalConfig(
            buy_threshold=0.55, sell_threshold=0.30, capital=10_000.0, risk_pct=0.01
        ),
    )


def _seed_signal(config: Config, ticker: str, date: str, proba: float) -> None:
    SignalStore(config.db_path).save(
        [
            Signal(
                date=pd.Timestamp(date),
                ticker=ticker,
                signal=BUY,
                proba=proba,
                entry=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                size_shares=10,
                notional=1000.0,
            )
        ]
    )


def _seed_closed_trade(
    config: Config,
    ticker: str,
    date_open: str,
    *,
    pnl_eur: float,
) -> None:
    paper = PaperStore(config.db_path)
    rows = pd.DataFrame(
        [
            {
                "date_open": date_open,
                "ticker": ticker,
                "signal": "BUY",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "size_shares": 10,
                "status": "open",
                "date_close": None,
                "exit_price": None,
                "pnl_pct": None,
                "pnl_eur": None,
            }
        ]
    )
    paper.insert_new(rows)
    paper.close_trade(
        date_open=pd.Timestamp(date_open),
        ticker=ticker,
        date_close=pd.Timestamp(date_open) + pd.Timedelta(days=5),
        exit_price=110.0 if pnl_eur > 0 else 95.0,
        status="closed_target" if pnl_eur > 0 else "closed_stop",
        pnl_pct=pnl_eur / 100.0,
        pnl_eur=pnl_eur,
    )


def test_calibration_empty_when_no_closed_trades(config):
    report = compute_calibration(config)
    assert report.n_trades_total == 0
    assert report.n_with_proba == 0
    assert len(report.buckets) == len(DEFAULT_BUCKETS)
    # All buckets are zero-count placeholders.
    assert all(b.n_trades == 0 for b in report.buckets)


def test_calibration_bucketizes_closed_trades_by_proba(config):
    """Two wins at proba 0.62 + two losses at proba 0.72 → one populated bucket each."""
    for i, (date, proba, pnl) in enumerate(
        [
            ("2024-01-02", 0.62, 50.0),
            ("2024-02-02", 0.62, 75.0),
            ("2024-03-02", 0.72, -40.0),
            ("2024-04-02", 0.72, -60.0),
        ]
    ):
        ticker = f"T{i}"
        _seed_signal(config, ticker, date, proba)
        _seed_closed_trade(config, ticker, date, pnl_eur=pnl)
    report = compute_calibration(config)
    assert report.n_trades_total == 4
    assert report.n_with_proba == 4
    # [0.60, 0.65) → 2 wins out of 2.
    bucket_low = next(b for b in report.buckets if b.low == pytest.approx(0.60))
    assert bucket_low.n_trades == 2
    assert bucket_low.win_rate == pytest.approx(1.0)
    # [0.70, 0.75) → 0 wins out of 2.
    bucket_high = next(b for b in report.buckets if b.low == pytest.approx(0.70))
    assert bucket_high.n_trades == 2
    assert bucket_high.win_rate == pytest.approx(0.0)
