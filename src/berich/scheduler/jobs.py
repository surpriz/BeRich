"""Scheduled job functions.

Each job is a plain function taking a :class:`~berich.config.Config` so it can be
called directly in tests or wired into APScheduler by the runner. Jobs are
idempotent: re-running a day's refresh/signals overwrites rather than duplicates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from berich.data import (
    NewsStore,
    RateLimitError,
    update_news_watchlist,
    update_watchlist,
)
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.monitoring import feature_drift
from berich.notifications import send_buy_signals_email
from berich.signals import (
    SignalStore,
    generate_signals,
    open_new_trades,
    update_open_trades,
)

if TYPE_CHECKING:
    from berich.config import Config
    from berich.monitoring import DriftReport

logger = logging.getLogger(__name__)

# Number of most-recent samples treated as the "current" window for drift.
DRIFT_CURRENT_SAMPLES = 2000


def daily_paper_job(config: Config) -> dict[str, int]:
    """Daily chain: refresh OHLCV → news (if configured) → signals → paper book.

    Returns a dict with sub-step counts. Every step is idempotent (signals
    upsert; paper opens skip duplicates; paper updates only touch open rows;
    news fetch walks forward from the cache tip; FinBERT scores only
    not-yet-scored rows), so the scheduler can re-run it safely and the user
    can run pieces manually via ``berich paper update`` / ``berich news ...``
    without colliding.

    The news sub-step is best-effort: if ``ALPHAVANTAGE_KEY`` is not set, or
    the daily quota is exhausted, we log a warning and continue with the
    rest of the chain rather than letting the whole job fail.
    """
    update_watchlist(config)
    news_rows = _try_news_refresh(config)
    finbert_scored = _try_finbert_score(config) if news_rows >= 0 else 0
    store = OhlcvStore(config.ohlcv_dir)
    signals = generate_signals(config, store)
    signal_store = SignalStore(config.db_path)
    saved = signal_store.save(signals)
    opened = open_new_trades(config, store, signal_store)
    closed = update_open_trades(config, store)
    # Email digest only fires when we actually opened paper trades — that filter
    # makes the "is there a new BUY for me to act on" semantics exact, instead
    # of spamming whenever the model emits a BUY proba on a ticker we already
    # held.
    notified = False
    if opened > 0:
        notified = send_buy_signals_email([s for s in signals if s.signal == "BUY"])
    logger.info(
        "daily_paper: %d news rows, %d finbert scored, %d signals saved,"
        " %d paper opened, %d paper closed, email=%s",
        max(news_rows, 0),
        finbert_scored,
        saved,
        opened,
        closed,
        notified,
    )
    return {
        "news_rows": max(news_rows, 0),
        "finbert_scored": finbert_scored,
        "signals_saved": saved,
        "trades_opened": opened,
        "trades_closed": closed,
        "email_sent": int(notified),
    }


def _try_news_refresh(config: Config) -> int:
    """Best-effort AV refresh; returns rows added, or -1 if news is opted out."""
    import os  # noqa: PLC0415 — local import keeps scheduler module-import lean

    if not os.environ.get("ALPHAVANTAGE_KEY"):
        logger.info("daily_paper: ALPHAVANTAGE_KEY unset, skipping news refresh")
        return -1
    try:
        reports = update_news_watchlist(config, max_pages_per_ticker=1)
    except RateLimitError as exc:
        logger.warning("daily_paper: news refresh skipped — %s", exc)
        return 0
    return sum(r.rows_added for r in reports)


def _try_finbert_score(config: Config) -> int:
    """Score unscored rows on the local GPU; returns count updated (0 if none)."""
    store = NewsStore(config.news_dir)
    if not store.has_any_data():
        return 0
    from berich.models.finbert_scorer import FinBertScorer  # noqa: PLC0415 — heavy import

    try:
        scorer = FinBertScorer()
    except (OSError, RuntimeError) as exc:
        logger.warning("daily_paper: FinBERT init failed — %s; skipping scoring", exc)
        return 0

    import pandas as pd  # noqa: PLC0415 — local to keep the lean import path clean

    total = 0
    for ticker in config.watchlist:
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        unscored = df[df["finbert_score"].isna()]
        if unscored.empty:
            continue
        texts = [
            f"{title} {summary}".strip()
            for title, summary in zip(
                unscored["title"].fillna("").to_list(),
                unscored["summary"].fillna("").to_list(),
                strict=False,
            )
        ]
        probs = scorer.score_texts(texts)
        scores = pd.DataFrame(
            {
                "url": unscored["url"].to_numpy(),
                "finbert_neg": probs[:, 0],
                "finbert_neu": probs[:, 1],
                "finbert_pos": probs[:, 2],
                "finbert_score": probs[:, 2] - probs[:, 0],
            }
        )
        total += store.update_finbert(ticker, scores)
    return total


def check_drift_job(config: Config) -> DriftReport:
    """Compare recent feature distributions to the training era; log if retrain needed."""
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)

    n_current = min(DRIFT_CURRENT_SAMPLES, len(dataset) // 2)
    reference = dataset.x.iloc[:-n_current]
    current = dataset.x.iloc[-n_current:]
    report = feature_drift(reference, current)

    level = logging.WARNING if report.should_retrain else logging.INFO
    logger.log(
        level,
        "drift: %d/%d features drifted (%.0f%%); retrain=%s",
        report.n_drifted,
        len(report.features),
        report.share_drifted * 100,
        report.should_retrain,
    )
    return report
