"""Scheduled job functions.

Each job is a plain function taking a :class:`~berich.config.Config` so it can be
called directly in tests or wired into APScheduler by the runner. Jobs are
idempotent: re-running a day's refresh/signals overwrites rather than duplicates.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
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
    from collections.abc import Callable

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


def refresh_universe_job(config: Config) -> dict[str, int]:
    """Refresh the wider long/short universe (mid/small-caps) the daily chain misses.

    ``daily_paper_job`` only refreshes the mega watchlist + multi-asset universes
    (``all_runtime_tickers``). The cross-sectional long/short model ranks the full
    ``longshort.universe`` (≈274 tickers), so those bars must be kept fresh too. Uses the
    parallel, liquidity-gated :func:`update_universe`; incremental fetch makes re-runs cheap.
    """
    from berich.data import update_universe  # noqa: PLC0415

    tickers = config.tickers_for_universe(config.longshort.universe)
    if not tickers:
        return {"tickers": 0, "warned": 0}
    reports = update_universe(config, tickers)
    warned = sum(1 for r in reports if not r.ok)
    logger.info("refresh_universe: %d tickers refreshed, %d warned", len(reports), warned)
    return {"tickers": len(reports), "warned": warned}


def longshort_signals_job(config: Config) -> dict[str, object]:
    """Generate + persist today's market-neutral long/short basket (paper book)."""
    from berich.signals import LongShortStore, generate_longshort_book  # noqa: PLC0415

    store = OhlcvStore(config.ohlcv_dir)
    book = generate_longshort_book(config, store)
    if book is None or not book.positions:
        logger.info("longshort_signals: no basket (empty cache or too few names)")
        return {"positions": 0, "saved": 0}
    saved = LongShortStore(config.db_path).save(book)
    logger.info(
        "longshort_signals: %d positions, gross=%.2f, %d saved",
        len(book.positions),
        book.gross_exposure,
        saved,
    )
    return {"positions": len(book.positions), "saved": saved}


def _zoo_factory(model_name: str, *, device: str | None = None) -> tuple[str, dict, Callable]:
    """Return ``(framework, hyperparams, factory)`` for a zoo model name."""
    if model_name == "lgbm":
        from berich.models import LGBMModel  # noqa: PLC0415

        return "lightgbm", {}, LGBMModel
    if model_name == "patchtst":
        from berich.models import PatchTSTConfig, PatchTSTModel  # noqa: PLC0415

        cfg = PatchTSTConfig(device=device)
        return "patchtst", cfg.as_dict(), lambda: PatchTSTModel(cfg)
    if model_name == "lstm":
        from berich.models import LSTMConfig, LSTMModel  # noqa: PLC0415

        cfg = LSTMConfig(device=device)
        return "lstm", cfg.as_dict(), lambda: LSTMModel(cfg)
    if model_name == "tft":
        from berich.models import TFTConfig, TFTModel  # noqa: PLC0415

        cfg = TFTConfig(device=device)
        return "tft", cfg.as_dict(), lambda: TFTModel(cfg)
    msg = f"unknown zoo model '{model_name}'"
    raise ValueError(msg)


def retrain_zoo_job(config: Config, *, date_str: str | None = None) -> dict[str, object]:
    """Nightly: retrain the enabled zoo, then promote the single best guard-passing model.

    Idempotent: each model saves to a *dated* artifact name (re-running a night overwrites
    that night's candidates), and only one model — the highest OOS Sharpe that beats both
    the LightGBM baseline and buy & hold — is promoted. If none clears the guard the active
    model is left untouched.
    """
    from berich.models import promote  # noqa: PLC0415
    from berich.training.deep import baseline_sharpe, train_deep_model  # noqa: PLC0415

    base = baseline_sharpe(config)
    day = date_str or datetime.now(UTC).date().isoformat()

    results = []
    for model_name in config.zoo.enabled_models:
        framework, hyperparams, factory = _zoo_factory(model_name)
        name = f"{model_name}-{day}"
        res = train_deep_model(
            config,
            name=name,
            framework=framework,
            model_factory=factory,
            hyperparams=hyperparams,
            baseline_sharpe=base,
            promote_if_passes=False,
        )
        results.append((name, res))
        logger.info(
            "zoo retrain %s: Sharpe=%.3f promotable=%s",
            name,
            res.strategy_sharpe,
            res.beats_buy_hold and res.beats_baseline,
        )

    promotable = [(n, r) for n, r in results if r.beats_buy_hold and r.beats_baseline]
    promoted = ""
    if promotable:
        best_name, _ = max(promotable, key=lambda nr: nr[1].strategy_sharpe)
        try:
            promote(best_name, registry_dir=config.models_dir)
            promoted = best_name
            logger.info("zoo retrain promoted '%s'", best_name)
        except ValueError as exc:
            logger.warning("zoo retrain promotion blocked: %s", exc)
    else:
        logger.info("zoo retrain: no candidate beat baseline + buy & hold; active unchanged")

    return {"baseline_sharpe": base, "candidates": len(results), "promoted": promoted}


def weekend_hpo_job(config: Config) -> dict[str, float]:
    """Weekend: run Optuna HPO for each searchable zoo model to keep the GPUs busy."""
    from berich.training.hpo import SUPPORTED_MODELS, run_hpo  # noqa: PLC0415

    out: dict[str, float] = {}
    for model_name in config.zoo.enabled_models:
        if model_name not in SUPPORTED_MODELS:
            continue
        study = run_hpo(config, model_name, n_trials=config.zoo.hpo_trials)
        out[model_name] = float(study.best_value)
        logger.info("weekend HPO %s best Sharpe=%.3f", model_name, study.best_value)
    return out


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
