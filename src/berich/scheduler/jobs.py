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
from berich.signals.service import LONG_SIGNALS

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
    # generate_signals now covers every *optimized* asset across all classes, served from each
    # asset's own model — no generic fallback, no separate multi-asset pass.
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
        notified = send_buy_signals_email([s for s in signals if s.signal in LONG_SIGNALS])
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


def _cfg_kwargs(config_cls: type, params: dict) -> dict:
    """Keep only the params that are real fields of ``config_cls`` (drops HPO extras)."""
    from dataclasses import fields  # noqa: PLC0415

    valid = {f.name for f in fields(config_cls)}
    return {k: v for k, v in params.items() if k in valid}


def _zoo_factory(
    model_name: str, *, device: str | None = None, params: dict | None = None
) -> tuple[str, dict, Callable]:
    """Return ``(framework, hyperparams, factory)`` for a zoo model name.

    ``params`` (best from the latest HPO study) overrides the defaults when supplied.
    """
    params = params or {}
    if model_name == "lgbm":
        from berich.models import LGBMModel  # noqa: PLC0415

        return "lightgbm", params, lambda: LGBMModel(**params)
    if model_name == "patchtst":
        from berich.models import PatchTSTConfig, PatchTSTModel  # noqa: PLC0415

        cfg = PatchTSTConfig(device=device, **_cfg_kwargs(PatchTSTConfig, params))
        return "patchtst", cfg.as_dict(), lambda: PatchTSTModel(cfg)
    if model_name == "lstm":
        from berich.models import LSTMConfig, LSTMModel  # noqa: PLC0415

        cfg = LSTMConfig(device=device, **_cfg_kwargs(LSTMConfig, params))
        return "lstm", cfg.as_dict(), lambda: LSTMModel(cfg)
    if model_name == "tft":
        from berich.models import TFTConfig, TFTModel  # noqa: PLC0415

        cfg = TFTConfig(device=device, **_cfg_kwargs(TFTConfig, params))
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
    from berich.training.hpo import best_params_for  # noqa: PLC0415

    base = baseline_sharpe(config)
    day = date_str or datetime.now(UTC).date().isoformat()

    results = []
    for model_name in config.zoo.enabled_models:
        # Use the best hyperparameters from the latest HPO study (weekend search), if any.
        params = best_params_for(config, model_name)
        framework, hyperparams, factory = _zoo_factory(model_name, params=params)
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


def retrain_asset_models_job(config: Config) -> dict[str, object]:
    """Nightly: retrain + force-promote the dedicated per-asset-class advisory models."""
    from berich.training.asset_class import train_asset_class_model  # noqa: PLC0415

    results = {}
    for asset_class in ("crypto", "forex", "commodities", "fr_stocks"):
        if not config.universes.get(asset_class):
            continue
        try:
            res = train_asset_class_model(config, asset_class)
            results[asset_class] = res.get("auc") if res.get("trained") else res.get("reason")
        except Exception:  # noqa: BLE001 — one class failing must not abort the others
            logger.warning("retrain_asset_models: %s failed", asset_class, exc_info=True)
            results[asset_class] = "error"
    logger.info("retrain_asset_models: %s", results)
    return results


def backup_job(config: Config) -> dict[str, object]:
    """Archive the training state (Optuna studies, models, signals DB) with rotation."""
    from berich.backup import create_backup  # noqa: PLC0415

    return create_backup(config, timestamp=datetime.now(UTC).isoformat())


def _ticker_hpo_task(
    device: str | None,
    config: Config,
    ticker: str,
    model_name: str,
    side: str,
    n_trials: int,
) -> None:
    """Module-level (picklable) per-ticker HPO call for the GPU pool worker.

    The GPU pool invokes ``fn(device, *args)``; ``device`` is the pinned ``cuda:N`` string in
    a worker or ``cpu`` in the sequential fallback. Failures are swallowed so one bad ticker
    never poisons the pool — the study simply keeps whatever trials it already had.
    """
    from berich.training.hpo import run_ticker_hpo  # noqa: PLC0415

    try:
        run_ticker_hpo(config, ticker, model_name, side, n_trials=n_trials, device=device)
    except Exception:  # noqa: BLE001 — one ticker/model HPO failure must not abort the sweep
        logger.warning("ticker HPO failed for %s/%s/%s", ticker, model_name, side, exc_info=True)


def ticker_nightly_refresh_job(config: Config) -> dict[str, int]:
    """Nightly: light top-up + re-tournament for tickers that already have a promoted model.

    For each tradeable ticker x side that is already promoted, run a few HPO trials
    (``zoo.ticker_nightly_hpo_trials``) into the shared study, then re-run the tournament to
    re-fit / re-promote the honest winner. Bounded and best-effort: only touches tickers that
    already cleared the gate once, and wraps each ticker's work so one failure never aborts the
    batch. The deep-model HPO runs on the GPU pool; the tournament re-fit runs in-process.
    """
    from berich.models.registry import load_active  # noqa: PLC0415
    from berich.training.gpu_pool import GpuTask, run_on_gpus  # noqa: PLC0415
    from berich.training.tournament import train_ticker_tournament  # noqa: PLC0415

    n_trials = config.zoo.ticker_nightly_hpo_trials
    deep_models = [m for m in config.zoo.ticker_tournament_models if m != "lgbm"]
    refreshed = 0
    promoted = 0
    skipped = 0
    failed = 0
    for ticker in config.tradeable_tickers():
        for side in config.zoo.ticker_sides:
            if load_active(config.model_dir_for_ticker(ticker, side)) is None:
                skipped += 1
                continue
            try:
                tasks = [
                    GpuTask(
                        _ticker_hpo_task,
                        args=(config, ticker, model_name, side, n_trials),
                        label=f"{ticker}/{model_name}/{side}",
                    )
                    for model_name in deep_models
                ]
                if tasks:
                    run_on_gpus(tasks, config.zoo.gpu_ids)
                result = train_ticker_tournament(config, ticker, side, calibrate=True)
                refreshed += 1
                promoted += int(result.promoted)
            except Exception:  # noqa: BLE001 — one bad ticker must not abort the nightly batch
                logger.warning("ticker_nightly_refresh: %s/%s failed", ticker, side, exc_info=True)
                failed += 1
    summary = {
        "refreshed": refreshed,
        "promoted": promoted,
        "skipped": skipped,
        "failed": failed,
        "signals_refreshed": refresh_signals(config) if refreshed else 0,
    }
    logger.info("ticker_nightly_refresh: %s", summary)
    return summary


def _hpo_and_tournament(config: Config, ticker: str, side: str, n_trials: int) -> bool:
    """Full first-pass HPO (deep models on the GPU pool) + tournament for one ticker x side.

    Returns whether a model was promoted. Deep-model HPO runs across the GPU pool; LightGBM is
    searched in-process by the tournament's own best-params lookup. Raises on hard failure so
    the caller can count it.
    """
    from berich.training.gpu_pool import GpuTask, run_on_gpus  # noqa: PLC0415
    from berich.training.tournament import train_ticker_tournament  # noqa: PLC0415

    deep_models = [m for m in config.zoo.ticker_tournament_models if m != "lgbm"]
    # LightGBM HPO runs in-process (CPU, fast); deep models go on the GPU pool.
    if "lgbm" in config.zoo.ticker_tournament_models:
        _ticker_hpo_task(None, config, ticker, "lgbm", side, n_trials)
    tasks = [
        GpuTask(
            _ticker_hpo_task,
            args=(config, ticker, model_name, side, n_trials),
            label=f"{ticker}/{model_name}/{side}",
        )
        for model_name in deep_models
    ]
    if tasks:
        run_on_gpus(tasks, config.zoo.gpu_ids)
    return train_ticker_tournament(config, ticker, side, calibrate=True).promoted


def _pending_hpo_targets(config: Config) -> list[tuple[str, str]]:
    """(ticker, side) pairs that have no per-asset HPO study yet, in config order.

    This is the queue's work-list: assets whose models still run on default params. Reusing the
    status scanner's trial counts means the queue is naturally resumable — a pair drops off the
    list as soon as its study has trials, so re-running the job continues where it left off.
    """
    from berich.training.status import _hpo_trial_counts, _hpo_trials_for  # noqa: PLC0415

    counts = _hpo_trial_counts(config.optuna_db)
    return [
        (ticker, side)
        for ticker in config.tradeable_tickers()
        for side in config.zoo.ticker_sides
        if _hpo_trials_for(counts, ticker, None, side) == 0
    ]


def refresh_signals(config: Config) -> int:
    """Regenerate + persist today's signals so the dashboard matches the current model set.

    Drops the stored table first so assets no longer optimized disappear (rather than lingering
    via the (date, ticker) upsert). Returns the number of signals written. Called after any job
    that mutates models, so /signals never drifts from /training. Best-effort: never raises.
    """
    import duckdb  # noqa: PLC0415

    try:
        store = OhlcvStore(config.ohlcv_dir)
        sigs = generate_signals(config, store)
        with duckdb.connect(str(config.db_path)) as con:
            con.execute("DELETE FROM signals")
        SignalStore(config.db_path).save(sigs)
    except Exception:  # noqa: BLE001 — a refresh failure must not abort the calling job
        logger.warning("refresh_signals failed", exc_info=True)
        return 0
    return len(sigs)


def ticker_hpo_queue_job(config: Config, *, max_assets: int = 1) -> dict[str, object]:
    """Sequential first-HPO queue: optimize the next ``max_assets`` un-searched ticker x sides.

    Processes assets **one at a time** (no two deep HPO searches overlap), so the GPUs aren't
    swamped by the all-at-once weekend sweep. Only touches pairs with no per-asset Optuna study
    yet, and is resumable: each call drains the next slice of the work-list. Run it on a short
    cron (or in a loop) to give every asset its first deep HPO without a thundering herd.
    """
    n_trials = config.zoo.ticker_initial_hpo_trials
    pending = _pending_hpo_targets(config)
    processed: list[str] = []
    promoted = 0
    failed = 0
    for ticker, side in pending[:max_assets]:
        try:
            if _hpo_and_tournament(config, ticker, side, n_trials):
                promoted += 1
            processed.append(f"{ticker}/{side}")
        except Exception:  # noqa: BLE001 — one bad asset must not stall the queue
            logger.warning("ticker_hpo_queue: %s/%s failed", ticker, side, exc_info=True)
            failed += 1
    # Keep /signals in lock-step with the model set we just changed.
    refreshed = refresh_signals(config) if processed else 0
    summary: dict[str, object] = {
        "processed": processed,
        "promoted": promoted,
        "failed": failed,
        "remaining": max(0, len(pending) - len(processed) - failed),
        "signals_refreshed": refreshed,
    }
    logger.info("ticker_hpo_queue: %s", summary)
    return summary


def ticker_initial_sweep_job(config: Config) -> dict[str, int]:
    """Weekend heavy sweep: full per-ticker HPO + tournament for every tradeable ticker x side.

    For each ticker x side, runs the full first-pass HPO (``zoo.ticker_initial_hpo_trials``) for
    each deep model across the GPU pool, then a tournament that promotes the honest winner. This
    is the cold-start that nightly_refresh later tops up. Best-effort per (ticker, side).
    """
    n_trials = config.zoo.ticker_initial_hpo_trials
    swept = 0
    promoted = 0
    failed = 0
    for ticker in config.tradeable_tickers():
        for side in config.zoo.ticker_sides:
            try:
                if _hpo_and_tournament(config, ticker, side, n_trials):
                    promoted += 1
                swept += 1
            except Exception:  # noqa: BLE001 — one bad ticker must not abort the weekend sweep
                logger.warning("ticker_initial_sweep: %s/%s failed", ticker, side, exc_info=True)
                failed += 1
    summary = {"swept": swept, "promoted": promoted, "failed": failed}
    logger.info("ticker_initial_sweep: %s", summary)
    return summary


def _run_hpo_round(config: Config, n_trials: int, label: str) -> dict[str, float]:
    """Run an Optuna round of ``n_trials`` per searchable model into the shared study."""
    from berich.training.hpo import SUPPORTED_MODELS, run_hpo  # noqa: PLC0415

    out: dict[str, float] = {}
    for model_name in config.zoo.enabled_models:
        if model_name not in SUPPORTED_MODELS:
            continue
        study = run_hpo(config, model_name, n_trials=n_trials)
        out[model_name] = float(study.best_value)
        logger.info("%s HPO %s best Sharpe=%.3f", label, model_name, study.best_value)
    return out


def nightly_hpo_job(config: Config) -> dict[str, float]:
    """Nightly: a light Optuna round (few trials) that accumulates into the shared study.

    Trials persist in the SQLite study, so a handful each night keeps improving the best
    params that ``retrain_zoo_job`` consumes — without the weekend's full GPU sweep.
    """
    return _run_hpo_round(config, config.zoo.nightly_hpo_trials, "nightly")


def weekend_hpo_job(config: Config) -> dict[str, float]:
    """Weekend: a deep Optuna sweep for each searchable zoo model to keep the GPUs busy."""
    return _run_hpo_round(config, config.zoo.hpo_trials, "weekend")


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
