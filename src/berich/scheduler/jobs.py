"""Scheduled job functions.

Each job is a plain function taking a :class:`~berich.config.Config` so it can be
called directly in tests or wired into APScheduler by the runner. Jobs are
idempotent: re-running a day's refresh/signals overwrites rather than duplicates.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
from berich.notifications import build_daily_digest, send_daily_digest_email
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

# Cross-process exclusive lock for the per-asset HPO+tournament queue. The standalone sweep
# service (scripts/run_full_sweep.py) and the scheduler's HPO jobs all take this, so only one
# driver ever trains a (ticker, side, strategy) at a time — no Optuna/registry write races. The
# sweep holds it for its whole life; the scheduler jobs try non-blocking and skip when it's held.
_HPO_LOCK_FILE = ".hpo_sweep.lock"


def acquire_hpo_lock(config: Config) -> int | None:
    """Try to take the HPO queue lock without blocking.

    Returns an open file descriptor (keep it open to HOLD the lock; ``os.close`` to release) or
    ``None`` if another process already holds it. The lock is advisory and auto-released if the
    holder dies (flock semantics), so a crashed driver never deadlocks the queue.
    """
    path = config.data_dir / _HPO_LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


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
    config.apply_active_risk_profile()  # honor a UI risk-profile switch without a restart
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
    # Daily briefing: a bilingual digest of the run + portfolio snapshot, fired on every weekday
    # run (not only when a trade opened) so the user gets a dependable morning email. It is
    # best-effort — assembly or SMTP failing must never abort the job.
    notified = False
    try:
        digest = build_daily_digest(config, store, signals)
        notified = send_daily_digest_email(digest)
    except Exception:  # noqa: BLE001 — a notification failure must not abort the daily chain
        logger.warning("daily_paper: digest email failed", exc_info=True)
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
    result = {
        "news_rows": max(news_rows, 0),
        "finbert_scored": finbert_scored,
        "signals_saved": saved,
        "trades_opened": opened,
        "trades_closed": closed,
        "email_sent": int(notified),
    }
    _write_daily_heartbeat(config, result)
    return result


def _write_daily_heartbeat(config: Config, result: dict[str, int]) -> None:
    """Stamp ``data/last_daily_run.json`` so an external watchdog can verify the run happened.

    The scheduler is a single process: if it dies before 22:30 nobody emails anything. The
    systemd watchdog timer (see ``scripts/check_daily_run.py``) reads this file at 23:15 and
    alerts when the evening run is missing. Best-effort — never aborts the daily chain.
    """
    try:
        payload = {"at": datetime.now(UTC).isoformat(), **result}
        path = config.data_dir / "last_daily_run.json"
        path.write_text(json.dumps(payload, indent=2))
    except OSError:
        logger.warning("daily_paper: heartbeat write failed", exc_info=True)


def intraday_paper_job(config: Config) -> dict[str, int]:
    """Hourly intraday (1h crypto) chain — fully isolated from the daily chain above.

    Refresh 1h Binance data → generate intraday signals → save to the SEPARATE intraday DuckDB →
    open promoted-tier intraday trades under the same money-management caps → walk/close open
    intraday trades. Disabled unless ``config.intraday.enabled``. Reuses the daily risk profile.
    No email digest (POC); the scheduler's EVENT_JOB_ERROR listener still alerts on failure. Never
    touches the daily DB, ``refresh_signals`` or the HPO lock.
    """
    if not config.intraday.enabled:
        return {"signals_saved": 0, "trades_opened": 0, "trades_closed": 0}
    config.apply_active_risk_profile()
    # Lazy imports: keep the intraday subsystem (ccxt adapter, intraday paper book) out of the
    # module-import path so the daily scheduler stays lean.
    from berich.data.binance_adapter import update_intraday  # noqa: PLC0415
    from berich.signals import SignalStore  # noqa: PLC0415
    from berich.signals.paper_intraday import (  # noqa: PLC0415
        open_new_intraday_trades,
        update_open_intraday_trades,
    )
    from berich.signals.service import generate_intraday_signals  # noqa: PLC0415

    store = OhlcvStore(config.ohlcv_intraday_dir, interval=config.intraday.interval)
    for ticker in config.intraday.tickers:
        try:
            update_intraday(config, store, ticker)
        except Exception:  # noqa: BLE001 — a transient ccxt error must not abort the chain
            logger.warning("intraday_paper: data refresh failed for %s", ticker, exc_info=True)

    signals = generate_intraday_signals(config, store)
    signal_store = SignalStore(config.intraday_db_path)
    saved = signal_store.save(signals)
    opened = open_new_intraday_trades(config, store, signal_store)
    closed = update_open_intraday_trades(config, store)
    logger.info("intraday_paper: %d signals saved, %d opened, %d closed", saved, opened, closed)
    return {"signals_saved": saved, "trades_opened": opened, "trades_closed": closed}


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


def cross_check_data_job(config: Config, *, sample_size: int = 6) -> dict[str, object]:
    """Weekly sanity check: compare recent yfinance closes to Stooq on a liquid sample.

    yfinance is the only OHLCV source — a silent upstream regression would poison labels and
    the paper book with no error raised anywhere. Mismatches beyond the tolerance email an
    alert (same channel as data-health); an unreachable Stooq just logs.
    """
    from berich.data.crosscheck import cross_check_ticker, stooq_symbol  # noqa: PLC0415

    store = OhlcvStore(config.ohlcv_dir)
    held = {
        str(t)
        for t in config.watchlist
        if stooq_symbol(str(t)) is not None  # mappable US names first
    }
    universe = [t for t in config.all_runtime_tickers() if stooq_symbol(t) is not None]
    sample = sorted(held.union(universe[: max(0, sample_size - len(held))]))[:sample_size]

    findings: list[dict[str, object]] = []
    for ticker in sample:
        findings.extend(cross_check_ticker(store, ticker))

    if findings:
        from berich.notifications import send_alert_email  # noqa: PLC0415

        lines = "\n".join(
            f"  {f['ticker']} {f['date']}: ours={f['ours']:.4f} stooq={f['stooq']:.4f} "
            f"(écart {cast('float', f['rel_diff']) * 100:.2f}%)"
            for f in findings
        )
        send_alert_email(
            subject="BeRich ALERTE : écart yfinance vs Stooq sur des clôtures récentes",
            body=(
                "Les clôtures yfinance en cache divergent de Stooq (source indépendante) :\n"
                f"{lines}\n\n"
                "Vérifier un éventuel problème de splits/ajustements côté yfinance avant le "
                "prochain run quotidien. / Cached yfinance closes disagree with Stooq — check "
                "for an upstream adjustment problem before the next daily run."
            ),
        )
    logger.info("cross_check_data: %d tickers checked, %d mismatches", len(sample), len(findings))
    return {"checked": sample, "mismatches": findings}


def backup_job(config: Config) -> dict[str, object]:
    """Archive the training state (Optuna studies, models, signals DB) with rotation.

    When ``BERICH_BACKUP_REMOTE`` is set (an rclone remote like ``gdrive:berich-backups``),
    the fresh archive is also copied off-site; a failed sync raises so the scheduler's
    EVENT_JOB_ERROR listener emails the alert — a silent local-only backup is exactly the
    failure mode this guards against.
    """
    from berich.backup import create_backup, sync_offsite  # noqa: PLC0415

    summary = create_backup(config, timestamp=datetime.now(UTC).isoformat())
    path = summary.get("path")
    if path is not None:
        summary["offsite_remote"] = sync_offsite(Path(str(path)))
    return summary


def _ticker_hpo_task(
    device: str | None,
    config: Config,
    ticker: str,
    model_name: str,
    side: str,
    n_trials: int,
    strategy: str = "fixed",
    interval: str = "1d",
) -> None:
    """Module-level (picklable) per-ticker x exit-strategy HPO call for the GPU pool worker.

    The GPU pool invokes ``fn(device, *args)``; ``device`` is the pinned ``cuda:N`` string in
    a worker or ``cpu`` in the sequential fallback. Failures are swallowed so one bad ticker
    never poisons the pool — the study simply keeps whatever trials it already had.
    """
    from berich.training.hpo import run_ticker_hpo  # noqa: PLC0415

    try:
        run_ticker_hpo(
            config,
            ticker,
            model_name,
            side,
            strategy=strategy,
            n_trials=n_trials,
            device=device,
            interval=interval,
        )
    except Exception:  # noqa: BLE001 — one ticker/model HPO failure must not abort the sweep
        logger.warning(
            "ticker HPO failed for %s/%s/%s/%s/%s",
            ticker,
            model_name,
            side,
            strategy,
            interval,
            exc_info=True,
        )


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

    lock = acquire_hpo_lock(config)
    if lock is None:
        logger.info("ticker_nightly_refresh: HPO lock held (sweep service running), skipping")
        return {"refreshed": 0, "promoted": 0, "skipped": 0, "failed": 0}
    n_trials = config.zoo.ticker_nightly_hpo_trials
    deep_models = [m for m in config.zoo.ticker_tournament_models if m != "lgbm"]
    refreshed = 0
    promoted = 0
    skipped = 0
    failed = 0
    for ticker in config.tradeable_tickers():
        for side in config.zoo.ticker_sides:
            for strategy in config.zoo.ticker_exit_strategies:
                if load_active(config.model_dir_for_ticker(ticker, side, strategy)) is None:
                    skipped += 1
                    continue
                try:
                    tasks = [
                        GpuTask(
                            _ticker_hpo_task,
                            args=(config, ticker, model_name, side, n_trials, strategy),
                            label=f"{ticker}/{model_name}/{side}/{strategy}",
                        )
                        for model_name in deep_models
                    ]
                    if tasks:
                        run_on_gpus(tasks, config.zoo.gpu_ids)
                    result = train_ticker_tournament(
                        config, ticker, side, strategy=strategy, calibrate=True
                    )
                    refreshed += 1
                    promoted += int(result.promoted)
                except Exception:  # noqa: BLE001 — one bad ticker must not abort the nightly batch
                    logger.warning(
                        "ticker_nightly_refresh: %s/%s/%s failed",
                        ticker,
                        side,
                        strategy,
                        exc_info=True,
                    )
                    failed += 1
    os.close(lock)  # release the HPO queue lock (per-iteration try/except keeps the loop safe)
    summary = {
        "refreshed": refreshed,
        "promoted": promoted,
        "skipped": skipped,
        "failed": failed,
        "signals_refreshed": refresh_signals(config) if refreshed else 0,
    }
    logger.info("ticker_nightly_refresh: %s", summary)
    return summary


def _hpo_and_tournament(
    config: Config,
    ticker: str,
    side: str,
    n_trials: int,
    strategy: str = "fixed",
    interval: str = "1d",
) -> bool:
    """Full first-pass HPO (deep models on the GPU pool) + tournament for one ticker/side/strategy.

    Returns whether a model was promoted. Deep-model HPO runs across the GPU pool; LightGBM is
    searched in-process by the tournament's own best-params lookup. Raises on hard failure so
    the caller can count it. ``interval="1h"`` trains the intraday namespace (own studies/registry).
    """
    from berich.training.gpu_pool import GpuTask, run_on_gpus  # noqa: PLC0415
    from berich.training.tournament import train_ticker_tournament  # noqa: PLC0415

    deep_models = [m for m in config.zoo.ticker_tournament_models if m != "lgbm"]
    # LightGBM HPO runs in-process (CPU, fast); deep models go on the GPU pool.
    if "lgbm" in config.zoo.ticker_tournament_models:
        _ticker_hpo_task(None, config, ticker, "lgbm", side, n_trials, strategy, interval)
    tasks = [
        GpuTask(
            _ticker_hpo_task,
            args=(config, ticker, model_name, side, n_trials, strategy, interval),
            label=f"{ticker}/{model_name}/{side}/{strategy}/{interval}",
        )
        for model_name in deep_models
    ]
    if tasks:
        run_on_gpus(tasks, config.zoo.gpu_ids)
    return train_ticker_tournament(
        config, ticker, side, strategy=strategy, interval=interval, calibrate=True
    ).promoted


def _pending_hpo_targets(config: Config) -> list[tuple[str, str, str]]:
    """(ticker, side, strategy) triples that have no per-asset HPO study yet, new arrivals first.

    This is the queue's work-list: (asset, exit strategy) pairs whose models still run on default
    params. Reusing the status scanner's per-strategy trial counts means the queue is naturally
    resumable — a triple drops off as soon as its study has trials, so re-running continues where
    it left off, and every strategy gets its own first deep HPO.

    **New arrivals jump the queue.** A ticker with *zero* HPO trials across every side+strategy has
    never been searched (e.g. just added to the universe), so all its triples are emitted ahead of
    those for partially-searched assets. Config order is preserved within each group, so onboarding
    fresh tickers never starves a half-done asset of its remaining strategies — it just defers them.
    """
    from berich.training.status import _hpo_trial_counts, _hpo_trials_for  # noqa: PLC0415

    counts = _hpo_trial_counts(config.optuna_db)
    sides = config.zoo.ticker_sides
    strategies = config.zoo.ticker_exit_strategies

    def is_new_arrival(ticker: str) -> bool:
        return all(_hpo_trials_for(counts, ticker, None, side) == 0 for side in sides)

    pending = [
        (ticker, side, strategy)
        for ticker in config.tradeable_tickers()
        for side in sides
        for strategy in strategies
        if _hpo_trials_for(counts, ticker, None, side, strategy) == 0
    ]
    pending.sort(key=lambda triple: not is_new_arrival(triple[0]))  # stable: new arrivals first
    return pending


def refresh_signals(config: Config) -> int:
    """Regenerate + persist today's signals so the dashboard matches the current model set.

    Replaces the whole table (assets no longer optimized disappear; newly promoted ones appear)
    atomically in a single connection + transaction, then CHECKPOINTs so other processes see
    the new state immediately. Returns the count written. Best-effort: never raises.
    """
    import duckdb  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from berich.signals.store import _INSERT_COLUMNS  # noqa: PLC0415

    try:
        store = OhlcvStore(config.ohlcv_dir)
        sigs = generate_signals(config, store)
        SignalStore(config.db_path)  # ensure schema/migrations exist
        rows = pd.DataFrame([s.as_row() for s in sigs])
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        with duckdb.connect(str(config.db_path)) as con:
            con.register("incoming", rows)
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM signals")
            con.execute(
                f"INSERT INTO signals ({_INSERT_COLUMNS}) "  # noqa: S608 — module constant
                f"SELECT {_INSERT_COLUMNS} FROM incoming"
            )
            con.execute("COMMIT")
            con.execute("CHECKPOINT")
    except Exception:  # noqa: BLE001 — a refresh failure must not abort the calling job
        logger.warning("refresh_signals failed", exc_info=True)
        return 0
    return len(sigs)


def refresh_signals_job(config: Config) -> dict[str, int]:
    """Periodically rewrite the served signals from the current promoted/optimized model set.

    The per-asset HPO queue promotes models throughout the day, but the signals table was only
    rewritten by the daily chain (22:30) and right after a queue run — so a promotion landing
    between those left /signals stale vs /training. This job closes that gap on a short cron.
    """
    config.apply_active_risk_profile()  # honor a UI risk-profile switch without a restart
    n = refresh_signals(config)
    logger.info("refresh_signals_job: %d signals written", n)
    return {"signals": n}


def ticker_hpo_queue_job(config: Config, *, max_assets: int = 1) -> dict[str, object]:
    """Sequential first-HPO queue: optimize the next ``max_assets`` un-searched ticker x sides.

    Processes assets **one at a time** (no two deep HPO searches overlap), so the GPUs aren't
    swamped by the all-at-once weekend sweep. Only touches pairs with no per-asset Optuna study
    yet, and is resumable: each call drains the next slice of the work-list. Run it on a short
    cron (or in a loop) to give every asset its first deep HPO without a thundering herd.
    """
    lock = acquire_hpo_lock(config)
    if lock is None:
        logger.info("ticker_hpo_queue: HPO lock held (sweep service running), skipping this round")
        return {"processed": [], "promoted": 0, "failed": 0, "skipped": "locked"}
    n_trials = config.zoo.ticker_initial_hpo_trials
    pending = _pending_hpo_targets(config)
    processed: list[str] = []
    promoted = 0
    failed = 0
    try:
        for ticker, side, strategy in pending[:max_assets]:
            try:
                if _hpo_and_tournament(config, ticker, side, n_trials, strategy):
                    promoted += 1
                processed.append(f"{ticker}/{side}/{strategy}")
            except Exception:  # noqa: BLE001 — one bad asset must not stall the queue
                logger.warning(
                    "ticker_hpo_queue: %s/%s/%s failed", ticker, side, strategy, exc_info=True
                )
                failed += 1
    finally:
        os.close(lock)
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
    lock = acquire_hpo_lock(config)
    if lock is None:
        logger.info("ticker_initial_sweep: HPO lock held (sweep service running), skipping")
        return {"swept": 0, "promoted": 0, "failed": 0}
    n_trials = config.zoo.ticker_initial_hpo_trials
    swept = 0
    promoted = 0
    failed = 0
    for ticker in config.tradeable_tickers():
        for side in config.zoo.ticker_sides:
            for strategy in config.zoo.ticker_exit_strategies:
                try:
                    if _hpo_and_tournament(config, ticker, side, n_trials, strategy):
                        promoted += 1
                    swept += 1
                except Exception:  # noqa: BLE001 — one bad ticker must not abort the weekend sweep
                    logger.warning(
                        "ticker_initial_sweep: %s/%s/%s failed",
                        ticker,
                        side,
                        strategy,
                        exc_info=True,
                    )
                    failed += 1
    os.close(lock)
    # Sweep-level multiple-testing control: walk back promotions that don't survive FDR across the
    # whole batch (each tournament only corrected for its own search). Demoted models fall back to
    # the observe tier, so nothing is lost — only real-capital trust is revoked.
    from berich.training.promotion import reconcile_sweep_fdr  # noqa: PLC0415

    fdr_demoted = reconcile_sweep_fdr(config)["demoted"]
    summary = {
        "swept": swept,
        "promoted": promoted,
        "failed": failed,
        "fdr_demoted": fdr_demoted if isinstance(fdr_demoted, int) else 0,
    }
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


# A promoted asset whose last cached bar is older than this many calendar days is STALE — its
# served signals are being computed on frozen data, which is an actionable, email-worthy problem.
STALE_DATA_DAYS = 7


def ticker_drift_monitor_job(config: Config) -> dict[str, object]:
    """Weekly per-asset DATA-HEALTH watch over the promoted universe.

    Honest post-mortem of v1: distribution-based drift alerts (PSI/KS on feature windows) are
    structurally noisy on daily financial features — calendar encodings differ between any two
    windows by construction, and slow features (mom_60, SMA ratios, realized vol) are so
    autocorrelated that adjacent quarters always "drift". The first weekly email flagged 33/47
    promoted assets at 64-95% under three different calibrations: cry-wolf, not signal. Model rot
    is already covered by something stronger — the continuous sweep retrains on fresh data and the
    hardened gate + FDR demote models whose OOS edge dies.

    So this job now alerts only on ACTIONABLE data problems for promoted assets: a stale OHLCV
    cache (last bar older than ``STALE_DATA_DAYS``) or frozen prices (no movement over the recent
    window). Drift shares are still computed and logged for visibility — just never emailed.
    """
    from berich.features.build import build_features, market_reference_for  # noqa: PLC0415
    from berich.models import load_active  # noqa: PLC0415
    from berich.monitoring import feature_drift, split_reference_recent  # noqa: PLC0415
    from berich.signals.service import _optimized_tickers  # noqa: PLC0415

    store = OhlcvStore(config.ohlcv_dir)
    now = datetime.now(UTC)
    scanned = 0
    stale: list[tuple[str, int]] = []
    frozen: list[str] = []
    drift_shares: list[tuple[str, float]] = []
    for ticker in _optimized_tickers(config):
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        # Only watch assets we'd actually trade: a promoted model on any side/strategy.
        if not _has_promoted_model(config, ticker, load_active):
            continue
        scanned += 1
        age_days = (now.date() - df.index[-1].date()).days
        if age_days > STALE_DATA_DAYS:
            stale.append((ticker, age_days))
        tail = df["close"].tail(5)
        if len(tail) >= 5 and float(tail.std()) == 0.0:  # noqa: PLR2004 — a week of identical closes
            frozen.append(ticker)
        market = store.load(market_reference_for(config.asset_class_for(ticker)))
        feats = build_features(df, market=market).dropna()
        split = split_reference_recent(feats)
        if split is not None:
            report = feature_drift(*split)
            drift_shares.append((ticker, report.psi_share_drifted))

    if stale or frozen:
        _alert_data_health(stale, frozen)
    if drift_shares:
        top = sorted(drift_shares, key=lambda x: -x[1])[:5]
        logger.info(
            "ticker_drift_monitor: drift shares (logged, not emailed — see docstring): %s",
            ", ".join(f"{t}={s * 100:.0f}%" for t, s in top),
        )
    summary: dict[str, object] = {
        "scanned": scanned,
        "stale": [t for t, _ in stale],
        "frozen": frozen,
        "median_drift_share": (
            sorted(s for _, s in drift_shares)[len(drift_shares) // 2] if drift_shares else 0.0
        ),
    }
    logger.info("ticker_drift_monitor: %s", summary)
    return summary


def _has_promoted_model(config: Config, ticker: str, load_active) -> bool:  # noqa: ANN001
    """True if any (side, strategy) for ``ticker`` has a promoted (active) model."""
    for side in config.zoo.ticker_sides:
        for strategy in config.zoo.ticker_exit_strategies:
            if load_active(config.model_dir_for_ticker(ticker, side, strategy)) is not None:
                return True
    return False


def _alert_data_health(stale: list[tuple[str, int]], frozen: list[str]) -> None:
    """Best-effort email for ACTIONABLE data problems on promoted assets (stale/frozen cache)."""
    from berich.notifications.email import send_alert_email  # noqa: PLC0415

    lines = [f"  STALE  {tkr}: last bar {age} days old" for tkr, age in stale]
    lines += [f"  FROZEN {tkr}: close unchanged across the last 5 bars" for tkr in frozen]
    send_alert_email(
        subject=f"[BeRich] data health: {len(stale) + len(frozen)} promoted asset(s) to check",
        body=(
            "These promoted assets are being served from a broken/stale data feed — their signals "
            "cannot be trusted until the cache refreshes:\n\n" + "\n".join(lines) + "\n"
        ),
    )
