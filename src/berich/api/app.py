"""FastAPI application factory.

Single-user, local-first: if ``BERICH_API_KEY`` is set in the environment, every
endpoint requires a matching ``X-API-Key`` header; otherwise auth is disabled. The
app reads persisted signals and the OHLCV cache directly and runs the (cached)
backtest on demand. Heavy ML imports stay inside the backtest path so app startup is
fast.

All endpoints live under the ``/api`` prefix so the reverse-proxy can route
``/api/*`` to the API and everything else to the Next.js frontend without
collisions (see ``docs/DEPLOY.md``). ``/api/health`` is intentionally exempt
from the API-key check so the proxy can probe liveness without owning the
secret.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import NewsStore
from berich.data.store import OhlcvStore
from berich.signals import (
    SignalStore,
    compute_calibration,
    explain_signal,
    get_equity_curve,
    get_open_positions,
    get_paper_metrics,
)
from berich.signals.paper import PROMOTED_TIER, PaperStore


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("BERICH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(config_path: str = str(DEFAULT_CONFIG_PATH)) -> FastAPI:  # noqa: C901, PLR0915
    """Build the FastAPI app bound to a config file.

    Each endpoint is a tiny inner function; the routes are kept inline (rather than
    split across many APIRouters) because the surface is small enough that the
    one-file layout is easier to read than a many-file router tree. The C901
    complexity warning is the cost of that choice and is explicitly suppressed.
    """
    config = Config.load(config_path)
    store = OhlcvStore(config.ohlcv_dir)
    signal_store = SignalStore(config.db_path)

    app = FastAPI(title="BeRich API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    router = APIRouter(prefix="/api")
    guard = [Depends(_require_api_key)]

    @router.get("/health")
    def health() -> dict[str, object]:
        """Liveness + freshness probe (no auth — used by Caddy and dashboards).

        Returns the basic ``status: ok`` plus four freshness numbers a human
        can scan at a glance to tell if the scheduler is alive: last refresh
        timestamps for OHLCV / news / signals, today's signal count, and the
        number of paper positions currently open.
        """
        return {
            "status": "ok",
            "ohlcv_last_refresh": _last_refresh(store, "AAPL"),
            "news_last_refresh": _last_news_refresh(config),
            "signals_last_date": _last_signals_date(signal_store),
            "n_signals_today": _n_signals_today(signal_store),
            "n_open_positions": _n_open_positions(config),
        }

    @router.get("/watchlist", dependencies=guard)
    def watchlist() -> list[str]:
        return config.watchlist

    @router.get("/signals", dependencies=guard)
    def signals() -> list[dict]:
        return signal_store.latest().to_dict(orient="records")

    @router.get("/signals/{ticker}/history", dependencies=guard)
    def signal_history(ticker: str) -> list[dict]:
        return signal_store.history(ticker.upper()).to_dict(orient="records")

    @router.get("/signals/{ticker}/explain", dependencies=guard)
    def signal_explain(ticker: str) -> dict:
        result = explain_signal(ticker.upper(), config, store)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"no explainable signal for {ticker} (cache or model framework mismatch)",
            )
        return result

    @router.get("/training", dependencies=guard)
    def training() -> list[dict]:
        """Per-asset inventory — only assets we've optimized (HPO run), like the signals page."""
        from berich.training.status import training_status

        return training_status(config, optimized_only=True)

    @router.get("/training/{ticker}", dependencies=guard)
    def training_for_ticker(ticker: str) -> list[dict]:
        """Both sides' training/HPO state for one asset, for the ticker drill-down panel."""
        from berich.training.status import training_status

        want = ticker.upper()
        return [r for r in training_status(config) if str(r["ticker"]).upper() == want]

    @router.get("/hpo-progress", dependencies=guard)
    def hpo_progress_endpoint() -> dict:
        """Lightweight HPO sweep coverage (combo grain) for the /training progress bar.

        Just the counts — no nvidia-smi/systemctl/journald — so /training can poll it cheaply.
        """
        from berich.ops import hpo_progress

        return hpo_progress(config)

    @router.get("/ops", dependencies=guard)
    def ops() -> dict:
        """Live machine status: GPUs, scheduler jobs, HPO queue progress, recent logs."""
        from berich.ops import ops_snapshot

        return ops_snapshot(config)

    @router.get("/config", dependencies=guard)
    def signal_config() -> dict:
        """Decision thresholds the dashboard needs to explain why a signal is LONG/SHORT/NEUTRAL."""
        s = config.signals
        return {
            "buy_threshold": s.buy_threshold,
            "short_threshold": s.short_threshold,
            "enable_short": s.enable_short,
            "horizon_days": config.labeling.horizon_days,
            "take_profit_atr": config.labeling.take_profit_atr,
            "stop_loss_atr": config.labeling.stop_loss_atr,
            "max_ticker_exposure_pct": s.max_ticker_exposure_pct,
            "max_book_exposure_pct": s.max_book_exposure_pct,
            "max_class_exposure_pct": s.max_class_exposure_pct,
            "drawdown_derisk_threshold": s.drawdown_derisk_threshold,
            "drawdown_halt_threshold": s.drawdown_halt_threshold,
            "max_open_positions": s.max_open_positions,
        }

    @router.get("/universes", dependencies=guard)
    def universes() -> dict[str, list[str]]:
        return {
            "us_stocks": config.universes.us_stocks,
            "fr_stocks": config.universes.fr_stocks,
            "forex": config.universes.forex,
            "crypto": config.universes.crypto,
            "commodities": config.universes.commodities,
        }

    @router.get("/longshort/basket", dependencies=guard)
    def longshort_basket() -> list[dict]:
        from berich.signals import LongShortStore

        return LongShortStore(config.db_path).latest().to_dict(orient="records")

    @router.get("/longshort/equity", dependencies=guard)
    def longshort_equity_route() -> dict:
        from berich.signals import longshort_equity

        return longshort_equity(config, store)

    @router.get("/prices/{ticker}", dependencies=guard)
    def prices(ticker: str, days: int = Query(default=365, ge=1, le=5000)) -> list[dict]:
        df = store.load(ticker.upper())
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"no data for {ticker}")
        tail = df.tail(days).reset_index()
        tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
        return tail.to_dict(orient="records")

    @router.get("/drift", dependencies=guard)
    def drift() -> dict:
        from berich.scheduler.jobs import check_drift_job

        report = check_drift_job(config)
        return {
            "n_drifted": report.n_drifted,
            "n_features": len(report.features),
            "should_retrain": report.should_retrain,
            "features": report.to_frame().to_dict(orient="records"),
        }

    @router.get("/backtest", dependencies=guard)
    def backtest(threshold: float = Query(default=0.5, ge=0.0, le=1.0)) -> dict:
        return _run_cached_backtest(config_path, round(threshold, 3))

    @router.get("/paper/positions", dependencies=guard)
    def paper_positions(
        strategy: str | None = Query(default=None),
        tier: str = Query(default=PROMOTED_TIER),
    ) -> dict:
        positions = get_open_positions(config, store, exit_strategy=strategy, tier=tier)
        return {
            "n": len(positions),
            "positions": [p.as_row() for p in positions],
        }

    @router.get("/paper/equity", dependencies=guard)
    def paper_equity(
        strategy: str | None = Query(default=None),
        tier: str = Query(default=PROMOTED_TIER),
    ) -> dict:
        curve = get_equity_curve(config, store, exit_strategy=strategy, tier=tier)
        metrics = get_paper_metrics(config, store, exit_strategy=strategy, tier=tier)
        if curve.empty:
            return {"dates": [], "equity_paper": [], "equity_spy": [], "metrics": metrics}
        return {
            "dates": curve["date"].tolist(),
            "equity_paper": curve["equity_paper"].tolist(),
            "equity_spy": [None if pd.isna(v) else float(v) for v in curve["equity_spy"]],
            "metrics": metrics,
        }

    @router.get("/paper/closed-trades", dependencies=guard)
    def paper_closed(
        limit: int = Query(default=50, ge=1, le=500),
        strategy: str | None = Query(default=None),
        tier: str = Query(default=PROMOTED_TIER),
    ) -> list[dict]:
        df = PaperStore(config.db_path).closed_trades(
            limit=limit, exit_strategy=strategy, tier=tier
        )
        if df.empty:
            return []
        df = df.copy()
        for col in ("date_open", "date_close"):
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        for col in ("created_at", "updated_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return df.to_dict(orient="records")

    @router.get("/paper/calibration", dependencies=guard)
    def paper_calibration() -> dict:
        report = compute_calibration(config)
        return {
            "n_trades_total": report.n_trades_total,
            "n_with_proba": report.n_with_proba,
            "is_well_calibrated": report.is_well_calibrated,
            "buckets": [b.as_row() for b in report.buckets],
        }

    @router.get("/paper/export.csv", dependencies=guard)
    def paper_export_csv() -> Response:
        df = PaperStore(config.db_path).all_trades()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=paper_trades.csv"},
        )

    app.include_router(router)
    return app


def _last_refresh(store: OhlcvStore, ticker: str) -> str | None:
    """ISO date of the most recent OHLCV bar for ``ticker`` (proxy for cache freshness)."""
    last = store.last_date(ticker)
    if last is None:
        return None
    # ``last_date`` returns a real Timestamp; the cast collapses the NaTType
    # branch ty insists on adding.
    return pd.Timestamp(last).date().isoformat()  # ty: ignore[unresolved-attribute]


def _last_news_refresh(config: Config) -> str | None:
    """Most recent ``time_published`` across the news cache, or ``None`` if no cache."""
    store = NewsStore(config.news_dir)
    if not store.has_any_data():
        return None
    latest: pd.Timestamp | None = None
    for ticker in config.watchlist:
        ts = store.last_time(ticker)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return None
    # ``last_time`` already guarantees a real Timestamp; the cast collapses
    # ty's NaTType union without a runtime branch.
    return pd.Timestamp(latest).isoformat()  # ty: ignore[unresolved-attribute]


def _last_signals_date(signal_store: SignalStore) -> str | None:
    latest = signal_store.latest()
    if latest.empty:
        return None
    return str(latest["date"].iloc[0])


def _n_signals_today(signal_store: SignalStore) -> int:
    latest = signal_store.latest()
    return len(latest)


def _n_open_positions(config: Config) -> int:
    return len(PaperStore(config.db_path).open_trades())


@lru_cache(maxsize=8)
def _run_cached_backtest(config_path: str, threshold: float) -> dict:
    """Run a walk-forward backtest, memoized by (config, threshold)."""
    from berich.backtest import BacktestConfig, run_backtest
    from berich.datasets import build_dataset
    from berich.labeling.triple_barrier import LabelConfig
    from berich.models import LGBMModel
    from berich.training import oof_predict

    config = Config.load(config_path)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    dataset = build_dataset(store, config.watchlist, label_cfg)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt_cfg = BacktestConfig(
        entry_threshold=threshold,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
    )
    result = run_backtest(prices, oof, bt_cfg)
    equity_dates = pd.DatetimeIndex(result.strategy_returns.index).strftime("%Y-%m-%d").tolist()
    return {
        "auc": oof.auc,
        "strategy": result.strategy.as_dict(),
        "benchmark": result.benchmark.as_dict(),
        "beats_buy_hold": result.beats_buy_hold,
        "equity": {
            "dates": equity_dates,
            "strategy": (1 + result.strategy_returns.fillna(0)).cumprod().tolist(),
            "benchmark": (1 + result.benchmark_returns.fillna(0)).cumprod().tolist(),
        },
    }
