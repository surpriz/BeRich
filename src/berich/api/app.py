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

import os
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.signals import (
    SignalStore,
    get_equity_curve,
    get_open_positions,
    get_paper_metrics,
)
from berich.signals.paper import PaperStore


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("BERICH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(config_path: str = str(DEFAULT_CONFIG_PATH)) -> FastAPI:  # noqa: C901
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
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/watchlist", dependencies=guard)
    def watchlist() -> list[str]:
        return config.watchlist

    @router.get("/signals", dependencies=guard)
    def signals() -> list[dict]:
        return signal_store.latest().to_dict(orient="records")

    @router.get("/signals/{ticker}/history", dependencies=guard)
    def signal_history(ticker: str) -> list[dict]:
        return signal_store.history(ticker.upper()).to_dict(orient="records")

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
    def paper_positions() -> dict:
        positions = get_open_positions(config, store)
        return {
            "n": len(positions),
            "positions": [p.as_row() for p in positions],
        }

    @router.get("/paper/equity", dependencies=guard)
    def paper_equity() -> dict:
        curve = get_equity_curve(config, store)
        metrics = get_paper_metrics(config, store)
        if curve.empty:
            return {"dates": [], "equity_paper": [], "equity_spy": [], "metrics": metrics}
        return {
            "dates": curve["date"].tolist(),
            "equity_paper": curve["equity_paper"].tolist(),
            "equity_spy": [None if pd.isna(v) else float(v) for v in curve["equity_spy"]],
            "metrics": metrics,
        }

    @router.get("/paper/closed-trades", dependencies=guard)
    def paper_closed(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
        df = PaperStore(config.db_path).closed_trades(limit=limit)
        if df.empty:
            return []
        df = df.copy()
        for col in ("date_open", "date_close"):
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
        for col in ("created_at", "updated_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return df.to_dict(orient="records")

    app.include_router(router)
    return app


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
