"""FastAPI application factory.

Single-user, local-first: if ``BERICH_API_KEY`` is set in the environment, every
endpoint requires a matching ``X-API-Key`` header; otherwise auth is disabled. The
app reads persisted signals and the OHLCV cache directly and runs the (cached)
backtest on demand. Heavy ML imports stay inside the backtest path so app startup is
fast.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.signals import SignalStore


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("BERICH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(config_path: str = str(DEFAULT_CONFIG_PATH)) -> FastAPI:
    """Build the FastAPI app bound to a config file."""
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
    guard = [Depends(_require_api_key)]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/watchlist", dependencies=guard)
    def watchlist() -> list[str]:
        return config.watchlist

    @app.get("/signals", dependencies=guard)
    def signals() -> list[dict]:
        return signal_store.latest().to_dict(orient="records")

    @app.get("/signals/{ticker}/history", dependencies=guard)
    def signal_history(ticker: str) -> list[dict]:
        return signal_store.history(ticker.upper()).to_dict(orient="records")

    @app.get("/prices/{ticker}", dependencies=guard)
    def prices(ticker: str, days: int = Query(default=365, ge=1, le=5000)) -> list[dict]:
        df = store.load(ticker.upper())
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"no data for {ticker}")
        tail = df.tail(days).reset_index()
        tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
        return tail.to_dict(orient="records")

    @app.get("/drift", dependencies=guard)
    def drift() -> dict:
        from berich.scheduler.jobs import check_drift_job

        report = check_drift_job(config)
        return {
            "n_drifted": report.n_drifted,
            "n_features": len(report.features),
            "should_retrain": report.should_retrain,
            "features": report.to_frame().to_dict(orient="records"),
        }

    @app.get("/backtest", dependencies=guard)
    def backtest(threshold: float = Query(default=0.5, ge=0.0, le=1.0)) -> dict:
        return _run_cached_backtest(config_path, round(threshold, 3))

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
