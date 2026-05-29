"""Train a dedicated advisory model for one non-US asset class.

Shared by the CLI (`berich train --asset-class ...`) and the nightly scheduler so both use
the same path: build the class panel with its own regime proxy, walk-forward OOS + backtest,
save to the per-class registry namespace, and force-promote (advisory — the dashboard flags
these universes experimental; the returned metrics state whether it beat the class B&H).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from berich.backtest import BacktestConfig, run_backtest
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.features.build import feature_columns, market_reference_for
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.training import oof_predict

if TYPE_CHECKING:
    from berich.config import Config

logger = logging.getLogger(__name__)


def train_asset_class_model(
    config: Config,
    asset_class: str,
    *,
    name: str | None = None,
) -> dict[str, object]:
    """Train + force-promote a base-22 model for ``asset_class``; return a metrics dict."""
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    tickers = config.universes.get(asset_class)
    if not tickers:
        return {"asset_class": asset_class, "trained": False, "reason": "no tickers configured"}

    market_ticker = market_reference_for(asset_class)
    dataset = build_dataset(store, tickers, label_cfg, market_ticker=market_ticker)
    if not len(dataset):
        return {"asset_class": asset_class, "trained": False, "reason": "no cached data"}

    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in tickers if (df := store.load(t)) is not None and not df.empty}
    result = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))
    model = LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)

    final_name = name or f"{asset_class}-lgbm"
    registry_dir = config.models_dir_for(asset_class)
    meta = ModelMetadata(
        name=final_name,
        framework="lightgbm",
        feature_columns=feature_columns(),
        metrics={
            "auc": oof.auc,
            "sharpe": result.strategy.sharpe,
            "benchmark_sharpe": result.benchmark.sharpe,
        },
        beats_buy_hold=result.beats_buy_hold,
        notes=f"dedicated {asset_class} model (advisory; market={market_ticker})",
    )
    save_model(model, meta, registry_dir=registry_dir)
    promote(final_name, registry_dir=registry_dir, force=True)  # advisory — UI flags experimental
    logger.info(
        "trained %s model '%s': AUC=%.4f Sharpe=%.3f vs class B&H %.3f (beats_bh=%s)",
        asset_class, final_name, oof.auc, result.strategy.sharpe,
        result.benchmark.sharpe, result.beats_buy_hold,
    )
    return {
        "asset_class": asset_class,
        "name": final_name,
        "n_tickers": len(tickers),
        "auc": oof.auc,
        "sharpe": result.strategy.sharpe,
        "benchmark_sharpe": result.benchmark.sharpe,
        "beats_buy_hold": result.beats_buy_hold,
        "trained": True,
    }


__all__ = ["train_asset_class_model"]
