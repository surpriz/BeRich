"""Deep-model training driver: walk-forward OOS + backtest + MLflow logging.

This is the Phase 3 entry point for sequence models. It reuses the existing
walk-forward harness (:func:`oof_predict`) and event-based backtester so the
guard rule applies to deep models identically to the LightGBM baseline — a model
is only promoted if its OOS Sharpe beats both LightGBM and buy & hold.

The MLflow run captures (a) the hyperparameters that produced the verdict and
(b) the verdict itself, so an unsuccessful experiment is still legible later.
The active artifact lives in the on-disk model registry; MLflow is the
experiment log, not the serving store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow

from berich.backtest import BacktestConfig, run_backtest
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.features.build import FEATURE_COLUMNS
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LSTMConfig, LSTMModel, ModelMetadata, promote, save_model
from berich.training import oof_predict

if TYPE_CHECKING:
    from berich.config import Config

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "berich-deep"


@dataclass
class DeepTrainResult:
    """What a deep-training run produced — used for logging and promotion gating."""

    name: str
    oos_auc: float
    strategy_sharpe: float
    benchmark_sharpe: float
    baseline_sharpe: float
    beats_buy_hold: bool
    beats_baseline: bool
    promoted: bool


def train_lstm(
    config: Config,
    *,
    name: str = "lstm-baseline",
    lstm_cfg: LSTMConfig | None = None,
    baseline_sharpe: float | None = None,
    entry_threshold: float = 0.5,
    force_promote: bool = False,
) -> DeepTrainResult:
    """Train and evaluate the LSTM under the same OOS + backtest discipline as LightGBM.

    Args:
        config: project config (data dirs, watchlist, labeling, threshold).
        name: artifact name for the registry.
        lstm_cfg: hyperparameters; defaults from :class:`LSTMConfig` if omitted.
        baseline_sharpe: LightGBM OOS Sharpe to beat. The model is only promoted if
            it strictly improves on this AND on buy & hold.
        entry_threshold: P(win) threshold for the backtest.
        force_promote: bypass the guard rule (use with extreme care).
    """
    cfg = lstm_cfg or LSTMConfig()
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    logger.info("dataset: %d samples x %d features", len(dataset), len(FEATURE_COLUMNS))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(cfg.as_dict())
        mlflow.log_param("device", str(LSTMModel(cfg).device))
        mlflow.log_param("entry_threshold", entry_threshold)
        mlflow.log_param("watchlist", ",".join(config.watchlist))
        mlflow.log_param("n_samples", len(dataset))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        oof = oof_predict(
            dataset,
            lambda: LSTMModel(cfg),
            embargo=label_cfg.horizon_days,
        )
        oos_auc = oof.auc
        logger.info("OOS AUC: %.4f", oos_auc)

        prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
        bt_cfg = BacktestConfig(
            entry_threshold=entry_threshold,
            horizon_days=label_cfg.horizon_days,
            atr_window=label_cfg.atr_window,
            take_profit_atr=label_cfg.take_profit_atr,
            stop_loss_atr=label_cfg.stop_loss_atr,
        )
        bt = run_backtest(prices, oof, bt_cfg)

        # Final model on all labeled history for serving (registry artifact).
        final = LSTMModel(cfg).fit(
            dataset.x,
            dataset.y,
            sample_weight=dataset.weight,
            tickers=dataset.tickers,
        )
        beats_baseline = baseline_sharpe is None or bt.strategy.sharpe > baseline_sharpe
        promotable = bt.beats_buy_hold and beats_baseline

        mlflow.log_metric("oos_auc", oos_auc)
        mlflow.log_metric("strategy_sharpe", bt.strategy.sharpe)
        mlflow.log_metric("benchmark_sharpe", bt.benchmark.sharpe)
        mlflow.log_metric("strategy_cagr", bt.strategy.cagr)
        mlflow.log_metric("strategy_total_return", bt.strategy.total_return)
        mlflow.log_metric("strategy_max_drawdown", bt.strategy.max_drawdown)
        mlflow.log_metric("strategy_n_trades", bt.strategy.n_trades)
        if baseline_sharpe is not None:
            mlflow.log_metric("baseline_sharpe", baseline_sharpe)
        mlflow.log_metric("beats_buy_hold", float(bt.beats_buy_hold))
        mlflow.log_metric("beats_baseline", float(beats_baseline))

        meta = ModelMetadata(
            name=name,
            framework="lstm",
            feature_columns=FEATURE_COLUMNS,
            metrics={
                "auc": oos_auc,
                "sharpe": bt.strategy.sharpe,
                "benchmark_sharpe": bt.benchmark.sharpe,
                "baseline_sharpe": float("nan") if baseline_sharpe is None else baseline_sharpe,
            },
            beats_buy_hold=bt.beats_buy_hold,
            notes=f"beats_baseline={beats_baseline}; best_val_auc={final.best_val_auc:.4f}",
        )
        artifact_dir = save_model(final, meta, registry_dir=config.models_dir)
        # joblib pickle of the LSTM (including torch state) lives in the registry —
        # log it as an MLflow artifact too so the experiment row is self-contained.
        mlflow.log_artifacts(str(artifact_dir), artifact_path="registry")

        promoted = False
        if promotable or force_promote:
            try:
                promote(name, registry_dir=config.models_dir, force=force_promote)
                promoted = True
            except ValueError as exc:
                logger.warning("promotion blocked: %s", exc)
        else:
            logger.info(
                "not promoting: beats_buy_hold=%s beats_baseline=%s",
                bt.beats_buy_hold,
                beats_baseline,
            )

        mlflow.log_metric("promoted", float(promoted))

        return DeepTrainResult(
            name=name,
            oos_auc=oos_auc,
            strategy_sharpe=bt.strategy.sharpe,
            benchmark_sharpe=bt.benchmark.sharpe,
            baseline_sharpe=float("nan") if baseline_sharpe is None else baseline_sharpe,
            beats_buy_hold=bt.beats_buy_hold,
            beats_baseline=beats_baseline,
            promoted=promoted,
        )
