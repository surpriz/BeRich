"""Deep / zoo model training driver: walk-forward OOS + backtest + MLflow logging.

This is the shared entry point for every non-LightGBM model (LSTM, PatchTST, TFT,
N-HiTS, ensembles). It reuses the existing walk-forward harness (:func:`oof_predict`)
and event-based backtester so the guard rule applies identically to the LightGBM
baseline — a model is only promoted if its OOS Sharpe beats both LightGBM and buy & hold.

The MLflow run captures (a) the hyperparameters that produced the verdict and
(b) the verdict itself, so an unsuccessful experiment is still legible later.
The active artifact lives in the on-disk model registry; MLflow is the
experiment log, not the serving store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlflow

from berich.backtest import BacktestConfig, run_backtest
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.features.build import FEATURE_COLUMNS
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, LSTMConfig, LSTMModel, ModelMetadata, promote, save_model
from berich.training import oof_predict

if TYPE_CHECKING:
    from collections.abc import Callable

    from berich.config import Config
    from berich.models.base import Model

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


def baseline_sharpe(config: Config) -> float:
    """Re-run the LightGBM baseline OOS + backtest to capture today's Sharpe-to-beat.

    Shared by the CLI, the scripts, and the nightly retrain job so every zoo candidate
    is gated against the same freshly-computed LightGBM bar.
    """
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))
    logger.info(
        "LightGBM baseline: AUC=%.4f Sharpe=%.3f vs B&H Sharpe=%.3f",
        oof.auc,
        bt.strategy.sharpe,
        bt.benchmark.sharpe,
    )
    return bt.strategy.sharpe


def train_deep_model(
    config: Config,
    *,
    name: str,
    framework: str,
    model_factory: Callable[[], Model],
    hyperparams: dict[str, Any] | None = None,
    baseline_sharpe: float | None = None,
    entry_threshold: float = 0.5,
    force_promote: bool = False,
    promote_if_passes: bool = True,
) -> DeepTrainResult:
    """Train and evaluate any model under the same OOS + backtest discipline as LightGBM.

    Args:
        config: project config (data dirs, watchlist, labeling, threshold).
        name: artifact name for the registry.
        framework: free-form framework tag stored in the metadata (e.g. ``"lstm"``).
        model_factory: zero-arg callable returning a *fresh* model. Called once per
            walk-forward fold for OOS, then once more for the final all-history fit.
        hyperparams: hyperparameters to log to MLflow (defaults to ``{}``).
        baseline_sharpe: LightGBM OOS Sharpe to beat. The model is only promoted if it
            strictly improves on this AND on buy & hold.
        entry_threshold: P(win) threshold for the backtest.
        force_promote: bypass the guard rule (use with extreme care).
    """
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    logger.info("dataset: %d samples x %d features", len(dataset), len(FEATURE_COLUMNS))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(hyperparams or {})
        mlflow.log_param("framework", framework)
        mlflow.log_param("device", str(getattr(model_factory(), "device", "cpu")))
        mlflow.log_param("entry_threshold", entry_threshold)
        mlflow.log_param("watchlist", ",".join(config.watchlist))
        mlflow.log_param("n_samples", len(dataset))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        oof = oof_predict(dataset, model_factory, embargo=label_cfg.horizon_days)
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
        final = model_factory().fit(
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

        best_val_auc = float(getattr(final, "best_val_auc", float("nan")))
        meta = ModelMetadata(
            name=name,
            framework=framework,
            feature_columns=FEATURE_COLUMNS,
            metrics={
                "auc": oos_auc,
                "sharpe": bt.strategy.sharpe,
                "benchmark_sharpe": bt.benchmark.sharpe,
                "baseline_sharpe": float("nan") if baseline_sharpe is None else baseline_sharpe,
            },
            beats_buy_hold=bt.beats_buy_hold,
            notes=f"beats_baseline={beats_baseline}; best_val_auc={best_val_auc:.4f}",
        )
        artifact_dir = save_model(final, meta, registry_dir=config.models_dir)
        # joblib pickle of the model (including torch state) lives in the registry —
        # log it as an MLflow artifact too so the experiment row is self-contained.
        mlflow.log_artifacts(str(artifact_dir), artifact_path="registry")

        promoted = False
        if promote_if_passes and (promotable or force_promote):
            try:
                promote(name, registry_dir=config.models_dir, force=force_promote)
                promoted = True
            except ValueError as exc:
                logger.warning("promotion blocked: %s", exc)
        else:
            logger.info(
                "not promoting (promote_if_passes=%s): beats_buy_hold=%s beats_baseline=%s",
                promote_if_passes,
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


def train_lstm(
    config: Config,
    *,
    name: str = "lstm-baseline",
    lstm_cfg: LSTMConfig | None = None,
    baseline_sharpe: float | None = None,
    entry_threshold: float = 0.5,
    force_promote: bool = False,
) -> DeepTrainResult:
    """Back-compat wrapper: train the LSTM through the generic :func:`train_deep_model`."""
    cfg = lstm_cfg or LSTMConfig()
    return train_deep_model(
        config,
        name=name,
        framework="lstm",
        model_factory=lambda: LSTMModel(cfg),
        hyperparams=cfg.as_dict(),
        baseline_sharpe=baseline_sharpe,
        entry_threshold=entry_threshold,
        force_promote=force_promote,
    )
