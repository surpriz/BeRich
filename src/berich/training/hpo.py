"""Optuna hyperparameter search over the walk-forward OOS Sharpe objective.

The objective for any zoo model is the same metric the guard rule cares about: out-of-sample
strategy Sharpe (``oof_predict`` -> ``run_backtest``). Trials are coordinated through an
SQLite RDB so several GPU workers can pull from one study concurrently; the best params then
feed the nightly final-fit + guard-gated promotion. MLflow records the search summary.

This module keeps imports of the heavy model classes lazy so importing it (e.g. from the
scheduler) doesn't pull torch on the fast paths.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from berich.backtest import BacktestConfig, run_backtest
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.training import oof_predict

if TYPE_CHECKING:
    from collections.abc import Callable

    import optuna

    from berich.config import Config
    from berich.datasets.assemble import SupervisedDataset
    from berich.models.base import Model

logger = logging.getLogger(__name__)

# Feature-engineering search: the HPO can toggle whole feature families on/off (jointly with
# model hyperparameters) to find which engineered groups actually help. Names match
# FEATURE_COLUMNS; any column not in a group (e.g. earnings/news add-ons) is always kept.
FEATURE_GROUPS: dict[str, list[str]] = {
    "momentum": ["ret_1", "ret_5", "mom_10", "mom_20", "mom_60", "mom_120"],
    "oscillators": ["rsi_14", "macd", "macd_signal", "macd_hist"],
    "volatility": ["atr_pct", "rvol_20"],
    "trend": ["close_sma20_ratio", "close_sma50_ratio", "dist_high_60", "dist_low_60"],
    "volume": ["volume_z20"],
    "calendar": ["month_sin", "month_cos", "days_to_month_end"],
    "market_regime": ["spy_ret_20", "spy_rvol_20"],
}


def _select_features(trial: optuna.Trial, available: list[str]) -> list[str]:
    """Trial-driven feature subset: each group is toggled on/off; extras always kept."""
    grouped = {c for cols in FEATURE_GROUPS.values() for c in cols}
    chosen: set[str] = set()
    for group, cols in FEATURE_GROUPS.items():
        if trial.suggest_categorical(f"feat_{group}", [True, False]):
            chosen.update(cols)
    chosen.update(c for c in available if c not in grouped)  # earnings/news add-ons, etc.
    return [c for c in available if c in chosen]

SUPPORTED_MODELS = ("lgbm", "lstm", "patchtst", "tft")


def _factory_from_trial(
    model_name: str, trial: optuna.Trial, *, device: str | None
) -> Callable[[], Model]:
    """Build a zero-arg model factory from a trial's suggested hyperparameters."""
    if model_name == "lgbm":
        from berich.models import LGBMModel  # noqa: PLC0415

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        return lambda: LGBMModel(**params)
    if model_name == "lstm":
        from berich.models import LSTMConfig, LSTMModel  # noqa: PLC0415

        cfg = LSTMConfig(
            lookback=trial.suggest_int("lookback", 20, 80, step=10),
            hidden=trial.suggest_categorical("hidden", [32, 64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            epochs=trial.suggest_int("epochs", 20, 60, step=10),
            device=device,
        )
        return lambda: LSTMModel(cfg)
    if model_name == "patchtst":
        from berich.models import PatchTSTConfig, PatchTSTModel  # noqa: PLC0415

        cfg = PatchTSTConfig(
            lookback=trial.suggest_int("lookback", 24, 80, step=8),
            patch_len=trial.suggest_categorical("patch_len", [8, 12, 16]),
            stride=trial.suggest_categorical("stride", [4, 6, 8]),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            num_layers=trial.suggest_int("num_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            epochs=trial.suggest_int("epochs", 20, 60, step=10),
            device=device,
        )
        return lambda: PatchTSTModel(cfg)
    if model_name == "tft":
        from berich.models import TFTConfig, TFTModel  # noqa: PLC0415

        cfg = TFTConfig(
            lookback=trial.suggest_int("lookback", 24, 80, step=8),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            epochs=trial.suggest_int("epochs", 20, 60, step=10),
            device=device,
        )
        return lambda: TFTModel(cfg)
    msg = f"unknown model '{model_name}' (expected one of {SUPPORTED_MODELS})"
    raise ValueError(msg)


def objective_for(
    model_name: str,
    dataset: SupervisedDataset,
    label_cfg: LabelConfig,
    prices: dict,
    *,
    device: str | None = None,
    entry_threshold: float = 0.5,
    search_features: bool = True,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective: maximize out-of-sample strategy Sharpe.

    When ``search_features`` is set, the trial also toggles feature families on/off so the
    search optimizes the feature set jointly with the model hyperparameters.
    """

    def objective(trial: optuna.Trial) -> float:
        ds = dataset
        if search_features:
            cols = _select_features(trial, list(dataset.x.columns))
            if not cols:
                return 0.0  # degenerate empty feature set
            ds = replace(dataset, x=dataset.x[cols])
            trial.set_user_attr("features", cols)
        factory: Callable[[], Model] = _factory_from_trial(model_name, trial, device=device)
        oof = oof_predict(ds, factory, embargo=label_cfg.horizon_days)
        bt = run_backtest(
            prices,
            oof,
            BacktestConfig(
                entry_threshold=entry_threshold,
                horizon_days=label_cfg.horizon_days,
                atr_window=label_cfg.atr_window,
                take_profit_atr=label_cfg.take_profit_atr,
                stop_loss_atr=label_cfg.stop_loss_atr,
            ),
        )
        trial.set_user_attr("oos_auc", oof.auc)
        trial.set_user_attr("sharpe", bt.strategy.sharpe)
        return bt.strategy.sharpe

    return objective


def run_hpo(
    config: Config,
    model_name: str,
    *,
    n_trials: int = 20,
    device: str | None = None,
    study_name: str | None = None,
) -> optuna.Study:
    """Run an Optuna study for one model and log the summary to MLflow.

    Uses the project's SQLite RDB so concurrent GPU workers can share the study
    (``load_if_exists=True``). Returns the completed study (``best_params`` / ``best_value``).
    """
    import optuna  # noqa: PLC0415

    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}

    storage = f"sqlite:///{config.optuna_db}"
    name = study_name or f"berich-hpo-{model_name}"
    config.optuna_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize", storage=storage, study_name=name, load_if_exists=True
    )
    objective = objective_for(model_name, dataset, label_cfg, prices, device=device)
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "HPO %s best Sharpe=%.3f params=%s", model_name, study.best_value, study.best_params
    )
    _log_to_mlflow(model_name, study)
    return study


LONGSHORT_MODELS = ("lgbm", "patchtst", "lstm", "tft")


def _ranker_factory_from_trial(
    model_name: str, trial: optuna.Trial, *, device: str | None
) -> Callable[[], Model]:
    """Build a cross-sectional ranker factory from a trial's hyperparameters."""
    if model_name == "lgbm":
        from berich.models import LGBMRanker  # noqa: PLC0415

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
        }
        return lambda: LGBMRanker(**params)
    if model_name == "patchtst":
        from berich.models import PatchTSTConfig, PatchTSTRanker  # noqa: PLC0415

        cfg = PatchTSTConfig(
            lookback=trial.suggest_int("lookback", 24, 80, step=8),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            num_layers=trial.suggest_int("num_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            device=device,
        )
        return lambda: PatchTSTRanker(cfg)
    if model_name == "lstm":
        from berich.models import LSTMConfig, LSTMRanker  # noqa: PLC0415

        cfg = LSTMConfig(
            lookback=trial.suggest_int("lookback", 20, 80, step=10),
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            device=device,
        )
        return lambda: LSTMRanker(cfg)
    if model_name == "tft":
        from berich.models import TFTConfig, TFTRanker  # noqa: PLC0415

        cfg = TFTConfig(
            lookback=trial.suggest_int("lookback", 24, 80, step=8),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            device=device,
        )
        return lambda: TFTRanker(cfg)
    msg = f"unknown ranker '{model_name}' (expected one of {LONGSHORT_MODELS})"
    raise ValueError(msg)


def run_longshort_hpo(
    config: Config,
    model_name: str,
    *,
    n_trials: int = 20,
    device: str | None = None,
) -> optuna.Study:
    """Optuna search for the long/short ranker, maximizing out-of-sample net Sharpe."""
    import optuna  # noqa: PLC0415

    from berich.backtest.longshort import LongShortConfig, run_longshort_backtest  # noqa: PLC0415
    from berich.datasets.cross_sectional import build_panel_dataset  # noqa: PLC0415
    from berich.labeling.cross_sectional import CrossSectionalLabelConfig  # noqa: PLC0415
    from berich.training.cross_sectional import oof_predict_cross_sectional  # noqa: PLC0415

    ls = config.longshort
    store = OhlcvStore(config.ohlcv_dir)
    tickers = config.tickers_for_universe(ls.universe)
    label_cfg = CrossSectionalLabelConfig(
        horizon_days=ls.horizon_days,
        beta_window=ls.beta_window,
        residualize=ls.residualize,
        standardize="rank" if ls.standardize == "rank" else "zscore",
    )
    panel = build_panel_dataset(
        store,
        tickers,
        label_cfg,
        market_ticker=ls.market_ticker,
        min_names_per_date=ls.min_names_per_date,
        cross_sectional=ls.cross_sectional_features,
    )
    prices = {t: df for t in tickers if (df := store.load(t)) is not None}
    bt_cfg = LongShortConfig(
        top_decile=ls.top_decile,
        bottom_decile=ls.bottom_decile,
        weighting=ls.weighting,
        rebalance_days=ls.rebalance_days,
        gross_leverage=ls.gross_leverage,
        target_vol=ls.target_vol,
        vol_lookback=ls.vol_lookback,
        fee_bps=ls.fee_bps,
        slippage_bps=ls.slippage_bps,
        borrow_bps_annual=ls.borrow_bps_annual,
        min_names=ls.min_names_per_date,
    )

    def objective(trial: optuna.Trial) -> float:
        factory = _ranker_factory_from_trial(model_name, trial, device=device)
        oof = oof_predict_cross_sectional(panel, factory, embargo=ls.horizon_days)
        res = run_longshort_backtest(prices, oof, bt_cfg, n_trials=ls.n_trials)
        trial.set_user_attr("rank_ic", oof.rank_ic)
        trial.set_user_attr("sharpe", res.significance.sharpe)
        return res.significance.sharpe

    storage = f"sqlite:///{config.optuna_db}"
    config.optuna_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=f"berich-longshort-{model_name}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    logger.info("longshort HPO %s best Sharpe=%.3f", model_name, study.best_value)
    return study


def best_params_for(config: Config, model_name: str, *, study_prefix: str = "berich-hpo") -> dict:
    """Best model hyperparameters from the latest HPO study (feature toggles excluded).

    Returns ``{}`` if no study exists yet, so the nightly retrain falls back to defaults.
    The ``feat_*`` feature-selection keys are dropped — production retrain keeps the full
    feature set for serving consistency (the search result is logged for inspection).
    """
    import optuna  # noqa: PLC0415

    try:
        study = optuna.load_study(
            study_name=f"{study_prefix}-{model_name}", storage=f"sqlite:///{config.optuna_db}"
        )
        best = study.best_params
    except (KeyError, ValueError):
        return {}
    except Exception:  # noqa: BLE001 — a missing/locked study must never break the retrain
        logger.warning("could not load HPO study for %s", model_name, exc_info=True)
        return {}
    return {k: v for k, v in best.items() if not k.startswith("feat_")}


def apply_hpo_best(config: Config, model_name: str = "lgbm") -> dict[str, object]:
    """Train + promote the final US model using the HPO study's best params *and features*.

    Reads the best trial, trains a LightGBM on its selected feature subset with its tuned
    params, fits a calibrator, and force-promotes it (advisory). This is how the feature-
    selection result from the search actually reaches production. LightGBM only for now.
    """
    import optuna  # noqa: PLC0415

    from berich.models import LGBMModel, ModelMetadata, promote, save_model  # noqa: PLC0415
    from berich.signals.calibration import fit_calibrator, save_calibrator  # noqa: PLC0415

    if model_name != "lgbm":
        msg = "apply_hpo_best currently supports only the lgbm model"
        raise ValueError(msg)
    study = optuna.load_study(
        study_name=f"berich-hpo-{model_name}", storage=f"sqlite:///{config.optuna_db}"
    )
    best = study.best_trial
    params = {k: v for k, v in best.params.items() if not k.startswith("feat_")}
    features = best.user_attrs.get("features")

    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    if features:
        dataset = replace(dataset, x=dataset.x[list(features)])

    oof = oof_predict(dataset, lambda: LGBMModel(**params), embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt = run_backtest(prices, oof, BacktestConfig(entry_threshold=config.signals.buy_threshold))
    model = LGBMModel(**params).fit(dataset.x, dataset.y, sample_weight=dataset.weight)

    name = f"{model_name}-hpo"
    meta = ModelMetadata(
        name=name,
        framework="lightgbm",
        feature_columns=list(dataset.x.columns),
        metrics={
            "auc": oof.auc,
            "sharpe": bt.strategy.sharpe,
            "benchmark_sharpe": bt.benchmark.sharpe,
        },
        beats_buy_hold=bt.beats_buy_hold,
        notes=f"HPO best: {len(dataset.x.columns)} features, params={params}",
    )
    artifact_dir = save_model(model, meta, registry_dir=config.models_dir)
    cal = fit_calibrator(oof.frame["proba"].to_numpy(), oof.frame["y_true"].to_numpy())
    save_calibrator(cal, artifact_dir=artifact_dir)
    promote(name, registry_dir=config.models_dir, force=True)
    logger.info(
        "applied HPO best '%s': %d features, Sharpe=%.3f",
        name, len(dataset.x.columns), bt.strategy.sharpe,
    )
    return {
        "name": name,
        "auc": oof.auc,
        "sharpe": bt.strategy.sharpe,
        "n_features": len(dataset.x.columns),
        "features": list(dataset.x.columns),
    }


def _log_to_mlflow(model_name: str, study: optuna.Study) -> None:
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_experiment(f"berich-hpo-{model_name}")
        with mlflow.start_run(run_name=f"{model_name}-best"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_sharpe", study.best_value)
            mlflow.log_metric("n_trials", len(study.trials))
    except Exception:  # noqa: BLE001 — MLflow logging is best-effort, never fails the search
        logger.debug("MLflow logging of HPO summary failed", exc_info=True)


__all__ = [
    "FEATURE_GROUPS",
    "LONGSHORT_MODELS",
    "SUPPORTED_MODELS",
    "apply_hpo_best",
    "best_params_for",
    "objective_for",
    "run_hpo",
    "run_longshort_hpo",
]
