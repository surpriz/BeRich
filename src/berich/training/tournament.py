"""Per-ticker model tournament.

Each asset gets its own uniquely-optimized model, per side (long / short). For one ticker
this runs every candidate framework (lgbm / lstm / patchtst / tft), each on its own best
hyperparameters + feature subset from :mod:`berich.training.hpo`, scores them walk-forward
out-of-sample, and applies the *honest* promotion gate:

- **long**: beat the ticker's own buy & hold (and AUC > 0.5);
- **short**: a positive, statistically significant Sharpe vs cash (deflated Sharpe + p-value
  from :mod:`berich.backtest.significance`) — a short has no buy-&-hold benchmark.

The winner (best strategy Sharpe for long, best deflated Sharpe for short) is saved and
promoted into the per-ticker registry namespace. If no candidate clears its gate, the best
by AUC is still saved for inspection but left unpromoted (advisory-only) — never bypass the
guard. One bad framework never aborts the tournament.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from berich.backtest import BacktestConfig, run_backtest
from berich.backtest.annualization import bars_per_year
from berich.backtest.significance import assess_sharpe
from berich.data.store import OhlcvStore
from berich.labeling.triple_barrier import LabelConfig
from berich.models import ModelMetadata, promote, save_model
from berich.models.registry import MAX_SHARPE_PVALUE, MIN_DEFLATED_SHARPE, MIN_TRADES
from berich.signals.calibration import (
    ProbaCalibrator,
    fit_calibrator,
    optimal_decision_threshold,
    save_calibrator,
)
from berich.training import oof_predict
from berich.training.hpo import (
    _ticker_dataset,
    best_for_ticker,
    best_horizon_for_ticker,
    ticker_trial_count,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from berich.config import Config
    from berich.models.base import Model

logger = logging.getLogger(__name__)

# Below this many labeled windows a per-ticker walk-forward is too thin to trust; skip it.
MIN_LABELED_ROWS = 250

# A model with no ranking skill (OOS AUC <= this) cannot be trusted even if its backtest
# looks good — it's the same coin-flip floor the OofResult.auc docstring describes.
AUC_FLOOR = 0.5

_FRAMEWORK = {
    "lgbm": "lightgbm",
    "lstm": "lstm",
    "patchtst": "patchtst",
    "tft": "tft",
}


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Coerce any None/NaN/Inf metric to 0.0 so the stored metadata is always valid floats.

    A degenerate short (too few trades → undefined deflated Sharpe) used to persist a None
    metric, which both broke ``ModelMetadata`` deserialization and the significance guard. 0.0
    fails the guard, which is the correct verdict for "significance couldn't be established".
    """
    return {
        k: (float(v) if v is not None and math.isfinite(v) else 0.0) for k, v in metrics.items()
    }


@dataclass
class CandidateResult:
    """One framework's OOS verdict for a ticker x side."""

    ticker: str
    side: str
    model_name: str
    oos_auc: float
    strategy_sharpe: float
    benchmark_sharpe: float
    beats_guard: bool
    framework: str
    n_features: int


@dataclass
class TournamentResult:
    """The full per-ticker tournament outcome for one side x exit strategy."""

    ticker: str
    side: str
    candidates: list[CandidateResult]
    winner: str | None
    promoted: bool
    advisory_only: bool
    strategy: str = "fixed"


if TYPE_CHECKING:
    # One trained candidate: (fitted model, metadata, calibrator-or-None, OOS verdict).
    _Trained = tuple[Model, ModelMetadata, ProbaCalibrator | None, CandidateResult]


def _model_from_params(model_name: str, params: dict, *, device: str | None) -> Model:
    """Construct a fresh model of ``model_name`` from a flat params dict (deep cfgs lazy)."""
    if model_name == "lgbm":
        from berich.models import LGBMModel  # noqa: PLC0415

        return LGBMModel(**params)
    if model_name == "lstm":
        from berich.models import LSTMConfig, LSTMModel  # noqa: PLC0415

        return LSTMModel(LSTMConfig(**params, device=device))
    if model_name == "patchtst":
        from berich.models import PatchTSTConfig, PatchTSTModel  # noqa: PLC0415

        return PatchTSTModel(PatchTSTConfig(**params, device=device))
    if model_name == "tft":
        from berich.models import TFTConfig, TFTModel  # noqa: PLC0415

        return TFTModel(TFTConfig(**params, device=device))
    msg = f"unknown model '{model_name}'"
    raise ValueError(msg)


def train_candidate(  # noqa: PLR0915 — linear train/backtest/gate/fit flow, clearer inline
    config: Config,
    store: OhlcvStore,
    ticker: str,
    side: str,
    model_name: str,
    *,
    strategy: str = "fixed",
    device: str | None = None,
    calibrate: bool = True,
    n_trials: int = 1,
    interval: str = "1d",
    factory_override: Callable[[], Model] | None = None,
    framework_override: str | None = None,
) -> tuple[Model, ModelMetadata, ProbaCalibrator | None, CandidateResult]:
    """Train one framework for a ticker x side x strategy: OOS, backtest, gate, fit, calibrator.

    ``strategy`` ("fixed" | "trailing" | "trailing_tp") re-labels and backtests under that exit
    rule so the model learns the target it will actually be served on. ``n_trials`` is the size of
    the search this candidate was selected from (HPO trials across the tournament's frameworks);
    it deflates the Sharpe so the significance gate corrects for data-mining.
    ``factory_override`` swaps in a custom model factory (e.g. an :class:`AveragingEnsemble`)
    trained on the FULL feature set instead of the HPO-selected subset — for the serve-time
    ensemble; ``framework_override`` then labels the artifact. Returns
    ``(model, metadata, calibrator_or_None, candidate_result)``. Nothing is saved or promoted
    here — that is the tournament's job once it has picked a winner.
    """
    params, features = best_for_ticker(
        config, ticker, model_name, side, strategy, interval=interval
    )
    # Reuse the HPO-chosen triple-barrier horizon so the candidate is trained, backtested AND
    # later served on the same horizon it was optimized for. None => the configured default.
    horizon = best_horizon_for_ticker(config, ticker, model_name, side, strategy, interval=interval)
    dataset, prices = _ticker_dataset(
        config, ticker, side, horizon_days=horizon, exit_mode=strategy, interval=interval
    )
    annualization = bars_per_year(interval)
    # An ensemble member slices the full frame to its own columns, so the ensemble candidate trains
    # on the FULL feature set; a single-model candidate keeps its HPO-selected subset.
    if features is not None and factory_override is None:
        keep = [c for c in features if c in dataset.x.columns]
        dataset = replace(dataset, x=dataset.x[keep])

    update: dict[str, object] = {"direction": side, "exit_mode": strategy}
    if horizon is not None:
        update["horizon_days"] = horizon
    label_cfg = LabelConfig(**config.labeling.model_dump()).model_copy(update=update)

    def factory() -> Model:
        if factory_override is not None:
            return factory_override()
        return _model_from_params(model_name, params, device=device)

    oof = oof_predict(dataset, factory, embargo=label_cfg.horizon_days)
    if interval != "1d":
        # _ticker_dataset already loaded the correct intraday frame; never read the daily cache
        # (the passed ``store`` is the daily one) for the same ticker.
        df = prices[ticker]
    else:
        df = store.load(ticker)
        if df is None or df.empty:  # defensive: _ticker_dataset already loaded it once
            df = prices[ticker]
    bt_config = BacktestConfig(
        entry_threshold=config.signals.buy_threshold,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
        direction=side,
        exit_mode=strategy,
        trailing_atr=label_cfg.trailing_atr,
        trailing_tp_atr=label_cfg.trailing_tp_atr,
        trailing_activation_atr=label_cfg.trailing_activation_atr,
        bars_per_year=annualization,
    )
    if interval != "1d":
        # The volume-proportional slippage reference is SPY share volume — meaningless for a
        # crypto pair quoted in coins. Charge a flat ~0.10%/side (Binance spot) instead.
        bt_config = bt_config.model_copy(
            update={"volume_proportional_slippage": False, "slippage_bps": 10.0}
        )
    bt = run_backtest({ticker: df}, oof, bt_config)

    framework = framework_override or _FRAMEWORK[model_name]
    auc = oof.auc
    n_trades = len(bt.trades)
    enough_trades = n_trades >= MIN_TRADES
    # Deflated Sharpe + p-value are computed for BOTH sides (the long gate stays beats-buy-&-hold,
    # but storing significance lets the sweep-level FDR pass and the observation tier reason about
    # longs too). ``n_trials`` corrects the DSR for the search this candidate was selected from.
    sig = assess_sharpe(bt.strategy_returns, n_trials=max(n_trials, 1), bars_per_year=annualization)
    if side == "short":
        beats_guard = (
            enough_trades
            and sig.sharpe > 0
            and sig.deflated_sharpe >= MIN_DEFLATED_SHARPE
            and sig.p_value < MAX_SHARPE_PVALUE
            and auc > AUC_FLOOR
        )
        metrics = _finite_metrics(
            {
                "auc": auc,
                "sharpe": sig.sharpe,
                "deflated_sharpe": sig.deflated_sharpe,
                "sharpe_pvalue": sig.p_value,
                "n_trades": float(n_trades),
                "n_trials": float(max(n_trials, 1)),
            }
        )
        benchmark_sharpe = 0.0  # a short's benchmark is cash
        beats_buy_hold = False
        notes = (
            f"per-ticker short {model_name}: dsr={sig.deflated_sharpe:.3f} "
            f"p={sig.p_value:.3f} AUC={auc:.3f} n={n_trades}"
        )
    else:
        # Long gate: beat buy & hold AND clear the same anti-luck significance floor as a short —
        # "beats a flat forex/commodity B&H" alone promoted edge-less longs (see registry).
        beats_guard = (
            bool(bt.beats_buy_hold)
            and auc > AUC_FLOOR
            and enough_trades
            and sig.sharpe > 0
            and sig.deflated_sharpe >= MIN_DEFLATED_SHARPE
            and sig.p_value < MAX_SHARPE_PVALUE
        )
        metrics = _finite_metrics(
            {
                "auc": auc,
                "sharpe": bt.strategy.sharpe,
                "benchmark_sharpe": bt.benchmark.sharpe,
                "deflated_sharpe": sig.deflated_sharpe,
                "sharpe_pvalue": sig.p_value,
                "n_trades": float(n_trades),
                "n_trials": float(max(n_trials, 1)),
            }
        )
        benchmark_sharpe = bt.benchmark.sharpe
        beats_buy_hold = bool(bt.beats_buy_hold)
        notes = (
            f"per-ticker long {model_name}: Sharpe={bt.strategy.sharpe:.3f} "
            f"vs B&H {bt.benchmark.sharpe:.3f} AUC={auc:.3f} n={n_trades}"
        )

    model = factory().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    oof_proba = oof.frame["proba"].to_numpy()
    oof_y = oof.frame["y_true"].to_numpy()
    calibrator: ProbaCalibrator | None = None
    if calibrate:
        calibrator = fit_calibrator(oof_proba, oof_y)

    # Per-asset decision threshold on the CALIBRATED OOF scale (the scale serving compares against),
    # tuned to this asset's payoff ratio. None => serving keeps the global threshold.
    oof_cal = calibrator.transform(oof_proba) if calibrator is not None else oof_proba
    decision_threshold = optimal_decision_threshold(
        oof_cal, oof_y, reward=label_cfg.take_profit_atr, risk=label_cfg.stop_loss_atr
    )

    meta = ModelMetadata(
        name=f"{model_name}-{side}",
        framework=framework,
        feature_columns=list(dataset.x.columns),
        metrics=metrics,
        beats_buy_hold=beats_buy_hold,
        strategy_type="long_only",
        decision_threshold=decision_threshold,
        side=cast("Literal['long', 'short']", side),
        ticker=ticker,
        horizon_days=label_cfg.horizon_days,
        exit_strategy=strategy,
        interval=interval,
        notes=f"[{strategy}] {notes}",
    )
    candidate = CandidateResult(
        ticker=ticker,
        side=side,
        model_name=model_name,
        oos_auc=auc,
        strategy_sharpe=metrics["sharpe"],
        benchmark_sharpe=benchmark_sharpe,
        beats_guard=beats_guard,
        framework=framework,
        n_features=len(dataset.x.columns),
    )
    return model, meta, calibrator, candidate


def _maybe_train_ensemble(
    config: Config,
    store: OhlcvStore,
    ticker: str,
    side: str,
    strategy: str,
    trained: list[_Trained],
    *,
    rank_key: Callable[[_Trained], float],
    device: str | None,
    calibrate: bool,
    n_trials: int,
    interval: str = "1d",
) -> _Trained | None:
    """Build + evaluate an :class:`AveragingEnsemble` over the top frameworks; ``None`` if N/A.

    Picks the best ``ensemble_top_n`` distinct frameworks by guard metric, rebuilds each as a
    member (its HPO params + feature subset), and trains the ensemble through ``train_candidate`` so
    it is gated/calibrated like a single model. Needs >= 2 distinct frameworks to be worthwhile.
    """
    from berich.models.ensemble import AveragingEnsemble, MemberSpec  # noqa: PLC0415

    top_n = max(2, config.signals.ensemble_top_n)
    ranked = sorted(trained, key=lambda t: (t[3].beats_guard, rank_key(t)), reverse=True)
    chosen_names: list[str] = []
    for t in ranked:
        name = t[3].model_name
        if name not in chosen_names:
            chosen_names.append(name)
        if len(chosen_names) >= top_n:
            break
    if len(chosen_names) < 2:  # noqa: PLR2004 — an "ensemble" of one is just that model
        return None

    specs: list[MemberSpec] = []
    for name in chosen_names:
        params_m, feats_m = best_for_ticker(config, ticker, name, side, strategy, interval=interval)

        def _mk(mn: str = name, p: dict = params_m) -> Model:
            return _model_from_params(mn, p, device=device)

        specs.append((_mk, list(feats_m) if feats_m is not None else []))

    def _ens_factory() -> Model:
        return AveragingEnsemble(specs)

    try:
        return train_candidate(
            config,
            store,
            ticker,
            side,
            "ensemble",
            strategy=strategy,
            device=device,
            calibrate=calibrate,
            n_trials=n_trials,
            interval=interval,
            factory_override=_ens_factory,
            framework_override="ensemble",
        )
    except Exception:  # noqa: BLE001 — a failed ensemble must not abort the tournament
        logger.warning(
            "ensemble candidate failed for %s/%s/%s", ticker, side, strategy, exc_info=True
        )
        return None


def _write_status(config: Config, result: TournamentResult, *, stamp: str) -> None:
    """Persist a per-(ticker, side, strategy) tournament summary for the dashboard's training tab.

    Written next to the per-ticker registry as ``status.json`` so the /api/training scan can
    report the full candidate slate (not just the saved winner), the verdict, and the run time.
    """
    out_dir = config.model_dir_for_ticker(result.ticker, result.side, result.strategy)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": result.ticker,
        "side": result.side,
        "strategy": result.strategy,
        "winner": result.winner,
        "promoted": result.promoted,
        "advisory_only": result.advisory_only,
        "trained_at": stamp,
        "candidates": [asdict(c) for c in result.candidates],
    }
    (out_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_ticker_tournament(
    config: Config,
    ticker: str,
    side: str = "long",
    *,
    strategy: str = "fixed",
    models: list[str] | None = None,
    device: str | None = None,
    calibrate: bool = True,
    force: bool = False,
    interval: str = "1d",
    trained_at: str | None = None,
) -> TournamentResult:
    """Run every candidate framework for one ticker x side x strategy and promote the honest winner.

    A winner is the gate-passing candidate with the best strategy Sharpe (long) / deflated
    Sharpe (short); it is saved + promoted into the per-ticker x strategy namespace. If none
    passes, the best-AUC candidate is saved unpromoted (advisory-only). Writes a ``status.json``
    summary for the dashboard and returns the full verdict. ``interval="1h"`` trains the intraday
    candidate into the interval-dimensioned namespace.
    """
    stamp = trained_at or datetime.now(UTC).isoformat()
    result = _run_tournament(
        config,
        ticker,
        side,
        strategy=strategy,
        models=models,
        device=device,
        calibrate=calibrate,
        force=force,
        interval=interval,
    )
    _write_status(config, result, stamp=stamp)
    return result


def _run_tournament(  # noqa: C901 — linear train/gate/promote flow, clearer inline
    config: Config,
    ticker: str,
    side: str = "long",
    *,
    strategy: str = "fixed",
    models: list[str] | None = None,
    device: str | None = None,
    calibrate: bool = True,
    force: bool = False,
    interval: str = "1d",
) -> TournamentResult:
    """Train + gate the candidate frameworks (no side effects beyond the registry)."""
    model_names = models or config.zoo.ticker_tournament_models
    registry_dir = config.model_dir_for_ticker(ticker, side, strategy, interval=interval)
    store = (
        OhlcvStore(config.ohlcv_intraday_dir, interval=interval)
        if interval != "1d"
        else OhlcvStore(config.ohlcv_dir)
    )

    def _skip(reason: str) -> TournamentResult:
        logger.info("tournament skipped for %s/%s/%s: %s", ticker, side, strategy, reason)
        return TournamentResult(
            ticker, side, [], None, promoted=False, advisory_only=True, strategy=strategy
        )

    try:
        probe, _ = _ticker_dataset(config, ticker, side, exit_mode=strategy, interval=interval)
    except ValueError as exc:
        return _skip(str(exc))
    if len(probe) < MIN_LABELED_ROWS:
        return _skip(f"only {len(probe)} labeled rows (< {MIN_LABELED_ROWS})")

    # The served winner is the best of every HPO trial across all the tournament's frameworks for
    # this (ticker, side, strategy). Feeding that total into the Deflated Sharpe corrects the gate
    # for the size of the search (anti data-mining). Floored at the framework count so a model
    # trained on defaults (no study) still counts as one trial per framework.
    search_trials = sum(
        ticker_trial_count(config, ticker, m, side, strategy, interval=interval)
        for m in model_names
    )
    n_trials = max(search_trials, len(model_names))

    trained: list[tuple[Model, ModelMetadata, ProbaCalibrator | None, CandidateResult]] = []
    for model_name in model_names:
        try:
            trained.append(
                train_candidate(
                    config,
                    store,
                    ticker,
                    side,
                    model_name,
                    strategy=strategy,
                    device=device,
                    calibrate=calibrate,
                    n_trials=n_trials,
                    interval=interval,
                )
            )
        except Exception:  # noqa: BLE001 — one bad framework must not abort the tournament
            logger.warning(
                "candidate %s failed for %s/%s/%s",
                model_name,
                ticker,
                side,
                strategy,
                exc_info=True,
            )

    candidates = [t[3] for t in trained]
    if not trained:
        return TournamentResult(
            ticker, side, candidates, None, promoted=False, advisory_only=True, strategy=strategy
        )

    def _rank_key(
        entry: tuple[Model, ModelMetadata, ProbaCalibrator | None, CandidateResult],
    ) -> float:
        meta = entry[1]
        return meta.metrics.get("deflated_sharpe" if side == "short" else "sharpe", 0.0)

    # Serve-time ensemble (Phase 3.1): blend the top frameworks into one averaging candidate that
    # competes for the winner slot under the same gate. Off unless ``ensemble_serving`` is set.
    if config.signals.ensemble_serving:
        ens = _maybe_train_ensemble(
            config,
            store,
            ticker,
            side,
            strategy,
            trained,
            rank_key=_rank_key,
            device=device,
            calibrate=calibrate,
            n_trials=n_trials,
            interval=interval,
        )
        if ens is not None:
            trained.append(ens)
            candidates.append(ens[3])

    passers = [t for t in trained if t[3].beats_guard]
    if passers:
        model, meta, calibrator, winner = max(passers, key=_rank_key)
        artifact_dir = save_model(model, meta, registry_dir=registry_dir)
        if calibrator is not None:
            save_calibrator(calibrator, artifact_dir=artifact_dir)
        promote(meta.name, registry_dir=registry_dir, force=force)
        logger.info("promoted per-ticker winner %s for %s/%s/%s", meta.name, ticker, side, strategy)
        return TournamentResult(
            ticker,
            side,
            candidates,
            winner.model_name,
            promoted=True,
            advisory_only=False,
            strategy=strategy,
        )

    # No honest winner — save the best-AUC candidate for inspection, leave it unpromoted.
    model, meta, calibrator, _ = max(trained, key=lambda t: t[3].oos_auc)
    artifact_dir = save_model(model, meta, registry_dir=registry_dir)
    if calibrator is not None:
        save_calibrator(calibrator, artifact_dir=artifact_dir)
    logger.info(
        "no per-ticker candidate cleared the gate for %s/%s/%s — advisory only",
        ticker,
        side,
        strategy,
    )
    return TournamentResult(
        ticker, side, candidates, None, promoted=False, advisory_only=True, strategy=strategy
    )


__all__ = [
    "CandidateResult",
    "TournamentResult",
    "train_candidate",
    "train_ticker_tournament",
]
