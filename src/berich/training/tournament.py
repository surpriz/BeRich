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
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from berich.backtest import BacktestConfig, run_backtest
from berich.backtest.significance import assess_sharpe
from berich.data.store import OhlcvStore
from berich.labeling.triple_barrier import LabelConfig
from berich.models import ModelMetadata, promote, save_model
from berich.models.registry import MAX_SHARPE_PVALUE, MIN_DEFLATED_SHARPE
from berich.signals.calibration import ProbaCalibrator, fit_calibrator, save_calibrator
from berich.training import oof_predict
from berich.training.hpo import _ticker_dataset, best_for_ticker, best_horizon_for_ticker

if TYPE_CHECKING:
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


def train_candidate(
    config: Config,
    store: OhlcvStore,
    ticker: str,
    side: str,
    model_name: str,
    *,
    strategy: str = "fixed",
    device: str | None = None,
    calibrate: bool = True,
) -> tuple[Model, ModelMetadata, ProbaCalibrator | None, CandidateResult]:
    """Train one framework for a ticker x side x strategy: OOS, backtest, gate, fit, calibrator.

    ``strategy`` ("fixed" | "trailing" | "trailing_tp") re-labels and backtests under that exit
    rule so the model learns the target it will actually be served on. Returns
    ``(model, metadata, calibrator_or_None, candidate_result)``. Nothing is saved or promoted
    here — that is the tournament's job once it has picked a winner.
    """
    params, features = best_for_ticker(config, ticker, model_name, side, strategy)
    # Reuse the HPO-chosen triple-barrier horizon so the candidate is trained, backtested AND
    # later served on the same horizon it was optimized for. None => the configured default.
    horizon = best_horizon_for_ticker(config, ticker, model_name, side, strategy)
    dataset, prices = _ticker_dataset(
        config, ticker, side, horizon_days=horizon, exit_mode=strategy
    )
    if features is not None:
        keep = [c for c in features if c in dataset.x.columns]
        dataset = replace(dataset, x=dataset.x[keep])

    update: dict[str, object] = {"direction": side, "exit_mode": strategy}
    if horizon is not None:
        update["horizon_days"] = horizon
    label_cfg = LabelConfig(**config.labeling.model_dump()).model_copy(update=update)

    def factory() -> Model:
        return _model_from_params(model_name, params, device=device)

    oof = oof_predict(dataset, factory, embargo=label_cfg.horizon_days)
    df = store.load(ticker)
    if df is None or df.empty:  # defensive: _ticker_dataset already loaded it once
        df = prices[ticker]
    bt = run_backtest(
        {ticker: df},
        oof,
        BacktestConfig(
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
        ),
    )

    framework = _FRAMEWORK[model_name]
    auc = oof.auc
    if side == "short":
        sig = assess_sharpe(bt.strategy_returns)
        beats_guard = (
            sig.sharpe > 0
            and sig.deflated_sharpe >= MIN_DEFLATED_SHARPE
            and sig.p_value < MAX_SHARPE_PVALUE
            and auc > AUC_FLOOR
        )
        metrics = {
            "auc": auc,
            "sharpe": sig.sharpe,
            "deflated_sharpe": sig.deflated_sharpe,
            "sharpe_pvalue": sig.p_value,
        }
        benchmark_sharpe = 0.0  # a short's benchmark is cash
        beats_buy_hold = False
        notes = (
            f"per-ticker short {model_name}: dsr={sig.deflated_sharpe:.3f} "
            f"p={sig.p_value:.3f} AUC={auc:.3f}"
        )
    else:
        beats_guard = bool(bt.beats_buy_hold) and auc > AUC_FLOOR
        metrics = {
            "auc": auc,
            "sharpe": bt.strategy.sharpe,
            "benchmark_sharpe": bt.benchmark.sharpe,
        }
        benchmark_sharpe = bt.benchmark.sharpe
        beats_buy_hold = bool(bt.beats_buy_hold)
        notes = (
            f"per-ticker long {model_name}: Sharpe={bt.strategy.sharpe:.3f} "
            f"vs B&H {bt.benchmark.sharpe:.3f} AUC={auc:.3f}"
        )

    model = factory().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    calibrator: ProbaCalibrator | None = None
    if calibrate:
        calibrator = fit_calibrator(oof.frame["proba"].to_numpy(), oof.frame["y_true"].to_numpy())

    meta = ModelMetadata(
        name=f"{model_name}-{side}",
        framework=framework,
        feature_columns=list(dataset.x.columns),
        metrics=metrics,
        beats_buy_hold=beats_buy_hold,
        strategy_type="long_only",
        side=cast("Literal['long', 'short']", side),
        ticker=ticker,
        horizon_days=label_cfg.horizon_days,
        exit_strategy=strategy,
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
    trained_at: str | None = None,
) -> TournamentResult:
    """Run every candidate framework for one ticker x side x strategy and promote the honest winner.

    A winner is the gate-passing candidate with the best strategy Sharpe (long) / deflated
    Sharpe (short); it is saved + promoted into the per-ticker x strategy namespace. If none
    passes, the best-AUC candidate is saved unpromoted (advisory-only). Writes a ``status.json``
    summary for the dashboard and returns the full verdict.
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
) -> TournamentResult:
    """Train + gate the candidate frameworks (no side effects beyond the registry)."""
    model_names = models or config.zoo.ticker_tournament_models
    registry_dir = config.model_dir_for_ticker(ticker, side, strategy)
    store = OhlcvStore(config.ohlcv_dir)

    def _skip(reason: str) -> TournamentResult:
        logger.info("tournament skipped for %s/%s/%s: %s", ticker, side, strategy, reason)
        return TournamentResult(
            ticker, side, [], None, promoted=False, advisory_only=True, strategy=strategy
        )

    try:
        probe, _ = _ticker_dataset(config, ticker, side, exit_mode=strategy)
    except ValueError as exc:
        return _skip(str(exc))
    if len(probe) < MIN_LABELED_ROWS:
        return _skip(f"only {len(probe)} labeled rows (< {MIN_LABELED_ROWS})")

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
