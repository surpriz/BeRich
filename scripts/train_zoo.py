"""Train a zoo model, the whole zoo, or run HPO — manual driver for the GPU box.

Examples:
    uv run python scripts/train_zoo.py --model zoo                 # nightly-equivalent retrain
    uv run python scripts/train_zoo.py --model patchtst --force    # train one model, force promote
    uv run python scripts/train_zoo.py --model lstm --hpo --trials 40
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.models import LGBMModel, PatchTSTConfig, PatchTSTModel, StackingEnsemble
from berich.scheduler.jobs import _zoo_factory, retrain_zoo_job
from berich.training.deep import baseline_sharpe, train_deep_model
from berich.training.hpo import run_hpo


def _ensemble_factory() -> StackingEnsemble:
    return StackingEnsemble([LGBMModel, lambda: PatchTSTModel(PatchTSTConfig())])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train the model zoo / run HPO")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument(
        "--model", default="zoo", choices=["lgbm", "lstm", "patchtst", "tft", "ensemble", "zoo"]
    )
    p.add_argument("--hpo", action="store_true", help="Run Optuna search instead of a single fit")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--force", action="store_true", help="Force promote (bypass guard)")
    args = p.parse_args(argv)
    config = Config.load(args.config)

    if args.model == "zoo":
        print(retrain_zoo_job(config))
        return 0

    if args.hpo:
        study = run_hpo(config, args.model, n_trials=args.trials)
        print(f"best Sharpe={study.best_value:.3f}  params={study.best_params}")
        return 0

    base = baseline_sharpe(config)
    if args.model == "ensemble":
        framework, hyperparams, factory = "stacking-ensemble", {}, _ensemble_factory
    else:
        framework, hyperparams, factory = _zoo_factory(args.model)

    res = train_deep_model(
        config,
        name=f"{args.model}-manual",
        framework=framework,
        model_factory=factory,
        hyperparams=hyperparams,
        baseline_sharpe=base,
        force_promote=args.force,
    )
    print(
        f"{args.model}: AUC={res.oos_auc:.4f} Sharpe={res.strategy_sharpe:.3f} "
        f"(baseline={res.baseline_sharpe:.3f}, B&H={res.benchmark_sharpe:.3f}) "
        f"promoted={res.promoted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
