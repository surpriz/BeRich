"""Run Optuna HPO across several zoo models and print a leaderboard.

Example:
    uv run python scripts/sweep_zoo.py --models lgbm patchtst lstm --trials 30
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.training.hpo import SUPPORTED_MODELS, run_hpo


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="HPO sweep across zoo models")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args(argv)
    config = Config.load(args.config)

    leaderboard = []
    for model in args.models:
        study = run_hpo(config, model, n_trials=args.trials)
        leaderboard.append((model, study.best_value, study.best_params))

    leaderboard.sort(key=lambda r: r[1], reverse=True)
    print("\n=== HPO leaderboard (OOS Sharpe) ===")
    for model, sharpe, params in leaderboard:
        print(f"{model:>10}  Sharpe={sharpe:7.3f}  {params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
