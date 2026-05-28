"""Train + evaluate + (gated) promote the PEAD model.

Builds event-level (ticker, earnings) dataset across the full universe
(mega + mid + small), runs walk-forward OOS at the event grain, executes
the event-driven backtest, and applies the Phase 7 promote gate:
``AUC > 0.55`` AND ``Sharpe > B&H Sharpe`` AND ``n_events >= 1000``.

A failing gate is the expected outcome (the project's running honest
verdict is "no edge above B&H on US daily/swing"). Either way the
metrics + top-importance ranking land in the run log so the next
iteration starts from a known state.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import EarningsStore, NewsStore
from berich.data.store import OhlcvStore
from berich.datasets.pead import build_pead_dataset
from berich.features.pead_features import PEAD_FEATURE_COLUMNS
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.training.pead import oof_predict_pead, run_pead_backtest

PROMOTE_MIN_AUC = 0.55
PROMOTE_MIN_EVENTS = 1000


def _print_importances(model: LGBMModel, k: int = 10) -> None:
    imp = np.asarray(model.feature_importances_, dtype=float)
    total = imp.sum() or 1.0
    pct = 100.0 * imp / total
    order = np.argsort(pct)[::-1]
    print(f"\nTop {k} PEAD features:")
    for rank, idx in enumerate(order[:k], start=1):
        print(f"  {rank:>2}. {PEAD_FEATURE_COLUMNS[idx]:<26}{pct[idx]:6.2f}%")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="PEAD (Post-Earnings Drift) training")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--universe", choices=["mega", "mid", "small", "all"], default="all")
    parser.add_argument("--label-horizon", choices=["5d", "20d"], default="5d")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--name", default="pead-lgbm")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir)
    if not news_store.has_any_data():
        print("(no news cache — news features will be neutral defaults)")
        news_arg = None
    else:
        news_arg = news_store

    tickers = config.tickers_for_universe(args.universe)
    print(
        f"Building PEAD dataset on universe='{args.universe}' "
        f"({len(tickers)} tickers, label={args.label_horizon})..."
    )
    dataset = build_pead_dataset(
        store,
        earnings_store,
        tickers,
        news_store=news_arg,
        label_horizon=args.label_horizon,
    )
    if not len(dataset):
        print("No PEAD events produced; check that earnings cache is populated.")
        return 1
    print(
        f"  events: {len(dataset)}  P(drift)={dataset.y.mean():.3f}  "
        f"date range: {dataset.entry_dates.min().date()} -> {dataset.entry_dates.max().date()}"
    )

    print("\nRunning walk-forward OOS on events...")
    oof = oof_predict_pead(dataset, LGBMModel, n_folds=5, min_train=500)
    print(f"  OOS AUC: {oof.auc:.4f}")

    print(f"\nEvent-driven backtest (threshold={args.threshold})...")
    bt = run_pead_backtest(dataset, oof, store, threshold=args.threshold)
    n_trades = int(bt.strategy.n_trades)
    print(
        f"  n_trades={n_trades}  Sharpe={bt.strategy.sharpe:.3f}  "
        f"vs window B&H={bt.benchmark.sharpe:.3f}  "
        f"beats={bt.beats_buy_hold}  total_return={bt.strategy.total_return:.3f}  "
        f"maxDD={bt.strategy.max_drawdown:.3f}"
    )

    # Fit a final model on the entire event set for the registry.
    final = LGBMModel().fit(dataset.x, dataset.y)
    _print_importances(final)

    gate_pass = (
        oof.auc > PROMOTE_MIN_AUC and bt.beats_buy_hold and len(dataset) >= PROMOTE_MIN_EVENTS
    )
    print(
        f"\nPromote gate (AUC > {PROMOTE_MIN_AUC} AND Sharpe > B&H AND events "
        f">= {PROMOTE_MIN_EVENTS}): {gate_pass}"
    )
    if not gate_pass:
        print("No promotion — PEAD code merged, registry untouched.")
        return 0

    meta = ModelMetadata(
        name=args.name,
        framework="lightgbm",
        feature_columns=list(PEAD_FEATURE_COLUMNS),
        metrics={
            "auc": oof.auc,
            "sharpe": bt.strategy.sharpe,
            "benchmark_sharpe": bt.benchmark.sharpe,
            "n_trades": float(n_trades),
            "n_events": float(len(dataset)),
        },
        beats_buy_hold=bt.beats_buy_hold,
        notes=(
            f"pead label={args.label_horizon} universe={args.universe} "
            f"n_events={len(dataset)} threshold={args.threshold}"
        ),
    )
    save_model(final, meta, registry_dir=config.models_dir)
    try:
        promote(args.name, registry_dir=config.models_dir, force=False)
        print(f"Promoted '{args.name}' as the active PEAD model.")
    except ValueError as exc:
        print(f"Saved '{args.name}' but registry blocked promotion: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
