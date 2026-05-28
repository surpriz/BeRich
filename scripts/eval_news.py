"""Phase 5b — comparative backtest on the news-coverage window.

Both runs (baseline 22 features vs 22+7 news features) are filtered to the
window where news exists (2022-04 onwards) so the comparison is honest. The
script prints the AUC/Sharpe table + top-10 feature importances, then applies
the Phase 5b promotion gate: ``AUC > 0.55`` AND ``strategy Sharpe > buy & hold
Sharpe``. If the gate fires, a final model is trained on the full history
and promotion is attempted (the registry's guard will still block it if the
metadata says it didn't beat buy & hold).
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import EarningsStore, NewsStore
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.datasets.assemble import SupervisedDataset
from berich.features.build import feature_columns
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.training import oof_predict

WINDOW_START = pd.Timestamp("2022-04-01")
PROMOTE_MIN_AUC = 0.55


def _filter_window(dataset: SupervisedDataset, start: pd.Timestamp) -> SupervisedDataset:
    """Restrict a SupervisedDataset to bars at or after ``start``."""
    mask = dataset.dates >= start
    return SupervisedDataset(
        x=dataset.x.loc[mask],
        y=dataset.y.loc[mask],
        weight=dataset.weight.loc[mask],
        dates=dataset.dates[mask],
        tickers=dataset.tickers.loc[mask],
    )


def _evaluate(
    config: Config,
    *,
    label_cfg: LabelConfig,
    with_news: bool,
) -> tuple[float, dict, dict, bool, int, LGBMModel, SupervisedDataset]:
    store = OhlcvStore(config.ohlcv_dir)
    # Earnings is always on for the comparison (Phase 5a code already merged).
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir) if with_news else None

    dataset = build_dataset(
        store,
        config.watchlist,
        label_cfg,
        earnings_store=earnings_store,
        news_store=news_store,
    )
    windowed = _filter_window(dataset, WINDOW_START)
    oof = oof_predict(windowed, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))
    # Train a final model on the windowed data — used by promote() below.
    final = LGBMModel().fit(windowed.x, windowed.y, sample_weight=windowed.weight)
    return (
        oof.auc,
        bt.strategy.as_dict(),
        bt.benchmark.as_dict(),
        bt.beats_buy_hold,
        len(windowed),
        final,
        windowed,
    )


def _print_table(rows: list[tuple[str, float, float, float, bool, int]]) -> None:
    print(f"\n{'variant':<22}{'samples':>9}{'AUC':>8}{'Sharpe':>9}{'B&H':>8}{'beats':>8}")
    for name, auc, sharpe, bench, beats, n in rows:
        print(f"{name:<22}{n:>9d}{auc:>8.4f}{sharpe:>9.3f}{bench:>8.3f}{beats!s:>8}")


def _print_top_importances(model: LGBMModel, cols: list[str], k: int = 10) -> None:
    imp = np.asarray(model.feature_importances_, dtype=float)
    total = imp.sum() or 1.0
    pct = 100.0 * imp / total
    order = np.argsort(pct)[::-1]
    print(f"\nTop {k} features (out of {len(cols)}):")
    for rank, idx in enumerate(order[:k], start=1):
        print(f"  {rank:>2}. {cols[idx]:<26}{pct[idx]:6.2f}%")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 5b news/sentiment evaluation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--promote-name", default="lgbm-news")
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Bypass the registry guard (NOT recommended; auto-mode never sets this)",
    )
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    print(f"\n== Baseline (22 + earnings) on window {WINDOW_START.date()}+ ==")
    base_auc, base_str, base_bench, base_beats, n_base, base_model, base_ds = _evaluate(
        config, label_cfg=label_cfg, with_news=False
    )
    print(
        f"  n={n_base}  AUC={base_auc:.4f}  "
        f"Sharpe={base_str['sharpe']:.3f}  B&H={base_bench['sharpe']:.3f}"
    )

    print(f"\n== With news (22 + earnings + 7 news) on window {WINDOW_START.date()}+ ==")
    news_auc, news_str, news_bench, news_beats, n_news, news_model, news_ds = _evaluate(
        config, label_cfg=label_cfg, with_news=True
    )
    print(
        f"  n={n_news}  AUC={news_auc:.4f}  "
        f"Sharpe={news_str['sharpe']:.3f}  B&H={news_bench['sharpe']:.3f}"
    )

    _print_table(
        [
            (
                "22 + earnings",
                base_auc,
                base_str["sharpe"],
                base_bench["sharpe"],
                base_beats,
                n_base,
            ),
            (
                "22 + earnings + news",
                news_auc,
                news_str["sharpe"],
                news_bench["sharpe"],
                news_beats,
                n_news,
            ),
        ]
    )

    _print_top_importances(news_model, feature_columns(earnings=True, news=True))

    # Promotion gate (Phase 5b rule, strict): both AUC and Sharpe must clear.
    gate_pass = news_auc > PROMOTE_MIN_AUC and news_str["sharpe"] > news_bench["sharpe"]
    print(f"\nPromote gate (AUC > {PROMOTE_MIN_AUC} AND Sharpe > B&H): {gate_pass}")
    if not gate_pass and not args.force_promote:
        print("No promotion — code merged behind --with-news, registry untouched.")
        return 0

    # Final model: re-fit on the full labeled history (not the windowed view)
    # so production serving sees the broadest possible training set, not just
    # the news-window subset. The metadata records the news=True schema.
    store = OhlcvStore(config.ohlcv_dir)
    full = build_dataset(
        store,
        config.watchlist,
        label_cfg,
        earnings_store=EarningsStore(config.earnings_dir),
        news_store=NewsStore(config.news_dir),
    )
    final = LGBMModel().fit(full.x, full.y, sample_weight=full.weight)
    meta = ModelMetadata(
        name=args.promote_name,
        framework="lightgbm",
        feature_columns=feature_columns(earnings=True, news=True),
        metrics={
            "auc": news_auc,
            "sharpe": news_str["sharpe"],
            "benchmark_sharpe": news_bench["sharpe"],
        },
        beats_buy_hold=news_beats,
        notes=(
            f"news-window={WINDOW_START.date()}+; baseline AUC={base_auc:.4f} "
            f"Sharpe={base_str['sharpe']:.3f}; full-history train rows={len(full)}"
        ),
    )
    save_model(final, meta, registry_dir=config.models_dir)
    print(f"\nSaved '{args.promote_name}' to registry.")
    try:
        promote(args.promote_name, registry_dir=config.models_dir, force=args.force_promote)
        print(f"Promoted '{args.promote_name}'.")
    except ValueError as exc:
        print(f"Not promoted: {exc}")
    # Silence unused-import warnings; base_model / base_ds / news_ds are returned
    # for parity in case a future caller wants them.
    _ = (base_model, base_ds, news_ds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
