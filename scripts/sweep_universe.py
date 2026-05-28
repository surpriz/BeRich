"""Phase 6 — comparative backtest across mega/mid/small/all universes.

Each universe runs the same LightGBM + 22-feature baseline (no earnings,
no news — we isolate the universe effect) with the volume-proportional
slippage model so small-caps pay realistic costs. The benchmark for each
run is an equal-weight buy & hold of the **same** universe — strategy
and benchmark always trade the same tickers so the Sharpe comparison is
honest.

Promote gate: AUC > 0.54 AND strategy Sharpe > universe-specific B&H.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.features.build import feature_columns
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.training import oof_predict

UNIVERSES = ("mega", "mid", "small", "all")
PROMOTE_MIN_AUC = 0.54


@dataclass
class Row:
    universe: str
    n_tickers: int
    n_samples: int
    p_win: float
    auc: float
    sharpe: float
    bench_sharpe: float
    beats_buy_hold: bool
    n_trades: int
    max_drawdown: float
    model: LGBMModel
    ticker_count_in_dataset: int


def _evaluate(config: Config, universe: str, label_cfg: LabelConfig) -> Row:
    store = OhlcvStore(config.ohlcv_dir)
    tickers = config.tickers_for_universe(universe)
    dataset = build_dataset(store, tickers, label_cfg)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in tickers if (df := store.load(t)) is not None and not df.empty}
    bt_cfg = BacktestConfig(
        entry_threshold=0.5,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
        volume_proportional_slippage=True,
    )
    bt = run_backtest(prices, oof, bt_cfg)
    # Final model on all labeled history — used for promote() if the gate fires.
    final = LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    return Row(
        universe=universe,
        n_tickers=len(tickers),
        n_samples=len(dataset),
        p_win=float(dataset.y.mean()),
        auc=oof.auc,
        sharpe=bt.strategy.sharpe,
        bench_sharpe=bt.benchmark.sharpe,
        beats_buy_hold=bt.beats_buy_hold,
        n_trades=int(bt.strategy.n_trades),
        max_drawdown=bt.strategy.max_drawdown,
        model=final,
        ticker_count_in_dataset=int(dataset.tickers.nunique()),
    )


def _print_table(rows: list[Row]) -> None:
    print(
        f"\n{'universe':<10}{'tickers':>9}{'in DS':>8}{'samples':>10}{'P(win)':>9}"
        f"{'AUC':>8}{'Sharpe':>9}{'B&H':>8}{'beats':>8}{'maxDD':>9}{'trades':>9}"
    )
    for r in rows:
        print(
            f"{r.universe:<10}{r.n_tickers:>9d}{r.ticker_count_in_dataset:>8d}"
            f"{r.n_samples:>10d}{r.p_win:>9.3f}{r.auc:>8.4f}"
            f"{r.sharpe:>9.3f}{r.bench_sharpe:>8.3f}{r.beats_buy_hold!s:>8}"
            f"{r.max_drawdown:>9.3f}{r.n_trades:>9d}"
        )


def _print_top_importances(model: LGBMModel, k: int = 10) -> None:
    cols = feature_columns()  # 22 base features
    imp = np.asarray(model.feature_importances_, dtype=float)
    total = imp.sum() or 1.0
    pct = 100.0 * imp / total
    order = np.argsort(pct)[::-1]
    print(f"\nTop {k} features for the 'all' universe model:")
    for rank, idx in enumerate(order[:k], start=1):
        print(f"  {rank:>2}. {cols[idx]:<22}{pct[idx]:6.2f}%")


def _maybe_promote(row: Row, config: Config, name: str = "lgbm-universe") -> bool:
    if not (row.auc > PROMOTE_MIN_AUC and row.sharpe > row.bench_sharpe):
        return False
    meta = ModelMetadata(
        name=name,
        framework="lightgbm",
        feature_columns=feature_columns(),
        metrics={
            "auc": row.auc,
            "sharpe": row.sharpe,
            "benchmark_sharpe": row.bench_sharpe,
        },
        beats_buy_hold=row.beats_buy_hold,
        notes=(
            f"phase-6 universe={row.universe}; n_tickers={row.n_tickers}; "
            f"n_samples={row.n_samples}; volume-proportional slippage"
        ),
    )
    save_model(row.model, meta, registry_dir=config.models_dir)
    try:
        promote(name, registry_dir=config.models_dir, force=False)
    except ValueError as exc:
        print(f"Saved '{name}' but registry blocked promotion: {exc}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 6 universe sweep")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    rows: list[Row] = []
    for universe in UNIVERSES:
        print(f"\n== Evaluating universe={universe} ==")
        row = _evaluate(config, universe, label_cfg)
        rows.append(row)
        print(
            f"   n_tickers={row.n_tickers} (in dataset: {row.ticker_count_in_dataset}) "
            f"n_samples={row.n_samples}  AUC={row.auc:.4f}  "
            f"Sharpe={row.sharpe:.3f}  vs B&H={row.bench_sharpe:.3f}"
        )

    _print_table(rows)
    all_row = next((r for r in rows if r.universe == "all"), None)
    if all_row is not None:
        _print_top_importances(all_row.model)

    winners = [r for r in rows if r.auc > PROMOTE_MIN_AUC and r.sharpe > r.bench_sharpe]
    if not winners:
        print(f"\nNo universe cleared the promote gate (AUC > {PROMOTE_MIN_AUC} AND Sharpe > B&H).")
        return 0
    print(f"\n{len(winners)} universe(s) cleared the gate; attempting promote...")
    for r in winners:
        promoted = _maybe_promote(r, config, name=f"lgbm-universe-{r.universe}")
        verdict = "PROMOTED" if promoted else "saved but not promoted"
        print(f"  universe={r.universe}: {verdict}  AUC={r.auc:.4f}  Sharpe={r.sharpe:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
