"""Triple-barrier label sweep: which (horizon, TP, SL) shape is actually learnable?

The Phase 3 baseline (h=10, TP/SL=2/1·ATR) does not beat buy & hold with either
LightGBM or an LSTM. Per the design notes the next investigation step is
labeling, not more features or models. This script holds features and the
LightGBM model constant and sweeps the label geometry — horizon length and
TP/SL multiples — then reports OOS AUC + Sharpe vs buy & hold for each.

Why LightGBM, not the LSTM: at ~30s per OOF run vs >10min for the LSTM the
LightGBM signal is the cheap probe of whether a label shape carries usable
information at all. A label that LightGBM can't extract a signal from is
unlikely to be rescued by a deeper model.

The script does not touch the model registry; nothing is promoted. Use
``uv run python scripts/sweep_labels.py`` to run it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel
from berich.training import oof_predict


@dataclass
class LabelVariant:
    """One labeling geometry to evaluate."""

    name: str
    horizon: int
    tp_atr: float
    sl_atr: float


@dataclass
class Row:
    """One sweep row's outcome (printed and returned for downstream comparison)."""

    variant: LabelVariant
    n_samples: int
    p_win: float
    auc: float
    sharpe: float
    bench_sharpe: float
    beats_buy_hold: bool
    n_trades: int
    metrics: dict[str, float] = field(default_factory=dict)


def _evaluate(config: Config, variant: LabelVariant) -> Row:
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(
        horizon_days=variant.horizon,
        atr_window=config.labeling.atr_window,
        take_profit_atr=variant.tp_atr,
        stop_loss_atr=variant.sl_atr,
    )
    dataset = build_dataset(store, config.watchlist, label_cfg)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt_cfg = BacktestConfig(
        entry_threshold=0.5,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
    )
    bt = run_backtest(prices, oof, bt_cfg)
    return Row(
        variant=variant,
        n_samples=len(dataset),
        p_win=float(dataset.y.mean()),
        auc=oof.auc,
        sharpe=bt.strategy.sharpe,
        bench_sharpe=bt.benchmark.sharpe,
        beats_buy_hold=bt.beats_buy_hold,
        n_trades=int(bt.strategy.n_trades),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Triple-barrier label sweep")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    config = Config.load(args.config)

    variants = [
        LabelVariant("h10_tp2_sl1", horizon=10, tp_atr=2.0, sl_atr=1.0),  # current baseline
        LabelVariant("h10_tp1.5_sl1.5", horizon=10, tp_atr=1.5, sl_atr=1.5),
        LabelVariant("h20_tp2_sl1", horizon=20, tp_atr=2.0, sl_atr=1.0),
        LabelVariant("h20_tp1.5_sl1.5", horizon=20, tp_atr=1.5, sl_atr=1.5),
    ]

    rows: list[Row] = []
    for v in variants:
        print(f"\n== Evaluating {v.name}: horizon={v.horizon} tp={v.tp_atr} sl={v.sl_atr}")
        row = _evaluate(config, v)
        rows.append(row)
        print(
            f"   n={row.n_samples}  P(win)={row.p_win:.3f}  AUC={row.auc:.4f}  "
            f"Sharpe={row.sharpe:.3f}  vs B&H={row.bench_sharpe:.3f}  "
            f"beats={row.beats_buy_hold}  trades={row.n_trades}"
        )

    print("\n=== Summary ===")
    print(f"{'variant':<20}{'P(win)':>9}{'AUC':>8}{'Sharpe':>9}{'B&H':>8}{'beats':>8}{'trades':>9}")
    for r in rows:
        print(
            f"{r.variant.name:<20}{r.p_win:>9.3f}{r.auc:>8.4f}{r.sharpe:>9.3f}"
            f"{r.bench_sharpe:>8.3f}{r.beats_buy_hold!s:>8}{r.n_trades:>9d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
