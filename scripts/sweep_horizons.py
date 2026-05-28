"""Phase 5b follow-up — sweep the triple-barrier horizon while holding everything else fixed.

Phases 3, 5a, 5b all used horizon_days=10. The label sweep in Phase 3 tested
10 vs 20 with various TP/SL ratios, but never short horizons. This script
runs the same LightGBM + 22-feature baseline at h in {3, 5, 7, 10} and
prints the AUC / Sharpe table so the verdict — "no edge at any short
horizon either" — is recorded rather than asserted.

Promote gate is the standard one: AUC > 0.54 AND strategy Sharpe > buy & hold.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel
from berich.training import oof_predict

HORIZONS = [3, 5, 7, 10]
PROMOTE_MIN_AUC = 0.54


@dataclass
class Row:
    horizon: int
    n_samples: int
    p_win: float
    auc: float
    sharpe: float
    bench_sharpe: float
    beats_buy_hold: bool
    n_trades: int


def _evaluate(config: Config, horizon: int) -> Row:
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(
        horizon_days=horizon,
        atr_window=config.labeling.atr_window,
        take_profit_atr=config.labeling.take_profit_atr,
        stop_loss_atr=config.labeling.stop_loss_atr,
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
        horizon=horizon,
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
    parser = argparse.ArgumentParser(description="Horizon sweep h=3/5/7/10")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)
    config = Config.load(args.config)

    rows: list[Row] = []
    for h in HORIZONS:
        print(f"\n== Evaluating horizon={h} ==")
        row = _evaluate(config, h)
        rows.append(row)
        print(
            f"   n={row.n_samples}  P(win)={row.p_win:.3f}  AUC={row.auc:.4f}  "
            f"Sharpe={row.sharpe:.3f}  vs B&H={row.bench_sharpe:.3f}  "
            f"beats={row.beats_buy_hold}  trades={row.n_trades}"
        )

    print("\n=== Summary ===")
    print(
        f"{'horizon':<10}{'samples':>9}{'P(win)':>9}{'AUC':>8}{'Sharpe':>9}"
        f"{'B&H':>8}{'beats':>8}{'trades':>9}"
    )
    for r in rows:
        print(
            f"h={r.horizon:<8}{r.n_samples:>9d}{r.p_win:>9.3f}{r.auc:>8.4f}"
            f"{r.sharpe:>9.3f}{r.bench_sharpe:>8.3f}{r.beats_buy_hold!s:>8}{r.n_trades:>9d}"
        )

    winners = [r for r in rows if r.auc > PROMOTE_MIN_AUC and r.sharpe > r.bench_sharpe]
    if winners:
        print(
            f"\n{len(winners)} horizon(s) cleared the promote gate: "
            + ", ".join(f"h={r.horizon}" for r in winners)
        )
    else:
        print(
            f"\nNo horizon cleared the promote gate (AUC > {PROMOTE_MIN_AUC} AND Sharpe > B&H)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
