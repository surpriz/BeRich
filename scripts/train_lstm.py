"""Train + evaluate + (gated-)promote the LSTM baseline against LightGBM and B&H.

This is the Phase 3b driver. It captures the LightGBM baseline Sharpe in a first
walk-forward pass so the guard rule has both bars to beat — LightGBM AND buy &
hold — before the LSTM can be promoted.

Run with: ``uv run python scripts/train_lstm.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, LSTMConfig
from berich.training import oof_predict
from berich.training.deep import train_lstm


def _baseline_sharpe(config: Config) -> float:
    """Re-run the LightGBM baseline OOS + backtest to capture today's Sharpe-to-beat."""
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))
    print(
        f"LightGBM baseline: AUC={oof.auc:.4f} Sharpe={bt.strategy.sharpe:.3f} "
        f"vs B&H Sharpe={bt.benchmark.sharpe:.3f}"
    )
    return bt.strategy.sharpe


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train LSTM swing classifier")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--name", default="lstm-baseline")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip-baseline", action="store_true", help="skip baseline recomputation")
    parser.add_argument("--force", action="store_true", help="bypass guard (last resort)")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    baseline = None if args.skip_baseline else _baseline_sharpe(config)

    lstm_cfg = LSTMConfig(
        lookback=args.lookback,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    result = train_lstm(
        config,
        name=args.name,
        lstm_cfg=lstm_cfg,
        baseline_sharpe=baseline,
        force_promote=args.force,
    )

    verdict = "PROMOTED" if result.promoted else "not promoted"
    base = "n/a" if baseline is None else f"{baseline:.3f}"
    print(
        f"\nLSTM '{result.name}' [{verdict}]: "
        f"AUC={result.oos_auc:.4f} "
        f"Sharpe={result.strategy_sharpe:.3f}  "
        f"(baseline={base}, B&H={result.benchmark_sharpe:.3f})  "
        f"beats_buy_hold={result.beats_buy_hold}  "
        f"beats_baseline={result.beats_baseline}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
