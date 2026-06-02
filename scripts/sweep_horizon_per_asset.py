"""Throwaway exploration: does a SHORTER horizon (5d) give a per-asset edge the 10d misses?

For each requested ticker, build the single-asset dataset at horizon h, run the per-asset
walk-forward OOS (LightGBM, default-regularized), backtest long with realistic costs, and
print AUC / strategy-Sharpe / buy&hold-Sharpe / beats-B&H for h in {5, 10} side by side.

This does NOT touch production (no registry writes, no promotion, no scheduler). It only
answers "is 5d worth investing in?" before we build multi-horizon support for real.

Usage:
    uv run python scripts/sweep_horizon_per_asset.py --tickers AAPL MSFT NVDA --horizons 5 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from berich.backtest import BacktestConfig, run_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore
from berich.datasets.assemble import build_ticker_dataset
from berich.features.build import market_reference_for
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel
from berich.training import oof_predict

logger = logging.getLogger("sweep_horizon_per_asset")


@dataclass
class Row:
    ticker: str
    horizon: int
    n_samples: int
    p_win: float
    auc: float
    sharpe: float
    bench_sharpe: float
    beats_buy_hold: bool
    n_trades: int


def _evaluate(config: Config, store: OhlcvStore, ticker: str, horizon: int) -> Row | None:
    df = store.load(ticker)
    if df is None or df.empty:
        logger.warning("no OHLCV for %s; skipping", ticker)
        return None
    market = store.load(market_reference_for(config.asset_class_for(ticker)))
    label_cfg = LabelConfig(
        horizon_days=horizon,
        atr_window=config.labeling.atr_window,
        take_profit_atr=config.labeling.take_profit_atr,
        stop_loss_atr=config.labeling.stop_loss_atr,
    )
    # micro=True matches the per-asset tournament's default feature set.
    dataset = build_ticker_dataset(df, label_cfg, ticker=ticker, market=market, micro=True)
    if len(dataset) < 250:  # noqa: PLR2004 — same thin-data floor as the tournament
        logger.warning("%s h=%d: only %d labeled rows; skipping", ticker, horizon, len(dataset))
        return None
    oof = oof_predict(dataset, LGBMModel, embargo=horizon)
    bt = run_backtest(
        {ticker: df},
        oof,
        BacktestConfig(
            entry_threshold=config.signals.buy_threshold,
            horizon_days=horizon,
            atr_window=label_cfg.atr_window,
            take_profit_atr=label_cfg.take_profit_atr,
            stop_loss_atr=label_cfg.stop_loss_atr,
            direction="long",
        ),
    )
    return Row(
        ticker=ticker,
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
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10])
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)

    rows: list[Row] = []
    for ticker in args.tickers:
        for h in args.horizons:
            row = _evaluate(config, store, ticker, h)
            if row is not None:
                rows.append(row)

    header = (
        f"{'ticker':<10}{'h':>4}{'samples':>9}{'P(win)':>8}"
        f"{'AUC':>8}{'Sharpe':>9}{'B&H':>8}{'beats':>7}{'trades':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.ticker:<10}{r.horizon:>4}{r.n_samples:>9}{r.p_win:>8.3f}"
            f"{r.auc:>8.4f}{r.sharpe:>9.3f}{r.bench_sharpe:>8.3f}"
            f"{'YES' if r.beats_buy_hold else 'no':>7}{r.n_trades:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
