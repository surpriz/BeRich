"""Command-line entry point.

v1 exposes a single ``data`` subcommand that refreshes the OHLCV cache. Later
phases add ``train``, ``backtest``, ``signals``, and ``serve`` here.
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import update_watchlist


def _cmd_data(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    reports = update_watchlist(config)
    failed = [r for r in reports if not r.ok]
    print(  # noqa: T201
        f"\n{len(reports)} tickers refreshed, {len(failed)} with warnings."
    )
    return 1 if failed else 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    # Imported lazily so `berich data` doesn't pay for lightgbm/sklearn import time.
    from berich.backtest import BacktestConfig, run_backtest
    from berich.data.store import OhlcvStore
    from berich.datasets import build_dataset
    from berich.labeling.triple_barrier import LabelConfig
    from berich.models import LGBMModel
    from berich.training import oof_predict

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    dataset = build_dataset(store, config.watchlist, label_cfg)
    print(f"Dataset: {len(dataset)} samples, P(win)={dataset.y.mean():.3f}")  # noqa: T201

    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    print(f"Out-of-sample AUC: {oof.auc:.4f}")  # noqa: T201

    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    bt_cfg = BacktestConfig(
        entry_threshold=args.threshold,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
    )
    result = run_backtest(prices, oof, bt_cfg)

    print("\n            strategy   buy&hold")  # noqa: T201
    for key in result.strategy.as_dict():
        s = result.strategy.as_dict()[key]
        b = result.benchmark.as_dict()[key]
        print(f"{key:>14}  {s:9.3f}  {b:9.3f}")  # noqa: T201
    verdict = "BEATS" if result.beats_buy_hold else "does NOT beat"
    print(f"\nStrategy {verdict} buy & hold on Sharpe.")  # noqa: T201
    return 0


def _cmd_signals(args: argparse.Namespace) -> int:
    from berich.data.store import OhlcvStore
    from berich.signals import SignalStore, generate_signals

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    signals = generate_signals(config, store)
    if not signals:
        print("No signals (empty cache?). Run `berich data` first.")  # noqa: T201
        return 1

    saved = SignalStore(config.db_path).save(signals)
    as_of = signals[0].date.date()
    print(f"\nSignals for {as_of} ({saved} saved to {config.db_path}):\n")  # noqa: T201
    header = (
        f"{'TICKER':<8}{'SIGNAL':<9}{'PROBA':>7}{'ENTRY':>10}"
        f"{'STOP':>10}{'TARGET':>10}{'SHARES':>8}"
    )
    print(header)  # noqa: T201
    for s in sorted(signals, key=lambda x: x.proba, reverse=True):
        print(  # noqa: T201
            f"{s.ticker:<8}{s.signal:<9}{s.proba:>7.3f}{s.entry:>10.2f}"
            f"{s.stop_loss:>10.2f}{s.take_profit:>10.2f}{s.size_shares:>8d}"
        )
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    from berich.scheduler.jobs import check_drift_job

    config = Config.load(args.config)
    report = check_drift_job(config)
    frame = report.to_frame()
    print(f"\nFeature drift ({report.n_drifted}/{len(report.features)} drifted):\n")  # noqa: T201
    print(frame.to_string(index=False))  # noqa: T201
    print(f"\nRetrain recommended: {report.should_retrain}")  # noqa: T201
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    from berich.scheduler import build_scheduler

    config = Config.load(args.config)
    scheduler = build_scheduler(config)
    print("Scheduler started (Ctrl-C to stop). Jobs:")  # noqa: T201
    for job in scheduler.get_jobs():
        print(f"  {job.id}: {job.trigger}")  # noqa: T201
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped.")  # noqa: T201
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from berich.api import create_app

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="berich", description="Swing-trading ML advisor")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Refresh the OHLCV cache from yfinance")
    p_data.set_defaults(func=_cmd_data)

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest of the LightGBM baseline")
    p_bt.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Min P(win) to enter a long (default: %(default)s)",
    )
    p_bt.set_defaults(func=_cmd_backtest)

    p_sig = sub.add_parser("signals", help="Generate today's signals for the watchlist")
    p_sig.set_defaults(func=_cmd_signals)

    p_drift = sub.add_parser("drift", help="Check feature drift vs the training era")
    p_drift.set_defaults(func=_cmd_drift)

    p_sched = sub.add_parser("schedule", help="Run the local scheduler (blocking)")
    p_sched.set_defaults(func=_cmd_schedule)

    p_serve = sub.add_parser("serve", help="Run the FastAPI backend")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
