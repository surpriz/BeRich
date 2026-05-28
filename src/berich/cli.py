"""Command-line entry point.

v1 exposes a single ``data`` subcommand that refreshes the OHLCV cache. Later
phases add ``train``, ``backtest``, ``signals``, and ``serve`` here.
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import update_earnings, update_watchlist


def _cmd_data(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    reports = update_watchlist(config)
    failed = [r for r in reports if not r.ok]
    print(f"\n{len(reports)} tickers refreshed, {len(failed)} with warnings.")  # noqa: T201
    if args.skip_earnings:
        return 1 if failed else 0
    earnings_reports = update_earnings(config)
    earn_failed = [r for r in earnings_reports if not r.ok]
    print(  # noqa: T201
        f"{len(earnings_reports)} earnings calendars refreshed, {len(earn_failed)} with warnings."
    )
    return 1 if (failed or earn_failed) else 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    # Imported lazily so `berich data` doesn't pay for lightgbm/sklearn import time.
    from berich.backtest import BacktestConfig, run_backtest
    from berich.data import EarningsStore
    from berich.data.store import OhlcvStore
    from berich.datasets import build_dataset
    from berich.labeling.triple_barrier import LabelConfig
    from berich.models import LGBMModel
    from berich.training import oof_predict

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    earnings_store = EarningsStore(config.earnings_dir) if args.with_earnings else None
    dataset = build_dataset(store, config.watchlist, label_cfg, earnings_store=earnings_store)
    feat_mode = "22 + earnings" if args.with_earnings else "22"
    print(  # noqa: T201
        f"Dataset: {len(dataset)} samples, P(win)={dataset.y.mean():.3f}, features={feat_mode}"
    )

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


def _cmd_paper(args: argparse.Namespace) -> int:
    from berich.data.store import OhlcvStore
    from berich.signals import (
        SignalStore,
        get_open_positions,
        get_paper_metrics,
        open_new_trades,
        update_open_trades,
    )
    from berich.signals.paper import PaperStore

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)

    if args.action == "update":
        signal_store = SignalStore(config.db_path)
        opened = open_new_trades(config, store, signal_store)
        closed = update_open_trades(config, store)
        print(f"paper: {opened} opened, {closed} closed.")  # noqa: T201
        return 0

    if args.action == "status":
        positions = get_open_positions(config, store)
        print(f"\nOpen positions ({len(positions)}):\n")  # noqa: T201
        if positions:
            header = (
                f"{'OPENED':<12}{'TICKER':<8}{'ENTRY':>9}{'CURRENT':>10}"
                f"{'MTM%':>8}{'MTM€':>10}{'DAYS':>6}"
            )
            print(header)  # noqa: T201
            for p in positions:
                print(  # noqa: T201
                    f"{p.date_open.date().isoformat():<12}{p.ticker:<8}"
                    f"{p.entry:>9.2f}{p.current_price:>10.2f}"
                    f"{p.mtm_pct * 100:>7.2f}%{p.mtm_eur:>10.2f}{p.days_held:>6d}"
                )

        closed = PaperStore(config.db_path).closed_trades(limit=10)
        print(f"\nRecent closed trades ({len(closed)}):\n")  # noqa: T201
        if not closed.empty:
            header = (
                f"{'CLOSED':<12}{'TICKER':<8}{'STATUS':<16}"
                f"{'ENTRY':>9}{'EXIT':>9}{'PNL%':>8}{'PNL€':>10}"
            )
            print(header)  # noqa: T201
            for _, r in closed.iterrows():
                print(  # noqa: T201
                    f"{r['date_close']!s:<12}{r['ticker']:<8}{r['status']:<16}"
                    f"{r['entry']:>9.2f}{r['exit_price']:>9.2f}"
                    f"{r['pnl_pct'] * 100:>7.2f}%{r['pnl_eur']:>10.2f}"
                )
        return 0

    # "equity" branch
    metrics = get_paper_metrics(config, store)
    print("\nPaper trading equity summary:\n")  # noqa: T201
    print(f"  capital starting    {metrics['capital']:>12.2f}")  # noqa: T201
    print(f"  open trades         {metrics['n_open']:>12d}")  # noqa: T201
    print(f"  closed trades       {metrics['n_closed']:>12d}")  # noqa: T201
    print(f"  win rate            {metrics['win_rate'] * 100:>11.2f}%")  # noqa: T201
    print(f"  total return paper  {metrics['total_return_paper'] * 100:>11.2f}%")  # noqa: T201
    import math

    spy = float(metrics["total_return_spy"])
    spy_str = f"{'n/a':>12}" if math.isnan(spy) else f"{spy * 100:>11.2f}%"
    print(f"  total return SPY    {spy_str}")  # noqa: T201
    print(f"  max drawdown paper  {metrics['max_drawdown_paper'] * 100:>11.2f}%")  # noqa: T201
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


def _cmd_train(args: argparse.Namespace) -> int:
    from berich.backtest import BacktestConfig, run_backtest
    from berich.data import EarningsStore
    from berich.data.store import OhlcvStore
    from berich.datasets import build_dataset
    from berich.features.build import feature_columns
    from berich.labeling.triple_barrier import LabelConfig
    from berich.models import LGBMModel, ModelMetadata, promote, save_model
    from berich.training import oof_predict

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    earnings_store = EarningsStore(config.earnings_dir) if args.with_earnings else None
    dataset = build_dataset(store, config.watchlist, label_cfg, earnings_store=earnings_store)
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    result = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))

    # Final model trained on all labeled history for serving.
    model = LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    meta = ModelMetadata(
        name=args.name,
        framework="lightgbm",
        feature_columns=feature_columns(earnings=args.with_earnings),
        metrics={
            "auc": oof.auc,
            "sharpe": result.strategy.sharpe,
            "benchmark_sharpe": result.benchmark.sharpe,
        },
        beats_buy_hold=result.beats_buy_hold,
    )
    save_model(model, meta, registry_dir=config.models_dir)
    print(  # noqa: T201
        f"Saved '{args.name}': AUC={oof.auc:.4f}, "
        f"Sharpe={result.strategy.sharpe:.3f} vs {result.benchmark.sharpe:.3f}, "
        f"beats_buy_hold={result.beats_buy_hold}"
    )
    try:
        promote(args.name, registry_dir=config.models_dir, force=args.force)
        print(f"Promoted '{args.name}' as the active serving model.")  # noqa: T201
    except ValueError as exc:
        print(f"Not promoted: {exc}")  # noqa: T201
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    import json

    from berich.models import list_models

    config = Config.load(args.config)
    metas = list_models(config.models_dir)
    pointer = config.models_dir / "active.json"
    active = json.loads(pointer.read_text())["name"] if pointer.exists() else None
    if not metas:
        print("No models in the registry. Run `berich train`.")  # noqa: T201
        return 0
    print(f"{'NAME':<20}{'FRAMEWORK':<12}{'AUC':>8}{'BEATS B&H':>11}{'ACTIVE':>8}")  # noqa: T201
    for m in metas:
        star = "  <--" if m.name == active else ""
        print(  # noqa: T201
            f"{m.name:<20}{m.framework:<12}{m.metrics.get('auc', float('nan')):>8.4f}"
            f"{m.beats_buy_hold!s:>11}{star:>8}"
        )
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

    p_data = sub.add_parser("data", help="Refresh OHLCV + earnings caches from yfinance")
    p_data.add_argument(
        "--skip-earnings",
        action="store_true",
        help="Skip the earnings-calendar refresh (Phase 5a). OHLCV only.",
    )
    p_data.set_defaults(func=_cmd_data)

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest of the LightGBM baseline")
    p_bt.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Min P(win) to enter a long (default: %(default)s)",
    )
    p_bt.add_argument(
        "--with-earnings",
        action="store_true",
        help="Include the 6 earnings features (Phase 5a). Default off for parity"
        " with the v0.1.0 baseline.",
    )
    p_bt.set_defaults(func=_cmd_backtest)

    p_sig = sub.add_parser("signals", help="Generate today's signals for the watchlist")
    p_sig.set_defaults(func=_cmd_signals)

    p_drift = sub.add_parser("drift", help="Check feature drift vs the training era")
    p_drift.set_defaults(func=_cmd_drift)

    p_paper = sub.add_parser("paper", help="Paper-trading tracker (no real money)")
    p_paper.add_argument(
        "action",
        choices=["update", "status", "equity"],
        help="update = open new BUY signals + walk open trades; status = list "
        "positions + recent closes; equity = summary metrics vs SPY benchmark",
    )
    p_paper.set_defaults(func=_cmd_paper)

    p_sched = sub.add_parser("schedule", help="Run the local scheduler (blocking)")
    p_sched.set_defaults(func=_cmd_schedule)

    p_serve = sub.add_parser("serve", help="Run the FastAPI backend")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_train = sub.add_parser("train", help="Train the baseline, save to the registry, promote")
    p_train.add_argument("--name", default="lgbm-baseline", help="Artifact name")
    p_train.add_argument(
        "--force",
        action="store_true",
        help="Promote even if it does not beat buy & hold",
    )
    p_train.add_argument(
        "--with-earnings",
        action="store_true",
        help="Include the 6 earnings features. The artifact metadata records"
        " the choice so serving stays in sync.",
    )
    p_train.set_defaults(func=_cmd_train)

    p_models = sub.add_parser("models", help="List models in the registry")
    p_models.set_defaults(func=_cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
