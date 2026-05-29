"""Command-line entry point.

v1 exposes a single ``data`` subcommand that refreshes the OHLCV cache. Later
phases add ``train``, ``backtest``, ``signals``, and ``serve`` here.
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import (
    update_earnings,
    update_news_watchlist,
    update_universe,
    update_watchlist,
)


def _cmd_data(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    if args.universe == "mega":
        reports = update_watchlist(config)
    else:
        tickers = config.tickers_for_universe(args.universe)
        if not tickers:
            print(f"No tickers configured for universe '{args.universe}'.")  # noqa: T201
            return 1
        print(  # noqa: T201
            f"Refreshing {len(tickers)} tickers for universe '{args.universe}' "
            f"(parallel, liquidity gate ON)..."
        )
        reports = update_universe(config, tickers)
    failed = [r for r in reports if not r.ok]
    print(  # noqa: T201
        f"\n{len(reports)} tickers refreshed, {len(failed)} with warnings "
        f"(universe={args.universe})."
    )

    earn_failed: list = []
    if not args.skip_earnings and args.universe == "mega":
        earnings_reports = update_earnings(config)
        earn_failed = [r for r in earnings_reports if not r.ok]
        print(  # noqa: T201
            f"{len(earnings_reports)} earnings calendars refreshed, "
            f"{len(earn_failed)} with warnings."
        )

    news_failed: list = []
    if args.with_news and args.universe == "mega":
        news_reports = update_news_watchlist(config)
        news_failed = [r for r in news_reports if not r.ok]
        print(  # noqa: T201
            f"{len(news_reports)} news feeds refreshed, {len(news_failed)} with warnings"
            " (FinBERT scoring queued; run `berich news score`)."
        )

    return 1 if (failed or earn_failed or news_failed) else 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    # Imported lazily so `berich data` doesn't pay for lightgbm/sklearn import time.
    from berich.backtest import BacktestConfig, run_backtest
    from berich.data import EarningsStore, NewsStore
    from berich.data.store import OhlcvStore
    from berich.datasets import build_dataset
    from berich.features.build import MARKET_TICKER, market_reference_for
    from berich.labeling.triple_barrier import LabelConfig
    from berich.models import LGBMModel
    from berich.training import oof_predict

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())

    asset_class = getattr(args, "asset_class", None)
    if asset_class:
        # Non-equity asset class: resolve from the multi-asset universes, use the
        # class-specific regime proxy, and skip earnings/news (no caches for these).
        tickers = config.universes.get(asset_class)
        market_ticker = market_reference_for(asset_class)
        can_use_extras = False
        scope_label = asset_class
    else:
        tickers = config.tickers_for_universe(args.universe)
        market_ticker = MARKET_TICKER
        # Earnings + news are only valid on the mega-cap watchlist (no caches for
        # the wider universes); silently skip when running on mid/small/all.
        can_use_extras = args.universe == "mega"
        scope_label = args.universe
    earnings_store = (
        EarningsStore(config.earnings_dir) if (args.with_earnings and can_use_extras) else None
    )
    news_store = NewsStore(config.news_dir) if (args.with_news and can_use_extras) else None
    dataset = build_dataset(
        store,
        tickers,
        label_cfg,
        market_ticker=market_ticker,
        earnings_store=earnings_store,
        news_store=news_store,
    )
    bits = ["22"]
    if earnings_store is not None:
        bits.append("earnings")
    if news_store is not None:
        bits.append("news")
    feat_mode = " + ".join(bits)
    print(  # noqa: T201
        f"Dataset: {len(dataset)} samples, P(win)={dataset.y.mean():.3f}, "
        f"features={feat_mode}, universe={scope_label} ({len(tickers)} tickers)"
    )

    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    print(f"Out-of-sample AUC: {oof.auc:.4f}")  # noqa: T201

    prices = {t: df for t in tickers if (df := store.load(t)) is not None and not df.empty}
    bt_cfg = BacktestConfig(
        entry_threshold=args.threshold,
        horizon_days=label_cfg.horizon_days,
        atr_window=label_cfg.atr_window,
        take_profit_atr=label_cfg.take_profit_atr,
        stop_loss_atr=label_cfg.stop_loss_atr,
        volume_proportional_slippage=args.volume_slippage,
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


def _cmd_news(args: argparse.Namespace) -> int:
    import pandas as pd

    from berich.data import NewsStore

    config = Config.load(args.config)
    store = NewsStore(config.news_dir)

    if args.action in {"fetch", "all"}:
        reports = update_news_watchlist(config)
        ok = sum(r.rows_added for r in reports)
        warn = sum(1 for r in reports if not r.ok)
        print(f"news fetch: +{ok} rows total, {warn} tickers warned.")  # noqa: T201

    if args.action in {"score", "all"}:
        from berich.models.finbert_scorer import FinBertScorer

        scorer = FinBertScorer()
        total_scored = 0
        for ticker in config.watchlist:
            df = store.load(ticker)
            if df is None or df.empty:
                continue
            unscored = df[df["finbert_score"].isna()].copy()
            if unscored.empty:
                continue
            # Concatenate title + summary so FinBERT sees more than the headline.
            texts = [
                f"{title} {summary}".strip()
                for title, summary in zip(
                    unscored["title"].fillna("").to_list(),
                    unscored["summary"].fillna("").to_list(),
                    strict=False,
                )
            ]
            probs = scorer.score_texts(texts)
            scores = pd.DataFrame(
                {
                    "url": unscored["url"].to_numpy(),
                    "finbert_neg": probs[:, 0],
                    "finbert_neu": probs[:, 1],
                    "finbert_pos": probs[:, 2],
                    "finbert_score": probs[:, 2] - probs[:, 0],
                }
            )
            updated = store.update_finbert(ticker, scores)
            total_scored += updated
            print(f"  {ticker}: scored {updated} rows")  # noqa: T201
        print(f"FinBERT total: {total_scored} rows scored.")  # noqa: T201
    return 0


def _cmd_pead(args: argparse.Namespace) -> int:
    """Event-level PEAD training / backtest dispatcher.

    ``train`` and ``backtest`` differ only in whether a passing promote gate
    actually writes to the registry — backtest is read-only and useful for
    quick parameter sweeps without risking accidental promotion.
    """
    import numpy as np

    from berich.data import EarningsStore, NewsStore
    from berich.data.store import OhlcvStore
    from berich.datasets.pead import build_pead_dataset
    from berich.features.pead_features import PEAD_FEATURE_COLUMNS
    from berich.models import LGBMModel, ModelMetadata, promote, save_model
    from berich.training.pead import oof_predict_pead, run_pead_backtest

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir)
    news_arg = news_store if news_store.has_any_data() else None

    tickers = config.tickers_for_universe(args.universe)
    print(  # noqa: T201
        f"PEAD universe={args.universe} tickers={len(tickers)} label={args.label_horizon}"
    )
    dataset = build_pead_dataset(
        store, earnings_store, tickers, news_store=news_arg, label_horizon=args.label_horizon
    )
    if not len(dataset):
        print("No PEAD events; run `berich data --universe ... ` then refetch earnings.")  # noqa: T201
        return 1
    print(f"  events={len(dataset)} P(drift)={dataset.y.mean():.3f}")  # noqa: T201

    oof = oof_predict_pead(dataset, LGBMModel, n_folds=5, min_train=500)
    bt = run_pead_backtest(dataset, oof, store, threshold=args.threshold)
    print(  # noqa: T201
        f"  OOS AUC={oof.auc:.4f}  Sharpe={bt.strategy.sharpe:.3f}  "
        f"vs window B&H={bt.benchmark.sharpe:.3f}  trades={bt.strategy.n_trades:.0f}"
    )
    final = LGBMModel().fit(dataset.x, dataset.y)
    imp = np.asarray(final.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1][:5]
    print(  # noqa: T201
        "  top features: "
        + ", ".join(f"{PEAD_FEATURE_COLUMNS[i]}={imp[i] / imp.sum() * 100:.1f}%" for i in order)
    )

    if args.action == "backtest":
        return 0

    gate_pass = (
        oof.auc > PROMOTE_MIN_AUC and bt.beats_buy_hold and len(dataset) >= PROMOTE_MIN_EVENTS_PEAD
    )
    print(  # noqa: T201
        f"  Gate (AUC > {PROMOTE_MIN_AUC} AND Sharpe > window B&H AND events"
        f" >= {PROMOTE_MIN_EVENTS_PEAD}): {gate_pass}"
    )
    if not gate_pass:
        print("  No promotion.")  # noqa: T201
        return 0

    meta = ModelMetadata(
        name=args.name,
        framework="lightgbm",
        feature_columns=list(PEAD_FEATURE_COLUMNS),
        metrics={
            "auc": oof.auc,
            "sharpe": bt.strategy.sharpe,
            "benchmark_sharpe": bt.benchmark.sharpe,
        },
        beats_buy_hold=bt.beats_buy_hold,
        notes=f"pead label={args.label_horizon} universe={args.universe}",
    )
    save_model(final, meta, registry_dir=config.models_dir)
    try:
        promote(args.name, registry_dir=config.models_dir, force=False)
        print(f"  Promoted '{args.name}'.")  # noqa: T201
    except ValueError as exc:
        print(f"  Saved '{args.name}' but registry blocked promotion: {exc}")  # noqa: T201
    return 0


PROMOTE_MIN_AUC = 0.55
PROMOTE_MIN_EVENTS_PEAD = 1000


def _cmd_longshort(args: argparse.Namespace) -> int:
    """Market-neutral long/short cross-sectional training / backtest dispatcher.

    ``backtest`` is read-only (OOS rank-IC + Sharpe-significance report); ``train``
    additionally saves a ``market_neutral`` artifact and runs it through the
    Sharpe-significance promotion gate (no buy-&-hold benchmark — see RESULTS.md).
    """
    from berich.backtest.longshort import LongShortConfig, run_longshort_backtest
    from berich.data.store import OhlcvStore
    from berich.datasets.cross_sectional import build_panel_dataset
    from berich.features.build import FEATURE_COLUMNS
    from berich.labeling.cross_sectional import CrossSectionalLabelConfig
    from berich.models import ModelMetadata, promote, save_model
    from berich.models.lightgbm_ranker import LGBMRanker
    from berich.training.cross_sectional import oof_predict_cross_sectional

    config = Config.load(args.config)
    ls = config.longshort
    store = OhlcvStore(config.ohlcv_dir)
    universe = args.universe or ls.universe
    tickers = config.tickers_for_universe(universe)
    label_cfg = CrossSectionalLabelConfig(
        horizon_days=ls.horizon_days,
        beta_window=ls.beta_window,
        residualize=ls.residualize,
        standardize="rank" if ls.standardize == "rank" else "zscore",
    )
    print(f"long/short universe={universe} tickers={len(tickers)} horizon={ls.horizon_days}d")  # noqa: T201
    panel = build_panel_dataset(
        store,
        tickers,
        label_cfg,
        market_ticker=ls.market_ticker,
        min_names_per_date=ls.min_names_per_date,
    )
    if not len(panel):
        print("Empty panel — too few names per date or no cached OHLCV. Run `berich data`.")  # noqa: T201
        return 1
    print(f"  panel rows={len(panel)} dates={panel.dates.nunique()}")  # noqa: T201

    oof = oof_predict_cross_sectional(panel, LGBMRanker, embargo=ls.horizon_days)
    prices = {t: df for t in tickers if (df := store.load(t)) is not None}
    bt_cfg = LongShortConfig(
        top_decile=ls.top_decile,
        bottom_decile=ls.bottom_decile,
        weighting=ls.weighting,
        rebalance_days=ls.rebalance_days,
        gross_leverage=ls.gross_leverage,
        target_vol=ls.target_vol,
        vol_lookback=ls.vol_lookback,
        fee_bps=ls.fee_bps,
        slippage_bps=ls.slippage_bps,
        borrow_bps_annual=ls.borrow_bps_annual,
        min_names=ls.min_names_per_date,
    )
    res = run_longshort_backtest(prices, oof, bt_cfg, n_trials=ls.n_trials)
    sig = res.significance
    print(  # noqa: T201
        f"  rank_IC={oof.rank_ic:.4f} (t={oof.ic_t_stat:.2f})  Sharpe={sig.sharpe:.3f}  "
        f"DSR={sig.deflated_sharpe:.3f}  p={sig.p_value:.3f}  boot_p={sig.bootstrap_p_value:.3f}"
    )
    print(  # noqa: T201
        f"  total_return={res.metrics.total_return:.1%}  maxDD={res.metrics.max_drawdown:.1%}  "
        f"rebalances={res.n_rebalances}  avg_gross={res.avg_gross_exposure:.2f}"
    )

    if args.action == "backtest":
        return 0

    registry_dir = config.models_dir_for("longshort")
    final = LGBMRanker().fit(panel.x, panel.y, sample_weight=panel.weight, tickers=panel.tickers)
    metrics = {**sig.as_dict(), "rank_ic": oof.rank_ic, "ic_t_stat": oof.ic_t_stat}
    meta = ModelMetadata(
        name=args.name,
        framework="lightgbm-ranker",
        feature_columns=list(FEATURE_COLUMNS),
        metrics=metrics,
        beats_buy_hold=False,  # not applicable — market-neutral gate is Sharpe-significance
        strategy_type="market_neutral",
        notes=f"longshort universe={universe} horizon={ls.horizon_days}d n_trials={ls.n_trials}",
    )
    save_model(final, meta, registry_dir=registry_dir)
    try:
        promote(args.name, registry_dir=registry_dir, force=args.force)
        print(f"  Promoted '{args.name}' (market_neutral).")  # noqa: T201
    except ValueError as exc:
        print(f"  Saved '{args.name}' but registry blocked promotion: {exc}")  # noqa: T201
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:  # noqa: C901,PLR0911,PLR0915 — multi-action dispatcher
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

    if args.action == "calibration":
        from berich.signals import compute_calibration

        report = compute_calibration(config)
        print(  # noqa: T201
            f"\nPaper calibration ({report.n_with_proba} of {report.n_trades_total} "
            f"closed trades have a recorded proba)\n"
        )
        if report.n_with_proba == 0:
            print("  No closed trades with a recorded proba yet.")  # noqa: T201
            return 0
        print(f"  {'bucket':<14}{'n':>5}{'mean_pred':>12}{'win_rate':>11}{'gap':>8}")  # noqa: T201
        for b in report.buckets:
            gap = b.win_rate - b.mean_predicted if b.n_trades else 0.0
            print(  # noqa: T201
                f"  [{b.low:.2f}, {b.high:.2f})  {b.n_trades:>5d}"
                f"{b.mean_predicted:>12.3f}{b.win_rate:>11.3f}{gap:>+8.3f}"
            )
        verdict = "OK" if report.is_well_calibrated else "OFF — model proba ≠ outcomes"
        print(f"\n  Verdict: {verdict}")  # noqa: T201
        return 0

    if args.action == "export":
        df = PaperStore(config.db_path).all_trades()
        if df.empty:
            print("No paper trades to export.")  # noqa: T201
            return 0
        df.to_csv(args.output, index=False)
        print(f"Exported {len(df)} paper trades to {args.output}")  # noqa: T201
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
    from berich.data import EarningsStore, NewsStore
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
    news_store = NewsStore(config.news_dir) if args.with_news else None
    dataset = build_dataset(
        store,
        config.watchlist,
        label_cfg,
        earnings_store=earnings_store,
        news_store=news_store,
    )
    oof = oof_predict(dataset, LGBMModel, embargo=label_cfg.horizon_days)
    prices = {t: df for t in config.watchlist if (df := store.load(t)) is not None}
    result = run_backtest(prices, oof, BacktestConfig(entry_threshold=0.5))

    # Final model trained on all labeled history for serving.
    model = LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)
    meta = ModelMetadata(
        name=args.name,
        framework="lightgbm",
        feature_columns=feature_columns(earnings=args.with_earnings, news=args.with_news),
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


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 — flat subcommand registry
    parser = argparse.ArgumentParser(prog="berich", description="Swing-trading ML advisor")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Refresh OHLCV + earnings (+ optional news) caches")
    p_data.add_argument(
        "--skip-earnings",
        action="store_true",
        help="Skip the earnings-calendar refresh (Phase 5a). OHLCV only.",
    )
    p_data.add_argument(
        "--with-news",
        action="store_true",
        help="Also fetch news via Alpha Vantage (Phase 5b). Costs API budget"
        " (25 req/day free tier). Run `berich news score` afterwards to FinBERT.",
    )
    p_data.add_argument(
        "--universe",
        choices=["mega", "mid", "small", "all"],
        default="mega",
        help="Which universe to refresh (Phase 6). Default 'mega' = the existing"
        " 10-ticker watchlist. 'mid'/'small'/'all' use the wider lists from"
        " config and apply liquidity gates; earnings + news refresh only run"
        " in 'mega' mode (those data sources are scoped to the watchlist).",
    )
    p_data.set_defaults(func=_cmd_data)

    p_news = sub.add_parser("news", help="News pipeline (Alpha Vantage fetch + FinBERT GPU score)")
    p_news.add_argument(
        "action",
        choices=["fetch", "score", "all"],
        help="fetch = AV refresh only; score = FinBERT on unscored rows;"
        " all = both (use after `berich data`)",
    )
    p_news.set_defaults(func=_cmd_news)

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
    p_bt.add_argument(
        "--with-news",
        action="store_true",
        help="Include the 7 news/sentiment features (Phase 5b). Requires the"
        " news cache to be populated and FinBERT-scored.",
    )
    p_bt.add_argument(
        "--universe",
        choices=["mega", "mid", "small", "all"],
        default="mega",
        help="Which universe to backtest (Phase 6). The benchmark buy & hold"
        " uses the same universe as the strategy for a fair comparison.",
    )
    p_bt.add_argument(
        "--volume-slippage",
        action="store_true",
        help="Use the volume-proportional slippage model (Phase 6). Recommended"
        " for wide universes — small-caps pay more per side than mega-caps.",
    )
    p_bt.add_argument(
        "--asset-class",
        choices=["crypto", "forex", "commodities", "fr_stocks"],
        default=None,
        help="Backtest a non-US-equity universe (Phase 10). Resolves tickers from the"
        " multi-asset config, uses a class-specific regime proxy (e.g. BTC for crypto),"
        " and disables earnings/news. Overrides --universe when set.",
    )
    p_bt.set_defaults(func=_cmd_backtest)

    p_sig = sub.add_parser("signals", help="Generate today's signals for the watchlist")
    p_sig.set_defaults(func=_cmd_signals)

    p_drift = sub.add_parser("drift", help="Check feature drift vs the training era")
    p_drift.set_defaults(func=_cmd_drift)

    p_pead = sub.add_parser("pead", help="Event-driven Post-Earnings Drift model (Phase 7)")
    p_pead.add_argument(
        "action",
        choices=["train", "backtest"],
        help="train = walk-forward + promote-gated train; backtest = OOS + report only.",
    )
    p_pead.add_argument(
        "--universe",
        choices=["mega", "mid", "small", "all"],
        default="all",
    )
    p_pead.add_argument("--label-horizon", choices=["5d", "20d"], default="5d")
    p_pead.add_argument("--threshold", type=float, default=0.5)
    p_pead.add_argument("--name", default="pead-lgbm")
    p_pead.set_defaults(func=_cmd_pead)

    p_ls = sub.add_parser(
        "longshort", help="Market-neutral long/short cross-sectional model (Phase 10)"
    )
    p_ls.add_argument(
        "action",
        choices=["train", "backtest"],
        help="train = walk-forward + significance-gated promote; backtest = OOS + report only.",
    )
    p_ls.add_argument(
        "--universe",
        choices=["mega", "mid", "small", "all"],
        default=None,
        help="Override the configured longshort universe (default: config value, 'all').",
    )
    p_ls.add_argument("--name", default="longshort-ranker")
    p_ls.add_argument(
        "--force", action="store_true", help="Promote even if the significance gate fails"
    )
    p_ls.set_defaults(func=_cmd_longshort)

    p_paper = sub.add_parser("paper", help="Paper-trading tracker (no real money)")
    p_paper.add_argument(
        "action",
        choices=["update", "status", "equity", "calibration", "export"],
        help="update = open new BUY signals + walk open trades; status = list "
        "positions + recent closes; equity = summary metrics vs SPY benchmark; "
        "calibration = predicted-proba vs realized win-rate buckets; export = "
        "dump all paper_trades rows to CSV (use --output).",
    )
    p_paper.add_argument(
        "--output",
        default="paper_trades.csv",
        help="Output file for the `export` action (default: %(default)s).",
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
    p_train.add_argument(
        "--with-news",
        action="store_true",
        help="Include the 7 news/sentiment features (Phase 5b).",
    )
    p_train.set_defaults(func=_cmd_train)

    p_models = sub.add_parser("models", help="List models in the registry")
    p_models.set_defaults(func=_cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs at INFO, which would leak the Alpha Vantage
    # API key into journalctl. Silence the noisy third-party loggers — our own
    # modules log the meaningful "news AAPL: +N rows" lines themselves.
    for noisy in ("httpx", "httpcore", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
