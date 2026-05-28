"""Sweep PEAD risk-management overlays — the last lever (Phase 8).

Plain PEAD (Phase 7) had Sharpe 0.849 on AUC 0.5346 with no risk overlay.
The hypothesis here is that proper risk management (regime gating, drawdown
protection, Kelly sizing, vol targeting) might lift Sharpe above the
window-B&H benchmark (1.013) even at the cost of total return.

Honest reporting: total return is printed alongside Sharpe in absolute
terms. A risk overlay that lifts Sharpe by butchering returns isn't an
edge — it's a quieter way to lose. The promote gate is still
``Sharpe > window-B&H AND AUC stable``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from berich.backtest.pead_engine import run_risk_aware_pead_backtest
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import EarningsStore, NewsStore
from berich.data.store import OhlcvStore
from berich.datasets.pead import build_pead_dataset
from berich.features.pead_features import PEAD_FEATURE_COLUMNS
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.risk import RiskOverlayConfig
from berich.training.pead import oof_predict_pead


@dataclass
class Variant:
    """One overlay configuration to evaluate."""

    name: str
    config: RiskOverlayConfig


VARIANTS: list[Variant] = [
    Variant(name="off", config=RiskOverlayConfig()),
    Variant(
        name="vol_gate",
        config=RiskOverlayConfig(use_regime_gate=True),
    ),
    Variant(
        name="dd_gate",
        config=RiskOverlayConfig(use_drawdown_gate=True),
    ),
    Variant(
        name="kelly",
        config=RiskOverlayConfig(use_kelly=True, use_inverse_vol=True),
    ),
    Variant(
        name="all",
        config=RiskOverlayConfig(
            use_regime_gate=True,
            use_drawdown_gate=True,
            use_kelly=True,
            use_inverse_vol=True,
            use_vol_target=True,
        ),
    ),
]


@dataclass
class Row:
    name: str
    auc: float
    sharpe: float
    bench_sharpe: float
    total_return: float
    max_dd: float
    n_trades: int
    n_considered: int
    n_gated: int
    beats_buy_hold: bool


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="PEAD + risk-management sweep")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--universe", choices=["mega", "mid", "small", "all"], default="all")
    parser.add_argument("--label-horizon", choices=["5d", "20d"], default="5d")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--name", default="pead-rm")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir)
    news_arg = news_store if news_store.has_any_data() else None
    tickers = config.tickers_for_universe(args.universe)

    print(f"Building PEAD dataset (universe={args.universe}, {len(tickers)} tickers)...")
    dataset = build_pead_dataset(
        store, earnings_store, tickers, news_store=news_arg, label_horizon=args.label_horizon
    )
    if not len(dataset):
        print("No events; check earnings cache.")
        return 1
    print(f"  events={len(dataset)} P(drift)={dataset.y.mean():.3f}")

    print("\nWalk-forward OOS (same model used by every overlay variant)...")
    oof = oof_predict_pead(dataset, LGBMModel, n_folds=5, min_train=500)
    print(f"  OOS AUC: {oof.auc:.4f}")

    rows: list[Row] = []
    for variant in VARIANTS:
        print(f"\n== Overlay: {variant.name} ==")
        result = run_risk_aware_pead_backtest(
            dataset, oof, store, config=variant.config, threshold=args.threshold
        )
        row = Row(
            name=variant.name,
            auc=oof.auc,
            sharpe=result.strategy.sharpe,
            bench_sharpe=result.benchmark.sharpe,
            total_return=result.strategy.total_return,
            max_dd=result.strategy.max_drawdown,
            n_trades=int(result.strategy.n_trades),
            n_considered=result.n_events_considered,
            n_gated=result.n_events_gated,
            beats_buy_hold=result.beats_buy_hold,
        )
        rows.append(row)
        print(
            f"  trades={row.n_trades} (of {row.n_considered}, gated {row.n_gated})  "
            f"Sharpe={row.sharpe:.3f}  total_return={row.total_return:.3f}  "
            f"max_DD={row.max_dd:.3f}  beats_B&H={row.beats_buy_hold}"
        )

    print(
        f"\n{'overlay':<10}{'AUC':>8}{'Sharpe':>9}{'B&H':>8}{'tot_ret':>10}"
        f"{'max_DD':>9}{'trades':>8}{'gated':>8}{'beats':>8}"
    )
    for r in rows:
        print(
            f"{r.name:<10}{r.auc:>8.4f}{r.sharpe:>9.3f}{r.bench_sharpe:>8.3f}"
            f"{r.total_return:>10.3f}{r.max_dd:>9.3f}{r.n_trades:>8d}{r.n_gated:>8d}"
            f"{r.beats_buy_hold!s:>8}"
        )

    winners = [r for r in rows if r.beats_buy_hold and r.n_trades >= 200]  # noqa: PLR2004
    if not winners:
        print("\nNo overlay variant cleared the promote gate (Sharpe > window B&H).")
        return 0

    winners.sort(key=lambda r: r.sharpe, reverse=True)
    best = winners[0]
    print(
        f"\nBest variant by Sharpe: {best.name}  "
        f"Sharpe={best.sharpe:.3f} (B&H {best.bench_sharpe:.3f}) "
        f"total_return={best.total_return:.3f}"
    )

    # Train the final model on the entire event set for the registry. The
    # overlay configuration is recorded in the notes so the deployed setup
    # is reproducible from metadata alone.
    final = LGBMModel().fit(dataset.x, dataset.y)
    meta = ModelMetadata(
        name=f"{args.name}-{best.name}",
        framework="lightgbm",
        feature_columns=list(PEAD_FEATURE_COLUMNS),
        metrics={
            "auc": best.auc,
            "sharpe": best.sharpe,
            "benchmark_sharpe": best.bench_sharpe,
            "total_return": best.total_return,
        },
        beats_buy_hold=best.beats_buy_hold,
        notes=(
            f"pead+risk overlay={best.name} threshold={args.threshold} "
            f"trades={best.n_trades}/{best.n_considered}"
        ),
    )
    save_model(final, meta, registry_dir=config.models_dir)
    try:
        promote(meta.name, registry_dir=config.models_dir, force=False)
        print(f"Promoted '{meta.name}'.")
    except ValueError as exc:
        print(f"Saved '{meta.name}' but registry blocked promotion: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
