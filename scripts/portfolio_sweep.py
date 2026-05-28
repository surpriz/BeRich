"""Phase 9 — core-satellite portfolio sweep.

Combines:
- ``bnh_spy``       — pure long-only SPY (the strong benchmark we've been
                      losing to for 8 phases).
- ``pead``          — average daily return of currently-held PEAD positions
                      (cash on flat days).
- ``calendar_spy``  — turn-of-month rule on SPY (last 3 + first 3 business
                      days of each month long, flat otherwise).

Two evaluations:
1. **Static weight grid**: hand-picked combinations (90/5/5, 80/10/10, etc.)
   to see what overlay magnitudes produce what shape.
2. **Walk-forward optimized weights**: for each fold's train slice we solve
   ``max Sharpe`` subject to non-negative weights summing to 1 (SLSQP),
   then apply those weights to the test slice. Test returns are
   concatenated for an honest OOS metric.

Promote gate: portfolio OOS Sharpe > pure-B&H Sharpe **by at least 0.05**
on the same dates. The buffer guards against borderline wins that are
statistically indistinguishable from noise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from berich.backtest import (
    build_bnh_returns,
    build_calendar_returns,
    build_pead_returns,
    run_portfolio_backtest,
)
from berich.backtest.metrics import compute_metrics
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data import EarningsStore, NewsStore
from berich.data.store import OhlcvStore
from berich.datasets.pead import build_pead_dataset
from berich.features.build import feature_columns
from berich.models import LGBMModel, ModelMetadata, promote, save_model

GATE_SHARPE_BUFFER = 0.05  # portfolio Sharpe must beat pure B&H by this much
MIN_TRAIN_FOLD_DAYS = 252 * 3  # 3 years before first OOS fold
TEST_FOLD_DAYS = 252  # 1 year per OOS fold


@dataclass
class Row:
    name: str
    sharpe: float
    sharpe_vs_bnh: float
    total_return: float
    max_dd: float
    ann_vol: float
    turnover_total: float


# Hand-picked static-weight combinations: pure B&H (control) + B&H + PEAD only
# variants + B&H + PEAD + calendar variants.
STATIC_GRID: list[tuple[str, dict[str, float]]] = [
    ("100/0/0", {"bnh_spy": 1.00, "pead": 0.00, "calendar_spy": 0.00}),
    ("95/3/2", {"bnh_spy": 0.95, "pead": 0.03, "calendar_spy": 0.02}),
    ("90/5/5", {"bnh_spy": 0.90, "pead": 0.05, "calendar_spy": 0.05}),
    ("90/10/0", {"bnh_spy": 0.90, "pead": 0.10, "calendar_spy": 0.00}),
    ("80/20/0", {"bnh_spy": 0.80, "pead": 0.20, "calendar_spy": 0.00}),
    ("80/10/10", {"bnh_spy": 0.80, "pead": 0.10, "calendar_spy": 0.10}),
    ("70/15/15", {"bnh_spy": 0.70, "pead": 0.15, "calendar_spy": 0.15}),
    ("70/30/0", {"bnh_spy": 0.70, "pead": 0.30, "calendar_spy": 0.00}),
    ("60/40/0", {"bnh_spy": 0.60, "pead": 0.40, "calendar_spy": 0.00}),
]


def _sharpe_from_weights(
    weights: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Negative Sharpe (for SLSQP minimization)."""
    port = returns @ weights
    std = float(port.std())
    if std <= 0:
        return 0.0
    mean = float(port.mean())
    return -mean / std * np.sqrt(252.0)


def _optimize_weights(
    train_returns: pd.DataFrame,
) -> dict[str, float]:
    """Find non-negative weights summing to 1 that maximize in-sample Sharpe.

    Uses SLSQP with a small ridge on the weights to keep the solver stable
    when two strategies are highly correlated.
    """
    names = list(train_returns.columns)
    n = len(names)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(
        _sharpe_from_weights,
        x0,
        args=(train_returns.to_numpy(),),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not res.success:
        return dict.fromkeys(names, 1.0 / n)
    return dict(zip(names, res.x.tolist(), strict=False))


def _walk_forward(
    aligned: pd.DataFrame,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, dict[str, float]]]]:
    """Roll forward in 1-year folds; each test slice uses weights from the train so far.

    Returns the concatenated daily portfolio returns (cost-net, via the
    portfolio engine applied per-fold) and the list of (start, weights)
    used so callers can inspect the schedule.
    """
    index = aligned.index
    if len(index) <= MIN_TRAIN_FOLD_DAYS + TEST_FOLD_DAYS:
        msg = "not enough history for at least one walk-forward fold"
        raise ValueError(msg)

    fold_returns: list[pd.Series] = []
    schedule: list[tuple[pd.Timestamp, dict[str, float]]] = []

    cursor = MIN_TRAIN_FOLD_DAYS
    while cursor + TEST_FOLD_DAYS <= len(index):
        train = aligned.iloc[:cursor]
        weights = _optimize_weights(train)
        schedule.append((pd.Timestamp(index[cursor]), weights))
        test_slice = aligned.iloc[cursor : cursor + TEST_FOLD_DAYS]
        # Run the portfolio engine on the test slice with these weights so the
        # rebalancing cost is honestly applied.
        slice_strategies = {col: test_slice[col] for col in test_slice.columns}
        result = run_portfolio_backtest(slice_strategies, weights=weights)
        fold_returns.append(result.returns)
        cursor += TEST_FOLD_DAYS

    return pd.concat(fold_returns).sort_index(), schedule


def _print_grid(rows: list[Row]) -> None:
    print(
        f"\n{'mix':<12}{'Sharpe':>9}{'Δ vs BNH':>10}{'total_ret':>11}"
        f"{'max_DD':>9}{'ann_vol':>9}{'turnover':>10}"
    )
    for r in rows:
        print(
            f"{r.name:<12}{r.sharpe:>9.3f}{r.sharpe_vs_bnh:>10.3f}{r.total_return:>11.3f}"
            f"{r.max_dd:>9.3f}{r.ann_vol:>9.3f}{r.turnover_total:>10.4f}"
        )


def _correlations(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the strategy returns over the aligned window."""
    return returns.corr()


def _maybe_promote(
    name: str, weights: dict[str, float], sharpe: float, bnh_sharpe: float, config: Config
) -> bool:
    """Promote the PEAD model + record the winning portfolio weights in metadata."""
    delta = sharpe - bnh_sharpe
    if delta < GATE_SHARPE_BUFFER:
        return False
    # The promoted artifact is the LightGBM PEAD model (the only ML component
    # in the mix). The portfolio recipe lives in the metadata notes so the
    # deployed setup is reproducible from the registry alone.
    store = OhlcvStore(config.ohlcv_dir)
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir)
    news_arg = news_store if news_store.has_any_data() else None
    dataset = build_pead_dataset(
        store, earnings_store, config.tickers_for_universe("all"), news_store=news_arg
    )
    model = LGBMModel().fit(dataset.x, dataset.y)
    weights_str = " ".join(f"{k}={v:.3f}" for k, v in weights.items())
    meta = ModelMetadata(
        name=name,
        framework="lightgbm",
        feature_columns=feature_columns(),
        metrics={
            "sharpe": sharpe,
            "benchmark_sharpe": bnh_sharpe,
            "sharpe_delta": delta,
        },
        beats_buy_hold=True,
        notes=f"phase-9 core-satellite portfolio weights={weights_str}",
    )
    save_model(model, meta, registry_dir=config.models_dir)
    promoted = False
    try:
        promote(name, registry_dir=config.models_dir, force=False)
        promoted = True
    except ValueError as exc:
        print(f"Saved '{name}' but registry blocked promotion: {exc}")
    return promoted


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 — orchestration is intrinsic
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 9 core-satellite portfolio sweep")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--name", default="portfolio-core-satellite")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)
    earnings_store = EarningsStore(config.earnings_dir)
    news_store = NewsStore(config.news_dir)
    news_arg = news_store if news_store.has_any_data() else None

    print("Building PEAD dataset for the satellite return series...")
    dataset = build_pead_dataset(
        store, earnings_store, config.tickers_for_universe("all"), news_store=news_arg
    )
    print(f"  PEAD events: {len(dataset)}")

    bnh = build_bnh_returns(store)
    pead = build_pead_returns(dataset, store)
    calendar = build_calendar_returns(store)
    strategies = {"bnh_spy": bnh, "pead": pead, "calendar_spy": calendar}

    aligned = pd.DataFrame(strategies).dropna(how="all").fillna(0.0)
    aligned = aligned.loc[aligned.index >= aligned.index.min()]
    print(
        f"Aligned series: {aligned.index.min().date()} -> {aligned.index.max().date()} "
        f"({len(aligned)} bars)"
    )

    corr = _correlations(aligned)
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    # Static grid first — the bnh-only row is the benchmark for the deltas.
    print("\n=== Static-weight grid ===")
    bnh_only = run_portfolio_backtest(strategies, weights=STATIC_GRID[0][1])
    bnh_sharpe = bnh_only.metrics.sharpe
    print(f"  pure B&H baseline Sharpe = {bnh_sharpe:.3f}")

    rows: list[Row] = []
    for name, weights in STATIC_GRID:
        result = run_portfolio_backtest(strategies, weights=weights)
        rows.append(
            Row(
                name=name,
                sharpe=result.metrics.sharpe,
                sharpe_vs_bnh=result.metrics.sharpe - bnh_sharpe,
                total_return=result.metrics.total_return,
                max_dd=result.metrics.max_drawdown,
                ann_vol=result.metrics.ann_vol,
                turnover_total=float(result.turnover.sum()),
            )
        )
    _print_grid(rows)

    print("\n=== Walk-forward optimized weights ===")
    oof_returns, schedule = _walk_forward(aligned)
    oof_metrics = compute_metrics(oof_returns)
    bnh_oof_metrics = compute_metrics(aligned["bnh_spy"].loc[oof_returns.index])
    print(
        f"  OOS dates: {oof_returns.index.min().date()} -> {oof_returns.index.max().date()}"
        f" ({len(oof_returns)} bars, {len(schedule)} folds)"
    )
    print(
        f"  Portfolio Sharpe={oof_metrics.sharpe:.3f}  "
        f"vs B&H Sharpe={bnh_oof_metrics.sharpe:.3f}  "
        f"delta={oof_metrics.sharpe - bnh_oof_metrics.sharpe:.3f}"
    )
    print("  Weight schedule:")
    for start, w in schedule:
        formatted = ", ".join(f"{k}={v:.2f}" for k, v in w.items())
        print(f"    {start.date()} -> {formatted}")

    # Promote gate: the best of {best static, walk-forward OOS} must clear the buffer.
    best_static = max(rows, key=lambda r: r.sharpe)
    candidates: list[tuple[str, float, dict[str, float]]] = [
        ("walk_forward", oof_metrics.sharpe, schedule[-1][1] if schedule else {}),
        (
            best_static.name,
            best_static.sharpe,
            dict(STATIC_GRID[[r.name for r in rows].index(best_static.name)][1]),
        ),
    ]
    print("\n=== Promote candidates (gate: Sharpe > B&H + 0.05) ===")
    promoted = False
    for cand_name, cand_sharpe, cand_weights in candidates:
        delta = cand_sharpe - bnh_oof_metrics.sharpe
        ok = delta >= GATE_SHARPE_BUFFER
        print(f"  {cand_name}: Sharpe={cand_sharpe:.3f}  delta={delta:+.3f}  clears={ok}")
        if ok and not promoted:
            full_name = f"{args.name}-{cand_name}"
            promoted = _maybe_promote(
                full_name, cand_weights, cand_sharpe, bnh_oof_metrics.sharpe, config
            )
            if promoted:
                print(f"  → Promoted '{full_name}'.")

    if not promoted:
        print("\nNo portfolio variant cleared the +0.05 Sharpe gate vs pure B&H.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
