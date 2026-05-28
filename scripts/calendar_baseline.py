"""Sanity baseline — pure turn-of-month calendar rule, no ML.

The Phase 6 importances showed the model leaning heavily on macro regime
+ calendar features (~73 % of total importance on the wide-universe
model). That raises a fair question: does a *no-ML* turn-of-month rule
already capture most of the calendar effect on the mega-cap watchlist?

The rule: hold each ticker long during the [J-3 last business days of
month, J+3 first business days of next month] window, flat otherwise.
Frais + slippage standards. Equal-weight portfolio across the 10
mega-caps. Compared to a 100 %-time-long buy & hold of the same
universe over the same dates.

If turn-of-month beats B&H by itself, the ML model isn't providing
incremental value above this trivial seasonality — an honest
red-flag we should record.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from berich.backtest.metrics import compute_metrics
from berich.config import DEFAULT_CONFIG_PATH, Config
from berich.data.store import OhlcvStore

WINDOW_DAYS = 3  # business days on either side of the month boundary
FEE_BPS = 1.0
SLIPPAGE_BPS = 5.0


def _turn_of_month_mask(index: pd.DatetimeIndex, window: int = WINDOW_DAYS) -> pd.Series:
    """Mark business days in the [last ``window`` of month, first ``window`` next month].

    Implemented per-month by ranking each business day inside the month from
    1..N: the last ``window`` ranks form the "end of month" window, the first
    ``window`` ranks form the "start of next month" window. Then the mask is
    the union of both — equivalent to the user's request without having to
    walk calendar dates explicitly.
    """
    df = pd.DataFrame({"date": index})
    df["year"] = index.year
    df["month"] = index.month
    df["rank_fwd"] = df.groupby(["year", "month"]).cumcount() + 1
    df["rank_bwd"] = df.groupby(["year", "month"]).cumcount(ascending=False) + 1
    in_first = df["rank_fwd"] <= window
    in_last = df["rank_bwd"] <= window
    return pd.Series((in_first | in_last).to_numpy(), index=index, name="in_window")


def _per_ticker_returns(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    fee: float,
    slippage: float,
) -> pd.Series:
    """Strategy daily returns: long when ``mask`` is True; pay fee+slippage on entry/exit edges."""
    close = df["close"]
    daily_ret = close.pct_change().fillna(0.0)
    # Position is the mask itself; transition days (0→1 or 1→0) pay the cost.
    position = mask.reindex(df.index).fillna(value=False).astype(int)
    held = daily_ret * position
    # Edge cost: on each transition we cross the spread once (fee + slippage).
    edge = (position != position.shift(1).fillna(0)).astype(int)
    cost = edge.astype(float) * (fee + slippage)
    return held - cost


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Turn-of-month calendar baseline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--window", type=int, default=WINDOW_DAYS)
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    store = OhlcvStore(config.ohlcv_dir)

    fee = FEE_BPS / 1e4
    slip = SLIPPAGE_BPS / 1e4

    strat_per_ticker: dict[str, pd.Series] = {}
    bench_per_ticker: dict[str, pd.Series] = {}
    for ticker in config.watchlist:
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        mask = _turn_of_month_mask(pd.DatetimeIndex(df.index), window=args.window)
        strat_per_ticker[ticker] = _per_ticker_returns(df, mask, fee=fee, slippage=slip)
        bench_per_ticker[ticker] = df["close"].pct_change().fillna(0.0)

    if not strat_per_ticker:
        print("No tickers cached; run `berich data` first.")
        return 1

    strat_daily = pd.DataFrame(strat_per_ticker).mean(axis=1).dropna()
    bench_daily = pd.DataFrame(bench_per_ticker).mean(axis=1).dropna()
    strat_metrics = compute_metrics(strat_daily)
    bench_metrics = compute_metrics(bench_daily)

    held_fraction = (
        np.mean(
            [
                _turn_of_month_mask(pd.DatetimeIndex(store.load(t).index)).mean()
                for t in strat_per_ticker
            ]  # type: ignore[union-attr]
        )
        if strat_per_ticker
        else 0.0
    )

    print(f"\nTurn-of-month strategy (J-{args.window} to J+{args.window}), 10 mega-caps:")
    print(f"  window time-in-market: {held_fraction * 100:.1f}% of trading days")
    print(f"  {'metric':<18}{'strategy':>12}{'buy & hold':>12}{'delta':>10}")
    for key in strat_metrics.as_dict():
        s = strat_metrics.as_dict()[key]
        b = bench_metrics.as_dict()[key]
        print(f"  {key:<18}{s:>12.4f}{b:>12.4f}{s - b:>10.4f}")
    beats = strat_metrics.sharpe > bench_metrics.sharpe
    verdict = "BEATS" if beats else "does NOT beat"
    print(f"\nTurn-of-month {verdict} buy & hold on Sharpe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
