"""Lot A7 — one gated walk-forward on BTC/USDT 1h, end-to-end through the unchanged guard.

The intraday POC's first real measurement: fetch deep 1h BTC/USDT history from Binance,
land it in the intraday store, then run the per-asset tournament with ``interval="1h"`` for
both sides. The promotion guard (``promote()`` / ``_gate_failure``) is byte-identical to the
daily path — it judges returns, not time — so a 1h candidate is held to the same bar: long
must beat BTC buy-&-hold, short needs a positive, significant (deflated) Sharpe, both with
>= 20 OOS trades, annualized with bars_per_year=8760 and charged ~0.10%/side Binance fees.

Honest reporting: if the 1h edge dies after costs it stays advisory/observe — no capital.

Usage:
    uv run python scripts/train_intraday_btc.py [--ticker BTC-USD] [--no-fetch] [--hpo N]
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.config import Config
from berich.data.binance_adapter import update_intraday
from berich.data.store import OhlcvStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_intraday_btc")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gated intraday (1h) walk-forward POC")
    parser.add_argument(
        "--ticker", default="BTC-USD", help="configured crypto ticker (yfinance syntax)"
    )
    parser.add_argument(
        "--no-fetch", action="store_true", help="skip the Binance refresh; use the cache"
    )
    parser.add_argument(
        "--hpo", type=int, default=0, help="HPO trials per framework before the tournament"
    )
    args = parser.parse_args()

    config = Config.load("config/berich.yaml")
    interval = config.intraday.interval
    store = OhlcvStore(config.ohlcv_intraday_dir, interval=interval)

    if not args.no_fetch:
        added = update_intraday(config, store, args.ticker)
        logger.info("fetched %d new %s %s bars", added, args.ticker, interval)

    df = store.load(args.ticker)
    if df is None or df.empty:
        logger.error("no intraday cache for %s — run without --no-fetch first", args.ticker)
        return 1
    logger.info(
        "%s intraday history: %d bars, %s -> %s",
        args.ticker,
        len(df),
        df.index.min(),
        df.index.max(),
    )

    # Lazy import: the tournament pulls in the model zoo (torch); keep it out of the fetch path.
    from berich.training.hpo import run_ticker_hpo  # noqa: PLC0415
    from berich.training.tournament import train_ticker_tournament  # noqa: PLC0415

    for side in config.zoo.ticker_sides:
        if args.hpo > 0:
            for model_name in config.zoo.ticker_tournament_models:
                run_ticker_hpo(
                    config, args.ticker, model_name, side, n_trials=args.hpo, interval=interval
                )
        result = train_ticker_tournament(config, args.ticker, side, interval=interval)
        verdict = "PROMOTED" if result.promoted else "advisory-only"
        logger.info(
            "%s %s/%s: winner=%s -> %s", interval, args.ticker, side, result.winner, verdict
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
