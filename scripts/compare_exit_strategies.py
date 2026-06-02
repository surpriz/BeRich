"""Head-to-head exit-strategy comparison: fixed vs trailing vs trailing_tp.

Reads the per-(ticker, side, strategy) ``status.json`` files the tournament writes under
``data/models/tickers/<SLUG>/<side>[/<strategy>]/`` and tabulates, per (ticker, side), each
strategy's guard verdict and best-candidate metric. This is the honest decision artifact: a
trailing variant is only ever *served* (see ``signals/service._select_strategy``) when it both
exists AND clears the same promotion guard as the fixed baseline — this table makes the verdict
visible side by side.

It does NOT train anything (run ``berich train --tournament --strategy all`` first); it only
summarizes what training already gated. Strategies with no status.json are shown as "—".

Usage:
    python scripts/compare_exit_strategies.py [--config config/berich.yaml] [--side long|short]
"""

from __future__ import annotations

import argparse
import json
import sys

from berich.config import Config

STRATEGIES = ("fixed", "trailing", "trailing_tp")


def _load_status(config: Config, ticker: str, side: str, strategy: str) -> dict | None:
    path = config.model_dir_for_ticker(ticker, side, strategy) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _best_metric(status: dict) -> float | None:
    """Best candidate's ranking metric (CandidateResult.strategy_sharpe holds the side metric)."""
    cands = status.get("candidates") or []
    vals = [c.get("strategy_sharpe") for c in cands if c.get("strategy_sharpe") is not None]
    return max(vals) if vals else None


def _cell(status: dict | None) -> str:
    if status is None:
        return f"{'—':>16}"
    metric = _best_metric(status)
    verdict = "PROMOTED" if status.get("promoted") else "advisory"
    metric_s = f"{metric:.3f}" if metric is not None else "  n/a"
    return f"{verdict:>9} {metric_s:>6}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/berich.yaml")
    ap.add_argument("--side", choices=["long", "short"], default=None)
    args = ap.parse_args()

    config = Config.load(args.config)
    sides = [args.side] if args.side else list(config.zoo.ticker_sides)

    header = f"{'ticker':<12}{'side':<7}" + "".join(f"{s:>16}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))

    any_trailing_promoted = False
    for ticker in config.tradeable_tickers():
        for side in sides:
            statuses = {s: _load_status(config, ticker, side, s) for s in STRATEGIES}
            if all(v is None for v in statuses.values()):
                continue
            row = f"{ticker:<12}{side:<7}" + "".join(_cell(statuses[s]) for s in STRATEGIES)
            print(row)
            for s in ("trailing", "trailing_tp"):
                if statuses[s] is not None and statuses[s].get("promoted"):
                    any_trailing_promoted = True

    print("-" * len(header))
    if any_trailing_promoted:
        print("At least one trailing variant cleared the guard — it will be served where it wins.")
    else:
        print(
            "No trailing variant cleared the guard: the fixed baseline still serves everywhere "
            "(the honest outcome — most assets are expected to stay advisory-only)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
