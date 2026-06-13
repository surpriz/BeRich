"""Head-to-head exit-strategy comparison: fixed vs trailing vs trailing_tp.

Reads the AUTHORITATIVE served set — each strategy's ``active.json`` pointer plus the served
model's ``metadata.json`` — under ``data/models/tickers/<SLUG>/<side>[/<strategy>]/``, and
tabulates, per (ticker, side), which strategies cleared the promotion guard, their guard metric
(Sharpe for long, deflated Sharpe for short) and OOS AUC, and which one *serving* actually picks
(``signals/service._select_strategy``: best promoted metric, ties → fixed).

Why active.json and not status.json: the continuous sweep promotes via ``active.json`` but does
not keep the tournament's ``status.json`` in sync, so the old status.json-based table badly
under-reported (it missed every trailing promotion). This reads what is genuinely served.

It does NOT train or load any model weights — only the small JSON artifacts. Strategies that did
not clear the guard for a (ticker, side) show as "—".

Usage:
    python scripts/compare_exit_strategies.py [--config config/berich.yaml] [--side long|short]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from berich.config import Config

STRATEGIES = ("fixed", "trailing", "trailing_tp")


def _served(
    config: Config, ticker: str, side: str, strategy: str
) -> tuple[bool, float | None, float | None]:
    """(promoted, guard_metric, auc) for one (ticker, side, strategy), from active.json + metadata.

    promoted is True iff an ``active.json`` exists (the gate passed and this is served). The
    guard metric is Sharpe for a long, deflated Sharpe for a short — exactly what serving ranks
    on. No model weights are loaded; only the JSON artifacts are read.
    """
    reg = config.model_dir_for_ticker(ticker, side, strategy)
    active = reg / "active.json"
    if not active.exists():
        return (False, None, None)
    try:
        name = json.loads(active.read_text(encoding="utf-8"))["name"]
        metrics = json.loads((reg / name / "metadata.json").read_text(encoding="utf-8")).get(
            "metrics", {}
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return (True, None, None)
    metric = metrics.get("deflated_sharpe" if side == "short" else "sharpe")
    return (True, metric, metrics.get("auc"))


def _served_winner(cells: dict[str, tuple[bool, float | None, float | None]]) -> str | None:
    """Replicate serving's choice: best promoted guard metric, ties → fixed."""
    promoted = [(s, m if m is not None else -9.0) for s, (ok, m, _a) in cells.items() if ok]
    if not promoted:
        return None
    return max(promoted, key=lambda sm: (sm[1], sm[0] == "fixed"))[0]


def _cell(c: tuple[bool, float | None, float | None]) -> str:
    ok, metric, _auc = c
    if not ok:
        return f"{'—':>18}"
    m = f"{metric:.3f}" if metric is not None else " n/a"
    return f"{'PROMU':>10} {m:>7}"


class _Stats:
    """Running aggregates as the table is printed (keeps ``main`` flat)."""

    def __init__(self) -> None:
        self.served: collections.Counter[str] = collections.Counter()
        self.promoted: collections.Counter[str] = collections.Counter()
        self.auc: dict[str, list[float]] = collections.defaultdict(list)
        self.competition = 0  # (ticker, side) with >= 2 strategies promoted
        self.trailing_wins = 0  # of those, a trailing variant is the served winner

    def add(self, cells: dict[str, tuple[bool, float | None, float | None]], winner: str) -> None:
        promoted_here = [s for s in STRATEGIES if cells[s][0]]
        for s in promoted_here:
            self.promoted[s] += 1
            if cells[s][2] is not None:
                self.auc[s].append(cells[s][2])  # type: ignore[arg-type]
        self.served[winner] += 1
        if len(promoted_here) >= 2:  # noqa: PLR2004 — "two or more strategies compete"
            self.competition += 1
            if winner in ("trailing", "trailing_tp"):
                self.trailing_wins += 1


def _print_summary(stats: _Stats, width: int) -> None:
    print("-" * width)
    print("\nRÉSUMÉ")
    print(f"{'stratégie':<14}{'promue':>8}{'servie (gagne)':>16}{'AUC moy':>10}")
    for s in STRATEGIES:
        aucs = stats.auc[s]
        auc = f"{sum(aucs) / len(aucs):.3f}" if aucs else "  n/a"
        print(f"{s:<14}{stats.promoted[s]:>8}{stats.served[s]:>16}{auc:>10}")
    print(
        f"\nVraie compétition (>=2 stratégies promues sur le même actif/sens) : {stats.competition}"
    )
    if stats.competition:
        pct = 100 * stats.trailing_wins / stats.competition
        print(f"  dont un trailing l'emporte : {stats.trailing_wins} ({pct:.0f}%)")
    total = sum(stats.served.values())
    if total:
        tr = stats.served["trailing"] + stats.served["trailing_tp"]
        print(f"\nSur {total} (actif, sens) servis : {tr} via trailing ({100 * tr / total:.0f}%).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/berich.yaml")
    ap.add_argument("--side", choices=["long", "short"], default=None)
    args = ap.parse_args()

    config = Config.load(args.config)
    sides = [args.side] if args.side else list(config.zoo.ticker_sides)

    header = f"{'ticker':<12}{'side':<7}" + "".join(f"{s:>18}" for s in STRATEGIES) + "   servi"
    print(header)
    print("-" * len(header))

    stats = _Stats()
    for ticker in config.tradeable_tickers():
        for side in sides:
            cells = {s: _served(config, ticker, side, s) for s in STRATEGIES}
            winner = _served_winner(cells)
            if winner is None:  # nothing promoted for this (ticker, side)
                continue
            print(
                f"{ticker:<12}{side:<7}"
                + "".join(_cell(cells[s]) for s in STRATEGIES)
                + f"   {winner}"
            )
            stats.add(cells, winner)

    _print_summary(stats, len(header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
