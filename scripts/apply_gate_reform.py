"""Forward-test decision + gate-reform kickoff (pre-registered in docs/GATE_REFORM.md).

Run when the committed book reaches ~30 closed trades (the decision rule archived in
docs/RESULTS.md). The script:

1. checks the trigger (closed committed trades >= --min-trades, default 30);
2. computes the pre-registered decision table: net expectancy per (asset class x side)
   segment on the committed book — concentrate on positive segments, cut the rest;
3. prints the gate-reform checklist (docs/GATE_REFORM.md) with the file anchors to edit.

Read-only by design: the reform items are code changes reviewed by a human (or Claude),
not blind string substitutions — this script is the guard that says WHEN, and the
decision table that says ON WHAT. It refuses to pretend the trigger is met (--force to
override, mirroring the registry's own escape hatch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKLIST = """\
GATE REFORM — checklist pré-enregistrée (docs/GATE_REFORM.md, figée le 2026-06-10)
  1. Gate = politique servie : backtester proba CALIBRÉE >= decision_threshold
     -> src/berich/training/tournament.py (train_candidate: bt_config + oof)
  2. Split temporel des OOF : 2/3 calibration+seuil, 1/3 gate
     -> src/berich/training/tournament.py (fit_calibrator / optimal_decision_threshold)
  3. MIN_TRADES 20 -> 50   -> src/berich/models/registry.py
  4. AUC_FLOOR 0.5 -> 0.52 -> src/berich/training/tournament.py
  5. Fill du stop au pire de (barrière, open) sur gap
     -> src/berich/backtest/engine.py (_resolve_exit*), signals/paper.py (_resolve_paper_exit)
  6. Gate long : Sharpe OU Calmar > B&H + significativité commune
     -> src/berich/models/registry.py (_gate_failure), training/tournament.py
  7. Cap dur par devise (budget = max_class_exposure_pct)
     -> src/berich/signals/paper.py (_apply_exposure_caps, class_of devise)
  8. Embargo horizon+2 (trailing) ; min_count adaptatif du seuil ; cost_bps en métadonnées
Après édition : laisser le sweep re-gater les 900 combos + FDR, puis NOUVEAU forward test.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/berich.yaml")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument(
        "--force", action="store_true", help="show the decision table even before the trigger"
    )
    args = parser.parse_args()

    import pandas as pd

    from berich.config import Config
    from berich.signals.paper import CLOSED_STATUSES, PROMOTED_TIER, PaperStore

    config = Config.load(args.config)
    trades = PaperStore(config.db_path).all_trades(tier=PROMOTED_TIER)
    closed = (trades[trades["status"].isin(CLOSED_STATUSES)] if not trades.empty else trades).copy()
    n_closed = len(closed)

    print(f"Trades fermés (committed) : {n_closed} / déclencheur {args.min_trades}")
    if n_closed < args.min_trades and not args.force:
        print("Déclencheur NON atteint — le forward test continue, aucune réforme à appliquer.")
        print("(--force pour afficher quand même la table de décision provisoire)")
        return 1

    if closed.empty:
        print("Aucun trade fermé — rien à analyser.")
        return 1

    closed["asset_class"] = closed["ticker"].map(config.asset_class_for)
    closed["side"] = closed["signal"].map(
        lambda s: "short" if str(s).upper() == "SHORT" else "long"
    )
    table = (
        closed.groupby(["asset_class", "side"])["pnl_pct"]
        .agg(n="count", expectancy="mean", total="sum")
        .reset_index()
        .sort_values("expectancy", ascending=False)
    )
    print("\nTable de décision pré-enregistrée — net expectancy par (classe x side) :")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(table.to_string(index=False))
    keep = table[table["expectancy"] > 0]
    cut = table[table["expectancy"] <= 0]
    print(f"\nGARDER (expectancy > 0) : {len(keep)} segment(s)")
    print(f"COUPER (expectancy <= 0): {len(cut)} segment(s)")
    print()
    print(CHECKLIST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
