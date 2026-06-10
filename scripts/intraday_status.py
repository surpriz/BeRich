"""Read-only status report for the intraday (1h) POC — run anytime to "faire un point".

Consolidates: HPO depth per 1h study, the guard verdict (tier + metrics) per (side, strategy),
the intraday paper book, the currently-served signal, and where the unified sweep is. Mutates
nothing. Usage: ``.venv/bin/python scripts/intraday_status.py``.
"""

from __future__ import annotations

import sqlite3

from berich.config import Config
from berich.data.store import OhlcvStore
from berich.models.registry import load_active, load_best, model_tier
from berich.signals.paper_intraday import IntradayPaperStore, get_intraday_paper_metrics


def main() -> int:
    c = Config.load("config/berich.yaml")
    print(f"==== POC INTRADAY — point du jour ({c.intraday.interval}, {c.intraday.tickers}) ====\n")

    # 1) HPO depth per 1h study.
    print("— Profondeur HPO (études Optuna 1h) —")
    if c.optuna_db.exists():
        con = sqlite3.connect(f"file:{c.optuna_db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT study_name, count(*) FROM studies s JOIN trials t ON t.study_id = s.study_id "
            "WHERE study_name LIKE '%-1h-%' GROUP BY study_name ORDER BY study_name"
        ).fetchall()
        con.close()
        deep = c.zoo.ticker_initial_hpo_trials
        done = sum(1 for _, n in rows if n >= deep)
        print(f"  {len(rows)} études, {done} à >= {deep} trials (profondes)")
    else:
        print("  (pas d'optuna.db)")

    # 2) Guard verdict per (side, strategy).
    print("\n— Verdict du garde-fou (tier + métriques) —")
    promoted_any = False
    for ticker in c.intraday.tickers:
        for side in c.zoo.ticker_sides:
            for strat in c.zoo.ticker_exit_strategies:
                d = c.model_dir_for_ticker(ticker, side, strat, interval=c.intraday.interval)
                lb = load_best(d)
                if lb is None:
                    print(f"  {ticker} {side}/{strat}: (pas de modèle)")
                    continue
                _, meta = lb
                promoted = load_active(d) is not None
                promoted_any = promoted_any or promoted
                tier = model_tier(meta, promoted=promoted)
                m = meta.metrics
                print(
                    f"  {ticker} {side}/{strat}: TIER={tier} | AUC={m.get('auc', 0):.3f} "
                    f"sharpe={m.get('sharpe', 0):.2f} dsr={m.get('deflated_sharpe', 0):.2f} "
                    f"p={m.get('sharpe_pvalue', 1):.3f} n={int(m.get('n_trades', 0))} "
                    f"beatsBH={meta.beats_buy_hold}"
                )

    # 3) Paper book.
    print("\n— Livre papier intraday —")
    store = OhlcvStore(c.ohlcv_intraday_dir, interval=c.intraday.interval)
    ps = IntradayPaperStore(c.intraday_db_path)
    allt = ps.all_trades()
    n_open = int((allt["status"] == "open").sum()) if len(allt) else 0
    print(f"  trades total: {len(allt)} | ouverts: {n_open}")
    for tier in ("promoted", "observe"):
        m = get_intraday_paper_metrics(c, store, tier=tier)
        print(
            f"  [{tier}] n_open={m['n_open']} n_closed={m['n_closed']} "
            f"ret={m['total_return_paper'] * 100:.2f}% vs B&H {m['total_return_bench'] * 100:.2f}%"
        )

    print(
        "\n— Synthèse —\n  "
        + (
            "AU MOINS UN modèle 1h est PROMU — le livre engagé peut trader."
            if promoted_any
            else "Aucun modèle 1h promu : signal advisory, zéro capital engagé (verdict honnête)."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
