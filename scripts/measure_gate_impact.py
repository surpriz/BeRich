"""Measure how the hardened promotion gate (Phase 1) would prune the CURRENT promoted set.

Read-only: re-evaluates each currently-promoted (ticker, side, strategy) under the new rules —
Deflated Sharpe fed the REAL trial count, the MIN_TRADES floor, realistic slippage — and reports
old (promoted) vs new (would-pass) verdicts, then applies Benjamini-Hochberg FDR across the
recomputed p-values. Nothing is saved or promoted/demoted; the registry is untouched.

Usage:
    uv run python scripts/measure_gate_impact.py --frameworks lightgbm
    uv run python scripts/measure_gate_impact.py --frameworks lightgbm,lstm --limit 20
    uv run python scripts/measure_gate_impact.py --all          # every framework (GPU, slow)
"""

from __future__ import annotations

import argparse
import json

from berich.backtest.multiple_testing import benjamini_hochberg
from berich.config import Config
from berich.data.store import OhlcvStore
from berich.models.registry import MAX_SHARPE_PVALUE, MIN_DEFLATED_SHARPE, MIN_TRADES
from berich.training.hpo import ticker_trial_count
from berich.training.tournament import _FRAMEWORK, train_candidate

_FRAMEWORK_TO_KEY = {v: k for k, v in _FRAMEWORK.items()}  # "lightgbm" -> "lgbm", ...


def _promoted(cfg: Config) -> list[dict]:
    root = cfg.models_dir / "tickers"
    out: list[dict] = []
    for active in root.rglob("active.json"):
        reg = active.parent
        try:
            name = json.loads(active.read_text())["name"]
            meta = json.loads((reg / name / "metadata.json").read_text())
        except Exception:  # noqa: BLE001, S112 — skip an unreadable artifact, keep scanning
            continue
        out.append(
            {
                "path": "/".join(reg.relative_to(root).parts),
                "ticker": meta.get("ticker"),
                "side": meta.get("side"),
                "strategy": meta.get("exit_strategy") or "fixed",
                "framework": meta.get("framework"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameworks", default="lightgbm", help="comma list, or use --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    store = OhlcvStore(cfg.ohlcv_dir)
    models = cfg.zoo.ticker_tournament_models
    keep_fw = None if args.all else set(args.frameworks.split(","))

    targets = [p for p in _promoted(cfg) if keep_fw is None or p["framework"] in keep_fw]
    if args.limit:
        targets = targets[: args.limit]
    print(f"Re-evaluating {len(targets)} promoted models under the hardened gate...\n")

    rows: list[dict] = []
    for p in targets:
        key = _FRAMEWORK_TO_KEY.get(p["framework"])
        if key is None:
            continue
        n_trials = max(
            sum(ticker_trial_count(cfg, p["ticker"], m, p["side"], p["strategy"]) for m in models),
            len(models),
        )
        try:
            _model, meta, _cal, cand = train_candidate(
                cfg,
                store,
                p["ticker"],
                p["side"],
                key,
                strategy=p["strategy"],
                calibrate=False,
                n_trials=n_trials,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {p['path']}: {exc}")
            continue
        m = meta.metrics
        rows.append(
            {
                "path": p["path"],
                "side": p["side"],
                "new_pass": cand.beats_guard,
                "n_trades": int(m.get("n_trades", 0)),
                "dsr": m.get("deflated_sharpe", 0.0),
                "pval": m.get("sharpe_pvalue", 1.0),
                "sharpe": m.get("sharpe", 0.0),
                "n_trials": n_trials,
            }
        )

    rows.sort(key=lambda r: (r["side"], r["path"]))
    print(f"{'model':<34}{'n_trades':>9}{'DSR':>7}{'p':>8}{'pass?':>7}")
    print("-" * 65)
    for r in rows:
        flag = "✓" if r["new_pass"] else "✗"
        print(f"{r['path']:<34}{r['n_trades']:>9}{r['dsr']:>7.2f}{r['pval']:>8.3f}{flag:>7}")

    passed = [r for r in rows if r["new_pass"]]
    thin = [r for r in rows if r["n_trades"] < MIN_TRADES]
    print(
        f"\nPer-model gate (real n_trials, MIN_TRADES={MIN_TRADES}, "
        f"DSR>={MIN_DEFLATED_SHARPE}, p<{MAX_SHARPE_PVALUE}):"
    )
    print(f"  {len(passed)}/{len(rows)} still pass | {len(thin)} fail the trade-count floor alone")

    # Sweep-level FDR across the recomputed p-values.
    res = benjamini_hochberg([r["pval"] for r in rows], alpha=0.10)
    fdr_keep = [rows[x.index]["path"] for x in res if x.rejected and rows[x.index]["new_pass"]]
    print(f"  After Benjamini-Hochberg FDR (alpha=0.10): {len(fdr_keep)}/{len(rows)} survive")
    for k in sorted(fdr_keep):
        print(f"     keep: {k}")


if __name__ == "__main__":
    main()
