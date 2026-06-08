# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# BeRich — AI agent context

  BeRich is a personal swing-trading advisory tool. ML predicts a **trend probability**
  (triple-barrier label: did the upper barrier hit before the lower one within N bars),
  not a price. Daily bars; per-asset models across **125 assets** (US 40 / FR 26 /
  forex 24 / crypto 18 / commodities 17), each carrying a **long and a short** side
  (Phase 12) and three exit strategies (Phase 13) = 750 (ticker, side, strategy) combos.
  The triple-barrier label and backtest engine are direction-aware.

  **Exit strategies (Phase 13).** Beyond the fixed TP/SL triple barrier, the label + backtest
  also simulate **trailing** exits via `LabelConfig.exit_mode` / `BacktestConfig.exit_mode`
  (`fixed` | `trailing` = ratcheting stop, no TP | `trailing_tp` = TP cap + ratchet; params
  `trailing_atr`, `trailing_activation_atr`). The trailing walk is **causal** (the trail
  distance is frozen at entry; the stop is set from the favorable extreme of *prior* bars, so a
  bar never sets-and-triggers its own stop). Each strategy is trained + gated as a **parallel**
  per-asset model under `data/models/tickers/<SLUG>/<side>[/<strategy>]/` (`fixed` keeps the
  legacy path) through the **same** `promote()` guard — `registry.py` is exit-strategy-agnostic.
  Serving picks the best **promoted** strategy per (ticker, side), ties → `fixed`
  (`signals/service._select_strategy`); the paper book walks the right exit per trade
  (`paper._resolve_trailing_exit`, status `closed_trail`).

  ## Non-negotiable design rules

  1. **The guard rule (hardened, Phase 14).** A model is only trusted / promoted if it
     clears its strategy's bar, walk-forward, out-of-sample, with realistic fees +
     volume-proportional slippage. The registry (`src/berich/models/registry.py`)
     enforces it in `_gate_failure` / `promote()`: **every** side needs ≥ `MIN_TRADES`
     (20) OOS trades AND a positive, significant Sharpe — deflated Sharpe ≥ 0.95
     computed with the **real trial count** (HPO trials × frameworks compared, fed from
     `tournament.py`), p < 0.05. **Long** must additionally beat that asset's buy & hold
     (`beats_buy_hold = True`). On top of the per-model gate, a **Benjamini–Hochberg FDR
     pass** (`backtest/multiple_testing.py`, `training/promotion.reconcile_sweep_fdr`,
     alpha = 0.1) runs across the whole sweep and **demotes** promoted models whose
     p-values don't survive multiple-testing control. Trust is **three-tier**
     (`registry.model_tier`): `promoted` (committed paper capital) / `observe` (shadow
     book, no capital) / `advisory` (display only). `promote()` refuses anything that
     fails (unless `force=True`).
  2. **No lookahead, ever.** Features are causal (see `src/berich/features/indicators.py`).
     Scalers fit on the train fold only (`src/berich/datasets/scaling.py`). Walk-forward
     splits keep an embargo gap equal to the label horizon.
  3. **Honest reporting.** When a model does not clear the gate, surface it — never hide
     it. The dashboard shows an "advisory only" banner; `berich train` refuses to promote;
     demotions are logged and dated on `/changelog`.
  4. **Avoid fragile deps.** `pandas-ta` (broken on numpy 2 / pandas 3) and Evidently
     (heavy, unstable API) are deliberately NOT used. Indicators are hand-rolled in
     `features/indicators.py`; drift is hand-rolled in `monitoring/drift.py` using PSI + KS.
     Apply the same scrutiny before adding any new data/ML dep.

  ## Architecture (modules behind the `Model` protocol)

  src/berich/
    data/          yfinance OHLCV + earnings + fundamentals + Alpha Vantage news, Parquet/cache
    features/      causal indicators, FEATURE_COLUMNS, earnings/news/microstructure/fundamental features
    labeling/      direction-aware triple-barrier labels (long/short) with sample weights
    datasets/      walk-forward splits, sequence windowing, StandardScaler
    models/        base.py Model protocol + lightgbm_model.py, lstm.py, finbert_scorer.py,
                   registry.py (gate, tiers, demote)
    training/      walk_forward.py (OOS), deep.py, hpo.py (Optuna), tournament.py (per-asset),
                   promotion.py (sweep-level FDR), status.py (training inventory), pead.py,
                   cross_sectional.py, gpu_pool.py
    backtest/      engine.py (direction-aware triple-barrier), multiple_testing.py (BH FDR),
                   pead_engine.py, portfolio.py, significance.py (deflated Sharpe), metrics.py
    signals/       service.py (dual long/short daily signals, per-asset decision_threshold),
                   calibration.py, paper.py (tiered books + money-management), store.py (DuckDB)
    risk/          gating.py, sizing.py, sizing_strategy.py (risk-based position sizing)
    monitoring/    PSI + KS feature drift (logged only — see data-health note below)
    notifications/ digest.py (build_daily_digest — portfolio + executions + good-to-know) +
                   email.py (bilingual Clarté daily digest + send_alert_email for job alerts)
    scheduler/     APScheduler jobs.py + runner.py (EVENT_JOB_ERROR -> email alert)
    api/           FastAPI backend (signals, explain, training, ops, universes, health,
                   risk-profile, brief-plan, replication)
    ops.py         live machine-status collectors (GPU/jobs/HPO/logs) for /api/ops
    backup.py      rotating local tar.gz of data/ (studies, models, signals DB)
  scripts/run_full_sweep.py   the continuous sweep loop (see Ops below)
  frontend/        Next.js dashboard — multi-asset tabs, ticker drill-down (lightweight-charts),
                   /brief (daily plan: positions + FORECAST orders, clearly flagged),
                   /wallet (paper book P&L), /copy (Réplication: executions-only morning
                   copy-trading list with broker-capital scaling), /training, /ops,
                   /changelog (bilingual dated journal of every system change),
                   /strategies, EN/FR i18n, three detail levels (simple/standard/expert,
                   new visitors default to simple = Brief-centred home), health footer

  `models/base.py` defines a 2-method `Model` protocol: `fit(x, y, sample_weight=None)`
  and `predict_proba(x) -> np.ndarray`. Every model implements it so training,
  backtesting, signal service, and the registry stay model-agnostic.

  Frontend i18n is home-grown (`frontend/app/lib/i18n/`, EN/FR JSON catalogs +
  React context with localStorage persistence) — `next-intl` was deliberately
  not adopted. The `/explain` endpoint returns SHAP-style feature contributions
  via LightGBM `predict_contrib`.

  ## CLI surface

  `berich data | news | backtest | signals | longshort | drift | pead | paper | train | hpo | models | backup | serve | schedule`
  (defined in `src/berich/cli.py`). The `train` command runs the walk-forward OOS,
  fits a final model on all labeled history, saves it to `data/models/<name>/`, and
  tries to promote it through the guard. As of Phase 12 `train` also drives the
  **per-asset tournament**: `berich train --ticker AAPL --side long|short --tournament`
  trains LightGBM/LSTM/PatchTST/TFT for one asset+side and keeps the walk-forward winner
  under `data/models/tickers/<TICKER>/<side>/`; `--all-tickers` sweeps every configured
  asset. As of Phase 13 `train --tournament --strategy fixed|trailing|trailing_tp|all`
  trains/compares exit strategies (each gated independently), and `berich backtest --exit-mode
  ...` backtests one; `scripts/compare_exit_strategies.py` tabulates the head-to-head verdicts.
  `berich hpo --ticker ... --model ... --side ... --trials N` runs a per-asset
  Optuna study (params + feature-group toggles, incl. news/FinBERT;
  `zoo.ticker_initial_hpo_trials = 100`). `berich backup [--keep N]` archives the
  training state. `news` runs the Alpha Vantage fetch + FinBERT GPU scoring; `pead`
  runs the event-driven Post-Earnings Drift model; `longshort` is the market-neutral
  cross-sectional ranker (Phase 10) — both are **retired** (see guardrails); `paper`
  drives the paper-trading book. The CLI uses lazy imports inside subcommands to keep
  startup fast (hence the `PLC0415` ruff ignore).

  ## Common commands

  - **Lint / format / types / tests** (run before claiming a task done):
    `uv run ruff format src/ tests/` · `uv run ruff check src/ tests/` ·
    `uv run ty check src/` · `uv run pytest -q` (337 tests).
  - **Single test**: `uv run pytest tests/test_pead.py -q` or
    `uv run pytest tests/test_signals.py::test_name -q`.
  - **Frontend** (`frontend/`): `npm run build` · `npm run dev` · `npm run lint`.
  - GPU-only deps (torch, lightning, mlflow, optuna, pytorch-forecasting) live in the
    `gpu` dependency group; the `news`/`pead`/LSTM training paths need them.
    Deployment (systemd units + Caddy HTTPS) is documented in `docs/DEPLOY.md`.
    Deploy flow: commit → push origin main → `systemctl restart berich-{api,scheduler,frontend,sweep}`
    (the frontend unit rebuilds via ExecStartPre).

  ## Current state (read before changing things)

  **Phase 14 — hardened validation + minimal money management (June 2026, deployed).**
  The over-optimistic gate (DSR computed with `n_trials=1`, no trade floor, longs
  exempt from significance) inflated the promoted count to ~119. Fixing it (real
  trial counts, `MIN_TRADES=20`, significance required on longs too, sweep-wide BH FDR,
  volume-proportional slippage on by default) cut the survivors to **46 promoted models
  on 31 tickers** — forex-heavy (JPY crosses carry 3 books each), crypto + commodity
  shorts, a handful of US/FR longs. Money management is deliberately **minimal**: fixed
  1 % risk per trade, plus a graduated drawdown kill-switch (derisk ×0.5 beyond −10 %,
  halt beyond −20 %), a per-asset-class exposure cap (40 %), and `max_open_positions`
  (15) — all wired into `paper.open_new_trades`. Kelly / vol-targeting exist in
  `risk/sizing.py` but stay **off**. Risk profiles (`prudent|balanced|offensive`,
  `RISK_PROFILES` in `config.py`, persisted in `data/risk_profile.json`, switchable from
  the UI) scale those knobs. Per-asset `decision_threshold` (tuned on calibrated OOF)
  overrides the global threshold when present. `ensemble_serving` and
  `regime_conditioning` flags exist but default **False** (validate before enabling).

  **Forward test in progress — the system is FROZEN.** The 46 survivors are paper-trade
  candidates, *not* validated edge. The committed paper book (tier `promoted` only,
  10 k€ base) runs autonomously; the decision rule, archived in `docs/RESULTS.md`, is:
  at ~30 closed committed trades, concentrate on (class × side) segments with positive
  net expectancy and cut the rest. Until then: **no system modifications except bug
  fixes**, no reacting to individual trades, no real money. Signals are served from each
  asset's own optimized candidate — there is **no generic global-model fallback**
  anymore, and the committed book only opens `tier == promoted` trades, so the old
  "paper book trades advisory assets" wrinkle is gone (observe-tier trades live in a
  shadow book with no capital).

  **Bake-off verdict (June 2026): PEAD and market-neutral are buried.** A pre-registered
  head-to-head showed PEAD trailing window B&H on both horizons and the cross-sectional
  ranker at rank-IC ≈ 0. Both are formally retired in `docs/RESULTS.md`.

  **The daily machine.** `daily_paper_job` executes weekdays at 22:30 Paris: refresh
  data → signals → close/trail open positions → open new promoted-tier trades under the
  money-management caps. Everything on `/brief` under "Nouveaux ordres" is a continuously
  recomputed **forecast** (flagged as such in the UI); only the 22:30 photo executes.
  `/copy` (Réplication) lists only what was actually executed in the last run — that is
  the page a human copy-trader follows, never `/brief`'s forecast.

  **Observability & ops (deployed).** Four systemd units: `berich-api`,
  `berich-scheduler`, `berich-frontend`, **`berich-sweep`**
  (`scripts/run_full_sweep.py --continuous`): a perpetual loop over all 750 combos —
  fresh OHLCV each cycle, **un-searched combos first** (so newly added assets get their
  deep 100-trial HPO before incumbents are re-deepened with 4-trial top-ups), config
  reloaded every cycle, FDR reconciliation at startup + every 30 combos + cycle end
  (the sweep holds the HPO lock perpetually, so scheduler-side FDR would never fire).
  Scheduler keeps `daily_paper_job` (22:30 weekdays), nightly refresh, a daily
  `backup_job` (21:00) and data-health checks; an `EVENT_JOB_ERROR` listener emails
  alerts (`NOTIFY_EMAIL`/`SMTP_*` in `/etc/berich/env`). **Drift emails were removed
  on purpose**: PSI/KS on daily features structurally cries wolf (calendar features,
  autocorrelated slow features) — three recalibrations all flagged ~everything. The
  monitor job now alerts only on *data health* (bars stale > 7 days, frozen prices);
  drift shares are logged, not emailed. Dashboard tabs: `/training` (per-asset status,
  winner, guard metrics, HPO trials), `/ops` (live GPU/jobs/sweep/logs, 5 s refresh).
  LightGBM pins `random_state=42` so served P(win) is reproducible.

  ## Exploration history & guardrails (read `docs/RESULTS.md` first)

  [`docs/RESULTS.md`](docs/RESULTS.md) is the authoritative log of all phases —
  full numbers, methodology, correlation matrices, and the per-phase promote-gate
  verdicts. **Before proposing any new edge search, read it** so you don't re-run a
  lever that already failed. Three hard guardrails:

  - **Do not reopen feature hunts in OHLCV or cross-asset macro.** Phase 3 settled
    that space: macro features dominate LightGBM importance yet don't lift AUC. If a
    proposal uses only `data/ohlcv/` plus standard cross-asset series, it's been tried.
  - **Do not re-litigate the portfolio overlays.** Phase 9 showed every B&H + PEAD /
    calendar blend reduces Sharpe vs pure B&H; the walk-forward optimizer never picked
    PEAD. The reframe is closed.
  - **Do not re-litigate PEAD or the daily cross-sectional market-neutral ranker.**
    The June 2026 pre-registered bake-off buried both (see `docs/RESULTS.md`,
    "Bake-off June 2026").

  Experiment scripts are kept under `scripts/` (`sweep_*.py`, `exploit_h3.py`,
  `calendar_baseline.py`, `portfolio_sweep.py`, `train_pead.py`, `train_lstm.py`,
  `measure_gate_impact.py`, …) and are reusable. The open question is no longer "which
  algorithm" but whether the 46 forward-test survivors hold up out-of-sample in the
  live paper book.

  ## House rules for code changes

  - Run `uv run ruff format src/ tests/`, `uv run ruff check src/ tests/`,
    `uv run ty check src/`, and `uv run pytest -q` before claiming a task done.
  - **Respect the forward-test freeze**: while the ~30-trade forward test runs, ship
    bug fixes and UI/observability work freely, but do not change labels, gates,
    sizing, thresholds, or the promoted set without an explicit user decision.
  - No comments that just restate what the code does; reserve comments for non-obvious
    *why*.
  - Keep features causal. Add a test whenever a new feature is added.
  - Honor the guard rule end-to-end: never bypass `promote()` without an explicit reason.
  - Every user-visible change gets a dated bilingual entry on `/changelog`
    (`frontend/app/changelog/page.tsx`, newest first).
  - New universe assets must be validated on yfinance first (>1000 daily bars, fresh
    last date); the continuous sweep picks them up automatically and trains them first.
  - The build backend is `setuptools` with `find(where=src)`; do not switch back to
    `uv_build` or hatchling — they silently dropped subpackages under uv 0.11.
  - `.gitignore` patterns for runtime caches (`data/`, `mlruns/`) are anchored with a
    leading `/` to avoid swallowing `src/berich/data/` etc.
  - Before any new edge search, read `docs/RESULTS.md` and respect the three guardrails
    in "Exploration history & guardrails" above.
