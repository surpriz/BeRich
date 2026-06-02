# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# BeRich — AI agent context

  BeRich is a personal swing-trading advisory tool. ML predicts a **trend probability**
  (triple-barrier label: did the upper barrier hit before the lower one within N bars),
  not a price. Daily bars; per-asset models across US/FR/forex/crypto/commodities, each
  carrying a **long and a short** side (Phase 12). The triple-barrier label and backtest
  engine are direction-aware.

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

  1. **The guard rule.** A model is only trusted / promoted if it clears its strategy's
     bar, walk-forward, out-of-sample, with realistic fees + slippage. The registry
     (`src/berich/models/registry.py`) enforces it in `_gate_failure` / `promote()`:
     **long** must beat that asset's buy & hold (`beats_buy_hold = True`); **short** has
     no buy-&-hold benchmark, so it needs a positive, significant Sharpe vs cash
     (deflated Sharpe ≥ 0.95, p < 0.05); market-neutral uses the same significance test.
     `promote()` refuses anything that fails (unless `force=True`). A side that fails
     stays advisory-only (no `active.json`).
  2. **No lookahead, ever.** Features are causal (see `src/berich/features/indicators.py`).
     Scalers fit on the train fold only (`src/berich/datasets/scaling.py`). Walk-forward
     splits keep an embargo gap equal to the label horizon.
  3. **Honest reporting.** When a model does not beat buy & hold, surface it — never hide
     it. The dashboard shows an "advisory only" banner; `berich train` refuses to promote.
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
    models/        base.py Model protocol + lightgbm_model.py, lstm.py, finbert_scorer.py, registry.py
    training/      walk_forward.py (OOS), deep.py, hpo.py (Optuna), tournament.py (per-asset),
                   status.py (training inventory), pead.py, cross_sectional.py, gpu_pool.py
    backtest/      engine.py (direction-aware triple-barrier), pead_engine.py, portfolio.py, significance.py, metrics.py
    signals/       service.py (dual long/short daily signals), calibration.py, paper.py, store.py (DuckDB)
    risk/          gating.py, sizing.py, sizing_strategy.py (risk-based position sizing)
    monitoring/    PSI + KS feature drift
    notifications/ email.py (signal digest + send_alert_email for job-failure alerts)
    scheduler/     APScheduler jobs.py + runner.py (EVENT_JOB_ERROR -> email alert)
    api/           FastAPI backend (signals, explain, training, ops, universes, health)
    ops.py         live machine-status collectors (GPU/jobs/HPO/logs) for /api/ops
    backup.py      rotating local tar.gz of data/ (studies, models, signals DB)
  frontend/        Next.js dashboard — multi-asset tabs, ticker drill-down (lightweight-charts),
                   /training (per-asset model inventory), /ops (live machine status),
                   EN/FR i18n, health footer

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
  cross-sectional ranker (Phase 10); `paper` drives the paper-trading book. The CLI uses
  lazy imports inside subcommands to keep startup fast (hence the `PLC0415` ruff ignore).

  ## Common commands

  - **Lint / format / types / tests** (run before claiming a task done):
    `uv run ruff format src/ tests/` · `uv run ruff check src/ tests/` ·
    `uv run ty check src/` · `uv run pytest -q` (264 tests).
  - **Single test**: `uv run pytest tests/test_pead.py -q` or
    `uv run pytest tests/test_signals.py::test_name -q`.
  - **Frontend** (`frontend/`): `npm run build` · `npm run dev` · `npm run lint`.
  - GPU-only deps (torch, lightning, mlflow, optuna, pytorch-forecasting) live in the
    `gpu` dependency group; the `news`/`pead`/LSTM training paths need them.
    Deployment (systemd units + Caddy HTTPS) is documented in `docs/DEPLOY.md`.

  ## Current state (read before changing things)

  **v0.4.0 — advisory infrastructure, no edge claim.** (Suite is now 264 tests, ruff +
  ty clean, frontend builds.) Across nine phases — OHLCV + cross-asset macro, earnings
  surprises, FinBERT news sentiment, mid/small-cap universes, short horizons,
  post-earnings drift (PEAD), a risk-management overlay, and a core-satellite portfolio
  reframe — **no combination beats buy & hold** walk-forward with realistic fees +
  slippage on the US daily long-only universe.
  The PEAD event book (AUC ≈ 0.535, trade-level Sharpe ≈ 0.849, max DD ≈ -6.7 %) is
  the closest to a usable signal but still does not clear the benchmark once blended.
  The guard rule (`promote()`) refuses every variant; the dashboard surfaces an
  "advisory only" banner. This is shipped on purpose.

  **Phase 12 — per-asset tournament + directional shorts (architecture, not an edge
  claim; shipped + deployed).** Training/HPO moved from per-class to **per-asset**: each
  configured asset (US/FR/forex/crypto/commodities) gets its own Optuna studies per
  `(ticker, model, side)` and a tournament (`training/tournament.py`) that keeps the
  walk-forward winner among LightGBM/LSTM/PatchTST/TFT. The label + backtest are
  direction-aware, so each asset can carry a **long and a short model**; the signal
  service (`signals/service.py`, `_decide`) emits LONG/SHORT/NEUTRAL with mirrored TP/SL
  and absolute-distance sizing capped at capital (no leverage). The per-asset guard is as
  in rule 1. Given ~400 labeled windows/asset, **expect most assets to stay
  advisory-only** — that is the honest outcome, not a bug.

  Current promoted set after the first 40-trial sweep: 4 forex long (lgbm), BNP.PA +
  TTE.PA long (tft/patchtst), USDCAD short (lstm) = 7. The forex "wins" beat a flat
  buy & hold (Sharpe ≈ 0), so the edge is weak — treat all 7 as paper-trade candidates,
  not validated. AAPL long *and* short re-run at 160 trials still fail the guard,
  confirming the daily long-only US universe is hard.

  **Observability & ops (deployed).** Scheduler jobs (`scheduler/runner.py`):
  `ticker_initial_sweep_job` (Sat 14:00, deep, GPU pool), `ticker_nightly_refresh_job`
  (23:30, light), **`ticker_hpo_queue_job`** (every 2h — sequential first-HPO queue, one
  un-searched `(ticker, side)` at a time so deep HPO never overlaps; resumable), and a
  daily **`backup_job`** (21:00). An `EVENT_JOB_ERROR` listener emails an alert
  (`send_alert_email`) when a job crashes (needs `NOTIFY_EMAIL`/`SMTP_*` in
  `/etc/berich/env`). Two dashboard tabs back this: **`/training`** (`/api/training` ←
  `training/status.py`: per-asset status, winner, guard metrics, HPO trial count) and
  **`/ops`** (`/api/ops` ← `ops.py`: live GPU/jobs/HPO-queue/logs, 5 s refresh).

  **Known wrinkle (by design, flagged):** when an asset has no promoted per-side model,
  `service.py` falls back to the global `lgbm-hpo` (whose `beats_buy_hold = False`), so
  the paper book can open positions on advisory-only assets. The per-asset *promotion*
  guard holds; the *serving* fallback does not. Decide explicitly before relying on the
  paper book as a validation signal. LightGBM now pins `random_state=42` so served P(win)
  is reproducible. See `docs/RESULTS.md` "Phase 12" for the full framing.

  ## Exploration history & guardrails (read `docs/RESULTS.md` first)

  [`docs/RESULTS.md`](docs/RESULTS.md) is the authoritative log of all nine phases —
  full numbers, methodology, correlation matrices, and the per-phase promote-gate
  verdicts. **Before proposing any new edge search, read it** so you don't re-run a
  lever that already failed. Two hard guardrails:

  - **Do not reopen feature hunts in OHLCV or cross-asset macro.** Phase 3 settled
    that space: macro features dominate LightGBM importance yet don't lift AUC. If a
    proposal uses only `data/ohlcv/` plus standard cross-asset series, it's been tried.
  - **Do not re-litigate the portfolio overlays.** Phase 9 showed every B&H + PEAD /
    calendar blend reduces Sharpe vs pure B&H; the walk-forward optimizer never picked
    PEAD. The reframe is closed.

  The recommended pivot for any future iteration is to **change the problem, not the
  algorithm** (crypto / intraday bars / market-neutral long-short) — see the final
  verdict in `docs/RESULTS.md`. Experiment scripts are kept under `scripts/`
  (`sweep_*.py`, `exploit_h3.py`, `calendar_baseline.py`, `portfolio_sweep.py`,
  `train_pead.py`, `train_lstm.py`, …) and are reusable for those directions.

  ## House rules for code changes

  - Run `uv run ruff format src/ tests/`, `uv run ruff check src/ tests/`,
    `uv run ty check src/`, and `uv run pytest -q` before claiming a task done.
  - No comments that just restate what the code does; reserve comments for non-obvious
    *why*.
  - Keep features causal. Add a test whenever a new feature is added.
  - Honor the guard rule end-to-end: never bypass `promote()` without an explicit reason.
  - The build backend is `setuptools` with `find(where=src)`; do not switch back to
    `uv_build` or hatchling — they silently dropped subpackages under uv 0.11.
  - `.gitignore` patterns for runtime caches (`data/`, `mlruns/`) are anchored with a
    leading `/` to avoid swallowing `src/berich/data/` etc.
  - Before any new edge search, read `docs/RESULTS.md` and respect the two guardrails
    in "Exploration history & guardrails" above (no OHLCV/macro re-hunts, no portfolio
    overlay re-litigation).
