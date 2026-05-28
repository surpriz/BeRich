# BeRich — AI agent context

  BeRich is a personal swing-trading advisory tool. ML predicts a **trend probability**
  (triple-barrier label: did the upper barrier hit before the lower one within N bars),
  not a price. Long-only, daily bars, US equities for v1.

  ## Non-negotiable design rules

  1. **The guard rule.** A model is only trusted / promoted if it beats **both** the
     LightGBM baseline **and** equal-weight buy & hold, walk-forward, out-of-sample, with
     realistic fees + slippage. The model registry (`src/berich/models/registry.py`)
     enforces this: `promote()` refuses any artifact whose metadata says
     `beats_buy_hold = False` (unless `force=True`).
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
    data/        yfinance ingestion + Parquet OHLCV cache
    features/    causal indicators + canonical FEATURE_COLUMNS
    labeling/    triple-barrier labels with sample weights
    datasets/    walk-forward splits, sequence windowing, StandardScaler
    models/      base.py Model protocol + lightgbm_model.py + registry.py
    training/    oof_predict (walk-forward out-of-sample probabilities)
    backtest/    event-based engine + risk/perf metrics
    signals/     daily signal generation + risk-based sizing + DuckDB persistence
    monitoring/  PSI + KS feature drift
    scheduler/   APScheduler jobs (daily signals, weekly drift)
    api/         FastAPI backend
  frontend/      Next.js dashboard (lightweight-charts)

  `models/base.py` defines a 2-method `Model` protocol: `fit(x, y, sample_weight=None)`
  and `predict_proba(x) -> np.ndarray`. Every model — LightGBM today, LSTM and TFT next —
  must implement it so training, backtesting, signal service, and the registry stay
  model-agnostic.

  ## CLI surface

  `berich data | backtest | signals | drift | train | models | serve | schedule`. The
  `train` command runs the walk-forward OOS, fits a final model on all labeled history,
  saves it to `data/models/<name>/`, and tries to promote it through the guard.

  ## Current state (read before changing things)

  **v0.1.0 — advisory infrastructure frozen, no edge claim.** All phases (0–6) are
  done and tested (47 pytest passing, ruff + ty clean, frontend builds). The final
  LightGBM baseline scores OOS AUC ≈ 0.517, Sharpe ≈ 0.43 vs buy & hold ≈ 1.15 —
  **does not beat buy & hold**. The guard rule (`promote()`) refuses the model;
  the dashboard surfaces an "advisory only" banner. This is shipped on purpose.

  ## Phase 3 outcome

  Phase 3 explored four levers — feature engineering (3a), LSTM (3b), label
  geometry sweep, and cross-asset macro features (VIX, TLT, HYG/LQD, sector
  ETFs). None lifted OOS AUC above ~0.52 nor Sharpe above buy & hold. The
  cross-asset features dominated LightGBM's importance ranking (top 5 = pure
  macro) yet did not lift AUC — the macro signal exists but is not convertible
  to a tradeable single-name probability at the 10-day horizon. Full numbers,
  methodology and verdicts in [`docs/RESULTS.md`](docs/RESULTS.md).

  Models / training driver from Phase 3 are kept in the codebase:
  - `src/berich/models/lstm.py` — LSTM behind the `Model` protocol.
  - `src/berich/training/deep.py` — MLflow-tracked OOF + backtest + guarded promote.
  - `scripts/feature_importances.py`, `scripts/sweep_labels.py`, `scripts/train_lstm.py`.
  These are reusable for future experiments. The `SECTOR_MAP` dict in
  `features/build.py` is kept as a helper (not wired) for the same reason.

  **Future work (not started): Phase 4 — exogenous information.** News / sentiment
  via FinBERT, earnings surprises, sector flows. This is a separate project; the
  v0.1.0 freeze is meant to keep the existing infrastructure stable while that
  work is scoped properly.

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
  - **Do not reopen feature hunts in OHLCV or cross-asset macro without a new
    data-source rationale.** Phase 3 conclusively explored that space; the next
    edge-search lives in Phase 4 (exogenous info — news, earnings, flows). If a
    proposal can be implemented with the data already in `data/ohlcv/` plus
    standard cross-asset macro series, it has already been tried — read
    `docs/RESULTS.md` first.
