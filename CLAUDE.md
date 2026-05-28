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

  Phases 0, 1, 2, 4, 5, 6 + model registry are done and tested (42 pytest passing, ruff +
  ty clean, frontend builds). LightGBM baseline OOS AUC ≈ 0.52, Sharpe ≈ 0.54 vs buy &
  hold ≈ 1.15 — **does NOT beat buy & hold yet**. This is the expected starting point;
  Phase 3 (deep models, better features) aims to clear the guard.

  ## Phase 3 — what to do next, in order

  The work below is staged from cheapest to heaviest. Don't skip 3a: a stronger feature
  set often gives more lift than swapping LightGBM for an LSTM.

  ### 3a. Improve the feature set
  Add to `features/build.py` while keeping every feature causal:
  - calendar: day-of-week, month, days-to-month-end (one-hot or sinusoidal encoding);
  - market regime: SPY trailing return / SPY rolling vol, broadcast to every non-SPY
    ticker with a one-day lag (no leakage from same-day SPY into AAPL);
  - longer momentum: `mom_60`, `mom_120`;
  - distance to N-day rolling high / low (mean-reversion proxy).
  Update `FEATURE_COLUMNS` and add a no-lookahead test for any cross-asset feature. Re-run
  `berich backtest`. Stop tuning here only once a thoughtful baseline has been tried.

  ### 3b. LSTM baseline (GPU)
  Add `models/lstm.py` exposing the `Model` protocol. Use `make_sequences` from
  `datasets/windows.py` for the `(n, lookback, n_features)` input. Train on CUDA, expose
  hyperparameters (lookback, hidden, dropout, lr, epochs, batch). Wire `training/deep.py`
  with an MLflow run that logs params, OOS AUC from `oof_predict`, and the backtest
  verdict. Persist via `save_model` + try `promote` (guard).

  ### 3c. Optuna HPO (GPU)
  One Optuna study per model class. Two trials run in parallel on the two GPUs (set
  `CUDA_VISIBLE_DEVICES` per worker). Search over lookback / hidden / lr / dropout.
  Promote only if the best trial beats LightGBM AND buy & hold on OOS Sharpe.

  ### 3d. TFT (GPU)
  `pytorch-forecasting`'s `TemporalFusionTransformer`. Heavier — uses its own
  `TimeSeriesDataSet`. Keep the `Model` protocol wrapper so the rest of the stack
  (`signals/`, `backtest/`, registry) does not change.

  ### 3e. Handoff back to local
  On the GPU box, after a successful `promote`, the artifact lives in `data/models/<name>/`.
  Sync to the local Mac with `scp -r jerome@<gpu-ip>:~/BeRich/data/models/<name>
  ~/Desktop/BeRich/data/models/`. `signals/service.py` calls `load_active()` and serves it
  with zero code change.

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
