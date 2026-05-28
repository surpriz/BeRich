# BeRich

Swing-trading advisory tool. Predicts the **probability that a swing long reaches its
target before its stop** (triple-barrier labeling) and validates every signal with a
rigorous walk-forward backtest before it is ever trusted.

> **Design principle:** we do *not* predict prices. We predict trend probability and
> only trust a model if it beats both a LightGBM baseline **and** buy & hold,
> out-of-sample, with realistic fees and slippage.

## Status — v0.1.0 frozen as advisory infrastructure

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Scaffolding + yfinance ingestion (adjusted, incremental Parquet cache) | ✅ done |
| 1 | Causal features + triple-barrier labels + walk-forward splits | ✅ done |
| 2 | LightGBM baseline + event-based backtest (ATR SL/TP, fees, slippage) vs buy & hold | ✅ done |
| 4 | Daily signal service + risk-based position sizing (DuckDB) | ✅ done |
| 6 | Drift monitoring (PSI + KS) + APScheduler automation | ✅ done |
| 5 | FastAPI API + Next.js dashboard | ✅ done |
| 3 | Feature engineering + LSTM/MLflow + label sweep + cross-asset experiment | ✅ **closed — no edge found, see [docs/RESULTS.md](docs/RESULTS.md)** |
| 4a | Paper-trading tracker (DuckDB-backed, daily roundtrip vs SPY) | ✅ done |
| 4b | Production deployment: systemd + Caddy HTTPS + 24/7 dashboard | ✅ done — see [docs/DEPLOY.md](docs/DEPLOY.md) |
| 5a | Earnings features (6 columns, yfinance) — code merged, no edge | ✅ done — AUC 0.5117 vs 0.5169 baseline, not promoted ([RESULTS](docs/RESULTS.md#phase-5a--earnings-features-free-data-no-gpu)) |

v0.1.0 ships the data → features → label → walk-forward backtest → signal →
sizing → drift → dashboard pipeline. The model produced does **not** beat
buy & hold; the registry refuses to promote it and the dashboard surfaces an
"advisory only" banner. No edge is claimed.

## Quickstart

```bash
uv sync --all-groups          # install deps
uv run berich data            # refresh OHLCV + earnings caches from yfinance
                              # (use --skip-earnings to refresh OHLCV only)
uv run berich backtest        # walk-forward backtest of the LightGBM baseline
uv run berich signals         # generate & persist today's signals + position sizing
uv run berich drift           # feature-drift check (PSI + KS) vs the training era
uv run berich train           # train baseline, backtest, save to registry, promote if it wins
uv run berich models          # list registry artifacts + which one is active
uv run berich paper update    # open new BUY signals + walk open paper trades
uv run berich paper status    # open positions (MTM) + recent closed trades
uv run berich paper equity    # paper P&L vs SPY benchmark summary
uv run berich serve           # FastAPI backend on http://127.0.0.1:8000
uv run berich schedule        # daily refresh+signals+paper + weekly drift
uv run pytest                 # run the test suite
```

### Dashboard

```bash
cd frontend
npm install
npm run dev                   # http://localhost:3000 (expects `berich serve` running)
```

Set `BERICH_API_KEY` in the backend env to require an `X-API-Key` header; the
frontend reads `NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_API_KEY`.

Configuration lives in [`config/berich.yaml`](config/berich.yaml): watchlist, fetch
settings, triple-barrier parameters, and signal thresholds / sizing.

## Layout

```
src/berich/
  config.py      # typed YAML config
  data/          # yfinance ingestion + Parquet OHLCV store
  features/      # causal technical indicators + feature matrix
  labeling/      # triple-barrier labeling
  datasets/      # walk-forward splits, scaling (fit-on-train), sequence windows
  models/        # common Model protocol + LightGBM baseline
  training/      # walk-forward out-of-sample prediction
  backtest/      # event-based engine + risk/perf metrics
  signals/       # daily signal generation + sizing + DuckDB persistence
  monitoring/    # PSI / KS feature-drift detection
  scheduler/     # APScheduler jobs (daily signals, weekly drift)
  api/           # FastAPI backend
  cli.py         # data / backtest / signals / drift / serve / schedule
frontend/        # Next.js dashboard (signals, backtest, drift)
```

## GPU handoff (kept for future work)

Deep models train on a GPU box; serving stays local. The seam is the **model
registry** (`src/berich/models/registry.py`) and the `Model` protocol:

```
GPU box   git clone + uv sync --all-groups   (torch, mlflow, optuna are in the gpu group)
          fit a model behind the Model protocol -> save_model(...) -> promote (guarded)
sync      scp -r data/models/<name>  ->  local data/models/
local     load_active() picks it up automatically; `berich signals` / `serve` use it,
          no code change. Falls back to the inline LightGBM baseline if nothing promoted.
```

`promote()` refuses any model whose metadata says it does not beat buy & hold (the
guard rule), so a worse model can never silently take over serving. Phase 3 used
this seam to train + evaluate an LSTM and a label sweep; the artifact was
correctly refused (see RESULTS.md).

## Phase 3 — honest result

The 10-ticker LightGBM baseline finishes Phase 3 at **OOS AUC ≈ 0.517, Sharpe
≈ 0.43** vs **buy & hold Sharpe ≈ 1.15** — the strategy does not beat buy &
hold. Deeper models (LSTM), wider feature sets (calendar, regime, longer
momentum, mean-reversion), cross-asset macro context (VIX, TLT, HYG/LQD,
sector ETFs), and alternative label geometries were all tested and none
clears the bar. The macro features dominate LightGBM's importance ranking
yet do not lift AUC — the signal exists at macro scale but does not convert
to a tradeable single-name probability at the 10-day horizon. Full numbers
and verdicts: [docs/RESULTS.md](docs/RESULTS.md).

## Paper trading (Phase 4a)

A simulator that follows the daily signals with fictive capital so the user can
build the discipline of running a swing book *without* claiming an edge. There
is no broker, no real money: every fill is taken from the cached OHLCV at the
signal's entry/stop/target and exits are decided by the same ATR-stop /
ATR-target / horizon rule as the backtest.

Daily workflow:

```bash
uv run berich data            # refresh the OHLCV cache
uv run berich signals         # generate today's signals
uv run berich paper update    # open any new BUY signals + walk open trades
uv run berich paper status    # what's open right now + last 10 closes
uv run berich paper equity    # paper return vs same-capital SPY buy & hold
```

Or just `uv run berich schedule` and the chain runs after the US close every
weekday. All three sub-steps are **idempotent** — re-running the same day is a
no-op once everything that can fire has fired.

Trades and metrics land in DuckDB (`data/berich.duckdb`, table `paper_trades`)
and are exposed by the API at `/paper/positions`, `/paper/equity`, and
`/paper/closed-trades`. The dashboard's "Paper trading" section plots paper
equity against an equal-capital SPY benchmark; if the paper line sits below
the dashed SPY line, that's the model losing to buy & hold, in plain sight.

This is **paper only**: not a recommendation, not a backtest, not a claim of
edge. Use it to learn how the signals behave day-to-day in the real cache
rather than under walk-forward retrospection.
