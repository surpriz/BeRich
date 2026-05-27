# BeRich

Swing-trading advisory tool. Predicts the **probability that a swing long reaches its
target before its stop** (triple-barrier labeling) and validates every signal with a
rigorous walk-forward backtest before it is ever trusted.

> **Design principle:** we do *not* predict prices. We predict trend probability and
> only trust a model if it beats both a LightGBM baseline **and** buy & hold,
> out-of-sample, with realistic fees and slippage.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Scaffolding + yfinance ingestion (adjusted, incremental Parquet cache) | ✅ done |
| 1 | Causal features (RSI, MACD, ATR, momentum, vol…) + triple-barrier labels + walk-forward splits | ✅ done |
| 2 | LightGBM baseline + event-based backtest (ATR SL/TP, fees, slippage) vs buy & hold | ✅ done |
| 4 | Daily signal service + risk-based position sizing (DuckDB) | ✅ done |
| 6 | Drift monitoring (PSI + KS) + APScheduler automation | ✅ done |
| 5 | FastAPI API + Next.js dashboard | ✅ done |
| 3 | LSTM / TFT on GPU + Optuna HPO + MLflow tracking | ⬜ todo (GPU machine) |

## Quickstart

```bash
uv sync --all-groups          # install deps
uv run berich data            # refresh the OHLCV cache from yfinance
uv run berich backtest        # walk-forward backtest of the LightGBM baseline
uv run berich signals         # generate & persist today's signals + position sizing
uv run berich drift           # feature-drift check (PSI + KS) vs the training era
uv run berich serve           # FastAPI backend on http://127.0.0.1:8000
uv run berich schedule        # local scheduler: daily signals + weekly drift
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

## Current baseline result

On the default 10-ticker US-equity watchlist the LightGBM baseline scores an
out-of-sample AUC near 0.51 and a Sharpe below buy & hold — i.e. **not yet a usable
signal**. This is the expected, honest starting point; the backtest harness exists
precisely to surface this rather than hide it. Phases 3+ (deep models, better
features, sizing) aim to clear the baseline + buy & hold bar.
