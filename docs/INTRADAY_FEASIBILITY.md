# Intraday long/short — feasibility study (crypto, 1-hour bars)

**Status:** study only — *no production code changed*. This document answers a single
question: *what would it take, and what do we need, to add intraday long & short trading
to BeRich, starting with crypto on 1-hour bars?* It ends with a go/no-go recommendation.

This is the natural continuation of the verdict in [`RESULTS.md`](RESULTS.md): *"change the
problem, not the algorithm — crypto / intraday bars / market-neutral long-short."* Here we
scope the first of those pivots.

---

## 1. Objective & scope

Today BeRich is a **daily-bar swing** system: three exit strategies (`fixed` / `trailing` /
`trailing_tp`) over horizons of a few days, each asset carrying a **long and a short** side,
all gated by the `promote()` guard.

This study scopes a **first intraday proof-of-concept**, deliberately narrowed to:

- **Asset class: crypto.** A 24/7 market means no overnight gaps, no session boundaries,
  and no US Pattern-Day-Trader (PDT) rule. Deep 1-hour history is freely available.
- **Granularity: 1 hour.** Closest to the current swing logic, least noisy, the regime
  where fees+slippage do not yet dominate the signal.
- **Goal:** measure whether an intraday edge exists *that clears the same guard* — long must
  beat the asset's buy & hold; short needs a positive, significant Sharpe vs cash. This is
  **not** an edge claim; it is a scoping study for a POC.

Explicitly **out of scope** for this first study (see §7): 5-minute scalping (transaction
costs dominate) and intraday *equities* (free history too short, PDT constraints).

---

## 2. What the architecture already does right (granularity-agnostic core)

The central finding is that most of the engine **counts in bars, not calendar days**, so it
transposes to intraday with no logic change:

- **`features/indicators.py`** — RSI / MACD / ATR / momentum / realized-vol all take a
  window in *bars*. A 14-bar ATR on 1h data is simply a 14-hour ATR.
- **`labeling/triple_barrier.py`** — triple-barrier and the causal `_trailing_touch`
  ratchet operate on a forward slice of *bars*; the ATR-scaled TP/SL barriers are
  frequency-agnostic. The causal trail (distance frozen at entry, stop set from the
  favorable extreme of *prior* bars) holds identically intraday.
- **`datasets/splits.py`** — walk-forward splits and the embargo gap are expressed as a
  *row count* equal to the label horizon. No daily assumption.
- **`models/registry.py`** — the `promote()` / `_gate_failure` guard decides on **Sharpe /
  buy-&-hold / significance**, not on time. It is exit-strategy- and granularity-agnostic.
- **The three exit strategies** (`fixed` / `trailing` / `trailing_tp`) are parameterized in
  ATR multiples and bar horizons → they map onto intraday directly.

This is why the work below is *concentrated fixes*, not a rewrite.

---

## 3. Blockers — what must change (with file:line)

Verified against the current tree. Ordered by severity.

| Severity | Location | Daily assumption | Fix for 1h crypto |
|---|---|---|---|
| **Fatal** | `data/store.py:67` — `.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()` | `.normalize()` strips the time component → every intraday bar collapses onto midnight of its day | keep the full `DatetimeIndex` (no `.normalize()`) for intraday frames |
| **Fatal** | `data/store.py:29` path `{TICKER}.parquet`; `:69` & `:75` dedup on the (normalized) index | one row per date; intraday bars seen as duplicates and dropped | separate cache namespace per interval (e.g. `{TICKER}_1h.parquet` or `data/ohlcv_1h/`); dedup on the datetime index |
| **High** | `backtest/metrics.py:18` `TRADING_DAYS = 252`; used `:61, :65, :77, :94` | Sharpe/vol annualized with `sqrt(252)`, CAGR years = `len/252` | centralize a `bars_per_year` (crypto 1h = 24×365 = **8760**); annualize with `sqrt(bars_per_year)` |
| **High** | `backtest/significance.py:27` `TRADING_DAYS = 252`; used `:130` (bootstrap Sharpe) | deflated-Sharpe annualization assumes 252 | same `bars_per_year` parameter |
| **High** | `risk/sizing.py:93` (`sqrt(252.0)`); `backtest/longshort.py:27,131,135` (`TRADING_DAYS`, borrow `/TRADING_DAYS`, vol target `sqrt(TRADING_DAYS)`) | vol targeting & borrow cost assume 252 trading days | `sqrt(bars_per_year)`; borrow cost `/ bars_per_year` |
| **High** | `features/build.py:135-157` (`_calendar_features`: `month_sin/cos`, `days_to_month_end` via `np.busday_count`); `features/microstructure.py:39` (`gap_open = open/close.shift(1) − 1`) | month-end/business-day seasonality and overnight gap are daily-equity concepts | crypto is 24/7: `gap_open` ≈ 0 bar-to-bar (continuous) → drop it; calendar features are degenerate intraday → drop, or replace with **hour-of-day / day-of-week** cyclical encodings |
| **Medium** | `horizon_days` everywhere — `config.py:39`, `labeling/triple_barrier.py`, `backtest/engine.py:43,213`, `signals/service.py`, `signals/paper.py` | the field is named "days" but is consumed as a **bar count** | semantically it is `horizon_bars`; for 1h, "1 day" = 24 bars, "5 days" = 120 bars. Either rename in the intraday subsystem or set the value in bars and document it |
| **Medium** | `scheduler/runner.py` — jobs fire once/day after the US close (22:30 Paris etc.) | once-daily cadence | crypto trades 24/7 → an hourly poll loop; retrain/refresh cadence to be chosen (it can stay much coarser than the bar interval) |
| **Medium** | `signals/paper.py:692` (`np.busday_count` days-held), `:774` (`pd.bdate_range` equity curve) | business-day stepping | continuous hourly stepping for mark-to-market and the equity curve (no weekend skip) |
| **Low** | `config/berich.yaml:8` (`interval: "1d"`), `config.py:31` (`interval: str = "1d"`) | daily default | add a dedicated `intraday` config block (interval, `bars_per_year`, `overnight_gap` flag) — leaves the daily default untouched |

**The single highest-leverage refactor** is centralizing the annualization factor: `sqrt(252)`
and `/252` appear in **7+ places**. A shared `bars_per_year` (resolved from the interval)
removes the whole class of bug at once.

---

## 4. Data sources — the "to be determined" decision

For crypto 1h history:

| Source | 1h history depth | Cost | Access | Verdict |
|---|---|---|---|---|
| **Binance public API (via `ccxt`)** | full, back to each pair's listing (years) | free | public REST `klines`, no key for market data | **recommended** — deep, free, clean |
| `yfinance` (`BTC-USD`, `interval="1h"`) | ~730 days (hard cap) | free | already integrated | fine as a quick smoke-test fallback; too short for a serious walk-forward |
| Coinbase / Kraken public API | deep | free | public REST | useful cross-check / second source |
| Polygon / Tiingo (crypto) | deep | paid | API key | unnecessary given Binance is free and deep |

**Recommendation: Binance via `ccxt`** for deep history, with `yfinance` 1h as a fallback.

**Dependency note (CLAUDE.md "avoid fragile deps").** `ccxt` would be a *new* dependency. It
is mature, widely used, and stable — unlike `pandas-ta` / Evidently, which were deliberately
rejected. It should still be added consciously: pin it, and keep the fetch behind a thin
adapter (mirroring how `data/ingest.py` wraps yfinance) so the rest of the pipeline stays
vendor-agnostic. A zero-dependency alternative exists too: Binance's `klines` REST endpoint
can be called directly with the existing HTTP stack if we prefer not to add `ccxt`.

**Starting universe:** the crypto pairs already configured (BTC, ETH, …), mapped to their
`USDT` spot pairs (e.g. `BTC/USDT`).

---

## 5. Intraday-crypto specifics to bake into the POC

- **Fees & slippage — model them honestly.** Binance spot is ~0.10%/side. At 1h this is
  survivable but must be charged in the backtest (the engine already supports flat /
  volume-proportional slippage). The "honest reporting" rule applies: if the edge dies
  after costs, say so.
- **24/7 simplifies everything.** No sessions, no overnight gaps, no PDT → the scheduler
  becomes a simple hourly loop and the label has no session boundary to handle.
- **Annualization = 8760.** `bars_per_year = 24 × 365` for continuous 1h bars; this is the
  number that makes the Sharpe meaningful (using 252 would inflate it ~6×).
- **Horizon in bars.** Pick the forward window explicitly in bars (e.g. 24 bars = 1 day,
  120 bars = 5 days) and map it onto the three existing exit strategies.
- **The guard is unchanged.** Long must beat the asset's buy & hold; short needs a positive,
  significant Sharpe (deflated Sharpe ≥ 0.95, p < 0.05). No `promote()` bypass.

---

## 6. Recommended approach — a **parallel** intraday subsystem (no daily refactor)

Do **not** retrofit intraday into the daily pipeline: that risks the 264-test suite and the
deployed production system. Instead add an intraday subsystem side-by-side:

- a **separate cache** namespace (`data/ohlcv_1h/` or `_1h` suffix), so daily Parquet files
  are never touched by the `.normalize()` change;
- a dedicated **`intraday` config block** plus a centralized `bars_per_year`;
- **intraday features** (calendar/gap features dropped or replaced by hour-of-day);
- **separately trained models**, with the model namespace under `data/models/tickers/`
  extended by a *timeframe* dimension (e.g. `<SLUG>/<side>/<strategy>/<interval>/`);
- the **daily pipeline keeps running intact**.

Files a future POC would touch (for information — **not** this phase): `data/store.py`
(conditional normalize + interval-aware path), `config.py` / `config/berich.yaml` (intraday
block), `backtest/metrics.py` + `backtest/significance.py` + `risk/sizing.py` + `backtest/longshort.py`
(centralized annualization), `features/build.py` + `features/microstructure.py` (intraday
feature set), a new `ccxt`/Binance adapter alongside `data/ingest.py`.

---

## 7. Effort estimate & risks

**POC effort** (one crypto pair, 1h, end-to-end), broken into independently testable lots:

1. **Data adapter** — Binance/ccxt fetch behind an adapter; land 1h klines into the store.
2. **Store** — interval-aware path + skip `.normalize()` for intraday; datetime dedup.
3. **Annualization** — replace hardcoded 252 with a resolved `bars_per_year` (one shared helper).
4. **Features** — intraday feature set (drop calendar/gap; optional hour-of-day encoding) + a causality test.
5. **Label / backtest** — horizon in bars; run `fixed` / `trailing` / `trailing_tp`.
6. **One gated walk-forward** — train + tournament on the pair, run it through `promote()`.

Each lot is small and verifiable on its own; lots 2–4 are the bulk of the real work, and lot
3 is mechanical but touches several files. Net: a contained spike, not a platform rewrite.

**Risks:**

- **New dependency** (`ccxt`) — mitigated by an adapter and a pin (or a direct-REST fallback).
- **Data volume** — 8760 bars/year/asset (≈ daily × 24); storage and retrain time grow
  accordingly. Manageable for a handful of pairs; plan before scaling the universe.
- **Retrain cost** — far more bars per fit; the continuous retrainer cadence must be re-tuned.
- **Edge uncertainty** — crypto 1h microstructure is noisy; an edge that clears the guard is
  not guaranteed. That is exactly what the POC is meant to find out.

**Anti-goals for this first study:** no 5m scalping (costs dominate), no intraday equities
(yfinance history too short, PDT rule).

---

## 8. Verdict — go / no-go

**Technically feasible.** The core (indicators, triple-barrier, causal trailing, walk-forward,
guard) is already bar-agnostic; the blockers are concentrated, identified, and mostly
mechanical (annualization centralization + a store/cache change + an intraday feature set).

**Data is available for free.** Binance via `ccxt` (or direct REST) gives deep, clean 1h
history — no paid vendor required.

**Risk is contained** by building a *parallel* intraday subsystem rather than refactoring the
deployed daily pipeline.

**Recommendation: GO for a narrow POC** — one pair (e.g. `BTC/USDT`, 1h) taken end-to-end
through the existing guard, to measure whether an intraday edge clears `promote()` before any
generalization to more pairs or finer granularities.
