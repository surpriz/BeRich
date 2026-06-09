# Phase 3 results — honest writeup

**TL;DR.** No edge in OHLCV + cross-asset macro for daily, long-only swing
trading on the 10-ticker watchlist. The pipeline (data, features, labels,
walk-forward backtest, drift, dashboard, sizing) is solid; the model does not
beat buy & hold. v0.1.0 is shipped as **advisory infrastructure with no edge
claim**.

## Methodology (fixed across all runs)

- **Data.** 10 US equities (AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM,
  JNJ, SPY), daily adjusted OHLCV via yfinance, ~4100 bars per name from
  2010-01-01 onwards. SPY also serves as buy & hold benchmark.
- **Labels.** Triple-barrier (López de Prado): for each bar `t`, place an
  upper barrier at `close[t] + 2 · ATR[t]` and a lower at `close[t] − 1 · ATR[t]`,
  watch the next 10 bars. `y = 1` iff the upper barrier is hit first. Sample
  weight = `|realized return|` to upweight decisive moves. Last `horizon`
  bars per ticker are NaN (incomplete forward window) and dropped.
- **Validation.** Walk-forward (expanding train, fixed test) with an embargo
  gap equal to the label horizon (10 bars). All scalers fit on the train
  fold only. The reported metric is **out-of-sample** AUC: the model never
  sees the rows it scores.
- **Backtest.** Event-based, long-only: enter at next bar's open when
  `P(win) ≥ threshold`, exit on ATR stop, ATR target, or time barrier. 1 bp
  commission and 5 bp slippage per side. Strategy Sharpe is compared to an
  equal-weight buy-and-hold of the same 10 tickers over the same OOS window.
- **Guard rule.** A model is only promoted to serving if its OOS Sharpe
  beats **both** the LightGBM baseline **and** buy & hold. Enforced in
  `src/berich/models/registry.py::promote`.

## Phase 3a — feature engineering

Starting point: 13 features (returns, RSI, MACD, ATR/vol ratios, SMA ratios,
volume z-score). Added in 3a:

- Calendar (sin/cos day-of-week, sin/cos month, days-to-month-end).
- SPY regime: `spy_ret_20`, `spy_rvol_20`, lagged 1 bar (no same-day leakage).
- Longer momentum: `mom_60`, `mom_120`.
- Mean-reversion: `dist_high_60`, `dist_low_60`.

**Result.**

| set | AUC | Sharpe | B&H Sharpe |
|-----|------|--------|------------|
| 13 features (pre-3a) | 0.5173 | 0.610 | 1.149 |
| 24 features (3a)     | 0.5132 | 0.358 | 1.149 |

More features did not help LightGBM. The importance diagnostic later showed
two features (`dow_sin` 0.4 %, `dow_cos` 0.6 %) at noise level — dropped in a
follow-up cleanup, taking the set to **22 features**.

## Phase 3b — LSTM baseline (GPU)

- Architecture: 2-layer LSTM, hidden 128, dropout 0.2, lookback 60, lr 1e-3,
  batch 256, early stop on a per-ticker val tail.
- Per-ticker sequence streams so lookback windows never mix bars from
  different symbols (the `Model` protocol carries an optional `tickers:
  pd.Series` kwarg for this; LightGBM ignores it).
- MLflow logs every run: hyperparameters, OOS AUC, backtest metrics, the
  serialized artifact, the promotion verdict.
- Trained on 1× RTX PRO 4000 Blackwell, ~10 min per OOF walk-forward.

**Result.**

| model | AUC | Sharpe |
|-------|------|--------|
| LightGBM baseline (24 feat) | 0.5132 | 0.358 |
| LSTM (lookback 60, h=128)   | **0.5234** | 0.141 |
| Buy & hold                  | —     | 1.149 |

LSTM nudges AUC up slightly (~0.01) but Sharpe regresses (lower win rate at
trade level). It beats neither the LightGBM baseline nor buy & hold;
`promote()` refused the artifact, as designed. No threshold-twisting to make
it pass.

## Label-geometry sweep

Hypothesis: maybe the (horizon=10, TP/SL=2/1·ATR) label shape isn't a fair
swing-trade target. Held features and LightGBM constant, varied
`(horizon, TP, SL)`:

| variant            | P(win) | AUC | Sharpe |
|--------------------|------|--------|--------|
| h=10 TP=2 SL=1 (current) | 0.325 | **0.5132** | 0.358 |
| h=10 TP=1.5 SL=1.5       | 0.486 | 0.4961 | 0.333 |
| h=20 TP=2 SL=1           | 0.393 | 0.4888 | 0.444 |
| h=20 TP=1.5 SL=1.5       | 0.558 | 0.4813 | 0.431 |

Three of four variants are sub-coin-flip (AUC < 0.5). The asymmetric
(2 · ATR up / 1 · ATR down) labeling is the only one where LightGBM extracts
*any* signal at all — and even there, Sharpe is far short of buy & hold.
Reshaping the label does not unlock an edge.

## Cross-asset features (later reverted)

Added: VIX level + 5d delta + term ratio (VIX9D/VIX), TLT 20d return + 60d
distance to high, HYG−LQD credit spread, sector-relative 20d return (per
ticker, mapped to its sector ETF). All lagged 1 bar.

LightGBM importance ranking flipped — **the top 5 became 100 % cross-asset**:

```
spy_rvol_20         7.20 %
tlt_dist_high_60    6.03 %
vix_level           5.33 %
spy_ret_20          5.11 %
tlt_ret_20          4.77 %
```

7 of the top 10 features were SPY / VIX / TLT / credit / sector. All ticker-
own indicators (mom_10/20, close_smaXX_ratio, volume_z20) dropped to the
bottom. The model clearly wants the macro context — and yet:

| set                  | n samples | AUC    | Sharpe |
|----------------------|-----------|--------|--------|
| 22 features (post-drop-dow) | 39341 | 0.5169 | 0.432 |
| 29 features (cross-asset)   | 34816 | 0.5053 | 0.381 |

(n drops because XLC only exists from late 2018 — META / GOOGL pre-2018 rows
fall out of the join.)

**The macro signal exists but does not convert** to a tradeable
single-name probability at the 10-day horizon. Knowing the regime well at
`t−1` does not help predict whether *this specific name* hits its TP before
its SL inside 10 trading days.

These features were reverted in commit `252fb6a`. The `SECTOR_MAP` helper
dict is retained in `features/build.py` as a reference; reopening the
cross-asset track requires a new data-source rationale (see CLAUDE.md
"House rules").

## Conclusion

- No edge found in OHLCV + cross-asset macro for the daily, long-only,
  single-name swing target on this 10-ticker universe.
- The asymmetric (2/1 · ATR) triple-barrier label is the one that produces
  a faint signal (~0.51–0.52 AUC). Alternative labelings tested are all
  worse.
- LightGBM and LSTM converge to similar verdicts; the LSTM's small AUC
  bump does not translate into a Sharpe improvement.
- **v0.1.0 ships as advisory infrastructure.** The dashboard displays an
  "advisory only — model does not beat buy & hold" banner; the registry
  refuses to promote a sub-buy-hold model; no edge is claimed.

## Phase 5a — earnings features (free data, no GPU)

The hypothesis: earnings surprises are exogenous information not encoded in
price or macro context, so they might lift the single-name signal. Six
features were added behind a `--with-earnings` flag (default off):
`days_to_next_earnings`, `days_since_last_earnings`, `last_surprise_pct`,
`surprise_4q_mean`, `pre_earnings_window`, `post_earnings_window`. Data comes
from `yfinance.Ticker(...).earnings_dates` and is cached under
`data/earnings/*.parquet`. Coverage is ~6 years (~25 quarters per stock);
ETFs and indices return empty — the feature builder substitutes neutral
defaults so SPY survives the join.

**Result on the same 10-ticker watchlist (LightGBM, walk-forward OOS):**

| feature set         | AUC    | Sharpe | B&H Sharpe |
|---------------------|--------|--------|------------|
| 22 (baseline)       | 0.5169 | 0.432  | 1.149      |
| 22 + 6 earnings     | 0.5117 | 0.399  | 1.149      |

Both AUC and Sharpe regress slightly. Promote threshold (AUC > 0.54 AND
Sharpe > buy & hold) is not met by a wide margin; no model was promoted.
The Phase 5a code is merged behind the `--with-earnings` flag so future
experiments can flip it without re-implementing the plumbing.

Importance ranking (top 10, 28-feature model) is informative:

```
spy_rvol_20             10.31 %
spy_ret_20               6.42 %
mom_120                  5.62 %
atr_pct                  4.75 %
month_cos                4.72 %
dist_low_60              4.45 %
days_to_month_end        4.43 %
month_sin                4.33 %
last_surprise_pct        4.33 %  ← earnings
rvol_20                  4.24 %
```

`last_surprise_pct` (rank 9, 4.33 %) and `surprise_4q_mean` (rank 13, 3.92 %)
are non-trivially used by the model — they carry real predictive content —
yet the net effect on AUC/Sharpe is mildly negative. Same pattern as the
cross-asset macro features in Phase 3: the model uses them, but the
information doesn't translate into a tradeable single-name probability at
the 10-day horizon. The pre/post-earnings binary windows are dead weight
(0.00 % importance — too sparse to learn from).

The honest read: earnings surprises *are* informative at the price-reaction
scale (which is well documented), but our triple-barrier, daily, single-name
target captures the wrong part of that information for a tradable edge.
Same verdict shape as Phase 3 — different data, same conclusion. Moving on
to Phase 5b (news/sentiment via FinBERT) is the next plausible probe; the
earnings feature plumbing stays in the repo for any future model that wants
to use it without re-implementing it.

## Phase 5b — news sentiment via Alpha Vantage + FinBERT (GPU)

The hypothesis: text-based news sentiment is qualitatively different from
price/macro/earnings (which all decompose to numbers from a single OHLCV
feed). Seven features were added behind a `--with-news` flag:
`news_count_5d`, `news_count_20d`, `sentiment_mean_5d`, `sentiment_std_5d`,
`sentiment_extreme_5d`, `sentiment_av_5d`, `sentiment_price_div`.

**Data pipeline:**
- Source: Alpha Vantage `NEWS_SENTIMENT` (free tier, 25 req/day, 1000
  articles/req). Cached under `data/news/<TICKER>.parquet`. Pagination
  walks forward from the cache tip; dedupe by URL.
- Scorer: `ProsusAI/finbert` (HuggingFace), loaded on `cuda:0` in half
  precision, batch 64, max 256 tokens (title + first paragraph of summary).
  Score: `pos - neg` from the (neg, neu, pos) softmax — symmetric in [-1, 1].
- ~22,400 articles scored end-to-end on the 10-ticker watchlist between
  2022-04-01 and the most recent fetch.

**Honest caveat on the window.** The news coverage starts only in April
2022 (AV's index goes back that far for most tickers and no further),
giving ~3.7 years of overlap with the OHLCV history. Walk-forward folds
on this window are short and statistically less reliable than the
~15-year backtest. The comparative table below filters the **baseline
to the same window** so it's apples-to-apples.

| variant                          | samples | AUC    | Sharpe | B&H Sharpe | beats |
|----------------------------------|---------|--------|--------|------------|-------|
| 22 + earnings (window-only)      | 10320   | 0.5087 | 0.276  | 1.149      | False |
| 22 + earnings + 7 news           | 10320   | 0.5003 | 0.241  | 1.149      | False |

News features regress both AUC and Sharpe modestly. Promote gate
(`AUC > 0.55 AND Sharpe > B&H`) is not met by a wide margin → **no model
promoted**.

Top-10 importance ranking (35-feature model, news on):

```
spy_rvol_20             7.48 %
last_surprise_pct       5.40 %  ← earnings
spy_ret_20              5.23 %
rvol_20                 4.96 %
surprise_4q_mean        4.88 %  ← earnings
atr_pct                 4.30 %
mom_120                 4.24 %
news_count_20d          3.82 %  ← news
month_sin               3.78 %
month_cos               3.69 %
```

`news_count_20d` slips into the top 10 — the model does use the *volume*
of coverage as a signal. The actual FinBERT sentiment columns
(`sentiment_mean_5d`, `sentiment_std_5d`, …) sit lower in the ranking and
don't help on this window. Same shape as every earlier exogenous-feature
experiment in Phase 3 and 5a: the model picks up the new signal, but it
doesn't translate to a tradable single-name probability at the 10-day
horizon.

**Why the news count works while the sentiment doesn't (best guess):**
the volume of news on a ticker correlates with already-elevated activity
(earnings windows, big macro days, idiosyncratic events), so the model
uses count as a regime proxy similar to realized volatility. The actual
FinBERT polarity adds nothing on top of that volume signal at the
10-day horizon — by the time we'd act, the news has been priced in.

The Phase 5b code is merged behind the `--with-news` flag; the news /
FinBERT pipeline + GPU scoring + scheduler integration are in place so a
future attempt with a different label (intraday? per-event reaction
window?) can reuse all of it.

## Horizon sweep (v0.2.0 follow-up)

Every result above used the default horizon `h=10`. The Phase 3 label
sweep tested 10 vs 20 with different TP/SL ratios but never shorter
horizons. Holding TP/SL at 2/1·ATR and the 22-feature baseline constant,
we ran h ∈ {3, 5, 7, 10}:

| h  | samples | P(win) | AUC    | Sharpe | B&H Sharpe | trades | beats |
|----|---------|--------|--------|--------|------------|--------|-------|
| 3  | 39421   | 0.101  | **0.5834** | 0.087  | 1.149      | 88     | False |
| 5  | 39401   | 0.195  | **0.5576** | 0.387  | 1.149      | 482    | False |
| 7  | 39381   | 0.261  | 0.5358 | 0.348  | 1.149      | 1135   | False |
| 10 | 39351   | 0.325  | 0.5153 | 0.398  | 1.149      | 1710   | False |

**AUC drops monotonically as the horizon lengthens** — at h=3 the model
scores 0.5834, a level we never saw across any feature-engineering
experiment, including news sentiment. The signal is real and the shorter
window makes it visible: there's less noise to confuse the classifier.
Two horizons (h=3, h=5) clear the AUC promote bar (0.54).

So why didn't we promote? **Sharpe never beats buy & hold** at any
horizon. At h=3 the model only fires 88 trades over ~15 years of data
(P(win)=10%, very few setups clear threshold 0.5), so even good
predictions don't accumulate to enough alpha; at h=5 we get 482 trades
but Sharpe still tops out at 0.387 vs B&H's 1.149. The model
**knows something** at short horizons but the long-only triple-barrier
construct doesn't convert that knowledge into spread above passive equity.

This is the most informative single result in the project to date: it
splits the previous "no edge anywhere" verdict into "no *tradable*
edge under this target shape, but the underlying directional signal at
3–5 day horizons is non-trivial". The next probe is naturally to change
the target itself — either lower the entry threshold to harvest more h=3
opportunities (and re-test the trade frequency / cost trade-off), or
move to per-event labels where the model's short-horizon edge could
translate into a different P&L mechanism.

## Phase 6 — universe expansion (mega vs mid vs small vs all)

The hypothesis: mega-caps are chased by everyone (lower edge potential);
small/mid caps have less analyst coverage and more documented anomalies, so
**if** a tradable edge exists for daily/swing US equities, it should be
visible there. We held the model + features constant (LightGBM on the 22
base features, no earnings or news) and ran the same backtest on four
universes with **volume-proportional slippage** so small-caps pay realistic
fill costs (slippage_bps ∝ √(SPY_volume / ticker_volume), capped at 100 bps).

Data: ~274 unique tickers cached (10 mega + 113 mid + 151 small), ~985k
total sample-rows. Each universe's benchmark is an equal-weight buy & hold
of **the same** tickers — apples-to-apples on Sharpe.

| universe | tickers | in DS | samples  | P(win) | AUC    | Sharpe  | B&H     | beats | max DD  | trades |
|----------|---------|-------|----------|--------|--------|---------|---------|-------|---------|--------|
| mega     | 10      | 10    | 39 351   | 0.325  | 0.5153 | **+0.31** | **+1.15** | False | -0.25   | 1 710  |
| mid      | 113     | 110   | 413 874  | 0.289  | 0.5108 | **-0.84** | +0.95   | False | -0.65   | 13 176 |
| small    | 151     | 141   | 532 135  | 0.276  | 0.5109 | **-1.00** | +0.78   | False | -0.73   | 13 384 |
| all      | 274     | 261   | 985 360  | 0.283  | 0.5069 | -0.92   | +0.87   | False | -0.69   | 27 634 |

**The hypothesis is rejected — backwards.** Two observations:

1. **AUC is essentially flat across universes** (0.5069–0.5153). The
   model has the same modest predictive ability everywhere; small/mid caps
   aren't "less efficient" in the sense of being easier to predict.
2. **Realistic slippage destroys the strategy on small/mid caps.** Sharpe
   goes from +0.31 (mega) to *negative* (mid/small/all). Buy & hold also
   degrades on these universes (1.15 → 0.95 → 0.78 — small-caps have lower
   risk-adjusted returns) but the strategy degrades much faster: gross
   directional signal exists, but ~50 bps/side slippage on a 13 000-trade
   book eats it whole.

Top-10 importances on the 'all' universe model are revealing:

```
spy_rvol_20             22.82 %
spy_ret_20              19.48 %
days_to_month_end       12.63 %
month_cos                9.22 %
month_sin                8.70 %
mom_120                  2.88 %
rvol_20                  2.29 %
dist_low_60              2.25 %
ret_5                    2.24 %
atr_pct                  1.97 %
```

The wider-universe model leans **heavily on macro regime + calendar**
(SPY features + month encoding = 73% of total importance). Per-ticker
features that were prominent on mega-caps (atr_pct, mom_120, rvol_20)
fall to <3% each. With 985k samples the model finds that the
cheapest-to-predict signal is "everyone moves with SPY and with the
month-end cycle" — and that's not enough to outrun friction.

**Verdict:** no promotion. The mega-cap baseline remains the only
universe where the strategy at least produces a positive (if sub-B&H)
Sharpe. The Phase 6 plumbing (universes config, parallel ingest,
liquidity gate, volume-proportional slippage) is merged behind
``--universe`` / ``--volume-slippage`` flags for future experiments;
the v0.2.0+ default production behavior is unchanged.

## h=3 exploit sweep (TP × SL × threshold)

The horizon-sweep finding ("AUC 0.5834 at h=3 but only 88 trades") begged
the question: is the signal harvestable with better trade engineering?
Tested 4 TP/SL combos × 4 entry thresholds (3 quantile modes + the legacy
0.5 baseline). Quantile thresholds set so the strategy fires on the
top-N% of bars by model confidence.

Best combo: **TP=1.5·ATR, SL=1.0·ATR, top-quintile threshold** (entry at
proba ≥ 0.402). Clears AUC gate at 0.5511 and produces a positive Sharpe
0.370 over 1 529 trades, but the buy & hold benchmark stays at 1.149 →
gate (AUC > 0.55 AND Sharpe > B&H AND trades ≥ 200) **not met, no
promotion**.

Symmetric barriers (1/1, 0.75/0.75) gave lower AUC (0.5067–0.5192) than
the asymmetric ones (1.5/1.0 at 0.5511, 1.0/1.5 at 0.5209) — the model
extracts more signal when the upper barrier is comfortably above the
lower, which is consistent with the Phase 3 label sweep result that the
asymmetric (2/1) labeling was the only one where LGBM extracted positive
predictive value at all.

Honest verdict on h=3: the signal **is** real and reproducible (AUC
0.55–0.58 depending on the label geometry) but it doesn't scale into
enough Sharpe to outrun a 15-year equity bull market in a long-only
book. Better trade engineering moved Sharpe from ~0.08 to ~0.37 — a
meaningful improvement, but the benchmark stays ~3× ahead.

## Calendar-only baseline (turn-of-month, no ML)

The Phase 6 importances showed the wider-universe model leaning heavily
on macro + calendar features (~73 % of total). That raised an honest
question: does a no-ML turn-of-month rule already capture most of the
calendar effect?

Rule: long during the last 3 + first 3 business days around each month
boundary (≈ 29 % time-in-market), flat the rest of the year, with the
same fee + slippage assumptions as the ML backtests. Equal-weight across
the 10 mega-caps; benchmark is the same universe held 100 % of the time.

| metric          | turn-of-month | buy & hold   | delta    |
|-----------------|---------------|--------------|----------|
| total return    | +164 %        | +3 029 %     | -2 865 % |
| CAGR            | 6.1 %         | 23.4 %       | -17.3 %  |
| ann vol         | 10.7 %        | 20.1 %       | -9.4 %   |
| **Sharpe**      | **0.61**      | **1.15**     | -0.54    |
| max drawdown    | -17.5 %       | -31.7 %      | +14.1 %  |

Turn-of-month delivers Sharpe 0.61 by itself — **higher than the
LightGBM baseline (~0.43) on the same universe**. The risk-adjusted
return is real (lower vol, smaller drawdowns, half the time in the
market), the absolute return is much smaller because the rule sits in
cash 71 % of the year. Still doesn't beat the full-time-long B&H Sharpe
(1.15), but the bar is lower.

The honest read: most of the calendar "lift" the ML model picks up via
``days_to_month_end`` and ``month_sin/cos`` was already capturable with
a no-ML rule. The ML model isn't providing meaningful incremental
information above this trivial seasonality on the mega-cap watchlist —
they both fail to beat B&H, but the calendar rule does so with less
machinery and slightly better Sharpe. Worth recording as a discipline
test for any future model: it must clear the turn-of-month baseline,
not just buy & hold.

## Phase 7 — PEAD (Post-Earnings Announcement Drift)

The most documented anomaly in the academic finance literature: stocks
that beat (miss) earnings expectations tend to drift up (down) for
several days after the announcement. Bernard & Thomas (1989) and
follow-up work make it the canonical "event-driven" effect. v0.2.0 had
exhausted the "predict every day" approach; this is the "predict
specific events" experiment.

**Implementation:**
- New event-level labelling (``src/berich/labeling/pead.py``): one row per
  (ticker, earnings announcement) instead of one per trading day. Entry
  date = first trading day strictly after the announcement (after-close
  convention). Forward returns over 5 and 20 trading days. Binary labels:
  ``drift_5d`` = 1 iff fwd 5d return > 2%; ``drift_20d`` = 1 iff > 5%.
- Event-level features (``src/berich/features/pead_features.py``): 11
  features computed strictly at the event date (no leakage from the
  forward window). Mix of earnings-specific (surprise_pct, surprise_4q_mean,
  days_since_last_earnings), pre-event price action
  (pre_announce_run_5d, pre_announce_vol_20d, pre_rsi_14), macro
  regime (market_regime_spy_rvol, sector_relative_5d), size proxy
  (log_volume_median), and news (news_count_pre_5d, sentiment_mean_pre_5d).
- Event-driven backtest: long the predicted-positive events at the
  entry-day **open** (with 10 bps slippage to model after-hours fills),
  exit at the close ``hold_days`` later (5 bps slippage). Standard fees.
  Benchmark = same (ticker, entry → exit) windows held long with no
  signal filter — i.e. "what if you blindly long every earnings window".
- Walk-forward chronological at the event grain, 5 folds, min 500 train events.

**Data:** earnings calendar fetched via yfinance for all 274 tickers
(mega + mid + small caps), ~6 800 cached announcements total. After
windowing (need 20 forward bars) and filtering to tickers with valid
OHLCV: **6 066 PEAD events** spanning 2010-01 to 2026-04. P(drift_5d) = 0.362.

**Result on the full universe:**

| metric                  | value     |
|-------------------------|-----------|
| events                  | 6 066     |
| OOS AUC                 | **0.5346** |
| n trades (threshold 0.5)| 1 174     |
| strategy Sharpe         | **0.849** |
| window-B&H Sharpe       | **1.013** |
| beats window B&H        | False     |
| strategy total return   | +737 %    |
| strategy max DD         | -78.7 %   |

Promote gate (AUC > 0.55 AND Sharpe > window B&H AND events >= 1000)
**not met** — AUC fell just shy of 0.55, Sharpe is below the
no-signal-filter window benchmark. No promotion.

Two readings of this:

1. **PEAD is the best AUC any model has produced on a sizable sample
   in this project** (0.5346 over 6 066 events vs ~0.51 on every
   daily-bar attempt). The directional signal exists at the event
   grain — substantially more than at the daily grain.

2. **It still doesn't beat blindly long every earnings window.** The
   benchmark — hold the stock from the day after each announcement to
   5 days later, every single event — earns Sharpe 1.013 in this
   sample. The model's filter (predict-positive → take the trade)
   raises trade-level AUC but lowers per-period Sharpe vs the
   "always-on" event exposure. Consistent with the literature finding
   that PEAD has decayed since the 2000s as the effect became widely
   known and traded out.

Top-10 PEAD feature importances:

```
market_regime_spy_rvol  20.43 %
surprise_pct            14.01 %
log_volume_median       13.31 %
pre_announce_vol_20d    12.74 %
surprise_4q_mean        12.31 %
pre_rsi_14              12.04 %
pre_announce_run_5d     11.57 %
days_since_last_earnings 3.09 %
sector_relative_5d       0.21 %
sentiment_mean_pre_5d    0.18 %
news_count_pre_5d        ...
```

The earnings-specific signals **are used** (surprise + 4q mean +
days-since together = ~30 % of importance) but the macro-regime feature
still leads the ranking — same pattern as Phases 5–6. News features
(sentiment + count) add nothing at the event grain either — they were
worth less than 0.5 % combined, mirroring Phase 5b's verdict.

The PEAD code stays merged behind ``berich pead {train|backtest}`` and
the paper-trade store now carries a ``source`` column so a future PEAD
promotion would track separately from the daily-LGBM book. Production
serving is unchanged.

## Phase 8 — PEAD + risk management overlay

Hypothesis: PEAD's directional signal is real (AUC 0.5346, Sharpe 0.849)
but no risk management has been applied yet — equal sizing, no regime
gating, no drawdown protection. Industry wisdom is that 80 % of a quant
strategy's Sharpe comes from risk management, not signal. This is the
last untested lever before the v0.3.0 verdict closes.

**Implementation:**
- New ``berich.risk`` module with three pure-math primitives
  (``kelly_fraction``, ``vol_target_size``, ``inverse_vol_size``), two
  binary gates (``regime_gate_spy`` based on rolling SPY-rvol quartile,
  ``drawdown_gate`` based on running peak equity), and a combiner
  ``RiskOverlay`` that runs them in order: threshold → gates → sizer.
- ``berich.backtest.pead_engine`` re-uses the same PEAD OOF probas but
  layers the overlay event-by-event. Gates see only realized history
  before each event (no test-window leakage). Benchmark = the same
  "blind long every event window" but scaled by the same per-event
  position so the Sharpe comparison isolates the overlay's effect.
- Walk-forward at the parameter level too: the regime-gate cutoff is a
  rolling quartile (252-day lookback ending at event-1, never the
  full sample); the drawdown gate watches only realized strategy
  equity to date; Kelly W/L are configuration constants tuned to the
  PEAD label thresholds (W=4 %, L=2 %).

**Comparative result on the 6 066-event PEAD dataset:**

| overlay     | AUC     | Sharpe   | window-B&H | total_return | max_DD | trades | gated | beats |
|-------------|---------|----------|-----------:|--------------|--------|--------|-------|-------|
| off         | 0.5346  | 0.849    | 1.013      | +17.7 %      | -6.7 % | 1 174  | 0     | False |
| vol_gate    | 0.5346  | 0.428    | 0.589      | +5.8 %       | -6.4 % | 746    | 428   | False |
| dd_gate     | 0.5346  | 0.849    | 1.013      | +17.7 %      | -6.7 % | 1 174  | 0     | False |
| kelly       | 0.5346  | 0.849    | 1.013      | +17.7 %      | -6.7 % | 1 174  | 0     | False |
| all         | 0.5346  | 0.428    | 0.589      | +5.8 %       | -6.4 % | 746    | 428   | False |

(Daily-return Sharpe; the per-event aggregation is averaged over the
event-day calendar, hence the modest total_return relative to the raw
1 174-trade Phase 7 figure that compounded each trade independently.)

**No overlay beats the buy-and-hold benchmark.** Three independent
findings:

1. **Drawdown gate never fired.** Strategy max DD on the sized book
   is only 6.7 %, well below the 20 % cut-off. PEAD is naturally
   drawdown-constrained at the daily-aggregated event level — the
   gate has nothing to protect against.
2. **Kelly + inverse-vol sizing leave Sharpe unchanged.** Because both
   strategy and benchmark are scaled by the same per-event size in
   our (honest) apples-to-apples comparison, a uniform multiplier
   cancels in the Sharpe ratio. The sizer matters in absolute return
   (it's hard-capped at 5 % per trade by ``max_position``), not in
   risk-adjusted return relative to the same-window benchmark.
3. **Regime gate skips winners.** Filtering out events when SPY's
   trailing rvol is in the top quartile removes 428 of 1 174 trades
   — but Sharpe drops from 0.849 to 0.428, and the *benchmark on the
   surviving events* also drops (1.013 → 0.589). High-vol-regime
   earnings events were ON AVERAGE profitable, so the regime gate
   filters out winners, not losers. The "don't trade during macro
   stress" intuition doesn't hold for PEAD specifically.

Promote gate (``Sharpe > window-B&H AND AUC stable``) **not met**.
No promotion. The risk overlay code stays merged behind
``scripts/pead_risk_managed.py`` for future use, but no live model
changes.

## Phase 9 — core-satellite portfolio (the reframe)

Hypothesis: the failures so far compared single-strategy Sharpe against
B&H, but real quant books are core-satellite — a beta core (B&H) plus
uncorrelated alpha overlays. The PEAD signal has the right shape (Sharpe
0.849 at the trade level, low drawdown, ~15 % time-in-market) for a
satellite, not a replacement. A 90 % B&H + 10 % PEAD blend, rebalanced
monthly, *should* clear pure B&H if PEAD is genuinely uncorrelated.

**Implementation:**
- ``backtest/portfolio.py``: ``run_portfolio_backtest`` takes named
  daily-return series + a weight policy (static or walk-forward) and
  simulates monthly rebalance with turnover-based cost (default 1.5 bps).
- ``backtest/strategies.py``: three composable return series —
  ``build_bnh_returns`` (SPY long), ``build_pead_returns`` (avg of
  currently-held PEAD events, cash on flat days), ``build_calendar_returns``
  (turn-of-month SPY).
- ``scripts/portfolio_sweep.py``: hand-picked static grid +
  walk-forward weight optimization (SLSQP, max in-sample Sharpe with
  weights ≥ 0 summing to 1, 1-year test folds after a 3-year warm-up).

**Correlation matrix** of the three components (2010-2026):

|                | bnh_spy | pead  | calendar_spy |
|----------------|---------|-------|--------------|
| bnh_spy        | 1.000   | 0.343 | 0.511        |
| pead           | 0.343   | 1.000 | 0.187        |
| calendar_spy   | 0.511   | 1.000 | 0.187        |

Calendar is too correlated with B&H (0.51) to diversify — it's
essentially a watered-down B&H with less time in market. PEAD has a
more moderate 0.34 correlation but, as we'll see, the absolute return
component is too thin to compensate.

**Static grid result** (B&H Sharpe = 0.850):

| mix         | Sharpe | Δ vs B&H | total_ret | max_DD | ann_vol | turnover |
|-------------|--------|---------:|-----------|--------|---------|----------|
| 100/0/0     | 0.850  |  0.000   | +789 %    | -33.7% | 16.8 %  | 0.00     |
| 95/3/2      | 0.843  | -0.006   | +723 %    | -32.2% | 16.3 %  | 0.69     |
| 90/5/5      | 0.838  | -0.011   | +664 %    | -30.7% | 15.8 %  | 1.25     |
| 90/10/0     | 0.819  | -0.030   | +652 %    | -30.9% | 16.1 %  | 1.61     |
| 80/20/0     | 0.770  | -0.080   | +531 %    | -28.2% | 15.7 %  | 2.88     |
| 80/10/10    | 0.820  | -0.030   | +554 %    | -27.6% | 14.8 %  | 2.28     |
| 70/15/15    | 0.792  | -0.058   | +458 %    | -24.6% | 14.0 %  | 3.10     |
| 70/30/0     | 0.703  | -0.147   | +425 %    | -28.3% | 15.6 %  | 3.79     |
| 60/40/0     | 0.621  | -0.229   | +333 %    | -29.4% | 16.0 %  | 4.34     |

**Pure B&H is best.** Every overlay reduces Sharpe — the satellites
trade some volatility reduction (ann_vol drops from 16.8 % to 14.0 %)
for a bigger absolute return loss. The overlays don't add risk-adjusted
return; they shave volatility off a strategy whose Sharpe already
captures the time-in-market advantage of always being long.

**Walk-forward weight optimization** (13 folds, 1 year each, 2012-11
→ 2025-06):

```
fold 1  (2012-11):  bnh=0.05, pead=0.00, calendar=0.95
fold 2  (2013-11):  bnh=0.22, pead=0.00, calendar=0.78
...
fold 7  (2018-09):  bnh=1.00, pead=0.00, calendar=0.00
...
fold 13 (2024-07):  bnh=0.58, pead=0.00, calendar=0.42

  OOS Portfolio Sharpe = 0.634
  OOS B&H Sharpe       = 0.848
  delta                = -0.214
```

The optimizer **never picked PEAD** in any fold — calendar dominated
the early years (low-vol bias paid off in choppy markets), then B&H
won as the bull market accelerated. The forecast-on-train / apply-on-test
discipline cost ~21 Sharpe points vs naive pure B&H. The instability
of the early calendar-heavy allocations + the cost of monthly
rebalancing make the dynamic policy *worse* than just holding SPY
through the whole period.

**Promote gate** (portfolio OOS Sharpe > pure B&H + 0.05) NOT MET by
any variant — best static is the pure-B&H control itself (delta 0),
walk-forward is -0.214. No promotion.

## Final verdict — v0.4.0 (still)

9 phases of exploration over OHLCV, macro cross-asset, earnings
surprises, FinBERT news sentiment, mid/small-cap universes, short
horizons, post-earnings drift, risk management, and core-satellite
portfolio reframing. **No combination tested publicly beats buy & hold
in walk-forward with realistic fees and slippage on the US daily
long-only universe.**

The Phase 7 PEAD result (AUC 0.5346, Sharpe 0.849, max DD -6.7 % on
the daily-aggregated event book) is the closest the project comes to
a usable signal — and Phase 8 confirms that adding standard risk
machinery on top doesn't unlock the additional Sharpe needed to
clear the benchmark.

Recommended pivot for any future iteration: **change the problem,
not the algorithm**. The US daily long-only space is too efficient
for the levers explored. Plausible next environments:

- **Crypto** (24/7, retail-dominated, less efficient micro-structure).
- **Intraday** (1-minute / 5-minute bars where short-horizon signals
  matter and PEAD-like reactions are sharper).
- **Market-neutral long/short** (cross-sectional ranking on the
  ~274-ticker universe with paired short positions — removes the
  ~10 %/yr equity beta from the benchmark, making "edge" possible
  even with thin AUC).

The **crypto / 1-hour intraday** pivot is now scoped in a dedicated study —
[`INTRADAY_FEASIBILITY.md`](INTRADAY_FEASIBILITY.md): the engine is already
bar-agnostic, the blockers are concentrated (store `.normalize()`, `sqrt(252)`
annualization, calendar/gap features), data is free (Binance/ccxt), and the verdict
is *go for a narrow BTC/USDT 1h POC* through the existing guard.

The v0.2.0+ infrastructure (data layer with multi-universe + earnings
+ news + GPU FinBERT, walk-forward backtest with realistic costs,
event-level PEAD + risk overlay, registry guard, paper book,
production deployment) is the platform on which any of these would
sit — none of the new direction requires throwing away what's built.

## Open project (not started)

After Phase 3 (OHLCV + macro), 5a (earnings surprises), 5b (news
sentiment), Phase 6 (universe expansion), the h=3 exploit sweep, the
calendar-only sanity check, Phase 7 PEAD, and Phase 8 risk
management, the project has pulled every reasonable lever short of
changing the very nature of the strategy. The remaining candidate
directions:

- Tune the entry threshold per horizon: at h=3 (AUC 0.58, see horizon
  sweep above) the strategy fires too few trades to accumulate alpha —
  try threshold 0.4 / 0.45 / 0.5 and see if the high-AUC model is
  harvestable with looser selection.
- Per-event labels (post-earnings drift, news-burst reaction window)
  instead of the calendar-bar triple-barrier.
- Intraday horizons (1-hour, 4-hour) where sentiment may matter more.
- Meta-labeling (Lopez de Prado): a tiny primary detector + an ML
  secondary that predicts the primary's reliability for sizing.

None are started. The current v0.2.0+ infrastructure (data layer with
OHLCV + earnings + news + multi-universe, walk-forward backtest with
realistic costs, registry guard, paper book, production deployment) is
the platform on which any of these would sit.

---

## Phase 10 — GPU model zoo, market-neutral long/short, adaptive advice (v0.5.0)

Infrastructure pivot per the prior verdict ("change the problem, not the algorithm").
All results below are walk-forward OOS with realistic fees + slippage; none clears the
guard yet, but each is an honest data point on the new platform.

**GPU zoo on US daily long-only (binary triple-barrier, mega watchlist):**
- LightGBM baseline: AUC ≈ 0.515, Sharpe ≈ 0.398.
- PatchTST (transformer, pure torch, GPU, 12 epochs): **AUC 0.524, Sharpe 0.492** — beats
  the LGBM baseline on both, still below buy & hold (Sharpe 1.149). Not promoted.
- Confirms the 9-phase verdict holds for deeper architectures: better models, no edge over
  passive equity on this problem.

**Crypto (long-only triple-barrier, BTC/ETH/SOL, BTC as regime proxy):**
- OOS **AUC 0.5397** — the highest single-name daily AUC in the project, above the ~0.51
  US-equity ceiling. Strategy Sharpe 0.257 vs crypto B&H 0.918 (bull-market dominated),
  but max DD -33% vs -88%. Directional signal is real and stronger than US equities.

**Market-neutral long/short (cross-sectional, 274 tickers, dollar-neutral, H=5):**
- Base 22 single-name features + LGBM ranker: rank-IC ≈ 0, Sharpe -0.04 → gate refuses.
- **+ within-date cross-sectional features (`*_xs`)**: rank-IC still ≈ 0 but Sharpe
  **-0.04 → +0.35** (total +28.7%), DSR 0.29, p 0.17. A real improvement from the relative
  features + inverse-vol construction; not yet statistically significant (DSR < 0.95).

**Verdict:** the platform now supports the recommended pivots end-to-end — GPU zoo
(LSTM/PatchTST/ensemble + Optuna HPO, dual-GPU, nightly retrain), market-neutral long/short
(rank-IC OOF, dollar-neutral backtest, deflated-Sharpe gate), crypto, adaptive SL/TP,
calibration, and meta-labeling. No combination clears the (adapted) guard yet; the deep
rankers + HPO on the full long/short panel and the crypto zoo are the open GPU search the
nightly/weekend jobs now automate.

---

## Phase 11 — Microstructure feature family (v0.5.1)

A genuinely new feature *source* (not the OHLCV momentum/macro hunt the guardrails close):
liquidity / effective-spread / intraday-position proxies, all causal from daily OHLCV —
`clv` (close location value), `gap_open`, `hl_range`, `parkinson_10`, `amihud_20`
(illiquidity), `roll_spread_20` (Roll effective spread). Added as an optional feature group
(`feature_columns(micro=True)`, `berich backtest --with-micro`) and a toggleable group in the
HPO feature search.

**Result (US mega watchlist, walk-forward OOS AUC):**
- base 22 features: AUC 0.5143
- base + microstructure (28): AUC 0.5251
- honest AUC-objective HPO (params + feature-family search): **AUC 0.5364**, and the search
  **independently selected the microstructure group** (alongside momentum + calendar). The
  deployed `lgbm-hpo` model uses all six microstructure features.

This is the largest legitimate discrimination gain from features since the early phases, from
a principled new source the honest search chose on its own. It still does **not** beat buy &
hold on Sharpe (the guard continues to refuse promotion of an edge claim) — but +0.022 AUC
from a new feature family is the most encouraging feature-side signal to date, and motivates
the next data sources (point-in-time fundamentals; intraday/microstructure from finer bars).

---

## Phase 11b — Point-in-time fundamentals (v0.5.2) — inconclusive (data depth)

Added a correct, no-lookahead fundamentals pipeline: `data/fundamentals.py` caches quarterly
statement line-items (revenue, net income, assets, equity, debt) per ticker; `features/
fundamental_features.py` derives net margin, ROE, debt/equity, and YoY revenue growth, each
made visible only after a conservative ~75-day filing lag (period-end + lag, forward-filled)
so a bar never sees a quarter before it was filed. Threaded as an optional family
(`feature_columns(fundamentals=True)`, `berich backtest --with-fundamentals`).

**Result: inconclusive — yfinance serves only ~5–7 recent quarters.** Over the 2010–2026
walk-forward the features are neutral-0 for ~90% of the panel, so OOS AUC is unchanged
(0.5143 -> 0.5143). The lever cannot be validated historically with this data depth. The
pipeline is causally correct and forward-looking: it will enrich live signals and future
backtests as history accrues, and is ready to plug into a deeper (paid) fundamentals source.
Not promoted into the active model (no evidence of benefit). Causality is unit-tested.

---

## Phase 12 — per-asset tournament + directional shorts (architecture, no edge claim)

This phase changes the *problem framing*, not by claiming an edge but by giving each asset
its own uniquely-trained, uniquely-optimized model and adding a directional short side. It
is shipped as **infrastructure under the same guard rule** — a per-asset side is only
promoted if it clears that asset's own benchmark out-of-sample.

**What changed.**

- **Per-ticker, not per-class.** Each configured asset (US + FR + forex + crypto +
  commodities) gets its own Optuna study per `(ticker, model, side)` — hyperparameters
  *and* feature-group toggles (incl. news/FinBERT) are searched per asset, so a feature
  family is kept only where it helps that asset. A **tournament** (`training/tournament.py`)
  trains LightGBM / LSTM / PatchTST / TFT per asset and the registry keeps the walk-forward
  winner. Artifacts live at `data/models/tickers/<TICKER>/<side>/`.
- **Directional shorts.** The triple-barrier label and the backtest engine are now
  direction-aware (`direction="short"` mirrors the barriers: profit below entry, stop above;
  realized return positive when a short wins). Each asset can carry a long model and a short
  model; the signal service picks LONG / SHORT / NEUTRAL by best expectancy with mirrored
  stop/target and absolute-distance sizing.
- **Per-asset honest guard.** Long side must beat *that asset's* buy & hold OOS; short side,
  having no buy-&-hold benchmark, must show a positive, statistically significant Sharpe vs
  cash (deflated Sharpe ≥ 0.95, p < 0.05). Assets failing the guard stay **advisory-only**
  (no `active.json`) and serve the fallback model behind the experimental banner — exactly as
  Phases 0–11 predict for a hard, efficient universe.

**Why this is not an edge claim.** Each asset has ~4 100 daily bars → ~400 labeled windows;
optimizing four model families per asset is a multiple-testing risk. The per-asset guard +
deflated Sharpe + strong regularization priors + AUC-objective HPO are the defense, and the
expected outcome — consistent with all prior phases — is that many assets remain
advisory-only. The value delivered is the **machinery** (unique per-asset optimization,
directional shorts, the dashboard showing direction / per-side P(win) / mirrored TP-SL / a
forward trend overlay), with the guard still refusing to overclaim.

**Cadence.** A heavy weekend deep sweep (`ticker_initial_sweep_job`, also runnable via
`berich train --all-tickers --tournament`) does the full per-asset HPO + tournament; a light
nightly job (`ticker_nightly_refresh_job`) tops up studies and re-fits the current winners.
The dual GPU is exploited via the existing `run_on_gpus` pool for the deep models.

## Bake-off June 2026 — hardened-harness re-validation (PEAD & market-neutral buried)

After the Phase-1-3 rigor overhaul (real-n_trials Deflated Sharpe, MIN_TRADES floor,
sweep-level Benjamini-Hochberg FDR, volume-proportional slippage, significance floor on
LONGS), the promoted set settled at **47 honest survivors** out of 600 (ticker × side ×
strategy) combos: forex 26 (13L/13S), crypto shorts 5/5, commodity shorts 5/5, plus a
handful of US/FR stock models. Median Sharpe 0.70, median DSR 0.97, FDR-stable.

A pre-registered bake-off (rule fixed BEFORE looking: pass the hardened gate → own book;
fail → buried, no re-litigation) re-tested the two remaining edge directions:

- **PEAD (read-only, universe=all, 274 tickers, 6 175 events)** —
  5d: AUC 0.535, Sharpe 0.827 vs window-B&H 1.006 (1 165 trades);
  20d: AUC 0.546, Sharpe 1.097 vs window-B&H 1.309 (1 437 trades).
  The drift exists (AUC > 0.5 on a large sample) but **holding through the earnings windows
  beat trading them** on both horizons. Same conclusion as Phases 7-9, now confirmed with
  realistic slippage and a big event sample. **Buried.**

- **Market-neutral cross-sectional (universe=all, 274 names, 988k panel rows)** —
  lgbm ranker: rank-IC 0.0007 (t = 0.26), Sharpe 0.08, DSR 0.10, p = 0.41;
  patchtst: rank-IC −0.0027, Sharpe −0.61. (lstm/tft OOM'd on the shared GPU; with zero IC
  on both evaluated rankers and Phase 10's prior negative, the verdict stands.)
  **No cross-sectional ranking power on this universe. Buried.**

**Guardrail (add to the two existing ones): do not re-litigate PEAD or the daily
cross-sectional market-neutral ranker.** Both are settled under the hardened harness.
The open direction is the **forward test of the 47 survivors** (systematic forex +
crypto/commodity shorts): concentrate capital per (class × side) segment once ~30 closed
paper trades accumulate with positive net realized expectancy; cut the segments that don't.

---

## Post-forward-test backlog — the US/FR stock gate (June 2026, deferred)

**Symptom (measured 2026-06-09, 125-asset universe, 55 promoted).** The promoted set is
forex-skewed: forex 30, crypto 6, US 4, FR 4, commodities 2. Cause is the **long gate's
`beats_buy_hold` requirement** colliding with how trending each class is. Average buy-&-hold
Sharpe a long model must beat, by class:

| Class | avg B&H Sharpe to beat | promoted | trained-but-advisory |
|---|---|---|---|
| forex | **+0.11** (flat) | 30 | 18 |
| commodities | +0.23 | 2 | 32 |
| crypto | +0.57 | 6 | 30 |
| FR stocks | +0.50 | 4 | 48 |
| US stocks | **+0.72** (strong trend) | 4 | 76 |

A daily-bar ML model rarely beats a strongly-trending stock's own buy & hold, so US/FR longs
almost never promote, while forex clears a ~flat bar trivially. The forex dominance is the gate
being honest — but it doubles as a warning (forex "wins" beat a near-zero benchmark = weak edge),
and it means the honest product for trending stocks is **buy & hold itself**, not a model.

**User decision (2026-06-09): fix this AFTER the forward test, not now** (changing the gate
re-shapes the promoted set and resets the ~30-trade counter). Candidate fixes to **pre-register
then evaluate** when the forward test closes:

1. **Drawdown-aware promotion (Calmar / MAR).** Promote a long if it beats B&H on return ÷
   max-drawdown rather than raw Sharpe — directly rewards the loss-control the tool is built for.
   A model capturing 80% of the uptrend at half the drawdown is valuable even if raw Sharpe < B&H.
2. **Risk-adjusted excess return.** Credit a model that matches B&H return at materially lower
   max-DD or time-in-market (lower exposure → capital freed for other segments).
3. **Neutralize the flat-benchmark advantage.** Rather than only lowering the stock bar, *raise*
   the forex bar: require forex longs to clear a higher deflated-Sharpe-vs-cash significance, so a
   ~flat B&H can't be cleared by a marginal model.
4. **Buy-&-hold stock sleeve.** A separate non-ML "hold" allocation for US/FR (and an index),
   tracked apart from the ML book — the honest baseline product where B&H is the edge.

**Guardrail:** any of these re-shapes the promoted set → must re-run the full sweep + FDR and
restart the forward-test trade count. Fix the new gate rule BEFORE running it (do not tune the
gate to make the assets we *want* pass).
