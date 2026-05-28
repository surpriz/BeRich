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

## Final verdict — v0.4.0

8 phases of exploration over OHLCV, macro cross-asset, earnings
surprises, FinBERT news sentiment, mid/small-cap universes, short
horizons, post-earnings drift, and risk management. **No combination
tested publicly beats buy & hold in walk-forward with realistic fees
and slippage on the US daily long-only universe.**

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
