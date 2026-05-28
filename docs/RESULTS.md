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

## Open project (not started)

The next plausible source of lift is **qualitatively different** exogenous
information — text rather than tabular:

- News sentiment via FinBERT on per-ticker headlines.
- Earnings *transcripts* (not just surprises) — language signal around the
  call vs the dry beat/miss number.
- Sector flows / 13F changes / insider filings.

These are larger projects than the OHLCV/macro/earnings tabular features
explored so far. They live in Phase 5b+ and are not started.
