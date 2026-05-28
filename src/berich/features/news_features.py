"""News / sentiment features (Phase 5b).

For each bar date ``t`` we summarize the cached news rows that were published
strictly before ``t`` (after-close rule, same as earnings — yesterday's news
is in, today's isn't). The features are:

- ``news_count_5d``        articles in the 5-day window ending at ``t-1``
- ``news_count_20d``       articles in the 20-day window
- ``sentiment_mean_5d``    recency-weighted mean of FinBERT ``pos - neg``
- ``sentiment_std_5d``     std of FinBERT ``pos - neg`` in the 5-day window
- ``sentiment_extreme_5d`` count of articles with ``|finbert_score| > 0.8``
- ``sentiment_av_5d``      mean of Alpha Vantage's own ``overall_sentiment_score``
- ``sentiment_price_div``  1 iff sentiment positive AND 5d return negative
                           (or symmetric: negative sentiment + positive return) —
                           a "news already priced / fading" divergence flag

Tickers without any cached news (or only unscored rows) get neutral defaults
so they survive the join + dropna step.

Causality: we use ``time_published < t`` strictly, where ``t`` is the bar's
midnight. That puts last night's after-close news in scope for tomorrow's
open and keeps same-day news out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_FEATURE_COLUMNS: list[str] = [
    "news_count_5d",
    "news_count_20d",
    "sentiment_mean_5d",
    "sentiment_std_5d",
    "sentiment_extreme_5d",
    "sentiment_av_5d",
    "sentiment_price_div",
]

WINDOW_5D = 5
WINDOW_20D = 20
RECENCY_HALF_LIFE_DAYS = 2.0
EXTREME_ABS_THRESHOLD = 0.8

# sentiment_price_div thresholds — a divergence flag fires only when both the
# sentiment signal and the recent return are non-trivially large in opposite
# directions. The 0.1/0.02 cutoffs are conservative: FinBERT pos-neg of 0.1
# means a moderately positive net signal, 2% 5d return is roughly one daily
# vol on this universe.
DIVERGENCE_SENTIMENT_THRESHOLD = 0.1
DIVERGENCE_RETURN_THRESHOLD = 0.02


def _neutral_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Defaults when the ticker has no news cache or no FinBERT scores yet."""
    return pd.DataFrame(
        dict.fromkeys(NEWS_FEATURE_COLUMNS, 0.0),
        index=index,
        columns=pd.Index(NEWS_FEATURE_COLUMNS),
    )


def _select_window(
    news_times: np.ndarray,
    news_values: np.ndarray,
    bar_date: np.datetime64,
    window_days: int,
) -> np.ndarray:
    """Articles published in ``[bar_date - window, bar_date)`` — strict past."""
    start = bar_date - np.timedelta64(window_days, "D")
    mask = (news_times >= start) & (news_times < bar_date)
    return news_values[mask]


def build_news_features(  # noqa: PLR0915 — straight-line aggregation, splitting hurts readability
    target_index: pd.DatetimeIndex,
    news: pd.DataFrame | None,
    *,
    close: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute the 7 sentiment features for every bar in ``target_index``.

    ``news`` is the per-ticker frame from :class:`NewsStore`; rows without a
    ``finbert_score`` are still counted for ``news_count_*`` but contribute 0
    to sentiment aggregates (they would otherwise drag the mean toward NaN).
    ``close`` is optional — only the divergence feature consumes it; when it's
    missing we set that feature to 0.
    """
    if news is None or news.empty:
        return _neutral_frame(target_index)

    news_sorted = news.dropna(subset=["time_published"]).copy()
    if news_sorted.empty:
        return _neutral_frame(target_index)
    news_sorted = news_sorted.sort_values("time_published")
    news_sorted["time_published"] = pd.to_datetime(news_sorted["time_published"])

    news_times = news_sorted["time_published"].to_numpy("datetime64[ns]")
    finbert = news_sorted["finbert_score"].astype(float).fillna(0.0).to_numpy()
    av = news_sorted["overall_sentiment_score"].astype(float).fillna(0.0).to_numpy()

    bar_dates = target_index.to_numpy("datetime64[ns]")
    n = len(bar_dates)
    out = _neutral_frame(target_index).copy()

    count_5d = np.zeros(n, dtype=float)
    count_20d = np.zeros(n, dtype=float)
    sent_mean_5d = np.zeros(n, dtype=float)
    sent_std_5d = np.zeros(n, dtype=float)
    sent_extreme_5d = np.zeros(n, dtype=float)
    sent_av_5d = np.zeros(n, dtype=float)

    half_life_ns = np.timedelta64(int(RECENCY_HALF_LIFE_DAYS * 86_400), "s")
    half_life_seconds = half_life_ns / np.timedelta64(1, "s")

    for i, t in enumerate(bar_dates):
        # 20d window — count only
        mask20 = (news_times >= t - np.timedelta64(WINDOW_20D, "D")) & (news_times < t)
        count_20d[i] = int(mask20.sum())

        mask5 = (news_times >= t - np.timedelta64(WINDOW_5D, "D")) & (news_times < t)
        if not mask5.any():
            continue
        count_5d[i] = int(mask5.sum())
        finbert_5d = finbert[mask5]
        av_5d = av[mask5]
        times_5d = news_times[mask5]

        # Recency-weighted mean: exponentially weight by (t - news_time).
        deltas_sec = (t - times_5d).astype("timedelta64[s]").astype(float)
        weights = np.exp(-deltas_sec / half_life_seconds)
        weight_sum = weights.sum()
        if weight_sum > 0:
            sent_mean_5d[i] = float(np.sum(weights * finbert_5d) / weight_sum)
        sent_std_5d[i] = float(np.std(finbert_5d)) if len(finbert_5d) > 1 else 0.0
        sent_extreme_5d[i] = float(np.sum(np.abs(finbert_5d) > EXTREME_ABS_THRESHOLD))
        sent_av_5d[i] = float(np.mean(av_5d))

    out["news_count_5d"] = count_5d
    out["news_count_20d"] = count_20d
    out["sentiment_mean_5d"] = sent_mean_5d
    out["sentiment_std_5d"] = sent_std_5d
    out["sentiment_extreme_5d"] = sent_extreme_5d
    out["sentiment_av_5d"] = sent_av_5d

    if close is not None and not close.empty:
        close_reindexed = close.reindex(target_index)
        ret_5d = close_reindexed.pct_change(WINDOW_5D)
        sentiment = pd.Series(sent_mean_5d, index=target_index)
        positive_divergence = (sentiment > DIVERGENCE_SENTIMENT_THRESHOLD) & (
            ret_5d < -DIVERGENCE_RETURN_THRESHOLD
        )
        negative_divergence = (sentiment < -DIVERGENCE_SENTIMENT_THRESHOLD) & (
            ret_5d > DIVERGENCE_RETURN_THRESHOLD
        )
        # ``fillna(False)`` keeps the early bars (NaN return) at 0 rather than NaN.
        divergence = (positive_divergence | negative_divergence).fillna(value=False)
        out["sentiment_price_div"] = divergence.astype(float).to_numpy()
    else:
        out["sentiment_price_div"] = 0.0

    return out
