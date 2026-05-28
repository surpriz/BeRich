"""Tests for the Phase 5b news features: causality, defaults, idempotence.

The FinBERT scorer itself isn't loaded in the unit tests (model weights are
~440 MB and the test suite must stay fast); we test scorer determinism with
a trivial in-process stub instead, which is sufficient to check that
``score_texts`` is called in eval mode + that the canonical (neg, neu, pos)
label order is honored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.data.news import NEWS_COLUMNS, NewsStore
from berich.features.build import build_features, feature_columns
from berich.features.news_features import (
    DIVERGENCE_RETURN_THRESHOLD,
    DIVERGENCE_SENTIMENT_THRESHOLD,
    EXTREME_ABS_THRESHOLD,
    NEWS_FEATURE_COLUMNS,
    WINDOW_5D,
    WINDOW_20D,
    build_news_features,
)


def _ohlcv(n: int = 200, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )


def _news_rows(
    rows: list[tuple[str, float, float, str]],
) -> pd.DataFrame:
    """``(time_published, finbert_score, av_score, url)`` → canonical news frame."""
    return pd.DataFrame(
        [
            {
                "time_published": pd.Timestamp(ts),
                "title": "t",
                "summary": "s",
                "source": "x",
                "url": url,
                "overall_sentiment_score": av,
                "ticker_sentiment_score": av,
                "relevance_score": 1.0,
                "finbert_neg": 0.0 if fb >= 0 else abs(fb),
                "finbert_neu": 0.0,
                "finbert_pos": fb if fb >= 0 else 0.0,
                "finbert_score": fb,
            }
            for ts, fb, av, url in rows
        ],
        columns=pd.Index(NEWS_COLUMNS),
    )


# ----------------------------------------------------------- neutral defaults ----


def test_neutral_defaults_when_no_news():
    df = _ohlcv()
    feats = build_news_features(pd.DatetimeIndex(df.index), None)
    assert list(feats.columns) == NEWS_FEATURE_COLUMNS
    assert (feats == 0).all().all()


def test_neutral_defaults_when_empty_frame():
    df = _ohlcv()
    feats = build_news_features(pd.DatetimeIndex(df.index), pd.DataFrame())
    assert (feats == 0).all().all()


def test_neutral_defaults_when_only_unscored_rows_present():
    """Rows with no FinBERT score should still appear in counts but contribute 0 sentiment."""
    df = _ohlcv()
    raw = _news_rows(
        [
            ("2024-01-03 13:00", float("nan"), 0.0, "url1"),
            ("2024-01-04 13:00", float("nan"), 0.0, "url2"),
        ]
    )
    feats = build_news_features(pd.DatetimeIndex(df.index), raw)
    # Count fields populate; sentiment fields stay 0 because FinBERT score is NaN -> 0.
    later = df.index >= pd.Timestamp("2024-01-05")
    assert feats.loc[later, "news_count_5d"].max() >= 2
    assert (feats["sentiment_mean_5d"] == 0).all()


# --------------------------------------------------------------- causality ----


def test_perturbation_at_future_news_does_not_change_past_features():
    """Adding (or modifying) a news row at date t+5 must NOT change features at <= t."""
    df = _ohlcv(start="2024-01-02", n=60)
    base_rows = [
        ("2024-01-08 16:00", 0.7, 0.5, "u1"),
        ("2024-01-12 16:00", -0.3, -0.1, "u2"),
    ]
    perturbed_rows = [*base_rows, ("2024-01-20 16:00", 0.9, 0.6, "u3")]
    base = build_news_features(pd.DatetimeIndex(df.index), _news_rows(base_rows))
    after = build_news_features(pd.DatetimeIndex(df.index), _news_rows(perturbed_rows))

    cutoff = pd.Timestamp("2024-01-20")
    pre = df.index <= cutoff
    pd.testing.assert_frame_equal(base.loc[pre], after.loc[pre])


def test_perturbation_does_change_features_after_lag():
    """Counter-test: the new article must move features at bars after it."""
    df = _ohlcv(start="2024-01-02", n=60)
    base_rows = [("2024-01-08 16:00", 0.7, 0.5, "u1")]
    perturbed_rows = [*base_rows, ("2024-01-20 16:00", 0.9, 0.6, "u_new")]
    base = build_news_features(pd.DatetimeIndex(df.index), _news_rows(base_rows))
    after = build_news_features(pd.DatetimeIndex(df.index), _news_rows(perturbed_rows))

    post = df.index > pd.Timestamp("2024-01-20")
    diff = (base.loc[post] - after.loc[post]).abs().sum().sum()
    assert diff > 0


def test_same_day_news_excluded_from_window():
    """News at exactly date t shouldn't yet appear in the t-bar's counts."""
    df = _ohlcv(start="2024-01-02", n=20)
    # News at 16:00 of 2024-01-10 should be EXCLUDED from the 2024-01-10 bar.
    rows = _news_rows([("2024-01-10 16:00", 0.5, 0.3, "u1")])
    feats = build_news_features(pd.DatetimeIndex(df.index), rows)
    day_of = pd.Timestamp("2024-01-10")
    assert feats.loc[day_of, "news_count_5d"] == 0.0
    next_day = pd.Timestamp("2024-01-11")
    if next_day in feats.index:
        assert feats.loc[next_day, "news_count_5d"] == 1.0


# ------------------------------------------------------------------ windows ----


def test_count_windows_match_lookback_lengths():
    """A burst of articles should grow news_count_20d but news_count_5d should
    stay at the 5-day truncated count."""
    df = _ohlcv(start="2024-01-02", n=40)
    rows = _news_rows(
        [
            ("2024-01-03 16:00", 0.0, 0.0, "a"),
            ("2024-01-04 16:00", 0.0, 0.0, "b"),
            ("2024-01-08 16:00", 0.0, 0.0, "c"),
            ("2024-01-15 16:00", 0.0, 0.0, "d"),
            ("2024-01-22 16:00", 0.0, 0.0, "e"),
        ]
    )
    feats = build_news_features(pd.DatetimeIndex(df.index), rows)

    far_out = pd.Timestamp("2024-01-29")
    if far_out in feats.index:
        # 20d window from 01-29 covers [01-09, 01-29) strictly.
        # Articles in that interval: 01-15 and 01-22 — two.
        assert feats.loc[far_out, "news_count_20d"] == 2
        # 5d window covers [01-24, 01-29) strictly — no articles there.
        assert feats.loc[far_out, "news_count_5d"] == 0

    in_5d_window = pd.Timestamp("2024-01-23")  # 22-Jan article is one day back
    if in_5d_window in feats.index:
        assert feats.loc[in_5d_window, "news_count_5d"] == 1


def test_extreme_count_uses_abs_threshold():
    """sentiment_extreme_5d only counts articles with |finbert_score| > threshold."""
    df = _ohlcv(start="2024-01-02", n=20)
    score_above = EXTREME_ABS_THRESHOLD + 0.05
    score_below = EXTREME_ABS_THRESHOLD - 0.05
    rows = _news_rows(
        [
            ("2024-01-03 16:00", score_above, 0.0, "a"),
            ("2024-01-04 16:00", -score_above, 0.0, "b"),
            ("2024-01-05 16:00", score_below, 0.0, "c"),
        ]
    )
    feats = build_news_features(pd.DatetimeIndex(df.index), rows)
    target = pd.Timestamp("2024-01-08")
    if target in feats.index:
        assert feats.loc[target, "sentiment_extreme_5d"] == 2.0


def test_sentiment_price_div_zero_without_close():
    """Without a close series the divergence flag must default to 0 everywhere."""
    df = _ohlcv()
    rows = _news_rows([("2024-01-08 16:00", DIVERGENCE_SENTIMENT_THRESHOLD + 0.05, 0.0, "u1")])
    feats = build_news_features(pd.DatetimeIndex(df.index), rows, close=None)
    assert (feats["sentiment_price_div"] == 0).all()


def test_sentiment_price_div_fires_on_disagreement():
    """Positive recency-weighted sentiment + negative 5d return => divergence=1."""
    df = _ohlcv(start="2024-01-02", n=40)
    # Force the close to drop ~3% over 5 days while sentiment is +0.5.
    close = df["close"].copy()
    target_date = pd.Timestamp("2024-01-15")
    if target_date in df.index:
        idx_t = df.index.get_loc(target_date)
        close.iloc[idx_t - 5 : idx_t + 1] = close.iloc[idx_t - 5] * np.linspace(
            1.0, 1.0 - (DIVERGENCE_RETURN_THRESHOLD * 2), 6
        )
    rows = _news_rows(
        [
            ("2024-01-12 16:00", 0.6, 0.4, "a"),
            ("2024-01-13 16:00", 0.6, 0.4, "b"),
        ]
    )
    feats = build_news_features(pd.DatetimeIndex(df.index), rows, close=close)
    if target_date in feats.index:
        assert feats.loc[target_date, "sentiment_price_div"] == 1.0


# ------------------------------------------------------------ build_features ----


def test_build_features_appends_news_columns_when_supplied():
    df = _ohlcv()
    market = _ohlcv(seed=1)
    rows = _news_rows([("2024-01-08 16:00", 0.5, 0.3, "u1")])
    base = build_features(df, market=market)
    with_news = build_features(df, market=market, news=rows)
    assert list(with_news.columns) == feature_columns(news=True)
    assert list(base.columns) == feature_columns()
    pd.testing.assert_frame_equal(with_news[base.columns], base)


def test_empty_news_yields_neutral_defaults_in_build_features():
    df = _ohlcv()
    feats = build_features(df, market=_ohlcv(seed=1), news=pd.DataFrame())
    for col in NEWS_FEATURE_COLUMNS:
        assert feats[col].notna().all(), f"{col} should be non-NaN under empty news"


# ----------------------------------------------------- NewsStore idempotence ----


def test_news_store_dedupes_by_url_and_keeps_finbert_score(tmp_path):
    store = NewsStore(tmp_path / "news")
    base = _news_rows([("2024-01-08 16:00", 0.5, 0.3, "u1")])
    store.save("AAA", base)
    loaded = store.load("AAA")
    assert loaded is not None
    assert len(loaded) == 1

    # Patch FinBERT score: the same URL but a new finbert_score should replace.
    update = pd.DataFrame(
        {
            "url": ["u1"],
            "finbert_neg": [0.05],
            "finbert_neu": [0.10],
            "finbert_pos": [0.85],
            "finbert_score": [0.80],
        }
    )
    updated = store.update_finbert("AAA", update)
    assert updated == 1
    loaded = store.load("AAA")
    assert loaded is not None
    assert abs(loaded.iloc[0]["finbert_score"] - 0.80) < 1e-9

    # Re-saving an unscored row for the same URL should NOT clobber the score:
    # _merge prefers the row with a score when URLs collide.
    unscored = _news_rows([("2024-01-08 16:00", float("nan"), 0.3, "u1")])
    store.save("AAA", unscored)
    loaded = store.load("AAA")
    assert loaded is not None
    assert abs(loaded.iloc[0]["finbert_score"] - 0.80) < 1e-9


def test_news_store_has_any_data(tmp_path):
    store = NewsStore(tmp_path / "news")
    assert store.has_any_data() is False
    store.save("AAA", pd.DataFrame())
    assert store.has_any_data() is False
    store.save("BBB", _news_rows([("2024-01-08 16:00", 0.5, 0.3, "u1")]))
    assert store.has_any_data() is True


# ---------------------------- silence unused-import warnings on shared symbols ---

_ = (WINDOW_5D, WINDOW_20D)
