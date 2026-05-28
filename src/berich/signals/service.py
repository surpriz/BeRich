"""Generate today's swing-trade advice for the watchlist.

The model is trained on all labeled history, then applied to each ticker's most
recent (fully-formed, causal) feature row to estimate P(win). The probability is
turned into a BUY / NEUTRAL / SELL call, and for BUY calls an ATR stop / target and
a risk-based position size are attached. This is the "conseil" surface: where to
enter, where the stop goes, and how big the position should be.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.data.earnings import EarningsStore
from berich.data.news import NewsStore
from berich.datasets.assemble import build_dataset
from berich.features.build import (
    MARKET_TICKER,
    build_features,
    feature_columns,
)
from berich.features.earnings_features import EARNINGS_FEATURE_COLUMNS
from berich.features.indicators import atr
from berich.features.news_features import NEWS_FEATURE_COLUMNS
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, load_active

if TYPE_CHECKING:
    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.models.base import Model

logger = logging.getLogger(__name__)

BUY = "BUY"
SELL = "SELL"
NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """One ticker's advice for a given date."""

    date: pd.Timestamp
    ticker: str
    signal: str
    proba: float
    entry: float
    stop_loss: float
    take_profit: float
    size_shares: int
    notional: float

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        # ``self.date`` is set by the producer and never NaT in practice; ignore
        # the stub's NaT union here rather than litter the call sites with casts.
        row["date"] = pd.Timestamp(self.date).date().isoformat()  # ty: ignore[unresolved-attribute]
        return row


def _classify(proba: float, config: Config) -> str:
    if proba >= config.signals.buy_threshold:
        return BUY
    if proba <= config.signals.sell_threshold:
        return SELL
    return NEUTRAL


def _size_position(entry: float, stop: float, config: Config) -> tuple[int, float]:
    """Risk-based sizing: risk at most ``risk_pct`` of capital to the stop."""
    stop_distance = entry - stop
    if stop_distance <= 0:
        return 0, 0.0
    risk_amount = config.signals.capital * config.signals.risk_pct
    shares = math.floor(risk_amount / stop_distance)
    return shares, shares * entry


def _earnings_store_if_available(config: Config) -> EarningsStore | None:
    """Return an EarningsStore iff the cache directory has at least one non-empty file."""
    store = EarningsStore(config.earnings_dir)
    return store if store.has_any_data() else None


def _news_store_if_available(config: Config) -> NewsStore | None:
    """Return a NewsStore iff the cache directory has at least one non-empty file."""
    store = NewsStore(config.news_dir)
    return store if store.has_any_data() else None


def _train_model(
    store: OhlcvStore,
    config: Config,
    label_cfg: LabelConfig,
    *,
    earnings_store: EarningsStore | None,
    news_store: NewsStore | None,
) -> Model:
    dataset = build_dataset(
        store,
        config.watchlist,
        label_cfg,
        earnings_store=earnings_store,
        news_store=news_store,
    )
    return LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)


def _resolve_model(
    store: OhlcvStore,
    config: Config,
    label_cfg: LabelConfig,
) -> tuple[Model, bool, bool]:
    """Return ``(model, with_earnings, with_news)`` — model + which modes it needs.

    Decision tree:
      - promoted model in the registry → trust its ``feature_columns`` to know
        which feature families it was trained with. If a model expects news
        but the cache is stale (empty), log a warning and re-train inline
        without news so we serve something usable rather than crash.
      - no promoted model → train a fresh LightGBM inline using whichever
        feature families have data on disk (v0.1.0 boxes that never ran
        the new ``berich data`` keep the 22-feature fallback for free).
    """
    active = load_active(config.models_dir)
    if active is not None:
        model, meta = active
        meta_has_earnings = all(c in meta.feature_columns for c in EARNINGS_FEATURE_COLUMNS)
        meta_has_news = all(c in meta.feature_columns for c in NEWS_FEATURE_COLUMNS)
        expected = feature_columns(earnings=meta_has_earnings, news=meta_has_news)
        if meta.feature_columns != expected:
            logger.warning(
                "active model '%s' feature columns differ from current; retraining baseline",
                meta.name,
            )
        elif meta_has_news and _news_store_if_available(config) is None:
            # Promoted model expects news features but the cache is empty;
            # rather than feed it all-zero columns and silently degrade the
            # signal, drop back to a freshly-trained earnings-or-base model.
            logger.warning(
                "active model '%s' expects news features but data/news/ is empty; "
                "falling back to a non-news baseline",
                meta.name,
            )
        else:
            logger.info(
                "serving promoted model '%s' (%s, earnings=%s, news=%s)",
                meta.name,
                meta.framework,
                meta_has_earnings,
                meta_has_news,
            )
            return model, meta_has_earnings, meta_has_news

    earnings_store = _earnings_store_if_available(config)
    news_store = _news_store_if_available(config)
    with_earnings = earnings_store is not None
    with_news = news_store is not None
    model = _train_model(
        store,
        config,
        label_cfg,
        earnings_store=earnings_store,
        news_store=news_store,
    )
    return model, with_earnings, with_news


def generate_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Serve the promoted (or freshly trained) model and emit one signal per ticker."""
    label_cfg = LabelConfig(**config.labeling.model_dump())
    model, with_earnings, with_news = _resolve_model(store, config, label_cfg)
    market = store.load(MARKET_TICKER)
    earnings_store = _earnings_store_if_available(config) if with_earnings else None
    news_store = _news_store_if_available(config) if with_news else None

    signals: list[Signal] = []
    for ticker in config.watchlist:
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        earnings_df = earnings_store.load(ticker) if earnings_store is not None else None
        news_df = news_store.load(ticker) if news_store is not None else None
        signal = _signal_for_ticker(
            ticker,
            df,
            model,
            config,
            label_cfg,
            market=market,
            earnings=earnings_df,
            news=news_df,
            with_earnings=with_earnings,
            with_news=with_news,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _signal_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    model: Model,
    config: Config,
    label_cfg: LabelConfig,
    *,
    market: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    with_earnings: bool = False,
    with_news: bool = False,
) -> Signal | None:
    # When a feature family is on we always pass an (optionally empty) frame so the
    # column shape stays consistent across tickers — empty frames get the neutral
    # defaults from the feature builder.
    earnings_arg = earnings if with_earnings else None
    if with_earnings and earnings_arg is None:
        earnings_arg = pd.DataFrame()
    news_arg = news if with_news else None
    if with_news and news_arg is None:
        news_arg = pd.DataFrame()
    feats = build_features(df, market=market, earnings=earnings_arg, news=news_arg).dropna()
    if feats.empty:
        return None
    last_date = feats.index[-1]
    cols = feature_columns(earnings=with_earnings, news=with_news)
    x = feats.loc[[last_date], cols]
    proba = float(model.predict_proba(x)[0])

    a = float(atr(df["high"], df["low"], df["close"], label_cfg.atr_window).loc[last_date])
    entry = float(df["close"].loc[last_date])
    if np.isnan(a):
        return None
    stop = entry - label_cfg.stop_loss_atr * a
    target = entry + label_cfg.take_profit_atr * a

    decision = _classify(proba, config)
    shares, notional = _size_position(entry, stop, config) if decision == BUY else (0, 0.0)

    return Signal(
        date=last_date,
        ticker=ticker,
        signal=decision,
        proba=round(proba, 4),
        entry=round(entry, 2),
        stop_loss=round(stop, 2),
        take_profit=round(target, 2),
        size_shares=shares,
        notional=round(notional, 2),
    )
