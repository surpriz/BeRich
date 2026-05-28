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
from berich.datasets.assemble import build_dataset
from berich.features.build import (
    MARKET_TICKER,
    build_features,
    feature_columns,
)
from berich.features.earnings_features import EARNINGS_FEATURE_COLUMNS
from berich.features.indicators import atr
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
    """Return an EarningsStore iff the cache directory has at least one non-empty file.

    Backward-compat with v0.1.0: if the user hasn't run ``berich data`` since
    Phase 5a landed, the directory is missing or empty and we silently fall
    back to the 22-feature mode used by every previously-trained model.
    """
    store = EarningsStore(config.earnings_dir)
    return store if store.has_any_data() else None


def _train_model(
    store: OhlcvStore,
    config: Config,
    label_cfg: LabelConfig,
    *,
    earnings_store: EarningsStore | None,
) -> Model:
    dataset = build_dataset(store, config.watchlist, label_cfg, earnings_store=earnings_store)
    return LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)


def _resolve_model(
    store: OhlcvStore,
    config: Config,
    label_cfg: LabelConfig,
) -> tuple[Model, bool]:
    """Return ``(model, with_earnings)`` — what to serve + whether it needs earnings.

    Decision tree:
      - promoted model in the registry → trust its ``feature_columns`` to know
        whether it was trained with earnings, use it as-is.
      - no promoted model → train a fresh LightGBM inline. Use earnings only
        if the local cache actually contains data (v0.1.0 boxes that never
        ran the new ``berich data`` get the 22-feature fallback for free).
    """
    active = load_active(config.models_dir)
    if active is not None:
        model, meta = active
        meta_has_earnings = all(c in meta.feature_columns for c in EARNINGS_FEATURE_COLUMNS)
        expected = feature_columns(earnings=meta_has_earnings)
        if meta.feature_columns != expected:
            logger.warning(
                "active model '%s' feature columns differ from current; retraining baseline",
                meta.name,
            )
        else:
            logger.info(
                "serving promoted model '%s' (%s, earnings=%s)",
                meta.name,
                meta.framework,
                meta_has_earnings,
            )
            return model, meta_has_earnings

    earnings_store = _earnings_store_if_available(config)
    with_earnings = earnings_store is not None
    model = _train_model(store, config, label_cfg, earnings_store=earnings_store)
    return model, with_earnings


def generate_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Serve the promoted (or freshly trained) model and emit one signal per ticker."""
    label_cfg = LabelConfig(**config.labeling.model_dump())
    model, with_earnings = _resolve_model(store, config, label_cfg)
    market = store.load(MARKET_TICKER)
    earnings_store = _earnings_store_if_available(config) if with_earnings else None

    signals: list[Signal] = []
    for ticker in config.watchlist:
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        earnings_df = earnings_store.load(ticker) if earnings_store is not None else None
        signal = _signal_for_ticker(
            ticker,
            df,
            model,
            config,
            label_cfg,
            market=market,
            earnings=earnings_df,
            with_earnings=with_earnings,
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
    with_earnings: bool = False,
) -> Signal | None:
    # When ``with_earnings`` is True we always pass an earnings frame (an empty
    # one falls back to neutral defaults inside build_earnings_features), so the
    # column shape stays consistent across all tickers in the same call.
    earnings_arg = earnings if with_earnings else None
    if with_earnings and earnings_arg is None:
        earnings_arg = pd.DataFrame()
    feats = build_features(df, market=market, earnings=earnings_arg).dropna()
    if feats.empty:
        return None
    last_date = feats.index[-1]
    cols = feature_columns(earnings=with_earnings)
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
