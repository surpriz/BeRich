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

from berich.datasets.assemble import build_dataset
from berich.features.build import FEATURE_COLUMNS, MARKET_TICKER, build_features
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


def _train_model(store: OhlcvStore, config: Config, label_cfg: LabelConfig) -> Model:
    dataset = build_dataset(store, config.watchlist, label_cfg)
    return LGBMModel().fit(dataset.x, dataset.y, sample_weight=dataset.weight)


def _resolve_model(store: OhlcvStore, config: Config, label_cfg: LabelConfig) -> Model:
    """Use the promoted registry model if one exists; otherwise train the baseline.

    This is the GPU handoff point: once an LSTM/TFT artifact is trained, saved, and
    promoted on the GPU box and synced here, serving picks it up automatically with no
    code change. Until then we fall back to training the LightGBM baseline inline.
    """
    active = load_active(config.models_dir)
    if active is not None:
        model, meta = active
        if meta.feature_columns != FEATURE_COLUMNS:
            logger.warning(
                "active model '%s' feature columns differ from current; retraining baseline",
                meta.name,
            )
        else:
            logger.info("serving promoted model '%s' (%s)", meta.name, meta.framework)
            return model
    return _train_model(store, config, label_cfg)


def generate_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Serve the promoted (or freshly trained) model and emit one signal per ticker."""
    label_cfg = LabelConfig(**config.labeling.model_dump())
    model = _resolve_model(store, config, label_cfg)
    market = store.load(MARKET_TICKER)

    signals: list[Signal] = []
    for ticker in config.watchlist:
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        signal = _signal_for_ticker(ticker, df, model, config, label_cfg, market=market)
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
) -> Signal | None:
    feats = build_features(df, market=market).dropna()
    if feats.empty:
        return None
    last_date = feats.index[-1]
    x = feats.loc[[last_date], FEATURE_COLUMNS]
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
