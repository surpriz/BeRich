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
    market_reference_for,
)
from berich.features.earnings_features import EARNINGS_FEATURE_COLUMNS
from berich.features.indicators import atr
from berich.features.news_features import NEWS_FEATURE_COLUMNS
from berich.features.volatility import forecast_vol
from berich.labeling.triple_barrier import LabelConfig, adaptive_barriers
from berich.models import LGBMModel, load_active
from berich.models.meta_labeler import PRIMARY_PROBA_COL
from berich.signals.calibration import ProbaCalibrator, load_calibrator

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
    # Enriched advice fields (optional; default to neutral/None for back-compat).
    proba_calibrated: float | None = None
    meta_proba: float | None = None
    acted: bool = True  # False when the meta-label filter vetoed a BUY
    ret_q10: float | None = None
    ret_q50: float | None = None
    ret_q90: float | None = None
    sigma_horizon: float | None = None
    sltp_method: str = "atr_fixed"  # "vol_scaled" | "quantile" | "atr_fixed"

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


def _load_calibrator(config: Config) -> ProbaCalibrator | None:
    """Load the calibrator saved next to the active model, if any."""
    active = load_active(config.models_dir)
    if active is None:
        return None
    return load_calibrator(config.models_dir / active[1].name)


def _load_meta_labeler(config: Config) -> Model | None:
    """Load the promoted meta-labeling model from the ``meta/`` namespace, if enabled."""
    if not config.signals.use_meta_label:
        return None
    active = load_active(config.models_dir / "meta")
    return active[0] if active is not None else None


def generate_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Serve the promoted (or freshly trained) model and emit one signal per ticker."""
    label_cfg = LabelConfig(**config.labeling.model_dump())
    model, with_earnings, with_news = _resolve_model(store, config, label_cfg)
    market = store.load(MARKET_TICKER)
    earnings_store = _earnings_store_if_available(config) if with_earnings else None
    news_store = _news_store_if_available(config) if with_news else None
    calibrator = _load_calibrator(config)
    meta_model = _load_meta_labeler(config)

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
            calibrator=calibrator,
            meta_model=meta_model,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _base_model(store: OhlcvStore, config: Config, label_cfg: LabelConfig) -> Model:
    """A base-22-feature LGBM for scoring non-US assets (no earnings/news caches there).

    Reuses the promoted model only if it is base-22; otherwise trains a quick LGBM on the
    US watchlist. Non-US signals are explicitly advisory/experimental (the UI says so).
    """
    active = load_active(config.models_dir)
    if active is not None and active[1].feature_columns == feature_columns():
        return active[0]
    return _train_model(store, config, label_cfg, earnings_store=None, news_store=None)


def _class_model(config: Config, asset_class: str, fallback: Model) -> Model:
    """Dedicated promoted model for an asset class if present, else the US fallback."""
    active = load_active(config.models_dir_for(asset_class))
    if active is not None and active[1].feature_columns == feature_columns():
        logger.info("serving dedicated %s model '%s'", asset_class, active[1].name)
        return active[0]
    return fallback


def generate_multi_asset_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Advisory signals for the non-US universes (FR stocks, forex, crypto, commodities).

    Uses the **dedicated** per-class model when one has been promoted
    (``data/models/<class>/``), otherwise falls back to a base-22 US model. Each class uses
    its own regime proxy (BTC for crypto, the dollar index for forex, …). Earnings/news are
    off (no caches); the US calibrator/meta filter are not applied to these experimental assets.
    """
    label_cfg = LabelConfig(**config.labeling.model_dump())
    fallback = _base_model(store, config, label_cfg)
    out: list[Signal] = []
    for asset_class in ("fr_stocks", "forex", "crypto", "commodities"):
        tickers = config.universes.get(asset_class)
        if not tickers:
            continue
        model = _class_model(config, asset_class, fallback)
        market = store.load(market_reference_for(asset_class))
        for ticker in tickers:
            df = store.load(ticker)
            if df is None or df.empty:
                continue
            signal = _signal_for_ticker(ticker, df, model, config, label_cfg, market=market)
            if signal is not None:
                out.append(signal)
    return out


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
    calibrator: ProbaCalibrator | None = None,
    meta_model: Model | None = None,
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
    raw_proba = float(model.predict_proba(x)[0])
    # Calibrated proba (if a calibrator was fit) drives the decision + sizing.
    proba_cal = float(calibrator.transform(np.array([raw_proba]))[0]) if calibrator else None
    eff_proba = proba_cal if proba_cal is not None else raw_proba

    a = float(atr(df["high"], df["low"], df["close"], label_cfg.atr_window).loc[last_date])
    entry = float(df["close"].loc[last_date])
    if np.isnan(a):
        return None

    q10 = q50 = q90 = None
    sigma_h: float | None = None
    if config.signals.adaptive_sltp:
        vf = forecast_vol(
            df["close"].loc[:last_date],
            horizon_days=label_cfg.horizon_days,
            method=config.signals.vol_method,
        )
        sigma_h = vf.horizon_sigma
        quantiles = None
        predict_quantiles = getattr(model, "predict_quantiles", None)
        if predict_quantiles is not None:
            q = np.asarray(predict_quantiles(x))[0]
            q10, q90 = float(q[0]), float(q[-1])
            q50 = float(q[len(q) // 2])
            quantiles = (q10, q90)
        stop, target, rationale = adaptive_barriers(entry, a, vf, label_cfg, quantiles=quantiles)
        sltp_method = str(rationale["method"])
    else:
        stop = entry - label_cfg.stop_loss_atr * a
        target = entry + label_cfg.take_profit_atr * a
        sltp_method = "atr_fixed"

    decision = _classify(eff_proba, config)

    # Meta-labeling precision filter: veto a BUY whose meta P(correct) is too low.
    meta_proba: float | None = None
    acted = True
    if meta_model is not None and decision == BUY:
        meta_x = x.copy()
        meta_x[PRIMARY_PROBA_COL] = eff_proba
        meta_proba = float(meta_model.predict_proba(meta_x)[0])
        if meta_proba < config.signals.meta_threshold:
            acted = False
            decision = NEUTRAL

    shares, notional = _size_position(entry, stop, config) if decision == BUY else (0, 0.0)

    return Signal(
        date=last_date,
        ticker=ticker,
        signal=decision,
        proba=round(raw_proba, 4),
        entry=round(entry, 2),
        stop_loss=round(stop, 2),
        take_profit=round(target, 2),
        size_shares=shares,
        notional=round(notional, 2),
        proba_calibrated=None if proba_cal is None else round(proba_cal, 4),
        meta_proba=None if meta_proba is None else round(meta_proba, 4),
        acted=acted,
        ret_q10=None if q10 is None else round(q10, 4),
        ret_q50=None if q50 is None else round(q50, 4),
        ret_q90=None if q90 is None else round(q90, 4),
        sigma_horizon=None if sigma_h is None else round(sigma_h, 6),
        sltp_method=sltp_method,
    )


def explain_signal(
    ticker: str,
    config: Config,
    store: OhlcvStore,
    *,
    top_k: int = 5,
) -> dict[str, object] | None:
    """SHAP-style explanation for the most recent signal of ``ticker``.

    Re-runs the same feature build as :func:`generate_signals` so the row
    sent to the model is identical to what produced the live proba, then
    asks the LightGBM booster for per-feature contributions. Returns the
    top ``top_k`` features by absolute contribution + the base value and
    a short list of the most-recent news headlines (the latter is best-
    effort — empty list when no news cache exists).

    Returns ``None`` when the ticker isn't in the cache or the resolved
    model isn't a LightGBM booster (the explain API is LGBM-only by
    design; other model frameworks expose contributions differently).
    """
    df = store.load(ticker)
    if df is None or df.empty:
        return None

    label_cfg = LabelConfig(**config.labeling.model_dump())
    model, with_earnings, with_news = _resolve_model(store, config, label_cfg)
    if not isinstance(model, LGBMModel):
        return None

    market = store.load(MARKET_TICKER)
    earnings_store = _earnings_store_if_available(config) if with_earnings else None
    news_store = _news_store_if_available(config) if with_news else None
    earnings_df = earnings_store.load(ticker) if earnings_store is not None else None
    news_df = news_store.load(ticker) if news_store is not None else None

    earnings_arg = earnings_df if with_earnings else None
    if with_earnings and earnings_arg is None:
        earnings_arg = pd.DataFrame()
    news_arg = news_df if with_news else None
    if with_news and news_arg is None:
        news_arg = pd.DataFrame()

    feats = build_features(df, market=market, earnings=earnings_arg, news=news_arg).dropna()
    if feats.empty:
        return None
    cols = feature_columns(earnings=with_earnings, news=with_news)
    last_date = feats.index[-1]
    x = feats.loc[[last_date], cols]
    proba = float(model.predict_proba(x)[0])

    contribs = model.feature_contributions(x)[0]
    # LightGBM tags ``pred_contrib`` with one extra column at the end for the
    # base value (the bias of the booster). Split it off before ranking.
    base_value = float(contribs[-1])
    feature_contribs = contribs[:-1]
    ranked = sorted(
        zip(cols, feature_contribs.tolist(), strict=False),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_k]

    recent_news: list[dict[str, object]] = []
    if news_df is not None and not news_df.empty:
        tail = (
            news_df.dropna(subset=["time_published"])
            .sort_values("time_published", ascending=False)
            .head(3)
        )
        for _, row in tail.iterrows():
            recent_news.append(
                {
                    "title": str(row["title"]) if pd.notna(row["title"]) else "",
                    "time_published": pd.Timestamp(row["time_published"]).isoformat(),  # ty: ignore[unresolved-attribute]
                    "source": str(row["source"]) if pd.notna(row["source"]) else "",
                    "url": str(row["url"]) if pd.notna(row["url"]) else "",
                    "finbert_score": (
                        float(row["finbert_score"]) if pd.notna(row["finbert_score"]) else None
                    ),
                }
            )

    return {
        "ticker": ticker.upper(),
        "date": pd.Timestamp(last_date).date().isoformat(),  # ty: ignore[unresolved-attribute]
        "proba": round(proba, 4),
        "base_value": base_value,
        "top_features": [{"feature": name, "contribution": float(value)} for name, value in ranked],
        "recent_news": recent_news,
    }
