"""Generate today's swing-trade advice for the watchlist.

Per ticker we serve up to two uniquely-trained models — a LONG model (P the upper
barrier is hit first) and, when one has been promoted, a SHORT model (P the mirrored
lower barrier is hit first). The two calibrated probabilities are turned into a
LONG / SHORT / NEUTRAL call by best expectancy, and for an actionable call an ATR
stop / target (mirrored for shorts) and a risk-based position size are attached.
This is the "conseil" surface: which side, where to enter, where the stop goes, and
how big the position should be.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from berich.data.earnings import EarningsStore
from berich.data.news import NewsStore
from berich.features.build import (
    build_features,
    feature_columns,
    market_reference_for,
)
from berich.features.earnings_features import EARNINGS_FEATURE_COLUMNS
from berich.features.indicators import atr
from berich.features.microstructure import MICRO_FEATURE_COLUMNS
from berich.features.news_features import NEWS_FEATURE_COLUMNS
from berich.features.volatility import forecast_vol
from berich.labeling.triple_barrier import LabelConfig, adaptive_barriers
from berich.models import LGBMModel, load_active, load_best
from berich.models.meta_labeler import PRIMARY_PROBA_COL
from berich.signals.calibration import ProbaCalibrator, load_calibrator

if TYPE_CHECKING:
    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.models.base import Model

logger = logging.getLogger(__name__)

# Directional decisions emitted to the dashboard / paper book.
LONG = "LONG"
SHORT = "SHORT"
NEUTRAL = "NEUTRAL"
# Legacy aliases: the long-only path historically emitted "BUY"/"SELL". `BUY` now maps
# to LONG so old consumers (and stored rows) keep working; `_classify` below is the
# legacy long-only classifier still used by tests. `SELL` stays a distinct constant.
BUY = LONG
SELL = "SELL"
# Signal strings that open a long position (new LONG + legacy BUY rows on disk).
LONG_SIGNALS: frozenset[str] = frozenset({LONG, "BUY"})


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
    # Direction of the call ("long" | "short"); NEUTRAL rows default to "long".
    direction: str = "long"
    # Per-side calibrated P(win): both populated when both models exist, else one/None.
    proba_long: float | None = None
    proba_short: float | None = None
    # Enriched advice fields (optional; default to neutral/None for back-compat).
    proba_calibrated: float | None = None
    meta_proba: float | None = None
    acted: bool = True  # False when the meta-label filter vetoed a BUY
    ret_q10: float | None = None
    ret_q50: float | None = None
    ret_q90: float | None = None
    sigma_horizon: float | None = None
    sltp_method: str = "atr_fixed"  # "vol_scaled" | "quantile" | "atr_fixed"
    # True when the acted side's per-asset model passed the guard (promoted); False = advisory
    # (served from the asset's own optimized-but-unpromoted candidate, never a generic fallback).
    promoted: bool = False

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        # ``self.date`` is set by the producer and never NaT in practice; ignore
        # the stub's NaT union here rather than litter the call sites with casts.
        row["date"] = pd.Timestamp(self.date).date().isoformat()  # ty: ignore[unresolved-attribute]
        return row


def _classify(proba: float, config: Config) -> str:
    """Legacy long-only classifier (kept for back-compat / tests)."""
    if proba >= config.signals.buy_threshold:
        return BUY
    if proba <= config.signals.sell_threshold:
        return SELL
    return NEUTRAL


def _decide(
    p_long: float | None, p_short: float | None, config: Config
) -> tuple[str, Literal["long", "short"]]:
    """Pick LONG / SHORT / NEUTRAL from the two calibrated win probabilities.

    Returns ``(signal, direction)``. With symmetric 2:1 barriers on both sides the
    expectancy ordering equals the probability ordering, so comparing probabilities is
    the expectancy comparison. ``None`` means that side has no eligible model.
    """
    sig = config.signals
    long_ok = (
        p_long is not None
        and p_long >= sig.buy_threshold
        and (p_short is None or p_long >= p_short)  # tie favors long (deterministic)
    )
    short_ok = (
        sig.enable_short
        and p_short is not None
        and p_short >= sig.short_threshold
        and (p_long is None or p_short > p_long)
    )
    if long_ok:
        return LONG, "long"
    if short_ok:
        return SHORT, "short"
    return NEUTRAL, "long"


def _price_decimals(price: float) -> int:
    """Decimal places to render a price at, scaled to its magnitude.

    Equities (~$10 to $1000) keep 2 decimals; low-priced instruments like FX pairs
    (~1.17) need 4 so the entry/stop/target don't collapse onto each other when rounded.
    """
    p = abs(price)
    if p >= 100:  # noqa: PLR2004 — magnitude buckets, not magic constants worth naming
        return 2
    if p >= 1:
        return 4
    return 6


def _size_position(entry: float, stop: float, config: Config) -> tuple[int, float]:
    """Risk-based sizing: risk at most ``risk_pct`` of capital to the stop.

    Uses the absolute entry-to-stop distance so it works for both longs (stop below
    entry) and shorts (stop above entry). The notional is capped at the account capital
    (no leverage by default) — without the cap a low-priced instrument (e.g. an FX pair
    near 1.17 with a tiny ATR stop) sizes to tens of thousands of units worth several
    times the account.
    """
    stop_distance = abs(entry - stop)
    if stop_distance <= 0 or entry <= 0:
        return 0, 0.0
    risk_amount = config.signals.capital * config.signals.risk_pct
    risk_shares = math.floor(risk_amount / stop_distance)
    capital_shares = math.floor(config.signals.capital / entry)  # no-leverage cap
    shares = min(risk_shares, capital_shares)
    return shares, shares * entry


def _earnings_store_if_available(config: Config) -> EarningsStore | None:
    """Return an EarningsStore iff the cache directory has at least one non-empty file."""
    store = EarningsStore(config.earnings_dir)
    return store if store.has_any_data() else None


def _news_store_if_available(config: Config) -> NewsStore | None:
    """Return a NewsStore iff the cache directory has at least one non-empty file."""
    store = NewsStore(config.news_dir)
    return store if store.has_any_data() else None


def _needs_news(cols: list[str]) -> bool:
    return any(c in NEWS_FEATURE_COLUMNS for c in cols)


def _needs_earnings(cols: list[str]) -> bool:
    return any(c in EARNINGS_FEATURE_COLUMNS for c in cols)


def _needs_micro(cols: list[str]) -> bool:
    return any(c in MICRO_FEATURE_COLUMNS for c in cols)


@dataclass
class _SideModel:
    """A per-asset model for one side, plus whether it cleared the guard (promoted)."""

    model: Model
    cols: list[str]
    calibrator: ProbaCalibrator | None
    promoted: bool


def _ticker_side_model(config: Config, ticker: str, side: str) -> _SideModel | None:
    """Per-asset model for one side: the promoted winner if any, else the best optimized
    candidate (advisory). Returns ``None`` only when the asset has no artifact for ``side``
    at all (not yet optimized). Never falls back to a generic/global model — an asset is served
    exclusively from its own trained models.
    """
    registry_dir = config.model_dir_for_ticker(ticker, side)
    loaded = load_best(registry_dir)
    if loaded is None:
        return None
    model, meta = loaded
    cal = load_calibrator(registry_dir / meta.name)
    promoted = load_active(registry_dir) is not None
    return _SideModel(model, list(meta.feature_columns), cal, promoted)


def _optimized_tickers(config: Config) -> list[str]:
    """Tickers that have had their per-asset HPO run (an Optuna study with >=1 trial).

    The dashboard surfaces only these — assets we've actually worked on/optimized — so it never
    shows raw, un-tuned fallbacks. As the nightly HPO queue advances, more assets appear here.
    """
    from berich.training.status import _hpo_trial_counts, _hpo_trials_for  # noqa: PLC0415

    counts = _hpo_trial_counts(config.optuna_db)
    sides = config.zoo.ticker_sides
    return [
        t
        for t in config.tradeable_tickers()
        if any(_hpo_trials_for(counts, t, None, s) > 0 for s in sides)
    ]


def _load_meta_labeler(config: Config) -> Model | None:
    """Load the promoted meta-labeling model from the ``meta/`` namespace, if enabled."""
    if not config.signals.use_meta_label:
        return None
    active = load_active(config.models_dir / "meta")
    return active[0] if active is not None else None


def generate_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Emit one signal per **optimized** asset, served only from that asset's own models.

    "Optimized" = the asset has had its per-asset HPO run (see :func:`_optimized_tickers`). An
    asset with no per-asset model (not yet worked on) produces no signal — the dashboard stays
    clean and only shows assets we've actually tuned. Each asset is scored with its own long
    (and, if present, short) model; there is no generic/global fallback.
    """
    label_cfg = LabelConfig(**config.labeling.model_dump())
    meta_model = _load_meta_labeler(config)
    earnings_store = _earnings_store_if_available(config)
    news_store = _news_store_if_available(config)
    market_cache: dict[str, pd.DataFrame | None] = {}

    signals: list[Signal] = []
    for ticker in _optimized_tickers(config):
        df = store.load(ticker)
        if df is None or df.empty:
            continue
        per_long = _ticker_side_model(config, ticker, "long")
        if per_long is None:
            continue  # only a short model exists for an asset we'd never go long on; skip long
        per_short = _ticker_side_model(config, ticker, "short")

        asset_class = config.asset_class_for(ticker)
        ref = market_reference_for(asset_class)
        if ref not in market_cache:
            market_cache[ref] = store.load(ref)
        market = market_cache[ref]

        all_cols = [*per_long.cols, *(per_short.cols if per_short else [])]
        with_earnings = _needs_earnings(all_cols)
        with_news = _needs_news(all_cols)
        earnings_df = (
            earnings_store.load(ticker) if (with_earnings and earnings_store is not None) else None
        )
        news_df = news_store.load(ticker) if (with_news and news_store is not None) else None
        signal = _signal_for_ticker(
            ticker,
            df,
            per_long.model,
            config,
            label_cfg,
            market=market,
            earnings=earnings_df,
            news=news_df,
            with_earnings=with_earnings,
            with_news=with_news,
            feature_cols=per_long.cols,
            calibrator=per_long.calibrator,
            meta_model=meta_model,
            short_model=per_short.model if per_short else None,
            short_feature_cols=per_short.cols if per_short else None,
            short_calibrator=per_short.calibrator if per_short else None,
            long_promoted=per_long.promoted,
            short_promoted=per_short.promoted if per_short else False,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def generate_multi_asset_signals(config: Config, store: OhlcvStore) -> list[Signal]:
    """Deprecated: non-US assets are now covered by :func:`generate_signals` (optimized-only,
    served from each asset's own model). Kept as a no-op so older callers don't break.
    """
    _ = (config, store)
    return []


def _signal_for_ticker(  # noqa: C901, PLR0915
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
    feature_cols: list[str] | None = None,
    calibrator: ProbaCalibrator | None = None,
    meta_model: Model | None = None,
    short_model: Model | None = None,
    short_feature_cols: list[str] | None = None,
    short_calibrator: ProbaCalibrator | None = None,
    long_promoted: bool = False,
    short_promoted: bool = False,
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
    with_micro = _needs_micro(feature_cols or []) or _needs_micro(short_feature_cols or [])
    feats = build_features(
        df, market=market, earnings=earnings_arg, news=news_arg, micro=with_micro
    ).dropna()
    if feats.empty:
        return None
    last_date = feats.index[-1]
    long_cols = (
        feature_cols
        if feature_cols is not None
        else feature_columns(earnings=with_earnings, news=with_news)
    )
    x = feats.loc[[last_date], long_cols]
    raw_long = float(model.predict_proba(x)[0])
    p_long = float(calibrator.transform(np.array([raw_long]))[0]) if calibrator else raw_long

    raw_short: float | None = None
    p_short: float | None = None
    if short_model is not None:
        scols = short_feature_cols or long_cols
        xs = feats.loc[[last_date], scols]
        raw_short = float(short_model.predict_proba(xs)[0])
        p_short = (
            float(short_calibrator.transform(np.array([raw_short]))[0])
            if short_calibrator
            else raw_short
        )

    a = float(atr(df["high"], df["low"], df["close"], label_cfg.atr_window).loc[last_date])
    entry = float(df["close"].loc[last_date])
    if np.isnan(a):
        return None

    decision, direction = _decide(p_long, p_short, config)

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
        # Quantile barriers only for the long side (the long model exposes them); the short
        # branch mirrors the ATR/vol barriers about entry instead.
        if direction == "long":
            predict_quantiles = getattr(model, "predict_quantiles", None)
            if predict_quantiles is not None:
                q = np.asarray(predict_quantiles(x))[0]
                q10, q90 = float(q[0]), float(q[-1])
                q50 = float(q[len(q) // 2])
                quantiles = (q10, q90)
        stop, target, rationale = adaptive_barriers(
            entry, a, vf, label_cfg, quantiles=quantiles, direction=direction
        )
        sltp_method = str(rationale["method"])
    elif direction == "short":
        stop = entry + label_cfg.stop_loss_atr * a
        target = entry - label_cfg.take_profit_atr * a
        sltp_method = "atr_fixed"
    else:
        stop = entry - label_cfg.stop_loss_atr * a
        target = entry + label_cfg.take_profit_atr * a
        sltp_method = "atr_fixed"

    # Meta-labeling precision filter: veto a LONG whose meta P(correct) is too low.
    # Shorts have no meta model, so they are never vetoed here.
    meta_proba: float | None = None
    acted = True
    if meta_model is not None and decision == LONG:
        meta_x = x.copy()
        meta_x[PRIMARY_PROBA_COL] = p_long
        meta_proba = float(meta_model.predict_proba(meta_x)[0])
        if meta_proba < config.signals.meta_threshold:
            acted = False
            decision = NEUTRAL
            direction = "long"

    shares, notional = _size_position(entry, stop, config) if decision != NEUTRAL else (0, 0.0)

    # Headline proba fields reflect the acted side.
    acted_raw = raw_short if decision == SHORT and raw_short is not None else raw_long
    acted_cal = p_short if decision == SHORT and p_short is not None else p_long
    acted_used_cal = (short_calibrator if decision == SHORT else calibrator) is not None
    acted_promoted = short_promoted if decision == SHORT else long_promoted

    return Signal(
        date=last_date,
        ticker=ticker,
        signal=decision,
        proba=round(acted_raw, 4),
        entry=round(entry, _price_decimals(entry)),
        stop_loss=round(stop, _price_decimals(entry)),
        take_profit=round(target, _price_decimals(entry)),
        size_shares=shares,
        notional=round(notional, 2),
        direction=direction,
        proba_long=round(p_long, 4),
        proba_short=None if p_short is None else round(p_short, 4),
        proba_calibrated=round(acted_cal, 4) if acted_used_cal else None,
        meta_proba=None if meta_proba is None else round(meta_proba, 4),
        acted=acted,
        ret_q10=None if q10 is None else round(q10, 4),
        ret_q50=None if q50 is None else round(q50, 4),
        ret_q90=None if q90 is None else round(q90, 4),
        sigma_horizon=None if sigma_h is None else round(sigma_h, 6),
        sltp_method=sltp_method,
        promoted=acted_promoted,
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

    # Explain the asset's OWN long model (the one that produced its served signal). Only
    # optimized assets have one; others (and non-LGBM winners) have no LGBM explanation.
    per_long = _ticker_side_model(config, ticker, "long")
    if per_long is None or not isinstance(per_long.model, LGBMModel):
        return None
    model, cols = per_long.model, per_long.cols
    with_earnings = _needs_earnings(cols)
    with_news = _needs_news(cols)
    with_micro = _needs_micro(cols)

    market = store.load(market_reference_for(config.asset_class_for(ticker)))
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

    feats = build_features(
        df, market=market, earnings=earnings_arg, news=news_arg, micro=with_micro
    ).dropna()
    if feats.empty:
        return None
    last_date = feats.index[-1]
    x = feats.loc[[last_date], cols]
    proba = float(model.predict_proba(x)[0])

    # Per-side calibrated win probabilities + the resulting direction, mirroring the live
    # signal service. Best-effort: a per-ticker long model overrides the global one for the
    # long proba; the short side is per-ticker-only (absent => never shorted). All branches
    # are null-guarded so the explain endpoint never fails on a missing side model.
    direction, p_long, p_short = _explain_directional(ticker, feats, last_date, config, model, cols)

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
        "direction": direction,
        "proba_long": None if p_long is None else round(p_long, 4),
        "proba_short": None if p_short is None else round(p_short, 4),
        "base_value": base_value,
        "top_features": [{"feature": name, "contribution": float(value)} for name, value in ranked],
        "recent_news": recent_news,
    }


def _explain_directional(
    ticker: str,
    feats: pd.DataFrame,
    last_date: pd.Timestamp,
    config: Config,
    long_model: Model,
    long_cols: list[str],
) -> tuple[str, float | None, float | None]:
    """Compute ``(direction, proba_long, proba_short)`` for the explain payload.

    ``long_model``/``long_cols`` are the asset's own long model (already resolved by the
    caller). The short side is per-ticker-only (absent => never shorted). Best-effort and fully
    null-guarded so the explain endpoint never fails on a missing/mismatched side model.
    """
    long_cal = _ticker_side_model(config, ticker, "long")
    cal = long_cal.calibrator if long_cal else None
    try:
        xl = feats.loc[[last_date], long_cols]
        raw_long = float(long_model.predict_proba(xl)[0])
        p_long: float | None = float(cal.transform(np.array([raw_long]))[0]) if cal else raw_long
    except (KeyError, ValueError):
        p_long = None

    p_short: float | None = None
    per_short = _ticker_side_model(config, ticker, "short")
    if per_short is not None:
        try:
            xs = feats.loc[[last_date], per_short.cols]
            raw_short = float(per_short.model.predict_proba(xs)[0])
            scal = per_short.calibrator
            p_short = float(scal.transform(np.array([raw_short]))[0]) if scal else raw_short
        except (KeyError, ValueError):
            p_short = None

    _, direction = _decide(p_long, p_short, config)
    return direction, p_long, p_short
