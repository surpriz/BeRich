"""Serve today's market-neutral long/short basket and track it as a paper book.

``generate_longshort_book`` scores the latest cross-section with the promoted (or freshly
trained) ranker, ranks names, and returns the dollar-neutral target weights — the actionable
advice (which names to long/short, at what weight). ``LongShortStore`` persists the daily
basket; ``longshort_equity`` replays the stored baskets against realized returns to give a
paper-trading equity curve (reusing the backtester's return engine).

Live serving uses the tabular ranker on the latest single cross-section. Deep (sequence)
rankers need a lookback window at inference, so live serving trains/serves an LGBM ranker;
the deep rankers remain available for research/backtest via ``berich longshort``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from berich.backtest.longshort import LongShortConfig, build_baskets, returns_from_weights
from berich.backtest.metrics import compute_metrics
from berich.datasets.cross_sectional import (
    _add_cross_sectional_features,
    build_panel_dataset,
)
from berich.features.build import FEATURE_COLUMNS, build_features
from berich.labeling.cross_sectional import CrossSectionalLabelConfig
from berich.models import LGBMRanker, load_active
from berich.training.cross_sectional import CrossSectionalOof

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.models.base import Model

_BASKET_SCHEMA = """
CREATE TABLE IF NOT EXISTS longshort_basket (
    date    DATE    NOT NULL,
    ticker  VARCHAR NOT NULL,
    side    VARCHAR NOT NULL,
    weight  DOUBLE  NOT NULL,
    score   DOUBLE  NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, ticker)
);
"""


@dataclass
class LongShortPosition:
    """One leg of the long/short basket."""

    date: pd.Timestamp
    ticker: str
    side: str  # "LONG" | "SHORT"
    weight: float
    score: float

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["date"] = pd.Timestamp(self.date).date().isoformat()  # ty: ignore[unresolved-attribute]
        return row


@dataclass
class LongShortBook:
    """Today's dollar-neutral basket."""

    date: pd.Timestamp
    positions: list[LongShortPosition]

    @property
    def gross_exposure(self) -> float:
        return float(sum(abs(p.weight) for p in self.positions))


def _serving_cross_section(
    store: OhlcvStore,
    tickers: list[str],
    *,
    market_ticker: str,
    cross_sectional: bool,
) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """Latest-date cross-sectional feature matrix (no labels), for live scoring."""
    market = store.load(market_ticker)
    rows = []
    for t in tickers:
        df = store.load(t)
        if df is None or df.empty:
            continue
        feats = build_features(df, market=market)[FEATURE_COLUMNS].dropna()
        if feats.empty:
            continue
        rows.append(feats.iloc[[-1]].assign(ticker=t))
    if not rows:
        return None
    panel = pd.concat(rows)
    latest = pd.Timestamp(panel.index.max())
    panel = pd.DataFrame(panel[panel.index == latest])
    if cross_sectional and len(panel) > 1:
        panel = _add_cross_sectional_features(panel)
    return panel, latest  # ty: ignore[invalid-return-type]


def _serving_ranker(config: Config, store: OhlcvStore) -> tuple[Model, list[str]]:
    """Promoted tabular ranker if available, else an LGBM ranker trained inline."""
    active = load_active(config.models_dir_for("longshort"))
    if active is not None and active[1].framework == "lightgbm-ranker":
        model, meta = active
        return model, list(meta.feature_columns)
    ls = config.longshort
    label_cfg = CrossSectionalLabelConfig(
        horizon_days=ls.horizon_days,
        beta_window=ls.beta_window,
        residualize=ls.residualize,
        standardize="rank" if ls.standardize == "rank" else "zscore",
    )
    panel = build_panel_dataset(
        store,
        config.tickers_for_universe(ls.universe),
        label_cfg,
        market_ticker=ls.market_ticker,
        min_names_per_date=ls.min_names_per_date,
        cross_sectional=ls.cross_sectional_features,
    )
    model = LGBMRanker().fit(panel.x, panel.y, sample_weight=panel.weight, tickers=panel.tickers)
    return model, list(panel.x.columns)


def generate_longshort_book(config: Config, store: OhlcvStore) -> LongShortBook | None:
    """Build today's dollar-neutral long/short basket from the latest cross-section."""
    ls = config.longshort
    tickers = config.tickers_for_universe(ls.universe)
    served = _serving_cross_section(
        store, tickers, market_ticker=ls.market_ticker, cross_sectional=ls.cross_sectional_features
    )
    if served is None:
        return None
    panel_df, latest = served

    model, feat_cols = _serving_ranker(config, store)
    x = panel_df.reindex(columns=feat_cols, fill_value=0.0)
    scores = model.predict_proba(x, tickers=panel_df["ticker"])
    score_by_ticker = dict(zip(panel_df["ticker"].to_numpy(), scores, strict=False))

    oof = CrossSectionalOof(
        frame=pd.DataFrame(
            {"score": scores, "y_true": 0.0, "ticker": panel_df["ticker"].to_numpy()},
            index=pd.DatetimeIndex([latest] * len(panel_df), name="date"),
        )
    )
    close = pd.DataFrame(
        {t: df["close"] for t in tickers if (df := store.load(t)) is not None}
    ).sort_index()
    ret = close.pct_change(fill_method=None)
    bt_cfg = LongShortConfig(
        top_decile=ls.top_decile,
        bottom_decile=ls.bottom_decile,
        weighting=ls.weighting,
        rebalance_days=ls.rebalance_days,
        gross_leverage=ls.gross_leverage,
        vol_lookback=ls.vol_lookback,
        min_names=ls.min_names_per_date,
    )
    baskets = build_baskets(oof, ret, bt_cfg)
    if baskets.empty:
        return LongShortBook(date=latest, positions=[])
    row = baskets.iloc[-1]
    positions = [
        LongShortPosition(
            date=latest,
            ticker=str(t),
            side="LONG" if w > 0 else "SHORT",
            weight=round(float(w), 6),
            score=round(float(score_by_ticker.get(t, 0.0)), 6),
        )
        for t, w in row.items()
        if w != 0
    ]
    return LongShortBook(date=latest, positions=positions)


class LongShortStore:
    """DuckDB persistence for daily long/short baskets."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(db_path)) as con:
            con.execute(_BASKET_SCHEMA)

    def save(self, book: LongShortBook) -> int:
        if not book.positions:
            return 0
        rows = pd.DataFrame([p.as_row() for p in book.positions])
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        with duckdb.connect(str(self.db_path)) as con:
            con.register("incoming", rows)
            con.execute(
                "DELETE FROM longshort_basket WHERE (date, ticker) IN "
                "(SELECT date, ticker FROM incoming)"
            )
            con.execute(
                "INSERT INTO longshort_basket (date, ticker, side, weight, score) "
                "SELECT date, ticker, side, weight, score FROM incoming"
            )
        return len(rows)

    def latest(self) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path)) as con:
            return con.execute(
                "SELECT * FROM longshort_basket "
                "WHERE date = (SELECT max(date) FROM longshort_basket) ORDER BY weight DESC"
            ).df()

    def weight_matrix(self) -> pd.DataFrame:
        """All stored baskets as a (date x ticker) weight matrix for equity replay."""
        with duckdb.connect(str(self.db_path)) as con:
            df = con.execute("SELECT date, ticker, weight FROM longshort_basket").df()
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="date", columns="ticker", values="weight", fill_value=0.0)


def longshort_equity(config: Config, store: OhlcvStore) -> dict[str, object]:
    """Paper-book equity: replay stored baskets against realized returns."""
    ls = config.longshort
    ls_store = LongShortStore(config.db_path)
    weights = ls_store.weight_matrix()
    if weights.empty:
        return {"n_baskets": 0}
    weights.index = pd.to_datetime(weights.index)
    tickers = list(weights.columns)
    close = pd.DataFrame(
        {t: df["close"] for t in tickers if (df := store.load(t)) is not None}
    ).sort_index()
    ret = close.pct_change(fill_method=None)
    cfg = LongShortConfig(
        fee_bps=ls.fee_bps,
        slippage_bps=ls.slippage_bps,
        borrow_bps_annual=ls.borrow_bps_annual,
        target_vol=ls.target_vol,
        vol_lookback=ls.vol_lookback,
    )
    net_ret, avg_gross = returns_from_weights(weights, ret, cfg)
    metrics = compute_metrics(net_ret)
    return {
        "n_baskets": len(weights),
        "sharpe": metrics.sharpe,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "avg_gross": avg_gross,
    }


__all__ = [
    "LongShortBook",
    "LongShortPosition",
    "LongShortStore",
    "generate_longshort_book",
    "longshort_equity",
]
