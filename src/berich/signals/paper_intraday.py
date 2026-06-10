"""Intraday (1h crypto) paper book — a parallel sibling of :mod:`berich.signals.paper`.

The daily paper engine counts in business days (``np.busday_count``) and steps the equity
curve over ``pd.bdate_range`` against a SPY benchmark — all wrong for a 24/7 crypto market.
This module reuses every *cadence-free* helper from ``paper.py`` (exit resolution, trailing
ratchet, exposure caps, tier routing, sizing, P&L) and re-implements only the three
calendar-bound primitives:

1. an :class:`IntradayPaperStore` keyed on a full ``TIMESTAMP`` (so multiple same-day entries
   per pair don't collide on the DATE primary key), in its OWN DuckDB file;
2. **hours-held** instead of business-days-held;
3. a **continuous hourly** equity curve (no weekend skip) benchmarked against the traded
   pair's own buy-&-hold, not SPY.

Isolation: this never touches ``config.db_path`` or ``paper.py``'s functions — only imports
its pure helpers. The committed book still opens ``tier == promoted`` only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.backtest.metrics import max_drawdown
from berich.labeling.triple_barrier import LabelConfig as _LabelConfig
from berich.signals.paper import (
    _TRADE_COLUMNS,
    CLOSED_STATUSES,
    LONG_OPEN,
    OBSERVE_TIER,
    OPEN,
    PROMOTED_TIER,
    SHORT_OPEN,
    OpenPosition,
    PaperStore,
    _apply_exposure_caps,
    _candidate_rows,
    _cap_open_positions,
    _direction_pnl,
    _fetch_scalar_int,
    _is_short,
    _open_trail_stop,
    _resolve_trade_exit,
    _signal_tiers,
    _ts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.signals.store import SignalStore

logger = logging.getLogger(__name__)

# Same columns as the daily book, but date_open / date_close are full TIMESTAMPs and the
# benchmark column is the traded pair (not SPY). Keeping the column NAMES identical lets the
# inherited SELECT helpers (all_trades / open_trades / closed_trades) work unchanged.
_INTRADAY_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    date_open     TIMESTAMP NOT NULL,
    ticker        VARCHAR   NOT NULL,
    signal        VARCHAR   NOT NULL,
    entry         DOUBLE    NOT NULL,
    stop          DOUBLE    NOT NULL,
    target        DOUBLE    NOT NULL,
    size_shares   BIGINT    NOT NULL,
    status        VARCHAR   NOT NULL DEFAULT 'open',
    date_close    TIMESTAMP,
    exit_price    DOUBLE,
    pnl_pct       DOUBLE,
    pnl_eur       DOUBLE,
    exit_strategy VARCHAR   NOT NULL DEFAULT 'fixed',
    tier          VARCHAR   NOT NULL DEFAULT 'promoted',
    cost_bps      DOUBLE,
    source        VARCHAR   NOT NULL DEFAULT 'intraday_1h',
    created_at    TIMESTAMP DEFAULT now(),
    updated_at    TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date_open, ticker, exit_strategy)
);
"""


class IntradayPaperStore(PaperStore):
    """DuckDB CRUD for the intraday ``paper_trades`` table (TIMESTAMP-keyed, own DB file).

    Inherits the SELECT helpers from :class:`PaperStore`; overrides only the schema bootstrap
    and the two writers that would otherwise truncate the entry/exit timestamps to a date.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_INTRADAY_SCHEMA)

    def insert_new(self, rows: pd.DataFrame) -> int:
        """Insert new (ts_open, ticker, exit_strategy) rows, preserving the full timestamp."""
        if rows.empty:
            return 0
        cols = _TRADE_COLUMNS
        rows = rows.copy()
        for col in cols:
            if col not in rows.columns:
                rows[col] = None
        rows = rows[list(cols)]
        rows["date_open"] = pd.to_datetime(rows["date_open"])  # keep time-of-day, no .dt.date
        rows["exit_strategy"] = rows["exit_strategy"].fillna("fixed")
        rows["tier"] = rows["tier"].fillna(PROMOTED_TIER)
        cols_csv = ", ".join(cols)
        insert_sql = (
            f"INSERT INTO paper_trades ({cols_csv}) "  # noqa: S608 — identifiers, not user input
            f"SELECT {cols_csv} FROM incoming "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM paper_trades p "
            "  WHERE p.date_open = incoming.date_open AND p.ticker = incoming.ticker "
            "    AND p.exit_strategy = incoming.exit_strategy"
            ")"
        )
        with self._connect() as con:
            con.register("incoming", rows)
            before = _fetch_scalar_int(con, "SELECT count(*) FROM paper_trades")
            con.execute(insert_sql)
            after = _fetch_scalar_int(con, "SELECT count(*) FROM paper_trades")
        return after - before

    def close_trade(
        self,
        *,
        date_open: pd.Timestamp,
        ticker: str,
        date_close: pd.Timestamp,
        exit_price: float,
        status: str,
        pnl_pct: float,
        pnl_eur: float,
        exit_strategy: str = "fixed",
    ) -> None:
        """Close a trade, keeping the full close timestamp (no date truncation)."""
        with self._connect() as con:
            con.execute(
                "UPDATE paper_trades SET "
                "status = ?, date_close = ?, exit_price = ?, "
                "pnl_pct = ?, pnl_eur = ?, updated_at = now() "
                "WHERE date_open = ? AND ticker = ? AND exit_strategy = ? AND status = ?",
                [
                    status,
                    pd.Timestamp(date_close),
                    exit_price,
                    pnl_pct,
                    pnl_eur,
                    pd.Timestamp(date_open),
                    ticker,
                    exit_strategy,
                    OPEN,
                ],
            )


def _hours_held(ts_open: pd.Timestamp, now: pd.Timestamp) -> int:
    """Whole hours a trade has been open — continuous, no weekend skip (crypto is 24/7)."""
    secs = (_ts(now) - _ts(ts_open)).total_seconds()
    return 0 if pd.isna(secs) else max(0, int(secs // 3600))


def open_new_intraday_trades(config: Config, store: OhlcvStore, signal_store: SignalStore) -> int:
    """Open an intraday paper trade per actionable signal, under the same money-management caps.

    Mirrors :func:`paper.open_new_trades` but against the intraday book/store. Idempotent on
    ``(ts_open, ticker, exit_strategy)``. ``tier == promoted`` opens committed capital; ``observe``
    routes to the shadow book; advisory is never opened.

    POC note: the shared :class:`SignalStore` keys signals by calendar date, so an entry opens at
    most once per (pair, strategy) per day (at that day's first hourly run). The position is then
    walked and exited at hourly resolution by :func:`update_open_intraday_trades` — i.e. one entry
    decision per day, hourly exits. Finer-grained entries would need a timestamp-keyed signal store.
    """
    latest = signal_store.latest()
    if latest.empty:
        return 0
    actionable = latest[latest["signal"].isin([*LONG_OPEN, *SHORT_OPEN])]
    actionable = actionable[actionable["size_shares"] > 0]
    if actionable.empty:
        return 0
    tier = _signal_tiers(actionable)
    paper = IntradayPaperStore(config.intraday_db_path)
    opened = 0
    for book_tier in (PROMOTED_TIER, OBSERVE_TIER):
        rows = _plan_intraday_book(config, store, paper, actionable[tier == book_tier], book_tier)
        if not rows.empty:
            opened += paper.insert_new(rows)
    logger.info("paper_intraday.open_new_intraday_trades: %d new trades opened", opened)
    return opened


def _intraday_committed_drawdown(config: Config, store: OhlcvStore) -> float:
    """Peak-to-now drawdown (>= 0) of the committed intraday book; 0.0 with no history."""
    eq = get_intraday_equity_curve(config, store, tier=PROMOTED_TIER)
    if eq.empty:
        return 0.0
    series = pd.Series(eq["equity_paper"].to_numpy(), dtype=float).dropna()
    if series.empty:
        return 0.0
    peak = float(series.cummax().iloc[-1])
    current = float(series.iloc[-1])
    if peak <= 0:
        return 0.0
    return max(0.0, 1.0 - current / peak)


def _derisk_intraday(rows: pd.DataFrame, config: Config, store: OhlcvStore) -> pd.DataFrame:
    """Graduated drawdown kill-switch on the committed intraday book (mirrors the daily rule)."""
    if rows.empty:
        return rows
    sig = config.signals
    dd = _intraday_committed_drawdown(config, store)
    if dd >= sig.drawdown_halt_threshold:
        logger.info("paper_intraday: drawdown %.1f%% >= halt — no committed trades", dd * 100)
        return rows.iloc[0:0]
    if dd < sig.drawdown_derisk_threshold:
        return rows
    rows = rows.copy()
    rows["size_shares"] = (rows["size_shares"] * sig.drawdown_derisk_factor).astype(int)
    return rows[rows["size_shares"] >= 1]


def _plan_intraday_book(
    config: Config,
    store: OhlcvStore,
    paper: IntradayPaperStore,
    book: pd.DataFrame,
    book_tier: str,
) -> pd.DataFrame:
    """Candidate intraday trades for one tier after all money-management guardrails."""
    if book.empty:
        return book
    sig = config.signals
    rows = _candidate_rows(book, book_tier)
    if book_tier == PROMOTED_TIER:
        rows = _derisk_intraday(rows, config, store)
        rows = _cap_open_positions(rows, paper.open_trades(tier=book_tier), sig.max_open_positions)
        if rows.empty:
            return rows
    return _apply_exposure_caps(
        rows,
        paper.open_trades(tier=book_tier),
        capital=float(sig.capital),
        max_ticker_pct=float(sig.max_ticker_exposure_pct),
        max_book_pct=float(sig.max_book_exposure_pct),
        max_class_pct=float(sig.max_class_exposure_pct),
        class_of=config.asset_class_for,
    )


def plan_committed_intraday_opens(
    config: Config, store: OhlcvStore, signal_store: SignalStore
) -> pd.DataFrame:
    """Dry-run of the committed intraday book's opens — the FORECAST order sheet for /intraday."""
    latest = signal_store.latest()
    if latest.empty:
        return latest
    actionable = latest[latest["signal"].isin([*LONG_OPEN, *SHORT_OPEN])]
    actionable = actionable[actionable["size_shares"] > 0]
    if actionable.empty:
        return actionable
    tier = _signal_tiers(actionable)
    paper = IntradayPaperStore(config.intraday_db_path)
    return _plan_intraday_book(
        config, store, paper, actionable[tier == PROMOTED_TIER], PROMOTED_TIER
    )


def update_open_intraday_trades(
    config: Config, store: OhlcvStore, label_cfg: _LabelConfig | None = None
) -> int:
    """Walk every open intraday trade forward against the 1h cache; close those that hit."""
    cfg = label_cfg or _LabelConfig(**config.labeling.model_dump())
    if config.intraday.horizon_bars:
        cfg = cfg.model_copy(update={"horizon_days": config.intraday.horizon_bars})
    paper = IntradayPaperStore(config.intraday_db_path)
    open_df = paper.open_trades()
    if open_df.empty:
        return 0
    closed_count = 0
    for _, row in open_df.iterrows():
        ticker = str(row["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            logger.warning("paper_intraday.update: no OHLCV for %s, leaving open", ticker)
            continue
        ts_open = _ts(row["date_open"])
        if ts_open not in ohlcv.index:
            logger.warning("paper_intraday.update: entry %s missing for %s", ts_open, ticker)
            continue
        entry_idx = int(ohlcv.index.get_loc(ts_open))
        short = _is_short(row["signal"])
        strategy = str(row["exit_strategy"]) if "exit_strategy" in row else "fixed"
        resolved = _resolve_trade_exit(ohlcv, row, entry_idx, cfg, short=short, strategy=strategy)
        if resolved is None:
            continue
        exit_idx, exit_price, status = resolved
        pnl_pct, pnl_eur = _direction_pnl(
            float(row["entry"]), float(exit_price), int(row["size_shares"]), short=short
        )
        paper.close_trade(
            date_open=ts_open,
            ticker=ticker,
            date_close=ohlcv.index[exit_idx],
            exit_price=float(exit_price),
            status=status,
            pnl_pct=float(pnl_pct),
            pnl_eur=float(pnl_eur),
            exit_strategy=strategy,
        )
        closed_count += 1
    logger.info("paper_intraday.update_open_intraday_trades: %d trades closed", closed_count)
    return closed_count


def get_open_intraday_positions(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> list[OpenPosition]:
    """Open intraday positions enriched with latest-close MTM; ``days_held`` carries HOURS held."""
    paper = IntradayPaperStore(config.intraday_db_path)
    df = paper.open_trades(tier=tier)
    if exit_strategy is not None and "exit_strategy" in df.columns:
        df = df[df["exit_strategy"].fillna("fixed") == exit_strategy]
    cfg = _LabelConfig(**config.labeling.model_dump())
    out: list[OpenPosition] = []
    now = pd.Timestamp.now()
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            continue
        current_price = float(ohlcv["close"].iloc[-1])
        entry = float(row["entry"])
        shares = int(row["size_shares"])
        short = _is_short(row["signal"])
        ts_open = _ts(row["date_open"])
        mtm_pct, mtm_eur = _direction_pnl(entry, current_price, shares, short=short)
        strategy = str(row["exit_strategy"]) if "exit_strategy" in row else "fixed"
        trail_stop = _open_trail_stop(
            ohlcv, ts_open, float(row["stop"]), cfg, strategy, short=short
        )
        out.append(
            OpenPosition(
                date_open=ts_open,
                ticker=ticker,
                direction="short" if short else "long",
                entry=entry,
                stop=float(row["stop"]),
                target=float(row["target"]),
                size_shares=shares,
                current_price=current_price,
                days_held=_hours_held(ts_open, now),  # HOURS for intraday (UI labels it "h")
                mtm_pct=mtm_pct,
                mtm_eur=mtm_eur,
                exit_strategy=strategy,
                trail_stop=trail_stop,
            )
        )
    return out


def get_intraday_equity_curve(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> pd.DataFrame:
    """Continuous hourly paper-equity series alongside the traded pair's buy-&-hold.

    Equity = capital + cumulative realized P&L + open-trade MTM, stepped over a continuous
    ``freq="1h"`` index (no weekend skip). The benchmark is the configured intraday pair's
    own buy & hold (not SPY) — the honest "long must beat buy & hold" reference.
    """
    paper = IntradayPaperStore(config.intraday_db_path)
    trades = paper.all_trades(exit_strategy, tier=tier)
    if trades.empty:
        return pd.DataFrame(columns=pd.Index(["date", "equity_paper", "equity_bench"]))

    capital = float(config.signals.capital)
    trades["date_open"] = pd.to_datetime(trades["date_open"])
    trades["date_close"] = pd.to_datetime(trades["date_close"])

    start = _ts(trades["date_open"].min()).floor("h")
    now = pd.Timestamp.now().floor("h")
    end = max(now, _ts(trades["date_open"].max()).floor("h"))
    max_close = trades["date_close"].max()
    if pd.notna(max_close):
        end = max(end, _ts(max_close).floor("h"))
    stamps = pd.date_range(start, end, freq="1h")

    cum_realized = pd.Series(0.0, index=stamps)
    closed = trades[trades["status"].isin(CLOSED_STATUSES)]
    if not closed.empty:
        by_close = closed.groupby("date_close")["pnl_eur"].sum().sort_index()
        for d_close, pnl in by_close.items():
            cum_realized.loc[cum_realized.index >= _ts(d_close)] += float(pnl)

    unrealized = pd.Series(0.0, index=stamps)
    for _, row in trades.iterrows():
        ticker_df = store.load(str(row["ticker"]))
        if ticker_df is None or ticker_df.empty:
            continue
        t_open = _ts(row["date_open"]).floor("h")
        is_closed = row["status"] in CLOSED_STATUSES
        t_close = _ts(row["date_close"]).floor("h") if is_closed else end + pd.Timedelta(hours=1)
        mask = (stamps >= t_open) & (stamps < t_close)
        active = stamps[mask]
        if len(active) == 0:
            continue
        closes = ticker_df["close"].reindex(active).ffill()
        entry_px = float(row["entry"])
        move = (entry_px - closes) if _is_short(row["signal"]) else (closes - entry_px)
        unrealized.loc[active] += (move * int(row["size_shares"])).fillna(0.0).to_numpy()

    equity_paper = capital + cum_realized + unrealized

    bench_ticker = (
        config.intraday.tickers[0] if config.intraday.tickers else str(trades["ticker"].iloc[0])
    )
    bench_df = store.load(bench_ticker)
    if bench_df is not None and not bench_df.empty:
        bench_series = bench_df["close"].reindex(stamps).ffill()
        anchor_series = bench_df["close"].loc[:start]
        if anchor_series.empty:
            equity_bench = pd.Series(np.nan, index=stamps)
        else:
            equity_bench = capital * bench_series / float(anchor_series.iloc[-1])
    else:
        equity_bench = pd.Series(np.nan, index=stamps)

    return pd.DataFrame(
        {
            "date": [s.isoformat() for s in stamps],
            "equity_paper": equity_paper.to_numpy(),
            "equity_bench": equity_bench.to_numpy(),
        }
    )


def get_intraday_paper_metrics(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> dict[str, float | int]:
    """Compact summary for the /intraday card: returns vs pair B&H, win rate, drawdown, n."""
    paper = IntradayPaperStore(config.intraday_db_path)
    trades = paper.all_trades(exit_strategy, tier=tier)
    capital = float(config.signals.capital)
    equity = get_intraday_equity_curve(config, store, exit_strategy=exit_strategy, tier=tier)

    n_open = int((trades["status"] == OPEN).sum()) if not trades.empty else 0
    closed = trades[trades["status"].isin(CLOSED_STATUSES)] if not trades.empty else trades
    n_closed = len(closed)
    wins = int((closed["pnl_eur"] > 0).sum()) if n_closed else 0
    win_rate = wins / n_closed if n_closed else 0.0

    if equity.empty:
        return {
            "n_open": n_open,
            "n_closed": n_closed,
            "win_rate": win_rate,
            "total_return_paper": 0.0,
            "total_return_bench": 0.0,
            "max_drawdown_paper": 0.0,
            "capital": capital,
        }

    eq_paper = pd.Series(equity["equity_paper"].to_numpy())
    eq_bench = pd.Series(equity["equity_bench"].to_numpy())
    bench_tail = eq_bench.dropna()
    total_return_bench = (
        float(bench_tail.iloc[-1] / capital - 1.0) if bench_tail.size else float("nan")
    )
    return {
        "n_open": n_open,
        "n_closed": n_closed,
        "win_rate": win_rate,
        "total_return_paper": float(eq_paper.iloc[-1] / capital - 1.0),
        "total_return_bench": total_return_bench,
        "max_drawdown_paper": float(max_drawdown(eq_paper / capital)),
        "capital": capital,
    }


def recent_intraday_executions(
    config: Config,
    store: OhlcvStore,
    *,
    hours: float = 2.0,
    tier: str | None = PROMOTED_TIER,
) -> dict[str, object]:
    """What the committed intraday book actually DID at the last hourly run — executions only.

    The 2h default spans one hourly run with margin; this is the EXECUTED list /intraday's
    replication section follows, never the forecast.
    """
    paper = IntradayPaperStore(config.intraday_db_path)
    df = paper.all_trades(tier=tier)
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    opened: list[dict[str, object]] = []
    closed: list[dict[str, object]] = []
    if not df.empty:
        df = df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["updated_at"] = pd.to_datetime(df["updated_at"])
        for _, r in df[(df["status"] == OPEN) & (df["created_at"] >= cutoff)].iterrows():
            opened.append(
                {
                    "ticker": str(r["ticker"]),
                    "direction": "short" if _is_short(r["signal"]) else "long",
                    "exit_strategy": str(r["exit_strategy"]),
                    "entry": float(r["entry"]),
                    "stop": float(r["stop"]),
                    "target": float(r["target"]),
                    "size_shares": int(r["size_shares"]),
                    "notional": float(r["entry"]) * int(r["size_shares"]),
                    "ts_open": _ts(r["date_open"]).isoformat(),
                }
            )
        for _, r in df[(df["status"] != OPEN) & (df["updated_at"] >= cutoff)].iterrows():
            closed.append(
                {
                    "ticker": str(r["ticker"]),
                    "direction": "short" if _is_short(r["signal"]) else "long",
                    "exit_strategy": str(r["exit_strategy"]),
                    "status": str(r["status"]),
                    "exit_price": float(r["exit_price"]) if pd.notna(r["exit_price"]) else None,
                    "pnl_pct": float(r["pnl_pct"]) if pd.notna(r["pnl_pct"]) else None,
                    "pnl_eur": float(r["pnl_eur"]) if pd.notna(r["pnl_eur"]) else None,
                    "ts_close": (
                        _ts(r["date_close"]).isoformat() if pd.notna(r["date_close"]) else None
                    ),
                }
            )
    adjust: list[dict[str, object]] = [
        {
            "ticker": p.ticker,
            "direction": p.direction,
            "exit_strategy": p.exit_strategy,
            "effective_stop": float(p.trail_stop if p.trail_stop is not None else p.stop),
            "target": float(p.target),
        }
        for p in get_open_intraday_positions(config, store, tier=tier)
        if (p.exit_strategy or "fixed") != "fixed"
    ]
    n_closed_total = int((df["status"] != OPEN).sum()) if not df.empty else 0
    return {
        "as_of": pd.Timestamp.now().isoformat(),
        "capital_base": float(config.signals.capital),
        "open": opened,
        "close": closed,
        "adjust": adjust,
        "closed_total": n_closed_total,
    }


__all__ = [
    "IntradayPaperStore",
    "get_intraday_equity_curve",
    "get_intraday_paper_metrics",
    "get_open_intraday_positions",
    "open_new_intraday_trades",
    "plan_committed_intraday_opens",
    "recent_intraday_executions",
    "update_open_intraday_trades",
]
