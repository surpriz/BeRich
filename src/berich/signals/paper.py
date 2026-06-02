"""Paper-trade tracker on top of the daily signal output.

DuckDB-backed simulation: each BUY signal opens a fictive long at the signal's
entry/stop/target, then we walk forward through the cached OHLCV and apply the
**same** ATR-stop / ATR-target / horizon rule as the backtest. Closed trades
carry their realized P&L; open trades carry a mark-to-market against the latest
close.

The point is daily discipline and an honest comparison vs same-capital SPY
buy & hold — not a claim of edge. The v0.1.0 model does not beat buy & hold
(see ``docs/RESULTS.md``); this tracker exists to make that visible day by
day. The schema and queries are idempotent: ``open_new_trades`` skips rows
already in the table (UNIQUE ``(date_open, ticker)``) and ``update_open_trades``
only touches rows with ``status='open'``, so re-running them is a no-op once
the world hasn't moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

from berich.backtest.metrics import max_drawdown
from berich.features.build import MARKET_TICKER
from berich.labeling.triple_barrier import LabelConfig as _LabelConfig

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.signals.store import SignalStore

logger = logging.getLogger(__name__)


def _ts(value: object) -> pd.Timestamp:
    """``pd.Timestamp(value)`` with a runtime assertion that the result is not NaT.

    The pandas type stubs widen ``pd.Timestamp(...)`` to ``Timestamp | NaTType``,
    which propagates through every downstream attribute access. We only ever feed
    this with values that are not nullable (row keys, today's date, opened-trade
    timestamps), so collapsing the union is correct and keeps the type-checker
    quiet without scattering ``# ty: ignore`` across the module.
    """
    ts = pd.Timestamp(value)  # ty: ignore[invalid-argument-type]
    if pd.isna(ts):
        msg = f"unexpected NaT timestamp from {value!r}"
        raise ValueError(msg)
    return ts  # ty: ignore[invalid-return-type]


def _fetch_scalar_int(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a single-row scalar-returning query and return it as an ``int``."""
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0])


OPEN = "open"
CLOSED_STOP = "closed_stop"
CLOSED_TARGET = "closed_target"
CLOSED_TIME = "closed_time"
CLOSED_STATUSES: tuple[str, ...] = (CLOSED_STOP, CLOSED_TARGET, CLOSED_TIME)

# Signal labels that open a position, by direction. The paper book opens both
# sides (promoted long + promoted short); ``signal`` on each row records which.
LONG_OPEN: tuple[str, ...] = ("LONG", "BUY")
SHORT_OPEN: tuple[str, ...] = ("SHORT",)


def _is_short(signal: object) -> bool:
    """True for a short trade. Direction is derived from the stored ``signal``."""
    return str(signal).upper() in SHORT_OPEN


def _direction_pnl(
    entry: float, exit_price: float, shares: int, *, short: bool
) -> tuple[float, float]:
    """Return ``(pnl_pct, pnl_eur)`` for a long or short leg.

    Short P&L is the mirror of long: you profit when price falls, so the per-unit
    move is ``entry - exit`` instead of ``exit - entry``.
    """
    move = (entry - exit_price) if short else (exit_price - entry)
    pnl_pct = move / entry if entry else 0.0
    return pnl_pct, move * shares


_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    date_open    DATE    NOT NULL,
    ticker       VARCHAR NOT NULL,
    signal       VARCHAR NOT NULL,
    entry        DOUBLE  NOT NULL,
    stop         DOUBLE  NOT NULL,
    target       DOUBLE  NOT NULL,
    size_shares  BIGINT  NOT NULL,
    status       VARCHAR NOT NULL DEFAULT 'open',
    date_close   DATE,
    exit_price   DOUBLE,
    pnl_pct      DOUBLE,
    pnl_eur      DOUBLE,
    created_at   TIMESTAMP DEFAULT now(),
    updated_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date_open, ticker)
);
"""

# Phase 7 — add the ``source`` column on existing tables. DuckDB's
# ``ADD COLUMN IF NOT EXISTS`` is a safe no-op on freshly-created tables and
# only fires the actual ALTER on tables that pre-date this migration. The
# default keeps the daily-LGBM book unaffected; PEAD signals will write
# ``'pead'`` so the two strategies can be tracked separately.
_MIGRATION_SOURCE_COLUMN = (
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'daily_lgbm';"
)

_TRADE_COLUMNS = (
    "date_open",
    "ticker",
    "signal",
    "entry",
    "stop",
    "target",
    "size_shares",
    "status",
    "date_close",
    "exit_price",
    "pnl_pct",
    "pnl_eur",
)


@dataclass
class OpenPosition:
    """A still-open paper trade with its mark-to-market against the latest close."""

    date_open: pd.Timestamp
    ticker: str
    direction: str
    entry: float
    stop: float
    target: float
    size_shares: int
    current_price: float
    days_held: int
    mtm_pct: float
    mtm_eur: float

    def as_row(self) -> dict[str, object]:
        return {
            "date_open": _ts(self.date_open).date().isoformat(),
            "ticker": self.ticker,
            "direction": self.direction,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "size_shares": self.size_shares,
            "current_price": self.current_price,
            "days_held": self.days_held,
            "mtm_pct": self.mtm_pct,
            "mtm_eur": self.mtm_eur,
        }


# ---------------------------------------------------------------- low-level store ----


class PaperStore:
    """DuckDB CRUD primitive for the ``paper_trades`` table.

    Single-purpose: own the table, expose typed reads/writes. Business logic
    (deciding which trades to open, when to close, how to mark to market) lives
    in the top-level functions below, not here.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)
            con.execute(_MIGRATION_SOURCE_COLUMN)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def all_trades(self) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute("SELECT * FROM paper_trades ORDER BY date_open, ticker").df()

    def open_trades(self) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM paper_trades WHERE status = ? ORDER BY date_open, ticker",
                [OPEN],
            ).df()

    def closed_trades(self, limit: int | None = None) -> pd.DataFrame:
        query = "SELECT * FROM paper_trades WHERE status <> ? ORDER BY date_close DESC, ticker"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self._connect() as con:
            return con.execute(query, [OPEN]).df()

    def insert_new(self, rows: pd.DataFrame) -> int:
        """Insert new (date_open, ticker) rows; skip those already present.

        Idempotent: re-running on the same input opens nothing the second time.
        """
        if rows.empty:
            return 0
        rows = rows.copy()
        for col in _TRADE_COLUMNS:
            if col not in rows.columns:
                rows[col] = None
        rows = rows[list(_TRADE_COLUMNS)]
        rows["date_open"] = pd.to_datetime(rows["date_open"]).dt.date
        # _TRADE_COLUMNS is a module-level constant of identifier strings, never user
        # input — safe to interpolate into SQL.
        cols_csv = ", ".join(_TRADE_COLUMNS)
        insert_sql = (
            f"INSERT INTO paper_trades ({cols_csv}) "  # noqa: S608 — see note above
            f"SELECT {cols_csv} FROM incoming "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM paper_trades p "
            "  WHERE p.date_open = incoming.date_open AND p.ticker = incoming.ticker"
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
    ) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE paper_trades SET "
                "status = ?, date_close = ?, exit_price = ?, "
                "pnl_pct = ?, pnl_eur = ?, updated_at = now() "
                "WHERE date_open = ? AND ticker = ? AND status = ?",
                [
                    status,
                    pd.Timestamp(date_close).date(),
                    exit_price,
                    pnl_pct,
                    pnl_eur,
                    pd.Timestamp(date_open).date(),
                    ticker,
                    OPEN,
                ],
            )


# --------------------------------------------------- exit-rule (mirrors backtest) ----


def _resolve_paper_exit(
    df: pd.DataFrame,
    entry_idx: int,
    horizon_days: int,
    stop: float,
    target: float,
    *,
    short: bool = False,
) -> tuple[int, float, str] | None:
    """Mirror ``backtest.engine._resolve_exit`` but return ``None`` when not enough bars.

    Same conservative tie-break: if both stop and target are touched inside the
    same bar, the stop wins (we'd rather assume the worst). When the cache hasn't
    accumulated enough bars to reach the time barrier yet, the trade stays open.

    Direction-aware: a long stops out when ``low`` breaks below ``stop`` and takes
    profit when ``high`` breaks above ``target``; a short is the mirror — stop is
    *above* the entry (touched by ``high``) and target *below* (touched by ``low``).
    """
    n = len(df)
    time_exit_idx = entry_idx + horizon_days
    last_walkable = min(time_exit_idx, n - 1)
    if last_walkable <= entry_idx:
        return None  # trade only just opened; nothing to walk yet

    high = df["high"]
    low = df["low"]
    close = df["close"]

    for j in range(entry_idx + 1, last_walkable + 1):
        if short:
            hit_stop = float(high.iloc[j]) >= stop
            hit_target = float(low.iloc[j]) <= target
        else:
            hit_stop = float(low.iloc[j]) <= stop
            hit_target = float(high.iloc[j]) >= target
        if hit_stop:  # stop wins the same-bar tie (conservative)
            return j, stop, CLOSED_STOP
        if hit_target:
            return j, target, CLOSED_TARGET

    if last_walkable >= time_exit_idx:
        return time_exit_idx, float(close.iloc[time_exit_idx]), CLOSED_TIME
    return None


# --------------------------------------------------------- top-level operations ----


def open_new_trades(
    config: Config,
    store: OhlcvStore,
    signal_store: SignalStore,
) -> int:
    """Open a paper trade for each BUY signal on the latest signal date.

    Already-open trades for the same ``(date_open, ticker)`` are left alone — this
    is the idempotency guarantee that makes the scheduler safe to re-run.
    """
    del store  # OhlcvStore not needed at open time; entry came from the signal
    latest = signal_store.latest()
    if latest.empty:
        return 0
    # Open both directions: a LONG (or legacy BUY) opens a long leg, a SHORT opens a short
    # leg. Each row's ``signal`` records the direction; the stop/target the signal service
    # emitted are already mirrored for shorts (stop above entry, target below).
    actionable = latest[latest["signal"].isin([*LONG_OPEN, *SHORT_OPEN])]
    actionable = actionable[actionable["size_shares"] > 0]
    # Only paper-trade PROMOTED assets — those whose per-asset model cleared the guard. The
    # book must measure the validated strategy, not advisory (optimized-but-unproven) signals.
    # Legacy rows predating the column have promoted=NULL; treat missing as not-promoted.
    if "promoted" in actionable.columns:
        actionable = actionable[actionable["promoted"].fillna(value=False).astype(bool)]
    else:
        actionable = actionable.iloc[0:0]  # no column yet => nothing known-promoted => open nothing
    if actionable.empty:
        return 0
    rows = pd.DataFrame(
        {
            "date_open": actionable["date"],
            "ticker": actionable["ticker"],
            "signal": actionable["signal"],
            "entry": actionable["entry"],
            "stop": actionable["stop_loss"],
            "target": actionable["take_profit"],
            "size_shares": actionable["size_shares"],
            "status": OPEN,
        }
    )
    paper = PaperStore(config.db_path)
    opened = paper.insert_new(rows)
    logger.info("paper.open_new_trades: %d new trades opened", opened)
    return opened


def update_open_trades(
    config: Config,
    store: OhlcvStore,
    label_cfg: _LabelConfig | None = None,
) -> int:
    """Walk every open paper trade forward against the cache; close those that hit.

    Idempotent: trades that haven't seen a stop/target/time exit yet stay open and
    are re-evaluated on the next call. Closed trades are immutable.
    """
    cfg = label_cfg or _LabelConfig(**config.labeling.model_dump())
    paper = PaperStore(config.db_path)
    open_df = paper.open_trades()
    if open_df.empty:
        return 0

    closed_count = 0
    for _, row in open_df.iterrows():
        ticker = str(row["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            logger.warning("paper.update: no OHLCV for %s, leaving trade open", ticker)
            continue
        date_open = _ts(row["date_open"])
        # Locate the entry bar; tolerate cache that doesn't yet have date_open.
        if date_open not in ohlcv.index:
            logger.warning(
                "paper.update: entry date %s missing for %s, leaving trade open",
                date_open.date(),
                ticker,
            )
            continue
        entry_idx = int(ohlcv.index.get_loc(date_open))
        short = _is_short(row["signal"])
        resolved = _resolve_paper_exit(
            ohlcv,
            entry_idx=entry_idx,
            horizon_days=cfg.horizon_days,
            stop=float(row["stop"]),
            target=float(row["target"]),
            short=short,
        )
        if resolved is None:
            continue
        exit_idx, exit_price, status = resolved
        entry = float(row["entry"])
        shares = int(row["size_shares"])
        pnl_pct, pnl_eur = _direction_pnl(entry, float(exit_price), shares, short=short)
        paper.close_trade(
            date_open=date_open,
            ticker=ticker,
            date_close=ohlcv.index[exit_idx],
            exit_price=float(exit_price),
            status=status,
            pnl_pct=float(pnl_pct),
            pnl_eur=float(pnl_eur),
        )
        closed_count += 1
    logger.info("paper.update_open_trades: %d trades closed", closed_count)
    return closed_count


def get_open_positions(config: Config, store: OhlcvStore) -> list[OpenPosition]:
    """Return open paper trades enriched with the latest-close MTM."""
    paper = PaperStore(config.db_path)
    df = paper.open_trades()
    out: list[OpenPosition] = []
    today = _ts(pd.Timestamp.today()).normalize()
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            continue
        current_price = float(ohlcv["close"].iloc[-1])
        entry = float(row["entry"])
        shares = int(row["size_shares"])
        short = _is_short(row["signal"])
        date_open = _ts(row["date_open"])
        days_held = max(0, int(np.busday_count(date_open.date(), today.date())))
        mtm_pct, mtm_eur = _direction_pnl(entry, current_price, shares, short=short)
        out.append(
            OpenPosition(
                date_open=date_open,
                ticker=ticker,
                direction="short" if short else "long",
                entry=entry,
                stop=float(row["stop"]),
                target=float(row["target"]),
                size_shares=shares,
                current_price=current_price,
                days_held=days_held,
                mtm_pct=mtm_pct,
                mtm_eur=mtm_eur,
            )
        )
    return out


# ---------------------------------------------------------------- equity curve ----


def get_equity_curve(config: Config, store: OhlcvStore) -> pd.DataFrame:
    """Build a daily paper-equity series alongside an equal-capital SPY buy & hold.

    Equity = ``capital + cumulative realized P&L + sum of open-trade MTM``. SPY
    benchmark is the same starting capital held in SPY from the first paper-trade
    date onward. Returns an empty frame if no paper trades have ever been opened.
    """
    paper = PaperStore(config.db_path)
    trades = paper.all_trades()
    if trades.empty:
        return pd.DataFrame(columns=pd.Index(["date", "equity_paper", "equity_spy"]))

    capital = float(config.signals.capital)
    trades["date_open"] = pd.to_datetime(trades["date_open"])
    trades["date_close"] = pd.to_datetime(trades["date_close"])

    start = _ts(trades["date_open"].min()).normalize()
    today = _ts(pd.Timestamp.today()).normalize()
    end = max(today, _ts(trades["date_open"].max()).normalize())
    max_close = trades["date_close"].max()
    if pd.notna(max_close):
        end = max(end, _ts(max_close).normalize())
    dates = pd.bdate_range(start, end)

    # Cumulative realized P&L: at every date d, sum pnl_eur of trades closed <= d.
    cum_realized = pd.Series(0.0, index=dates)
    closed = trades[trades["status"].isin(CLOSED_STATUSES)]
    if not closed.empty:
        by_close = closed.groupby("date_close")["pnl_eur"].sum().sort_index()
        for d_close, pnl in by_close.items():
            cum_realized.loc[cum_realized.index >= _ts(d_close)] += float(pnl)

    # Unrealized P&L: per open or pre-close trade, MTM = (close[d] - entry) * shares
    # for d in [date_open, date_close) (open trades go up to ``today``).
    unrealized = pd.Series(0.0, index=dates)
    for _, row in trades.iterrows():
        ticker_df = store.load(str(row["ticker"]))
        if ticker_df is None or ticker_df.empty:
            continue
        d_open = _ts(row["date_open"]).normalize()
        is_closed = row["status"] in CLOSED_STATUSES
        d_close = _ts(row["date_close"]).normalize() if is_closed else today + pd.Timedelta(days=1)
        mask = (dates >= d_open) & (dates < d_close)
        active = dates[mask]
        if len(active) == 0:
            continue
        closes = ticker_df["close"].reindex(active).ffill()
        entry_px = float(row["entry"])
        move = (entry_px - closes) if _is_short(row["signal"]) else (closes - entry_px)
        mtm = move * int(row["size_shares"])
        unrealized.loc[active] += mtm.fillna(0.0).to_numpy()

    equity_paper = capital + cum_realized + unrealized

    spy_df = store.load(MARKET_TICKER)
    if spy_df is not None and not spy_df.empty:
        spy_series = spy_df["close"].reindex(dates).ffill()
        # Anchor the benchmark at the first available SPY close at/before start.
        spy_anchor_series = spy_df["close"].loc[:start]
        if spy_anchor_series.empty:
            equity_spy = pd.Series(np.nan, index=dates)
        else:
            anchor = float(spy_anchor_series.iloc[-1])
            equity_spy = capital * spy_series / anchor
    else:
        equity_spy = pd.Series(np.nan, index=dates)

    return pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "equity_paper": equity_paper.to_numpy(),
            "equity_spy": equity_spy.to_numpy(),
        }
    )


# ------------------------------------------------------ summary metrics for UI ----


def get_paper_metrics(config: Config, store: OhlcvStore) -> dict[str, float | int]:
    """Compact summary used by the dashboard card: returns, win rate, drawdowns, n."""
    paper = PaperStore(config.db_path)
    trades = paper.all_trades()
    capital = float(config.signals.capital)
    equity = get_equity_curve(config, store)

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
            "total_return_spy": 0.0,
            "max_drawdown_paper": 0.0,
            "capital": capital,
        }

    eq_paper = pd.Series(equity["equity_paper"].to_numpy())
    eq_spy = pd.Series(equity["equity_spy"].to_numpy())
    total_return_paper = float(eq_paper.iloc[-1] / capital - 1.0)
    total_return_spy = (
        float(eq_spy.dropna().iloc[-1] / capital - 1.0) if eq_spy.dropna().size else float("nan")
    )
    return {
        "n_open": n_open,
        "n_closed": n_closed,
        "win_rate": win_rate,
        "total_return_paper": total_return_paper,
        "total_return_spy": total_return_spy,
        "max_drawdown_paper": float(max_drawdown(eq_paper / capital)),
        "capital": capital,
    }


__all__ = [
    "CLOSED_STATUSES",
    "CLOSED_STOP",
    "CLOSED_TARGET",
    "CLOSED_TIME",
    "OPEN",
    "OpenPosition",
    "PaperStore",
    "get_equity_curve",
    "get_open_positions",
    "get_paper_metrics",
    "open_new_trades",
    "update_open_trades",
]
