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
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import duckdb
import numpy as np
import pandas as pd

from berich.backtest.engine import BacktestConfig as _BacktestConfig
from berich.backtest.engine import estimated_cost_bps
from berich.backtest.metrics import max_drawdown
from berich.features.build import MARKET_TICKER
from berich.features.indicators import atr
from berich.labeling.triple_barrier import LabelConfig as _LabelConfig

if TYPE_CHECKING:
    from collections.abc import Callable
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
CLOSED_TRAIL = "closed_trail"  # ratcheted (armed) trailing stop hit — distinct from a fixed stop
CLOSED_STATUSES: tuple[str, ...] = (CLOSED_STOP, CLOSED_TARGET, CLOSED_TIME, CLOSED_TRAIL)

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
    date_open     DATE    NOT NULL,
    ticker        VARCHAR NOT NULL,
    signal        VARCHAR NOT NULL,
    entry         DOUBLE  NOT NULL,
    stop          DOUBLE  NOT NULL,
    target        DOUBLE  NOT NULL,
    size_shares   BIGINT  NOT NULL,
    status        VARCHAR NOT NULL DEFAULT 'open',
    date_close    DATE,
    exit_price    DOUBLE,
    pnl_pct       DOUBLE,
    pnl_eur       DOUBLE,
    exit_strategy VARCHAR NOT NULL DEFAULT 'fixed',
    tier          VARCHAR NOT NULL DEFAULT 'promoted',
    cost_bps      DOUBLE,
    created_at    TIMESTAMP DEFAULT now(),
    updated_at    TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date_open, ticker, exit_strategy)
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

# Exit strategy of the trade (so the daily walk applies the right rule): "fixed" keeps the
# historical TP/SL barrier; "trailing" / "trailing_tp" drive the ratcheting-stop walk. Like the
# ``source`` migration this is a safe no-op on fresh tables and only ALTERs pre-existing ones;
# the default keeps every legacy row on the fixed rule.
_MIGRATION_EXIT_STRATEGY_COLUMN = (
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_strategy VARCHAR DEFAULT 'fixed';"
)

# Trust tier of the trade: "promoted" trades make up the committed-capital book; "observe" trades
# are a paper-only shadow book (no capital engaged) tracking near-miss models live. A safe no-op on
# fresh tables; the default keeps every legacy row in the committed book (all prior trades were
# promoted-only).
_MIGRATION_TIER_COLUMN = (
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS tier VARCHAR DEFAULT 'promoted';"
)

# Estimated round-trip friction (commissions + volume-proportional slippage, bps) frozen at open.
# Charged against the realized P&L when the trade closes so the forward test pays the same costs
# the promotion gate's backtest assumed. NULL on legacy rows = closed at zero cost (history is
# never rewritten).
_MIGRATION_COST_COLUMN = "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS cost_bps DOUBLE;"

PROMOTED_TIER = "promoted"
OBSERVE_TIER = "observe"

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
    "exit_strategy",
    "tier",
    "cost_bps",
)


def _upgrade_paper_pk(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuild a legacy ``(date_open, ticker)`` PK as ``(date_open, ticker, exit_strategy)``.

    Lets the fixed and trailing books hold a trade on the same asset/day. Preserves every row
    (legacy rows carry ``exit_strategy='fixed'``); no-op once the key already includes it.
    """
    pk = con.execute(
        "SELECT constraint_column_names FROM duckdb_constraints() "
        "WHERE table_name = 'paper_trades' AND constraint_type = 'PRIMARY KEY'"
    ).fetchall()
    if not pk or "exit_strategy" in list(pk[0][0]):
        return
    old_cols = [r[0] for r in con.execute("DESCRIBE paper_trades").fetchall()]
    con.execute("ALTER TABLE paper_trades RENAME TO _paper_trades_old")
    con.execute(_SCHEMA)
    con.execute(_MIGRATION_SOURCE_COLUMN)
    con.execute(_MIGRATION_EXIT_STRATEGY_COLUMN)
    con.execute(_MIGRATION_TIER_COLUMN)
    con.execute(_MIGRATION_COST_COLUMN)
    new_cols = {r[0] for r in con.execute("DESCRIBE paper_trades").fetchall()}
    common = ", ".join(c for c in old_cols if c in new_cols)
    con.execute(
        f"INSERT INTO paper_trades ({common}) "  # noqa: S608 — identifiers from DESCRIBE, not user input
        f"SELECT {common} FROM _paper_trades_old"
    )
    con.execute("DROP TABLE _paper_trades_old")


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
    exit_strategy: str = "fixed"
    # Live ratcheting stop for an open trailing trade (the effective stop after the high-water
    # mark since entry); None for fixed trades, where ``stop`` is already the effective level.
    trail_stop: float | None = None
    # Pending exit: the daily job closes positions only at 22:30 on weekdays, so a trade whose
    # stop/target/time barrier was already breached in the cached bars sits OPEN (and its live
    # MTM at the current price wildly overstates the outcome) until the next run. These fields
    # carry that resolved-but-not-yet-booked exit so the UI can flag it and show the REAL P&L.
    # "closed_stop" | "closed_target" | "closed_time" | "closed_trail" when already resolved.
    pending_exit: str | None = None
    pending_exit_price: float | None = None
    # Realized P&L fraction at that exit (the capped/locked one — what the close will book).
    pending_exit_pct: float | None = None

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
            "exit_strategy": self.exit_strategy,
            "trail_stop": self.trail_stop,
            "pending_exit": self.pending_exit,
            "pending_exit_price": self.pending_exit_price,
            "pending_exit_pct": self.pending_exit_pct,
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
            con.execute(_MIGRATION_EXIT_STRATEGY_COLUMN)
            con.execute(_MIGRATION_TIER_COLUMN)
            con.execute(_MIGRATION_COST_COLUMN)
            _upgrade_paper_pk(con)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def all_trades(
        self, exit_strategy: str | None = None, *, tier: str | None = None
    ) -> pd.DataFrame:
        sql = "SELECT * FROM paper_trades"
        clauses: list[str] = []
        params: list[object] = []
        if exit_strategy is not None:
            clauses.append("exit_strategy = ?")
            params.append(exit_strategy)
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY date_open, ticker"
        with self._connect() as con:
            return con.execute(sql, params).df()

    def open_trades(self, *, tier: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM paper_trades WHERE status = ?"
        params: list[object] = [OPEN]
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier)
        sql += " ORDER BY date_open, ticker"
        with self._connect() as con:
            return con.execute(sql, params).df()

    def closed_trades(
        self, limit: int | None = None, exit_strategy: str | None = None, *, tier: str | None = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM paper_trades WHERE status <> ?"
        params: list[object] = [OPEN]
        if exit_strategy is not None:
            query += " AND exit_strategy = ?"
            params.append(exit_strategy)
        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)
        query += " ORDER BY date_close DESC, ticker"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self._connect() as con:
            return con.execute(query, params).df()

    def insert_new(self, rows: pd.DataFrame) -> int:
        """Insert new (date_open, ticker, exit_strategy) rows; skip those already present.

        Idempotent: re-running on the same input opens nothing the second time. Keying on the
        exit strategy too lets the fixed and trailing books hold a trade on the same asset/day.
        """
        if rows.empty:
            return 0
        rows = rows.copy()
        for col in _TRADE_COLUMNS:
            if col not in rows.columns:
                rows[col] = None
        rows = rows[list(_TRADE_COLUMNS)]
        rows["date_open"] = pd.to_datetime(rows["date_open"]).dt.date
        # exit_strategy is part of the PK and NOT NULL — coerce missing/NaN to the fixed default
        # so legacy callers that don't set it still open a (fixed-book) trade.
        rows["exit_strategy"] = rows["exit_strategy"].fillna("fixed")
        # tier is NOT NULL — a caller that doesn't set it opens into the committed (promoted) book.
        rows["tier"] = rows["tier"].fillna(PROMOTED_TIER)
        # _TRADE_COLUMNS is a module-level constant of identifier strings, never user
        # input — safe to interpolate into SQL.
        cols_csv = ", ".join(_TRADE_COLUMNS)
        insert_sql = (
            f"INSERT INTO paper_trades ({cols_csv}) "  # noqa: S608 — see note above
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
        with self._connect() as con:
            con.execute(
                "UPDATE paper_trades SET "
                "status = ?, date_close = ?, exit_price = ?, "
                "pnl_pct = ?, pnl_eur = ?, updated_at = now() "
                "WHERE date_open = ? AND ticker = ? AND exit_strategy = ? AND status = ?",
                [
                    status,
                    pd.Timestamp(date_close).date(),
                    exit_price,
                    pnl_pct,
                    pnl_eur,
                    pd.Timestamp(date_open).date(),
                    ticker,
                    exit_strategy,
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


def _trail_scalars(
    df: pd.DataFrame, entry_idx: int, cfg: _LabelConfig, strategy: str, *, short: bool
) -> tuple[float, float, float] | None:
    """Return ``(entry_ref, trail_dist, activation_level)`` for a trailing trade, or ``None``.

    The trail distance is the entry-frozen ATR (the per-strategy multiple at the entry bar — wide
    ``trailing_atr`` for pure trailing, tight ``trailing_tp_atr`` for trailing_tp) and the
    activation sits ``trailing_activation_atr``*ATR on the favorable side of the entry close —
    the same construction the backtest/label use. ``None`` when ATR isn't warmed at the entry
    bar (caller then falls back to the fixed stop).
    """
    a = float(atr(df["high"], df["low"], df["close"], cfg.atr_window).iloc[entry_idx])
    if pd.isna(a):
        return None
    entry_ref = float(df["close"].iloc[entry_idx])
    trail_mult = cfg.trailing_tp_atr if strategy == "trailing_tp" else cfg.trailing_atr
    trail_dist = trail_mult * a
    activation_level = (
        entry_ref - cfg.trailing_activation_atr * a
        if short
        else entry_ref + cfg.trailing_activation_atr * a
    )
    return entry_ref, trail_dist, activation_level


def _resolve_trailing_exit(  # noqa: C901, PLR0912 — the bar-by-bar ratchet is irreducibly branchy
    df: pd.DataFrame,
    entry_idx: int,
    horizon_days: int,
    init_stop: float,
    target: float | None,
    *,
    short: bool,
    entry_ref: float,
    trail_dist: float,
    activation_level: float,
) -> tuple[int, float, str] | None:
    """Trailing-stop walk mirroring ``backtest.engine._resolve_exit_trailing``.

    Causal ratcheting stop (the bar never sets-and-triggers its own stop); arms once the
    favorable extreme passes ``activation_level``, then reports ``CLOSED_TRAIL`` (vs the pre-arm
    ``CLOSED_STOP``). ``target=None`` means no TP cap (pure trailing). Returns ``None`` when the
    cache hasn't reached the time barrier and nothing has triggered yet (trade stays open).
    """
    n = len(df)
    time_exit_idx = entry_idx + horizon_days
    last_walkable = min(time_exit_idx, n - 1)
    if last_walkable <= entry_idx:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]
    running_ext = entry_ref
    cur_stop = init_stop
    armed = False
    for j in range(entry_idx + 1, last_walkable + 1):
        if short:
            hit_stop = float(high.iloc[j]) >= cur_stop
            hit_target = target is not None and float(low.iloc[j]) <= target
        else:
            hit_stop = float(low.iloc[j]) <= cur_stop
            hit_target = target is not None and float(high.iloc[j]) >= target
        if hit_stop:
            return j, cur_stop, (CLOSED_TRAIL if armed else CLOSED_STOP)
        if hit_target:
            return j, float(target), CLOSED_TARGET
        if short:
            running_ext = min(running_ext, float(low.iloc[j]))
            if running_ext <= activation_level:
                armed = True
            if armed:
                cur_stop = min(cur_stop, running_ext + trail_dist)
        else:
            running_ext = max(running_ext, float(high.iloc[j]))
            if running_ext >= activation_level:
                armed = True
            if armed:
                cur_stop = max(cur_stop, running_ext - trail_dist)

    if last_walkable >= time_exit_idx:
        return time_exit_idx, float(close.iloc[time_exit_idx]), CLOSED_TIME
    return None


def _current_trail_stop(
    df: pd.DataFrame,
    entry_idx: int,
    init_stop: float,
    *,
    short: bool,
    trail_dist: float,
    activation_level: float,
) -> float:
    """Live ratcheting stop for an open trailing trade: walk every bar since entry to the latest
    close and return the effective stop (the high-water-mark stop once armed, else the initial).
    """
    high = df["high"]
    low = df["low"]
    entry_ref = float(df["close"].iloc[entry_idx])
    running_ext = entry_ref
    cur_stop = init_stop
    armed = False
    for j in range(entry_idx + 1, len(df)):
        if short:
            running_ext = min(running_ext, float(low.iloc[j]))
            if running_ext <= activation_level:
                armed = True
            if armed:
                cur_stop = min(cur_stop, running_ext + trail_dist)
        else:
            running_ext = max(running_ext, float(high.iloc[j]))
            if running_ext >= activation_level:
                armed = True
            if armed:
                cur_stop = max(cur_stop, running_ext - trail_dist)
    return cur_stop


def _resolve_trade_exit(
    df: pd.DataFrame,
    row: pd.Series,
    entry_idx: int,
    cfg: _LabelConfig,
    *,
    short: bool,
    strategy: str,
) -> tuple[int, float, str] | None:
    """Dispatch one open trade to its exit walk: fixed barrier or ratcheting trailing stop.

    Trailing falls back to the fixed stop when ATR isn't warmed at the entry bar (degenerate);
    ``trailing`` drops the TP cap, ``trailing_tp`` keeps the stored target as the cap.
    """
    stop = float(row["stop"])
    target = float(row["target"])
    if strategy == "fixed":
        return _resolve_paper_exit(
            df,
            entry_idx=entry_idx,
            horizon_days=cfg.horizon_days,
            stop=stop,
            target=target,
            short=short,
        )
    scal = _trail_scalars(df, entry_idx, cfg, strategy, short=short)
    if scal is None:
        return _resolve_paper_exit(
            df,
            entry_idx=entry_idx,
            horizon_days=cfg.horizon_days,
            stop=stop,
            target=target,
            short=short,
        )
    entry_ref, trail_dist, activation_level = scal
    return _resolve_trailing_exit(
        df,
        entry_idx,
        cfg.horizon_days,
        stop,
        target if strategy == "trailing_tp" else None,
        short=short,
        entry_ref=entry_ref,
        trail_dist=trail_dist,
        activation_level=activation_level,
    )


# --------------------------------------------------------- top-level operations ----


# A candidate shrunk by the caps to below this fraction of capital is dropped, not opened: a
# sub-floor sliver of leftover budget (e.g. a 0.57 € / 1-share forex position) is not a real trade
# — it pollutes the trade count, the win rate and the per-segment expectancy the forward-test
# decision reads, while engaging no meaningful capital.
MIN_POSITION_NOTIONAL_PCT = 0.005


def _apply_exposure_caps(
    rows: pd.DataFrame,
    open_df: pd.DataFrame,
    *,
    capital: float,
    max_ticker_pct: float,
    max_book_pct: float,
    max_class_pct: float = 1.0,
    class_of: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Scale down / drop candidate trades so open notional stays within the exposure budget.

    Pyramiding is allowed, but bounded: exposure is measured at cost basis (``entry * shares``)
    and capped per name (``capital * max_ticker_pct``), across the whole book
    (``capital * max_book_pct``), and — when ``class_of`` is given — per asset class
    (``capital * max_class_pct``), so the book can't pile into one correlated bucket (e.g. several
    USD pairs, or all tech) when many signals fire at once. Caps are enforced PER exit-strategy
    book — each book gets its own budget, consumed first by already-open trades, then by new
    candidates in row order. A candidate that breaches a cap is shrunk to the remaining budget;
    one that can't fit at least ``MIN_POSITION_NOTIONAL_PCT`` of capital is dropped (no dust
    positions). Returns the surviving rows with ``size_shares`` adjusted.
    """
    if rows.empty:
        return rows
    cap_ticker = capital * max_ticker_pct
    cap_book = capital * max_book_pct
    cap_class = capital * max_class_pct
    book_used: dict[str, float] = {}
    ticker_used: dict[tuple[str, str], float] = {}
    class_used: dict[tuple[str, str], float] = {}

    def _cls(ticker: str) -> str:
        return class_of(ticker) if class_of is not None else ""

    if not open_df.empty:
        has_strat = "exit_strategy" in open_df.columns
        for _, orow in open_df.iterrows():
            strat = (
                str(orow["exit_strategy"])
                if has_strat and pd.notna(orow["exit_strategy"])
                else "fixed"
            )
            tkr = str(orow["ticker"])
            notional = float(orow["entry"]) * float(orow["size_shares"])
            ticker_used[(strat, tkr)] = ticker_used.get((strat, tkr), 0.0) + notional
            book_used[strat] = book_used.get(strat, 0.0) + notional
            class_used[(strat, _cls(tkr))] = class_used.get((strat, _cls(tkr)), 0.0) + notional

    kept_idx: list[object] = []
    new_shares: list[int] = []
    for idx, row in rows.iterrows():
        strat = str(row["exit_strategy"]) if pd.notna(row.get("exit_strategy")) else "fixed"
        tkr = str(row["ticker"])
        cls = _cls(tkr)
        entry = float(row["entry"])
        want = int(row["size_shares"])
        if entry <= 0 or want <= 0:
            continue
        budget = min(
            want * entry,
            cap_ticker - ticker_used.get((strat, tkr), 0.0),
            cap_book - book_used.get(strat, 0.0),
            cap_class - class_used.get((strat, cls), 0.0),
        )
        fit = min(want, math.floor(budget / entry)) if budget > 0 else 0
        added = fit * entry
        # Drop dust: when only a sliver of budget is left, a 1-share / sub-floor position is not a
        # real trade. The cap is doing its job — this name simply doesn't fit — so skip it rather
        # than open a token position that pollutes the stats.
        if fit <= 0 or added < capital * MIN_POSITION_NOTIONAL_PCT:
            continue
        ticker_used[(strat, tkr)] = ticker_used.get((strat, tkr), 0.0) + added
        book_used[strat] = book_used.get(strat, 0.0) + added
        class_used[(strat, cls)] = class_used.get((strat, cls), 0.0) + added
        kept_idx.append(idx)
        new_shares.append(fit)

    out = rows.loc[kept_idx].copy()
    out["size_shares"] = new_shares
    return out


def open_new_trades(
    config: Config,
    store: OhlcvStore,
    signal_store: SignalStore,
) -> int:
    """Open a paper trade for each BUY signal on the latest signal date.

    Already-open trades for the same ``(date_open, ticker)`` are left alone — this is the
    idempotency guarantee that makes the scheduler safe to re-run. Candidates pass through the
    money-management guardrails before opening: a graduated drawdown kill-switch and a
    max-concurrent-positions cap on the committed book (Phase 2 — protect real capital), then
    ``_apply_exposure_caps`` (per-name, per-book and per-asset-class concentration limits).
    """
    latest = signal_store.latest()
    if latest.empty:
        return 0
    # Open both directions: a LONG (or legacy BUY) opens a long leg, a SHORT opens a short
    # leg. Each row's ``signal`` records the direction; the stop/target the signal service
    # emitted are already mirrored for shorts (stop above entry, target below).
    actionable = latest[latest["signal"].isin([*LONG_OPEN, *SHORT_OPEN])]
    actionable = actionable[actionable["size_shares"] > 0]
    if actionable.empty:
        return 0

    # Route by trust tier into two independent books, each with its OWN capital budget:
    #   - committed book: PROMOTED models (cleared the guard) — this is the real-capital track.
    #   - observation book: OBSERVE models (near-miss, paper-only) — tracks forward evidence with
    #     NO capital engaged, so it never competes with or pollutes the committed book.
    # Advisory rows are never opened. ``tier`` is authoritative; we fall back to the legacy
    # ``promoted`` flag for rows written before the tier column existed.
    tier = _signal_tiers(actionable)
    paper = PaperStore(config.db_path)
    opened = 0
    for book_tier in (PROMOTED_TIER, OBSERVE_TIER):
        rows = _plan_book(config, store, paper, actionable[tier == book_tier], book_tier)
        if not rows.empty:
            opened += paper.insert_new(rows)
    logger.info("paper.open_new_trades: %d new trades opened", opened)
    return opened


def _plan_book(
    config: Config,
    store: OhlcvStore,
    paper: PaperStore,
    book: pd.DataFrame,
    book_tier: str,
) -> pd.DataFrame:
    """Candidate trades that WOULD open for one tier after all money-management guardrails.

    The single source of truth shared by ``open_new_trades`` (which then inserts) and
    ``plan_committed_opens`` (which returns it for the Brief's order sheet), so the planned orders
    shown to the user are exactly what the book opens. Empty frame when nothing survives.
    """
    if book.empty:
        return book
    sig = config.signals
    rows = _candidate_rows(book, book_tier)
    # Same money-management philosophy on both books — drawdown kill-switch + position cap — each
    # measured on its own tier (own equity, own open positions), so the diversified panel is a
    # genuinely managed book, not an unbounded shadow. The committed book is unchanged (tier
    # defaults to promoted).
    rows = _derisk_for_drawdown(rows, config, store, book_tier)
    rows = _cap_open_positions(rows, paper.open_trades(tier=book_tier), sig.max_open_positions)
    if rows.empty:
        return rows
    rows = _apply_exposure_caps(
        rows,
        paper.open_trades(tier=book_tier),
        capital=float(sig.capital),
        max_ticker_pct=float(sig.max_ticker_exposure_pct),
        max_book_pct=float(sig.max_book_exposure_pct),
        max_class_pct=float(sig.max_class_exposure_pct),
        class_of=config.asset_class_for,
    )
    if not rows.empty:
        # Freeze the gate's round-trip friction estimate at open; the close charges it
        # against realized P&L so the forward test isn't cheaper than the backtest was.
        rows = rows.copy()
        rows["cost_bps"] = [_ticker_cost_bps(store, str(t)) for t in rows["ticker"]]
    return rows


def plan_committed_opens(
    config: Config, store: OhlcvStore, signal_store: SignalStore
) -> pd.DataFrame:
    """Dry-run of the committed book's opens today — what ``open_new_trades`` WOULD open, not raw
    signals. Sizes are already scaled to the per-name / per-book / per-class caps and the drawdown
    kill-switch, and they account for positions already open (which consume budget). This is the
    portfolio-coherent order sheet the Brief shows, so its sizes sum to a real allocation rather
    than one full-size order per signal. Returns rows with ``entry/stop/target/size_shares/...``.
    """
    latest = signal_store.latest()
    if latest.empty:
        return latest
    actionable = latest[latest["signal"].isin([*LONG_OPEN, *SHORT_OPEN])]
    actionable = actionable[actionable["size_shares"] > 0]
    if actionable.empty:
        return actionable
    tier = _signal_tiers(actionable)
    paper = PaperStore(config.db_path)
    return _plan_book(config, store, paper, actionable[tier == PROMOTED_TIER], PROMOTED_TIER)


# A single currency carrying more than this share of the book's capital is one correlated bet
# (e.g. three open JPY crosses) that the per-class caps can't see — surfaced, never blocked.
CURRENCY_CONCENTRATION_WARN_PCT = 0.5


def currency_concentration(config: Config, *, tier: str = PROMOTED_TIER) -> list[dict[str, object]]:
    """Aggregate open forex exposure per currency (cost basis) — observability only.

    The exposure caps are per asset CLASS, so several crosses sharing a leg (EURJPY + GBPJPY +
    AUDJPY) look diversified to them while being one JPY bet. Each open ``XXXYYY=X`` pair
    contributes its notional to BOTH legs' buckets; the dashboard and the daily digest flag any
    currency above ``CURRENCY_CONCENTRATION_WARN_PCT`` of capital. No sizing is changed here
    (the forward test is frozen).
    """
    open_df = PaperStore(config.db_path).open_trades(tier=tier)
    if open_df.empty:
        return []
    capital = float(config.signals.capital)
    buckets: dict[str, dict[str, float]] = {}
    for _, row in open_df.iterrows():
        ticker = str(row["ticker"]).upper()
        if not (ticker.endswith("=X") and len(ticker) == 8):  # noqa: PLR2004 — XXXYYY=X
            continue
        notional = float(row["entry"]) * float(row["size_shares"])
        for ccy in (ticker[:3], ticker[3:6]):
            bucket = buckets.setdefault(ccy, {"notional": 0.0, "n": 0.0})
            bucket["notional"] += notional
            bucket["n"] += 1
    out: list[dict[str, object]] = [
        {
            "currency": ccy,
            "notional": round(bucket["notional"], 2),
            "pct_capital": round(bucket["notional"] / capital, 4) if capital else 0.0,
            "n_positions": int(bucket["n"]),
        }
        for ccy, bucket in buckets.items()
    ]
    out.sort(key=lambda r: -float(r["notional"]))  # type: ignore[arg-type]
    return out


def _ticker_cost_bps(store: OhlcvStore, ticker: str) -> float:
    """Round-trip friction (bps) for one ticker, on the same cost model as the gate's backtest."""
    cfg = _BacktestConfig()
    df = store.load(ticker)
    if df is None or df.empty or "volume" not in df.columns:
        return 2.0 * (cfg.fee_bps + cfg.slippage_bps)
    return float(estimated_cost_bps(df, cfg))


def _book_drawdown(config: Config, store: OhlcvStore, tier: str = PROMOTED_TIER) -> float:
    """Current peak-to-now drawdown (>= 0) of one tier's paper book, 0.0 with no history.

    Each book is measured on its own equity curve, so the committed and the observe (diversified
    panel) books de-risk off their own drawdown, not a shared one.
    """
    eq = get_equity_curve(config, store, tier=tier)
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


def _derisk_for_drawdown(
    rows: pd.DataFrame, config: Config, store: OhlcvStore, tier: str = PROMOTED_TIER
) -> pd.DataFrame:
    """Scale (or zero) candidate sizes per the graduated drawdown kill-switch on ``tier``'s book.

    Full size below the de-risk threshold; ``drawdown_derisk_factor`` between de-risk and halt;
    nothing (empty frame) at or above the halt threshold. Sizes are floored to whole shares and
    rows that shrink below one share are dropped. Applied to both the committed and the observe
    (diversified panel) books, each off its own drawdown.
    """
    if rows.empty:
        return rows
    sig = config.signals
    dd = _book_drawdown(config, store, tier)
    if dd >= sig.drawdown_halt_threshold:
        logger.info(
            "paper.open_new_trades: %s drawdown %.1f%% >= halt — no new trades", tier, dd * 100
        )
        return rows.iloc[0:0]
    if dd < sig.drawdown_derisk_threshold:
        return rows
    factor = sig.drawdown_derisk_factor
    out = rows.copy()
    out["size_shares"] = (out["size_shares"].astype(int) * factor).apply(math.floor).astype(int)
    return out[out["size_shares"] > 0]


def _cap_open_positions(rows: pd.DataFrame, open_df: pd.DataFrame, max_open: int) -> pd.DataFrame:
    """Keep at most ``max_open`` concurrent committed positions (already-open count toward it).

    ``max_open <= 0`` disables the cap. New candidates are kept in row order (highest-proba first,
    as ``signal_store.latest`` sorts), so the strongest signals get the remaining slots.
    """
    if max_open <= 0 or rows.empty:
        return rows
    already = 0 if open_df.empty else len(open_df)
    remaining = max_open - already
    if remaining <= 0:
        return rows.iloc[0:0]
    return rows.iloc[:remaining]


def _signal_tiers(actionable: pd.DataFrame) -> pd.Series:
    """Route each signal row: committed (``promoted``), observation (``observe``), or dropped.

    The authoritative ``promoted`` guard flag always wins — a promoted signal goes to the
    committed book regardless of the ``tier`` hint (so the real-capital book is unchanged from
    before). Among non-promoted rows, an explicit ``tier == "observe"`` routes to the shadow book;
    everything else is advisory and never opened.
    """
    n = len(actionable)
    index = actionable.index
    promoted = (
        actionable["promoted"].fillna(value=False).astype(bool)
        if "promoted" in actionable.columns
        else pd.Series([False] * n, index=index)
    )
    tier_col = (
        actionable["tier"].fillna("")
        if "tier" in actionable.columns
        else pd.Series([""] * n, index=index)
    )
    routed = [
        PROMOTED_TIER if p else (OBSERVE_TIER if t == OBSERVE_TIER else "advisory")
        for p, t in zip(promoted, tier_col, strict=False)
    ]
    return pd.Series(routed, index=index)


def _candidate_rows(book: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Project signal rows into the paper_trades insert shape, stamping the book ``tier``.

    Display-only signal fields (horizon, calibrated proba, net expectancy) ride along for the
    Brief's order sheet; ``insert_new`` persists only ``_TRADE_COLUMNS`` so they never hit the DB.
    """
    exit_strategy = (
        book["exit_strategy"].fillna("fixed") if "exit_strategy" in book.columns else "fixed"
    )
    extras = {
        col: book[col]
        for col in ("horizon_days", "proba_calibrated", "exp_return_net")
        if col in book.columns
    }
    return pd.DataFrame(
        {
            "date_open": book["date"],
            "ticker": book["ticker"],
            "signal": book["signal"],
            "entry": book["entry"],
            "stop": book["stop_loss"],
            "target": book["take_profit"],
            "size_shares": book["size_shares"],
            "status": OPEN,
            "exit_strategy": exit_strategy,
            "tier": tier,
            **extras,
        }
    )


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
        strategy = str(row["exit_strategy"]) if "exit_strategy" in row else "fixed"
        resolved = _resolve_trade_exit(ohlcv, row, entry_idx, cfg, short=short, strategy=strategy)
        if resolved is None:
            continue
        exit_idx, exit_price, status = resolved
        entry = float(row["entry"])
        shares = int(row["size_shares"])
        pnl_pct, pnl_eur = _direction_pnl(entry, float(exit_price), shares, short=short)
        # Charge the friction frozen at open (NULL on legacy rows = zero, history untouched).
        cost_bps = row.get("cost_bps") if "cost_bps" in row.index else None
        if cost_bps is not None and pd.notna(cost_bps):
            cost_frac = float(cost_bps) / 1e4
            pnl_pct -= cost_frac
            pnl_eur -= entry * shares * cost_frac
        paper.close_trade(
            date_open=date_open,
            ticker=ticker,
            date_close=ohlcv.index[exit_idx],
            exit_price=float(exit_price),
            status=status,
            pnl_pct=float(pnl_pct),
            pnl_eur=float(pnl_eur),
            exit_strategy=strategy,
        )
        closed_count += 1
    logger.info("paper.update_open_trades: %d trades closed", closed_count)
    return closed_count


def get_open_positions(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> list[OpenPosition]:
    """Return open paper trades enriched with the latest-close MTM (and live trailing stop).

    ``exit_strategy`` filters to one book (the dashboard toggle's selection); ``None`` = all.
    ``tier`` selects the committed book (``"promoted"``, the default), the observation shadow book
    (``"observe"``), or both (``None``).
    """
    paper = PaperStore(config.db_path)
    df = paper.open_trades(tier=tier)
    if exit_strategy is not None and "exit_strategy" in df.columns:
        df = df[df["exit_strategy"].fillna("fixed") == exit_strategy]
    cfg = _LabelConfig(**config.labeling.model_dump())
    out: list[OpenPosition] = []
    today = _ts(pd.Timestamp.today()).normalize()
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            continue
        # Mark to the last VALID close, not iloc[-1]: a provisional/partial yfinance bar can carry
        # volume but NaN OHLC, which would otherwise propagate NaN -> null into current_price / MTM
        # and blank the position on /wallet. Skip the position only if no close is usable at all.
        closes = ohlcv["close"].dropna()
        if closes.empty:
            continue
        current_price = float(closes.iloc[-1])
        entry = float(row["entry"])
        shares = int(row["size_shares"])
        short = _is_short(row["signal"])
        date_open = _ts(row["date_open"])
        days_held = max(0, int(np.busday_count(date_open.date(), today.date())))
        mtm_pct, mtm_eur = _direction_pnl(entry, current_price, shares, short=short)
        strategy = str(row["exit_strategy"]) if "exit_strategy" in row else "fixed"
        trail_stop = _open_trail_stop(
            ohlcv, date_open, float(row["stop"]), cfg, strategy, short=short
        )
        # Has this trade ALREADY hit its barrier in the cached bars but not yet been booked?
        # The daily close job runs only weekday 22:30, so between the breach and the next run the
        # live MTM (at current_price) badly misstates the outcome. Resolve the would-be exit the
        # exact same way ``update_open_trades`` will, so the UI shows the real, capped P&L.
        pending_exit = pending_exit_price = pending_exit_pct = None
        if date_open in ohlcv.index:
            entry_idx = int(ohlcv.index.get_loc(date_open))
            resolved = _resolve_trade_exit(
                ohlcv, row, entry_idx, cfg, short=short, strategy=strategy
            )
            if resolved is not None:
                _idx, pending_exit_price, pending_exit = resolved
                pending_exit_pct = _direction_pnl(
                    entry, float(pending_exit_price), shares, short=short
                )[0]
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
                exit_strategy=strategy,
                trail_stop=trail_stop,
                pending_exit=pending_exit,
                pending_exit_price=pending_exit_price,
                pending_exit_pct=pending_exit_pct,
            )
        )
    return out


def _open_trail_stop(
    ohlcv: pd.DataFrame,
    date_open: pd.Timestamp,
    init_stop: float,
    cfg: _LabelConfig,
    strategy: str,
    *,
    short: bool,
) -> float | None:
    """Live ratcheting stop for an open trailing trade, or ``None`` for a fixed trade / when the
    entry bar isn't in the cache or ATR isn't warmed."""
    if strategy == "fixed" or date_open not in ohlcv.index:
        return None
    entry_idx = int(ohlcv.index.get_loc(date_open))
    scal = _trail_scalars(ohlcv, entry_idx, cfg, strategy, short=short)
    if scal is None:
        return None
    _entry_ref, trail_dist, activation_level = scal
    return _current_trail_stop(
        ohlcv,
        entry_idx,
        init_stop,
        short=short,
        trail_dist=trail_dist,
        activation_level=activation_level,
    )


# ---------------------------------------------------------------- equity curve ----


def get_equity_curve(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> pd.DataFrame:
    """Build a daily paper-equity series alongside an equal-capital SPY buy & hold.

    Equity = ``capital + cumulative realized P&L + sum of open-trade MTM``. SPY
    benchmark is the same starting capital held in SPY from the first paper-trade
    date onward. ``exit_strategy`` restricts the book to one strategy (toggle); ``None`` = all.
    ``tier`` selects the committed book (``"promoted"``, default), the observation shadow book
    (``"observe"``), or both (``None``). Returns an empty frame if no such trades exist.
    """
    paper = PaperStore(config.db_path)
    trades = paper.all_trades(exit_strategy, tier=tier)
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


def get_paper_metrics(
    config: Config,
    store: OhlcvStore,
    *,
    exit_strategy: str | None = None,
    tier: str | None = PROMOTED_TIER,
) -> dict[str, float | int]:
    """Compact summary used by the dashboard card: returns, win rate, drawdowns, n.

    ``exit_strategy`` restricts the book to one strategy (the dashboard toggle); ``None`` = all.
    ``tier`` selects the committed book (``"promoted"``, default) or the observation shadow book
    (``"observe"``); ``None`` = both.
    """
    paper = PaperStore(config.db_path)
    trades = paper.all_trades(exit_strategy, tier=tier)
    capital = float(config.signals.capital)
    equity = get_equity_curve(config, store, exit_strategy=exit_strategy, tier=tier)

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


class RecentExecutions(TypedDict):
    """Executions at the last daily run — shared payload for /replication and the email digest."""

    as_of: str
    capital_base: float
    open: list[dict[str, object]]
    close: list[dict[str, object]]
    adjust: list[dict[str, object]]
    closed_total: int


def recent_executions(
    config: Config,
    store: OhlcvStore,
    *,
    hours: float = 30.0,
    tier: str | None = PROMOTED_TIER,
) -> RecentExecutions:
    """What the committed book actually DID at the last daily run — executions, never forecasts.

    Returns the trades ``open``ed and ``close``d within the last ``hours`` plus ``adjust`` (open
    trailing positions whose effective ratcheting stop to mirror today). The 30h default spans one
    daily run with a margin. This is the single source of truth behind both the ``/replication``
    endpoint and the daily email digest, so the morning copy list never drifts between the two.
    """
    paper = PaperStore(config.db_path)
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
                    "date_open": str(_ts(r["date_open"]).date()),
                }
            )
        done = df[(df["status"] != OPEN) & (df["updated_at"] >= cutoff)]
        for _, r in done.iterrows():
            closed.append(
                {
                    "ticker": str(r["ticker"]),
                    "direction": "short" if _is_short(r["signal"]) else "long",
                    "exit_strategy": str(r["exit_strategy"]),
                    "status": str(r["status"]),
                    "exit_price": float(r["exit_price"]) if pd.notna(r["exit_price"]) else None,
                    "pnl_pct": float(r["pnl_pct"]) if pd.notna(r["pnl_pct"]) else None,
                    "pnl_eur": float(r["pnl_eur"]) if pd.notna(r["pnl_eur"]) else None,
                    "date_close": (
                        str(_ts(r["date_close"]).date()) if pd.notna(r["date_close"]) else None
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
        for p in get_open_positions(config, store, tier=tier)
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
    "CLOSED_STATUSES",
    "CLOSED_STOP",
    "CLOSED_TARGET",
    "CLOSED_TIME",
    "CLOSED_TRAIL",
    "OBSERVE_TIER",
    "OPEN",
    "PROMOTED_TIER",
    "OpenPosition",
    "PaperStore",
    "get_equity_curve",
    "get_open_positions",
    "get_paper_metrics",
    "open_new_trades",
    "plan_committed_opens",
    "recent_executions",
    "update_open_trades",
]
