"""DuckDB persistence for generated signals.

Signals are kept so the dashboard and any later evaluation can show history and
compare past advice to realized outcomes. The primary key is ``(date, ticker)`` so
re-running generation for the same day overwrites rather than duplicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from berich.signals.service import Signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    date         DATE    NOT NULL,
    ticker       VARCHAR NOT NULL,
    signal       VARCHAR NOT NULL,
    proba        DOUBLE  NOT NULL,
    entry        DOUBLE  NOT NULL,
    stop_loss    DOUBLE  NOT NULL,
    take_profit  DOUBLE  NOT NULL,
    size_shares  BIGINT  NOT NULL,
    notional     DOUBLE  NOT NULL,
    created_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, ticker)
);
"""

# Enriched-advice columns added after the original schema, as (name, "TYPE [DEFAULT ...]").
# CRITICAL: we only ADD a column when it is genuinely missing. Re-running
# ``ADD COLUMN IF NOT EXISTS <col> ... DEFAULT <d>`` on a column that already exists does NOT
# no-op in DuckDB — it RE-APPLIES the default, wiping every stored value back to <d>. Since the
# store is constructed on every access, that silently reset e.g. ``promoted`` to FALSE on each
# read. So `_apply_migrations` checks the live column set first (see __init__).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("proba_calibrated", "DOUBLE"),
    ("meta_proba", "DOUBLE"),
    ("acted", "BOOLEAN DEFAULT TRUE"),
    ("ret_q10", "DOUBLE"),
    ("ret_q50", "DOUBLE"),
    ("ret_q90", "DOUBLE"),
    ("sigma_horizon", "DOUBLE"),
    ("sltp_method", "VARCHAR DEFAULT 'atr_fixed'"),
    ("direction", "VARCHAR DEFAULT 'long'"),
    ("proba_long", "DOUBLE"),
    ("proba_short", "DOUBLE"),
    ("promoted", "BOOLEAN DEFAULT FALSE"),
    ("exp_return_gross", "DOUBLE"),
    ("exp_return_net", "DOUBLE"),
    ("cost_bps_roundtrip", "DOUBLE"),
    ("exit_strategy", "VARCHAR DEFAULT 'fixed'"),
    ("trail_atr", "DOUBLE"),
    ("trail_activation_atr", "DOUBLE"),
)

_INSERT_COLUMNS = (
    "date, ticker, signal, proba, entry, stop_loss, take_profit, size_shares, notional, "
    "proba_calibrated, meta_proba, acted, ret_q10, ret_q50, ret_q90, sigma_horizon, sltp_method, "
    "direction, proba_long, proba_short, promoted, "
    "exp_return_gross, exp_return_net, cost_bps_roundtrip, "
    "exit_strategy, trail_atr, trail_activation_atr"
)


class SignalStore:
    """Read/write signals in a DuckDB file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)
            existing = {row[0] for row in con.execute("DESCRIBE signals").fetchall()}
            for name, decl in _MIGRATIONS:
                # Only add a genuinely-missing column. DuckDB re-applies the DEFAULT (wiping
                # stored values) if we ALTER an existing column, so guard on `existing`.
                if name not in existing:
                    con.execute(f"ALTER TABLE signals ADD COLUMN {name} {decl}")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def save(self, signals: list[Signal]) -> int:
        """Upsert a batch of signals; return the number written."""
        if not signals:
            return 0
        rows = pd.DataFrame([s.as_row() for s in signals])
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        with self._connect() as con:
            con.register("incoming", rows)
            con.execute(
                "DELETE FROM signals WHERE (date, ticker) IN (SELECT date, ticker FROM incoming)"
            )
            con.execute(
                f"INSERT INTO signals ({_INSERT_COLUMNS}) "  # noqa: S608 — column list is a module constant
                f"SELECT {_INSERT_COLUMNS} FROM incoming"
            )
        return len(rows)

    def latest(self) -> pd.DataFrame:
        """Return the most recent signal *per ticker*, newest-proba first.

        Latest-per-ticker (rather than a single global max date) so multi-asset universes
        with different trading calendars — crypto trades weekends, equities don't — all
        surface their freshest signal instead of only whichever class closed most recently.
        """
        with self._connect() as con:
            return con.execute(
                "SELECT * EXCLUDE (rn) FROM ("
                "  SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn"
                "  FROM signals"
                ") WHERE rn = 1 ORDER BY proba DESC"
            ).df()

    def history(self, ticker: str) -> pd.DataFrame:
        """Return the full signal history for one ticker, oldest first."""
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY date", [ticker]
            ).df()
