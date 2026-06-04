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
    date          DATE    NOT NULL,
    ticker        VARCHAR NOT NULL,
    signal        VARCHAR NOT NULL,
    proba         DOUBLE  NOT NULL,
    entry         DOUBLE  NOT NULL,
    stop_loss     DOUBLE  NOT NULL,
    take_profit   DOUBLE  NOT NULL,
    size_shares   BIGINT  NOT NULL,
    notional      DOUBLE  NOT NULL,
    exit_strategy VARCHAR NOT NULL DEFAULT 'fixed',
    created_at    TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, ticker, exit_strategy)
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
    ("tier", "VARCHAR DEFAULT 'advisory'"),
)

_INSERT_COLUMNS = (
    "date, ticker, signal, proba, entry, stop_loss, take_profit, size_shares, notional, "
    "proba_calibrated, meta_proba, acted, ret_q10, ret_q50, ret_q90, sigma_horizon, sltp_method, "
    "direction, proba_long, proba_short, promoted, "
    "exp_return_gross, exp_return_net, cost_bps_roundtrip, "
    "exit_strategy, trail_atr, trail_activation_atr, tier"
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
            _upgrade_pk(con)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def save(self, signals: list[Signal]) -> int:
        """Upsert a batch of signals; return the number written.

        Dedup is on (date, ticker, exit_strategy) so the fixed and trailing variants of the same
        asset coexist as separate rows — the dashboard's strategy toggle reads one or the other.
        """
        if not signals:
            return 0
        rows = pd.DataFrame([s.as_row() for s in signals])
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        with self._connect() as con:
            con.register("incoming", rows)
            con.execute(
                "DELETE FROM signals WHERE (date, ticker, exit_strategy) IN "
                "(SELECT date, ticker, exit_strategy FROM incoming)"
            )
            con.execute(
                f"INSERT INTO signals ({_INSERT_COLUMNS}) "  # noqa: S608 — column list is a module constant
                f"SELECT {_INSERT_COLUMNS} FROM incoming"
            )
        return len(rows)

    def latest(self) -> pd.DataFrame:
        """Return the most recent signal *per (ticker, exit strategy)*, newest-proba first.

        Latest-per-(ticker, strategy) (rather than a single global max date) so multi-asset
        universes with different trading calendars all surface their freshest signal, and the
        fixed and trailing variants of an asset both appear (the UI toggle filters by strategy).
        """
        with self._connect() as con:
            return con.execute(
                "SELECT * EXCLUDE (rn) FROM ("
                "  SELECT *, row_number() OVER ("
                "    PARTITION BY ticker, exit_strategy ORDER BY date DESC) AS rn"
                "  FROM signals"
                ") WHERE rn = 1 ORDER BY proba DESC"
            ).df()

    def history(self, ticker: str) -> pd.DataFrame:
        """Return the full signal history for one ticker, oldest first."""
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY date", [ticker]
            ).df()


def _upgrade_pk(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuild a legacy ``(date, ticker)`` primary key as ``(date, ticker, exit_strategy)``.

    Pre-Phase-13 tables key one row per (date, ticker), which rejects the second exit strategy
    for the same asset. We rebuild the table with the 3-column key, preserving every row (old
    rows carry the default ``exit_strategy='fixed'``). No-op once the key already includes it.
    """
    pk = con.execute(
        "SELECT constraint_column_names FROM duckdb_constraints() "
        "WHERE table_name = 'signals' AND constraint_type = 'PRIMARY KEY'"
    ).fetchall()
    if not pk or "exit_strategy" in list(pk[0][0]):
        return
    old_cols = [r[0] for r in con.execute("DESCRIBE signals").fetchall()]
    con.execute("ALTER TABLE signals RENAME TO _signals_old")
    con.execute(_SCHEMA)
    new_existing = {r[0] for r in con.execute("DESCRIBE signals").fetchall()}
    for name, decl in _MIGRATIONS:
        if name not in new_existing:
            con.execute(f"ALTER TABLE signals ADD COLUMN {name} {decl}")
    new_cols = {r[0] for r in con.execute("DESCRIBE signals").fetchall()}
    common = ", ".join(c for c in old_cols if c in new_cols)
    con.execute(
        f"INSERT INTO signals ({common}) SELECT {common} FROM _signals_old"  # noqa: S608 — identifiers
    )
    con.execute("DROP TABLE _signals_old")
