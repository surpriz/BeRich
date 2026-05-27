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


class SignalStore:
    """Read/write signals in a DuckDB file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)

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
                "INSERT INTO signals "
                "(date, ticker, signal, proba, entry, stop_loss, take_profit, "
                "size_shares, notional) "
                "SELECT date, ticker, signal, proba, entry, stop_loss, take_profit, "
                "size_shares, notional FROM incoming"
            )
        return len(rows)

    def latest(self) -> pd.DataFrame:
        """Return all signals for the most recent date in the table."""
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM signals WHERE date = (SELECT max(date) FROM signals) "
                "ORDER BY proba DESC"
            ).df()

    def history(self, ticker: str) -> pd.DataFrame:
        """Return the full signal history for one ticker, oldest first."""
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY date", [ticker]
            ).df()
