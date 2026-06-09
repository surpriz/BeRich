"""Intraday paper book (Lot B1) — the cadence primitives the daily engine gets wrong.

Imports the model zoo transitively (via berich.signals), so this runs in the GPU-equipped CI,
not the lean sandbox. Tests the genuinely-new code in isolation from the signal-generation path:
the TIMESTAMP-keyed store, hours-held, the continuous hourly equity curve, and DB isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.config import Config, SignalConfig
from berich.data.store import OhlcvStore
from berich.signals.paper import OPEN, PROMOTED_TIER
from berich.signals.paper_intraday import (
    IntradayPaperStore,
    _hours_held,
    get_intraday_equity_curve,
    get_open_intraday_positions,
)


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config(data_dir=tmp_path, signals=SignalConfig(capital=10_000.0))
    cfg.intraday.enabled = True
    cfg.intraday.tickers = ["BTC-USD"]
    return cfg


def _hourly(store: OhlcvStore, ticker: str, start: str, n: int, px: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    close = px + np.arange(n) * 0.5
    df = pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1_000.0},
        index=idx,
    )
    store.save(ticker, df)
    return df


def _open_row(
    ts_open: pd.Timestamp, ticker: str = "BTC-USD", strategy: str = "fixed"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date_open": ts_open,
                "ticker": ticker,
                "signal": "LONG",
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "size_shares": 10,
                "status": OPEN,
                "exit_strategy": strategy,
                "tier": PROMOTED_TIER,
            }
        ]
    )


def test_hours_held_is_continuous_no_weekend_skip():
    fri = pd.Timestamp("2024-01-05 12:00")  # Friday
    mon = pd.Timestamp("2024-01-08 12:00")  # Monday — 72 calendar hours later
    assert _hours_held(fri, mon) == 72  # busday_count would have skipped the weekend


def test_intraday_store_keeps_distinct_same_day_timestamps(config):
    store = IntradayPaperStore(config.intraday_db_path)
    store.insert_new(_open_row(pd.Timestamp("2024-01-02 09:00")))
    # A second entry the SAME day at a different hour must NOT collide on the PK.
    store.insert_new(_open_row(pd.Timestamp("2024-01-02 14:00")))
    trades = store.all_trades()
    assert len(trades) == 2
    opens = pd.to_datetime(trades["date_open"]).dt.strftime("%H:%M").tolist()
    assert set(opens) == {"09:00", "14:00"}  # time-of-day preserved (no .normalize / .date)


def test_intraday_book_is_isolated_from_daily_db(config):
    IntradayPaperStore(config.intraday_db_path).insert_new(
        _open_row(pd.Timestamp("2024-01-02 09:00"))
    )
    assert config.intraday_db_path.exists()
    # Writing the intraday book never creates/touches the daily DB.
    assert not config.db_path.exists()


def test_intraday_equity_curve_is_hourly_and_benchmarked_on_pair(config):
    store = OhlcvStore(config.ohlcv_intraday_dir, interval="1h")
    _hourly(store, "BTC-USD", "2024-01-05 00:00", 100)
    paper = IntradayPaperStore(config.intraday_db_path)
    paper.insert_new(_open_row(pd.Timestamp("2024-01-05 02:00")))

    curve = get_intraday_equity_curve(config, store, tier=PROMOTED_TIER)
    assert not curve.empty
    assert "equity_bench" in curve.columns  # pair B&H, not SPY
    stamps = pd.to_datetime(curve["date"])
    # Continuous hourly steps (no weekend gap).
    deltas = stamps.diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(hours=1)]


def test_open_positions_report_hours_held(config):
    store = OhlcvStore(config.ohlcv_intraday_dir, interval="1h")
    _hourly(store, "BTC-USD", "2024-01-05 00:00", 50)
    paper = IntradayPaperStore(config.intraday_db_path)
    paper.insert_new(_open_row(pd.Timestamp("2024-01-05 02:00")))
    positions = get_open_intraday_positions(config, store, tier=PROMOTED_TIER)
    assert len(positions) == 1
    assert positions[0].days_held >= 0  # carries hours-held for intraday
