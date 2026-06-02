"""Paper-trading roundtrip tests.

Covers the contract that matters at the day-to-day level:
- a steady uptrend closes a trade at the target,
- a steady downtrend closes a trade at the stop,
- a flat-then-time-barrier sequence closes a trade with reason "closed_time",
- update is idempotent: running it again after the world stops moving doesn't
  change any trade's status, and re-opening on the same signal date is a no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.config import Config, LabelingConfig, SignalConfig
from berich.data.store import OhlcvStore
from berich.signals import (
    PaperStore,
    open_new_trades,
    update_open_trades,
)
from berich.signals.paper import (
    CLOSED_STOP,
    CLOSED_TARGET,
    CLOSED_TIME,
    CLOSED_TRAIL,
    OPEN,
    get_open_positions,
)
from berich.signals.service import BUY, SHORT, Signal
from berich.signals.store import SignalStore


@pytest.fixture
def config(tmp_path) -> Config:
    """Project config rooted in a tmp directory so each test gets a clean DB + cache."""
    return Config(
        data_dir=tmp_path,
        watchlist=["AAA"],
        labeling=LabelingConfig(horizon_days=10, atr_window=14),
        signals=SignalConfig(
            buy_threshold=0.55, sell_threshold=0.30, capital=10_000.0, risk_pct=0.01
        ),
    )


@pytest.fixture
def ohlcv_store(config) -> OhlcvStore:
    return OhlcvStore(config.ohlcv_dir)


def _save_ohlcv(store: OhlcvStore, ticker: str, df: pd.DataFrame) -> None:
    """Round-trip a frame through OhlcvStore (which expects the canonical schema)."""
    full = df.copy()
    if "volume" not in full.columns:
        full["volume"] = 1_000.0
    full.index.name = "date"
    store.save(ticker, full)


def _ramp(*, start: float, end: float, n: int, start_date: str) -> pd.DataFrame:
    """Smooth-ramp OHLCV from ``start`` to ``end`` over ``n`` business days."""
    idx = pd.bdate_range(start_date, periods=n)
    close = np.linspace(start, end, n)
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close},
        index=idx,
    )


def _flat(*, level: float, n: int, start_date: str) -> pd.DataFrame:
    idx = pd.bdate_range(start_date, periods=n)
    close = np.full(n, level)
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05, "close": close},
        index=idx,
    )


def _signal(
    date: pd.Timestamp,
    ticker: str,
    entry: float,
    stop: float,
    target: float,
    *,
    promoted: bool = True,
    exit_strategy: str = "fixed",
) -> Signal:
    return Signal(
        date=date,
        ticker=ticker,
        signal=BUY,
        proba=0.65,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        size_shares=10,
        notional=entry * 10,
        promoted=promoted,
        exit_strategy=exit_strategy,
    )


def _seed_signal_table(
    config: Config,
    date: pd.Timestamp,
    *,
    entry: float,
    stop: float,
    target: float,
) -> SignalStore:
    """Persist a BUY signal so ``open_new_trades`` has something to read."""
    store = SignalStore(config.db_path)
    store.save([_signal(date, "AAA", entry=entry, stop=stop, target=target)])
    return store


def _short_signal(
    date: pd.Timestamp,
    ticker: str,
    entry: float,
    stop: float,
    target: float,
    *,
    promoted: bool = True,
) -> Signal:
    # A short mirrors a long: stop is *above* entry, target *below*.
    return Signal(
        date=date,
        ticker=ticker,
        signal=SHORT,
        proba=0.65,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        size_shares=10,
        notional=entry * 10,
        promoted=promoted,
        direction="short",
    )


def test_short_downtrend_closes_at_target(config, ohlcv_store):
    # Short entry 100, target 90 (below), stop 105 (above). A slide to 80 hits the
    # target first and the short is profitable (price fell as predicted).
    df = _ramp(start=100.0, end=80.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    sigstore = SignalStore(config.db_path)
    sigstore.save([_short_signal(df.index[0], "AAA", entry=100.0, stop=105.0, target=90.0)])

    opened = open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)

    assert opened == 1
    assert closed == 1
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == CLOSED_TARGET
    assert trade["exit_price"] == pytest.approx(90.0)
    assert trade["pnl_pct"] > 0  # short profits when price falls
    assert trade["pnl_eur"] == pytest.approx((100.0 - 90.0) * 10)


def test_short_uptrend_closes_at_stop(config, ohlcv_store):
    # Short entry 100, stop 105 (above), target 90 (below). A climb to 120 hits the
    # stop first and the short loses.
    df = _ramp(start=100.0, end=120.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    sigstore = SignalStore(config.db_path)
    sigstore.save([_short_signal(df.index[0], "AAA", entry=100.0, stop=105.0, target=90.0)])

    open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)

    assert closed == 1
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == CLOSED_STOP
    assert trade["exit_price"] == pytest.approx(105.0)
    assert trade["pnl_pct"] < 0


def test_advisory_short_opens_no_paper_trade(config, ohlcv_store):
    # An advisory (non-promoted) SHORT must not open a paper trade either.
    df = _ramp(start=100.0, end=80.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [_short_signal(df.index[0], "AAA", entry=100.0, stop=105.0, target=90.0, promoted=False)]
    )
    assert open_new_trades(config, ohlcv_store, sigstore) == 0
    assert PaperStore(config.db_path).all_trades().empty


def test_advisory_signal_opens_no_paper_trade(config, ohlcv_store):
    # A non-promoted (advisory) LONG signal must NOT open a paper trade — the book tracks
    # only the validated strategy.
    df = _ramp(start=100.0, end=120.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [_signal(df.index[0], "AAA", entry=100.0, stop=95.0, target=110.0, promoted=False)]
    )
    assert open_new_trades(config, ohlcv_store, sigstore) == 0
    assert PaperStore(config.db_path).all_trades().empty


def test_uptrend_closes_at_target(config, ohlcv_store):
    # Entry at 100, target 110, stop 95. 20-day climb to 120 -> target must hit first.
    df = _ramp(start=100.0, end=120.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    opened = open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)

    assert opened == 1
    assert closed == 1
    rows = PaperStore(config.db_path).all_trades()
    assert len(rows) == 1
    trade = rows.iloc[0]
    assert trade["status"] == CLOSED_TARGET
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["pnl_pct"] > 0


def test_downtrend_closes_at_stop(config, ohlcv_store):
    # Entry at 100, stop 95, target 110. 20-day slide to 80 -> stop must hit first.
    df = _ramp(start=100.0, end=80.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)

    assert closed == 1
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == CLOSED_STOP
    assert trade["exit_price"] == pytest.approx(95.0)
    assert trade["pnl_pct"] < 0


def test_flat_market_closes_at_time_barrier(config, ohlcv_store):
    # 15 flat bars at 100 — neither stop (95) nor target (110) is touched, so the
    # trade must close at the horizon (10) with reason CLOSED_TIME.
    df = _flat(level=100.0, n=15, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)

    assert closed == 1
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == CLOSED_TIME
    # Time barrier is at entry_idx + horizon_days = 0 + 10 = index 10.
    expected_date = df.index[10]
    assert pd.Timestamp(trade["date_close"]) == pd.Timestamp(expected_date.date())


def test_update_is_idempotent_for_open_and_closed(config, ohlcv_store):
    df = _ramp(start=100.0, end=120.0, n=20, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    open_new_trades(config, ohlcv_store, sigstore)
    first = update_open_trades(config, ohlcv_store)
    # Re-running update once the trade is closed must be a no-op (closed trades
    # are immutable; nothing in 'open' state remains).
    second = update_open_trades(config, ohlcv_store)
    third = update_open_trades(config, ohlcv_store)
    assert first == 1
    assert second == 0
    assert third == 0


def test_open_new_trades_is_idempotent(config, ohlcv_store):
    df = _flat(level=100.0, n=5, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    first = open_new_trades(config, ohlcv_store, sigstore)
    second = open_new_trades(config, ohlcv_store, sigstore)
    assert first == 1
    assert second == 0
    assert len(PaperStore(config.db_path).all_trades()) == 1


def test_open_trade_stays_open_when_history_too_short(config, ohlcv_store):
    # Only 3 bars of history after entry — not enough to hit the time barrier (10)
    # and price stays flat — so the trade should remain in OPEN state.
    df = _flat(level=100.0, n=4, start_date="2024-01-02")
    _save_ohlcv(ohlcv_store, "AAA", df)
    entry_date = df.index[0]
    sigstore = _seed_signal_table(config, entry_date, entry=100.0, stop=95.0, target=110.0)

    open_new_trades(config, ohlcv_store, sigstore)
    closed = update_open_trades(config, ohlcv_store)
    assert closed == 0
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == OPEN


# ----------------------------------------------------------------- trailing exit ----


def _trail_config(tmp_path) -> Config:
    """Config with a short ATR window so the entry bar has warmed ATR a few bars in."""
    return Config(
        data_dir=tmp_path,
        watchlist=["AAA"],
        labeling=LabelingConfig(
            horizon_days=10,
            atr_window=3,
            stop_loss_atr=1.0,
            take_profit_atr=2.0,
            trailing_atr=2.5,
            trailing_activation_atr=1.0,
        ),
        signals=SignalConfig(buy_threshold=0.55, capital=10_000.0, risk_pct=0.01),
    )


def _trail_ohlcv() -> pd.DataFrame:
    # 6 warmup bars (ATR -> 2), entry at idx 5 (close 100), a rise that arms+ratchets the stop
    # to ~106, then bar 8 reverses and breaks it. Padding keeps the time barrier (idx 15) clear.
    close = [100.0] * 6 + [106.0, 110.0, 104.0] + [104.0] * 7
    high = [101.0] * 6 + [107.0, 111.0, 108.0] + [105.0] * 7
    low = [99.0] * 6 + [105.0, 109.0, 104.0] + [103.0] * 7
    idx = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def test_trailing_long_rides_then_exits_on_reversal(tmp_path):
    config = _trail_config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    df = _trail_ohlcv()
    _save_ohlcv(store, "AAA", df)
    entry_date = df.index[5]
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [_signal(entry_date, "AAA", entry=100.0, stop=98.0, target=102.0, exit_strategy="trailing")]
    )

    assert open_new_trades(config, store, sigstore) == 1
    assert update_open_trades(config, store) == 1
    trade = PaperStore(config.db_path).all_trades().iloc[0]
    assert trade["status"] == CLOSED_TRAIL  # ratcheted (armed) stop, not a fixed stop
    assert trade["exit_price"] == pytest.approx(106.0)  # high 111 - trail 5
    assert trade["exit_price"] > 102.0  # captured more than the fixed take-profit would
    assert trade["pnl_pct"] > 0


def test_trailing_open_position_exposes_live_trail_stop(tmp_path):
    config = _trail_config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    # Rise only (no reversal yet) so the trade stays open with an armed, ratcheted stop.
    close = [100.0] * 6 + [106.0, 110.0]
    high = [101.0] * 6 + [107.0, 111.0]
    low = [99.0] * 6 + [105.0, 109.0]
    idx = pd.bdate_range("2024-01-02", periods=len(close))
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    _save_ohlcv(store, "AAA", df)
    entry_date = df.index[5]
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [_signal(entry_date, "AAA", entry=100.0, stop=98.0, target=102.0, exit_strategy="trailing")]
    )

    open_new_trades(config, store, sigstore)
    update_open_trades(config, store)  # not enough bars to hit anything -> stays open
    positions = get_open_positions(config, store)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.exit_strategy == "trailing"
    assert pos.trail_stop == pytest.approx(106.0)  # high 111 - trail 5, ratcheted up from 98


def test_fixed_and_trailing_books_coexist_and_filter(tmp_path):
    # The same asset can be paper-traded under both strategies as separate books, and the
    # metrics/positions read can be scoped to one book (the dashboard toggle).
    config = _trail_config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    df = _trail_ohlcv()
    _save_ohlcv(store, "AAA", df)
    entry_date = df.index[5]
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [
            _signal(entry_date, "AAA", entry=100.0, stop=98.0, target=102.0, exit_strategy="fixed"),
            _signal(
                entry_date, "AAA", entry=100.0, stop=98.0, target=102.0, exit_strategy="trailing"
            ),
        ]
    )
    assert open_new_trades(config, store, sigstore) == 2  # one trade per book
    update_open_trades(config, store)
    all_trades = PaperStore(config.db_path).all_trades()
    assert set(all_trades["exit_strategy"]) == {"fixed", "trailing"}
    # Scoped reads return only the requested book.
    fixed_only = PaperStore(config.db_path).all_trades("fixed")
    assert set(fixed_only["exit_strategy"]) == {"fixed"}


def test_trailing_idempotent_after_close(tmp_path):
    config = _trail_config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    _save_ohlcv(store, "AAA", _trail_ohlcv())
    entry_date = _trail_ohlcv().index[5]
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [_signal(entry_date, "AAA", entry=100.0, stop=98.0, target=102.0, exit_strategy="trailing")]
    )

    open_new_trades(config, store, sigstore)
    assert update_open_trades(config, store) == 1
    assert update_open_trades(config, store) == 0  # closed trades are immutable
