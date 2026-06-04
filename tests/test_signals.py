"""Tests for signal classification, position sizing, and DuckDB persistence."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from berich.config import Config, SignalConfig
from berich.signals.calibration import optimal_decision_threshold
from berich.signals.service import (
    BUY,
    LONG,
    NEUTRAL,
    SELL,
    SHORT,
    Signal,
    _classify,
    _decide,
    _expected_return,
    _price_decimals,
    _regime_threshold_bump,
    _size_position,
)
from berich.signals.store import SignalStore


def _config() -> Config:
    return Config(
        signals=SignalConfig(
            buy_threshold=0.55, sell_threshold=0.30, capital=10_000.0, risk_pct=0.01
        )
    )


def _ls_config(*, enable_short: bool = True) -> Config:
    return Config(
        signals=SignalConfig(
            buy_threshold=0.55,
            short_threshold=0.55,
            enable_short=enable_short,
            capital=10_000.0,
            risk_pct=0.01,
        )
    )


def test_classify_thresholds():
    cfg = _config()
    assert _classify(0.60, cfg) == BUY
    assert _classify(0.45, cfg) == NEUTRAL
    assert _classify(0.20, cfg) == SELL


def test_decide_picks_long():
    assert _decide(0.60, 0.20, _ls_config()) == (LONG, "long")


def test_decide_picks_short():
    assert _decide(0.20, 0.60, _ls_config()) == (SHORT, "short")


def test_decide_neutral_when_both_below_thresholds():
    assert _decide(0.40, 0.40, _ls_config()) == (NEUTRAL, "long")


def test_decide_tie_favors_long():
    # Both clear their threshold and are equal -> deterministic long bias.
    assert _decide(0.60, 0.60, _ls_config()) == (LONG, "long")


def test_decide_no_short_model_never_shorts():
    assert _decide(0.60, None, _ls_config()) == (LONG, "long")
    assert _decide(0.40, None, _ls_config()) == (NEUTRAL, "long")


def test_decide_short_suppressed_when_disabled():
    # A strong short is ignored when enable_short is False.
    assert _decide(0.20, 0.90, _ls_config(enable_short=False)) == (NEUTRAL, "long")


def test_decide_per_asset_threshold_overrides_global():
    cfg = _ls_config()  # global buy/short thresholds ~0.5
    # A per-asset long bar of 0.40 lets a 0.45 long through that the global bar would reject.
    assert _decide(0.45, 0.10, cfg, long_threshold=0.40) == (LONG, "long")
    # A stricter per-asset long bar of 0.70 rejects a 0.60 long the global bar would accept.
    assert _decide(0.60, 0.10, cfg, long_threshold=0.70) == (NEUTRAL, "long")
    # Per-asset short bar likewise overrides on the short side.
    assert _decide(0.10, 0.45, cfg, short_threshold=0.40) == (SHORT, "short")


def test_optimal_decision_threshold_prefers_high_precision_bucket():
    # Below 0.5 the labels are coin-flips; at/above 0.6 they're almost all wins. With a 2:1 payoff
    # the optimizer should pick a threshold in the high-precision region, not the noisy low one.
    rng = np.random.default_rng(0)
    low = rng.uniform(0.30, 0.59, 200)
    high = rng.uniform(0.60, 0.70, 200)
    proba = np.concatenate([low, high])
    y = np.concatenate([rng.integers(0, 2, 200), np.ones(200, dtype=int)])
    tau = optimal_decision_threshold(proba, y, reward=2.0, risk=1.0, min_count=20)
    assert tau is not None
    assert tau >= 0.55


def test_optimal_decision_threshold_none_when_too_few():
    assert optimal_decision_threshold(np.array([0.6, 0.7]), np.array([1, 1]), min_count=20) is None


def test_decide_threshold_bump_makes_entries_stricter():
    cfg = _ls_config()
    # A 0.55 long clears the base bar but not once the high-vol regime adds a 0.10 bump.
    assert _decide(0.55, 0.10, cfg) == (LONG, "long")
    assert _decide(0.55, 0.10, cfg, threshold_bump=0.10) == (NEUTRAL, "long")


def test_regime_threshold_bump_high_vs_low_vol():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    # Calm history then a volatile tail -> the latest 20d rvol sits in the top quantile.
    rets = np.concatenate([rng.normal(0, 0.005, 240), rng.normal(0, 0.05, 60)])
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    cfg = Config()
    cfg.signals.regime_conditioning = True
    assert _regime_threshold_bump(close, idx[-1], cfg) == cfg.signals.regime_threshold_bump
    # Off by default -> no bump regardless of regime.
    assert _regime_threshold_bump(close, idx[-1], Config()) == 0.0


def test_size_position_risk_based():
    cfg = _config()
    # Risk 1% of 10k = $100; stop distance $5 -> 20 shares, $2000 notional.
    shares, notional = _size_position(entry=100.0, stop=95.0, config=cfg)
    assert shares == 20
    assert notional == 2000.0


def test_size_position_uses_absolute_distance():
    # A short's stop sits ABOVE entry; |100 - 105| == 5 must size like the long case.
    cfg = _config()
    assert _size_position(entry=100.0, stop=105.0, config=cfg) == _size_position(
        entry=100.0, stop=95.0, config=cfg
    )


def test_size_position_capped_at_capital_no_leverage():
    # A low-priced FX pair with a tiny stop distance would risk-size to tens of thousands
    # of units; the no-leverage cap keeps notional <= capital.
    cfg = _config()  # capital 10k, risk 1%
    shares, notional = _size_position(entry=1.1664, stop=1.1602, config=cfg)
    assert notional <= cfg.signals.capital
    assert shares == int(cfg.signals.capital // 1.1664)


def test_expected_return_triple_barrier_expectancy():
    # Long: entry 100, target 110 (+10% reward), stop 95 (-5% risk), P(win)=0.4.
    # E = 0.4*0.10 - 0.6*0.05 = 0.04 - 0.03 = +0.01.
    er = _expected_return(0.4, entry=100.0, stop=95.0, target=110.0)
    assert er == pytest.approx(0.01)


def test_expected_return_direction_agnostic_via_abs():
    # Short: entry 100, target 90 (reward), stop 105 (risk). Same |distances| as a 10/5 long
    # scaled, so the abs-based formula yields P(win)*0.10 - P(loss)*0.05.
    er = _expected_return(0.4, entry=100.0, stop=105.0, target=90.0)
    assert er == pytest.approx(0.4 * 0.10 - 0.6 * 0.05)


def test_price_decimals_scales_with_magnitude():
    assert _price_decimals(212.5) == 2  # equity
    assert _price_decimals(1.1664) == 4  # FX pair
    assert _price_decimals(0.42) == 6  # sub-unit (e.g. some crypto/penny)


def test_size_position_rejects_nonpositive_stop_distance():
    cfg = _config()
    shares, notional = _size_position(entry=100.0, stop=100.0, config=cfg)
    assert shares == 0
    assert notional == 0.0


def _signal(ticker: str, proba: float, date: str = "2024-01-05") -> Signal:
    return Signal(
        date=pd.Timestamp(date),
        ticker=ticker,
        signal=BUY,
        proba=proba,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size_shares=20,
        notional=2000.0,
    )


def test_signal_store_roundtrip_and_upsert(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6), _signal("MSFT", 0.4)])

    latest = store.latest()
    assert len(latest) == 2
    # Ordered by proba descending.
    assert latest.iloc[0]["ticker"] == "AAPL"

    # Re-saving the same (date, ticker) overwrites instead of duplicating.
    store.save([_signal("AAPL", 0.7)])
    latest = store.latest()
    assert len(latest) == 2
    aapl = latest[latest["ticker"] == "AAPL"].iloc[0]
    assert abs(aapl["proba"] - 0.7) < 1e-9


def test_promoted_persists_across_store_instances(tmp_path):
    # Regression: a promoted signal must read back True from a SEPARATE SignalStore instance
    # (the API/paper book open their own). Guards the recurring "promoted shows False" bug.
    db = tmp_path / "berich.duckdb"
    sig = Signal(
        date=pd.Timestamp("2024-01-05"),
        ticker="BNP.PA",
        signal=LONG,
        proba=0.6,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size_shares=20,
        notional=2000.0,
        promoted=True,
    )
    SignalStore(db).save([sig])
    # Fresh instance (new connection) — the value must still be True.
    reread = SignalStore(db).latest()
    row = reread[reread["ticker"] == "BNP.PA"].iloc[0]
    assert bool(row["promoted"]) is True


def test_signal_store_history(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6, "2024-01-04"), _signal("AAPL", 0.5, "2024-01-05")])
    hist = store.history("AAPL")
    assert len(hist) == 2
    assert list(hist["date"]) == sorted(hist["date"])


def test_signal_store_roundtrips_direction_and_per_side_proba(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    short_sig = Signal(
        date=pd.Timestamp("2024-01-05"),
        ticker="TSLA",
        signal=SHORT,
        proba=0.61,
        entry=100.0,
        stop_loss=105.0,  # short stop sits ABOVE entry
        take_profit=90.0,  # short target sits BELOW entry
        size_shares=20,
        notional=2000.0,
        direction="short",
        proba_long=0.30,
        proba_short=0.61,
    )
    store.save([short_sig])
    row = store.history("TSLA").iloc[0]
    assert row["direction"] == "short"
    assert abs(row["proba_short"] - 0.61) < 1e-9
    assert abs(row["proba_long"] - 0.30) < 1e-9
    # Mirrored barriers: target below entry below stop.
    assert row["take_profit"] < row["entry"] < row["stop_loss"]


def test_legacy_signal_defaults_long_direction(tmp_path):
    # A Signal built the old way (no direction kwarg) persists as a long row.
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6)])
    row = store.history("AAPL").iloc[0]
    assert row["direction"] == "long"


def test_signal_store_roundtrips_trailing_fields(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    trailing = Signal(
        date=pd.Timestamp("2024-01-05"),
        ticker="BNP.PA",
        signal=LONG,
        proba=0.6,
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        size_shares=20,
        notional=2000.0,
        sltp_method="trailing",
        exit_strategy="trailing",
        trail_atr=2.5,
        trail_activation_atr=1.0,
    )
    store.save([trailing])
    row = store.history("BNP.PA").iloc[0]
    assert row["exit_strategy"] == "trailing"
    assert row["trail_atr"] == pytest.approx(2.5)
    assert row["trail_activation_atr"] == pytest.approx(1.0)


def test_legacy_signal_defaults_fixed_exit_strategy(tmp_path):
    # A Signal built the old way persists as a fixed-exit row (back-compat default).
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6)])
    row = store.history("AAPL").iloc[0]
    assert row["exit_strategy"] == "fixed"


def test_fixed_and_trailing_signals_coexist_for_same_asset(tmp_path):
    # The toggle relies on one row per (date, ticker, exit_strategy): both must survive a save.
    store = SignalStore(tmp_path / "berich.duckdb")
    common = {
        "date": pd.Timestamp("2024-01-05"),
        "ticker": "BNP.PA",
        "signal": LONG,
        "proba": 0.6,
        "entry": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "size_shares": 20,
        "notional": 2000.0,
    }
    store.save(
        [
            Signal(**common, exit_strategy="fixed"),
            Signal(**common, exit_strategy="trailing", sltp_method="trailing"),
        ]
    )
    latest = store.latest()
    rows = latest[latest["ticker"] == "BNP.PA"]
    assert set(rows["exit_strategy"]) == {"fixed", "trailing"}


def test_legacy_two_column_pk_is_upgraded_in_place(tmp_path):
    # A pre-Phase-13 DB keyed (date, ticker) must be rebuilt to (date, ticker, exit_strategy)
    # WITHOUT losing rows, so the prod DB can hold a second strategy after deploy.
    db = tmp_path / "berich.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute(
            "CREATE TABLE signals (date DATE, ticker VARCHAR, signal VARCHAR, proba DOUBLE, "
            "entry DOUBLE, stop_loss DOUBLE, take_profit DOUBLE, size_shares BIGINT, "
            "notional DOUBLE, PRIMARY KEY (date, ticker))"
        )
        con.execute(
            "INSERT INTO signals VALUES ('2024-01-05', 'AAPL', 'LONG', 0.6, 100, 95, 110, 10, 1000)"
        )
    # Constructing the store runs the PK upgrade; the legacy row is preserved as a fixed-exit row.
    store = SignalStore(db)
    hist = store.history("AAPL")
    assert len(hist) == 1
    assert hist.iloc[0]["exit_strategy"] == "fixed"
    # And a trailing variant can now be added for the same (date, ticker).
    store.save([_signal("AAPL", 0.6, "2024-01-05")])  # fixed, overwrites
    store.save(
        [
            Signal(
                date=pd.Timestamp("2024-01-05"),
                ticker="AAPL",
                signal=LONG,
                proba=0.6,
                entry=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                size_shares=10,
                notional=1000.0,
                exit_strategy="trailing",
            )
        ]
    )
    assert set(store.history("AAPL")["exit_strategy"]) == {"fixed", "trailing"}
