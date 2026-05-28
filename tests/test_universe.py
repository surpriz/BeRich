"""Phase 6 tests: volume-proportional slippage + liquidity gate + universe resolver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.backtest.engine import BacktestConfig, _slippage_for_ticker
from berich.config import Config
from berich.data.ingest import MIN_HISTORY_BARS, MIN_MEDIAN_VOLUME, _liquidity_warning
from berich.data.store import OhlcvStore


def _ohlcv(*, n: int = 600, volume: float = 1_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


# ----------------------------------------------------- volume slippage model ----


def test_constant_slippage_when_disabled():
    cfg = BacktestConfig(slippage_bps=5.0, volume_proportional_slippage=False)
    # Volume in the frame is irrelevant when the proportional model is off.
    slip_low_vol = _slippage_for_ticker(_ohlcv(volume=100_000), cfg)
    slip_high_vol = _slippage_for_ticker(_ohlcv(volume=100_000_000), cfg)
    assert slip_low_vol == pytest.approx(5.0 / 1e4)
    assert slip_high_vol == pytest.approx(5.0 / 1e4)


def test_proportional_slippage_scales_with_volume():
    """At the reference volume slippage equals the base; below it scales up
    as sqrt(volume_ref / median_volume)."""
    cfg = BacktestConfig(
        slippage_bps=5.0,
        volume_proportional_slippage=True,
        volume_ref=80_000_000.0,
        slippage_cap_bps=100.0,
    )
    # Median volume == reference -> slippage == base.
    slip_ref = _slippage_for_ticker(_ohlcv(volume=80_000_000), cfg)
    assert slip_ref == pytest.approx(5.0 / 1e4, rel=0.01)
    # Median volume 100x lower -> sqrt(100) == 10x more slippage.
    slip_thin = _slippage_for_ticker(_ohlcv(volume=800_000), cfg)
    assert slip_thin == pytest.approx(50.0 / 1e4, rel=0.05)


def test_proportional_slippage_is_capped():
    """An extremely thin name should be capped at slippage_cap_bps, not infinite."""
    cfg = BacktestConfig(
        slippage_bps=5.0,
        volume_proportional_slippage=True,
        volume_ref=80_000_000.0,
        slippage_cap_bps=100.0,
    )
    slip = _slippage_for_ticker(_ohlcv(volume=100), cfg)
    assert slip == pytest.approx(100.0 / 1e4)


def test_proportional_slippage_handles_zero_volume():
    """A ticker with zero median volume falls back to the base rate (no NaN)."""
    cfg = BacktestConfig(slippage_bps=5.0, volume_proportional_slippage=True)
    df = _ohlcv()
    df["volume"] = 0.0
    slip = _slippage_for_ticker(df, cfg)
    assert slip == pytest.approx(5.0 / 1e4)


# -------------------------------------------------- liquidity gate (ingest) ----


def test_liquidity_warning_skips_history_too_short(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", _ohlcv(n=MIN_HISTORY_BARS - 10, volume=10_000_000))
    warn = _liquidity_warning(store, "AAA")
    assert warn is not None
    assert "history" in warn


def test_liquidity_warning_skips_volume_too_low(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", _ohlcv(volume=MIN_MEDIAN_VOLUME // 2))
    warn = _liquidity_warning(store, "AAA")
    assert warn is not None
    assert "median volume" in warn


def test_liquidity_warning_clean_for_good_ticker(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", _ohlcv(n=MIN_HISTORY_BARS + 100, volume=MIN_MEDIAN_VOLUME * 10))
    assert _liquidity_warning(store, "AAA") is None


def test_liquidity_warning_missing_ticker(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    warn = _liquidity_warning(store, "GHOST")
    assert warn is not None
    assert "no cached" in warn


# ------------------------------------------------------- universe resolver ----


def test_tickers_for_universe_mega_aliases_watchlist():
    config = Config(watchlist=["AAA", "BBB"])
    assert config.tickers_for_universe("mega") == ["AAA", "BBB"]


def test_tickers_for_universe_all_dedupes():
    config = Config(
        watchlist=["AAA", "BBB"],
        mid_cap_universe=["BBB", "CCC"],
        small_cap_universe=["CCC", "DDD"],
    )
    # Dedup keeps first-occurrence order across mega -> mid -> small.
    assert config.tickers_for_universe("all") == ["AAA", "BBB", "CCC", "DDD"]


def test_tickers_for_universe_unknown_raises():
    config = Config(watchlist=["AAA"])
    with pytest.raises(ValueError, match="unknown universe"):
        config.tickers_for_universe("crypto")
