"""Per-ticker config plumbing: filesystem-safe slugs, registry dirs, tradeable set."""

from __future__ import annotations

import pytest

from berich.config import AssetUniverses, Config, SignalConfig, safe_ticker_slug


def test_safe_ticker_slug_replaces_unsafe_chars():
    assert safe_ticker_slug("EURUSD=X") == "EURUSD_X"
    assert safe_ticker_slug("GC=F") == "GC_F"
    # Path-valid characters are preserved (just upper-cased).
    assert safe_ticker_slug("BTC-USD") == "BTC-USD"
    assert safe_ticker_slug("mc.pa") == "MC.PA"


def test_model_dir_for_ticker_namespaces_by_side():
    cfg = Config()
    long_dir = cfg.model_dir_for_ticker("EURUSD=X", "long")
    short_dir = cfg.model_dir_for_ticker("EURUSD=X", "short")
    assert long_dir == cfg.models_dir / "tickers" / "EURUSD_X" / "long"
    assert short_dir == cfg.models_dir / "tickers" / "EURUSD_X" / "short"
    # The per-ticker subtree never collides with the legacy class dirs.
    assert "tickers" in long_dir.parts


def test_model_dir_for_ticker_rejects_unknown_side():
    with pytest.raises(ValueError, match="unknown side"):
        Config().model_dir_for_ticker("AAPL", "sideways")


def test_tradeable_tickers_unions_universes():
    cfg = Config(
        universes=AssetUniverses(
            us_stocks=["AAPL", "SPY"],
            crypto=["BTC-USD"],
            forex=["EURUSD=X"],
        )
    )
    tradeable = cfg.tradeable_tickers()
    assert set(tradeable) == {"AAPL", "SPY", "BTC-USD", "EURUSD=X"}


def test_short_threshold_default_mirrors_buy_threshold():
    sig = SignalConfig()
    assert sig.short_threshold == pytest.approx(sig.buy_threshold)
    assert sig.enable_short is True
