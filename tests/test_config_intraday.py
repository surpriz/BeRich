"""Intraday config block + interval-dimensioned model paths (intraday POC)."""

from __future__ import annotations

from pathlib import Path

from berich.config import Config

REPO_YAML = Path(__file__).resolve().parents[1] / "config" / "berich.yaml"


def test_intraday_defaults_off():
    cfg = Config()
    assert cfg.intraday.enabled is False
    assert cfg.data.interval == "1d"  # daily default untouched


def test_intraday_paths_are_separate():
    cfg = Config()
    assert cfg.ohlcv_intraday_dir != cfg.ohlcv_dir
    assert cfg.intraday_db_path != cfg.db_path
    assert cfg.ohlcv_intraday_dir == cfg.data_dir / "ohlcv_1h"


def test_yaml_intraday_block_paused_but_intact():
    # The intraday POC is currently paused (enabled: false) to give 100% of the GPU back to swing,
    # but the 1h block is kept intact so it can be re-enabled with a single flag + restart.
    cfg = Config.load(REPO_YAML)
    assert cfg.intraday.enabled is False
    assert cfg.intraday.interval == "1h"
    assert cfg.intraday.bars_per_year == 8760
    assert cfg.intraday.horizon_bars == 24
    assert cfg.intraday.tickers == ["BTC-USD"]
    assert cfg.intraday.overnight_gap is False
    assert cfg.data.interval == "1d"  # daily block unchanged


def test_model_dir_interval_dimension():
    cfg = Config()
    # interval="1d" returns the byte-identical legacy path (no interval leaf).
    legacy = cfg.model_dir_for_ticker("BTC-USD", "long", "fixed")
    assert legacy == cfg.models_dir / "tickers" / "BTC-USD" / "long"
    assert cfg.model_dir_for_ticker("BTC-USD", "long", "fixed", interval="1d") == legacy
    # 1h appends an interval leaf.
    assert (
        cfg.model_dir_for_ticker("BTC-USD", "long", "fixed", interval="1h")
        == cfg.models_dir / "tickers" / "BTC-USD" / "long" / "1h"
    )
    # Composes with a trailing strategy.
    assert (
        cfg.model_dir_for_ticker("BTC-USD", "short", "trailing", interval="1h")
        == cfg.models_dir / "tickers" / "BTC-USD" / "short" / "trailing" / "1h"
    )
