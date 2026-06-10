"""Tests for the daily video-script generator (facts only, French, always well-formed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import Config, LabelingConfig, SignalConfig
from berich.data.store import OhlcvStore
from berich.notifications.video_script import DISCLAIMER, build_video_script
from berich.signals import open_new_trades, update_open_trades
from berich.signals.service import BUY, Signal
from berich.signals.store import SignalStore


def _config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path,
        watchlist=["AAA"],
        labeling=LabelingConfig(horizon_days=10, atr_window=14),
        signals=SignalConfig(buy_threshold=0.55, capital=10_000.0, risk_pct=0.01),
    )


def _ramp(start: float, end: float, n: int = 20) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(start, end, n)
    df = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1e6},
        index=idx,
    )
    df.index.name = "date"
    return df


def test_empty_book_still_produces_wellformed_script(tmp_path):
    config = _config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    out = build_video_script(config, store)
    assert out["title"].startswith("Le journal du robot")
    script = str(out["script"])
    assert DISCLAIMER in script  # the disclaimer is non-negotiable
    assert "Aucune ouverture ni clôture" in script
    assert "entièrement à plat" in script


def test_closed_trade_is_narrated_with_net_pnl(tmp_path):
    config = _config(tmp_path)
    store = OhlcvStore(config.ohlcv_dir)
    df = _ramp(100.0, 120.0)
    store.save("AAA", df)
    sigstore = SignalStore(config.db_path)
    sigstore.save(
        [
            Signal(
                date=df.index[0],
                ticker="AAA",
                signal=BUY,
                proba=0.65,
                entry=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                size_shares=10,
                notional=1000.0,
                promoted=True,
                tier="promoted",
            )
        ]
    )
    open_new_trades(config, store, sigstore)
    assert update_open_trades(config, store) == 1

    script = str(build_video_script(config, store)["script"])
    assert "AAA" in script
    assert "objectif atteint" in script  # the close and its reason are narrated
    assert "net de frais" in script
    assert DISCLAIMER in script
