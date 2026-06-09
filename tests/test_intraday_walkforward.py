"""Lot A7 — the intraday (1h) tournament lands in the interval namespace and honors the guard.

Mirrors tests/test_tournament.py but on hourly bars with interval="1h": artifacts must land
under ``.../<side>/1h/``, metadata must record interval=="1h", and a random-walk series must
NOT be promoted (the guard is live, force=False). Requires the model zoo (torch) to import the
tournament, so it runs in the GPU-equipped CI/deploy env, not the lean sandbox.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import Config
from berich.data.store import OhlcvStore
from berich.models.registry import ACTIVE_POINTER, META_FILE


def _hourly_ohlcv(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    rets = rng.normal(0.0, 0.01, n)  # zero-drift random walk — no edge to find
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000, 10_000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def _intraday_config(tmp_path, ticker: str = "BTC-USD") -> Config:
    cfg = Config(data_dir=tmp_path, universes={"crypto": [ticker]})
    cfg.intraday.enabled = True
    cfg.intraday.interval = "1h"
    cfg.intraday.tickers = [ticker]
    store = OhlcvStore(cfg.ohlcv_intraday_dir, interval="1h")
    store.save(ticker, _hourly_ohlcv())
    # The crypto regime proxy is BTC-USD itself; with a single pair it is its own market.
    return cfg


def test_intraday_tournament_advisory_and_namespaced(tmp_path):
    from berich.training.tournament import train_ticker_tournament  # noqa: PLC0415

    cfg = _intraday_config(tmp_path)
    result = train_ticker_tournament(
        cfg, "BTC-USD", "long", models=["lgbm"], calibrate=False, interval="1h"
    )

    # Random-walk crypto cannot beat buy & hold -> advisory only, nothing promoted (guard live).
    assert result.promoted is False
    assert result.winner is None

    registry_dir = cfg.model_dir_for_ticker("BTC-USD", "long", interval="1h")
    assert registry_dir.parts[-1] == "1h"  # interval-dimensioned namespace
    assert not (registry_dir / ACTIVE_POINTER).exists()
    assert (registry_dir / "lgbm-long" / META_FILE).exists()

    import json  # noqa: PLC0415

    meta = json.loads((registry_dir / "lgbm-long" / META_FILE).read_text())
    assert meta["interval"] == "1h"
