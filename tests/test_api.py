"""API tests using FastAPI's TestClient against a temp cache + signal store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from berich.api import create_app
from berich.data.store import OhlcvStore
from berich.signals.service import Signal
from berich.signals.store import SignalStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("BERICH_API_KEY", raising=False)
    data_dir = tmp_path / "data"

    # Seed an OHLCV cache for one ticker.
    idx = pd.bdate_range("2023-01-01", periods=300)
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300))
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )
    OhlcvStore(data_dir / "ohlcv").save("AAPL", df)

    # Seed one persisted signal.
    SignalStore(data_dir / "berich.duckdb").save(
        [
            Signal(
                date=pd.Timestamp("2024-01-05"),
                ticker="AAPL",
                signal="BUY",
                proba=0.6,
                entry=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                size_shares=20,
                notional=2000.0,
            )
        ]
    )

    cfg = {"data_dir": str(data_dir), "watchlist": ["AAPL"]}
    cfg_path = tmp_path / "berich.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return TestClient(create_app(str(cfg_path)))


def test_health(client):
    # /api/health is exempt from auth so the reverse proxy can probe liveness
    # without owning the API key.
    assert client.get("/api/health").json() == {"status": "ok"}


def test_watchlist(client):
    assert client.get("/api/watchlist").json() == ["AAPL"]


def test_signals(client):
    rows = client.get("/api/signals").json()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["signal"] == "BUY"


def test_prices(client):
    rows = client.get("/api/prices/AAPL?days=30").json()
    assert len(rows) == 30
    assert {"date", "open", "high", "low", "close", "volume"} <= set(rows[0])


def test_prices_unknown_ticker_404(client):
    assert client.get("/api/prices/ZZZZ").status_code == 404


def test_health_open_under_api_key(tmp_path, monkeypatch):
    """Health probe must stay 200 even when an API key is configured."""
    monkeypatch.setenv("BERICH_API_KEY", "secret")
    data_dir = tmp_path / "data"
    OhlcvStore(data_dir / "ohlcv")
    cfg_path = tmp_path / "berich.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_dir": str(data_dir), "watchlist": []}))
    client = TestClient(create_app(str(cfg_path)))
    assert client.get("/api/health").status_code == 200


def test_api_key_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("BERICH_API_KEY", "secret")
    data_dir = tmp_path / "data"
    OhlcvStore(data_dir / "ohlcv")  # create dir
    cfg_path = tmp_path / "berich.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_dir": str(data_dir), "watchlist": []}))
    client = TestClient(create_app(str(cfg_path)))

    assert client.get("/api/watchlist").status_code == 401
    assert client.get("/api/watchlist", headers={"X-API-Key": "secret"}).status_code == 200
