"""Polish v2 tests: multi-asset config + SHAP explain endpoint + universes API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from berich.api import create_app
from berich.config import (
    ASSET_UNIVERSE_NAMES,
    AssetUniverses,
    Config,
)
from berich.data.store import OhlcvStore
from berich.datasets.assemble import build_ticker_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.signals.service import Signal
from berich.signals.store import SignalStore

# ---------------------------------------------------------- multi-asset config ----


def test_asset_universes_get_returns_each_named_list():
    u = AssetUniverses(
        us_stocks=["AAPL"],
        fr_stocks=["AIR.PA"],
        forex=["EURUSD=X"],
        crypto=["BTC-USD"],
        commodities=["GC=F"],
    )
    for name in ASSET_UNIVERSE_NAMES:
        assert isinstance(u.get(name), list)
    assert u.get("us_stocks") == ["AAPL"]
    assert u.get("forex") == ["EURUSD=X"]


def test_asset_universes_unknown_raises():
    u = AssetUniverses()
    with pytest.raises(ValueError, match="unknown asset universe"):
        u.get("ascii_art")


def test_asset_universes_all_tickers_dedupes_across_universes():
    u = AssetUniverses(us_stocks=["AAPL", "MSFT"], crypto=["MSFT", "BTC-USD"])
    assert u.all_tickers() == ["AAPL", "MSFT", "BTC-USD"]


def test_asset_class_lookup_returns_first_match():
    u = AssetUniverses(us_stocks=["AAPL"], crypto=["BTC-USD"])
    assert u.asset_class("AAPL") == "us_stocks"
    assert u.asset_class("BTC-USD") == "crypto"
    assert u.asset_class("UNKNOWN") is None


def test_config_loads_multi_asset_yaml(tmp_path):
    """A YAML with the new ``universes`` block must validate end-to-end."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "watchlist": ["AAPL", "MSFT"],
                "universes": {
                    "us_stocks": ["AAPL", "MSFT"],
                    "fr_stocks": ["AIR.PA"],
                    "forex": ["EURUSD=X"],
                    "crypto": ["BTC-USD"],
                    "commodities": ["GC=F"],
                },
            }
        )
    )
    config = Config.load(cfg_path)
    assert config.universes.us_stocks == ["AAPL", "MSFT"]
    assert config.universes.fr_stocks == ["AIR.PA"]
    # all_runtime_tickers unions watchlist + universes without duplicates.
    runtime = config.all_runtime_tickers()
    assert "AAPL" in runtime
    assert "AIR.PA" in runtime
    assert "EURUSD=X" in runtime
    # Counting upper-cased symbols, AAPL appears once even though it's in both
    # the watchlist and us_stocks.
    upper_set = {t.upper() for t in runtime}
    assert len(runtime) == len(upper_set)


def test_asset_class_for_falls_back_to_us_stocks_on_legacy_watchlist():
    """Tickers in the legacy ``watchlist`` but not in any explicit universe
    must classify as ``us_stocks`` — that's what the model was trained on."""
    config = Config(watchlist=["AAPL"], universes=AssetUniverses())
    assert config.asset_class_for("AAPL") == "us_stocks"
    # Truly unknown tickers report ``unknown`` so the UI shows the
    # experimental banner explicitly.
    assert config.asset_class_for("MYSTERY") == "unknown"


# --------------------------------------------------------- explain endpoint ----


@pytest.fixture
def client_with_data(tmp_path, monkeypatch):
    """API client backed by a seeded OHLCV cache big enough to fit features+model."""
    monkeypatch.delenv("BERICH_API_KEY", raising=False)
    data_dir = tmp_path / "data"
    store = OhlcvStore(data_dir / "ohlcv")

    # Build a 600-bar synthetic frame: enough history for SMA50 + ATR + the
    # 22-feature pipeline to produce non-NaN rows.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=600)
    close = 100 + np.cumsum(rng.normal(0, 1, 600))
    # volume must vary so the volume_z20 z-score has a non-zero std and produces
    # non-NaN values; a constant volume causes every row's z-score to be NaN
    # and the dataset assembly drops every sample under dropna.
    volume = rng.integers(900_000, 1_100_000, size=600)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    for ticker in ("AAPL", "SPY"):
        store.save(ticker, ohlcv)

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

    # Seed AAPL's own promoted long model so explain has an optimized per-asset model to
    # explain (serving no longer falls back to a generic global model).
    cfg_obj = Config(data_dir=data_dir, universes=AssetUniverses(us_stocks=["AAPL"]))
    ds = build_ticker_dataset(ohlcv, LabelConfig(), ticker="AAPL", market=ohlcv)
    model = LGBMModel(n_estimators=20).fit(ds.x, ds.y)
    reg = cfg_obj.model_dir_for_ticker("AAPL", "long")
    meta = ModelMetadata(
        name="lgbm-long",
        framework="lightgbm",
        feature_columns=list(ds.x.columns),
        metrics={"auc": 0.6, "sharpe": 1.2, "benchmark_sharpe": 0.3, "n_trades": 30.0},
        beats_buy_hold=True,
        ticker="AAPL",
        side="long",
    )
    save_model(model, meta, registry_dir=reg)
    promote("lgbm-long", registry_dir=reg)

    cfg = {"data_dir": str(data_dir), "watchlist": ["AAPL", "SPY"]}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return TestClient(create_app(str(cfg_path)))


def test_explain_endpoint_returns_top_features_and_base_value(client_with_data):
    response = client_with_data.get("/api/signals/AAPL/explain")
    assert response.status_code == 200
    payload = response.json()
    expected_keys = {"ticker", "date", "proba", "base_value", "top_features", "recent_news"}
    assert expected_keys <= set(payload)
    assert payload["ticker"] == "AAPL"
    assert 0.0 <= payload["proba"] <= 1.0
    assert isinstance(payload["base_value"], float)
    # top_features returns at most top_k items, each with feature + contribution.
    assert 1 <= len(payload["top_features"]) <= 5
    for entry in payload["top_features"]:
        assert {"feature", "contribution"} <= set(entry)
        assert isinstance(entry["contribution"], float)
    # No news cache seeded → empty list, not an error.
    assert payload["recent_news"] == []


def test_explain_endpoint_404_on_unknown_ticker(client_with_data):
    response = client_with_data.get("/api/signals/ZZZZ/explain")
    assert response.status_code == 404


def test_universes_endpoint_shape(client_with_data):
    """The /api/universes endpoint returns the dict of asset-class lists."""
    response = client_with_data.get("/api/universes")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == set(ASSET_UNIVERSE_NAMES)
    # Default (no universes configured) → every list is empty.
    for lst in payload.values():
        assert isinstance(lst, list)
