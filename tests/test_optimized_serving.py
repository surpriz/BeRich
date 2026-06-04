"""The signal service must serve ONLY optimized assets, from their own models (no fallback)."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from berich.config import AssetUniverses, Config
from berich.data.store import OhlcvStore
from berich.datasets.assemble import build_ticker_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.signals.service import generate_signals
from berich.training.hpo import ticker_study_name


def _ohlcv(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    vol = rng.integers(900_000, 1_100_000, size=n)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": vol},
        index=idx,
    )


def _seed_optimized_long(cfg: Config, ticker: str, ohlcv: pd.DataFrame) -> None:
    """Give a ticker a real per-asset long model + 1-trial HPO study so it counts as optimized."""
    ds = build_ticker_dataset(ohlcv, LabelConfig(), ticker=ticker, market=ohlcv)
    model = LGBMModel(n_estimators=20).fit(ds.x, ds.y)
    reg = cfg.model_dir_for_ticker(ticker, "long")
    meta = ModelMetadata(
        name="lgbm-long",
        framework="lightgbm",
        feature_columns=list(ds.x.columns),
        metrics={"auc": 0.6, "sharpe": 1.0, "benchmark_sharpe": 0.2, "n_trades": 30.0},
        beats_buy_hold=True,
        ticker=ticker,
        side="long",
    )
    save_model(model, meta, registry_dir=reg)
    promote("lgbm-long", registry_dir=reg)
    # One Optuna trial so _optimized_tickers() recognizes it as worked-on.
    study = optuna.create_study(
        study_name=ticker_study_name(ticker, "lgbm", "long"),
        storage=f"sqlite:///{cfg.optuna_db}",
        direction="maximize",
        load_if_exists=True,
    )
    study.add_trial(optuna.trial.create_trial(value=0.6, params={}, distributions={}))


def test_only_optimized_assets_are_served(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    a, b = _ohlcv(0), _ohlcv(1)
    store.save("AAA", a)
    store.save("BBB", b)
    store.save("SPY", _ohlcv(2))
    cfg = Config(data_dir=tmp_path, universes=AssetUniverses(us_stocks=["AAA", "BBB"]))

    # Nothing optimized yet → no signals at all (clean dashboard).
    assert generate_signals(cfg, store) == []

    # Optimize only AAA → only AAA is served, and from its own promoted model.
    _seed_optimized_long(cfg, "AAA", a)
    sigs = generate_signals(cfg, store)
    tickers = {s.ticker for s in sigs}
    assert tickers == {"AAA"}
    assert sigs[0].promoted is True
