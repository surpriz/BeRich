"""Tests for the shared SequenceClassifier base and its LSTM / PatchTST subclasses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.models import (
    LSTMConfig,
    LSTMModel,
    PatchTSTConfig,
    PatchTSTModel,
    TFTConfig,
    TFTModel,
)
from berich.models.base import Model


def _panel(n_per_ticker: int = 80, n_features: int = 6, seed: int = 0):
    """Two-ticker date-sorted panel mimicking build_dataset output."""
    rng = np.random.default_rng(seed)
    frames, ys, tks = [], [], []
    for ticker in ("AAA", "BBB"):
        feats = rng.standard_normal((n_per_ticker, n_features))
        # Make the label weakly learnable from the features so AUC isn't degenerate.
        logits = feats[:, 0] - feats[:, 1]
        labels = (logits + rng.standard_normal(n_per_ticker) * 0.5 > 0).astype(float)
        frames.append(pd.DataFrame(feats, columns=[f"f{i}" for i in range(n_features)]))
        ys.append(pd.Series(labels))
        tks.append(pd.Series([ticker] * n_per_ticker))
    x = pd.concat(frames, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    tickers = pd.concat(tks, ignore_index=True)
    return x, y, tickers


def test_lstm_is_model_and_predicts_in_range():
    x, y, tickers = _panel()
    cfg = LSTMConfig(lookback=10, hidden=16, num_layers=1, epochs=3, batch_size=32, device="cpu")
    model = LSTMModel(cfg).fit(x, y, tickers=tickers)
    assert isinstance(model, Model)
    proba = model.predict_proba(x, tickers=tickers)
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_patchtst_is_model_and_predicts_in_range():
    x, y, tickers = _panel()
    cfg = PatchTSTConfig(
        lookback=24,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=3,
        batch_size=32,
        device="cpu",
    )
    model = PatchTSTModel(cfg).fit(x, y, tickers=tickers)
    assert isinstance(model, Model)
    proba = model.predict_proba(x, tickers=tickers)
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_tft_is_model_and_predicts_in_range():
    x, y, tickers = _panel()
    cfg = TFTConfig(
        lookback=20, d_model=16, n_heads=2, num_layers=1, epochs=3, batch_size=32, device="cpu"
    )
    model = TFTModel(cfg).fit(x, y, tickers=tickers)
    assert isinstance(model, Model)
    proba = model.predict_proba(x, tickers=tickers)
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_config_as_dict_excludes_device():
    cfg = PatchTSTConfig(device="cuda:1", patch_len=4)
    d = cfg.as_dict()
    assert "device" not in d
    assert d["patch_len"] == 4
    assert d["lookback"] == PatchTSTConfig().lookback


def test_tickers_keep_streams_separate():
    """A ticker absent from training falls back to 0.5 (no cross-ticker leakage)."""
    x, y, tickers = _panel()
    cfg = LSTMConfig(lookback=10, hidden=8, num_layers=1, epochs=2, device="cpu")
    model = LSTMModel(cfg).fit(x, y, tickers=tickers)
    unseen = pd.Series(["ZZZ"] * 20)
    new_x = x.iloc[:20].reset_index(drop=True)
    proba = model.predict_proba(new_x, tickers=unseen)
    assert np.allclose(proba, 0.5)
