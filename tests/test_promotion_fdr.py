"""Sweep-level FDR reconciliation: demote promotions that don't survive multiple-testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import AssetUniverses, Config
from berich.models import LGBMModel, ModelMetadata, load_active, promote, save_model
from berich.training.promotion import reconcile_sweep_fdr


def _model() -> LGBMModel:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 1, (120, 3)), columns=["a", "b", "c"])
    y = pd.Series((x["a"] + rng.normal(0, 0.1, 120) > 0).astype(int))
    return LGBMModel(n_estimators=10).fit(x, y)


def _promote_short(cfg: Config, ticker: str, *, pval: float) -> None:
    """Force-promote a short model for ``ticker`` with a given Sharpe p-value."""
    reg = cfg.model_dir_for_ticker(ticker, "short", "fixed")
    meta = ModelMetadata(
        name="lgbm-short",
        framework="lightgbm",
        feature_columns=["a", "b", "c"],
        metrics={"sharpe": 1.0, "deflated_sharpe": 0.97, "sharpe_pvalue": pval, "n_trades": 30.0},
        side="short",
        ticker=ticker,
    )
    save_model(_model(), meta, registry_dir=reg)
    promote("lgbm-short", registry_dir=reg, force=True)


def _cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path,
        universes=AssetUniverses(us_stocks=["AAA", "BBB", "CCC", "DDD"]),
    )


def test_no_promotions_is_noop(tmp_path):
    summary = reconcile_sweep_fdr(_cfg(tmp_path))
    assert summary["promoted_before"] == 0
    assert summary["demoted"] == 0


def test_weak_promotions_demoted_strong_kept(tmp_path):
    cfg = _cfg(tmp_path)
    # One genuinely-strong promotion + three null ones. BH at alpha=0.1 over m=4: only rank1
    # (0.001 <= 1/4*0.1=0.025) passes, so the three p=0.5 nulls are demoted.
    _promote_short(cfg, "AAA", pval=0.001)
    _promote_short(cfg, "BBB", pval=0.5)
    _promote_short(cfg, "CCC", pval=0.5)
    _promote_short(cfg, "DDD", pval=0.5)

    summary = reconcile_sweep_fdr(cfg, alpha=0.1)
    assert summary["promoted_before"] == 4
    assert summary["demoted"] == 3
    # The strong one survives (still promoted); the demoted ones lost their active pointer.
    assert load_active(cfg.model_dir_for_ticker("AAA", "short", "fixed")) is not None
    assert load_active(cfg.model_dir_for_ticker("BBB", "short", "fixed")) is None


def test_all_strong_promotions_kept(tmp_path):
    cfg = _cfg(tmp_path)
    for tkr in ("AAA", "BBB", "CCC"):
        _promote_short(cfg, tkr, pval=1e-4)
    summary = reconcile_sweep_fdr(cfg, alpha=0.1)
    assert summary["demoted"] == 0
    assert summary["promoted_after"] == 3
