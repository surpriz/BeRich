"""Tests for ingestion helpers that don't require network access."""

from __future__ import annotations

import pandas as pd

from berich.data.ingest import _check_integrity


def _frame(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        },
        index=idx,
    )


def test_integrity_clean_series_has_no_warnings():
    df = _frame(["2024-01-02", "2024-01-03", "2024-01-04"])
    assert _check_integrity(df) == []


def test_integrity_flags_nans():
    df = _frame(["2024-01-02", "2024-01-03"])
    df.loc["2024-01-03", "close"] = float("nan")
    warnings = _check_integrity(df)
    assert any("NaN" in w for w in warnings)


def test_integrity_flags_large_gap():
    # ~3 week gap between bars.
    df = _frame(["2024-01-02", "2024-01-25"])
    warnings = _check_integrity(df)
    assert any("gap" in w for w in warnings)
