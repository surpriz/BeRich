"""Test the Chronos optional-dependency guard (chronos is not installed in CI)."""

from __future__ import annotations

import importlib.util

import pytest

from berich.models.chronos_scorer import DEFAULT_MODEL, ChronosForecaster

_CHRONOS_INSTALLED = importlib.util.find_spec("chronos") is not None


def test_default_model_is_a_bolt_variant():
    assert "chronos" in DEFAULT_MODEL


@pytest.mark.skipif(_CHRONOS_INSTALLED, reason="chronos is installed; guard path not exercised")
def test_missing_chronos_raises_actionable_error():
    with pytest.raises(ImportError, match="chronos-forecasting"):
        ChronosForecaster()
