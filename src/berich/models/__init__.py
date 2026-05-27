"""Predictive models exposing a common ``fit`` / ``predict_proba`` interface."""

from berich.models.base import Model
from berich.models.lightgbm_model import LGBMModel

__all__ = ["LGBMModel", "Model"]
