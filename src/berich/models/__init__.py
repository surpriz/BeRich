"""Predictive models exposing a common ``fit`` / ``predict_proba`` interface."""

from berich.models.base import Model
from berich.models.ensemble import StackingEnsemble
from berich.models.lightgbm_model import LGBMModel
from berich.models.lightgbm_ranker import LGBMRanker
from berich.models.lstm import LSTMConfig, LSTMModel
from berich.models.meta_labeler import MetaLabeler
from berich.models.patchtst import PatchTSTConfig, PatchTSTModel
from berich.models.registry import (
    ModelMetadata,
    list_models,
    load_active,
    load_model,
    promote,
    save_model,
)
from berich.models.sequence_ranker import LSTMRanker, PatchTSTRanker, TFTRanker
from berich.models.tft import TFTConfig, TFTModel

__all__ = [
    "LGBMModel",
    "LGBMRanker",
    "LSTMConfig",
    "LSTMModel",
    "LSTMRanker",
    "MetaLabeler",
    "Model",
    "ModelMetadata",
    "PatchTSTConfig",
    "PatchTSTModel",
    "PatchTSTRanker",
    "StackingEnsemble",
    "TFTConfig",
    "TFTModel",
    "TFTRanker",
    "list_models",
    "load_active",
    "load_model",
    "promote",
    "save_model",
]
