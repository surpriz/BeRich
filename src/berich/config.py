"""Typed configuration loaded from a YAML file.

A single :class:`Config` object is the entry point for every module: it knows the
watchlist, the data-fetch settings, the labeling parameters, and where the cache
lives. Loading is explicit (`Config.load(path)`) so tests can point at fixtures.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config/berich.yaml")


class DataConfig(BaseModel):
    """Market-data fetch settings."""

    interval: str = "1d"
    start_date: date = date(2010, 1, 1)
    auto_adjust: bool = True


class LabelingConfig(BaseModel):
    """Triple-barrier labeling parameters (consumed in Phase 1)."""

    horizon_days: int = 10
    atr_window: int = 14
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 1.0


class SignalConfig(BaseModel):
    """Daily signal thresholds and position-sizing parameters."""

    buy_threshold: float = 0.55
    sell_threshold: float = 0.30
    capital: float = 10_000.0
    risk_pct: float = 0.01


class Config(BaseModel):
    """Top-level project configuration."""

    data_dir: Path = Path("data")
    data: DataConfig = Field(default_factory=DataConfig)
    watchlist: list[str] = Field(default_factory=list)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)

    @property
    def ohlcv_dir(self) -> Path:
        """Directory holding one Parquet file per ticker."""
        return self.data_dir / "ohlcv"

    @property
    def db_path(self) -> Path:
        """DuckDB catalog for generated signals and run history."""
        return self.data_dir / "berich.duckdb"

    @property
    def models_dir(self) -> Path:
        """Registry directory holding trained model artifacts + metadata."""
        return self.data_dir / "models"

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
        """Load and validate configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
