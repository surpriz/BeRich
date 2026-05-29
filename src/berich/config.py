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
    # Adaptive SL/TP: scale the ATR barriers by a vol forecast (and use a model's return
    # quantiles when available) instead of fixed multiples. On by default — it improves
    # the advice without changing the model.
    adaptive_sltp: bool = True
    vol_method: str = "ewma"  # "ewma" | "garch"
    # Meta-labeling precision filter: when on and a meta artifact exists, a BUY is
    # downgraded to NEUTRAL if the meta model's P(signal correct) < meta_threshold.
    use_meta_label: bool = False
    meta_threshold: float = 0.5


class LongShortConfigModel(BaseModel):
    """Market-neutral long/short cross-sectional parameters (the ``longshort`` block)."""

    universe: str = "all"  # cross-sectional ranking needs breadth, not the 10-name mega set
    market_ticker: str = "SPY"
    # Label
    horizon_days: int = 5
    beta_window: int = 60
    residualize: bool = True
    standardize: str = "zscore"  # "zscore" | "rank"
    min_names_per_date: int = 20
    cross_sectional_features: bool = True  # append within-date relative (*_xs) features
    # Portfolio
    top_decile: float = 0.1
    bottom_decile: float = 0.1
    weighting: str = "inverse_vol"  # "equal" | "inverse_vol"
    rebalance_days: int = 5
    gross_leverage: float = 1.0
    target_vol: float = 0.10
    vol_lookback: int = 20
    fee_bps: float = 1.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    # Significance: deflate the Sharpe for the many trials already run (see docs/RESULTS.md).
    n_trials: int = 9


class ZooConfig(BaseModel):
    """Model-zoo training / nightly-retrain configuration."""

    enabled_models: list[str] = Field(default_factory=lambda: ["lgbm", "patchtst"])
    gpu_ids: list[int] | None = None  # None = all visible GPUs
    hpo_trials: int = 20


UNIVERSE_NAMES = ("mega", "mid", "small", "all")

# Phase polish v2 — multi-asset universes. The model was trained on US stocks
# only; everything below is best-effort signal generation with the same
# pipeline. The frontend shows an "experimental" banner on non-US universes
# so users know the validation envelope.
ASSET_UNIVERSE_NAMES = ("us_stocks", "fr_stocks", "forex", "crypto", "commodities")
US_STOCKS_UNIVERSE = "us_stocks"


class AssetUniverses(BaseModel):
    """Per-asset-class ticker lists. Empty lists are fine — they're just skipped."""

    us_stocks: list[str] = Field(default_factory=list)
    fr_stocks: list[str] = Field(default_factory=list)
    forex: list[str] = Field(default_factory=list)
    crypto: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)

    def get(self, name: str) -> list[str]:
        """Return the ticker list for ``name`` (raises on unknown universe)."""
        if name not in ASSET_UNIVERSE_NAMES:
            msg = f"unknown asset universe '{name}' (expected one of {ASSET_UNIVERSE_NAMES})"
            raise ValueError(msg)
        return list(getattr(self, name))

    def all_tickers(self) -> list[str]:
        """Union of every populated universe, deduped by upper(ticker)."""
        seen: set[str] = set()
        out: list[str] = []
        for name in ASSET_UNIVERSE_NAMES:
            for ticker in self.get(name):
                upper = ticker.upper()
                if upper in seen:
                    continue
                seen.add(upper)
                out.append(ticker)
        return out

    def asset_class(self, ticker: str) -> str | None:
        """Reverse-lookup: which universe owns ``ticker``? Returns None if absent."""
        upper = ticker.upper()
        for name in ASSET_UNIVERSE_NAMES:
            for member in self.get(name):
                if member.upper() == upper:
                    return name
        return None


class Config(BaseModel):
    """Top-level project configuration."""

    data_dir: Path = Path("data")
    data: DataConfig = Field(default_factory=DataConfig)
    watchlist: list[str] = Field(default_factory=list)
    # Phase 6 — wider universes. Either may be empty; the helpers below treat
    # missing entries as "no extra tickers".
    mid_cap_universe: list[str] = Field(default_factory=list)
    small_cap_universe: list[str] = Field(default_factory=list)
    # Phase polish v2 — multi-asset. When populated, the union of every
    # universe is added to the runtime ticker set; ``watchlist`` is kept for
    # backward compat with the daily-LGBM signals path (US stocks only).
    universes: AssetUniverses = Field(default_factory=AssetUniverses)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    longshort: LongShortConfigModel = Field(default_factory=LongShortConfigModel)
    zoo: ZooConfig = Field(default_factory=ZooConfig)

    def tickers_for_universe(self, name: str) -> list[str]:
        """Resolve ``mega | mid | small | all`` to a deduplicated list of tickers.

        ``mega`` aliases to the existing ``watchlist`` so the default 10-ticker
        production behavior is unchanged when the universe arg isn't passed.
        ``all`` is the union of every populated universe (order: mega → mid →
        small) with stable ordering and dedup on first occurrence.
        """
        if name == "mega":
            return list(self.watchlist)
        if name == "mid":
            return list(self.mid_cap_universe)
        if name == "small":
            return list(self.small_cap_universe)
        if name == "all":
            seen: set[str] = set()
            out: list[str] = []
            for ticker in [*self.watchlist, *self.mid_cap_universe, *self.small_cap_universe]:
                upper = ticker.upper()
                if upper in seen:
                    continue
                seen.add(upper)
                out.append(ticker)
            return out
        msg = f"unknown universe '{name}' (expected one of {UNIVERSE_NAMES})"
        raise ValueError(msg)

    def all_runtime_tickers(self) -> list[str]:
        """Every ticker the daily scheduler should ingest.

        Union of the legacy ``watchlist`` (mega-cap, the only universe the model was
        actually trained on), all populated multi-asset ``universes``, and the per-class
        regime-proxy tickers (e.g. the dollar index for forex) so the market-context
        features stay fresh. Deduped by upper(ticker), first-occurrence order.
        """
        from berich.features.build import market_reference_for  # noqa: PLC0415 — avoid import cycle

        proxies = [
            market_reference_for(ac) for ac in ASSET_UNIVERSE_NAMES if self.universes.get(ac)
        ]
        seen: set[str] = set()
        out: list[str] = []
        for ticker in [*self.watchlist, *self.universes.all_tickers(), *proxies]:
            upper = ticker.upper()
            if upper in seen:
                continue
            seen.add(upper)
            out.append(ticker)
        return out

    def asset_class_for(self, ticker: str) -> str:
        """Map ``ticker`` to its asset class label for the dashboard / honesty banner.

        Defaults to ``us_stocks`` for anything in the legacy watchlist
        (which is what the model was trained on), and falls back to
        ``"unknown"`` only for tickers we can't classify at all — the
        frontend treats unknown the same as the multi-asset universes
        (i.e. shows the experimental banner).
        """
        explicit = self.universes.asset_class(ticker)
        if explicit is not None:
            return explicit
        for member in self.watchlist:
            if member.upper() == ticker.upper():
                return US_STOCKS_UNIVERSE
        return "unknown"

    @property
    def ohlcv_dir(self) -> Path:
        """Directory holding one Parquet file per ticker."""
        return self.data_dir / "ohlcv"

    @property
    def db_path(self) -> Path:
        """DuckDB catalog for generated signals and run history."""
        return self.data_dir / "berich.duckdb"

    @property
    def optuna_db(self) -> Path:
        """SQLite RDB backing Optuna studies (shared across GPU workers)."""
        return self.data_dir / "optuna.db"

    @property
    def models_dir(self) -> Path:
        """Registry directory holding trained model artifacts + metadata."""
        return self.data_dir / "models"

    def models_dir_for(self, asset_class: str) -> Path:
        """Registry directory namespaced by asset class.

        ``us_stocks`` keeps the historical root (``data/models/``) so existing
        artifacts and the daily-LGBM serving path are unchanged; every other class
        (and the long/short track) gets its own subdirectory.
        """
        if asset_class in ("us_stocks", "mega", ""):
            return self.models_dir
        return self.models_dir / asset_class

    @property
    def earnings_dir(self) -> Path:
        """Per-ticker earnings cache (Phase 5a). Empty by default until populated."""
        return self.data_dir / "earnings"

    @property
    def news_dir(self) -> Path:
        """Per-ticker news cache (Phase 5b). Empty by default until populated."""
        return self.data_dir / "news"

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
        """Load and validate configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
