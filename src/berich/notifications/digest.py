"""Assemble the structured payload behind the daily email digest.

Pure read + shape: this module never opens, closes or sizes a trade — it queries the existing
paper-book / signal / training read APIs and packs the result into a :class:`DailyDigest` that
``email.py`` renders. Keeping assembly here (data) separate from rendering there (presentation)
mirrors the rest of the codebase (``PaperStore`` owns CRUD; top-level functions own logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pandas as pd

from berich.signals.paper import (
    CURRENCY_CONCENTRATION_WARN_PCT,
    currency_concentration,
    get_equity_curve,
    get_open_positions,
    get_paper_metrics,
    recent_executions,
)

if TYPE_CHECKING:
    from berich.config import Config
    from berich.data.store import OhlcvStore
    from berich.signals.service import Signal

# A held asset whose last cached bar is older than this is being marked-to-market on frozen data —
# worth surfacing as a "good to know" line (the weekly monitor still owns the real alerting).
_STALE_DATA_DAYS = 7


@dataclass(frozen=True)
class DailyDigest:
    """Everything the daily email needs, already computed — the renderer only formats it."""

    date: str  # ISO "YYYY-MM-DD" of the run
    # --- portfolio snapshot (committed/promoted book) ---
    capital: float
    equity: float
    total_return_paper: float
    total_return_spy: float
    current_drawdown: float  # ≥ 0, peak-to-current as a fraction
    n_open: int
    n_closed_total: int
    win_rate: float
    # --- the morning copy list + today's activity (from recent_executions) ---
    opened: list[dict] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)
    adjust: list[dict] = field(default_factory=list)
    # --- good to know ---
    open_positions: list[dict] = field(default_factory=list)  # ticker/direction/mtm/days_held
    stale_tickers: list[str] = field(default_factory=list)
    # Currencies whose open forex exposure exceeds the concentration warning threshold —
    # several crosses sharing a leg are one correlated bet the per-class caps can't see.
    concentrated_currencies: list[dict] = field(default_factory=list)
    n_promoted_models: int = 0
    n_promoted_tickers: int = 0
    signals_total: int = 0
    longs_total: int = 0
    shorts_total: int = 0

    @property
    def has_activity(self) -> bool:
        """True when the run opened or closed at least one committed trade."""
        return bool(self.opened or self.closed)


def _current_drawdown(equity: pd.DataFrame) -> float:
    """Peak-to-current drawdown of the paper equity curve as a non-negative fraction."""
    if equity.empty:
        return 0.0
    series = pd.Series(equity["equity_paper"].to_numpy(), dtype=float).dropna()
    if series.empty:
        return 0.0
    peak = float(series.cummax().iloc[-1])
    last = float(series.iloc[-1])
    return max(0.0, 1.0 - last / peak) if peak else 0.0


def _promoted_counts(config: Config) -> tuple[int, int]:
    """(promoted models, distinct tickers) by counting ``active.json`` pointers on disk — cheap."""
    root = config.models_dir / "tickers"
    if not root.exists():
        return 0, 0
    pointers = list(root.rglob("active.json"))
    tickers = {p.relative_to(root).parts[0] for p in pointers}
    return len(pointers), len(tickers)


def _stale_held_tickers(store: OhlcvStore, positions: list[dict]) -> list[str]:
    """Held tickers whose cached OHLCV tip is older than ``_STALE_DATA_DAYS`` — frozen MTM risk."""
    today = pd.Timestamp.today().normalize()
    stale: list[str] = []
    for pos in positions:
        df = store.load(pos["ticker"])
        if df is None or df.empty:
            continue
        age_days = (today - df.index[-1].normalize()).days
        if age_days > _STALE_DATA_DAYS:
            stale.append(pos["ticker"])
    return stale


def build_daily_digest(config: Config, store: OhlcvStore, signals: list[Signal]) -> DailyDigest:
    """Gather the committed-book snapshot, the run's executions, and the good-to-know lines.

    Robust to an empty paper book (early forward test): every read API returns sane zeros, so the
    digest is always well-formed even before the first trade closes.
    """
    # Lazy like every other service-layer import in the scheduler path.
    from berich.signals.service import LONG_SIGNALS, SHORT  # noqa: PLC0415

    metrics = get_paper_metrics(config, store)
    equity = get_equity_curve(config, store)
    capital = float(metrics["capital"])
    executions = recent_executions(config, store)

    positions = [
        {
            "ticker": p.ticker,
            "direction": p.direction,
            "mtm_pct": p.mtm_pct,
            "mtm_eur": p.mtm_eur,
            "days_held": p.days_held,
            "exit_strategy": p.exit_strategy,
        }
        for p in get_open_positions(config, store)
    ]
    n_promoted_models, n_promoted_tickers = _promoted_counts(config)
    longs = sum(1 for s in signals if s.signal in LONG_SIGNALS)
    shorts = sum(1 for s in signals if s.signal == SHORT)

    return DailyDigest(
        date=str(pd.Timestamp.today().date()),
        capital=capital,
        equity=capital * (1.0 + float(metrics["total_return_paper"])),
        total_return_paper=float(metrics["total_return_paper"]),
        total_return_spy=float(metrics["total_return_spy"]),
        current_drawdown=_current_drawdown(equity),
        n_open=int(metrics["n_open"]),
        n_closed_total=int(metrics["n_closed"]),
        win_rate=float(metrics["win_rate"]),
        opened=executions["open"],
        closed=executions["close"],
        adjust=executions["adjust"],
        open_positions=positions,
        stale_tickers=_stale_held_tickers(store, positions),
        concentrated_currencies=[
            c
            for c in currency_concentration(config)
            if cast("float", c["pct_capital"]) >= CURRENCY_CONCENTRATION_WARN_PCT
        ],
        n_promoted_models=n_promoted_models,
        n_promoted_tickers=n_promoted_tickers,
        signals_total=len(signals),
        longs_total=longs,
        shorts_total=shorts,
    )
