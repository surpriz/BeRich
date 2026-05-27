"""Risk/performance metrics computed from a daily-returns series.

Returns are simple daily portfolio returns (already net of costs). All ratios are
annualized with 252 trading days. These are the numbers that decide whether a model
is trustworthy: a high accuracy with a negative Sharpe or a brutal drawdown is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerfMetrics:
    """Headline backtest metrics for one strategy."""

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int

    def as_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_trades": float(self.n_trades),
        }


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Compound a daily-returns series into an equity curve starting at 1.0."""
    return (1.0 + daily_returns.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fractional drop of an equity curve (<= 0)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(daily_returns: pd.Series, *, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of a daily-returns series."""
    excess = daily_returns.dropna() - risk_free / TRADING_DAYS
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def compute_metrics(
    daily_returns: pd.Series,
    *,
    trade_returns: list[float] | None = None,
) -> PerfMetrics:
    """Build a :class:`PerfMetrics` from daily returns and (optional) trade returns."""
    daily_returns = daily_returns.dropna()
    equity = equity_curve(daily_returns)
    total = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    years = len(daily_returns) / TRADING_DAYS if len(daily_returns) else np.nan
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years and years > 0 else 0.0

    trades = trade_returns or []
    win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0

    return PerfMetrics(
        total_return=total,
        cagr=cagr,
        ann_vol=float(daily_returns.std() * np.sqrt(TRADING_DAYS)) if len(daily_returns) else 0.0,
        sharpe=sharpe_ratio(daily_returns),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        n_trades=len(trades),
    )
