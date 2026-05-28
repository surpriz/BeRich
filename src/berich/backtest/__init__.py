"""Walk-forward backtest engine, portfolio combiner, and risk/perf metrics."""

from berich.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.backtest.portfolio import (
    PortfolioBacktestResult,
    run_portfolio_backtest,
)
from berich.backtest.strategies import (
    build_bnh_returns,
    build_calendar_returns,
    build_pead_returns,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PerfMetrics",
    "PortfolioBacktestResult",
    "build_bnh_returns",
    "build_calendar_returns",
    "build_pead_returns",
    "compute_metrics",
    "run_backtest",
    "run_portfolio_backtest",
]
