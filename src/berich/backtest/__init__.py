"""Walk-forward backtest engine, portfolio combiner, and risk/perf metrics."""

from berich.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from berich.backtest.longshort import (
    LongShortConfig,
    LongShortResult,
    build_baskets,
    run_longshort_backtest,
)
from berich.backtest.metrics import PerfMetrics, compute_metrics
from berich.backtest.portfolio import (
    PortfolioBacktestResult,
    run_portfolio_backtest,
)
from berich.backtest.significance import SharpeSignificance, assess_sharpe
from berich.backtest.strategies import (
    build_bnh_returns,
    build_calendar_returns,
    build_pead_returns,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LongShortConfig",
    "LongShortResult",
    "PerfMetrics",
    "PortfolioBacktestResult",
    "SharpeSignificance",
    "assess_sharpe",
    "build_baskets",
    "build_bnh_returns",
    "build_calendar_returns",
    "build_pead_returns",
    "compute_metrics",
    "run_backtest",
    "run_longshort_backtest",
    "run_portfolio_backtest",
]
