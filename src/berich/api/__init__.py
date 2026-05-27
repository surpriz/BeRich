"""FastAPI backend exposing signals, prices, backtest, and drift to the dashboard."""

from berich.api.app import create_app

__all__ = ["create_app"]
