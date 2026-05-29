"""Train / backtest the market-neutral long/short cross-sectional model.

Thin wrapper over the same path as ``berich longshort`` so the GPU box can run it
non-interactively. Example:
    uv run python scripts/train_longshort.py backtest --universe all
    uv run python scripts/train_longshort.py train --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from berich.cli import _cmd_longshort
from berich.config import DEFAULT_CONFIG_PATH


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Market-neutral long/short train/backtest")
    p.add_argument("action", choices=["train", "backtest"])
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--universe", choices=["mega", "mid", "small", "all"], default=None)
    p.add_argument("--name", default="longshort-ranker")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    return _cmd_longshort(args)


if __name__ == "__main__":
    sys.exit(main())
