"""Read-only inventory of per-asset training/optimization state for the dashboard.

Scans the per-ticker registry (``data/models/tickers/<TICKER>/<side>/``) and the Optuna RDB
(``data/optuna.db``) to answer: which assets have been trained, with which model, when, are
they promoted or advisory-only, and has a per-asset HPO search actually run. Nothing here
trains or mutates — it's the backing data for the ``/api/training`` endpoint and its tab.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from typing import TYPE_CHECKING, cast

from berich.config import safe_ticker_slug
from berich.models.registry import ACTIVE_POINTER, META_FILE, ModelMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from berich.config import Config

logger = logging.getLogger(__name__)

_STATUS_FILE = "status.json"
_SIDES = ("long", "short")
_STRATEGIES = ("fixed", "trailing", "trailing_tp")


def _hpo_trial_counts(optuna_db: Path) -> dict[str, int]:
    """Map per-ticker study name -> completed-trial count (empty if the RDB is absent)."""
    if not optuna_db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{optuna_db}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.warning("could not open optuna db at %s", optuna_db, exc_info=True)
        return {}
    try:
        rows = con.execute(
            "SELECT st.study_name, count(t.trial_id) "
            "FROM studies st LEFT JOIN trials t ON t.study_id = st.study_id "
            "GROUP BY st.study_name"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    return {name: int(n) for name, n in rows}


# Intraday timeframe segments a study name may carry (``berich-hpo-...-<side>[-<strategy>]
# [-<interval>]-<metric>``). Daily studies carry NO interval segment, so a daily query must
# exclude these — otherwise an intraday (e.g. ``-1h-``) study pollutes the daily trial count.
_INTRADAY_SEGMENTS = ("1h", "4h")


def _study_matches(
    name: str,
    slug: str,
    side: str,
    model: str | None,
    strategy: str | None,
    interval: str = "1d",
) -> bool:
    """Whether an Optuna study name belongs to this (ticker, side[, model][, strategy], interval).

    Study names are ``berich-hpo-<SLUG>-<model>-<side>[-<strategy>][-<interval>]-<metric>`` (fixed
    has no strategy segment, daily has no interval segment), so ``strategy="fixed"`` matches names
    with no trailing strategy segment, a trailing strategy matches names carrying its segment, and
    ``interval="1d"`` matches names with NO intraday segment (an intraday timeframe matches only its
    own ``-<interval>-`` segment). ``model``/``strategy`` are ``None`` = any.
    """
    if not name.startswith(f"berich-hpo-{slug}-") or f"-{side}-" not in name:
        return False
    if model is not None and f"-{model}-" not in name:
        return False
    if strategy == "fixed" and ("-trailing-" in name or "-trailing_tp-" in name):
        return False
    if strategy not in (None, "fixed") and f"-{strategy}-" not in name:
        return False
    # Interval gate: daily excludes any intraday segment; an intraday interval requires its own.
    if interval == "1d":
        return not any(f"-{seg}-" in name for seg in _INTRADAY_SEGMENTS)
    return f"-{interval}-" in name


def _hpo_trials_for(
    counts: dict[str, int],
    ticker: str,
    model: str | None,
    side: str,
    strategy: str | None = None,
    interval: str = "1d",
) -> int:
    """Sum HPO trials across that ticker+side's studies (see ``_study_matches`` for filtering)."""
    slug = safe_ticker_slug(ticker)
    return sum(
        n
        for name, n in counts.items()
        if _study_matches(name, slug, side, model, strategy, interval)
    )


def _naive_local_to_utc_iso(raw: str | None) -> str | None:
    """Normalize an Optuna trial timestamp (naive, server-local) to an ISO-UTC string.

    Optuna stores ``datetime_complete`` without a timezone, in the server's local time, whereas
    the registry's ``trained_at`` is UTC-aware — so we anchor the naive value to the local zone
    and convert, keeping every timestamp the dashboard sees on a single (UTC) clock.
    """
    if not raw:
        return None
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(datetime.UTC).isoformat()


def _hpo_last_trial_times(optuna_db: Path) -> dict[str, str]:
    """Map per-ticker study name -> ISO-UTC timestamp of its most recent completed trial."""
    if not optuna_db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{optuna_db}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.warning("could not open optuna db at %s", optuna_db, exc_info=True)
        return {}
    try:
        rows = con.execute(
            "SELECT st.study_name, max(t.datetime_complete) "
            "FROM studies st JOIN trials t ON t.study_id = st.study_id "
            "WHERE t.datetime_complete IS NOT NULL "
            "GROUP BY st.study_name"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    times: dict[str, str] = {}
    for name, raw in rows:
        iso = _naive_local_to_utc_iso(raw)
        if iso is not None:
            times[name] = iso
    return times


def _hpo_last_for(
    times: dict[str, str],
    ticker: str,
    model: str | None,
    side: str,
    strategy: str | None = None,
    interval: str = "1d",
) -> str | None:
    """Most recent HPO trial completion (ISO-UTC) for that ticker+side, or None.

    Same study-name matching as ``_hpo_trials_for`` (including the interval gate, so an intraday
    timeframe reads only its own studies); ISO-UTC strings share one format so the lexicographic
    max is the chronological max.
    """
    slug = safe_ticker_slug(ticker)
    stamps = [
        ts
        for name, ts in times.items()
        if _study_matches(name, slug, side, model, strategy, interval)
    ]
    return max(stamps) if stamps else None


def hpo_combo_sort_key(
    counts: dict[str, int],
    times: dict[str, str],
    ticker: str,
    side: str,
    strategy: str | None = None,
    interval: str = "1d",
) -> tuple[bool, str]:
    """Re-fit ordering key for one (ticker, side, strategy, interval) combo in the continuous sweep.

    Sorts un-searched combos first — an asset with no trained model yet is effectively infinitely
    stale, so it must get its first deep HPO before anything is re-deepened — then already-searched
    combos oldest-HPO-first, so the most stale model is always the next re-fit. Reading the real
    last-trial time from the Optuna RDB (rather than a fixed config order) bounds the worst-case
    staleness and is robust to restarts: the sweep resumes on the genuinely oldest combo instead of
    redoing the top of the list while the tail starves.
    """
    searched = _hpo_trials_for(counts, ticker, None, side, strategy, interval) > 0
    last = _hpo_last_for(times, ticker, None, side, strategy, interval) or ""
    return (searched, last)


def _strategy_entry(
    config: Config,
    ticker: str,
    side: str,
    strategy: str,
    counts: dict[str, int],
    times: dict[str, str],
) -> dict[str, object] | None:
    """Status of one (ticker, side, exit strategy), or ``None`` if that strategy isn't trained.

    Mirrors the per-asset tournament verdict for a single exit-strategy namespace: promoted
    (cleared the guard) / advisory_only (saved but unpromoted) / never_trained, plus the winner,
    headline metrics, horizon, this strategy's own HPO trial count, and the full candidate slate.
    """
    reg = config.model_dir_for_ticker(ticker, side, strategy)
    if not reg.exists():
        return None
    item: dict[str, object] = {
        "strategy": strategy,
        "status": "never_trained",
        "winner": None,
        "framework": None,
        "trained_at": None,
        "metrics": {},
        "candidates": [],
        "horizon_days": None,
        "hpo_trials": _hpo_trials_for(counts, ticker, None, side, strategy),
        "last_hpo_at": _hpo_last_for(times, ticker, None, side, strategy),
    }
    status_path = reg / _STATUS_FILE
    if status_path.exists():
        try:
            summary = json.loads(status_path.read_text(encoding="utf-8"))
            item["trained_at"] = summary.get("trained_at")
            item["candidates"] = summary.get("candidates", [])
        except (OSError, ValueError):
            logger.warning("unreadable status.json for %s/%s/%s", ticker, side, strategy)

    if (reg / ACTIVE_POINTER).exists():
        promoted = _promoted_meta(reg, ticker, side)
        if promoted is not None:
            _fill_from_meta(item, promoted, status="promoted", winner=promoted.name)
            return item

    # No promoted pointer: advisory-only if a candidate artifact was saved. Exclude the nested
    # trailing-strategy subdirs (they have no META_FILE of their own) so the fixed namespace
    # doesn't count them as its own candidates.
    saved = [d for d in reg.iterdir() if d.is_dir() and (d / META_FILE).exists()]
    if saved or item["candidates"]:
        item["status"] = "advisory_only"
        if saved and not item["metrics"]:
            meta = _read_meta(saved[0])
            if meta is not None:
                _fill_from_meta(item, meta, status="advisory_only", winner=None)
    return item


def _served_strategy(strategies: list[dict[str, object]], side: str) -> dict[str, object] | None:
    """The exit strategy that serves this (ticker, side): mirrors ``service._select_strategy``.

    Best PROMOTED strategy by guard metric (Sharpe long / deflated Sharpe short), ties → fixed;
    if none is promoted, the fixed advisory if present, else the first trained. ``None`` when no
    strategy has been trained yet.
    """
    trained = [s for s in strategies if s["status"] != "never_trained"]
    if not trained:
        return None
    metric_key = "deflated_sharpe" if side == "short" else "sharpe"

    def rank(s: dict[str, object]) -> tuple[float, bool]:
        metric = cast("dict[str, float]", s["metrics"]).get(metric_key, 0.0)
        return metric, s["strategy"] == "fixed"

    promoted = [s for s in trained if s["status"] == "promoted"]
    if promoted:
        return max(promoted, key=rank)
    return next((s for s in trained if s["strategy"] == "fixed"), trained[0])


def _side_entry(
    config: Config, ticker: str, side: str, counts: dict[str, int], times: dict[str, str]
) -> dict[str, object]:
    """Build one (ticker, side) row across all exit strategies.

    The headline fields (status/winner/metrics/...) reflect the SERVED strategy (the one the
    signal service would pick); ``strategies`` carries every trained exit strategy so the
    dashboard can show fixed vs trailing vs trailing_tp side by side.
    """
    per_strategy = (_strategy_entry(config, ticker, side, st, counts, times) for st in _STRATEGIES)
    strategies = [s for s in per_strategy if s is not None]
    served = _served_strategy(strategies, side)
    return {
        "ticker": ticker,
        "asset_class": config.asset_class_for(ticker),
        "side": side,
        "status": served["status"] if served else "never_trained",
        "winner": served["winner"] if served else None,
        "framework": served["framework"] if served else None,
        "trained_at": served["trained_at"] if served else None,
        "metrics": served["metrics"] if served else {},
        "candidates": served["candidates"] if served else [],
        "hpo_trials": _hpo_trials_for(counts, ticker, None, side),
        "last_hpo_at": _hpo_last_for(times, ticker, None, side),
        "horizon_days": served["horizon_days"] if served else None,
        "served_strategy": served["strategy"] if served else "fixed",
        "strategies": strategies,
    }


def _read_meta(artifact_dir: Path) -> ModelMetadata | None:
    """Load one artifact's metadata, or None if unreadable."""
    try:
        return ModelMetadata.model_validate_json(
            (artifact_dir / META_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _promoted_meta(reg: Path, ticker: str, side: str) -> ModelMetadata | None:
    """Metadata of the active (promoted) artifact under ``reg``, or None if unreadable."""
    try:
        name = json.loads((reg / ACTIVE_POINTER).read_text(encoding="utf-8"))["name"]
    except (OSError, ValueError, KeyError):
        logger.warning("unreadable active pointer for %s/%s", ticker, side, exc_info=True)
        return None
    return _read_meta(reg / name)


def _fill_from_meta(
    entry: dict[str, object], meta: ModelMetadata, *, status: str, winner: str | None
) -> None:
    """Populate an entry's status/framework/metrics/trained_at from a model's metadata."""
    entry["status"] = status
    if winner is not None:
        entry["winner"] = winner
    entry["framework"] = meta.framework
    entry["metrics"] = meta.metrics
    entry["horizon_days"] = meta.horizon_days
    if entry["trained_at"] is None:
        entry["trained_at"] = meta.created_at


def training_status(config: Config, *, optimized_only: bool = False) -> list[dict[str, object]]:
    """Per-(ticker, side) training inventory across every configured tradeable asset.

    ``optimized_only`` keeps only assets that have had their per-asset HPO run (an Optuna
    study with >=1 trial on either side) — the set the dashboard's /training tab shows, so it
    never lists assets we haven't actually worked on. The default (full scan) backs /ops,
    which needs the global done/pending counts.
    """
    counts = _hpo_trial_counts(config.optuna_db)
    times = _hpo_last_trial_times(config.optuna_db)
    rows = [
        _side_entry(config, ticker, side, counts, times)
        for ticker in config.tradeable_tickers()
        for side in _SIDES
    ]
    if not optimized_only:
        return rows
    optimized = {r["ticker"] for r in rows if cast("int", r["hpo_trials"]) > 0}
    return [r for r in rows if r["ticker"] in optimized]


__all__ = ["training_status"]
