"""Single source of truth for the annualization factor (bars per year).

Daily-bar metrics annualize with 252 trading days. Intraday subsystems count in
bars, not calendar days: continuous 1-hour crypto bars run 24x365 = 8760 a year.
Centralizing the factor here removes the hardcoded ``sqrt(252)`` / ``/252`` that
otherwise scatters across metrics, significance and sizing — feeding the wrong
number inflates an intraday Sharpe by ~sqrt(8760/252) ~= 6x.
"""

from __future__ import annotations

DEFAULT_BARS_PER_YEAR = 252

_BARS_PER_YEAR: dict[str, int] = {
    "1d": 252,
    "1h": 24 * 365,  # 8760 — crypto trades continuously, no session/weekend skip
}


def bars_per_year(interval: str = "1d") -> int:
    """Bars per year for an interval string; falls back to the daily 252."""
    return _BARS_PER_YEAR.get(interval, DEFAULT_BARS_PER_YEAR)
