"""Statistical significance of a Sharpe ratio for the market-neutral guard.

A market-neutral strategy has no buy-&-hold benchmark to beat — its bar is a Sharpe that
is *positive and unlikely to be luck*. After nine phases of trials the naive Sharpe is
selection-biased, so the promotion gate uses the **Deflated Sharpe Ratio** (Bailey &
López de Prado): the probability the true Sharpe is positive after correcting for the
number of trials, return skew, and kurtosis. A stationary-bootstrap p-value (which
respects autocorrelation in daily returns) is reported alongside as a robustness check.

All probabilities are in ``[0, 1]``. ``deflated_sharpe`` is the DSR probability; the guard
promotes when it clears ~0.95.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from berich.backtest.metrics import sharpe_ratio

if TYPE_CHECKING:
    import pandas as pd

TRADING_DAYS = 252
_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


@dataclass
class SharpeSignificance:
    """Significance verdict for a daily-returns series."""

    sharpe: float  # annualized
    t_stat: float  # PSR z-score (sr_period / se_sr)
    p_value: float  # one-sided P(true Sharpe <= 0)
    deflated_sharpe: float  # DSR probability in [0, 1]
    bootstrap_p_value: float  # stationary-bootstrap P(annualized Sharpe <= 0)

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "sharpe_t_stat": self.t_stat,
            "sharpe_pvalue": self.p_value,
            "deflated_sharpe": self.deflated_sharpe,
            "bootstrap_pvalue": self.bootstrap_p_value,
        }


def _expected_max_sr(se_sr: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe across ``n_trials`` independent trials."""
    if n_trials <= 1:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return se_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def _bootstrap_pvalue(returns: np.ndarray, *, avg_block: int, n_boot: int, seed: int) -> float:
    """Stationary-bootstrap P(annualized Sharpe <= 0) over ``n_boot`` resamples."""
    n = len(returns)
    if n < 3:  # noqa: PLR2004
        return float("nan")
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / max(avg_block, 1)
    non_positive = 0
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        restart = rng.random(n) < p_restart
        for i in range(1, n):
            idx[i] = rng.integers(n) if restart[i] else (idx[i - 1] + 1) % n
        sample = returns[idx]
        std = sample.std()
        sr = 0.0 if std == 0 else sample.mean() / std * math.sqrt(TRADING_DAYS)
        if sr <= 0:
            non_positive += 1
    return non_positive / n_boot


def assess_sharpe(
    daily_returns: pd.Series,
    *,
    n_trials: int = 1,
    avg_block: int = 10,
    n_boot: int = 500,
    seed: int = 42,
) -> SharpeSignificance:
    """Compute Sharpe + PSR/DSR + bootstrap significance for a daily-returns series."""
    r = daily_returns.dropna()
    n = len(r)
    ann_sharpe = sharpe_ratio(r)

    if n < 3:  # noqa: PLR2004
        return SharpeSignificance(
            ann_sharpe, float("nan"), float("nan"), float("nan"), float("nan")
        )

    arr = r.to_numpy(dtype=float)
    std = float(arr.std())
    if std == 0:
        return SharpeSignificance(
            ann_sharpe, float("nan"), float("nan"), float("nan"), float("nan")
        )

    sr_period = float(arr.mean()) / std  # non-annualized per-period Sharpe
    mean = float(arr.mean())
    skew = float(np.mean(((arr - mean) / std) ** 3))
    kurt = float(np.mean(((arr - mean) / std) ** 4))  # Pearson kurtosis (normal == 3)

    # Standard error of the Sharpe estimate (Lo / Mertens), used by PSR and DSR.
    se_sr = math.sqrt(
        max(1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period**2, 1e-9) / (n - 1)
    )

    t_stat = sr_period / se_sr
    p_value = 1.0 - _norm_cdf(t_stat)  # one-sided P(true Sharpe <= 0)

    sr0 = _expected_max_sr(se_sr, n_trials)
    deflated = _norm_cdf((sr_period - sr0) / se_sr)

    boot_p = _bootstrap_pvalue(arr, avg_block=avg_block, n_boot=n_boot, seed=seed)

    return SharpeSignificance(
        sharpe=ann_sharpe,
        t_stat=float(t_stat),
        p_value=float(p_value),
        deflated_sharpe=float(deflated),
        bootstrap_p_value=float(boot_p),
    )


__all__ = ["SharpeSignificance", "assess_sharpe"]
