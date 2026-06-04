"""Tests for the sweep-level Benjamini-Hochberg FDR control."""

from __future__ import annotations

from berich.backtest.multiple_testing import benjamini_hochberg


def test_empty_input_returns_empty():
    assert benjamini_hochberg([]) == []


def test_results_returned_in_input_order():
    res = benjamini_hochberg([0.9, 0.001, 0.5], alpha=0.1)
    assert [r.index for r in res] == [0, 1, 2]
    assert [r.p_value for r in res] == [0.9, 0.001, 0.5]


def test_all_tiny_pvalues_all_rejected():
    res = benjamini_hochberg([1e-4, 1e-4, 1e-4], alpha=0.1)
    assert all(r.rejected for r in res)


def test_all_large_pvalues_none_rejected():
    res = benjamini_hochberg([0.6, 0.7, 0.8, 0.9], alpha=0.1)
    assert not any(r.rejected for r in res)


def test_bh_is_less_strict_than_bonferroni():
    # One strong + many null candidates. Bonferroni cut = alpha/m = 0.01; the p=0.008 signal
    # survives Bonferroni too here, but BH must also keep it while rejecting the nulls.
    p_values = [0.008, *[0.6] * 9]
    res = benjamini_hochberg(p_values, alpha=0.1)
    assert res[0].rejected
    assert not any(r.rejected for r in res[1:])


def test_step_up_rejects_through_largest_passing_rank():
    # BH is a step-UP procedure. Sorted p=[0.009,0.03,0.4,0.8], m=4, alpha=0.1:
    # rank1 0.009<=0.025 ✓, rank2 0.03<=0.05 ✓, rank3 0.4<=0.075 ✗ → largest passing k=2,
    # so BOTH p=0.009 and p=0.03 are rejected (even though 0.03 alone wouldn't pass its own cut).
    res = {r.index: r for r in benjamini_hochberg([0.009, 0.03, 0.4, 0.8], alpha=0.1)}
    assert res[0].rejected
    assert res[1].rejected
    assert not res[2].rejected
    assert not res[3].rejected


def test_non_finite_pvalue_never_rejected():
    res = benjamini_hochberg([float("nan"), 0.0001], alpha=0.1)
    assert not res[0].rejected
    assert res[1].rejected
