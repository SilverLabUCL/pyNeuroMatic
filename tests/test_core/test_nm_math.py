"""Tests for pyneuromatic.core.nm_math."""
from __future__ import annotations

import math

import numpy as np
import pytest

import pyneuromatic.core.nm_math as nm_math
from pyneuromatic.core.nm_math import (
    VALID_ARITH_OPS,
    VALID_INEQUALITY_OPS,
    apply_arithmetic,
    apply_inequality,
    array_stats,
    compute_ref_value,
    find_level_crossings,
    inequality_condition_str,
    inequality_mask,
    interp_x,
    linear_regression,
    time_window_to_slice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARR = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# TestValidInequalityOps
# ---------------------------------------------------------------------------


class TestValidInequalityOps:
    def test_single_ops(self):
        assert {">", ">=", "<", "<=", "==", "!="}.issubset(VALID_INEQUALITY_OPS)

    def test_range_ops(self):
        assert {"<<", "<=<=", "<=<", "<<="}.issubset(VALID_INEQUALITY_OPS)

    def test_total_count(self):
        assert len(VALID_INEQUALITY_OPS) == 10

    def test_unknown_op_not_in_set(self):
        assert "^" not in VALID_INEQUALITY_OPS
        assert ">" in VALID_INEQUALITY_OPS


# ---------------------------------------------------------------------------
# TestInequalityMask
# ---------------------------------------------------------------------------


class TestInequalityMask:
    def test_gt(self):
        mask = inequality_mask(ARR, ">", 3.0)
        np.testing.assert_array_equal(mask, [False, False, False, True, True])

    def test_gte(self):
        mask = inequality_mask(ARR, ">=", 3.0)
        np.testing.assert_array_equal(mask, [False, False, True, True, True])

    def test_lt(self):
        mask = inequality_mask(ARR, "<", 3.0)
        np.testing.assert_array_equal(mask, [True, True, False, False, False])

    def test_lte(self):
        mask = inequality_mask(ARR, "<=", 3.0)
        np.testing.assert_array_equal(mask, [True, True, True, False, False])

    def test_eq(self):
        mask = inequality_mask(ARR, "==", 3.0)
        np.testing.assert_array_equal(mask, [False, False, True, False, False])

    def test_neq(self):
        mask = inequality_mask(ARR, "!=", 3.0)
        np.testing.assert_array_equal(mask, [True, True, False, True, True])

    def test_range_open_open(self):
        # a < y < b  →  "<<"
        mask = inequality_mask(ARR, "<<", 1.0, 4.0)
        np.testing.assert_array_equal(mask, [False, True, True, False, False])

    def test_range_closed_closed(self):
        # a <= y <= b  →  "<=<="
        mask = inequality_mask(ARR, "<=<=", 2.0, 4.0)
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_range_closed_open(self):
        # a <= y < b  →  "<=<"
        mask = inequality_mask(ARR, "<=<", 2.0, 4.0)
        np.testing.assert_array_equal(mask, [False, True, True, False, False])

    def test_range_open_closed(self):
        # a < y <= b  →  "<<="
        mask = inequality_mask(ARR, "<<=", 2.0, 4.0)
        np.testing.assert_array_equal(mask, [False, False, True, True, False])

    def test_nan_is_false(self):
        arr = np.array([1.0, float("nan"), 3.0])
        mask = inequality_mask(arr, ">", 0.0)
        assert mask[0] is np.bool_(True)
        assert not mask[1]  # NaN comparison → False
        assert mask[2] is np.bool_(True)

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="unknown operator"):
            inequality_mask(ARR, "^", 1.0)

    def test_range_op_without_b_raises(self):
        with pytest.raises(ValueError, match="requires b"):
            inequality_mask(ARR, "<<", 1.0)  # b not provided


# ---------------------------------------------------------------------------
# TestApplyInequality
# ---------------------------------------------------------------------------


class TestApplyInequality:
    def test_binary_output_true(self):
        result = apply_inequality(ARR, ">", 3.0, binary_output=True)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0, 1.0])
        assert result.dtype == float

    def test_binary_output_false(self):
        result = apply_inequality(ARR, ">", 3.0, binary_output=False)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert math.isnan(result[2])
        assert result[3] == 4.0
        assert result[4] == 5.0


# ---------------------------------------------------------------------------
# TestInequalityConditionStr
# ---------------------------------------------------------------------------


class TestInequalityConditionStr:
    def test_gt(self):
        assert inequality_condition_str(">", 5.0, None) == "y > 5"

    def test_gte(self):
        assert inequality_condition_str(">=", 5.0, None) == "y >= 5"

    def test_lt(self):
        assert inequality_condition_str("<", 5.0, None) == "y < 5"

    def test_lte(self):
        assert inequality_condition_str("<=", 5.0, None) == "y <= 5"

    def test_eq(self):
        assert inequality_condition_str("==", 0.0, None) == "y == 0"

    def test_neq(self):
        assert inequality_condition_str("!=", 0.0, None) == "y != 0"

    def test_range_open_open(self):
        assert inequality_condition_str("<<", 2.0, 5.0) == "2 < y < 5"

    def test_range_closed_closed(self):
        assert inequality_condition_str("<=<=", 2.0, 5.0) == "2 <= y <= 5"

    def test_range_closed_open(self):
        assert inequality_condition_str("<=<", 2.0, 5.0) == "2 <= y < 5"

    def test_range_open_closed(self):
        assert inequality_condition_str("<<=", 2.0, 5.0) == "2 < y <= 5"


# ---------------------------------------------------------------------------
# TestApplyArithmetic
# ---------------------------------------------------------------------------


class TestApplyArithmetic:
    ARR = np.array([2.0, 4.0, 6.0])

    def test_multiply(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 2.0, "x"), [4.0, 8.0, 12.0])

    def test_divide(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 2.0, "/"), [1.0, 2.0, 3.0])

    def test_add(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 1.0, "+"), [3.0, 5.0, 7.0])

    def test_subtract(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 1.0, "-"), [1.0, 3.0, 5.0])

    def test_assign(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 9.0, "="), [9.0, 9.0, 9.0])

    def test_exponentiate(self):
        np.testing.assert_array_equal(apply_arithmetic(self.ARR, 2.0, "**"), [4.0, 16.0, 36.0])

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            apply_arithmetic(self.ARR, 1.0, "^")

    def test_valid_arith_ops_constant(self):
        assert VALID_ARITH_OPS == {"x", "/", "+", "-", "=", "**"}


# ---------------------------------------------------------------------------
# TestComputeRefValue
# ---------------------------------------------------------------------------


class TestComputeRefValue:
    ARR = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_mean(self):
        assert compute_ref_value(self.ARR, "mean", 1) == pytest.approx(3.0)

    def test_min(self):
        assert compute_ref_value(self.ARR, "min", 1) == 1.0

    def test_max(self):
        assert compute_ref_value(self.ARR, "max", 1) == 5.0

    def test_mean_at_min(self):
        # min is at index 0; n_mean=1 → just arr[0]
        val = compute_ref_value(self.ARR, "mean@min", 1)
        assert val == pytest.approx(1.0)

    def test_mean_at_max(self):
        # max is at index 4; n_mean=3 → mean of arr[3:6] = mean([4,5]) = 4.5
        val = compute_ref_value(self.ARR, "mean@max", 3)
        assert val == pytest.approx(4.5)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_ref_value(np.array([]), "mean", 1))

    def test_unknown_fxn_returns_nan(self):
        assert math.isnan(compute_ref_value(self.ARR, "unknown", 1))


# ---------------------------------------------------------------------------
# TestTimeWindowToSlice
# ---------------------------------------------------------------------------


class TestTimeWindowToSlice:
    ARR = np.zeros(10)
    XD = {"start": 0.0, "delta": 1.0}

    def test_basic_slice(self):
        s = time_window_to_slice(self.ARR, self.XD, 2.0, 4.0)
        assert s == slice(2, 5)

    def test_clips_low(self):
        s = time_window_to_slice(self.ARR, self.XD, -5.0, 3.0)
        assert s.start == 0

    def test_clips_high(self):
        s = time_window_to_slice(self.ARR, self.XD, 7.0, 20.0)
        assert s.stop == len(self.ARR)

    def test_zero_delta_returns_empty(self):
        xd = {"start": 0.0, "delta": 0.0}
        s = time_window_to_slice(self.ARR, xd, 0.0, 5.0)
        assert s == slice(0, 0)

    def test_non_unit_delta(self):
        xd = {"start": 0.0, "delta": 0.5}
        arr = np.zeros(20)
        s = time_window_to_slice(arr, xd, 1.0, 2.0)
        # i0 = round((1.0-0)/0.5)=2, i1 = round((2.0-0)/0.5)+1=5
        assert s == slice(2, 5)


# ---------------------------------------------------------------------------
# TestArrayStats
# ---------------------------------------------------------------------------


class TestArrayStats:
    def test_returns_dict_with_expected_keys(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = array_stats(arr)
        for key in ("mean", "std", "sem", "rms", "N", "NaNs", "INFs", "min", "max"):
            assert key in r

    def test_mean(self):
        arr = np.array([2.0, 4.0, 6.0])
        r = array_stats(arr)
        assert r["mean"] == pytest.approx(4.0)

    def test_min_max(self):
        arr = np.array([1.0, 5.0, 3.0])
        r = array_stats(arr)
        assert r["min"] == 1.0
        assert r["max"] == 5.0

    def test_nan_count(self):
        arr = np.array([1.0, float("nan"), 3.0])
        r = array_stats(arr)
        assert r["NaNs"] == 1

    def test_ignore_nans(self):
        arr = np.array([2.0, float("nan"), 4.0])
        r = array_stats(arr, ignore_nans=True)
        assert r["N"] == 2
        assert r["mean"] == pytest.approx(3.0)

    def test_empty_after_nan_removal(self):
        arr = np.array([float("nan")])
        r = array_stats(arr, ignore_nans=True)
        assert r["N"] == 0
        assert math.isnan(r["mean"])

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            array_stats([1.0, 2.0])


# ---------------------------------------------------------------------------
# TestInterpX
# ---------------------------------------------------------------------------


class TestInterpX:
    def test_midpoint(self):
        # line from (0,0) to (2,2); crosses y=1 at x=1
        assert interp_x(1.0, 0.0, 0.0, 2.0, 2.0) == pytest.approx(1.0)

    def test_crossing_below_to_above(self):
        # line from (0,-1) to (1,1); crosses y=0 at x=0.5
        assert interp_x(0.0, 0.0, -1.0, 1.0, 1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestFindLevelCrossings
# ---------------------------------------------------------------------------


class TestFindLevelCrossings:
    def test_returns_tuple_of_arrays(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, xv = find_level_crossings(arr, 0.0)
        assert isinstance(idx, np.ndarray)
        assert isinstance(xv, np.ndarray)
        assert len(idx) == len(xv)

    def test_no_crossings(self):
        arr = np.array([1.0, 2.0, 3.0])
        idx, xv = find_level_crossings(arr, -1.0)
        assert len(idx) == 0

    def test_rising_only(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, _ = find_level_crossings(arr, 0.0, func_name="level+")
        # two rising crossings
        assert len(idx) == 2

    def test_falling_only(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, _ = find_level_crossings(arr, 0.0, func_name="level-")
        # one falling crossing (1→-1 at index 2)
        assert len(idx) == 1

    def test_unknown_func_name_raises(self):
        with pytest.raises(ValueError):
            find_level_crossings(np.array([1.0, -1.0]), 0.0, func_name="bad")

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            find_level_crossings([1.0, -1.0], 0.0)


# ---------------------------------------------------------------------------
# TestLinearRegression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    def test_perfect_line(self):
        # y = 2x + 1
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        m, b = linear_regression(y, xarray=x)
        assert m == pytest.approx(2.0, abs=1e-10)
        assert b == pytest.approx(1.0, abs=1e-10)

    def test_uniform_xscale(self):
        x = np.linspace(0, 9, 10)
        y = 3 * x - 2
        m, b = linear_regression(y, xstart=0.0, xdelta=1.0)
        assert m == pytest.approx(3.0, abs=1e-10)
        assert b == pytest.approx(-2.0, abs=1e-10)

    def test_ignore_nans(self):
        y = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        m, b = linear_regression(y, xarray=x, ignore_nans=True)
        assert m == pytest.approx(1.0, abs=1e-10)

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            linear_regression([1.0, 2.0, 3.0])
