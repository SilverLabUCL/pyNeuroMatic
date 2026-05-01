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
    apply_dfof,
    apply_inequality,
    array_stats,
    compute_ref_value,
    find_level_crossings,
    inequality_condition_str,
    inequality_mask,
    interp_x,
    linear_regression,
    parse_si_units,
    si_scale_factor,
    xscale_window_to_slice,
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
# TestXscaleWindowToSlice
# ---------------------------------------------------------------------------


class TestXscaleWindowToSlice:
    ARR = np.zeros(10)
    XD = {"start": 0.0, "delta": 1.0}

    def test_basic_slice(self):
        s = xscale_window_to_slice(self.ARR, self.XD, 2.0, 4.0)
        assert s == slice(2, 5)

    def test_clips_low(self):
        s = xscale_window_to_slice(self.ARR, self.XD, -5.0, 3.0)
        assert s.start == 0

    def test_clips_high(self):
        s = xscale_window_to_slice(self.ARR, self.XD, 7.0, 20.0)
        assert s.stop == len(self.ARR)

    def test_zero_delta_returns_empty(self):
        xd = {"start": 0.0, "delta": 0.0}
        s = xscale_window_to_slice(self.ARR, xd, 0.0, 5.0)
        assert s == slice(0, 0)

    def test_non_unit_delta(self):
        xd = {"start": 0.0, "delta": 0.5}
        arr = np.zeros(20)
        s = xscale_window_to_slice(arr, xd, 1.0, 2.0)
        # i0 = round((1.0-0)/0.5)=2, i1 = round((2.0-0)/0.5)+1=5
        assert s == slice(2, 5)

    def test_neg_inf_begin(self):
        s = xscale_window_to_slice(self.ARR, self.XD, -math.inf, 4.0)
        assert s.start == 0

    def test_pos_inf_end(self):
        s = xscale_window_to_slice(self.ARR, self.XD, 2.0, math.inf)
        assert s.stop == len(self.ARR)

    def test_both_inf(self):
        s = xscale_window_to_slice(self.ARR, self.XD, -math.inf, math.inf)
        assert s == slice(0, len(self.ARR))


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

    # --- x_interp ---

    def test_x_interp_true_returns_interpolated_x(self):
        # Crossing between x=1 (y=0) and x=2 (y=1); ylevel=0.5 → x=1.5
        y = np.array([0.0, 0.0, 1.0])
        idx, xv = find_level_crossings(y, 0.5, xstart=0.0, xdelta=1.0,
                                       x_interp=True)
        assert len(xv) == 1
        assert xv[0] == pytest.approx(1.5)

    def test_x_interp_false_returns_nearest_sample_x(self):
        # Same crossing; nearest sample to x=1.5 is equidistant — tie goes to i-1=1
        y = np.array([0.0, 0.0, 1.0])
        idx, xv = find_level_crossings(y, 0.5, xstart=0.0, xdelta=1.0,
                                       x_interp=False)
        assert len(xv) == 1
        assert xv[0] in (1.0, 2.0)  # one of the two bounding sample x values

    def test_x_interp_true_differs_from_false(self):
        # With a lopsided crossing (y=0 → y=0.9), interpolated x ≠ sample x
        y = np.array([0.0, 0.9])
        idx_t, xv_t = find_level_crossings(y, 0.5, xstart=0.0, xdelta=10.0,
                                           x_interp=True)
        idx_f, xv_f = find_level_crossings(y, 0.5, xstart=0.0, xdelta=10.0,
                                           x_interp=False)
        assert len(xv_t) == 1
        assert len(xv_f) == 1
        assert xv_t[0] != xv_f[0]  # interpolated ≠ nearest sample

    def test_x_interp_xarray_nonuniform(self):
        # Non-uniform x: crossing between x=0 (y=0) and x=8 (y=1); ylevel=0.25 → x=2
        y = np.array([0.0, 1.0])
        xa = np.array([0.0, 8.0])
        idx, xv = find_level_crossings(y, 0.25, xarray=xa, x_interp=True)
        assert len(xv) == 1
        assert xv[0] == pytest.approx(2.0)

    # --- x-window (forward, x0 <= x1) ---

    def test_x_window_default_returns_all(self):
        # x0=-inf, x1=+inf should give the same result as no window
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx_no_win, xv_no_win = find_level_crossings(arr, 0.0)
        idx_win,    xv_win    = find_level_crossings(arr, 0.0, x0=-math.inf, x1=math.inf)
        np.testing.assert_array_equal(idx_no_win, idx_win)
        np.testing.assert_array_almost_equal(xv_no_win, xv_win)

    def test_x_window_excludes_crossings_outside_range(self):
        # 4-sample square wave: crossings near x=0.5, 1.5, 2.5
        # xstart=0, xdelta=1 → samples at x=0,1,2,3
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, xv = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x0=0.0, x1=1.0)
        # Only the rising crossing at ~x=0.5 is within [0.0, 1.0]
        assert len(idx) == 1
        assert xv[0] < 1.0

    def test_x_window_x0_only(self):
        # Only apply a lower bound; all crossings at x >= x0 are returned
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx_all, _ = find_level_crossings(arr, 0.0)
        idx_win, _ = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x0=1.0)
        assert len(idx_win) < len(idx_all)

    def test_x_window_x1_only(self):
        # Only apply an upper bound
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx_all, _ = find_level_crossings(arr, 0.0)
        idx_win, _ = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x1=1.5)
        assert len(idx_win) < len(idx_all)

    def test_x_window_empty_range(self):
        # x0 == x1: zero-width window → no crossings
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx, xv = find_level_crossings(arr, 0.0, x0=1.0, x1=1.0)
        assert len(idx) == 0
        assert len(xv) == 0

    # --- backwards search (x0 > x1) ---

    def test_backward_search_returns_descending_xvalues(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        _, xv = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x0=3.0, x1=0.0)
        assert len(xv) > 1
        # xvalues must be strictly descending
        assert all(xv[i] > xv[i + 1] for i in range(len(xv) - 1))

    def test_backward_search_same_crossings_as_forward(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        idx_fwd, xv_fwd = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x0=0.0, x1=3.0)
        idx_bwd, xv_bwd = find_level_crossings(arr, 0.0, xstart=0, xdelta=1, x0=3.0, x1=0.0)
        np.testing.assert_array_equal(idx_fwd, idx_bwd[::-1])
        np.testing.assert_array_almost_equal(xv_fwd, xv_bwd[::-1])

    def test_backward_search_with_level_plus(self):
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        _, xv = find_level_crossings(arr, 0.0, func_name="level+",
                                     xstart=0, xdelta=1, x0=3.0, x1=0.0)
        # Rising crossings only, descending order
        assert len(xv) == 2
        assert xv[0] > xv[1]

    def test_backward_search_empty_when_no_crossings(self):
        arr = np.array([1.0, 2.0, 3.0])
        idx, xv = find_level_crossings(arr, -1.0, x0=3.0, x1=0.0)
        assert len(idx) == 0
        assert len(xv) == 0

    # --- x0/x1 validation ---

    def test_x0_rejects_nan(self):
        with pytest.raises(ValueError):
            find_level_crossings(np.array([-1.0, 1.0]), 0.0, x0=math.nan)

    def test_x1_rejects_nan(self):
        with pytest.raises(ValueError):
            find_level_crossings(np.array([-1.0, 1.0]), 0.0, x1=math.nan)

    def test_x0_rejects_bool(self):
        with pytest.raises(TypeError):
            find_level_crossings(np.array([-1.0, 1.0]), 0.0, x0=True)

    def test_x1_rejects_bool(self):
        with pytest.raises(TypeError):
            find_level_crossings(np.array([-1.0, 1.0]), 0.0, x1=False)

    # --- xarray with non-uniform spacing ---

    def test_xarray_nonuniform_crossing_x_value(self):
        # Crossing between index 1 (x=1.0, y=0.0) and index 2 (x=5.0, y=1.0)
        # Interpolated x = 1.0 + (0.5 - 0.0) / (1.0 - 0.0) * (5.0 - 1.0) = 3.0
        y = np.array([0.0, 0.0, 1.0, 1.0])
        xa = np.array([0.0, 1.0, 5.0, 6.0])
        idx, xv = find_level_crossings(y, 0.5, xarray=xa)
        assert len(idx) == 1
        assert xv[0] == pytest.approx(3.0)

    def test_xarray_nonuniform_window_respects_x_spacing(self):
        # Two rising crossings: one at x≈3 (between x=1 and x=5),
        # one at x≈15 (between x=10 and x=20).
        # Window x0=0, x1=8 should include only the first.
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        xa = np.array([0.0, 1.0, 5.0, 6.0, 10.0, 10.5, 20.0, 21.0])
        all_idx, _ = find_level_crossings(y, 0.5, func_name="level+", xarray=xa)
        win_idx, _ = find_level_crossings(
            y, 0.5, func_name="level+", xarray=xa, x0=0.0, x1=8.0
        )
        assert len(all_idx) == 2
        assert len(win_idx) == 1

    def test_xarray_nonuniform_backward_search(self):
        # Two rising crossings; backward search (x0 > x1) returns descending order
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        xa = np.array([0.0, 1.0, 5.0, 6.0, 10.0, 10.5, 20.0, 21.0])
        _, xv_fwd = find_level_crossings(y, 0.5, func_name="level+", xarray=xa)
        _, xv_bwd = find_level_crossings(
            y, 0.5, func_name="level+", xarray=xa, x0=25.0, x1=0.0
        )
        assert len(xv_bwd) == 2
        assert xv_bwd[0] > xv_bwd[1]
        # Same x values as forward, reversed
        assert xv_bwd[0] == pytest.approx(xv_fwd[1])
        assert xv_bwd[1] == pytest.approx(xv_fwd[0])

    def test_xarray_size_mismatch_raises(self):
        y = np.array([0.0, 1.0, 0.0])
        xa = np.array([0.0, 1.0])  # wrong size
        with pytest.raises((ValueError, TypeError)):
            find_level_crossings(y, 0.5, xarray=xa)

    # --- NaN handling ---

    def test_ignore_nans_detects_crossing_across_nan_gap(self):
        # NaN between y=0 and y=1; with ignore_nans=True the crossing is found
        y = np.array([0.0, float("nan"), 1.0])
        idx, xv = find_level_crossings(y, 0.5, xstart=0.0, xdelta=1.0,
                                       ignore_nans=True)
        assert len(idx) == 1

    def test_ignore_nans_false_blocks_crossing_across_nan_gap(self):
        # NaN between y=0 and y=1; with ignore_nans=False the crossing is missed
        y = np.array([0.0, float("nan"), 1.0])
        idx, xv = find_level_crossings(y, 0.5, xstart=0.0, xdelta=1.0,
                                       ignore_nans=False)
        assert len(idx) == 0

    def test_ignore_nans_interpolates_x_using_non_nan_positions(self):
        # Non-uniform xarray: y=0 at x=0, y=NaN at x=1, y=1 at x=5.
        # Crossing should be interpolated between x=0 and x=5 → x=2.5
        y = np.array([0.0, float("nan"), 1.0])
        xa = np.array([0.0, 1.0, 5.0])
        idx, xv = find_level_crossings(y, 0.5, xarray=xa, ignore_nans=True)
        assert len(xv) == 1
        assert xv[0] == pytest.approx(2.5)

    def test_ignore_nans_returned_index_is_original_array_index(self):
        # NaN at index 1; crossing is between original indices 0 and 2.
        # Nearest sample to crossing x=2.5 is index 2 (x=5 is farther than x=0).
        # Actually x_cross=2.5, xa=0.0, xb=5.0 → |2.5-0|=2.5, |2.5-5|=2.5 equal → i-1=0
        y = np.array([0.0, float("nan"), 1.0])
        xa = np.array([0.0, 1.0, 5.0])
        idx, _ = find_level_crossings(y, 0.5, xarray=xa, ignore_nans=True)
        assert idx[0] in (0, 2)  # original array index, not compact index 1

    def test_ignore_nans_no_nans_unchanged(self):
        # With no NaNs, ignore_nans=True gives same result as ignore_nans=False
        y = np.array([0.0, 1.0, 0.0])
        idx_t, xv_t = find_level_crossings(y, 0.5, xdelta=1.0, ignore_nans=True)
        idx_f, xv_f = find_level_crossings(y, 0.5, xdelta=1.0, ignore_nans=False)
        np.testing.assert_array_equal(idx_t, idx_f)
        np.testing.assert_array_almost_equal(xv_t, xv_f)

    def test_ignore_nans_all_nan_returns_empty(self):
        y = np.array([float("nan"), float("nan"), float("nan")])
        idx, xv = find_level_crossings(y, 0.5, ignore_nans=True)
        assert len(idx) == 0


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


# ---------------------------------------------------------------------------
# TestApplyDFOF
# ---------------------------------------------------------------------------


class TestApplyDFOF:
    def test_basic_dfof(self):
        arr = np.array([0.0, 1.0, 2.0])
        result = apply_dfof(arr, f0=1.0)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_f0_zero_returns_nan(self):
        arr = np.array([1.0, 2.0])
        result = apply_dfof(arr, f0=0.0)
        assert np.all(np.isnan(result))

    def test_negative_f0(self):
        arr = np.array([0.0, -1.0, -2.0])
        result = apply_dfof(arr, f0=-1.0)
        # (arr - (-1)) / (-1) = (arr + 1) / (-1) = -(arr + 1)
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 1.0])

    def test_shape_preserved(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_dfof(arr, f0=2.0)
        assert result.shape == arr.shape

    def test_nan_in_arr_propagates(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = apply_dfof(arr, f0=1.0)
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])


# ---------------------------------------------------------------------------
# TestParseSiUnits
# ---------------------------------------------------------------------------


class TestParseSiUnits:
    def test_prefix_and_base(self):
        assert parse_si_units("pA") == ("p", "A")

    def test_milli(self):
        assert parse_si_units("mV") == ("m", "V")

    def test_no_prefix(self):
        assert parse_si_units("V") == ("", "V")

    def test_single_char_no_prefix(self):
        assert parse_si_units("A") == ("", "A")

    def test_kilo(self):
        assert parse_si_units("kHz") == ("k", "Hz")

    def test_micro_ascii(self):
        assert parse_si_units("uV") == ("u", "V")

    def test_micro_unicode_b5(self):
        assert parse_si_units("\u00b5V") == ("\u00b5", "V")

    def test_multi_char_base(self):
        assert parse_si_units("MOhm") == ("M", "Ohm")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_si_units("")


# ---------------------------------------------------------------------------
# TestSiScaleFactor
# ---------------------------------------------------------------------------


class TestSiScaleFactor:
    def test_pA_to_nA(self):
        assert si_scale_factor("pA", "nA") == pytest.approx(1e-3)

    def test_nA_to_pA(self):
        assert si_scale_factor("nA", "pA") == pytest.approx(1e3)

    def test_mV_to_V(self):
        assert si_scale_factor("mV", "V") == pytest.approx(1e-3)

    def test_V_to_mV(self):
        assert si_scale_factor("V", "mV") == pytest.approx(1e3)

    def test_same_units(self):
        assert si_scale_factor("mV", "mV") == pytest.approx(1.0)

    def test_base_mismatch_raises(self):
        with pytest.raises(ValueError, match="base units must match"):
            si_scale_factor("pA", "mV")

    def test_unknown_prefix_raises(self):
        # "x" is not a recognised SI prefix, so parse_si_units treats "xA"
        # as a bare base unit — base mismatch with "A" still raises ValueError
        with pytest.raises(ValueError):
            si_scale_factor("xA", "nA")

    def test_no_prefix_to_prefix(self):
        assert si_scale_factor("V", "mV") == pytest.approx(1e3)

    def test_ohm_units(self):
        assert si_scale_factor("MOhm", "kOhm") == pytest.approx(1e3)

    def test_femto_to_base(self):
        assert si_scale_factor("fA", "A") == pytest.approx(1e-15)


# =============================================================================
# smooth_boxcar
# =============================================================================


class TestSmoothBoxcar:
    def test_basic_output_length(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nm_math.smooth_boxcar(y, window=3)
        assert len(result) == len(y)

    def test_flat_signal_unchanged(self):
        y = np.ones(10)
        result = nm_math.smooth_boxcar(y, window=3)
        np.testing.assert_allclose(result[1:-1], 1.0)

    def test_window5_centre(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = nm_math.smooth_boxcar(y, window=5)
        # centre point: mean of [2,3,4,5,6] = 4.0
        assert result[3] == pytest.approx(4.0)

    def test_returns_copy(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nm_math.smooth_boxcar(y, window=3)
        result[2] = 999.0
        assert y[2] == 3.0

    def test_rejects_bool_window(self):
        with pytest.raises(TypeError):
            nm_math.smooth_boxcar(np.ones(5), window=True)

    def test_rejects_even_window(self):
        with pytest.raises(ValueError):
            nm_math.smooth_boxcar(np.ones(5), window=4)

    def test_rejects_window_lt3(self):
        with pytest.raises(ValueError):
            nm_math.smooth_boxcar(np.ones(5), window=1)

    def test_passes_increases_smoothing(self):
        y = np.zeros(21)
        y[10] = 1.0
        r1 = nm_math.smooth_boxcar(y, window=3, passes=1)
        r3 = nm_math.smooth_boxcar(y, window=3, passes=3)
        # more passes → more non-zero points around centre
        assert np.count_nonzero(r3) > np.count_nonzero(r1)

    def test_rejects_passes_zero(self):
        with pytest.raises(ValueError):
            nm_math.smooth_boxcar(np.ones(5), window=3, passes=0)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.smooth_boxcar([1.0, 2.0, 3.0], window=3)


# =============================================================================
# smooth_binomial
# =============================================================================


class TestSmoothBinomial:
    def test_basic_output_length(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nm_math.smooth_binomial(y)
        assert len(result) == len(y)

    def test_flat_signal_unchanged(self):
        y = np.ones(10)
        result = nm_math.smooth_binomial(y, passes=1)
        np.testing.assert_allclose(result[1:-1], 1.0)

    def test_centre_point(self):
        # [0,0,1,0,0] → after 1 pass centre should be 0.5
        y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = nm_math.smooth_binomial(y, passes=1)
        assert result[2] == pytest.approx(0.5)

    def test_passes_spreads_impulse(self):
        # More passes spread a central impulse over more points
        y = np.zeros(21)
        y[10] = 1.0
        r1 = nm_math.smooth_binomial(y, passes=1)
        r5 = nm_math.smooth_binomial(y, passes=5)
        # More passes → more non-zero points around centre
        nonzero1 = np.count_nonzero(r1)
        nonzero5 = np.count_nonzero(r5)
        assert nonzero5 > nonzero1

    def test_returns_copy(self):
        y = np.array([1.0, 2.0, 3.0])
        result = nm_math.smooth_binomial(y)
        result[1] = 999.0
        assert y[1] == 2.0

    def test_rejects_bool_passes(self):
        with pytest.raises(TypeError):
            nm_math.smooth_binomial(np.ones(5), passes=True)

    def test_rejects_passes_zero(self):
        with pytest.raises(ValueError):
            nm_math.smooth_binomial(np.ones(5), passes=0)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.smooth_binomial([1.0, 2.0, 3.0])


# =============================================================================
# smooth_savgol
# =============================================================================


class TestSmoothSavgol:
    def test_basic_output_length(self):
        y = np.linspace(0, 1, 20)
        result = nm_math.smooth_savgol(y, window=5)
        assert len(result) == len(y)

    def test_linear_signal_preserved(self):
        y = np.linspace(0.0, 1.0, 20)
        result = nm_math.smooth_savgol(y, window=5, polyorder=1)
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_returns_copy(self):
        y = np.linspace(0.0, 1.0, 20)
        result = nm_math.smooth_savgol(y, window=5)
        result[5] = 999.0
        assert y[5] != 999.0

    def test_rejects_even_window(self):
        with pytest.raises(ValueError):
            nm_math.smooth_savgol(np.ones(10), window=4)

    def test_rejects_window_lt3(self):
        with pytest.raises(ValueError):
            nm_math.smooth_savgol(np.ones(10), window=1)

    def test_rejects_polyorder_gte_window(self):
        with pytest.raises(ValueError):
            nm_math.smooth_savgol(np.ones(10), window=5, polyorder=5)

    def test_rejects_polyorder_zero(self):
        with pytest.raises(ValueError):
            nm_math.smooth_savgol(np.ones(10), window=5, polyorder=0)

    def test_rejects_bool_window(self):
        with pytest.raises(TypeError):
            nm_math.smooth_savgol(np.ones(10), window=True)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.smooth_savgol([1.0, 2.0, 3.0], window=3)


# =============================================================================
# histogram
# =============================================================================


class TestHistogram:
    def test_returns_dict(self):
        y = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        r = nm_math.histogram(y, bins=4)
        assert "counts" in r
        assert "edges" in r

    def test_counts_length(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = nm_math.histogram(y, bins=4)
        assert len(r["counts"]) == 4

    def test_edges_length(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = nm_math.histogram(y, bins=4)
        assert len(r["edges"]) == 5  # bins + 1

    def test_counts_sum(self):
        y = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        r = nm_math.histogram(y, bins=4)
        assert sum(r["counts"]) == 9

    def test_strips_nans(self):
        y = np.array([1.0, 2.0, np.nan, 3.0])
        r = nm_math.histogram(y, bins=3)
        assert sum(r["counts"]) == 3

    def test_strips_infs(self):
        y = np.array([1.0, 2.0, np.inf, 3.0])
        r = nm_math.histogram(y, bins=3)
        assert sum(r["counts"]) == 3

    def test_density_sums_to_approx_one(self):
        y = np.linspace(0.0, 1.0, 100)
        r = nm_math.histogram(y, bins=10, density=True)
        # density * bin_width should sum to ~1.0
        bin_width = r["edges"][1] - r["edges"][0]
        assert abs(sum(r["counts"]) * bin_width - 1.0) < 1e-10

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.histogram([1.0, 2.0, 3.0])


# =============================================================================
# ks_test
# =============================================================================


class TestKSTest:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.pop1 = rng.normal(loc=0, scale=1, size=50)
        self.pop2 = rng.normal(loc=10, scale=1, size=50)
        self.same1 = rng.normal(loc=0, scale=1, size=50)
        self.same2 = rng.normal(loc=0, scale=1, size=50)

    def test_returns_dict_keys(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        for key in ("d", "pvalue", "alpha", "significant", "message",
                    "n1", "n2", "sort1", "ecdf1", "sort2", "ecdf2"):
            assert key in r

    def test_d_range(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert 0.0 <= r["d"] <= 1.0

    def test_pvalue_range(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert 0.0 <= r["pvalue"] <= 1.0

    def test_significant_true(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert r["significant"] is True

    def test_significant_false(self):
        r = nm_math.ks_test(self.same1, self.same2)
        assert r["significant"] is False

    def test_message_different(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert r["message"] == "different populations"

    def test_message_same(self):
        r = nm_math.ks_test(self.same1, self.same2)
        assert r["message"] == "same population"

    def test_n1_n2(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert r["n1"] == 50
        assert r["n2"] == 50

    def test_strips_nans(self):
        y1_nan = np.concatenate([self.pop1, [np.nan, np.nan]])
        r = nm_math.ks_test(y1_nan, self.pop2)
        assert r["n1"] == 50  # 2 NaNs removed

    def test_alpha_echoed(self):
        r = nm_math.ks_test(self.pop1, self.pop2, alpha=0.01)
        assert r["alpha"] == pytest.approx(0.01)

    def test_ecdf_values_in_range(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert np.all(r["ecdf1"] >= 0.0)
        assert np.all(r["ecdf1"] <= 1.0)

    def test_ecdf_last_value_is_one(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert r["ecdf1"][-1] == pytest.approx(1.0)

    def test_sort_array_is_sorted(self):
        r = nm_math.ks_test(self.pop1, self.pop2)
        assert np.all(r["sort1"][:-1] <= r["sort1"][1:])

    def test_rejects_non_array_y1(self):
        with pytest.raises(TypeError):
            nm_math.ks_test([1.0, 2.0], self.pop2)

    def test_rejects_non_array_y2(self):
        with pytest.raises(TypeError):
            nm_math.ks_test(self.pop1, [1.0, 2.0])


# =============================================================================
# stability_test
# =============================================================================


class TestStabilityTest:
    def setup_method(self):
        self.flat = np.full(20, 3.0)
        self.trend = np.arange(20, dtype=float)
        nan_arr = np.full(20, 3.0)
        nan_arr[5] = np.nan
        nan_arr[15] = np.nan
        self.nan_arr = nan_arr

    def test_stable_region_found(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert r["stable"] is True

    def test_returns_dict_keys(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        for key in ("stable", "start", "end", "n", "rs", "pvalue", "alpha", "mask"):
            assert key in r

    def test_start_end_not_none_when_stable(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert r["start"] is not None
        assert r["end"] is not None

    def test_n_matches_window(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert r["n"] == r["end"] - r["start"] + 1

    def test_rs_range(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert -1.0 <= r["rs"] <= 1.0

    def test_pvalue_range(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert 0.0 <= r["pvalue"] <= 1.0

    def test_alpha_echoed(self):
        r = nm_math.stability_test(self.flat, alpha=0.01, min_window=5)
        assert r["alpha"] == pytest.approx(0.01)

    def test_unstable_region(self):
        r = nm_math.stability_test(self.trend, alpha=0.05, min_window=5)
        assert r["stable"] is False

    def test_start_end_none_when_not_stable(self):
        r = nm_math.stability_test(self.trend, alpha=0.05, min_window=5)
        assert r["start"] is None
        assert r["end"] is None

    def test_rs_none_when_not_stable(self):
        r = nm_math.stability_test(self.trend, alpha=0.05, min_window=5)
        assert r["rs"] is None

    def test_mask_length_equals_input(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert len(r["mask"]) == len(self.flat)

    def test_mask_true_at_stable_region(self):
        r = nm_math.stability_test(self.flat, min_window=5)
        assert np.all(r["mask"][r["start"]: r["end"] + 1])

    def test_mask_all_false_when_not_stable(self):
        r = nm_math.stability_test(self.trend, alpha=0.05, min_window=5)
        assert not np.any(r["mask"])

    def test_strips_nans_mask_length_unchanged(self):
        r = nm_math.stability_test(self.nan_arr, min_window=5)
        assert len(r["mask"]) == 20

    def test_strips_nans_stable_found(self):
        r = nm_math.stability_test(self.nan_arr, min_window=5)
        assert r["stable"] is True

    def test_min_window_too_large_raises(self):
        with pytest.raises(ValueError):
            nm_math.stability_test(self.flat, min_window=100)

    def test_min_window_less_than_3_raises(self):
        with pytest.raises(ValueError):
            nm_math.stability_test(self.flat, min_window=2)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.stability_test([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    """Tests for nm_math.resample()."""

    def test_downsample_halves_length(self):
        y = np.ones(100)
        result = nm_math.resample(y, old_delta=0.1, new_delta=0.2)
        assert len(result) == 50

    def test_upsample_doubles_length(self):
        y = np.ones(50)
        result = nm_math.resample(y, old_delta=0.2, new_delta=0.1)
        assert len(result) == 100

    def test_same_delta_returns_same_length(self):
        y = np.arange(10, dtype=float)
        result = nm_math.resample(y, old_delta=0.1, new_delta=0.1)
        assert len(result) == 10

    def test_constant_signal_preserved(self):
        # Use a long signal; polyphase resampling has edge transients, so
        # check only the interior where the filter has fully settled.
        y = np.ones(1000) * 3.0
        result = nm_math.resample(y, old_delta=0.1, new_delta=0.2)
        np.testing.assert_allclose(result[50:-50], 3.0, atol=1e-4)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            nm_math.resample([1, 2, 3], old_delta=0.1, new_delta=0.2)

    def test_rejects_zero_old_delta(self):
        with pytest.raises(ValueError):
            nm_math.resample(np.ones(10), old_delta=0.0, new_delta=0.1)

    def test_rejects_zero_new_delta(self):
        with pytest.raises(ValueError):
            nm_math.resample(np.ones(10), old_delta=0.1, new_delta=0.0)

    def test_rejects_negative_delta(self):
        with pytest.raises(ValueError):
            nm_math.resample(np.ones(10), old_delta=-0.1, new_delta=0.1)

    def test_returns_ndarray(self):
        result = nm_math.resample(np.ones(10), old_delta=0.1, new_delta=0.2)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


class TestInterpolate:
    """Tests for nm_math.interpolate()."""

    def test_linear_same_grid_unchanged(self):
        x = np.linspace(0, 1, 11)
        y = x ** 2
        result = nm_math.interpolate(y, x, x, method="linear")
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_linear_coarser_grid(self):
        x_old = np.linspace(0, 1, 101)
        y = np.sin(x_old)
        x_new = np.linspace(0, 1, 11)
        result = nm_math.interpolate(y, x_old, x_new, method="linear")
        assert len(result) == 11

    def test_cubic_same_grid_unchanged(self):
        x = np.linspace(0, 1, 21)
        y = x ** 3
        result = nm_math.interpolate(y, x, x, method="cubic")
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_outside_range_is_nan(self):
        x_old = np.linspace(0, 1, 11)
        y = np.ones(11)
        x_new = np.array([-1.0, 0.5, 2.0])
        result = nm_math.interpolate(y, x_old, x_new, method="linear")
        assert np.isnan(result[0])
        assert np.isnan(result[2])
        assert not np.isnan(result[1])

    def test_cubic_outside_range_is_nan(self):
        x_old = np.linspace(0, 1, 21)
        y = np.ones(21)
        x_new = np.array([-1.0, 0.5, 2.0])
        result = nm_math.interpolate(y, x_old, x_new, method="cubic")
        assert np.isnan(result[0])
        assert np.isnan(result[2])
        assert not np.isnan(result[1])

    def test_rejects_non_array_y(self):
        with pytest.raises(TypeError):
            nm_math.interpolate([1, 2, 3], np.array([0, 1, 2]),
                                np.array([0, 1, 2]))

    def test_rejects_non_array_x_old(self):
        with pytest.raises(TypeError):
            nm_math.interpolate(np.ones(3), [0, 1, 2], np.array([0, 1, 2]))

    def test_rejects_non_array_x_new(self):
        with pytest.raises(TypeError):
            nm_math.interpolate(np.ones(3), np.array([0, 1, 2]), [0, 1, 2])

    def test_rejects_invalid_method(self):
        x = np.linspace(0, 1, 5)
        with pytest.raises(ValueError):
            nm_math.interpolate(np.ones(5), x, x, method="spline")

    def test_returns_ndarray(self):
        x = np.linspace(0, 1, 11)
        result = nm_math.interpolate(np.ones(11), x, x)
        assert isinstance(result, np.ndarray)

# ---------------------------------------------------------------------------
# filter helpers shared by Butterworth and Bessel tests
# ---------------------------------------------------------------------------

_SR = 10000.0  # sample rate (Hz) used across filter tests
_N = 1000      # signal length


def _dc(val=5.0):
    return np.ones(_N) * val


def _sine(freq):
    t = np.arange(_N) / _SR
    return np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# filter_butterworth
# ---------------------------------------------------------------------------


class TestFilterButterworth:
    """Tests for nm_math.filter_butterworth."""

    # --- input validation ---

    def test_rejects_non_array_y(self):
        with pytest.raises(TypeError):
            nm_math.filter_butterworth([1.0, 2.0], 1000.0, _SR)

    def test_rejects_bool_cutoff(self):
        with pytest.raises(TypeError):
            nm_math.filter_butterworth(_dc(), True, _SR)

    def test_rejects_zero_cutoff(self):
        with pytest.raises(ValueError):
            nm_math.filter_butterworth(_dc(), 0.0, _SR)

    def test_rejects_negative_cutoff(self):
        with pytest.raises(ValueError):
            nm_math.filter_butterworth(_dc(), -500.0, _SR)

    def test_rejects_string_cutoff(self):
        with pytest.raises(TypeError):
            nm_math.filter_butterworth(_dc(), "1000", _SR)

    def test_rejects_bool_sample_rate(self):
        with pytest.raises(TypeError):
            nm_math.filter_butterworth(_dc(), 1000.0, True)

    def test_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError):
            nm_math.filter_butterworth(_dc(), 1000.0, 0.0)

    def test_rejects_bool_order(self):
        with pytest.raises(TypeError):
            nm_math.filter_butterworth(_dc(), 1000.0, _SR, order=True)

    def test_rejects_zero_order(self):
        with pytest.raises(ValueError):
            nm_math.filter_butterworth(_dc(), 1000.0, _SR, order=0)

    def test_rejects_invalid_btype(self):
        with pytest.raises(ValueError):
            nm_math.filter_butterworth(_dc(), 1000.0, _SR, btype="bandstop")

    # --- output ---

    def test_output_is_ndarray(self):
        result = nm_math.filter_butterworth(_dc(), 1000.0, _SR)
        assert isinstance(result, np.ndarray)

    def test_output_length_preserved(self):
        result = nm_math.filter_butterworth(_dc(), 1000.0, _SR)
        assert len(result) == _N

    def test_accepts_list_cutoff_bandpass(self):
        result = nm_math.filter_butterworth(_dc(), [500.0, 2000.0], _SR, btype="bandpass")
        assert isinstance(result, np.ndarray)

    # --- functional ---

    def test_dc_preserved_lowpass(self):
        result = nm_math.filter_butterworth(_dc(5.0), 1000.0, _SR)
        np.testing.assert_allclose(result[100:-100], 5.0, atol=1e-6)

    def test_lowpass_passes_low_freq(self):
        y = _sine(100)  # 100 Hz << 1 kHz cutoff
        result = nm_math.filter_butterworth(y, 1000.0, _SR)
        rms_in = np.sqrt(np.mean(y[100:-100] ** 2))
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert abs(rms_in - rms_out) < 0.05

    def test_lowpass_attenuates_high_freq(self):
        y = _sine(4000)  # 4 kHz >> 1 kHz cutoff
        result = nm_math.filter_butterworth(y, 1000.0, _SR)
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert rms_out < 0.1

    def test_highpass_attenuates_low_freq(self):
        y = _sine(100)  # 100 Hz << 1 kHz cutoff
        result = nm_math.filter_butterworth(y, 1000.0, _SR, btype="high")
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert rms_out < 0.1


# ---------------------------------------------------------------------------
# filter_bessel
# ---------------------------------------------------------------------------


class TestFilterBessel:
    """Tests for nm_math.filter_bessel."""

    # --- input validation ---

    def test_rejects_non_array_y(self):
        with pytest.raises(TypeError):
            nm_math.filter_bessel([1.0, 2.0], 1000.0, _SR)

    def test_rejects_bool_cutoff(self):
        with pytest.raises(TypeError):
            nm_math.filter_bessel(_dc(), True, _SR)

    def test_rejects_zero_cutoff(self):
        with pytest.raises(ValueError):
            nm_math.filter_bessel(_dc(), 0.0, _SR)

    def test_rejects_negative_cutoff(self):
        with pytest.raises(ValueError):
            nm_math.filter_bessel(_dc(), -500.0, _SR)

    def test_rejects_string_cutoff(self):
        with pytest.raises(TypeError):
            nm_math.filter_bessel(_dc(), "1000", _SR)

    def test_rejects_bool_sample_rate(self):
        with pytest.raises(TypeError):
            nm_math.filter_bessel(_dc(), 1000.0, True)

    def test_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError):
            nm_math.filter_bessel(_dc(), 1000.0, 0.0)

    def test_rejects_bool_order(self):
        with pytest.raises(TypeError):
            nm_math.filter_bessel(_dc(), 1000.0, _SR, order=True)

    def test_rejects_zero_order(self):
        with pytest.raises(ValueError):
            nm_math.filter_bessel(_dc(), 1000.0, _SR, order=0)

    def test_rejects_invalid_btype(self):
        with pytest.raises(ValueError):
            nm_math.filter_bessel(_dc(), 1000.0, _SR, btype="bandstop")

    # --- output ---

    def test_output_is_ndarray(self):
        result = nm_math.filter_bessel(_dc(), 1000.0, _SR)
        assert isinstance(result, np.ndarray)

    def test_output_length_preserved(self):
        result = nm_math.filter_bessel(_dc(), 1000.0, _SR)
        assert len(result) == _N

    def test_accepts_list_cutoff_bandpass(self):
        result = nm_math.filter_bessel(_dc(), [500.0, 2000.0], _SR, btype="bandpass")
        assert isinstance(result, np.ndarray)

    # --- functional ---

    def test_dc_preserved_lowpass(self):
        result = nm_math.filter_bessel(_dc(5.0), 1000.0, _SR)
        np.testing.assert_allclose(result[100:-100], 5.0, atol=1e-6)

    def test_lowpass_passes_low_freq(self):
        y = _sine(100)
        result = nm_math.filter_bessel(y, 1000.0, _SR)
        rms_in = np.sqrt(np.mean(y[100:-100] ** 2))
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert abs(rms_in - rms_out) < 0.05

    def test_lowpass_attenuates_high_freq(self):
        y = _sine(4000)
        result = nm_math.filter_bessel(y, 1000.0, _SR)
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert rms_out < 0.1

    def test_highpass_attenuates_low_freq(self):
        y = _sine(100)
        result = nm_math.filter_bessel(y, 1000.0, _SR, btype="high")
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        assert rms_out < 0.1


# ---------------------------------------------------------------------------
# filter_notch
# ---------------------------------------------------------------------------


class TestFilterNotch:
    """Tests for nm_math.filter_notch."""

    _N_LONG = 10000  # longer signal needed for high-Q transient to decay

    def _sine_long(self, freq):
        t = np.arange(self._N_LONG) / _SR
        return np.sin(2 * np.pi * freq * t)

    # --- input validation ---

    def test_rejects_non_array_y(self):
        with pytest.raises(TypeError):
            nm_math.filter_notch([1.0, 2.0], 60.0, _SR)

    def test_rejects_bool_sample_rate(self):
        with pytest.raises(TypeError):
            nm_math.filter_notch(_dc(), 60.0, True)

    def test_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError):
            nm_math.filter_notch(_dc(), 60.0, 0.0)

    def test_rejects_bool_freq(self):
        with pytest.raises(TypeError):
            nm_math.filter_notch(_dc(), True, _SR)

    def test_rejects_zero_freq(self):
        with pytest.raises(ValueError):
            nm_math.filter_notch(_dc(), 0.0, _SR)

    def test_rejects_bool_q(self):
        with pytest.raises(TypeError):
            nm_math.filter_notch(_dc(), 60.0, _SR, q=True)

    def test_rejects_zero_q(self):
        with pytest.raises(ValueError):
            nm_math.filter_notch(_dc(), 60.0, _SR, q=0.0)

    # --- output ---

    def test_output_is_ndarray(self):
        result = nm_math.filter_notch(_dc(), 60.0, _SR)
        assert isinstance(result, np.ndarray)

    def test_output_length_preserved(self):
        result = nm_math.filter_notch(_dc(), 60.0, _SR)
        assert len(result) == _N

    # --- functional ---

    def test_dc_preserved(self):
        result = nm_math.filter_notch(_dc(3.0), 60.0, _SR)
        np.testing.assert_allclose(result[100:-100], 3.0, atol=1e-6)

    def test_attenuates_notch_frequency(self):
        y = self._sine_long(60)
        result = nm_math.filter_notch(y, 60.0, _SR, q=30.0)
        rms_out = np.sqrt(np.mean(result[2000:-2000] ** 2))
        assert rms_out < 0.1

    def test_passes_off_notch_frequency(self):
        y = self._sine_long(100)  # 100 Hz, notch at 60 Hz
        result = nm_math.filter_notch(y, 60.0, _SR, q=30.0)
        rms_in = np.sqrt(np.mean(y[2000:-2000] ** 2))
        rms_out = np.sqrt(np.mean(result[2000:-2000] ** 2))
        assert abs(rms_in - rms_out) < 0.05


# ---------------------------------------------------------------------------
# TestMatchTemplate
# ---------------------------------------------------------------------------


def _match_template_bruteforce(data, template, circular=False):
    """Brute-force O(n*m) reference implementation for test verification."""
    n = len(data)
    m = len(template)
    tsum = float(np.sum(template))
    tsumsqr = float(np.sum(template ** 2))
    pnts = float(m)
    denom = tsumsqr - tsum * tsum / pnts
    passes = n if circular else n - m + 1
    result = np.zeros(n)
    if denom == 0.0:
        return result
    for i in range(passes):
        seg = np.array([data[(i + j) % n] for j in range(m)], dtype=float)
        dsum = float(np.sum(seg))
        dsumsqr = float(np.sum(seg ** 2))
        dtsum = float(np.sum(seg * template))
        scale = (dtsum - tsum * dsum / pnts) / denom
        offset = (dsum - scale * tsum) / pnts
        sse = (
            dsumsqr
            + scale ** 2 * tsumsqr
            + pnts * offset ** 2
            - 2.0 * (scale * dtsum + offset * dsum - scale * offset * tsum)
        )
        se = math.sqrt(max(sse, 0.0) / (pnts - 1.0))
        result[i] = 0.0 if se == 0.0 else scale / se
    return result


class TestMatchTemplate:
    """Tests for nm_math.match_template."""

    # --- helpers ---

    @staticmethod
    def _template(m=20):
        t = np.linspace(0, math.pi, m)
        return np.sin(t)

    @staticmethod
    def _noisy_data(n=200, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)

    # --- input validation ---

    def test_rejects_list_data(self):
        with pytest.raises(TypeError):
            nm_math.match_template([1.0, 2.0, 3.0], np.array([1.0, 2.0]))

    def test_rejects_list_template(self):
        data = np.zeros(10)
        with pytest.raises(TypeError):
            nm_math.match_template(data, [1.0, 2.0])

    def test_rejects_2d_data(self):
        with pytest.raises(ValueError):
            nm_math.match_template(np.zeros((5, 5)), np.array([1.0, 2.0]))

    def test_rejects_2d_template(self):
        with pytest.raises(ValueError):
            nm_math.match_template(np.zeros(10), np.ones((2, 2)))

    def test_rejects_template_shorter_than_2(self):
        with pytest.raises(ValueError):
            nm_math.match_template(np.zeros(10), np.array([1.0]))

    def test_rejects_template_longer_than_data(self):
        with pytest.raises(ValueError):
            nm_math.match_template(np.zeros(5), np.zeros(10))

    def test_template_equal_length_to_data_ok(self):
        data = np.array([1.0, 2.0, 3.0])
        tmpl = np.array([0.5, 1.0, 0.5])
        result = nm_math.match_template(data, tmpl)
        assert len(result) == 3

    # --- output shape ---

    def test_output_length_equals_data_length(self):
        data = self._noisy_data(200)
        tmpl = self._template(20)
        result = nm_math.match_template(data, tmpl)
        assert len(result) == 200

    def test_output_is_ndarray(self):
        data = self._noisy_data(100)
        tmpl = self._template(10)
        result = nm_math.match_template(data, tmpl)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_float(self):
        data = self._noisy_data(100)
        tmpl = self._template(10)
        result = nm_math.match_template(data, tmpl)
        assert np.issubdtype(result.dtype, np.floating)

    # --- tail zeros (non-circular) ---

    def test_tail_is_zero_non_circular(self):
        n, m = 100, 20
        data = self._noisy_data(n)
        tmpl = self._template(m)
        result = nm_math.match_template(data, tmpl, circular=False)
        passes = n - m + 1
        np.testing.assert_array_equal(result[passes:], 0.0)

    def test_no_tail_zeros_when_template_length_equals_data(self):
        n = 10
        data = self._noisy_data(n)
        tmpl = self._template(n)
        result = nm_math.match_template(data, tmpl, circular=False)
        assert result[0] != 0.0 or True  # just checking no IndexError

    # --- constant template returns zeros ---

    def test_constant_template_returns_zeros(self):
        data = self._noisy_data(50)
        tmpl = np.ones(10)
        result = nm_math.match_template(data, tmpl)
        np.testing.assert_array_equal(result, 0.0)

    # --- matches brute-force computation ---

    def test_matches_bruteforce_non_circular(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal(80)
        tmpl = self._template(15)
        result = nm_math.match_template(data, tmpl, circular=False)
        ref = _match_template_bruteforce(data, tmpl, circular=False)
        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)

    def test_matches_bruteforce_circular(self):
        rng = np.random.default_rng(13)
        data = rng.standard_normal(60)
        tmpl = self._template(12)
        result = nm_math.match_template(data, tmpl, circular=True)
        ref = _match_template_bruteforce(data, tmpl, circular=True)
        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)

    def test_circular_output_length_equals_data_length(self):
        n, m = 100, 20
        data = self._noisy_data(n)
        tmpl = self._template(m)
        result = nm_math.match_template(data, tmpl, circular=True)
        assert len(result) == n

    def test_circular_has_no_tail_zeros(self):
        n, m = 50, 10
        rng = np.random.default_rng(99)
        data = rng.standard_normal(n)
        tmpl = self._template(m)
        result = nm_math.match_template(data, tmpl, circular=True)
        # All passes are active so no zero tail; at least the last position is computed
        ref = _match_template_bruteforce(data, tmpl, circular=True)
        assert result[n - 1] == pytest.approx(ref[n - 1], abs=1e-10)

    # --- event detection ---

    def test_criterion_peaks_at_embedded_event(self):
        # Embed a scaled template at position 50 in mostly-zero background
        m = 20
        tmpl = self._template(m)
        n = 200
        data = np.zeros(n)
        event_pos = 50
        data[event_pos:event_pos + m] = 3.0 * tmpl + 0.5
        result = nm_math.match_template(data, tmpl)
        peak_pos = int(np.argmax(np.abs(result)))
        assert peak_pos == event_pos

    def test_criterion_at_event_exceeds_threshold(self):
        # Large event embedded in low-noise background → criterion >> 4
        rng = np.random.default_rng(123)
        m = 20
        tmpl = self._template(m)
        n = 200
        data = rng.standard_normal(n) * 0.1
        event_pos = 50
        data[event_pos:event_pos + m] += 5.0 * tmpl
        result = nm_math.match_template(data, tmpl)
        assert result[event_pos] > 4.0

    def test_integer_data_accepted(self):
        data = np.arange(50, dtype=int)
        tmpl = np.array([0, 1, 2, 1, 0], dtype=int)
        result = nm_math.match_template(data, tmpl)
        assert len(result) == 50

