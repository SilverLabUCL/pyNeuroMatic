"""Tests for pyneuromatic.core.nm_math."""
from __future__ import annotations

import math

import numpy as np
import pytest

import pyneuromatic.core.nm_math as nm_math
from pyneuromatic.core.nm_math import (
    VALID_INEQUALITY_OPS,
    apply_inequality,
    inequality_condition_str,
    inequality_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARR = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


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
