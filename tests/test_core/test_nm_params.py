"""Tests for parameter descriptions (#555)."""

import math

import pytest

from pyneuromatic.core.nm_params import (
    BOOLEAN, CHOICE, INTEGER, NUMBER, X_RANGE, XRANGE, NMParam, keywords)


def test_check_converts_and_names_the_parameter():
    rate = NMParam("rate", NUMBER, "Rate", minimum=0.0, maximum=10.0)
    assert rate.check("2.5") == 2.5
    with pytest.raises(ValueError, match="Rate: 'fast' is not a number"):
        rate.check("fast")
    with pytest.raises(ValueError, match="Rate: at most 10"):
        rate.check(11)
    count = NMParam("n", INTEGER, "Points")
    assert count.check("3") == 3
    with pytest.raises(ValueError, match="not a whole number"):
        count.check("3.5")
    mode = NMParam("mode", CHOICE, "Mode", choices=(("1d", "End to end"), ("2d", "Rows")))
    assert mode.check("2d") == "2d"
    with pytest.raises(ValueError):
        mode.check("3d")
    assert NMParam("on", BOOLEAN, "On").check(1) is True


def test_x_range_gives_two_keywords():
    assert X_RANGE.kind == XRANGE
    assert X_RANGE.check(("-inf", "60")) == (-math.inf, 60.0)
    with pytest.raises(ValueError, match="begin must be less than end"):
        X_RANGE.check((5, 1))
    assert keywords((X_RANGE,), {"x": (0, 1)}) == {"xbgn": 0.0, "xend": 1.0}


def test_bad_descriptions():
    with pytest.raises(ValueError):
        NMParam("x", "colour", "X")
    with pytest.raises(ValueError):
        NMParam("x", XRANGE, "X")  # no names
    with pytest.raises(ValueError):
        NMParam("m", CHOICE, "M")  # no choices


def test_nan_only_where_allowed_and_blank_said():
    """Replace Values' value may be NaN (#559); a blank number says to enter one."""
    value = NMParam("old", NUMBER, "Replace", nan=True)
    assert math.isnan(value.check("nan")) and value.check("-inf") == -math.inf
    with pytest.raises(ValueError, match="Rate: NaN is not a number"):
        NMParam("rate", NUMBER, "Rate").check("nan")
    with pytest.raises(ValueError, match="Points: enter a number"):
        NMParam("n", INTEGER, "Points").check(" ")
    with pytest.raises(ValueError, match="NaN is not a number"):  # nor a whole one
        NMParam("n", INTEGER, "Points", nan=True).check("nan")
