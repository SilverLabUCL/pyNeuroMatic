# -*- coding: utf-8 -*-
"""Tests for pyneuromatic.analysis.nm_tool_utilities."""
import math

import numpy as np
import pytest

from pyneuromatic.core.nm_data import NMData
import pyneuromatic.core.nm_math as nm_math
from pyneuromatic.analysis.nm_tool_utilities import (
    _sample_rate_from_xscale,
    find_level_crossings_nmdata,
    filter_bessel_nmdata,
    filter_butterworth_nmdata,
    filter_notch_nmdata,
    linear_regression_nmdata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(yarray, xstart=0.0, xdelta=0.0001, xunits="s"):
    """Return an NMData with the given array and x-scale."""
    d = NMData(name="TestData")
    d.nparray = np.asarray(yarray, dtype=float)
    d.xscale.start = xstart
    d.xscale.delta = xdelta
    d.xscale.units = xunits
    return d


def _sine_data(freq=50.0, n=1000, xdelta=0.0001, xunits="s"):
    """1000-point sine wave, sample rate = 1/xdelta."""
    t = np.arange(n) * xdelta
    y = np.sin(2 * math.pi * freq * t)
    return _make_data(y, xstart=0.0, xdelta=xdelta, xunits=xunits)


# ---------------------------------------------------------------------------
# _sample_rate_from_xscale (Hz)
# ---------------------------------------------------------------------------

class TestSampleRateFromXscale:
    def test_seconds_units(self):
        data = _make_data([0], xdelta=0.0001, xunits="s")
        assert _sample_rate_from_xscale(data) == pytest.approx(10000.0)

    def test_ms_units_gives_correct_sample_rate(self):
        # delta=0.1 ms → interval 0.0001 s → 10000 Hz
        data = _make_data([0], xdelta=0.1, xunits="ms")
        assert _sample_rate_from_xscale(data) == pytest.approx(10000.0)

    def test_no_units_uses_delta_directly(self):
        # no units → factor=1.0 → sample rate = 1/0.0001
        data = _make_data([0], xdelta=0.0001, xunits="")
        assert _sample_rate_from_xscale(data) == pytest.approx(10000.0)

    def test_zero_delta_raises_ValueError(self):
        # NMScaleX rejects delta=0 at assignment, so bypass via internal attr
        data = _make_data([0], xdelta=0.0001, xunits="s")
        data.xscale._delta = 0
        with pytest.raises((ValueError, ZeroDivisionError)):
            _sample_rate_from_xscale(data)


# ---------------------------------------------------------------------------
# find_level_crossings_nmdata
# ---------------------------------------------------------------------------

class TestFindLevelCrossingsNMData:
    def _square_data(self):
        # Square wave: 0..0 then 1..1 crossing at index 5
        y = np.array([0.0] * 5 + [1.0] * 5)
        return _make_data(y, xstart=0.0, xdelta=0.001)

    def test_returns_same_as_nm_math_direct(self):
        data = self._square_data()
        result_wrapper = find_level_crossings_nmdata(data, 0.5)
        result_direct = nm_math.find_level_crossings(
            data.nparray,
            0.5,
            xstart=data.xscale.start,
            xdelta=data.xscale.delta,
        )
        np.testing.assert_array_equal(result_wrapper[0], result_direct[0])
        np.testing.assert_array_almost_equal(result_wrapper[1], result_direct[1])

    def test_returns_tuple_of_two_arrays(self):
        data = self._square_data()
        indexes, xvalues = find_level_crossings_nmdata(data, 0.5)
        assert len(indexes) == len(xvalues)

    def test_x_window_passed_through(self):
        # Two crossings at ~0.005 s and later; restrict to [0, 0.003] should
        # return only the first crossing.
        y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        data = _make_data(y, xstart=0.0, xdelta=0.001)
        all_idx, _ = find_level_crossings_nmdata(data, 0.5)
        win_idx, _ = find_level_crossings_nmdata(data, 0.5, x0=0.0, x1=0.003)
        assert len(win_idx) < len(all_idx)

    def test_backward_search_passed_through(self):
        # x0 > x1 → descending order
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        data = _make_data(y, xstart=0.0, xdelta=0.001)
        fwd_idx, fwd_x = find_level_crossings_nmdata(data, 0.5)
        bwd_idx, bwd_x = find_level_crossings_nmdata(
            data, 0.5, x0=0.008, x1=0.0
        )
        assert len(bwd_x) >= 1
        # x values should be in descending order
        if len(bwd_x) > 1:
            assert bwd_x[0] > bwd_x[-1]

    def test_raises_if_nparray_is_none(self):
        data = _make_data([0.0, 1.0, 0.0])
        data.nparray = None
        with pytest.raises((TypeError, AttributeError, ValueError)):
            find_level_crossings_nmdata(data, 0.5)

    def test_uses_xarray_when_set(self):
        # Non-uniform x spacing — xarray takes precedence over xscale
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        xa = np.array([0.0, 0.5, 1.0, 2.0, 3.0])  # irregular spacing
        data = _make_data(y, xstart=0.0, xdelta=0.001)
        data.xarray = xa
        result_wrapper = find_level_crossings_nmdata(data, 0.5)
        result_direct = nm_math.find_level_crossings(
            y, 0.5, xarray=xa
        )
        np.testing.assert_array_equal(result_wrapper[0], result_direct[0])
        np.testing.assert_array_almost_equal(result_wrapper[1], result_direct[1])

    def test_xarray_window_passed_through(self):
        # Window applied correctly with non-uniform xarray
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        xa = np.array([0.0, 1.0, 2.0, 10.0, 11.0])  # large gap in middle
        data = _make_data(y, xstart=0.0, xdelta=0.001)
        data.xarray = xa
        all_idx, _ = find_level_crossings_nmdata(data, 0.5)
        win_idx, _ = find_level_crossings_nmdata(data, 0.5, x0=0.0, x1=3.0)
        assert len(win_idx) < len(all_idx)


# ---------------------------------------------------------------------------
# linear_regression_nmdata
# ---------------------------------------------------------------------------

class TestLinearRegressionNMData:
    def test_returns_slope_and_intercept(self):
        # y = 2x + 3 at x = 0, 0.001, 0.002, ...
        xdelta = 0.001
        n = 10
        x = np.arange(n) * xdelta
        y = 2.0 * x + 3.0
        data = _make_data(y, xstart=0.0, xdelta=xdelta)
        slope, intercept = linear_regression_nmdata(data)
        assert slope == pytest.approx(2.0, rel=1e-6)
        assert intercept == pytest.approx(3.0, rel=1e-6)

    def test_matches_nm_math_direct(self):
        y = np.linspace(1.0, 5.0, 20)
        data = _make_data(y, xstart=0.0, xdelta=0.01)
        result_wrapper = linear_regression_nmdata(data)
        result_direct = nm_math.linear_regression(
            data.nparray,
            xstart=data.xscale.start,
            xdelta=data.xscale.delta,
        )
        assert result_wrapper[0] == pytest.approx(result_direct[0])
        assert result_wrapper[1] == pytest.approx(result_direct[1])

    def test_ignores_nans(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        data = _make_data(y, xstart=0.0, xdelta=1.0)
        slope, intercept = linear_regression_nmdata(data, ignore_nans=True)
        assert np.isfinite(slope)
        assert np.isfinite(intercept)

    def test_uses_xarray_when_set(self):
        # Non-uniform x spacing — xarray takes precedence over xscale
        xa = np.array([0.0, 1.0, 3.0, 6.0, 10.0])  # irregular spacing
        y = 2.0 * xa + 1.0
        data = _make_data(y, xstart=0.0, xdelta=0.001)
        data.xarray = xa
        result_wrapper = linear_regression_nmdata(data)
        result_direct = nm_math.linear_regression(y, xarray=xa)
        assert result_wrapper[0] == pytest.approx(result_direct[0])
        assert result_wrapper[1] == pytest.approx(result_direct[1])


# ---------------------------------------------------------------------------
# filter_butterworth_nmdata
# ---------------------------------------------------------------------------

class TestFilterButterworthNMData:
    def test_output_shape_matches_input(self):
        data = _sine_data(freq=10.0)
        out = filter_butterworth_nmdata(data, cutoff=100.0)
        assert out.shape == data.nparray.shape

    def test_lowpass_attenuates_high_freq(self):
        # Low-frequency signal should pass; high-frequency should be attenuated
        n, xdelta = 2000, 0.0001  # 10 kHz sample rate
        t = np.arange(n) * xdelta
        low = np.sin(2 * math.pi * 10 * t)    # 10 Hz — well below cutoff
        high = np.sin(2 * math.pi * 1000 * t)  # 1000 Hz — above cutoff
        data_low = _make_data(low, xdelta=xdelta)
        data_high = _make_data(high, xdelta=xdelta)
        out_low = filter_butterworth_nmdata(data_low, cutoff=200.0)
        out_high = filter_butterworth_nmdata(data_high, cutoff=200.0)
        assert np.std(out_low) > np.std(out_high)

    def test_matches_nm_math_direct(self):
        data = _sine_data(freq=10.0)
        sr = 1.0 / (data.xscale.delta * 1.0)  # units="s", factor=1
        expected = nm_math.filter_butterworth(data.nparray, 100.0, sr, 4, "low")
        result = filter_butterworth_nmdata(data, cutoff=100.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_delta_raises(self):
        data = _make_data(np.ones(10), xdelta=0.0001)
        data.xscale._delta = 0
        with pytest.raises((ValueError, ZeroDivisionError)):
            filter_butterworth_nmdata(data, cutoff=100.0)

    def test_raises_if_xarray_set(self):
        data = _sine_data(freq=10.0)
        data.xarray = np.arange(data.nparray.size, dtype=float) * 0.0001
        with pytest.raises(ValueError, match="non-uniform xarray"):
            filter_butterworth_nmdata(data, cutoff=100.0)


# ---------------------------------------------------------------------------
# filter_bessel_nmdata
# ---------------------------------------------------------------------------

class TestFilterBesselNMData:
    def test_output_shape_matches_input(self):
        data = _sine_data(freq=10.0)
        out = filter_bessel_nmdata(data, cutoff=100.0)
        assert out.shape == data.nparray.shape

    def test_matches_nm_math_direct(self):
        data = _sine_data(freq=10.0)
        sr = 1.0 / data.xscale.delta
        expected = nm_math.filter_bessel(data.nparray, 100.0, sr, 4, "low")
        result = filter_bessel_nmdata(data, cutoff=100.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_lowpass_attenuates_high_freq(self):
        n, xdelta = 2000, 0.0001
        t = np.arange(n) * xdelta
        data_low = _make_data(np.sin(2 * math.pi * 10 * t), xdelta=xdelta)
        data_high = _make_data(np.sin(2 * math.pi * 1000 * t), xdelta=xdelta)
        out_low = filter_bessel_nmdata(data_low, cutoff=200.0)
        out_high = filter_bessel_nmdata(data_high, cutoff=200.0)
        assert np.std(out_low) > np.std(out_high)

    def test_raises_if_xarray_set(self):
        data = _sine_data(freq=10.0)
        data.xarray = np.arange(data.nparray.size, dtype=float) * 0.0001
        with pytest.raises(ValueError, match="non-uniform xarray"):
            filter_bessel_nmdata(data, cutoff=100.0)


# ---------------------------------------------------------------------------
# filter_notch_nmdata
# ---------------------------------------------------------------------------

class TestFilterNotchNMData:
    def test_output_shape_matches_input(self):
        data = _sine_data(freq=10.0)
        out = filter_notch_nmdata(data, freq=50.0)
        assert out.shape == data.nparray.shape

    def test_matches_nm_math_direct(self):
        data = _sine_data(freq=10.0)
        sr = 1.0 / data.xscale.delta
        expected = nm_math.filter_notch(data.nparray, 50.0, sr, 30.0)
        result = filter_notch_nmdata(data, freq=50.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_notch_attenuates_target_freq(self):
        # A sine at exactly the notch frequency should be strongly attenuated
        n, xdelta = 4000, 0.0001  # 10 kHz
        t = np.arange(n) * xdelta
        y_notch = np.sin(2 * math.pi * 50 * t)    # exactly 50 Hz
        y_other = np.sin(2 * math.pi * 200 * t)   # away from notch
        data_notch = _make_data(y_notch, xdelta=xdelta)
        data_other = _make_data(y_other, xdelta=xdelta)
        out_notch = filter_notch_nmdata(data_notch, freq=50.0)
        out_other = filter_notch_nmdata(data_other, freq=50.0)
        # The notch-freq signal should be more attenuated than the other
        assert np.std(out_notch) < np.std(out_other)

    def test_raises_if_xarray_set(self):
        data = _sine_data(freq=10.0)
        data.xarray = np.arange(data.nparray.size, dtype=float) * 0.0001
        with pytest.raises(ValueError, match="non-uniform xarray"):
            filter_notch_nmdata(data, freq=50.0)
