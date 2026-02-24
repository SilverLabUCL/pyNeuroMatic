#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_stat_utilities: find_level_crossings, linear_regression, stat, stats.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import unittest

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_stat_utilities as nsmm
import pyneuromatic.core.nm_utilities as nmu

NM = NMManager(quiet=True)


def _make_data(n=100, name="recordA0", with_nans=False, units=True):
    """Create a test NMData with optional NaNs."""
    ydata = np.random.normal(loc=0, scale=1, size=n)
    if with_nans:
        ydata[3] = math.nan
        ydata[66] = math.nan
    kwargs = {"name": name, "nparray": ydata,
              "xscale": {"label": "Time", "units": "ms", "start": 0,
                         "delta": 1}}
    if units:
        kwargs["yscale"] = {"label": "Membrane Current", "units": "pA"}
    return NMData(NM, **kwargs)


# =========================================================================
# find_level_crossings()
# =========================================================================

class TestFindLevelCrossings(unittest.TestCase):
    """Tests for find_level_crossings()."""

    def setUp(self):
        self.data = _make_data(n=100)
        self.ydata = self.data.nparray

    def test_func_name_type_error(self):
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nsmm.find_level_crossings(self.ydata, ylevel=0, func_name=b)

    def test_func_name_value_error(self):
        with self.assertRaises(ValueError):
            nsmm.find_level_crossings(self.ydata, ylevel=0, func_name="bad")

    def test_ylevel_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nsmm.find_level_crossings(self.ydata, ylevel=b)

    def test_xstart_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nsmm.find_level_crossings(self.ydata, ylevel=0, xstart=b)

    def test_xdelta_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nsmm.find_level_crossings(self.ydata, ylevel=0, xdelta=b)

    def test_returns_tuple_of_equal_length_arrays(self):
        r = nsmm.find_level_crossings(self.ydata, ylevel=0)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)
        self.assertIsInstance(r[0], np.ndarray)
        self.assertIsInstance(r[1], np.ndarray)
        self.assertEqual(r[0].size, r[1].size)

    def test_crossings_match_manual_count(self):
        ylevel = 0
        r = nsmm.find_level_crossings(
            self.ydata, ylevel=ylevel,
            func_name="level", i_nearest=False)
        # Build expected indexes manually
        expected = []
        yon = self.ydata[0] >= ylevel
        for i in range(1, len(self.ydata)):
            above = self.ydata[i] >= ylevel
            if above != yon:
                yon = above
                expected.append(i)
        self.assertEqual(list(r[0]), expected)

    def test_level_plus_positive_slopes_only(self):
        r = nsmm.find_level_crossings(
            self.ydata, ylevel=0, func_name="level+", i_nearest=False)
        for i in r[0]:
            self.assertGreater(self.ydata[i], self.ydata[i - 1])

    def test_level_minus_negative_slopes_only(self):
        r = nsmm.find_level_crossings(
            self.ydata, ylevel=0, func_name="level-", i_nearest=False)
        for i in r[0]:
            self.assertLess(self.ydata[i], self.ydata[i - 1])

    def test_with_xarray(self):
        n = 50
        ydata = np.linspace(-1, 1, n)
        xdata = np.linspace(0, 10, n)
        r = nsmm.find_level_crossings(
            ydata, ylevel=0, func_name="level+", xarray=xdata)
        self.assertEqual(r[0].size, 1)
        self.assertAlmostEqual(r[1][0], 5.0, places=0)

    def test_no_crossings_returns_empty(self):
        ydata = np.ones(20)  # never crosses 10
        r = nsmm.find_level_crossings(ydata, ylevel=10)
        self.assertEqual(r[0].size, 0)
        self.assertEqual(r[1].size, 0)

    def test_xarray_size_mismatch_raises(self):
        ydata = np.zeros(10)
        xdata = np.zeros(5)  # wrong size
        with self.assertRaises(ValueError):
            nsmm.find_level_crossings(ydata, ylevel=0, xarray=xdata)


# =========================================================================
# linear_regression()
# =========================================================================

class TestLinearRegression(unittest.TestCase):
    """Tests for linear_regression()."""

    def test_yarray_type_error(self):
        with self.assertRaises(TypeError):
            nsmm.linear_regression([1, 2, 3])

    def test_xarray_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            nsmm.linear_regression(
                np.array([1.0, 2.0, 3.0]), xarray=np.array([1.0, 2.0]))

    def test_returns_tuple(self):
        result = nsmm.linear_regression(
            np.array([1.0, 2.0, 3.0]), xstart=0, xdelta=1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_slope_and_intercept_from_xstart_xdelta(self):
        # y = 2x + 1, x = [0, 1, 2, 3, 4]
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        m, b = nsmm.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(b, 1.0)

    def test_slope_and_intercept_from_xarray(self):
        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        m, b = nsmm.linear_regression(ydata, xarray=xdata)
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(b, 1.0)

    def test_flat_line(self):
        ydata = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        m, b = nsmm.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, 0.0)
        self.assertAlmostEqual(b, 5.0)

    def test_negative_slope(self):
        ydata = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
        m, b = nsmm.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, -2.0)
        self.assertAlmostEqual(b, 9.0)

    def test_ignore_nans(self):
        ydata = np.array([1.0, math.nan, 5.0, 7.0, 9.0])
        m, b = nsmm.linear_regression(ydata, xstart=0, xdelta=1,
                                      ignore_nans=True)
        self.assertFalse(math.isnan(m))
        self.assertFalse(math.isnan(b))
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(b, 1.0)


# =========================================================================
# stat()
# =========================================================================

class TestStat(unittest.TestCase):
    """Tests for the stat() function."""

    def setUp(self):
        self.data = _make_data(n=100)
        self.datanan = _make_data(n=100, with_nans=True)

    def test_data_type_error(self):
        for b in nmu.badtypes():
            with self.assertRaises(TypeError):
                nsmm.stat(b, {"name": "mean"})

    def test_func_type_error(self):
        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                nsmm.stat(self.data, b)

    def test_func_name_type_error(self):
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nsmm.stat(self.data, {"name": b})

    def test_x_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[3, 3.14]):
            with self.assertRaises(TypeError):
                nsmm.stat(self.data, func, x0=b)
            with self.assertRaises(TypeError):
                nsmm.stat(self.data, func, x1=b)

    def test_results_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[None, {}]):
            with self.assertRaises(TypeError):
                nsmm.stat(self.data, func, results=b)

    def test_xarray_size_mismatch_raises(self):
        data = NMData(NM, name="d", nparray=np.array([1, 2, 3, 4]),
                      xarray=np.array([1, 2, 3, 4]))
        data.xarray = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nsmm.stat(data, {"name": "max"})

    def test_unknown_func_raises(self):
        with self.assertRaises(ValueError):
            nsmm.stat(self.data, {"name": "unknownfunc"})

    def test_i0_out_of_bounds(self):
        r = nsmm.stat(self.datanan, {"name": "max"}, x0=-100, xclip=False)
        self.assertIsNone(r["i0"])
        self.assertIn("error", r)

    def test_i1_out_of_bounds(self):
        r = nsmm.stat(self.datanan, {"name": "max"}, x1=200, xclip=False)
        self.assertIsNone(r["i1"])
        self.assertIn("error", r)

    def test_max_xclip_result_keys(self):
        r = nsmm.stat(self.datanan, {"name": "max"},
                      x0=-100, x1=200, xclip=True)
        keys = ["data", "i0", "i1", "n", "nans", "infs",
                "s", "sunits", "i", "x", "xunits"]
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(r["i0"], 0)
        self.assertEqual(r["i1"], len(self.datanan.nparray) - 1)
        self.assertEqual(r["sunits"], self.datanan.yscale.units)
        self.assertEqual(r["xunits"], self.datanan.xscale.units)

    def test_value_at_x0(self):
        r = nsmm.stat(self.datanan, {"name": "value@x0"}, x0=10)
        self.assertEqual(r["i0"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_value_at_x1(self):
        r = nsmm.stat(self.datanan, {"name": "value@x1"}, x1=10)
        self.assertEqual(r["i1"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_max_single_point(self):
        r = nsmm.stat(self.datanan, {"name": "max"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_min_single_point(self):
        r = nsmm.stat(self.datanan, {"name": "min"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_mean(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "mean"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("s", r)
        self.assertIn("sunits", r)

    def test_mean_plus_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "mean+var"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["var"], np.var(ydata))

    def test_mean_plus_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "mean+std"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["std"], np.std(ydata))

    def test_mean_plus_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "mean+sem"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("sem", r)
        self.assertAlmostEqual(r["sem"], np.std(ydata) / math.sqrt(5))

    def test_median(self):
        ydata = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "median"})
        self.assertAlmostEqual(r["s"], 3.0)

    def test_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "var"})
        self.assertAlmostEqual(r["s"], np.var(ydata))
        self.assertIn("**2", r["sunits"])

    def test_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "std"})
        self.assertAlmostEqual(r["s"], np.std(ydata))

    def test_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "sem"})
        self.assertAlmostEqual(r["s"], np.std(ydata) / math.sqrt(5))

    def test_rms(self):
        ydata = np.array([3.0, 4.0])  # rms = 5/sqrt(2) = 3.535...
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "rms"})
        expected = math.sqrt((9 + 16) / 2)
        self.assertAlmostEqual(r["s"], expected)

    def test_sum(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "sum"})
        self.assertAlmostEqual(r["s"], 15.0)

    def test_slope(self):
        # y = 2x + 1
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "slope"})
        self.assertAlmostEqual(r["s"], 2.0)
        self.assertIn("b", r)

    def test_count(self):
        ydata = np.array([1.0, 2.0, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "count"})
        self.assertEqual(r["n"], 5)
        self.assertEqual(r["nans"], 1)
        r = nsmm.stat(data, {"name": "count"}, ignore_nans=True)
        self.assertEqual(r["n"], 4)

    def test_count_nans(self):
        ydata = np.array([1.0, math.nan, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "count_nans"})
        self.assertEqual(r["nans"], 2)

    def test_count_infs(self):
        ydata = np.array([1.0, math.inf, -math.inf, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "count_infs"})
        self.assertEqual(r["infs"], 2)

    def test_area(self):
        ydata = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nsmm.stat(data, {"name": "area"})
        self.assertAlmostEqual(r["s"], 10.0)  # sum(y) * delta = 10 * 1

    def test_pathlength(self):
        # flat line: dy=0 at each step, so pathlength = (n-1) * dx
        ydata = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 2},
                      yscale={"units": "mV"})
        data.xscale.units = "mV"  # must match yunits for pathlength
        r = nsmm.stat(data, {"name": "pathlength"})
        self.assertAlmostEqual(r["s"], 4 * 2.0)  # 4 steps * dx=2

    def test_pathlength_diagonal(self):
        # y increases by 1 each step, dx=1: each segment = sqrt(1²+1²) = sqrt(2)
        ydata = np.array([0.0, 1.0, 2.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1},
                      yscale={"units": "mV"})
        data.xscale.units = "mV"
        r = nsmm.stat(data, {"name": "pathlength"})
        self.assertAlmostEqual(r["s"], 3 * math.sqrt(2))

    def test_pathlength_unit_mismatch_raises(self):
        ydata = np.array([1.0, 2.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1},
                      yscale={"units": "mV"})
        data.xscale.units = "ms"  # different string units → ValueError
        with self.assertRaises(ValueError):
            nsmm.stat(data, {"name": "pathlength"})

    def test_mean_with_nans_ignored(self):
        r = nsmm.stat(self.datanan, {"name": "mean"}, ignore_nans=True)
        self.assertFalse(math.isnan(r["s"]))

    def test_mean_with_nans_not_ignored(self):
        r = nsmm.stat(self.datanan, {"name": "mean"}, ignore_nans=False)
        self.assertTrue(math.isnan(r["s"]))


# =========================================================================
# stats()
# =========================================================================

class TestStats(unittest.TestCase):
    """Tests for the stats() summary statistics function."""

    def setUp(self):
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.arr_nan = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])

    # --- type validation ---

    def test_rejects_non_array(self):
        with self.assertRaises(TypeError):
            nsmm.stats([1, 2, 3])

    def test_rejects_non_dict_results(self):
        with self.assertRaises(TypeError):
            nsmm.stats(self.arr, results="bad")

    # --- result keys ---

    def test_returns_dict(self):
        r = nsmm.stats(self.arr)
        self.assertIsInstance(r, dict)

    def test_all_keys_present(self):
        r = nsmm.stats(self.arr)
        for key in ("N", "NaNs", "INFs", "mean", "std", "sem", "rms", "min", "max"):
            self.assertIn(key, r)

    # --- correct values ---

    def test_N(self):
        r = nsmm.stats(self.arr)
        self.assertEqual(r["N"], 5)

    def test_NaNs(self):
        r = nsmm.stats(self.arr_nan)
        self.assertEqual(r["NaNs"], 1)

    def test_INFs(self):
        arr = np.array([1.0, float("inf"), 3.0])
        r = nsmm.stats(arr)
        self.assertEqual(r["INFs"], 1)

    def test_mean(self):
        r = nsmm.stats(self.arr)
        self.assertAlmostEqual(r["mean"], 3.0)

    def test_std(self):
        r = nsmm.stats(self.arr)
        self.assertAlmostEqual(r["std"], float(np.std(self.arr, ddof=1)))

    def test_sem(self):
        r = nsmm.stats(self.arr)
        self.assertAlmostEqual(r["sem"], r["std"] / math.sqrt(5))

    def test_min(self):
        r = nsmm.stats(self.arr)
        self.assertAlmostEqual(r["min"], 1.0)

    def test_max(self):
        r = nsmm.stats(self.arr)
        self.assertAlmostEqual(r["max"], 5.0)

    def test_rms(self):
        r = nsmm.stats(self.arr)
        expected = math.sqrt(np.mean(np.square(self.arr)))
        self.assertAlmostEqual(r["rms"], expected)

    # --- ignore_nans ---

    def test_ignore_nans_N(self):
        r = nsmm.stats(self.arr_nan, ignore_nans=True)
        self.assertEqual(r["N"], 4)

    def test_ignore_nans_mean(self):
        r = nsmm.stats(self.arr_nan, ignore_nans=True)
        self.assertAlmostEqual(r["mean"], (1 + 3 + 4 + 5) / 4)

    def test_ignore_nans_false_propagates_nan(self):
        r = nsmm.stats(self.arr_nan, ignore_nans=False)
        self.assertTrue(math.isnan(r["mean"]))

    # --- empty array ---

    def test_empty_array_returns_nans(self):
        r = nsmm.stats(np.array([]), ignore_nans=True)
        self.assertEqual(r["N"], 0)
        self.assertTrue(math.isnan(r["mean"]))

    # --- populates provided dict ---

    def test_populates_provided_dict(self):
        r = {"existing": 99}
        nsmm.stats(self.arr, results=r)
        self.assertIn("mean", r)
        self.assertEqual(r["existing"], 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
