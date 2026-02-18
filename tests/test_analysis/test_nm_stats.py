#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_stats: stats functions, NMStatsFunc class hierarchy,
NMStatsWin and NMStatsWinContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import math
import unittest

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_tool_stats as nms
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


def _make_linear(n=5, slope=2.0, intercept=1.0):
    """Create NMData whose y = slope*x + intercept."""
    xdata = np.arange(float(n))
    ydata = slope * xdata + intercept
    data = NMData(NM, name="d", nparray=ydata,
                  xscale={"start": 0, "delta": 1})
    return data, xdata, ydata


# =========================================================================
# badvalue()
# =========================================================================

class TestBadValue(unittest.TestCase):
    """Tests for the badvalue() utility function."""

    def test_none_is_bad(self):
        self.assertTrue(nms.badvalue(None))

    def test_nan_is_bad(self):
        self.assertTrue(nms.badvalue(math.nan))

    def test_inf_is_bad(self):
        self.assertTrue(nms.badvalue(math.inf))
        self.assertTrue(nms.badvalue(-math.inf))

    def test_valid_numbers_are_ok(self):
        self.assertFalse(nms.badvalue(0.0))
        self.assertFalse(nms.badvalue(1.0))
        self.assertFalse(nms.badvalue(-1.0))


# =========================================================================
# NMStatsFunc base class
# =========================================================================

class TestNMStatsFunc(unittest.TestCase):
    """Tests for NMStatsFunc base class."""

    def test_name_property(self):
        t = nms.NMStatsFunc("test")
        self.assertEqual(t.name, "test")

    def test_needs_baseline_false(self):
        t = nms.NMStatsFunc("test")
        self.assertFalse(t.needs_baseline)

    def test_to_dict(self):
        t = nms.NMStatsFunc("test")
        self.assertEqual(t.to_dict(), {"name": "test"})

    def test_getitem(self):
        t = nms.NMStatsFunc("test")
        self.assertEqual(t["name"], "test")
        with self.assertRaises(KeyError):
            t["nonexistent"]

    def test_eq_with_instance(self):
        t1 = nms.NMStatsFuncBasic("mean")
        t2 = nms.NMStatsFuncBasic("mean")
        t3 = nms.NMStatsFuncBasic("median")
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_eq_with_dict(self):
        t = nms.NMStatsFuncBasic("mean")
        self.assertEqual(t, {"name": "mean"})

    def test_eq_with_other_type(self):
        t = nms.NMStatsFuncBasic("mean")
        self.assertEqual(t.__eq__(42), NotImplemented)

    def test_repr_contains_class_and_name(self):
        t = nms.NMStatsFuncBasic("mean")
        self.assertIn("NMStatsFuncBasic", repr(t))
        self.assertIn("mean", repr(t))

    def test_deepcopy_resets_parent(self):
        t = nms.NMStatsFuncBasic("mean", parent=object())
        t2 = copy.deepcopy(t)
        self.assertIsNone(t2._parent)
        self.assertEqual(t, t2)

    def test_validate_baseline_is_noop(self):
        nms.NMStatsFunc("test").validate_baseline(None)  # must not raise

    def test_compute_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            nms.NMStatsFunc("test").compute(None, 0, 1, False, False, None, {})


# =========================================================================
# NMStatsFuncBasic
# =========================================================================

class TestNMStatsFuncBasic(unittest.TestCase):
    """Tests for NMStatsFuncBasic."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncBasic("badfuncname")

    def test_all_valid_names(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms.NMStatsFuncBasic(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_needs_baseline_false(self):
        self.assertFalse(nms.NMStatsFuncBasic("mean").needs_baseline)

    def test_from_dict_round_trip(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms._stats_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatsFuncBasic)
            self.assertEqual(t.name, f)


# =========================================================================
# NMStatsFuncMaxMin
# =========================================================================

class TestNMStatsFuncMaxMin(unittest.TestCase):
    """Tests for NMStatsFuncMaxMin."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncMaxMin("badfuncname")

    def test_mean_at_max_requires_imean(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncMaxMin("mean@max")

    def test_imean_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncMaxMin("mean@max", imean=b)

    def test_imean_value_errors(self):
        for b in [-10, math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncMaxMin("mean@max", imean=b)

    def test_imean_overflow(self):
        with self.assertRaises(OverflowError):
            nms.NMStatsFuncMaxMin("mean@max", imean=math.inf)

    def test_max_min_without_imean(self):
        for f in ("max", "min"):
            t = nms.NMStatsFuncMaxMin(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_imean_upgrades_to_mean_at(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nms.NMStatsFuncMaxMin(f, imean=10)
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t.to_dict()["imean"], 10)

    def test_from_dict(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nms._stats_func_from_dict({"name": f, "imean": 10})
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t["imean"], 10)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "max", "badkey": 0})


# =========================================================================
# NMStatsFuncLevel
# =========================================================================

class TestNMStatsFuncLevel(unittest.TestCase):
    """Tests for NMStatsFuncLevel (explicit ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncLevel("badfuncname", ylevel=10)

    def test_missing_ylevel_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncLevel("level")

    def test_ylevel_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncLevel("level", ylevel=b)

    def test_ylevel_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncLevel("level", ylevel=b)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_LEVEL:
            t = nms.NMStatsFuncLevel(f, ylevel=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["ylevel"], 10)
            self.assertFalse(t.needs_baseline)

    def test_from_dict(self):
        t = nms._stats_func_from_dict({"name": "level+", "ylevel": 10})
        self.assertIsInstance(t, nms.NMStatsFuncLevel)
        self.assertEqual(t.name, "level+")
        self.assertEqual(t["ylevel"], 10)

    def test_from_dict_missing_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "level"})

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "level", "badkey": 0})


# =========================================================================
# NMStatsFuncLevelNstd
# =========================================================================

class TestNMStatsFuncLevelNstd(unittest.TestCase):
    """Tests for NMStatsFuncLevelNstd (nstd-based ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncLevelNstd("badfuncname", nstd=2)

    def test_missing_nstd_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncLevelNstd("level")

    def test_nstd_value_errors(self):
        for b in [math.nan, math.inf, "badvalue", 0]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncLevelNstd("level", nstd=b)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_LEVEL:
            t = nms.NMStatsFuncLevelNstd(f, nstd=2)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["nstd"], 2)
            self.assertTrue(t.needs_baseline)

    def test_negative_nstd_valid(self):
        t = nms.NMStatsFuncLevelNstd("level", nstd=-2)
        self.assertEqual(t.to_dict()["nstd"], -2)

    def test_validate_baseline(self):
        t = nms.NMStatsFuncLevelNstd("level", nstd=2)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        with self.assertRaises(RuntimeError):
            t.validate_baseline("mean")
        t.validate_baseline("mean+std")  # ok

    def test_from_dict(self):
        t = nms._stats_func_from_dict({"name": "level-", "nstd": -2})
        self.assertIsInstance(t, nms.NMStatsFuncLevelNstd)
        self.assertEqual(t.name, "level-")
        self.assertEqual(t["nstd"], -2)

    def test_from_dict_both_keys_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict(
                {"name": "level", "ylevel": 10, "nstd": 2})


# =========================================================================
# NMStatsFuncRiseTime
# =========================================================================

class TestNMStatsFuncRiseTime(unittest.TestCase):
    """Tests for NMStatsFuncRiseTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime("badfuncname", p0=10, p1=90)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime("falltime+", p0=90, p1=10)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime("risetime+")
        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime("risetime+", p1=90)

    def test_missing_p1_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime("risetime+", p0=10)

    def test_p0_p1_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(TypeError):
                nms.NMStatsFuncRiseTime("risetime+", p0=10, p1=b)

    def test_p0_p1_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime("risetime+", p0=10, p1=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime("risetime+", p0=b, p1=90)

    def test_p0_ge_p1_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime("risetime+", p0=90, p1=10)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_RISETIME:
            t = nms.NMStatsFuncRiseTime(f, p0=10, p1=90)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 10)
            self.assertEqual(t.to_dict()["p1"], 90)

    def test_float_p_values(self):
        t = nms.NMStatsFuncRiseTime("risetime+", p0=10.5, p1=89.5)
        self.assertEqual(t.to_dict()["p0"], 10.5)
        self.assertEqual(t.to_dict()["p1"], 89.5)

    def test_needs_baseline(self):
        t = nms.NMStatsFuncRiseTime("risetime+", p0=10, p1=90)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatsFuncRiseTime("risetime+", p0=10, p1=90)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nms._stats_func_from_dict(
            {"name": "risetime+", "p0": 10, "p1": 90})
        self.assertIsInstance(t, nms.NMStatsFuncRiseTime)
        self.assertEqual(t.name, "risetime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict(
                {"name": "risetime+", "p0": 10, "badkey": 90})


# =========================================================================
# NMStatsFuncFallTime
# =========================================================================

class TestNMStatsFuncFallTime(unittest.TestCase):
    """Tests for NMStatsFuncFallTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime("badfuncname", p0=90)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime("risetime+", p0=10, p1=90)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatsFuncFallTime("falltime+")

    def test_p0_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFallTime("falltime+", p0=b)

    def test_p0_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFallTime("falltime+", p0=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFallTime("falltime+", p0=b)

    def test_p0_le_p1_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime("falltime+", p0=10, p1=90)

    def test_p0_without_p1(self):
        t = nms.NMStatsFuncFallTime("falltime+", p0=36)
        self.assertEqual(t.to_dict()["p0"], 36)
        self.assertIsNone(t.to_dict()["p1"])

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_FALLTIME:
            t = nms.NMStatsFuncFallTime(f, p0=90, p1=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 90)
            self.assertEqual(t.to_dict()["p1"], 10)

    def test_needs_baseline(self):
        t = nms.NMStatsFuncFallTime("falltime+", p0=90, p1=10)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatsFuncFallTime("falltime+", p0=90, p1=10)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nms._stats_func_from_dict(
            {"name": "falltime+", "p0": 90, "p1": 10})
        self.assertIsInstance(t, nms.NMStatsFuncFallTime)
        self.assertEqual(t.name, "falltime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict(
                {"name": "falltime+", "p0": 90, "badkey": 10})


# =========================================================================
# NMStatsFuncFWHM
# =========================================================================

class TestNMStatsFuncFWHM(unittest.TestCase):
    """Tests for NMStatsFuncFWHM."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFWHM("badfuncname")

    def test_p_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFWHM("fwhm+", p0=50, p1=b)

    def test_p_value_errors(self):
        for b in [-10, 110, math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFWHM("fwhm+", p0=50, p1=b)

    def test_defaults_to_50_50(self):
        for f in nms.FUNC_NAMES_FWHM:
            t = nms.NMStatsFuncFWHM(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 50)
            self.assertEqual(t.to_dict()["p1"], 50)

    def test_custom_p_values(self):
        t = nms.NMStatsFuncFWHM("fwhm+", p0=45, p1=55)
        self.assertEqual(t.to_dict()["p0"], 45)
        self.assertEqual(t.to_dict()["p1"], 55)

    def test_needs_baseline(self):
        self.assertTrue(nms.NMStatsFuncFWHM("fwhm+").needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatsFuncFWHM("fwhm+")
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict_defaults(self):
        t = nms._stats_func_from_dict({"name": "fwhm+"})
        self.assertEqual(t.name, "fwhm+")
        self.assertEqual(t["p0"], 50)
        self.assertEqual(t["p1"], 50)

    def test_from_dict_custom_values(self):
        t = nms._stats_func_from_dict({"name": "fwhm-", "p0": 45, "p1": 55})
        self.assertEqual(t["p0"], 45)
        self.assertEqual(t["p1"], 55)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "fwhm+", "badkey": 0})


# =========================================================================
# _stats_func_from_dict factory
# =========================================================================

class TestStatsFuncFromDict(unittest.TestCase):
    """Tests for the _stats_func_from_dict() factory function."""

    def test_none_returns_none(self):
        self.assertIsNone(nms._stats_func_from_dict(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(nms._stats_func_from_dict({}))

    def test_none_name_returns_none(self):
        self.assertIsNone(nms._stats_func_from_dict({"name": None}))

    def test_missing_name_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"badkey": "mean"})

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            nms._stats_func_from_dict({"name": "badname"})

    def test_bad_input_type_raises(self):
        with self.assertRaises(TypeError):
            nms._stats_func_from_dict(42)

    def test_bad_name_type_raises(self):
        with self.assertRaises(TypeError):
            nms._stats_func_from_dict({"name": 42})

    def test_string_shorthand(self):
        t = nms._stats_func_from_dict("mean")
        self.assertIsInstance(t, nms.NMStatsFuncBasic)
        self.assertEqual(t.name, "mean")

    def test_basic_func_round_trip(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms._stats_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatsFuncBasic)
            self.assertEqual(t.name, f)

    def test_maxmin_without_imean(self):
        for f in ("max", "min"):
            t = nms._stats_func_from_dict({"name": f})
            self.assertEqual(t.name, f)

    def test_maxmin_with_imean(self):
        for f in ("mean@max", "mean@min"):
            t = nms._stats_func_from_dict({"name": f, "imean": 5})
            self.assertEqual(t.name, f)

    def test_level_ylevel(self):
        t = nms._stats_func_from_dict({"name": "level", "ylevel": 5})
        self.assertIsInstance(t, nms.NMStatsFuncLevel)

    def test_level_nstd(self):
        t = nms._stats_func_from_dict({"name": "level", "nstd": 2})
        self.assertIsInstance(t, nms.NMStatsFuncLevelNstd)

    def test_risetime_round_trip(self):
        for f in nms.FUNC_NAMES_RISETIME:
            t = nms._stats_func_from_dict({"name": f, "p0": 10, "p1": 90})
            self.assertIsInstance(t, nms.NMStatsFuncRiseTime)
            self.assertEqual(t.name, f)

    def test_falltime_round_trip(self):
        for f in nms.FUNC_NAMES_FALLTIME:
            t = nms._stats_func_from_dict({"name": f, "p0": 90, "p1": 10})
            self.assertIsInstance(t, nms.NMStatsFuncFallTime)
            self.assertEqual(t.name, f)

    def test_fwhm_round_trip(self):
        for f in nms.FUNC_NAMES_FWHM:
            t = nms._stats_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatsFuncFWHM)
            self.assertEqual(t.name, f)


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
                nms.find_level_crossings(self.ydata, ylevel=0, func_name=b)

    def test_func_name_value_error(self):
        with self.assertRaises(ValueError):
            nms.find_level_crossings(self.ydata, ylevel=0, func_name="bad")

    def test_ylevel_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(self.ydata, ylevel=b)

    def test_xstart_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(self.ydata, ylevel=0, xstart=b)

    def test_xdelta_type_error(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(self.ydata, ylevel=0, xdelta=b)

    def test_returns_tuple_of_equal_length_arrays(self):
        r = nms.find_level_crossings(self.ydata, ylevel=0)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 2)
        self.assertIsInstance(r[0], np.ndarray)
        self.assertIsInstance(r[1], np.ndarray)
        self.assertEqual(r[0].size, r[1].size)

    def test_crossings_match_manual_count(self):
        ylevel = 0
        r = nms.find_level_crossings(
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
        r = nms.find_level_crossings(
            self.ydata, ylevel=0, func_name="level+", i_nearest=False)
        for i in r[0]:
            self.assertGreater(self.ydata[i], self.ydata[i - 1])

    def test_level_minus_negative_slopes_only(self):
        r = nms.find_level_crossings(
            self.ydata, ylevel=0, func_name="level-", i_nearest=False)
        for i in r[0]:
            self.assertLess(self.ydata[i], self.ydata[i - 1])

    def test_with_xarray(self):
        n = 50
        ydata = np.linspace(-1, 1, n)
        xdata = np.linspace(0, 10, n)
        r = nms.find_level_crossings(
            ydata, ylevel=0, func_name="level+", xarray=xdata)
        self.assertEqual(r[0].size, 1)
        self.assertAlmostEqual(r[1][0], 5.0, places=0)

    def test_no_crossings_returns_empty(self):
        ydata = np.ones(20)  # never crosses 10
        r = nms.find_level_crossings(ydata, ylevel=10)
        self.assertEqual(r[0].size, 0)
        self.assertEqual(r[1].size, 0)


# =========================================================================
# linear_regression()
# =========================================================================

class TestLinearRegression(unittest.TestCase):
    """Tests for linear_regression()."""

    def test_yarray_type_error(self):
        with self.assertRaises(TypeError):
            nms.linear_regression([1, 2, 3])

    def test_xarray_size_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            nms.linear_regression(
                np.array([1.0, 2.0, 3.0]), xarray=np.array([1.0, 2.0]))

    def test_returns_tuple(self):
        result = nms.linear_regression(
            np.array([1.0, 2.0, 3.0]), xstart=0, xdelta=1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_slope_and_intercept_from_xstart_xdelta(self):
        # y = 2x + 1, x = [0, 1, 2, 3, 4]
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        m, b = nms.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(b, 1.0)

    def test_slope_and_intercept_from_xarray(self):
        xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        m, b = nms.linear_regression(ydata, xarray=xdata)
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(b, 1.0)

    def test_flat_line(self):
        ydata = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        m, b = nms.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, 0.0)
        self.assertAlmostEqual(b, 5.0)

    def test_negative_slope(self):
        ydata = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
        m, b = nms.linear_regression(ydata, xstart=0, xdelta=1)
        self.assertAlmostEqual(m, -2.0)
        self.assertAlmostEqual(b, 9.0)

    def test_ignore_nans(self):
        ydata = np.array([1.0, math.nan, 5.0, 7.0, 9.0])
        m, b = nms.linear_regression(ydata, xstart=0, xdelta=1,
                                     ignore_nans=True)
        self.assertFalse(math.isnan(m))
        self.assertFalse(math.isnan(b))


# =========================================================================
# stats()
# =========================================================================

class TestStats(unittest.TestCase):
    """Tests for the stats() function."""

    def setUp(self):
        self.data = _make_data(n=100)
        self.datanan = _make_data(n=100, with_nans=True)

    def test_data_type_error(self):
        for b in nmu.badtypes(ok=[]):
            with self.assertRaises(TypeError):
                nms.stats(b, {"name": "mean"})

    def test_func_type_error(self):
        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                nms.stats(self.data, b)

    def test_func_name_type_error(self):
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nms.stats(self.data, {"name": b})

    def test_x_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[3, 3.14]):
            with self.assertRaises(TypeError):
                nms.stats(self.data, func, x0=b)
            with self.assertRaises(TypeError):
                nms.stats(self.data, func, x1=b)

    def test_results_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[None, {}]):
            with self.assertRaises(TypeError):
                nms.stats(self.data, func, results=b)

    def test_xarray_size_mismatch_raises(self):
        data = NMData(NM, name="d", nparray=np.array([1, 2, 3, 4]),
                      xarray=np.array([1, 2, 3, 4]))
        data.xarray = np.array([1, 2, 3])
        with self.assertRaises(RuntimeError):
            nms.stats(data, {"name": "max"})

    def test_unknown_func_raises(self):
        with self.assertRaises(ValueError):
            nms.stats(self.data, {"name": "unknownfunc"})

    def test_i0_out_of_bounds(self):
        r = nms.stats(self.datanan, {"name": "max"}, x0=-100, xclip=False)
        self.assertIsNone(r["i0"])
        self.assertIn("error", r)

    def test_i1_out_of_bounds(self):
        r = nms.stats(self.datanan, {"name": "max"}, x1=200, xclip=False)
        self.assertIsNone(r["i1"])
        self.assertIn("error", r)

    def test_max_xclip_result_keys(self):
        r = nms.stats(self.datanan, {"name": "max"},
                      x0=-100, x1=200, xclip=True)
        keys = ["data", "i0", "i1", "n", "nans", "infs",
                "s", "sunits", "i", "x", "xunits"]
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(r["i0"], 0)
        self.assertEqual(r["i1"], len(self.datanan.nparray) - 1)
        self.assertEqual(r["sunits"], self.datanan.yscale.units)
        self.assertEqual(r["xunits"], self.datanan.xscale.units)

    def test_value_at_x0(self):
        r = nms.stats(self.datanan, {"name": "value@x0"}, x0=10)
        self.assertEqual(r["i0"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_value_at_x1(self):
        r = nms.stats(self.datanan, {"name": "value@x1"}, x1=10)
        self.assertEqual(r["i1"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_max_single_point(self):
        r = nms.stats(self.datanan, {"name": "max"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_min_single_point(self):
        r = nms.stats(self.datanan, {"name": "min"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_mean(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "mean"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("s", r)
        self.assertIn("sunits", r)

    def test_mean_plus_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "mean+var"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["var"], np.var(ydata))

    def test_mean_plus_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "mean+std"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["std"], np.std(ydata))

    def test_mean_plus_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "mean+sem"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("sem", r)
        self.assertAlmostEqual(r["sem"], np.std(ydata) / math.sqrt(5))

    def test_median(self):
        ydata = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "median"})
        self.assertAlmostEqual(r["s"], 3.0)

    def test_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "var"})
        self.assertAlmostEqual(r["s"], np.var(ydata))
        self.assertIn("**2", r["sunits"])

    def test_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "std"})
        self.assertAlmostEqual(r["s"], np.std(ydata))

    def test_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "sem"})
        self.assertAlmostEqual(r["s"], np.std(ydata) / math.sqrt(5))

    def test_rms(self):
        ydata = np.array([3.0, 4.0])  # rms = 5/sqrt(2) = 3.535...
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "rms"})
        expected = math.sqrt((9 + 16) / 2)
        self.assertAlmostEqual(r["s"], expected)

    def test_sum(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "sum"})
        self.assertAlmostEqual(r["s"], 15.0)

    def test_slope(self):
        # y = 2x + 1
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "slope"})
        self.assertAlmostEqual(r["s"], 2.0)
        self.assertIn("b", r)

    def test_count(self):
        ydata = np.array([1.0, 2.0, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "count"})
        self.assertEqual(r["n"], 5)
        self.assertEqual(r["nans"], 1)
        r = nms.stats(data, {"name": "count"}, ignore_nans=True)
        self.assertEqual(r["n"], 4)

    def test_count_nans(self):
        ydata = np.array([1.0, math.nan, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "count_nans"})
        self.assertEqual(r["nans"], 2)

    def test_count_infs(self):
        ydata = np.array([1.0, math.inf, -math.inf, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "count_infs"})
        self.assertEqual(r["infs"], 2)

    def test_area(self):
        ydata = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stats(data, {"name": "area"})
        self.assertAlmostEqual(r["s"], 10.0)  # sum(y) * delta = 10 * 1

    def test_mean_with_nans_ignored(self):
        r = nms.stats(self.datanan, {"name": "mean"}, ignore_nans=True)
        self.assertFalse(math.isnan(r["s"]))

    def test_mean_with_nans_not_ignored(self):
        r = nms.stats(self.datanan, {"name": "mean"}, ignore_nans=False)
        self.assertTrue(math.isnan(r["s"]))


# =========================================================================
# NMStatsWin
# =========================================================================

class TestNMStatsWin(unittest.TestCase):
    """Tests for NMStatsWin."""

    def setUp(self):
        n = 100
        self.win0 = {
            "on": True,
            "func": {"name": "mean"},
            "x0": 0,
            "x1": n + 10,
            "transform": None,
            "bsln_on": True,
            "bsln_func": {"name": "mean"},
            "bsln_x0": 0,
            "bsln_x1": 10,
        }
        self.win1 = {
            "on": True,
            "func": {"name": "max"},
            "x0": 0,
            "x1": n + 10,
            "transform": None,
            "bsln_on": True,
            "bsln_func": {"name": "mean"},
            "bsln_x0": 0,
            "bsln_x1": 10,
        }
        self.w0 = nms.NMStatsWin(NM, "w0", win=self.win0)
        self.w1 = nms.NMStatsWin(NM, "w1", win=self.win1)
        self.data = _make_data(n=n)
        self.datanan = _make_data(n=n, with_nans=True)

    def test_init_type_errors(self):
        for b in nmu.badtypes(ok=[{}, None]):
            with self.assertRaises(TypeError):
                nms.NMStatsWin(win=b)
        with self.assertRaises(TypeError):
            nms.NMStatsWin(copy=NM)  # unexpected kwarg

    def test_eq_different_funcs(self):
        self.assertFalse(self.w0 == self.w1)

    def test_eq_same_after_copy(self):
        c = self.w0.copy()
        self.assertTrue(self.w0 == c)

    def test_eq_after_win_and_name_set(self):
        self.w1._win_set(self.win0)
        self.assertFalse(self.w0 == self.w1)  # names still differ
        self.w1.name = self.w0.name
        self.assertTrue(self.w0 == self.w1)
        self.w1.x0 = -1
        self.assertFalse(self.w0 == self.w1)

    def test_to_dict_keys(self):
        keys = ["name", "on", "func", "x0", "x1", "transform",
                "bsln_on", "bsln_func", "bsln_x0", "bsln_x1"]
        d = self.w0.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(set(keys), set(d.keys()))
        self.assertEqual(d["name"], "w0")

    def test_win_set_type_error(self):
        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                self.w0._win_set(b)

    def test_win_set_bad_key_raises(self):
        with self.assertRaises(KeyError):
            self.w0._win_set({"badkey": "max"})

    def test_win_set_partial_update(self):
        self.w0._win_set({"on": False, "bsln_on": False})
        self.assertFalse(self.w0.on)
        self.assertFalse(self.w0.bsln_on)

    def test_func_is_dict_with_name(self):
        self.assertIsInstance(self.w1.func, dict)
        self.assertIn("name", self.w1.func)

    def test_func_set_none_clears(self):
        self.w1._func_set(None)
        self.assertEqual(self.w1.func, {})

    def test_func_set_none_name_clears(self):
        self.w1._func_set({"name": None})
        self.assertEqual(self.w1.func, {})

    def test_func_set_type_errors(self):
        for b in nmu.badtypes(ok=[None, {}, "string"]):
            with self.assertRaises(TypeError):
                self.w0._func_set(b)
        for b in nmu.badtypes(ok=[None, "string"]):
            with self.assertRaises(TypeError):
                self.w0._func_set({"name": b})

    def test_func_set_bad_key_raises(self):
        with self.assertRaises(KeyError):
            self.w1.func = {"badkey": "max"}

    def test_func_set_bad_name_raises(self):
        with self.assertRaises(ValueError):
            self.w1.func = {"name": "badname"}

    def test_func_set_maxmin_imean(self):
        self.w1._func_set({"name": "mean@max", "imean": 7})
        self.w1._func_set({"imean": "3"})
        self.assertEqual(self.w1.func["name"], "mean@max")
        self.assertEqual(self.w1.func["imean"], 3)
        self.w1._func_set({"imean": True})
        self.assertEqual(self.w1.func["imean"], 1)

    def test_func_set_level(self):
        self.w1._func_set({"name": "level+", "ylevel": -10})
        self.assertEqual(self.w1.func["name"], "level+")
        self.assertEqual(self.w1.func["ylevel"], -10)
        self.w1._func_set({"ylevel": "10"})
        self.assertEqual(self.w1.func["ylevel"], 10)

    def test_func_set_risetime(self):
        self.w1._func_set({"name": "risetime+", "p0": 10, "p1": 90})
        self.assertEqual(self.w1.func["name"], "risetime+")
        self.assertEqual(self.w1.func["p0"], 10)
        self.assertEqual(self.w1.func["p1"], 90)
        self.w1._func_set({"p0": "20", "p1": "80"})
        self.assertEqual(self.w1.func["p0"], 20)
        self.assertEqual(self.w1.func["p1"], 80)
        with self.assertRaises(ValueError):
            self.w1._func_set({"p0": 75, "p1": 25})

    def test_func_set_falltime(self):
        self.w1._func_set({"name": "falltime+", "p0": 90, "p1": 10})
        self.assertEqual(self.w1.func["name"], "falltime+")
        self.assertEqual(self.w1.func["p0"], 90)
        self.assertEqual(self.w1.func["p1"], 10)
        self.w1._func_set({"p0": 36})
        self.assertEqual(self.w1.func["p0"], 36)
        self.assertIsNone(self.w1.func["p1"])
        with self.assertRaises(ValueError):
            self.w1._func_set({"p0": 25, "p1": 75})

    def test_func_set_fwhm(self):
        for f in ["fwhm+", "fwhm-"]:
            self.w1._func_set({"name": f})
            self.assertEqual(self.w1.func["name"], f)
            self.assertEqual(self.w1.func["p0"], 50)
            self.assertEqual(self.w1.func["p1"], 50)
            self.w1._func_set({"p0": "40", "p1": "40"})
            self.assertEqual(self.w1.func["p0"], 40)
            self.assertEqual(self.w1.func["p1"], 40)

    def test_func_names_complete(self):
        fnames = [
            "max", "min", "mean@max", "mean@min",
            "median", "mean", "mean+var", "mean+std", "mean+sem",
            "var", "std", "sem", "rms", "sum", "pathlength", "area", "slope",
            "level", "level+", "level-", "value@x0", "value@x1",
            "count", "count_nans", "count_infs",
            "risetime+", "falltime+", "risetimeslope+", "falltimeslope+",
            "fwhm+", "risetime-", "falltime-", "risetimeslope-",
            "falltimeslope-", "fwhm-",
        ]
        self.assertEqual(len(nms.FUNC_NAMES), len(fnames))
        for f in fnames:
            self.assertIn(f, nms.FUNC_NAMES)

    def test_bsln_func_names_complete(self):
        fnames = ("median", "mean", "mean+var", "mean+std", "mean+sem")
        self.assertEqual(len(nms.FUNC_NAMES_BSLN), len(fnames))
        for f in fnames:
            self.assertIn(f, nms.FUNC_NAMES_BSLN)

    def test_x_set_type_errors(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                self.w0._x_set("x0", b)

    def test_x_set_nan_raises(self):
        with self.assertRaises(ValueError):
            self.w0._x_set("x0", math.nan)

    def test_x_set_values(self):
        self.w0._x_set("x0", "0")
        self.assertEqual(self.w0.x0, 0)
        self.w0._x_set("x0", -99)
        self.assertEqual(self.w0.x0, -99)
        self.w0._x_set("x1", 99)
        self.assertEqual(self.w0.x1, 99)
        self.w0._x_set("bsln_x0", -99)
        self.assertEqual(self.w0.bsln_x0, -99)
        self.w0._x_set("bsln_x1", 99)
        self.assertEqual(self.w0.bsln_x1, 99)

    def test_x_set_inf_clamps_by_axis(self):
        self.w0._x_set("x0", "inf")
        self.assertTrue(math.isinf(self.w0.x0))
        self.assertLess(self.w0.x0, 0)   # x0 → -inf
        self.w0._x_set("x1", "inf")
        self.assertTrue(math.isinf(self.w0.x1))
        self.assertGreater(self.w0.x1, 0)  # x1 → +inf

    def test_transform_set_and_roundtrip(self):
        from pyneuromatic.core.nm_transform import (
            NMTransformInvert, NMTransformLog, NMTransformLn,
        )
        self.assertIsNone(self.w0.transform)

        self.w0.transform = [NMTransformInvert(), NMTransformLog()]
        self.assertEqual(len(self.w0.transform), 2)
        self.assertIsInstance(self.w0.transform[0], NMTransformInvert)
        self.assertIsInstance(self.w0.transform[1], NMTransformLog)

        # to_dict serialises to list of dicts
        win = self.w0.to_dict()
        self.assertIsInstance(win["transform"], list)
        self.assertEqual(win["transform"][0], {"type": "NMTransformInvert"})

        # round-trip via NMStatsWin constructor
        w2 = nms.NMStatsWin(win=win)
        self.assertEqual(len(w2.transform), 2)
        self.assertIsInstance(w2.transform[0], NMTransformInvert)

        # set via dicts
        self.w0.transform = [{"type": "NMTransformInvert"},
                              {"type": "NMTransformLn"}]
        self.assertIsInstance(self.w0.transform[1], NMTransformLn)

        self.w0.transform = None
        self.assertIsNone(self.w0.transform)
        self.assertIsNone(self.w0.to_dict()["transform"])

        for b in nmu.badtypes(ok=[None, []]):
            with self.assertRaises(TypeError):
                self.w0.transform = b

    def test_bsln_func_set(self):
        self.assertIsInstance(self.w1.bsln_func, dict)
        self.assertIn("name", self.w1.bsln_func)
        for b in nmu.badtypes(ok=[None, {}, "string"]):
            with self.assertRaises(TypeError):
                self.w0._bsln_func_set(b)
        with self.assertRaises(KeyError):
            self.w1.bsln_func = {"badkey": "max"}
        with self.assertRaises(ValueError):
            self.w1.bsln_func = {"name": "badname"}
        with self.assertRaises(ValueError):
            self.w1._bsln_func_set({"name": "mean@max", "imean": 7})

    def test_results_list(self):
        self.assertIsInstance(self.w1.results, list)
        self.w1.results.append("test0")
        self.assertIn("test0", self.w1.results)
        self.w1.results.clear()
        self.assertEqual(len(self.w1.results), 0)

    def test_compute_type_error(self):
        for b in nmu.badtypes(ok=[None]):
            with self.assertRaises(TypeError):
                self.w0.compute(b)

    def test_compute_max_result_keys(self):
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0]["id"], "bsln")
        self.assertEqual(r[1]["id"], "main")
        bsln_keys = ["win", "id", "func", "x0", "x1", "data", "i0", "i1",
                     "n", "nans", "infs", "s", "sunits"]
        for k in bsln_keys:
            self.assertIn(k, r[0])
        main_keys = bsln_keys + ["i", "x", "xunits", "Δs"]
        for k in main_keys:
            self.assertIn(k, r[1])

    def test_compute_delta_s(self):
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        if r[1]["Δs"]:
            self.assertAlmostEqual(r[1]["Δs"], r[1]["s"] - r[0]["s"])

    def test_compute_x1_out_of_bounds(self):
        r = self.w1.compute(self.datanan, xclip=False, ignore_nans=True)
        self.assertIsNone(r[1]["i1"])
        self.assertIn("error", r[1])

    def test_compute_nans_not_ignored(self):
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=False)
        self.assertTrue(np.isnan(r[0]["s"]))
        self.assertTrue(np.isnan(r[1]["s"]))
        self.assertTrue(np.isnan(r[1]["Δs"]))

    def test_compute_mean_at_max_warning(self):
        self.w1.func = {"name": "mean@max", "imean": 0}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertIn("warning", r[1]["func"])

    def test_compute_mean_at_max_with_imean(self):
        self.w1.func = {"name": "mean@max", "imean": 5}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(r[1]["func"]["name"], "mean@max")
        self.assertEqual(r[1]["func"]["imean"], 5)

    def test_compute_mean_at_min(self):
        self.w1.func = {"name": "mean@min", "imean": 5}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(r[1]["func"]["name"], "mean@min")

    def test_compute_level_ylevel(self):
        ylevel = 10
        self.w1.func = {"name": "level+", "ylevel": ylevel}
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertEqual(len(r), 2)
        self.assertIn("ylevel", r[1]["func"])
        self.assertEqual(r[1]["func"]["ylevel"], ylevel)

    def test_compute_level_nstd_requires_mean_std_baseline(self):
        self.w1.func = {"name": "level+", "nstd": 2}
        self.w1.bsln_on = True
        with self.assertRaises(RuntimeError):
            self.w1.bsln_func = "mean"  # need mean+std
            self.w1.compute(self.data, xclip=True, ignore_nans=True)

    def test_compute_level_nstd(self):
        nstd = 2
        self.w1.func = {"name": "level+", "nstd": nstd}
        self.w1.bsln_on = True
        self.w1.bsln_func = "mean+std"
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertIn("ylevel", r[1]["func"])
        self.assertAlmostEqual(
            round(r[1]["Δs"] * 1000),
            round(nstd * r[0]["std"] * 1000))

    def test_compute_level_nstd_negative(self):
        nstd = -2
        self.w1.func = {"name": "level-", "nstd": nstd}
        self.w1.bsln_func = "mean+std"
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertAlmostEqual(
            round(r[1]["Δs"] * 1000),
            round(nstd * r[0]["std"] * 1000))

    def test_compute_falltime_p0_only(self):
        self.w1.bsln_x0 = 0
        self.w1.bsln_x1 = 10
        self.w1.x0 = -math.inf
        self.w1.x1 = math.inf
        self.w1.func = {"name": "falltime+", "p0": 36}
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertEqual(r[0]["id"], "bsln")
        self.assertEqual(r[1]["id"], "falltime+")
        self.assertIn("Δs", r[1])
        if not math.isnan(r[1]["Δs"]):
            self.assertEqual(len(r), 3)
            self.assertAlmostEqual(r[1]["Δs"], r[1]["s"] - r[0]["s"])
            self.assertIn("Δx", r[2])

    def test_compute_peak_funcs(self):
        flist = [
            "risetime+", "risetime-", "risetimeslope+", "risetimeslope-",
            "falltime+", "falltime-", "falltimeslope+", "falltimeslope-",
            "fwhm+", "fwhm-",
        ]
        for f in flist:
            rise = "rise" in f
            fall = "fall" in f
            fwhm = "fwhm" in f
            slope = "slope" in f
            p0, p1 = (10, 90) if rise else (90, 10) if fall else (50, 50)

            self.w1.func = {"name": f, "p0": p0, "p1": p1}

            with self.assertRaises(RuntimeError):
                self.w1.bsln_on = False
                self.w1.compute(self.data, xclip=True)

            self.w1.bsln_on = True
            r = self.w1.compute(self.datanan, xclip=True, ignore_nans=False)
            self.assertEqual(len(r), 2)
            self.assertTrue(math.isnan(r[1]["Δs"]))
            self.assertEqual(r[0]["id"], "bsln")
            peak = "max" if "+" in f else "min"
            self.assertEqual(r[1]["func"]["name"], peak)

            r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
            if math.isnan(r[1]["Δs"]):
                continue
            self.assertAlmostEqual(r[1]["Δs"], r[1]["s"] - r[0]["s"])
            self.assertGreaterEqual(len(r), 4)
            self.assertEqual(r[2]["p0"], p0)
            self.assertEqual(r[3]["p1"], p1)
            self.assertIn("Δx", r[3])
            if not math.isnan(r[3]["Δx"]):
                self.assertAlmostEqual(r[3]["Δx"], r[3]["x"] - r[2]["x"])
            expected_len = (5 if slope and not math.isnan(r[3]["Δx"])
                            else 4)
            self.assertEqual(len(r), expected_len)
            for i in range(1, expected_len):
                self.assertEqual(r[i]["id"], f)
            if slope and expected_len == 5:
                if "error" not in r[4]:
                    self.assertIn("s", r[4])
                    self.assertIn("b", r[4])


# =========================================================================
# NMStatsWinContainer
# =========================================================================

class TestNMStatsWinContainer(unittest.TestCase):
    """Tests for NMStatsWinContainer."""

    def test_new_creates_window(self):
        c = nms.NMStatsWinContainer()
        w = c.new()
        self.assertIsInstance(w, nms.NMStatsWin)
        self.assertEqual(len(c), 1)
        self.assertEqual(c.selected_name, "w0")

    def test_sequential_names(self):
        c = nms.NMStatsWinContainer()
        w0 = c.new()
        w1 = c.new()
        self.assertEqual(w0.name, "w0")
        self.assertEqual(w1.name, "w1")

    def test_len(self):
        c = nms.NMStatsWinContainer()
        self.assertEqual(len(c), 0)
        c.new()
        c.new()
        self.assertEqual(len(c), 2)

    def test_getitem(self):
        c = nms.NMStatsWinContainer()
        w = c.new()
        self.assertIs(c["w0"], w)

    def test_contains(self):
        c = nms.NMStatsWinContainer()
        c.new()
        self.assertIn("w0", c)
        self.assertNotIn("w99", c)

    def test_iter(self):
        c = nms.NMStatsWinContainer()
        c.new()
        c.new()
        wins = list(c)
        self.assertEqual(len(wins), 2)
        self.assertIsInstance(wins[0], nms.NMStatsWin)

    def test_custom_prefix(self):
        c = nms.NMStatsWinContainer(name_prefix="win")
        w = c.new()
        self.assertEqual(w.name, "win0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
