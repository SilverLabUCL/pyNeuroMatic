#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_stats: stat functions, NMStatFunc class hierarchy,
NMStatWin and NMStatWinContainer.

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
# NMStatFunc base class
# =========================================================================

class TestNMStatFunc(unittest.TestCase):
    """Tests for NMStatFunc base class."""

    def test_name_property(self):
        t = nms.NMStatFunc("test")
        self.assertEqual(t.name, "test")

    def test_needs_baseline_false(self):
        t = nms.NMStatFunc("test")
        self.assertFalse(t.needs_baseline)

    def test_to_dict(self):
        t = nms.NMStatFunc("test")
        self.assertEqual(t.to_dict(), {"name": "test"})

    def test_getitem(self):
        t = nms.NMStatFunc("test")
        self.assertEqual(t["name"], "test")
        with self.assertRaises(KeyError):
            t["nonexistent"]

    def test_eq_with_instance(self):
        t1 = nms.NMStatFuncBasic("mean")
        t2 = nms.NMStatFuncBasic("mean")
        t3 = nms.NMStatFuncBasic("median")
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_eq_with_dict(self):
        t = nms.NMStatFuncBasic("mean")
        self.assertEqual(t, {"name": "mean"})

    def test_eq_with_other_type(self):
        t = nms.NMStatFuncBasic("mean")
        self.assertEqual(t.__eq__(42), NotImplemented)

    def test_repr_contains_class_and_name(self):
        t = nms.NMStatFuncBasic("mean")
        self.assertIn("NMStatFuncBasic", repr(t))
        self.assertIn("mean", repr(t))

    def test_deepcopy_resets_parent(self):
        t = nms.NMStatFuncBasic("mean", parent=object())
        t2 = copy.deepcopy(t)
        self.assertIsNone(t2._parent)
        self.assertEqual(t, t2)

    def test_validate_baseline_is_noop(self):
        t = nms.NMStatFunc("test")
        t.validate_baseline(None)  # must not raise

    def test_compute_raises_not_implemented(self):
        t = nms.NMStatFunc("test")
        with self.assertRaises(NotImplementedError):
            t.compute(None, 0, 1, False, False, None, {})


# =========================================================================
# NMStatFuncBasic
# =========================================================================

class TestNMStatFuncBasic(unittest.TestCase):
    """Tests for NMStatFuncBasic."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncBasic("badfuncname")

    def test_all_valid_names(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms.NMStatFuncBasic(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_needs_baseline_false(self):
        self.assertFalse(nms.NMStatFuncBasic("mean").needs_baseline)

    def test_from_dict_round_trip(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatFuncBasic)
            self.assertEqual(t.name, f)


# =========================================================================
# NMStatFuncMaxMin
# =========================================================================

class TestNMStatFuncMaxMin(unittest.TestCase):
    """Tests for NMStatFuncMaxMin."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncMaxMin("badfuncname")

    def test_mean_at_max_requires_n_avg(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncMaxMin("mean@max")

    def test_n_avg_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncMaxMin("mean@max", n_avg=b)

    def test_n_avg_value_errors(self):
        for b in [-10, math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncMaxMin("mean@max", n_avg=b)

    def test_n_avg_overflow(self):
        with self.assertRaises(OverflowError):
            nms.NMStatFuncMaxMin("mean@max", n_avg=math.inf)

    def test_max_min_without_n_avg(self):
        for f in ("max", "min"):
            t = nms.NMStatFuncMaxMin(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_n_avg_upgrades_to_mean_at(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nms.NMStatFuncMaxMin(f, n_avg=10)
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t.to_dict()["n_avg"], 10)

    def test_from_dict(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nms._stat_func_from_dict({"name": f, "n_avg": 10})
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t["n_avg"], 10)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict({"name": "max", "badkey": 0})


# =========================================================================
# NMStatFuncLevel
# =========================================================================

class TestNMStatFuncLevel(unittest.TestCase):
    """Tests for NMStatFuncLevel (explicit ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncLevel("badfuncname", ylevel=10)

    def test_missing_ylevel_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncLevel("level")

    def test_ylevel_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncLevel("level", ylevel=b)

    def test_ylevel_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncLevel("level", ylevel=b)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_LEVEL:
            t = nms.NMStatFuncLevel(f, ylevel=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["ylevel"], 10)
            self.assertFalse(t.needs_baseline)

    def test_from_dict(self):
        t = nms._stat_func_from_dict({"name": "level+", "ylevel": 10})
        self.assertIsInstance(t, nms.NMStatFuncLevel)
        self.assertEqual(t.name, "level+")
        self.assertEqual(t["ylevel"], 10)

    def test_from_dict_missing_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict({"name": "level"})

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict({"name": "level", "badkey": 0})


# =========================================================================
# NMStatFuncLevelNstd
# =========================================================================

class TestNMStatFuncLevelNstd(unittest.TestCase):
    """Tests for NMStatFuncLevelNstd (n_std-based ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncLevelNstd("badfuncname", n_std=2)

    def test_missing_n_std_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncLevelNstd("level")

    def test_n_std_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncLevelNstd("level", n_std=b)

    def test_n_std_value_errors(self):
        for b in [math.nan, math.inf, "badvalue", 0]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncLevelNstd("level", n_std=b)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_LEVEL:
            t = nms.NMStatFuncLevelNstd(f, n_std=2)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["n_std"], 2)
            self.assertTrue(t.needs_baseline)

    def test_negative_n_std_valid(self):
        t = nms.NMStatFuncLevelNstd("level", n_std=-2)
        self.assertEqual(t.to_dict()["n_std"], -2)

    def test_validate_baseline(self):
        t = nms.NMStatFuncLevelNstd("level", n_std=2)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        with self.assertRaises(RuntimeError):
            t.validate_baseline("mean")
        t.validate_baseline("mean+std")  # ok

    def test_from_dict(self):
        t = nms._stat_func_from_dict({"name": "level-", "n_std": -2})
        self.assertIsInstance(t, nms.NMStatFuncLevelNstd)
        self.assertEqual(t.name, "level-")
        self.assertEqual(t["n_std"], -2)

    def test_from_dict_both_keys_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict(
                {"name": "level", "ylevel": 10, "n_std": 2})


# =========================================================================
# NMStatFuncRiseTime
# =========================================================================

class TestNMStatFuncRiseTime(unittest.TestCase):
    """Tests for NMStatFuncRiseTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncRiseTime("badfuncname", p0=10, p1=90)
        with self.assertRaises(ValueError):
            nms.NMStatFuncRiseTime("falltime+", p0=10, p1=90)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncRiseTime("risetime+")
        with self.assertRaises(KeyError):
            nms.NMStatFuncRiseTime("risetime+", p1=90)

    def test_missing_p1_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncRiseTime("risetime+", p0=10)

    def test_p0_p1_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(TypeError):
                nms.NMStatFuncRiseTime("risetime+", p0=10, p1=b)

    def test_p0_p1_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(ValueError):
                nms.NMStatFuncRiseTime("risetime+", p0=10, p1=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncRiseTime("risetime+", p0=b, p1=90)

    def test_p0_ge_p1_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncRiseTime("risetime+", p0=90, p1=10)

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_RISETIME:
            t = nms.NMStatFuncRiseTime(f, p0=10, p1=90)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 10)
            self.assertEqual(t.to_dict()["p1"], 90)

    def test_float_p_values(self):
        t = nms.NMStatFuncRiseTime("risetime+", p0=10.5, p1=89.5)
        self.assertEqual(t.to_dict()["p0"], 10.5)
        self.assertEqual(t.to_dict()["p1"], 89.5)

    def test_needs_baseline(self):
        t = nms.NMStatFuncRiseTime("risetime+", p0=10, p1=90)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatFuncRiseTime("risetime+", p0=10, p1=90)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nms._stat_func_from_dict(
            {"name": "risetime+", "p0": 10, "p1": 90})
        self.assertIsInstance(t, nms.NMStatFuncRiseTime)
        self.assertEqual(t.name, "risetime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict(
                {"name": "risetime+", "p0": 10, "badkey": 90})


# =========================================================================
# NMStatFuncFallTime
# =========================================================================

class TestNMStatFuncFallTime(unittest.TestCase):
    """Tests for NMStatFuncFallTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncFallTime("badfuncname", p0=90)
        with self.assertRaises(ValueError):
            nms.NMStatFuncFallTime("risetime+", p0=10, p1=90)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nms.NMStatFuncFallTime("falltime+")

    def test_p0_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncFallTime("falltime+", p0=b)

    def test_p0_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncFallTime("falltime+", p0=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncFallTime("falltime+", p0=b)

    def test_p1_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncFallTime("falltime+", p0=90, p1=b)

    def test_p0_le_p1_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncFallTime("falltime+", p0=10, p1=90)

    def test_p0_without_p1(self):
        t = nms.NMStatFuncFallTime("falltime+", p0=36)
        self.assertEqual(t.to_dict()["p0"], 36)
        self.assertIsNone(t.to_dict()["p1"])

    def test_valid_construction(self):
        for f in nms.FUNC_NAMES_FALLTIME:
            t = nms.NMStatFuncFallTime(f, p0=90, p1=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 90)
            self.assertEqual(t.to_dict()["p1"], 10)

    def test_needs_baseline(self):
        t = nms.NMStatFuncFallTime("falltime+", p0=90, p1=10)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatFuncFallTime("falltime+", p0=90, p1=10)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nms._stat_func_from_dict(
            {"name": "falltime+", "p0": 90, "p1": 10})
        self.assertIsInstance(t, nms.NMStatFuncFallTime)
        self.assertEqual(t.name, "falltime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict(
                {"name": "falltime+", "p0": 90, "badkey": 10})


# =========================================================================
# NMStatFuncFWHM
# =========================================================================

class TestNMStatFuncFWHM(unittest.TestCase):
    """Tests for NMStatFuncFWHM."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nms.NMStatFuncFWHM("badfuncname")

    def test_p_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(TypeError):
                nms.NMStatFuncFWHM("fwhm+", p0=50, p1=b)

    def test_p_value_errors(self):
        for b in [-10, 110, math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(ValueError):
                nms.NMStatFuncFWHM("fwhm+", p0=50, p1=b)

    def test_defaults_to_50_50(self):
        for f in nms.FUNC_NAMES_FWHM:
            t = nms.NMStatFuncFWHM(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 50)
            self.assertEqual(t.to_dict()["p1"], 50)

    def test_custom_p_values(self):
        t = nms.NMStatFuncFWHM("fwhm+", p0=45, p1=55)
        self.assertEqual(t.to_dict()["p0"], 45)
        self.assertEqual(t.to_dict()["p1"], 55)

    def test_needs_baseline(self):
        self.assertTrue(nms.NMStatFuncFWHM("fwhm+").needs_baseline)

    def test_validate_baseline(self):
        t = nms.NMStatFuncFWHM("fwhm+")
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict_defaults(self):
        t = nms._stat_func_from_dict({"name": "fwhm+"})
        self.assertEqual(t.name, "fwhm+")
        self.assertEqual(t["p0"], 50)
        self.assertEqual(t["p1"], 50)

    def test_from_dict_custom_values(self):
        t = nms._stat_func_from_dict({"name": "fwhm-", "p0": 45, "p1": 55})
        self.assertEqual(t["p0"], 45)
        self.assertEqual(t["p1"], 55)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict({"name": "fwhm+", "badkey": 0})


# =========================================================================
# _stat_func_from_dict factory
# =========================================================================

class TestStatFuncFromDict(unittest.TestCase):
    """Tests for the _stat_func_from_dict() factory function."""

    def test_none_returns_none(self):
        self.assertIsNone(nms._stat_func_from_dict(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(nms._stat_func_from_dict({}))

    def test_none_name_returns_none(self):
        self.assertIsNone(nms._stat_func_from_dict({"name": None}))

    def test_missing_name_key_raises(self):
        with self.assertRaises(KeyError):
            nms._stat_func_from_dict({"badkey": "mean"})

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            nms._stat_func_from_dict({"name": "badname"})

    def test_bad_input_type_raises(self):
        with self.assertRaises(TypeError):
            nms._stat_func_from_dict(42)

    def test_bad_name_type_raises(self):
        with self.assertRaises(TypeError):
            nms._stat_func_from_dict({"name": 42})

    def test_string_shorthand(self):
        t = nms._stat_func_from_dict("mean")
        self.assertIsInstance(t, nms.NMStatFuncBasic)
        self.assertEqual(t.name, "mean")

    def test_basic_func_round_trip(self):
        for f in nms.FUNC_NAMES_BASIC:
            t = nms._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatFuncBasic)
            self.assertEqual(t.name, f)

    def test_maxmin_without_n_avg(self):
        for f in ("max", "min"):
            t = nms._stat_func_from_dict({"name": f})
            self.assertEqual(t.name, f)

    def test_maxmin_with_n_avg(self):
        for f in ("mean@max", "mean@min"):
            t = nms._stat_func_from_dict({"name": f, "n_avg": 5})
            self.assertEqual(t.name, f)

    def test_level_ylevel(self):
        t = nms._stat_func_from_dict({"name": "level", "ylevel": 5})
        self.assertIsInstance(t, nms.NMStatFuncLevel)

    def test_level_nstd(self):
        t = nms._stat_func_from_dict({"name": "level", "n_std": 2})
        self.assertIsInstance(t, nms.NMStatFuncLevelNstd)

    def test_risetime_round_trip(self):
        for f in nms.FUNC_NAMES_RISETIME:
            t = nms._stat_func_from_dict({"name": f, "p0": 10, "p1": 90})
            self.assertIsInstance(t, nms.NMStatFuncRiseTime)
            self.assertEqual(t.name, f)

    def test_falltime_round_trip(self):
        for f in nms.FUNC_NAMES_FALLTIME:
            t = nms._stat_func_from_dict({"name": f, "p0": 90, "p1": 10})
            self.assertIsInstance(t, nms.NMStatFuncFallTime)
            self.assertEqual(t.name, f)

    def test_fwhm_round_trip(self):
        for f in nms.FUNC_NAMES_FWHM:
            t = nms._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nms.NMStatFuncFWHM)
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

    def test_xarray_size_mismatch_raises(self):
        ydata = np.zeros(10)
        xdata = np.zeros(5)  # wrong size
        with self.assertRaises(ValueError):
            nms.find_level_crossings(ydata, ylevel=0, xarray=xdata)


# =========================================================================
# linear_regression()
# =========================================================================

class TestLinearRegression(unittest.TestCase):
    """Tests for linear_regression()."""

    def test_yarray_type_error(self):
        with self.assertRaises(TypeError):
            nms.linear_regression([1, 2, 3])

    def test_xarray_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
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
                nms.stat(b, {"name": "mean"})

    def test_func_type_error(self):
        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                nms.stat(self.data, b)

    def test_func_name_type_error(self):
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nms.stat(self.data, {"name": b})

    def test_x_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[3, 3.14]):
            with self.assertRaises(TypeError):
                nms.stat(self.data, func, x0=b)
            with self.assertRaises(TypeError):
                nms.stat(self.data, func, x1=b)

    def test_results_type_error(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[None, {}]):
            with self.assertRaises(TypeError):
                nms.stat(self.data, func, results=b)

    def test_xarray_size_mismatch_raises(self):
        data = NMData(NM, name="d", nparray=np.array([1, 2, 3, 4]),
                      xarray=np.array([1, 2, 3, 4]))
        data.xarray = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nms.stat(data, {"name": "max"})

    def test_unknown_func_raises(self):
        with self.assertRaises(ValueError):
            nms.stat(self.data, {"name": "unknownfunc"})

    def test_i0_out_of_bounds(self):
        r = nms.stat(self.datanan, {"name": "max"}, x0=-100, xclip=False)
        self.assertIsNone(r["i0"])
        self.assertIn("error", r)

    def test_i1_out_of_bounds(self):
        r = nms.stat(self.datanan, {"name": "max"}, x1=200, xclip=False)
        self.assertIsNone(r["i1"])
        self.assertIn("error", r)

    def test_max_xclip_result_keys(self):
        r = nms.stat(self.datanan, {"name": "max"},
                      x0=-100, x1=200, xclip=True)
        keys = ["data", "i0", "i1", "n", "nans", "infs",
                "s", "sunits", "i", "x", "xunits"]
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(r["i0"], 0)
        self.assertEqual(r["i1"], len(self.datanan.nparray) - 1)
        self.assertEqual(r["sunits"], self.datanan.yscale.units)
        self.assertEqual(r["xunits"], self.datanan.xscale.units)

    def test_value_at_x0(self):
        r = nms.stat(self.datanan, {"name": "value@x0"}, x0=10)
        self.assertEqual(r["i0"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_value_at_x1(self):
        r = nms.stat(self.datanan, {"name": "value@x1"}, x1=10)
        self.assertEqual(r["i1"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_max_single_point(self):
        r = nms.stat(self.datanan, {"name": "max"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_min_single_point(self):
        r = nms.stat(self.datanan, {"name": "min"}, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

    def test_mean(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "mean"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("s", r)
        self.assertIn("sunits", r)

    def test_mean_plus_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "mean+var"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["var"], np.var(ydata))

    def test_mean_plus_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "mean+std"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertAlmostEqual(r["std"], np.std(ydata))

    def test_mean_plus_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "mean+sem"})
        self.assertAlmostEqual(r["s"], 3.0)
        self.assertIn("sem", r)
        self.assertAlmostEqual(r["sem"], np.std(ydata) / math.sqrt(5))

    def test_median(self):
        ydata = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "median"})
        self.assertAlmostEqual(r["s"], 3.0)

    def test_var(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "var"})
        self.assertAlmostEqual(r["s"], np.var(ydata))
        self.assertIn("**2", r["sunits"])

    def test_std(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "std"})
        self.assertAlmostEqual(r["s"], np.std(ydata))

    def test_sem(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "sem"})
        self.assertAlmostEqual(r["s"], np.std(ydata) / math.sqrt(5))

    def test_rms(self):
        ydata = np.array([3.0, 4.0])  # rms = 5/sqrt(2) = 3.535...
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "rms"})
        expected = math.sqrt((9 + 16) / 2)
        self.assertAlmostEqual(r["s"], expected)

    def test_sum(self):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "sum"})
        self.assertAlmostEqual(r["s"], 15.0)

    def test_slope(self):
        # y = 2x + 1
        ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "slope"})
        self.assertAlmostEqual(r["s"], 2.0)
        self.assertIn("b", r)

    def test_count(self):
        ydata = np.array([1.0, 2.0, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "count"})
        self.assertEqual(r["n"], 5)
        self.assertEqual(r["nans"], 1)
        r = nms.stat(data, {"name": "count"}, ignore_nans=True)
        self.assertEqual(r["n"], 4)

    def test_count_nans(self):
        ydata = np.array([1.0, math.nan, math.nan, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "count_nans"})
        self.assertEqual(r["nans"], 2)

    def test_count_infs(self):
        ydata = np.array([1.0, math.inf, -math.inf, 4.0, 5.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "count_infs"})
        self.assertEqual(r["infs"], 2)

    def test_area(self):
        ydata = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1})
        r = nms.stat(data, {"name": "area"})
        self.assertAlmostEqual(r["s"], 10.0)  # sum(y) * delta = 10 * 1

    def test_pathlength(self):
        # flat line: dy=0 at each step, so pathlength = (n-1) * dx
        ydata = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 2},
                      yscale={"units": "mV"})
        data.xscale.units = "mV"  # must match yunits for pathlength
        r = nms.stat(data, {"name": "pathlength"})
        self.assertAlmostEqual(r["s"], 4 * 2.0)  # 4 steps * dx=2

    def test_pathlength_diagonal(self):
        # y increases by 1 each step, dx=1: each segment = sqrt(1²+1²) = sqrt(2)
        ydata = np.array([0.0, 1.0, 2.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1},
                      yscale={"units": "mV"})
        data.xscale.units = "mV"
        r = nms.stat(data, {"name": "pathlength"})
        self.assertAlmostEqual(r["s"], 3 * math.sqrt(2))

    def test_pathlength_unit_mismatch_raises(self):
        ydata = np.array([1.0, 2.0, 3.0])
        data = NMData(NM, name="d", nparray=ydata,
                      xscale={"start": 0, "delta": 1},
                      yscale={"units": "mV"})
        data.xscale.units = "ms"  # different string units → ValueError
        with self.assertRaises(ValueError):
            nms.stat(data, {"name": "pathlength"})

    def test_mean_with_nans_ignored(self):
        r = nms.stat(self.datanan, {"name": "mean"}, ignore_nans=True)
        self.assertFalse(math.isnan(r["s"]))

    def test_mean_with_nans_not_ignored(self):
        r = nms.stat(self.datanan, {"name": "mean"}, ignore_nans=False)
        self.assertTrue(math.isnan(r["s"]))


# =========================================================================
# NMStatWin
# =========================================================================

class TestNMStatWin(unittest.TestCase):
    """Tests for NMStatWin."""

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
        self.w0 = nms.NMStatWin(NM, "w0", win=self.win0)
        self.w1 = nms.NMStatWin(NM, "w1", win=self.win1)
        self.data = _make_data(n=n)
        self.datanan = _make_data(n=n, with_nans=True)

    def test_init_type_errors(self):
        for b in nmu.badtypes(ok=[{}, None]):
            with self.assertRaises(TypeError):
                nms.NMStatWin(win=b)
        with self.assertRaises(TypeError):
            nms.NMStatWin(copy=NM)  # unexpected kwarg

    def test_eq_different_funcs(self):
        self.assertFalse(self.w0 == self.w1)

    def test_eq_same_after_copy(self):
        c = self.w0.copy()
        self.assertTrue(self.w0 == c)

    def test_eq_after_win_set(self):
        w0 = nms.NMStatWin(name="same")
        w1 = nms.NMStatWin(name="same")
        w0._win_set(self.win0)
        w1._win_set(self.win0)
        self.assertTrue(w0 == w1)
        w1.x0 = -1
        self.assertFalse(w0 == w1)

    def test_name_setter_removed(self):
        w = nms.NMStatWin(name="w0")
        with self.assertRaises(AttributeError):
            w.name = "w1"

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

    def test_func_set_maxmin_n_avg(self):
        self.w1._func_set({"name": "mean@max", "n_avg": 7})
        self.w1._func_set({"n_avg": "3"})
        self.assertEqual(self.w1.func["name"], "mean@max")
        self.assertEqual(self.w1.func["n_avg"], 3)

    def test_func_set_maxmin_n_avg_bool_raises(self):
        self.w1._func_set({"name": "mean@max", "n_avg": 7})
        with self.assertRaises(TypeError):
            self.w1._func_set({"n_avg": True})

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

        # round-trip via NMStatWin constructor
        w2 = nms.NMStatWin(win=win)
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
            self.w1._bsln_func_set({"name": "mean@max", "n_avg": 7})

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
        self.w1.func = {"name": "mean@max", "n_avg": 0}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertIn("warning", r[1]["func"])

    def test_compute_mean_at_max_with_n_avg(self):
        self.w1.func = {"name": "mean@max", "n_avg": 5}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(r[1]["func"]["name"], "mean@max")
        self.assertEqual(r[1]["func"]["n_avg"], 5)

    def test_compute_mean_at_min(self):
        self.w1.func = {"name": "mean@min", "n_avg": 5}
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
        self.w1.func = {"name": "level+", "n_std": 2}
        self.w1.bsln_on = True
        with self.assertRaises(RuntimeError):
            self.w1.bsln_func = "mean"  # need mean+std
            self.w1.compute(self.data, xclip=True, ignore_nans=True)

    def test_compute_level_nstd(self):
        n_std = 2
        self.w1.func = {"name": "level+", "n_std": n_std}
        self.w1.bsln_on = True
        self.w1.bsln_func = "mean+std"
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertIn("ylevel", r[1]["func"])
        self.assertAlmostEqual(
            round(r[1]["Δs"] * 1000),
            round(n_std * r[0]["std"] * 1000))

    def test_compute_level_nstd_negative(self):
        n_std = -2
        self.w1.func = {"name": "level-", "n_std": n_std}
        self.w1.bsln_func = "mean+std"
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertAlmostEqual(
            round(r[1]["Δs"] * 1000),
            round(n_std * r[0]["std"] * 1000))

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
            self.assertIn("dx", r[2])

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
            self.assertIn("dx", r[3])
            if not math.isnan(r[3]["dx"]):
                self.assertAlmostEqual(r[3]["dx"], r[3]["x"] - r[2]["x"])
            expected_len = (5 if slope and not math.isnan(r[3]["dx"])
                            else 4)
            self.assertEqual(len(r), expected_len)
            for i in range(1, expected_len):
                self.assertEqual(r[i]["id"], f)
            if slope and expected_len == 5:
                if "error" not in r[4]:
                    self.assertIn("s", r[4])
                    self.assertIn("b", r[4])


# =========================================================================
# NMStatWinContainer
# =========================================================================

class TestNMStatWinContainer(unittest.TestCase):
    """Tests for NMStatWinContainer."""

    def test_new_creates_window(self):
        c = nms.NMStatWinContainer()
        w = c.new()
        self.assertIsInstance(w, nms.NMStatWin)
        self.assertEqual(len(c), 1)
        self.assertEqual(c.selected_name, "w0")

    def test_sequential_names(self):
        c = nms.NMStatWinContainer()
        w0 = c.new()
        w1 = c.new()
        self.assertEqual(w0.name, "w0")
        self.assertEqual(w1.name, "w1")

    def test_len(self):
        c = nms.NMStatWinContainer()
        self.assertEqual(len(c), 0)
        c.new()
        c.new()
        self.assertEqual(len(c), 2)

    def test_getitem(self):
        c = nms.NMStatWinContainer()
        w = c.new()
        self.assertIs(c["w0"], w)

    def test_contains(self):
        c = nms.NMStatWinContainer()
        c.new()
        self.assertIn("w0", c)
        self.assertNotIn("w99", c)

    def test_iter(self):
        c = nms.NMStatWinContainer()
        c.new()
        c.new()
        wins = list(c)
        self.assertEqual(len(wins), 2)
        self.assertIsInstance(wins[0], nms.NMStatWin)

    def test_custom_prefix(self):
        c = nms.NMStatWinContainer(name_prefix="win")
        w = c.new()
        self.assertEqual(w.name, "win0")


class TestNMToolStats(unittest.TestCase):
    """Tests for NMToolStats save flags and _save_history()."""

    def setUp(self):
        self.tool = nms.NMToolStats()

    # --- defaults ---

    def test_save_history_default(self):
        self.assertFalse(self.tool.save_history)

    def test_save_cache_default(self):
        self.assertTrue(self.tool.save_cache)

    def test_save_numpy_default(self):
        self.assertFalse(self.tool.save_numpy)

    # --- save_history ---

    def test_save_history_set_true(self):
        self.tool.save_history = True
        self.assertTrue(self.tool.save_history)

    def test_save_history_set_false(self):
        self.tool.save_history = True
        self.tool.save_history = False
        self.assertFalse(self.tool.save_history)

    def test_save_history_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.save_history = 1

    def test_save_history_rejects_none(self):
        with self.assertRaises(TypeError):
            self.tool.save_history = None

    # --- save_cache ---

    def test_save_cache_set_false(self):
        self.tool.save_cache = False
        self.assertFalse(self.tool.save_cache)

    def test_save_cache_set_true(self):
        self.tool.save_cache = False
        self.tool.save_cache = True
        self.assertTrue(self.tool.save_cache)

    def test_save_cache_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.save_cache = "yes"

    # --- save_numpy ---

    def test_save_numpy_set_true(self):
        self.tool.save_numpy = True
        self.assertTrue(self.tool.save_numpy)

    def test_save_numpy_set_false(self):
        self.tool.save_numpy = True
        self.tool.save_numpy = False
        self.assertFalse(self.tool.save_numpy)

    def test_save_numpy_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.save_numpy = 0

    # --- _save_history ---

    def test__save_history_empty_results(self):
        # Should not raise with no results
        self.tool._save_history(quiet=True)

    def test__save_history_with_results(self):
        # Populate results via compute then call print
        data = _make_data()
        w = list(self.tool.windows)[0]
        w.func = "mean"
        w.compute(data)
        self.tool._save_history(quiet=True)

    # --- _save_numpy ---

    def _setup_folder(self):
        """Create a real NMFolder and wire it into the tool's selection."""
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.analysis.nm_tool import SELECT_LEVELS
        folder = NMFolder(name="TestFolder")
        self.tool._select = {level: None for level in SELECT_LEVELS}
        self.tool._select["folder"] = folder
        return folder

    def _run_compute(self, func="mean", n_waves=3):
        """Compute stat for n_waves and accumulate into tool results."""
        w = list(self.tool.windows)[0]
        w.func = func
        for k in range(n_waves):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in self.tool._NMToolStats__results:
                self.tool._NMToolStats__results[wname].append(results)
            else:
                self.tool._NMToolStats__results[wname] = [results]

    def test_save_numpy_returns_toolfolder(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self._setup_folder()
        self._run_compute()
        f = self.tool._save_numpy()
        self.assertIsInstance(f, NMToolFolder)

    def test_save_numpy_folder_named_stats0(self):
        self._setup_folder()
        self._run_compute()
        f = self.tool._save_numpy()
        self.assertEqual(f.name, "stats0")

    def test_save_numpy_second_run_named_stats1(self):
        self._setup_folder()
        self._run_compute()
        self.tool._save_numpy()
        self.tool._NMToolStats__results.clear()
        self._run_compute()
        f = self.tool._save_numpy()
        self.assertEqual(f.name, "stats1")

    def test_save_numpy_creates_data_array(self):
        self._setup_folder()
        self._run_compute(n_waves=3)
        f = self.tool._save_numpy()
        self.assertIn("ST_w0_data", f.data)

    def test_save_numpy_data_array_length(self):
        self._setup_folder()
        self._run_compute(n_waves=3)
        f = self.tool._save_numpy()
        d = f.data.get("ST_w0_data")
        self.assertEqual(len(d.nparray), 3)

    def test_save_numpy_creates_s_array(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._save_numpy()
        self.assertIn("ST_w0_main_s", f.data)

    def test_save_numpy_s_array_length(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._save_numpy()
        d = f.data.get("ST_w0_main_s")
        self.assertEqual(len(d.nparray), 3)

    def test_save_numpy_no_folder_returns_none(self):
        from pyneuromatic.analysis.nm_tool import SELECT_LEVELS
        self.tool._select = {level: None for level in SELECT_LEVELS}
        # folder is None — should return None
        result = self.tool._save_numpy()
        self.assertIsNone(result)

    def test_save_numpy_no_results_raises(self):
        self._setup_folder()
        with self.assertRaises(RuntimeError):
            self.tool._save_numpy()


class TestStats(unittest.TestCase):
    """Tests for the stats() summary statistics function."""

    def setUp(self):
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.arr_nan = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])

    # --- type validation ---

    def test_rejects_non_array(self):
        with self.assertRaises(TypeError):
            nms.stats([1, 2, 3])

    def test_rejects_non_dict_results(self):
        with self.assertRaises(TypeError):
            nms.stats(self.arr, results="bad")

    # --- result keys ---

    def test_returns_dict(self):
        r = nms.stats(self.arr)
        self.assertIsInstance(r, dict)

    def test_all_keys_present(self):
        r = nms.stats(self.arr)
        for key in ("N", "NaNs", "INFs", "mean", "std", "sem", "rms", "min", "max"):
            self.assertIn(key, r)

    # --- correct values ---

    def test_N(self):
        r = nms.stats(self.arr)
        self.assertEqual(r["N"], 5)

    def test_NaNs(self):
        r = nms.stats(self.arr_nan)
        self.assertEqual(r["NaNs"], 1)

    def test_INFs(self):
        arr = np.array([1.0, float("inf"), 3.0])
        r = nms.stats(arr)
        self.assertEqual(r["INFs"], 1)

    def test_mean(self):
        r = nms.stats(self.arr)
        self.assertAlmostEqual(r["mean"], 3.0)

    def test_std(self):
        r = nms.stats(self.arr)
        self.assertAlmostEqual(r["std"], float(np.std(self.arr, ddof=1)))

    def test_sem(self):
        r = nms.stats(self.arr)
        self.assertAlmostEqual(r["sem"], r["std"] / math.sqrt(5))

    def test_min(self):
        r = nms.stats(self.arr)
        self.assertAlmostEqual(r["min"], 1.0)

    def test_max(self):
        r = nms.stats(self.arr)
        self.assertAlmostEqual(r["max"], 5.0)

    def test_rms(self):
        r = nms.stats(self.arr)
        expected = math.sqrt(np.mean(np.square(self.arr)))
        self.assertAlmostEqual(r["rms"], expected)

    # --- ignore_nans ---

    def test_ignore_nans_N(self):
        r = nms.stats(self.arr_nan, ignore_nans=True)
        self.assertEqual(r["N"], 4)

    def test_ignore_nans_mean(self):
        r = nms.stats(self.arr_nan, ignore_nans=True)
        self.assertAlmostEqual(r["mean"], (1 + 3 + 4 + 5) / 4)

    def test_ignore_nans_false_propagates_nan(self):
        r = nms.stats(self.arr_nan, ignore_nans=False)
        self.assertTrue(math.isnan(r["mean"]))

    # --- empty array ---

    def test_empty_array_returns_nans(self):
        r = nms.stats(np.array([]), ignore_nans=True)
        self.assertEqual(r["N"], 0)
        self.assertTrue(math.isnan(r["mean"]))

    # --- populates provided dict ---

    def test_populates_provided_dict(self):
        r = {"existing": 99}
        nms.stats(self.arr, results=r)
        self.assertIn("mean", r)
        self.assertEqual(r["existing"], 99)


class TestNMToolStats2(unittest.TestCase):
    """Tests for NMToolStats2."""

    def setUp(self):
        self.tool2 = nms.NMToolStats2()
        # Build a NMToolFolder with some ST_ arrays
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_main_s", nparray=arr.copy())
        self.tf.data.new("ST_w0_bsln_s", nparray=arr.copy() * 0.1)
        self.tf.data.new("ST_w0_data",
                         nparray=np.array(["p0", "p1", "p2", "p3", "p4"],
                                          dtype=object))

    # --- defaults ---

    def test_save_history_default(self):
        self.assertFalse(self.tool2.save_history)

    def test_save_cache_default(self):
        self.assertTrue(self.tool2.save_cache)

    def test_save_numpy_default(self):
        self.assertFalse(self.tool2.save_numpy)

    # --- save flag setters ---

    def test_save_history_set(self):
        self.tool2.save_history = True
        self.assertTrue(self.tool2.save_history)

    def test_save_cache_set(self):
        self.tool2.save_cache = False
        self.assertFalse(self.tool2.save_cache)

    def test_save_numpy_set(self):
        self.tool2.save_numpy = True
        self.assertTrue(self.tool2.save_numpy)

    def test_save_history_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool2.save_history = 1

    def test_save_cache_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool2.save_cache = "yes"

    def test_save_numpy_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool2.save_numpy = 0

    # --- ignore_nans ---

    def test_ignore_nans_default(self):
        self.assertTrue(self.tool2.ignore_nans)

    def test_ignore_nans_set(self):
        self.tool2.ignore_nans = False
        self.assertFalse(self.tool2.ignore_nans)

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool2.ignore_nans = 1

    def test_ignore_nans_instance_used_by_compute(self):
        arr_nan = np.array([1.0, np.nan, 3.0])
        self.tf.data.new("ST_w0_nan_s", nparray=arr_nan)
        self.tool2.ignore_nans = True
        r = self.tool2.compute(self.tf, select="ST_w0_nan_s")
        self.assertEqual(r["ST_w0_nan_s"]["N"], 2)

    def test_ignore_nans_param_overrides_instance(self):
        arr_nan = np.array([1.0, np.nan, 3.0])
        self.tf.data.new("ST_w0_nan_s", nparray=arr_nan)
        self.tool2.ignore_nans = True
        r = self.tool2.compute(self.tf, select="ST_w0_nan_s", ignore_nans=False)
        self.assertEqual(r["ST_w0_nan_s"]["N"], 3)  # NaN counted

    # --- compute() type validation ---

    def test_compute_rejects_non_toolfolder(self):
        with self.assertRaises(TypeError):
            self.tool2.compute("bad")

    def test_compute_rejects_non_string_select(self):
        with self.assertRaises(TypeError):
            self.tool2.compute(self.tf, select=123)

    def test_compute_unknown_select_raises(self):
        with self.assertRaises(KeyError):
            self.tool2.compute(self.tf, select="ST_w0_missing")

    # --- compute() results ---

    def test_compute_all_returns_dict(self):
        r = self.tool2.compute(self.tf, select="all")
        self.assertIsInstance(r, dict)

    def test_compute_all_keys_are_st_arrays(self):
        r = self.tool2.compute(self.tf, select="all")
        self.assertIn("ST_w0_main_s", r)
        self.assertIn("ST_w0_bsln_s", r)

    def test_compute_all_excludes_data_array(self):
        r = self.tool2.compute(self.tf, select="all")
        self.assertNotIn("ST_w0_data", r)

    def test_compute_single_array(self):
        r = self.tool2.compute(self.tf, select="ST_w0_main_s")
        self.assertIn("ST_w0_main_s", r)
        self.assertEqual(len(r), 1)

    def test_compute_mean_correct(self):
        r = self.tool2.compute(self.tf, select="ST_w0_main_s")
        self.assertAlmostEqual(r["ST_w0_main_s"]["mean"], 3.0)

    def test_compute_N_correct(self):
        r = self.tool2.compute(self.tf, select="ST_w0_main_s")
        self.assertEqual(r["ST_w0_main_s"]["N"], 5)

    # --- _save_numpy_results ---

    def test_save_numpy_creates_st2_data(self):
        self.tool2.save_cache = False
        self.tool2.save_numpy = True
        self.tool2.compute(self.tf, select="all")
        self.assertIn("ST2_data", self.tf.data)

    def test_save_numpy_creates_st2_mean(self):
        self.tool2.save_cache = False
        self.tool2.save_numpy = True
        self.tool2.compute(self.tf, select="all")
        self.assertIn("ST2_mean", self.tf.data)

    def test_save_numpy_st2_data_length(self):
        self.tool2.save_cache = False
        self.tool2.save_numpy = True
        self.tool2.compute(self.tf, select="all")
        d = self.tf.data.get("ST2_data")
        self.assertEqual(len(d.nparray), 2)  # 2 ST_ numeric arrays

    def test_save_numpy_st2_mean_value(self):
        self.tool2.save_cache = False
        self.tool2.save_numpy = True
        self.tool2.compute(self.tf, select="ST_w0_main_s")
        d = self.tf.data.get("ST2_mean")
        self.assertAlmostEqual(d.nparray[0], 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
