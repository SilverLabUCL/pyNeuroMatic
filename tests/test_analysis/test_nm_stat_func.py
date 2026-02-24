#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_stat_func: NMStatFunc class hierarchy and _stat_func_from_dict.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import math
import unittest

from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_stat_func as nmsf
import pyneuromatic.core.nm_utilities as nmu

NM = NMManager(quiet=True)


# =========================================================================
# badvalue()
# =========================================================================

class TestBadValue(unittest.TestCase):
    """Tests for the badvalue() utility function."""

    def test_none_is_bad(self):
        self.assertTrue(nmsf.badvalue(None))

    def test_nan_is_bad(self):
        self.assertTrue(nmsf.badvalue(math.nan))

    def test_inf_is_bad(self):
        self.assertTrue(nmsf.badvalue(math.inf))
        self.assertTrue(nmsf.badvalue(-math.inf))

    def test_valid_numbers_are_ok(self):
        self.assertFalse(nmsf.badvalue(0.0))
        self.assertFalse(nmsf.badvalue(1.0))
        self.assertFalse(nmsf.badvalue(-1.0))


# =========================================================================
# NMStatFunc base class
# =========================================================================

class TestNMStatFunc(unittest.TestCase):
    """Tests for NMStatFunc base class."""

    def test_name_property(self):
        t = nmsf.NMStatFunc("test")
        self.assertEqual(t.name, "test")

    def test_needs_baseline_false(self):
        t = nmsf.NMStatFunc("test")
        self.assertFalse(t.needs_baseline)

    def test_to_dict(self):
        t = nmsf.NMStatFunc("test")
        self.assertEqual(t.to_dict(), {"name": "test"})

    def test_getitem(self):
        t = nmsf.NMStatFunc("test")
        self.assertEqual(t["name"], "test")
        with self.assertRaises(KeyError):
            t["nonexistent"]

    def test_eq_with_instance(self):
        t1 = nmsf.NMStatFuncBasic("mean")
        t2 = nmsf.NMStatFuncBasic("mean")
        t3 = nmsf.NMStatFuncBasic("median")
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_eq_with_dict(self):
        t = nmsf.NMStatFuncBasic("mean")
        self.assertEqual(t, {"name": "mean"})

    def test_eq_with_other_type(self):
        t = nmsf.NMStatFuncBasic("mean")
        self.assertEqual(t.__eq__(42), NotImplemented)

    def test_repr_contains_class_and_name(self):
        t = nmsf.NMStatFuncBasic("mean")
        self.assertIn("NMStatFuncBasic", repr(t))
        self.assertIn("mean", repr(t))

    def test_deepcopy_resets_parent(self):
        t = nmsf.NMStatFuncBasic("mean", parent=object())
        t2 = copy.deepcopy(t)
        self.assertIsNone(t2._parent)
        self.assertEqual(t, t2)

    def test_validate_baseline_is_noop(self):
        t = nmsf.NMStatFunc("test")
        t.validate_baseline(None)  # must not raise

    def test_compute_raises_not_implemented(self):
        t = nmsf.NMStatFunc("test")
        with self.assertRaises(NotImplementedError):
            t.compute(None, 0, 1, False, False, None, {})


# =========================================================================
# NMStatFuncBasic
# =========================================================================

class TestNMStatFuncBasic(unittest.TestCase):
    """Tests for NMStatFuncBasic."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncBasic("badfuncname")

    def test_all_valid_names(self):
        for f in nmsf.FUNC_NAMES_BASIC:
            t = nmsf.NMStatFuncBasic(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_needs_baseline_false(self):
        self.assertFalse(nmsf.NMStatFuncBasic("mean").needs_baseline)

    def test_from_dict_round_trip(self):
        for f in nmsf.FUNC_NAMES_BASIC:
            t = nmsf._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nmsf.NMStatFuncBasic)
            self.assertEqual(t.name, f)


# =========================================================================
# NMStatFuncMaxMin
# =========================================================================

class TestNMStatFuncMaxMin(unittest.TestCase):
    """Tests for NMStatFuncMaxMin."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncMaxMin("badfuncname")

    def test_mean_at_max_requires_n_avg(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncMaxMin("mean@max")

    def test_n_avg_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncMaxMin("mean@max", n_avg=b)

    def test_n_avg_value_errors(self):
        for b in [-10, math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncMaxMin("mean@max", n_avg=b)

    def test_n_avg_overflow(self):
        with self.assertRaises(OverflowError):
            nmsf.NMStatFuncMaxMin("mean@max", n_avg=math.inf)

    def test_max_min_without_n_avg(self):
        for f in ("max", "min"):
            t = nmsf.NMStatFuncMaxMin(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test_n_avg_upgrades_to_mean_at(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nmsf.NMStatFuncMaxMin(f, n_avg=10)
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t.to_dict()["n_avg"], 10)

    def test_from_dict(self):
        for f in ["max", "min", "mean@max", "mean@min"]:
            t = nmsf._stat_func_from_dict({"name": f, "n_avg": 10})
            expected = ("mean@" + f) if f in ("max", "min") else f
            self.assertEqual(t.name, expected)
            self.assertEqual(t["n_avg"], 10)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict({"name": "max", "badkey": 0})


# =========================================================================
# NMStatFuncLevel
# =========================================================================

class TestNMStatFuncLevel(unittest.TestCase):
    """Tests for NMStatFuncLevel (explicit ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncLevel("badfuncname", ylevel=10)

    def test_missing_ylevel_raises(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncLevel("level")

    def test_ylevel_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncLevel("level", ylevel=b)

    def test_ylevel_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncLevel("level", ylevel=b)

    def test_valid_construction(self):
        for f in nmsf.FUNC_NAMES_LEVEL:
            t = nmsf.NMStatFuncLevel(f, ylevel=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["ylevel"], 10)
            self.assertFalse(t.needs_baseline)

    def test_from_dict(self):
        t = nmsf._stat_func_from_dict({"name": "level+", "ylevel": 10})
        self.assertIsInstance(t, nmsf.NMStatFuncLevel)
        self.assertEqual(t.name, "level+")
        self.assertEqual(t["ylevel"], 10)

    def test_from_dict_missing_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict({"name": "level"})

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict({"name": "level", "badkey": 0})


# =========================================================================
# NMStatFuncLevelNstd
# =========================================================================

class TestNMStatFuncLevelNstd(unittest.TestCase):
    """Tests for NMStatFuncLevelNstd (n_std-based ylevel)."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncLevelNstd("badfuncname", n_std=2)

    def test_missing_n_std_raises(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncLevelNstd("level")

    def test_n_std_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncLevelNstd("level", n_std=b)

    def test_n_std_value_errors(self):
        for b in [math.nan, math.inf, "badvalue", 0]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncLevelNstd("level", n_std=b)

    def test_valid_construction(self):
        for f in nmsf.FUNC_NAMES_LEVEL:
            t = nmsf.NMStatFuncLevelNstd(f, n_std=2)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["n_std"], 2)
            self.assertTrue(t.needs_baseline)

    def test_negative_n_std_valid(self):
        t = nmsf.NMStatFuncLevelNstd("level", n_std=-2)
        self.assertEqual(t.to_dict()["n_std"], -2)

    def test_validate_baseline(self):
        t = nmsf.NMStatFuncLevelNstd("level", n_std=2)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        with self.assertRaises(RuntimeError):
            t.validate_baseline("mean")
        t.validate_baseline("mean+std")  # ok

    def test_from_dict(self):
        t = nmsf._stat_func_from_dict({"name": "level-", "n_std": -2})
        self.assertIsInstance(t, nmsf.NMStatFuncLevelNstd)
        self.assertEqual(t.name, "level-")
        self.assertEqual(t["n_std"], -2)

    def test_from_dict_both_keys_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict(
                {"name": "level", "ylevel": 10, "n_std": 2})


# =========================================================================
# NMStatFuncRiseTime
# =========================================================================

class TestNMStatFuncRiseTime(unittest.TestCase):
    """Tests for NMStatFuncRiseTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncRiseTime("badfuncname", p0=10, p1=90)
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncRiseTime("falltime+", p0=10, p1=90)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncRiseTime("risetime+")
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncRiseTime("risetime+", p1=90)

    def test_missing_p1_raises(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncRiseTime("risetime+", p0=10)

    def test_p0_p1_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncRiseTime("risetime+", p0=10, p1=b)

    def test_p0_p1_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncRiseTime("risetime+", p0=b, p1=90)
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncRiseTime("risetime+", p0=10, p1=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncRiseTime("risetime+", p0=b, p1=90)

    def test_p0_ge_p1_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncRiseTime("risetime+", p0=90, p1=10)

    def test_valid_construction(self):
        for f in nmsf.FUNC_NAMES_RISETIME:
            t = nmsf.NMStatFuncRiseTime(f, p0=10, p1=90)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 10)
            self.assertEqual(t.to_dict()["p1"], 90)

    def test_float_p_values(self):
        t = nmsf.NMStatFuncRiseTime("risetime+", p0=10.5, p1=89.5)
        self.assertEqual(t.to_dict()["p0"], 10.5)
        self.assertEqual(t.to_dict()["p1"], 89.5)

    def test_needs_baseline(self):
        t = nmsf.NMStatFuncRiseTime("risetime+", p0=10, p1=90)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nmsf.NMStatFuncRiseTime("risetime+", p0=10, p1=90)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nmsf._stat_func_from_dict(
            {"name": "risetime+", "p0": 10, "p1": 90})
        self.assertIsInstance(t, nmsf.NMStatFuncRiseTime)
        self.assertEqual(t.name, "risetime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict(
                {"name": "risetime+", "p0": 10, "badkey": 90})


# =========================================================================
# NMStatFuncFallTime
# =========================================================================

class TestNMStatFuncFallTime(unittest.TestCase):
    """Tests for NMStatFuncFallTime."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncFallTime("badfuncname", p0=90)
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncFallTime("risetime+", p0=10, p1=90)

    def test_missing_p0_raises(self):
        with self.assertRaises(KeyError):
            nmsf.NMStatFuncFallTime("falltime+")

    def test_p0_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncFallTime("falltime+", p0=b)

    def test_p0_value_errors(self):
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncFallTime("falltime+", p0=b)
        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncFallTime("falltime+", p0=b)

    def test_p1_type_errors(self):
        for b in [[], (), {}, set(), NM, True, False]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncFallTime("falltime+", p0=90, p1=b)

    def test_p0_le_p1_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncFallTime("falltime+", p0=10, p1=90)

    def test_p0_without_p1(self):
        t = nmsf.NMStatFuncFallTime("falltime+", p0=36)
        self.assertEqual(t.to_dict()["p0"], 36)
        self.assertIsNone(t.to_dict()["p1"])

    def test_valid_construction(self):
        for f in nmsf.FUNC_NAMES_FALLTIME:
            t = nmsf.NMStatFuncFallTime(f, p0=90, p1=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 90)
            self.assertEqual(t.to_dict()["p1"], 10)

    def test_needs_baseline(self):
        t = nmsf.NMStatFuncFallTime("falltime+", p0=90, p1=10)
        self.assertTrue(t.needs_baseline)

    def test_validate_baseline(self):
        t = nmsf.NMStatFuncFallTime("falltime+", p0=90, p1=10)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict(self):
        t = nmsf._stat_func_from_dict(
            {"name": "falltime+", "p0": 90, "p1": 10})
        self.assertIsInstance(t, nmsf.NMStatFuncFallTime)
        self.assertEqual(t.name, "falltime+")

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict(
                {"name": "falltime+", "p0": 90, "badkey": 10})


# =========================================================================
# NMStatFuncFWHM
# =========================================================================

class TestNMStatFuncFWHM(unittest.TestCase):
    """Tests for NMStatFuncFWHM."""

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf.NMStatFuncFWHM("badfuncname")

    def test_p_type_errors(self):
        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(TypeError):
                nmsf.NMStatFuncFWHM("fwhm+", p0=50, p1=b)

    def test_p_value_errors(self):
        for b in [-10, 110, math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(ValueError):
                nmsf.NMStatFuncFWHM("fwhm+", p0=50, p1=b)

    def test_defaults_to_50_50(self):
        for f in nmsf.FUNC_NAMES_FWHM:
            t = nmsf.NMStatFuncFWHM(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 50)
            self.assertEqual(t.to_dict()["p1"], 50)

    def test_custom_p_values(self):
        t = nmsf.NMStatFuncFWHM("fwhm+", p0=45, p1=55)
        self.assertEqual(t.to_dict()["p0"], 45)
        self.assertEqual(t.to_dict()["p1"], 55)

    def test_needs_baseline(self):
        self.assertTrue(nmsf.NMStatFuncFWHM("fwhm+").needs_baseline)

    def test_validate_baseline(self):
        t = nmsf.NMStatFuncFWHM("fwhm+")
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")
        t.validate_baseline("median")

    def test_from_dict_defaults(self):
        t = nmsf._stat_func_from_dict({"name": "fwhm+"})
        self.assertEqual(t.name, "fwhm+")
        self.assertEqual(t["p0"], 50)
        self.assertEqual(t["p1"], 50)

    def test_from_dict_custom_values(self):
        t = nmsf._stat_func_from_dict({"name": "fwhm-", "p0": 45, "p1": 55})
        self.assertEqual(t["p0"], 45)
        self.assertEqual(t["p1"], 55)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict({"name": "fwhm+", "badkey": 0})


# =========================================================================
# _stat_func_from_dict factory
# =========================================================================

class TestStatFuncFromDict(unittest.TestCase):
    """Tests for the _stat_func_from_dict() factory function."""

    def test_none_returns_none(self):
        self.assertIsNone(nmsf._stat_func_from_dict(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(nmsf._stat_func_from_dict({}))

    def test_none_name_returns_none(self):
        self.assertIsNone(nmsf._stat_func_from_dict({"name": None}))

    def test_missing_name_key_raises(self):
        with self.assertRaises(KeyError):
            nmsf._stat_func_from_dict({"badkey": "mean"})

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            nmsf._stat_func_from_dict({"name": "badname"})

    def test_bad_input_type_raises(self):
        with self.assertRaises(TypeError):
            nmsf._stat_func_from_dict(42)

    def test_bad_name_type_raises(self):
        with self.assertRaises(TypeError):
            nmsf._stat_func_from_dict({"name": 42})

    def test_string_shorthand(self):
        t = nmsf._stat_func_from_dict("mean")
        self.assertIsInstance(t, nmsf.NMStatFuncBasic)
        self.assertEqual(t.name, "mean")

    def test_basic_func_round_trip(self):
        for f in nmsf.FUNC_NAMES_BASIC:
            t = nmsf._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nmsf.NMStatFuncBasic)
            self.assertEqual(t.name, f)

    def test_maxmin_without_n_avg(self):
        for f in ("max", "min"):
            t = nmsf._stat_func_from_dict({"name": f})
            self.assertEqual(t.name, f)

    def test_maxmin_with_n_avg(self):
        for f in ("mean@max", "mean@min"):
            t = nmsf._stat_func_from_dict({"name": f, "n_avg": 5})
            self.assertEqual(t.name, f)

    def test_level_ylevel(self):
        t = nmsf._stat_func_from_dict({"name": "level", "ylevel": 5})
        self.assertIsInstance(t, nmsf.NMStatFuncLevel)

    def test_level_nstd(self):
        t = nmsf._stat_func_from_dict({"name": "level", "n_std": 2})
        self.assertIsInstance(t, nmsf.NMStatFuncLevelNstd)

    def test_risetime_round_trip(self):
        for f in nmsf.FUNC_NAMES_RISETIME:
            t = nmsf._stat_func_from_dict({"name": f, "p0": 10, "p1": 90})
            self.assertIsInstance(t, nmsf.NMStatFuncRiseTime)
            self.assertEqual(t.name, f)

    def test_falltime_round_trip(self):
        for f in nmsf.FUNC_NAMES_FALLTIME:
            t = nmsf._stat_func_from_dict({"name": f, "p0": 90, "p1": 10})
            self.assertIsInstance(t, nmsf.NMStatFuncFallTime)
            self.assertEqual(t.name, f)

    def test_fwhm_round_trip(self):
        for f in nmsf.FUNC_NAMES_FWHM:
            t = nmsf._stat_func_from_dict({"name": f})
            self.assertIsInstance(t, nmsf.NMStatFuncFWHM)
            self.assertEqual(t.name, f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
