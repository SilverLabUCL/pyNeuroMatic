#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_stat_win: NMStatWin and NMStatWinContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import unittest

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_stat_win as nmsw
from pyneuromatic.analysis.nm_tool_stats import FUNC_NAMES
from pyneuromatic.analysis.nm_stat_func import FUNC_NAMES_BSLN
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
        self.w0 = nmsw.NMStatWin(NM, "w0", win=self.win0)
        self.w1 = nmsw.NMStatWin(NM, "w1", win=self.win1)
        self.data = _make_data(n=n)
        self.datanan = _make_data(n=n, with_nans=True)

    def test_init_type_errors(self):
        for b in nmu.badtypes(ok=[{}, None]):
            with self.assertRaises(TypeError):
                nmsw.NMStatWin(win=b)
        with self.assertRaises(TypeError):
            nmsw.NMStatWin(copy=NM)  # unexpected kwarg

    def test_eq_different_funcs(self):
        self.assertFalse(self.w0 == self.w1)

    def test_eq_same_after_copy(self):
        c = self.w0.copy()
        self.assertTrue(self.w0 == c)

    def test_eq_after_win_set(self):
        w0 = nmsw.NMStatWin(name="same")
        w1 = nmsw.NMStatWin(name="same")
        w0._win_set(self.win0)
        w1._win_set(self.win0)
        self.assertTrue(w0 == w1)
        w1.x0 = -1
        self.assertFalse(w0 == w1)

    def test_name_setter_removed(self):
        w = nmsw.NMStatWin(name="w0")
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
        self.assertEqual(len(FUNC_NAMES), len(fnames))
        for f in fnames:
            self.assertIn(f, FUNC_NAMES)

    def test_bsln_func_names_complete(self):
        fnames = ("median", "mean", "mean+var", "mean+std", "mean+sem")
        self.assertEqual(len(FUNC_NAMES_BSLN), len(fnames))
        for f in fnames:
            self.assertIn(f, FUNC_NAMES_BSLN)

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
        w2 = nmsw.NMStatWin(win=win)
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
        c = nmsw.NMStatWinContainer()
        w = c.new()
        self.assertIsInstance(w, nmsw.NMStatWin)
        self.assertEqual(len(c), 1)
        self.assertEqual(c.selected_name, "w0")

    def test_sequential_names(self):
        c = nmsw.NMStatWinContainer()
        w0 = c.new()
        w1 = c.new()
        self.assertEqual(w0.name, "w0")
        self.assertEqual(w1.name, "w1")

    def test_len(self):
        c = nmsw.NMStatWinContainer()
        self.assertEqual(len(c), 0)
        c.new()
        c.new()
        self.assertEqual(len(c), 2)

    def test_getitem(self):
        c = nmsw.NMStatWinContainer()
        w = c.new()
        self.assertIs(c["w0"], w)

    def test_contains(self):
        c = nmsw.NMStatWinContainer()
        c.new()
        self.assertIn("w0", c)
        self.assertNotIn("w99", c)

    def test_iter(self):
        c = nmsw.NMStatWinContainer()
        c.new()
        c.new()
        wins = list(c)
        self.assertEqual(len(wins), 2)
        self.assertIsInstance(wins[0], nmsw.NMStatWin)

    def test_custom_prefix(self):
        c = nmsw.NMStatWinContainer(name_prefix="win")
        w = c.new()
        self.assertEqual(w.name, "win0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
