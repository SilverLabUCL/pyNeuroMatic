#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_stats: NMToolStats and NMToolStats2 tool classes.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.tools.nm_tool_stats as nms
import pyneuromatic.tools.nm_stat_win as nmsw
from pyneuromatic.tools.nm_tool_stats import NMToolStatsConfig

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


class TestNMToolStats(unittest.TestCase):
    """Tests for NMToolStats results_to_* flags and _write_results_to_history()."""

    def setUp(self):
        self.tool = nms.NMToolStats()

    # --- defaults ---

    def test_results_to_history_default(self):
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_cache_default(self):
        self.assertFalse(self.tool.results_to_cache)  # off: memory; scripts turn it on (#480)

    def test_results_to_numpy_default(self):
        self.assertTrue(self.tool.results_to_numpy)

    # --- _write_results_to_history ---

    def test_results_to_history_empty_results(self):
        # Should not raise with no results
        self.tool._write_results_to_history(quiet=True)

    def test_results_to_history_with_results(self):
        # Populate results via compute then call print
        data = _make_data()
        w = list(self.tool.windows)[0]
        w.func = "mean"
        w.compute(data)
        self.tool._write_results_to_history(quiet=True)

    # --- _write_results_to_numpy ---

    def _setup_folder(self):
        """Create a real NMFolder and wire it into the tool's selection."""
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.tools.nm_tool import HIERARCHY_SELECT_KEYS
        folder = NMFolder(name="TestFolder")
        self.tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        self.tool._select["folder"] = folder
        return folder

    def _run_compute(self, func="mean", n_arrays=3):
        """Compute stat for n_arrays and accumulate into tool results."""
        w = list(self.tool.windows)[0]
        w.func = func
        for k in range(n_arrays):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in self.tool._NMToolStats__results:
                self.tool._NMToolStats__results[wname].append(results)
            else:
                self.tool._NMToolStats__results[wname] = [results]

    def _run_compute_with_bsln(self, func="mean", bsln_func="mean", n_arrays=3):
        """Compute stat with baseline enabled for n_arrays."""
        w = list(self.tool.windows)[0]
        w.func = func
        w._bsln_on_set(True, quiet=True)
        w._bsln_func_set({"name": bsln_func}, quiet=True)
        for k in range(n_arrays):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in self.tool._NMToolStats__results:
                self.tool._NMToolStats__results[wname].append(results)
            else:
                self.tool._NMToolStats__results[wname] = [results]

    def test_results_to_numpy_returns_toolfolder(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder
        self._setup_folder()
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertIsInstance(f, NMToolFolder)

    def test_results_to_numpy_folder_named_stats_0_no_dataseries(self):
        # No dataseries selected → fallback name "Stats_0"
        self._setup_folder()
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertEqual(f.name, "Stats_0")

    def test_results_to_numpy_second_run_named_stats_1_no_dataseries(self):
        # Second run with no dataseries → "Stats_1"
        self._setup_folder()
        self._run_compute()
        self.tool._write_results_to_numpy()
        self.tool._NMToolStats__results.clear()
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertEqual(f.name, "Stats_1")

    def test_results_to_numpy_folder_named_with_dataseries(self):
        # Dataseries selected → name uses dataseries name
        from pyneuromatic.core.nm_dataseries import NMDataSeries
        self._setup_folder()
        ds = NMDataSeries(name="Record")
        self.tool._select["dataseries"] = ds
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertEqual(f.name, "Stats_Record_0")

    def test_results_to_numpy_folder_named_with_dataseries_and_channel(self):
        # Dataseries + channel → Stats_{ds}_{ch}_N
        from pyneuromatic.core.nm_dataseries import NMDataSeries
        from pyneuromatic.core.nm_channel import NMChannel
        self._setup_folder()
        self.tool._select["dataseries"] = NMDataSeries(name="Record")
        self.tool._select["channel"] = NMChannel(name="A")
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertEqual(f.name, "Stats_Record_A_0")

    def test_results_to_numpy_folder_named_channel_only(self):
        # Channel but no dataseries → Stats_{ch}_N
        from pyneuromatic.core.nm_channel import NMChannel
        self._setup_folder()
        self.tool._select["channel"] = NMChannel(name="B")
        self._run_compute()
        f = self.tool._write_results_to_numpy()
        self.assertEqual(f.name, "Stats_B_0")

    def test_results_to_numpy_creates_data_array(self):
        self._setup_folder()
        self._run_compute(n_arrays=3)
        f = self.tool._write_results_to_numpy()
        self.assertIn("ST_data_names", f.data)

    def test_results_to_numpy_data_array_length(self):
        self._setup_folder()
        self._run_compute(n_arrays=3)
        f = self.tool._write_results_to_numpy()
        d = f.data.get("ST_data_names")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_creates_mean_y_array(self):
        self._setup_folder()
        self._run_compute(func="mean", n_arrays=3)
        f = self.tool._write_results_to_numpy()
        self.assertIn("ST_w0_mean_y", f.data)

    def test_results_to_numpy_mean_y_array_length(self):
        self._setup_folder()
        self._run_compute(func="mean", n_arrays=3)
        f = self.tool._write_results_to_numpy()
        d = f.data.get("ST_w0_mean_y")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_compound_mean_std_splits(self):
        # mean+std → ST_w0_mean_y AND ST_w0_std_y
        self._setup_folder()
        self._run_compute(func="mean+std", n_arrays=3)
        f = self.tool._write_results_to_numpy()
        self.assertIn("ST_w0_mean_y", f.data)
        self.assertIn("ST_w0_std_y", f.data)
        self.assertNotIn("ST_w0_mean_std_y", f.data)
        # Its other arrays are the mean's: ST_w0_mean_n, not ST_w0_mean_std_n
        self.assertFalse([n for n in f.data if "mean_std" in n])
        self.assertIn("ST_w0_mean_n", f.data)

    def test_results_to_numpy_compound_delta_is_the_means(self):
        # mean+std/var/sem with a baseline: Δ is ST_w0_mean_dy, not ST_w0_mean_std_ds
        for func in ("mean+std", "mean+var", "mean+sem"):
            self.setUp()
            self._setup_folder()
            self._run_compute_with_bsln(func=func)
            f = self.tool._write_results_to_numpy()
            self.assertIn("ST_w0_mean_dy", f.data, func)
            self.assertIn("ST_w0_%s_y" % func[len("mean+"):], f.data, func)
            self.assertFalse([n for n in f.data if n.startswith("ST_w0_mean_" + func[5:])],
                             func)

    def test_results_to_numpy_bsln_uses_bsln_prefix(self):
        # baseline result → ST_w0_bsln_y
        self._setup_folder()
        self._run_compute_with_bsln()
        f = self.tool._write_results_to_numpy()
        self.assertIn("ST_w0_bsln_y", f.data)

    def _run_win_on_arrays(self, win, ys):
        """Compute window w0 configured by *win* on each y array in *ys*."""
        w = list(self.tool.windows)[0]
        w._win_set(win, quiet=True)
        results = self.tool._NMToolStats__results
        for k, y in enumerate(ys):
            data = NMData(NM, name="recordA%d" % k, nparray=y,
                          xscale={"start": 0.0, "delta": 1.0})
            w.compute(data)
            results.setdefault(w.name, []).append(w.results)

    @staticmethod
    def _ramp(base=-70.0, height=50.0):
        """Baseline for x 0-9, linear rise over x 10-15, then plateau."""
        return np.r_[np.full(10, base), np.linspace(base, base + height, 6),
                     np.full(5, base + height)]

    def test_results_to_numpy_saves_delta_s(self):
        self._setup_folder()
        self._run_win_on_arrays(
            {"func": "max", "xbgn": 10.0, "bsln_on": True,
             "bsln_func": "mean", "bsln_xbgn": 0.0, "bsln_xend": 9.0},
            [self._ramp(height=h) for h in (50.0, 51.0, 52.0)],
        )
        f = self.tool._write_results_to_numpy()
        self.assertEqual(list(f.data.get("ST_w0_max_dy").nparray),
                         [50.0, 51.0, 52.0])
        # Δ is _dy (as _y and _dx), in the baseline's units (#488)
        self.assertFalse([n for n in f.data if n.endswith("_ds")])
        self.assertEqual(f.data.get("ST_w0_max_dy").yscale.units,
                         f.data.get("ST_w0_bsln_y").yscale.units)

    def test_delta_only_for_a_point_or_average_on_the_trace(self):
        # Decided by measurement, not units (#488): std, sum... are in y
        # units but their Δ means nothing; data without units can't compare
        from pyneuromatic.tools.nm_stat_func import (
            FUNC_NAMES_BASIC, FUNC_NAMES_BASIC_DELTA)
        no_delta = ("var", "std", "sem", "rms", "sum", "pathlength", "area",
                    "slope", "count", "count_nans", "count_infs")
        self.assertEqual(set(FUNC_NAMES_BASIC),
                         set(FUNC_NAMES_BASIC_DELTA) | set(no_delta))
        for units in (True, False):
            for func in FUNC_NAMES_BASIC + ("max", "min", "level+"):
                self.setUp()
                self._setup_folder()
                w = list(self.tool.windows)[0]
                win = {"func": func, "bsln_on": True, "bsln_func": "mean",
                       "bsln_xbgn": 0.0, "bsln_xend": 9.0}
                if func == "level+":
                    win["func"] = {"name": "level+", "ylevel": 0.5}
                w._win_set(win, quiet=True)
                data = _make_data(units=units)
                if func == "pathlength":  # needs x and y in the same units
                    u = "mV" if units else ""
                    data = NMData(NM, name="recordA0", nparray=data.nparray,
                                  xscale={"units": u, "start": 0, "delta": 1},
                                  yscale={"units": u})
                w.compute(data)
                self.tool._NMToolStats__results[w.name] = [w.results]
                f = self.tool._write_results_to_numpy()
                saved = [n for n in f.data if n.endswith("_dy")]
                if func in no_delta:
                    self.assertEqual(saved, [], (func, units))
                    self.assertFalse(any("Δs" in r for r in w.results), (func, units))
                else:
                    self.assertEqual(len(saved), 1, (func, units, saved))

    def test_results_to_numpy_multistep_arrays_one_value_per_data(self):
        # Rise time records peak, p0 and p1 results per data array; each step
        # gets its own arrays, aligned with ST_w0_data. The last array has an
        # all-NaN baseline, so only its peak step runs; the others are NaN.
        self._setup_folder()
        bad = self._ramp()
        bad[:10] = math.nan
        self._run_win_on_arrays(
            {"func": {"name": "risetime+", "p0": 10, "p1": 90},
             "xbgn": 9.0, "bsln_on": True, "bsln_func": "mean",
             "bsln_xbgn": 0.0, "bsln_xend": 8.0},
            [self._ramp(), self._ramp(height=60.0), bad],
        )
        f = self.tool._write_results_to_numpy()
        for name in f.data:
            self.assertEqual(len(f.data.get(name).nparray), 3, name)
        np.testing.assert_allclose(f.data.get("ST_w0_rt_p_y").nparray[:2],
                                   [-20.0, -10.0])
        np.testing.assert_allclose(f.data.get("ST_w0_rt_p_dy").nparray[:2],
                                   [50.0, 60.0])
        p0_x = f.data.get("ST_w0_rt_p_p0_x").nparray
        p1_x = f.data.get("ST_w0_rt_p_p1_x").nparray
        dx = f.data.get("ST_w0_rt_p_dx").nparray
        np.testing.assert_allclose(p0_x[:2], [10.5, 10.5])
        np.testing.assert_allclose(p1_x[:2], [14.5, 14.5])
        np.testing.assert_allclose(dx[:2], [4.0, 4.0])
        self.assertTrue(math.isnan(p0_x[2]) and math.isnan(dx[2]))
        # Notes name the window's func and the step, not the helper func
        note = f.data.get("ST_w0_rt_p_p1_x").notes[0]["note"]
        self.assertIn("func=risetime+, id=risetime+, step=p1", note)

    # --- _sanitize_func_name / _st_array_name ---

    def test_sanitize_plain_name_unchanged(self):
        self.assertEqual(nms.NMToolStats._sanitize_func_name("mean"), "mean")

    def test_sanitize_risetime_plus(self):
        self.assertEqual(nms.NMToolStats._sanitize_func_name("risetime+"), "rt_p")

    def test_sanitize_falltime_minus(self):
        self.assertEqual(nms.NMToolStats._sanitize_func_name("falltime-"), "ft_m")

    def test_sanitize_fwhm_plus(self):
        self.assertEqual(nms.NMToolStats._sanitize_func_name("fwhm+"), "fwhm_p")

    def test_sanitize_pathlength(self):
        self.assertEqual(nms.NMToolStats._sanitize_func_name("pathlength"), "pathlen")

    def test_st_array_name_main_mean(self):
        n = nms.NMToolStats._st_array_name("w0", "mean", "main", "s")
        self.assertEqual(n, "ST_w0_mean_y")

    def test_st_array_name_main_mean_x(self):
        n = nms.NMToolStats._st_array_name("w0", "mean", "main", "x")
        self.assertEqual(n, "ST_w0_mean_x")

    def test_st_array_name_bsln(self):
        n = nms.NMToolStats._st_array_name("w0", "mean", "bsln", "s")
        self.assertEqual(n, "ST_w0_bsln_y")

    def test_st_array_name_bsln_std(self):
        n = nms.NMToolStats._st_array_name("w0", "mean+std", "bsln", "std")
        self.assertEqual(n, "ST_w0_bsln_std")

    def test_st_array_name_compound_primary(self):
        n = nms.NMToolStats._st_array_name("w0", "mean+std", "main", "s")
        self.assertEqual(n, "ST_w0_mean_y")

    def test_st_array_name_compound_secondary(self):
        n = nms.NMToolStats._st_array_name("w0", "mean+std", "main", "std")
        self.assertEqual(n, "ST_w0_std_y")

    def test_st_array_name_complex_func(self):
        n = nms.NMToolStats._st_array_name("w0", "risetime+", "risetime+", "dx")
        self.assertEqual(n, "ST_w0_rt_p_dx")

    def test_st_array_name_step(self):
        name = nms.NMToolStats._st_array_name
        self.assertEqual(name("w0", "level+", "risetime+", "x", "p0"),
                         "ST_w0_rt_p_p0_x")
        # dx is recorded by a single step, so it keeps the plain name
        self.assertEqual(name("w0", "level+", "risetime+", "dx", "p1"),
                         "ST_w0_rt_p_dx")
        self.assertEqual(name("w0", "max", "main", "Δs"), "ST_w0_max_dy")

    def test_results_to_numpy_no_folder_returns_none(self):
        from pyneuromatic.tools.nm_tool import HIERARCHY_SELECT_KEYS
        self.tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        # folder is None — should return None
        result = self.tool._write_results_to_numpy()
        self.assertIsNone(result)

    def test_results_to_numpy_no_results_raises(self):
        self._setup_folder()
        with self.assertRaises(RuntimeError):
            self.tool._write_results_to_numpy()


class TestNMToolStats2(unittest.TestCase):
    """Tests for NMToolStats2."""

    def setUp(self):
        # Build a NMToolFolder with some ST_ arrays
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=arr.copy())
        self.tf.data.new("ST_w0_bsln_y", nparray=arr.copy() * 0.1)
        self.tf.data.new("ST_w0_data",
                         nparray=np.array(["p0", "p1", "p2", "p3", "p4"],
                                          dtype=object))

    # --- stats() ignore_nans parameter ---

    def test_ignore_nans_default_excludes_nans(self):
        arr_nan = np.array([1.0, np.nan, 3.0])
        self.tf.data.new("ST_w0_nan_s", nparray=arr_nan)
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_nan_s")
        self.assertEqual(r["ST_w0_nan_s"]["N"], 2)

    def test_ignore_nans_false_counts_nans(self):
        arr_nan = np.array([1.0, np.nan, 3.0])
        self.tf.data.new("ST_w0_nan_s", nparray=arr_nan)
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_nan_s", ignore_nans=False)
        self.assertEqual(r["ST_w0_nan_s"]["N"], 3)

    # --- stats() type validation ---

    def test_stats_rejects_non_toolfolder(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.stats("bad")

    def test_stats_rejects_non_string_select(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.stats(self.tf, select=123)

    def test_stats_unknown_select_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.stats(self.tf, select="ST_w0_missing")

    # --- stats() results ---

    def test_stats_all_returns_dict(self):
        r = nms.NMToolStats2.stats(self.tf, select="all")
        self.assertIsInstance(r, dict)

    def test_stats_all_keys_are_st_arrays(self):
        r = nms.NMToolStats2.stats(self.tf, select="all")
        self.assertIn("ST_w0_mean_y", r)
        self.assertIn("ST_w0_bsln_y", r)

    def test_stats_all_excludes_data_array(self):
        r = nms.NMToolStats2.stats(self.tf, select="all")
        self.assertNotIn("ST_w0_data", r)

    def test_stats_single_array(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_mean_y")
        self.assertIn("ST_w0_mean_y", r)
        self.assertEqual(len(r), 1)

    def test_stats_mean_correct(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_mean_y")
        self.assertAlmostEqual(r["ST_w0_mean_y"]["mean"], 3.0)

    def test_stats_N_correct(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_mean_y")
        self.assertEqual(r["ST_w0_mean_y"]["N"], 5)

    # --- results_to_numpy parameter ---

    def test_results_to_numpy_creates_st2_data(self):
        nms.NMToolStats2.stats(self.tf, select="all", results_to_numpy=True)
        self.assertIn("ST2_data", self.tf.data)

    def test_results_to_numpy_creates_st2_mean(self):
        nms.NMToolStats2.stats(self.tf, select="all", results_to_numpy=True)
        self.assertIn("ST2_mean", self.tf.data)

    def test_results_to_numpy_st2_data_length(self):
        nms.NMToolStats2.stats(self.tf, select="all", results_to_numpy=True)
        d = self.tf.data.get("ST2_data")
        self.assertEqual(len(d.nparray), 2)  # 2 ST_ numeric arrays

    def test_results_to_numpy_st2_mean_value(self):
        nms.NMToolStats2.stats(self.tf, select="ST_w0_mean_y", results_to_numpy=True)
        d = self.tf.data.get("ST2_mean")
        self.assertAlmostEqual(d.nparray[0], 3.0)


class TestNMToolStats2Histogram(unittest.TestCase):
    """Tests for NMToolStats2.histogram() — NM wrapper behaviour only.
    Pure math tests are in test_nm_math.py::TestHistogram.
    """

    def setUp(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=arr)

    # --- saved as H_<array> (#492) ---

    def H(self, name="H_ST_w0_mean_y"):
        return self.tf.data.get(name)

    def test_histogram_saves_one_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertIn("H_ST_w0_mean_y", self.tf.data)
        self.assertNotIn("H_ST_w0_mean_y_edges", self.tf.data)  # equal bins
        self.assertNotIn("H_ST_w0_mean_y_counts", self.tf.data)

    def test_histogram_number_of_bins_and_empty_bins(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4, pad=0)
        self.assertEqual(list(self.H().nparray), [1, 2, 3, 3])  # 1 to 5
        self.assertAlmostEqual(self.H().xscale.start, 1.0)
        self.assertAlmostEqual(self.H().xscale.delta, 1.0)  # (5 - 1) / 4
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)  # 1 empty each end
        self.assertEqual(list(self.H().nparray), [0, 1, 2, 3, 3, 0])
        self.assertAlmostEqual(self.H().xscale.start, 0.0)

    def test_histogram_binwidth_round_edges_and_padded(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", binwidth=2)
        # 1 to 5 in bins of 2 from 0 (a multiple of 2), and an empty bin each end
        self.assertEqual(list(self.H().nparray), [0, 1, 5, 3, 0])
        self.assertAlmostEqual(self.H().xscale.start, -2.0)
        self.assertAlmostEqual(self.H().xscale.delta, 2.0)

    def test_histogram_auto(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y")
        counts = self.H().nparray
        self.assertEqual((counts[0], counts[-1]), (0, 0))
        self.assertEqual(counts.sum(), 9)

    def test_histogram_xrange_as_given(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=5, xrange=(0, 10))
        self.assertEqual(list(self.H().nparray), [1, 5, 3, 0, 0])  # 0, 2, ... 10; no padding
        self.assertAlmostEqual(self.H().xscale.start, 0.0)

    def test_histogram_unequal_bins_save_edges(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=[0, 2, 3, 6])
        self.assertEqual(list(self.H("H_ST_w0_mean_y_edges").nparray), [0, 2, 3, 6])
        self.assertEqual(list(self.H().nparray), [1, 2, 6])
        # Made again with equal bins: the edges are no longer needed
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=2)
        self.assertNotIn("H_ST_w0_mean_y_edges", self.tf.data)

    def test_histogram_again_replaces_it(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4, pad=0)
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=2, pad=0)
        self.assertEqual(len(self.H().nparray), 2)

    def test_histogram_outputs_and_scales(self):
        self.tf.data.get("ST_w0_mean_y").yscale.label = "Vm"
        self.tf.data.get("ST_w0_mean_y").yscale.units = "mV"
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", binwidth=0.5)
        self.assertEqual((self.H().xscale.label, self.H().xscale.units), ("Vm", "mV"))
        self.assertEqual((self.H().yscale.label, self.H().yscale.units), ("count", ""))
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", binwidth=0.5,
                                   output="probability")
        self.assertAlmostEqual(self.H().nparray.sum(), 1.0)  # heights
        self.assertEqual((self.H().yscale.label, self.H().yscale.units), ("probability", ""))
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", binwidth=0.5, output="density")
        self.assertAlmostEqual(self.H().nparray.sum() * 0.5, 1.0)  # areas
        self.assertEqual((self.H().yscale.label, self.H().yscale.units), ("density", "1/mV"))

    def test_histogram_note(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", binwidth=2)
        self.assertEqual([n["note"] for n in self.H().notes], [
            "NMHistogram(array=ST_w0_mean_y, binwidth=2, nbins=5, xbgn=-2, xend=8, "
            "pad=1, output=count, n=9)"])

    def test_histogram_no_save_skips_arrays(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4, save_to_numpy=False)
        self.assertNotIn("H_ST_w0_mean_y", self.tf.data)

    def test_histogram_bad_values(self):
        with self.assertRaises(ValueError):
            nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", output="counts")
        self.tf.data.new("ST_w0_nan_y", nparray=np.array([np.nan, np.nan]))
        with self.assertRaises(ValueError):
            nms.NMToolStats2.histogram(self.tf, "ST_w0_nan_y")

    # --- type and key validation ---

    def test_histogram_rejects_non_toolfolder(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.histogram("bad", "ST_w0_mean_y")

    def test_histogram_rejects_non_string_name(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.histogram(self.tf, 123)

    def test_histogram_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.histogram(self.tf, "ST_w0_missing")


class TestNMToolStats2Inequality(unittest.TestCase):
    """Tests for NMToolStats2.inequality()."""

    def setUp(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder
        from pyneuromatic.core.nm_dataseries import NMDataSeries

        self.tf = NMToolFolder(name="stats0")
        # ST_w0_mean_y: values 1..5
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=self.arr.copy())
        # ST_w0_data: array names for epoch set creation
        self.array_names = ["RecordA0", "RecordA1", "RecordA2",
                           "RecordA3", "RecordA4"]
        self.tf.data.new(
            "ST_w0_data",
            nparray=np.array(self.array_names, dtype=object),
        )

        # Build a real NMDataSeries with 5 epochs for set-creation tests
        self.nm = NMManager(quiet=True)
        assert self.nm.folders is not None
        self.folder = self.nm.folders.new("f0")
        self.ds = self.folder.dataseries.new("Record")
        for i in range(5):
            self.ds.epochs.new("E%d" % i)

    # --- condition string ---

    def test_condition_str_greater_than(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y > 3")

    def test_condition_str_greater_than_or_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">=", 3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y >= 3")

    def test_condition_str_less_than(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<", 4, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y < 4")

    def test_condition_str_less_than_or_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<=", 4, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y <= 4")

    def test_condition_str_range_exclusive(self):
        # "<<" → a < y < b
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<<", 2, 5, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 < y < 5")

    def test_condition_str_range_inclusive(self):
        # "<=<=" → a <= y <= b
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<=<=", 2, 4, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 <= y <= 4")

    def test_condition_str_range_half_open_left(self):
        # "<=<" → a <= y < b
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<=<", 2, 5, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 <= y < 5")

    def test_condition_str_range_half_open_right(self):
        # "<<=" → a < y <= b
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<<=", 2, 5, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 < y <= 5")

    def test_condition_str_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "==", 3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y == 3")

    def test_condition_str_not_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "!=", 3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y != 3")

    # --- mask and counts ---

    def test_greater_than_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3, save_to_numpy=False
        )
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)
        self.assertEqual(r["failures"], 3)

    def test_less_than_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<", 3, save_to_numpy=False
        )
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)
        self.assertEqual(r["failures"], 3)

    def test_range_mask(self):
        # "<<" → 2 < y < 5 → [3, 4] pass (not 2, not 5)
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "<<", 2, 5, save_to_numpy=False
        )
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)

    def test_equal_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "==", 3, save_to_numpy=False
        )
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 1)

    def test_not_equal_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", "!=", 3, save_to_numpy=False
        )
        self.assertEqual(r["successes"], 4)

    def test_nan_neither_passes_nor_fails(self):
        # A row without a value: NaN in the result, in neither set (#499)
        self.tf.data.new(
            "ST_w0_nan_y",
            nparray=np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        )
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_nan_y", ">", 3, save_to_numpy=False,
            dataseries=self.ds, set_name_success="Pass", set_name_failure="Fail",
        )
        self.assertEqual((r["successes"], r["failures"], r["nans"]), (2, 2, 1))
        np.testing.assert_array_equal(r["result"], [0.0, np.nan, 0.0, 1.0, 1.0])
        names = lambda key: [ep.name for ep in self.ds.epochs.sets.get_items(key)]
        self.assertEqual(names("Pass"), ["E3", "E4"])
        self.assertEqual(names("Fail"), ["E0", "E2"])  # not E1

    def test_command_writes_numpy_numbers_plainly(self):
        from pyneuromatic.tools.nm_tool_stats import NMToolStats
        import pyneuromatic.core.nm_command_history as nmch
        tool = NMToolStats()
        written = []
        original = nmch.add_nm_command
        nmch.add_nm_command = written.append
        try:
            tool.inequality(self.tf, "ST_w0_mean_y", ">", np.float64(3.0))
        finally:
            nmch.add_nm_command = original
        # A numpy number (e.g. from an array) is written plainly: replays without numpy
        self.assertTrue(written and "'>', 3.0" in written[0] and "np." not in written[0])

    def test_again_replaces_the_array_and_the_sets(self):
        nms.NMToolStats2.inequality(self.tf, "ST_w0_mean_y", ">", 3,
                                    dataseries=self.ds, set_name_success="Pass")
        nms.NMToolStats2.inequality(self.tf, "ST_w0_mean_y", ">", 1,
                                    dataseries=self.ds, set_name_success="Pass")
        iq = self.tf.data.get("IQ_ST_w0_mean_y")
        self.assertEqual(list(iq.nparray), [0, 1, 1, 1, 1])
        names = [ep.name for ep in self.ds.epochs.sets.get_items("Pass")]
        self.assertEqual(names, ["E1", "E2", "E3", "E4"])  # replaced, not added to
        self.assertEqual([n["note"] for n in iq.notes], [
            "NMInequality(array=ST_w0_mean_y, condition=y > 1, output=binary, "
            "pass=4, fail=1, nan=0)"])

    # --- result array ---

    def test_binary_output_true(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3, save_to_numpy=False
        )
        np.testing.assert_array_equal(
            r["result"], np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        )

    def test_binary_output_false(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3,
            binary_output=False, save_to_numpy=False
        )
        expected = np.array([np.nan, np.nan, np.nan, 4.0, 5.0])
        np.testing.assert_array_equal(
            np.isnan(r["result"]),
            np.isnan(expected),
        )
        np.testing.assert_array_equal(
            r["result"][~np.isnan(r["result"])],
            expected[~np.isnan(expected)],
        )

    # --- save_to_numpy ---

    def test_save_to_numpy_creates_iq_array(self):
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3, save_to_numpy=True
        )
        self.assertIn("IQ_ST_w0_mean_y", self.tf.data)

    def test_save_to_numpy_false_does_not_create_array(self):
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3, save_to_numpy=False
        )
        self.assertNotIn("IQ_ST_w0_mean_y", self.tf.data)

    # --- epoch sets ---

    def test_set_name_success_creates_epoch_set(self):
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3,
            dataseries=self.ds,
            set_name_success="Successes",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get_items("Successes")
        self.assertIsNotNone(set_epochs)

    def test_set_name_success_contains_correct_epochs(self):
        # ">" 3 → E3, E4 pass
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3,
            dataseries=self.ds,
            set_name_success="Pass",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get_items("Pass")
        epoch_names = [ep.name for ep in set_epochs]
        self.assertIn("E3", epoch_names)
        self.assertIn("E4", epoch_names)
        self.assertNotIn("E0", epoch_names)

    def test_set_name_failure_contains_failing_epochs(self):
        # ">" 3 → E0, E1, E2 fail
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", ">", 3,
            dataseries=self.ds,
            set_name_failure="Fail",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get_items("Fail")
        epoch_names = [ep.name for ep in set_epochs]
        self.assertIn("E0", epoch_names)
        self.assertIn("E1", epoch_names)
        self.assertIn("E2", epoch_names)
        self.assertNotIn("E3", epoch_names)

    # --- type and value validation ---

    def test_bad_toolfolder_raises_typeerror(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.inequality("not_a_folder", "ST_w0_mean_y",
                                        ">", 1)

    def test_bad_name_raises_typeerror(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.inequality(self.tf, 123, ">", 1)

    def test_unknown_name_raises_keyerror(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.inequality(self.tf, "ST_w0_missing", ">", 1)

    def test_unknown_op_raises_valueerror(self):
        with self.assertRaises(ValueError):
            nms.NMToolStats2.inequality(
                self.tf, "ST_w0_mean_y", "??", 1, save_to_numpy=False
            )

    def test_range_op_without_b_raises_valueerror(self):
        with self.assertRaises(ValueError):
            nms.NMToolStats2.inequality(
                self.tf, "ST_w0_mean_y", "<<", 1, save_to_numpy=False
            )


class TestNMToolStats2KSTest(unittest.TestCase):
    """Tests for NMToolStats2.ks_test() — NM wrapper behaviour only.
    Pure math tests are in test_nm_math.py::TestKSTest.
    """

    def setUp(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder

        rng = np.random.default_rng(42)
        self.tf = NMToolFolder(name="stats0")
        self.pop1 = rng.normal(loc=0, scale=1, size=50)
        self.pop2 = rng.normal(loc=10, scale=1, size=50)
        self.tf.data.new("ST_w0_pop1_y", nparray=self.pop1.copy())
        self.tf.data.new("ST_w0_pop2_y", nparray=self.pop2.copy())

    # --- save_to_numpy ---

    def test_ks_test_saves_sort_arrays(self):
        nms.NMToolStats2.ks_test(
            self.tf, "ST_w0_pop1_y", "ST_w0_pop2_y", save_to_numpy=True
        )
        self.assertIn("KS_ST_w0_pop1_y_sort", self.tf.data)
        self.assertIn("KS_ST_w0_pop2_y_sort", self.tf.data)

    def test_ks_test_saves_ecdf_arrays(self):
        nms.NMToolStats2.ks_test(
            self.tf, "ST_w0_pop1_y", "ST_w0_pop2_y", save_to_numpy=True
        )
        self.assertIn("KS_ST_w0_pop1_y_ecdf", self.tf.data)
        self.assertIn("KS_ST_w0_pop2_y_ecdf", self.tf.data)

    def test_ks_test_no_save(self):
        nms.NMToolStats2.ks_test(
            self.tf, "ST_w0_pop1_y", "ST_w0_pop2_y", save_to_numpy=False
        )
        self.assertNotIn("KS_ST_w0_pop1_y_sort", self.tf.data)
        self.assertNotIn("KS_ST_w0_pop1_y_ecdf", self.tf.data)

    # --- type and key validation ---

    def test_ks_test_rejects_bad_toolfolder(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.ks_test("bad", "ST_w0_pop1_y", "ST_w0_pop2_y")

    def test_ks_test_rejects_bad_name1(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.ks_test(self.tf, 123, "ST_w0_pop2_y")

    def test_ks_test_rejects_bad_name2(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.ks_test(self.tf, "ST_w0_pop1_y", 123)

    def test_ks_test_unknown_name1_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.ks_test(self.tf, "ST_w0_missing", "ST_w0_pop2_y")

    def test_ks_test_unknown_name2_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.ks_test(self.tf, "ST_w0_pop1_y", "ST_w0_missing")


class TestNMToolStats2StabilityTest(unittest.TestCase):
    """Tests for NMToolStats2.stability_test() — NM wrapper behaviour only.
    Pure math tests are in test_nm_math.py::TestStabilityTest.
    """

    def setUp(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder

        self.tf = NMToolFolder(name="stats0")
        self.flat = np.full(20, 3.0)
        self.tf.data.new("ST_w0_flat_y", nparray=self.flat.copy())

        array_names = ["RecordA%d" % i for i in range(20)]
        self.tf.data.new(
            "ST_w0_data",
            nparray=np.array(array_names, dtype=object),
        )
        self.nm = NMManager(quiet=True)
        assert self.nm.folders is not None
        self.folder = self.nm.folders.new("f0")
        self.ds = self.folder.dataseries.new("Record")
        for i in range(20):
            self.ds.epochs.new("E%d" % i)

    # --- save_to_numpy ---

    def test_save_creates_mask_array(self):
        nms.NMToolStats2.stability_test(
            self.tf, "ST_w0_flat_y", min_window=5, save_to_numpy=True
        )
        self.assertIn("STAB_ST_w0_flat_y_mask", self.tf.data)

    def test_no_save_no_array(self):
        nms.NMToolStats2.stability_test(
            self.tf, "ST_w0_flat_y", min_window=5, save_to_numpy=False
        )
        self.assertNotIn("STAB_ST_w0_flat_y_mask", self.tf.data)

    # --- epoch sets ---

    def test_set_name_stable_creates_epoch_set(self):
        nms.NMToolStats2.stability_test(
            self.tf, "ST_w0_flat_y", min_window=5,
            dataseries=self.ds, set_name_stable="Stable",
        )
        self.assertIsNotNone(self.ds.epochs.sets.get_items("Stable"))

    def test_set_name_stable_contains_correct_epochs(self):
        nms.NMToolStats2.stability_test(
            self.tf, "ST_w0_flat_y", min_window=5,
            dataseries=self.ds, set_name_stable="Stable",
        )
        epoch_names = [ep.name for ep in self.ds.epochs.sets.get_items("Stable")]
        self.assertIn("E0", epoch_names)
        self.assertIn("E19", epoch_names)

    # --- type and key validation ---

    def test_rejects_bad_toolfolder(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.stability_test("bad", "ST_w0_flat_y")

    def test_rejects_bad_name(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.stability_test(self.tf, 123)

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.stability_test(self.tf, "ST_w0_missing")


class TestNMToolStats2AddEpochSetsFromMask(unittest.TestCase):
    """Direct tests for NMToolStats2._add_epoch_sets_from_mask()."""

    def setUp(self):
        from pyneuromatic.tools.nm_tool_folder import NMToolFolder

        self.tf = NMToolFolder(name="stats0")
        # mask: [T, T, F, F, T] → true indices 0,1,4; false indices 2,3
        self.mask = np.array([True, True, False, False, True])
        array_names = ["RecordA0", "RecordA1", "RecordA2", "RecordA3", "RecordA4"]
        self.tf.data.new("ST_w0_mean_y", nparray=np.zeros(5))
        self.tf.data.new(
            "ST_w0_data",
            nparray=np.array(array_names, dtype=object),
        )
        self.nm = NMManager(quiet=True)
        assert self.nm.folders is not None
        self.folder = self.nm.folders.new("f0")
        self.ds = self.folder.dataseries.new("Record")
        for i in range(5):
            self.ds.epochs.new("E%d" % i)

    def test_true_set_created(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask, set_name_true="Pass"
        )
        self.assertIsNotNone(self.ds.epochs.sets.get_items("Pass"))

    def test_true_set_contains_correct_epochs(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask, set_name_true="Pass"
        )
        epoch_names = [ep.name for ep in self.ds.epochs.sets.get_items("Pass")]
        self.assertIn("E0", epoch_names)
        self.assertIn("E1", epoch_names)
        self.assertIn("E4", epoch_names)
        self.assertNotIn("E2", epoch_names)
        self.assertNotIn("E3", epoch_names)

    def test_false_set_created(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask, set_name_false="Fail"
        )
        self.assertIsNotNone(self.ds.epochs.sets.get_items("Fail"))

    def test_false_set_contains_correct_epochs(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask, set_name_false="Fail"
        )
        epoch_names = [ep.name for ep in self.ds.epochs.sets.get_items("Fail")]
        self.assertIn("E2", epoch_names)
        self.assertIn("E3", epoch_names)
        self.assertNotIn("E0", epoch_names)

    def test_both_sets_created_in_one_call(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask,
            set_name_true="Pass", set_name_false="Fail",
        )
        self.assertIsNotNone(self.ds.epochs.sets.get_items("Pass"))
        self.assertIsNotNone(self.ds.epochs.sets.get_items("Fail"))

    def test_no_true_set_when_name_is_none(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, self.mask, set_name_true=None
        )
        self.assertIsNone(self.ds.epochs.sets.get_items("Pass"))

    def test_all_false_mask_skips_true_set(self):
        all_false = np.zeros(5, dtype=bool)
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "ST_w0_mean_y", self.ds, all_false, set_name_true="Pass"
        )
        self.assertIsNone(self.ds.epochs.sets.get_items("Pass"))

    def test_missing_data_array_does_not_raise(self):
        # No ST_w0_data → should return silently
        tf2 = self.tf.__class__(name="stats1")
        tf2.data.new("ST_w0_mean_y", nparray=np.zeros(5))
        nms.NMToolStats2._add_epoch_sets_from_mask(
            tf2, "ST_w0_mean_y", self.ds, self.mask, set_name_true="Pass"
        )  # no exception

    def test_name_with_too_few_parts_does_not_raise(self):
        nms.NMToolStats2._add_epoch_sets_from_mask(
            self.tf, "short", self.ds, self.mask, set_name_true="Pass"
        )  # no exception


class TestNMToolStatsConfig(unittest.TestCase):
    """NMToolStatsConfig schema and defaults."""

    def setUp(self):
        self.cfg = NMToolStatsConfig()

    def test_ignore_nans_default(self):
        self.assertTrue(self.cfg.ignore_nans)

    def test_results_to_history_default(self):
        self.assertFalse(self.cfg.results_to_history)

    def test_results_to_cache_default(self):
        self.assertFalse(self.cfg.results_to_cache)  # #480

    def test_results_to_numpy_default(self):
        self.assertTrue(self.cfg.results_to_numpy)

    def test_set_ignore_nans(self):
        self.cfg.ignore_nans = False
        self.assertFalse(self.cfg.ignore_nans)

    def test_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            self.cfg.ignore_nans = 1  # int, not bool

    def test_toml_type(self):
        self.assertEqual(NMToolStatsConfig._TOML_TYPE, "stats_config")

    def test_save_load_round_trip(self):
        self.cfg.ignore_nans = False
        self.cfg.results_to_numpy = True
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stats.toml"
            self.cfg.save(path)
            cfg2 = NMToolStatsConfig.load(path)
        self.assertFalse(cfg2.ignore_nans)
        self.assertTrue(cfg2.results_to_numpy)

    def test_stats_tool_has_config(self):
        t = nms.NMToolStats()
        self.assertIsNotNone(t.config)

    def test_stats_tool_config_is_stats_config(self):
        t = nms.NMToolStats()
        self.assertIsInstance(t.config, NMToolStatsConfig)

    def test_stats_tool_config_defaults(self):
        t = nms.NMToolStats()
        self.assertTrue(t.config.ignore_nans)

    def test_overwrite_default(self):
        self.assertFalse(self.cfg.overwrite)

    def test_overwrite_set_true(self):
        self.cfg.overwrite = True
        self.assertTrue(self.cfg.overwrite)

    def test_overwrite_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.cfg.overwrite = 1

    def test_to_dict_contains_all_keys(self):
        d = self.cfg.to_dict()
        for key in ("ignore_nans", "overwrite",
                    "results_to_history", "results_to_cache", "results_to_numpy"):
            self.assertIn(key, d)

    def test_to_dict_does_not_contain_xclip(self):
        self.assertNotIn("xclip", self.cfg.to_dict())


class TestNMToolStatsOverwrite(unittest.TestCase):
    """overwrite flag controls subfolder reuse vs. new numbered subfolders."""

    def _setup(self, overwrite=False):
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.tools.nm_tool import HIERARCHY_SELECT_KEYS
        tool = nms.NMToolStats()
        tool.overwrite = overwrite
        folder = NMFolder(name="F")
        tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        tool._select["folder"] = folder
        return tool, folder

    def _fill_results(self, tool, func="mean", n=3):
        w = list(tool.windows)[0]
        w.func = func
        for k in range(n):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in tool._NMToolStats__results:
                tool._NMToolStats__results[wname].append(results)
            else:
                tool._NMToolStats__results[wname] = [results]

    def test_overwrite_false_default(self):
        self.assertFalse(nms.NMToolStats().overwrite)

    def test_overwrite_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats().overwrite = 1

    def test_overwrite_false_creates_stats_0(self):
        tool, folder = self._setup(overwrite=False)
        self._fill_results(tool)
        tool._write_results_to_numpy()
        self.assertIn("stats_0", folder.toolfolders)

    def test_overwrite_false_creates_stats_1_on_second_run(self):
        tool, folder = self._setup(overwrite=False)
        self._fill_results(tool)
        tool._write_results_to_numpy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool)
        tool._write_results_to_numpy()
        self.assertIn("stats_0", folder.toolfolders)
        self.assertIn("stats_1", folder.toolfolders)

    def test_overwrite_true_creates_stats_0(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool)
        tool._write_results_to_numpy()
        self.assertIn("stats_0", folder.toolfolders)

    def test_overwrite_true_reuses_stats_0_on_second_run(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool)
        tool._write_results_to_numpy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool)
        tool._write_results_to_numpy()
        self.assertIn("stats_0", folder.toolfolders)
        self.assertNotIn("stats_1", folder.toolfolders)

    def test_overwrite_true_replaces_st_arrays(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool, n=3)
        f = tool._write_results_to_numpy()
        arr_first = f.data.get("ST_w0_mean_y").nparray.copy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool, n=5)
        f2 = tool._write_results_to_numpy()
        self.assertIs(f, f2)
        arr_second = f2.data.get("ST_w0_mean_y").nparray
        self.assertEqual(len(arr_second), 5)
        self.assertEqual(len(arr_first), 3)


class TestNMToolStatsNotes(unittest.TestCase):
    """Notes are attached to numeric ST_ arrays in _write_results_to_numpy()."""

    def setUp(self):
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.tools.nm_tool import HIERARCHY_SELECT_KEYS
        self.tool = nms.NMToolStats()
        folder = NMFolder(name="F")
        self.tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        self.tool._select["folder"] = folder
        w = list(self.tool.windows)[0]
        w.func = "mean"
        w.xbgn = 10.0
        w.xend = 50.0
        self.n_arrays = 4
        for k in range(self.n_arrays):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in self.tool._NMToolStats__results:
                self.tool._NMToolStats__results[wname].append(results)
            else:
                self.tool._NMToolStats__results[wname] = [results]
        self.f = self.tool._write_results_to_numpy()
        self.st_array = self.f.data.get("ST_w0_mean_y")

    def _note_text(self):
        notes = getattr(self.st_array, "notes", None)
        if notes is None:
            return ""
        return " ".join(str(n) for n in notes)

    def test_st_array_has_note(self):
        notes = getattr(self.st_array, "notes", None)
        self.assertIsNotNone(notes)
        self.assertGreater(len(list(notes)), 0)

    def test_note_contains_win(self):
        self.assertIn("win=w0", self._note_text())

    def test_note_contains_func(self):
        self.assertIn("func=mean", self._note_text())

    def test_note_contains_id(self):
        self.assertIn("id=main", self._note_text())

    def test_note_contains_xbgn(self):
        self.assertIn("xbgn=10.0", self._note_text())

    def test_note_contains_xend(self):
        self.assertIn("xend=50.0", self._note_text())

    def test_note_contains_n(self):
        self.assertIn("n=%d" % self.n_arrays, self._note_text())

    def test_st_data_array_has_no_note(self):
        st_data = self.f.data.get("ST_data_names")
        if st_data is None:
            self.skipTest("ST_data_names not created for this func")
        notes = getattr(st_data, "notes", None)
        if notes is not None:
            self.assertEqual(len(list(notes)), 0)

    def test_bsln_array_note_uses_bsln_x_range(self):
        # Set up a run with baseline enabled; ST_w0_bsln_y note should
        # record the baseline x-range (bsln_xbgn/bsln_xend), not the main xbgn/xend.
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.tools.nm_tool import HIERARCHY_SELECT_KEYS
        tool = nms.NMToolStats()
        folder = NMFolder(name="F")
        tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        tool._select["folder"] = folder
        w = list(tool.windows)[0]
        w.func = "mean"
        w.xbgn = 10.0
        w.xend = 50.0
        w._bsln_on_set(True, quiet=True)
        w._bsln_func_set({"name": "mean"}, quiet=True)
        w._x_set("bsln_xbgn", 0.0, quiet=True)
        w._x_set("bsln_xend", 20.0, quiet=True)
        for k in range(3):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in tool._NMToolStats__results:
                tool._NMToolStats__results[wname].append(results)
            else:
                tool._NMToolStats__results[wname] = [results]
        f = tool._write_results_to_numpy()
        bsln_arr = f.data.get("ST_w0_bsln_y")
        self.assertIsNotNone(bsln_arr)
        notes = getattr(bsln_arr, "notes", None)
        self.assertIsNotNone(notes)
        note_text = " ".join(str(n) for n in notes)
        self.assertIn("id=bsln", note_text)
        self.assertIn("xbgn=0.0", note_text)
        self.assertIn("xend=20.0", note_text)
        # Main xbgn/xend should NOT appear in the bsln note
        self.assertNotIn("xbgn=10.0", note_text)


class TestNMToolStatsCommandHistory(unittest.TestCase):
    """Tests for command history logging of NMToolStats, NMStatWin, NMStatWinContainer."""

    def setUp(self):
        from pyneuromatic.core.nm_command_history import (
            NMCommandHistory, set_command_history,
        )
        self._ch = NMCommandHistory(enabled=True, quiet=True, log_to_nm_history=False)
        set_command_history(self._ch)
        self.tool = nms.NMToolStats()
        self.w0 = self.tool.windows['w0']

    # ------------------------------------------------------------------
    # NMStatWin.func setter

    def _func_command(self, func):
        self._ch.clear()
        self.w0.func = func
        self.assertEqual(len(self._ch.buffer), 1)
        return self._ch.buffer[0]['command']

    def _compute_notes(self, **kwargs):
        from pyneuromatic.core.nm_manager import NMManager
        nm = NMManager(quiet=True)
        data = nm.folders.new("F").data.new(
            "RecordA0", nparray=np.arange(20.0), xscale={"start": 0.0, "delta": 1.0})
        count = len(nm.history.buffer)
        self.w0.compute(data, **kwargs)
        return [e["message"] for e in nm.history.buffer[count:] if "compute" in e["message"]]

    def test_compute_note(self):
        self.w0.func = "max"
        self.w0.xbgn, self.w0.xend = 1.5, 15.0
        self.assertEqual(self._compute_notes(), ["compute func=max, xbgn=1.5, xend=15.0"])
        self.w0.bsln_on = True
        self.w0.bsln_func = "mean"
        self.w0.bsln_xbgn, self.w0.bsln_xend = 0.5, 2.5
        self.assertEqual(self._compute_notes(), [
            "compute func=max, xbgn=1.5, xend=15.0, bsln=mean 0.5 to 2.5"])
        self.assertEqual(self._compute_notes(note=False), [])

    def test_delete_last_window_keeps_names_in_sequence(self):
        windows = self.tool.windows
        windows.new()
        windows.new(select=True)  # w2
        self._ch.clear()
        self.assertEqual(windows.delete_last(), "w2")
        self.assertEqual([e["command"] for e in self._ch.buffer],
                         ["nm.stats.windows.delete_last()"])
        self.assertEqual([w.name for w in windows], ["w0", "w1"])
        self.assertEqual(windows.selected_name, "w1")  # w2 was selected
        windows.new()
        self.assertEqual([w.name for w in windows], ["w0", "w1", "w2"])  # w2 again

    def test_the_first_window_is_not_deleted(self):
        windows = self.tool.windows
        self.assertEqual([w.name for w in windows], ["w0"])
        self._ch.clear()
        with self.assertRaises(RuntimeError):
            windows.delete_last()
        self.assertEqual([w.name for w in windows], ["w0"])
        self.assertEqual(self._ch.buffer, [])

    def test_delete_last_keeps_another_selection(self):
        windows = self.tool.windows
        windows.new()  # w1, w0 still selected
        windows.delete_last()
        self.assertEqual(windows.selected_name, "w0")

    def test_reset_returns_to_defaults_as_one_command(self):
        w0 = self.w0
        w0.func = "mean"
        w0.xbgn, w0.xend = 2.0, 8.0
        w0.on = False
        w0.bsln_on = True
        w0.bsln_xbgn, w0.bsln_xend = 0.0, 1.0
        self._ch.clear()
        w0.reset()
        self.assertEqual([e["command"] for e in self._ch.buffer],
                         ["nm.stats.windows['w0'].reset()"])
        from pyneuromatic.tools.nm_stat_win import NMStatWin
        self.assertEqual(w0.to_dict(), NMStatWin("w0").to_dict())
        self.assertEqual(w0.name, "w0")
        self.assertTrue(w0.on and w0.func == {} and not w0.bsln_on)
        self.assertEqual((w0.xbgn, w0.xend), (-math.inf, math.inf))

    def test_func_setter_logs_a_name_alone(self):
        """As the setter takes it, so a script runs it (#448)."""
        for name in ('mean', 'max'):
            self.assertEqual(self._func_command(name),
                             "nm.stats.windows['w0'].func = %r" % name)

    def test_func_setter_logs_a_dict_with_parameters(self):
        for func in (
            {'name': 'mean@max', 'n_mean': 3},
            {'name': 'level', 'ylevel': 10.0},
            {'name': 'level', 'n_std': 2.0},
            {'name': 'risetime+', 'p0': 10.0, 'p1': 90.0},
            {'name': 'decaytime+', 'p0': 50.0},
            {'name': 'fwhm+', 'p0': 30.0, 'p1': 70.0},
        ):
            self.assertEqual(self._func_command(dict(func)),
                             "nm.stats.windows['w0'].func = %r" % func)

    def test_func_setter_logs_default_parameters(self):
        self.assertIn("{'name': 'decaytime+', 'p0': ", self._func_command('decaytime+'))
        self.assertEqual(self._func_command('fwhm+'),
                         "nm.stats.windows['w0'].func = "
                         "{'name': 'fwhm+', 'p0': 50.0, 'p1': 50.0}")

    def test_func_setter_logs_none(self):
        """Clearing is recorded too, so a script matches the session."""
        self.assertEqual(self._func_command(None), "nm.stats.windows['w0'].func = None")

    def test_func_command_replays(self):
        self.w0.func = {'name': 'risetime+', 'p0': 20.0, 'p1': 80.0}
        command = self._ch.buffer[-1]['command']
        other = nms.NMToolStats()
        exec(command, {'nm': type('NM', (), {'stats': other})})
        self.assertEqual(other.windows['w0'].func, self.w0.func)

    # ------------------------------------------------------------------
    # NMStatWin scalar setters

    def test_xbgn_setter_logs(self):
        self._ch.clear()
        self.w0.xbgn = 100.0
        self.assertEqual(len(self._ch.buffer), 1)
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].xbgn", cmd)
        self.assertIn("100.0", cmd)

    def test_xend_setter_logs(self):
        self._ch.clear()
        self.w0.xend = 200.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].xend", cmd)
        self.assertIn("200.0", cmd)

    def test_on_setter_logs_false(self):
        self._ch.clear()
        self.w0.on = False
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].on", cmd)
        self.assertIn("False", cmd)

    def test_bsln_on_setter_logs_true(self):
        self._ch.clear()
        self.w0.bsln_on = True
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_on", cmd)
        self.assertIn("True", cmd)

    def test_bsln_xbgn_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_xbgn = -10.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_xbgn", cmd)
        self.assertIn("-10.0", cmd)

    def test_bsln_xend_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_xend = 0.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_xend", cmd)
        self.assertIn("0.0", cmd)

    def test_bsln_func_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_func = 'mean'
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_func", cmd)
        self.assertIn("'mean'", cmd)

    def test_bsln_func_setter_logs_none(self):
        self._ch.clear()
        self.w0.bsln_func = None
        self.assertEqual([e['command'] for e in self._ch.buffer],
                         ["nm.stats.windows['w0'].bsln_func = None"])

    # ------------------------------------------------------------------
    # NMStatWinContainer.new()

    def test_windows_new_logs(self):
        self._ch.clear()
        self.tool.windows.new()
        self.assertEqual(len(self._ch.buffer), 1)
        cmd = self._ch.buffer[0]['command']
        # No argument: new() names the window itself (#448)
        self.assertEqual(cmd, "nm.stats.windows.new()")
        self.assertEqual(self.tool.windows.selected_name, 'w0')

    def test_windows_new_select_logs(self):
        self._ch.clear()
        self.tool.windows.new(select=True)
        self.assertEqual([e['command'] for e in self._ch.buffer],
                         ["nm.stats.windows.new(select=True)"])
        self.assertEqual(self.tool.windows.selected_name, 'w1')

    def test_windows_selected_name_logs_changes_only(self):
        self.tool.windows.new()
        self._ch.clear()
        self.tool.windows.selected_name = 'w0'  # unchanged
        self.tool.windows.selected_name = 'w1'
        self.assertEqual([e['command'] for e in self._ch.buffer],
                         ["nm.stats.windows.selected_name = 'w1'"])
        with self.assertRaises(KeyError):
            self.tool.windows.selected_name = 'w9'

    # ------------------------------------------------------------------
    # NMToolStats tool-level flag setters

    def test_ignore_nans_setter_logs(self):
        self._ch.clear()
        self.tool.ignore_nans = True
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.ignore_nans", cmd)
        self.assertIn("True", cmd)

    def test_results_to_history_setter_logs(self):
        self._ch.clear()
        self.tool.results_to_history = True
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.results_to_history", cmd)
        self.assertIn("True", cmd)

    def test_results_to_cache_setter_logs_false(self):
        self._ch.clear()
        self.tool.results_to_cache = False
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.results_to_cache", cmd)
        self.assertIn("False", cmd)

    def test_results_to_numpy_setter_logs(self):
        self._ch.clear()
        self.tool.results_to_numpy = True
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.results_to_numpy", cmd)
        self.assertIn("True", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestNMToolStatsAllEpochs(unittest.TestCase):
    """results_all_epochs: ST_ arrays with one point per epoch of the series."""

    def setUp(self):
        from pyneuromatic.core.nm_manager import NMManager

        self.nm = NMManager(quiet=True)
        self.folder = self.nm.folders.new("Demo")
        self.series = self.folder.new_dataseries("Record", n_epochs=10, n_points=4)
        for i, epoch in enumerate(self.series.epochs.values()):
            epoch.data[0].nparray = np.full(4, float(i * 10))  # mean = 10 * epoch
        self.tool = nms.NMToolStats()
        list(self.tool.windows)[0].func = "mean"
        self.tool.results_to_cache = False

    def _run(self, epochs):
        channel = self.series.channels["A"]
        targets = [
            {"folder": self.folder, "dataseries": self.series, "channel": channel,
             "epoch": self.series.epochs[name], "data": self.series.epochs[name].data[0]}
            for name in epochs
        ]
        self.tool.run_all(targets)
        return self.tool.toolfolder

    def test_default_off(self):
        self.assertFalse(self.tool.results_all_epochs)
        self.assertFalse(nms.NMToolStatsConfig().results_all_epochs)

    def test_off_one_point_per_analysed_epoch(self):
        f = self._run(["E1", "E3", "E7"])
        self.assertEqual(list(f.data["ST_w0_mean_y"].nparray), [10.0, 30.0, 70.0])

    def test_on_one_point_per_epoch(self):
        self.tool.results_all_epochs = True
        f = self._run(["E1", "E3", "E7"])
        values = f.data["ST_w0_mean_y"].nparray
        self.assertEqual(len(values), 10)
        self.assertEqual([values[i] for i in (1, 3, 7)], [10.0, 30.0, 70.0])
        self.assertTrue(all(math.isnan(values[i]) for i in (0, 2, 4, 5, 6, 8, 9)))
        # Every epoch's array is named, analysed or not
        paths = list(f.data["ST_data_names"].nparray)
        self.assertEqual([p.rsplit(".", 1)[-1] for p in paths],
                         [f"RecordA{i}" for i in range(10)])
        self.assertIn("per epoch", f.data["ST_w0_mean_y"].notes.note)

    def test_on_with_every_epoch_is_unchanged(self):
        self.tool.results_all_epochs = True
        f = self._run(list(self.series.epochs.keys()))
        self.assertEqual(list(f.data["ST_w0_mean_y"].nparray), [10.0 * i for i in range(10)])

    def test_on_without_a_data_series_is_unchanged(self):
        self.tool.results_all_epochs = True
        data = [self.folder.data.new(f"W{i}", nparray=np.full(3, float(i))) for i in range(3)]
        self.tool.run_all([{"folder": self.folder, "data": d} for d in data])
        values = self.tool.toolfolder.data["ST_w0_mean_y"].nparray
        self.assertEqual(list(values), [0.0, 1.0, 2.0])

    def test_type_checked(self):
        with self.assertRaises(TypeError):
            self.tool.results_all_epochs = 1

    def test_config_round_trip(self):
        config = nms.NMToolStatsConfig()
        config.results_all_epochs = True
        self.assertTrue(config.to_dict()["results_all_epochs"])


    def test_stats2_ignores_unanalysed_epochs(self):
        # Stats2 summaries of per-epoch ST_ arrays skip the NaN epochs
        self.tool.results_all_epochs = True
        f = self._run(["E1", "E3", "E7"])
        summary = nms.NMToolStats2.stats(f, select="ST_w0_mean_y")["ST_w0_mean_y"]
        self.assertEqual(summary["N"], 3)
        self.assertAlmostEqual(summary["mean"], (10 + 30 + 70) / 3)


    def test_off_names_only_the_analysed_arrays(self):
        f = self._run(["E1", "E3"])
        paths = list(f.data["ST_data_names"].nparray)
        self.assertEqual([p.rsplit(".", 1)[-1] for p in paths], ["RecordA1", "RecordA3"])

    def test_names_follow_the_channel_run_on(self):
        series = self.folder.new_dataseries("Two", n_channels=2, n_epochs=3, n_points=2)
        channel = series.channels["B"]
        self.tool.results_all_epochs = True
        self.tool.run_all([
            {"folder": self.folder, "dataseries": series, "channel": channel,
             "epoch": series.epochs["E1"], "data": series.epochs["E1"].data[1]},
        ])
        paths = list(self.tool.toolfolder.data["ST_data_names"].nparray)
        self.assertEqual([p.rsplit(".", 1)[-1] for p in paths], ["TwoB0", "TwoB1", "TwoB2"])


class TestNMToolRunEach(unittest.TestCase):
    """run_each: one run, and one subfolder, per epoch set or group."""

    def setUp(self):
        from pyneuromatic.core.nm_manager import NMManager

        self.nm = NMManager(quiet=True)
        self.folder = self.nm.folders.new("Demo")
        self.series = self.folder.new_dataseries("Record", n_epochs=6, n_points=3)
        for i, epoch in enumerate(self.series.epochs.values()):
            epoch.data[0].nparray = np.full(3, float(i))  # mean = epoch number
        sets = self.series.epochs.sets
        sets.add("Set0", [self.series.epochs["E0"], self.series.epochs["E1"]])
        sets.add("Set1", [self.series.epochs[e] for e in ("E2", "E3", "E4")])
        sets.add("Set2")  # empty
        self.series.epochs.groups.assign_cyclic(list(self.series.epochs.keys()), n_groups=2)
        self.tool = nms.NMToolStats()
        list(self.tool.windows)[0].func = "mean"
        self.tool.results_to_cache = False

    def _targets(self, names):
        channel = self.series.channels["A"]
        return [{"folder": self.folder, "dataseries": self.series, "channel": channel,
                 "epoch": self.series.epochs[n], "data": self.series.epochs[n].data[0]}
                for n in names]

    def _means(self, folder):
        return list(folder.data["ST_w0_mean_y"].nparray)

    def test_one_run_per_set(self):
        sets = self.series.epochs.sets
        batches = [(self._targets(sets.get_items(name, get_keys=True) or []), {"epoch": name})
                   for name in ("Set0", "Set1", "Set2")]
        done = self.tool.run_each(batches)
        self.assertEqual([keys["epoch"] for keys, _ in done], ["Set0", "Set1"])  # Set2 empty
        names = [folder.name for _, folder in done]
        self.assertEqual(names, ["Stats_Record_A_S0_0", "Stats_Record_A_S1_0"])
        self.assertEqual(self._means(done[0][1]), [0.0, 1.0])
        self.assertEqual(self._means(done[1][1]), [2.0, 3.0, 4.0])
        self.assertIs(self.tool.toolfolder, done[-1][1])  # the last run's

    def test_one_run_per_group(self):
        groups = self.series.epochs.groups
        batches = [(self._targets(groups.get_items(n)), {"epoch": None, "group": n})
                   for n in groups.group_numbers]
        done = self.tool.run_each(batches)
        self.assertEqual([f.name for _, f in done],
                         ["Stats_Record_A_G0_0", "Stats_Record_A_G1_0"])
        self.assertEqual(self._means(done[0][1]), [0.0, 2.0, 4.0])
        self.assertEqual(self._means(done[1][1]), [1.0, 3.0, 5.0])

    def test_per_epoch_arrays_in_each_run(self):
        self.tool.results_all_epochs = True
        sets = self.series.epochs.sets
        done = self.tool.run_each([
            (self._targets(sets.get_items("Set0", get_keys=True)), {"epoch": "Set0"}),
        ])
        values = done[0][1].data["ST_w0_mean_y"].nparray
        self.assertEqual(len(values), 6)  # one point per epoch
        self.assertEqual(list(values[:2]), [0.0, 1.0])
        self.assertTrue(all(math.isnan(v) for v in values[2:]))

    def test_nothing_to_run(self):
        self.assertEqual(self.tool.run_each([([], {"epoch": "Set2"})]), [])


class TestRunSelection(unittest.TestCase):
    """NMTool.run_selection: a run as one command, which a script repeats (#448)."""

    def setUp(self):
        self.nm = NMManager(quiet=True, command_history=True)
        self._ch = self.nm.command_history
        self.folder = self.nm.folders.new("Demo")
        self.series = self.folder.new_dataseries("Record", n_epochs=4, n_points=3)
        for i, data in enumerate(self.folder.data.values()):
            data.nparray[:] = float(i)
        self.series.epochs.sets.add("Set1", ["E1", "E2"])
        self.series.epochs.groups.assign_cyclic(list(self.series.epochs.keys()), n_groups=2)
        self.nm.tool_add("stats")
        self.tool = self.nm.stats
        self.tool.windows["w0"].func = "mean"
        self._ch.clear()

    def _commands(self):
        return [e["command"] for e in self._ch.buffer]

    def _means(self):
        return [r[0]["s"] for r in self.tool.results["w0"]]

    def test_series_epoch_set(self):
        self.assertTrue(self.tool.run_selection(self.series, channel="A", epochs="Set1"))
        self.assertEqual(self._means(), [1.0, 2.0])
        self.assertEqual(self.tool.toolfolder.name, "Stats_Record_A_S1_0")
        self.assertEqual(self._commands(), [
            "nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'], channel='A', "
            "epochs='Set1')"])

    def test_series_set_and_group(self):
        self.tool.run_selection(self.series, channel="A", epochs="Set1", group=1,
                         group_operator="AND")
        self.assertEqual(self._means(), [1.0])
        self.assertEqual(self.tool.toolfolder.name, "Stats_Record_A_S1andG1_0")
        self.assertIn("group=1, group_operator='AND'", self._commands()[0])

    def test_series_one_epoch_group_and_list(self):
        self.tool.run_selection(self.series, epochs="E3")  # the selected channel
        self.assertEqual(self._means(), [3.0])
        self.assertEqual(self.tool.toolfolder.name, "Stats_Record_A_E3_0")
        self.tool.run_selection(self.series, group=0)
        self.assertEqual(self._means(), [0.0, 2.0])
        self.tool.run_selection(self.series, epochs=["E0", "E1", "E2"])
        self.assertEqual(self._commands()[-1],
                         "nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'], "
                         "epochs=['E0', 'E1', 'E2'])")

    def test_folder_data(self):
        self.folder.data.sets.add("DS", ["RecordA0", "RecordA3"])
        self.tool.run_selection(self.folder, data="DS")
        self.assertEqual(self._means(), [0.0, 3.0])
        self.assertEqual(self.tool.toolfolder.name, "Stats_DS_0")
        self.tool.run_selection(self.folder, data="RecordA2")
        self.assertEqual(self._means(), [2.0])
        self.assertEqual(self._commands()[-1],
                         "nm.stats.run_selection(nm.folders['Demo'], data='RecordA2')")

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            self.tool.run_selection(self.folder)  # no data
        with self.assertRaises(ValueError):
            self.tool.run_selection(self.folder, data="RecordA0", epochs="all")
        with self.assertRaises(KeyError):
            self.tool.run_selection(self.series, channel="Z", epochs="all")
        self.series.epochs.sets.add("Empty", [])
        self._ch.clear()
        with self.assertRaises(RuntimeError):
            self.tool.run_selection(self.series, epochs="Empty")  # nothing to analyze
        self.assertEqual(self._commands(), [])

    def test_compute_notes_are_inner_calls(self):
        """A run's own notes (each compute, its subfolder) are inner calls
        (DEBUG, not kept by default): the command is the run's one entry."""
        before = len(self.nm.history.buffer)
        self.tool.run_selection(self.series, epochs="all")
        self.assertEqual([e["kind"] for e in self.nm.history.buffer[before:]], ["command"])

    def test_command_replays(self):
        self.tool.run_selection(self.series, channel="A", epochs="Set1", group=1, group_operator="OR")
        expected = self._means()
        self.tool.run_init()
        exec(self._commands()[-1], {"nm": self.nm})
        self.assertEqual(self._means(), expected)
        self.assertEqual(self.tool.toolfolder.name, "Stats_Record_A_S1orG1_1")
