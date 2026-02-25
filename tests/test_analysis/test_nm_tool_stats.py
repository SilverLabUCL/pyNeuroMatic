#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_stats: NMToolStats and NMToolStats2 tool classes.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import unittest

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_tool_stats as nms
import pyneuromatic.analysis.nm_stat_win as nmsw

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
    """Tests for NMToolStats results_to_* flags and _results_to_history()."""

    def setUp(self):
        self.tool = nms.NMToolStats()

    # --- defaults ---

    def test_results_to_history_default(self):
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_cache_default(self):
        self.assertTrue(self.tool.results_to_cache)

    def test_results_to_numpy_default(self):
        self.assertFalse(self.tool.results_to_numpy)

    # --- results_to_history ---

    def test_results_to_history_set_true(self):
        self.tool.results_to_history = True
        self.assertTrue(self.tool.results_to_history)

    def test_results_to_history_set_false(self):
        self.tool.results_to_history = True
        self.tool.results_to_history = False
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_history_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_history = 1

    def test_results_to_history_rejects_none(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_history = None

    # --- results_to_cache ---

    def test_results_to_cache_set_false(self):
        self.tool.results_to_cache = False
        self.assertFalse(self.tool.results_to_cache)

    def test_results_to_cache_set_true(self):
        self.tool.results_to_cache = False
        self.tool.results_to_cache = True
        self.assertTrue(self.tool.results_to_cache)

    def test_results_to_cache_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_cache = "yes"

    # --- results_to_numpy ---

    def test_results_to_numpy_set_true(self):
        self.tool.results_to_numpy = True
        self.assertTrue(self.tool.results_to_numpy)

    def test_results_to_numpy_set_false(self):
        self.tool.results_to_numpy = True
        self.tool.results_to_numpy = False
        self.assertFalse(self.tool.results_to_numpy)

    def test_results_to_numpy_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_numpy = 0

    # --- _results_to_history ---

    def test_results_to_history_empty_results(self):
        # Should not raise with no results
        self.tool._results_to_history(quiet=True)

    def test_results_to_history_with_results(self):
        # Populate results via compute then call print
        data = _make_data()
        w = list(self.tool.windows)[0]
        w.func = "mean"
        w.compute(data)
        self.tool._results_to_history(quiet=True)

    # --- _results_to_numpy ---

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

    def test_results_to_numpy_returns_toolfolder(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self._setup_folder()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertIsInstance(f, NMToolFolder)

    def test_results_to_numpy_folder_named_stats0(self):
        self._setup_folder()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats0")

    def test_results_to_numpy_second_run_named_stats1(self):
        self._setup_folder()
        self._run_compute()
        self.tool._results_to_numpy()
        self.tool._NMToolStats__results.clear()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats1")

    def test_results_to_numpy_creates_data_array(self):
        self._setup_folder()
        self._run_compute(n_waves=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_data", f.data)

    def test_results_to_numpy_data_array_length(self):
        self._setup_folder()
        self._run_compute(n_waves=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_data")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_creates_s_array(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_main_s", f.data)

    def test_results_to_numpy_s_array_length(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_main_s")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_no_folder_returns_none(self):
        from pyneuromatic.analysis.nm_tool import SELECT_LEVELS
        self.tool._select = {level: None for level in SELECT_LEVELS}
        # folder is None — should return None
        result = self.tool._results_to_numpy()
        self.assertIsNone(result)

    def test_results_to_numpy_no_results_raises(self):
        self._setup_folder()
        with self.assertRaises(RuntimeError):
            self.tool._results_to_numpy()


class TestNMToolStats2(unittest.TestCase):
    """Tests for NMToolStats2."""

    def setUp(self):
        # Build a NMToolFolder with some ST_ arrays
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_main_s", nparray=arr.copy())
        self.tf.data.new("ST_w0_bsln_s", nparray=arr.copy() * 0.1)
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
        self.assertIn("ST_w0_main_s", r)
        self.assertIn("ST_w0_bsln_s", r)

    def test_stats_all_excludes_data_array(self):
        r = nms.NMToolStats2.stats(self.tf, select="all")
        self.assertNotIn("ST_w0_data", r)

    def test_stats_single_array(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_main_s")
        self.assertIn("ST_w0_main_s", r)
        self.assertEqual(len(r), 1)

    def test_stats_mean_correct(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_main_s")
        self.assertAlmostEqual(r["ST_w0_main_s"]["mean"], 3.0)

    def test_stats_N_correct(self):
        r = nms.NMToolStats2.stats(self.tf, select="ST_w0_main_s")
        self.assertEqual(r["ST_w0_main_s"]["N"], 5)

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
        nms.NMToolStats2.stats(self.tf, select="ST_w0_main_s", results_to_numpy=True)
        d = self.tf.data.get("ST2_mean")
        self.assertAlmostEqual(d.nparray[0], 3.0)


class TestNMToolStats2Histogram(unittest.TestCase):
    """Tests for NMToolStats2.histogram()."""

    def setUp(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_main_s", nparray=arr)

    # --- return value ---

    def test_histogram_returns_dict(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s")
        self.assertIsInstance(r, dict)
        self.assertIn("counts", r)
        self.assertIn("edges", r)

    def test_histogram_counts_length(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        self.assertEqual(len(r["counts"]), 4)

    def test_histogram_edges_length(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        self.assertEqual(len(r["edges"]), 5)  # bins + 1

    def test_histogram_counts_sum(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        self.assertEqual(sum(r["counts"]), 9)  # all 9 values counted

    # --- saved arrays (save_to_numpy=True, default) ---

    def test_histogram_saves_counts_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        self.assertIn("HIST_ST_w0_main_s_counts", self.tf.data)

    def test_histogram_saves_edges_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        self.assertIn("HIST_ST_w0_main_s_edges", self.tf.data)

    def test_histogram_xscale_start(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        d = self.tf.data.get("HIST_ST_w0_main_s_counts")
        self.assertAlmostEqual(d.xscale.start, 1.0)

    def test_histogram_xscale_delta(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4)
        d = self.tf.data.get("HIST_ST_w0_main_s_counts")
        self.assertAlmostEqual(d.xscale.delta, 1.0)  # (5 - 1) / 4 = 1.0

    # --- save_to_numpy=False ---

    def test_histogram_no_save_skips_arrays(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4, save_to_numpy=False)
        self.assertNotIn("HIST_ST_w0_main_s_counts", self.tf.data)
        self.assertNotIn("HIST_ST_w0_main_s_edges", self.tf.data)

    def test_histogram_no_save_still_returns_dict(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_main_s", bins=4,
                                 save_to_numpy=False)
        self.assertIn("counts", r)
        self.assertIn("edges", r)

    # --- NaN/Inf handling ---

    def test_histogram_strips_nans(self):
        arr_nan = np.array([1.0, 2.0, np.nan, 3.0])
        self.tf.data.new("ST_w0_nan_s", nparray=arr_nan)
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_nan_s", bins=3)
        self.assertEqual(sum(r["counts"]), 3)  # NaN excluded

    def test_histogram_strips_infs(self):
        arr_inf = np.array([1.0, 2.0, np.inf, 3.0])
        self.tf.data.new("ST_w0_inf_s", nparray=arr_inf)
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_inf_s", bins=3)
        self.assertEqual(sum(r["counts"]), 3)  # Inf excluded

    # --- type validation ---

    def test_histogram_rejects_non_toolfolder(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.histogram("bad", "ST_w0_main_s")

    def test_histogram_rejects_non_string_name(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.histogram(self.tf, 123)

    def test_histogram_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.histogram(self.tf, "ST_w0_missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
