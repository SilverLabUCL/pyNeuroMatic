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
import pyneuromatic.analysis.nm_tool_stats as nms
import pyneuromatic.analysis.nm_stat_win as nmsw
from pyneuromatic.analysis.nm_tool_stats import NMToolStatsConfig

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
        from pyneuromatic.analysis.nm_tool import HIERARCHY_SELECT_KEYS
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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self._setup_folder()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertIsInstance(f, NMToolFolder)

    def test_results_to_numpy_folder_named_stats_0_no_dataseries(self):
        # No dataseries selected → fallback name "stats_0"
        self._setup_folder()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats_0")

    def test_results_to_numpy_second_run_named_stats_1_no_dataseries(self):
        # Second run with no dataseries → "stats_1"
        self._setup_folder()
        self._run_compute()
        self.tool._results_to_numpy()
        self.tool._NMToolStats__results.clear()
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats_1")

    def test_results_to_numpy_folder_named_with_dataseries(self):
        # Dataseries selected → name uses dataseries name
        from pyneuromatic.core.nm_dataseries import NMDataSeries
        self._setup_folder()
        ds = NMDataSeries(name="Record")
        self.tool._select["dataseries"] = ds
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats_Record_0")

    def test_results_to_numpy_folder_named_with_dataseries_and_channel(self):
        # Dataseries + channel → stats_{ds}_{ch}_N
        from pyneuromatic.core.nm_dataseries import NMDataSeries
        from pyneuromatic.core.nm_channel import NMChannel
        self._setup_folder()
        self.tool._select["dataseries"] = NMDataSeries(name="Record")
        self.tool._select["channel"] = NMChannel(name="A")
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats_Record_A_0")

    def test_results_to_numpy_folder_named_channel_only(self):
        # Channel but no dataseries → stats_{ch}_N
        from pyneuromatic.core.nm_channel import NMChannel
        self._setup_folder()
        self.tool._select["channel"] = NMChannel(name="B")
        self._run_compute()
        f = self.tool._results_to_numpy()
        self.assertEqual(f.name, "stats_B_0")

    def test_results_to_numpy_creates_data_array(self):
        self._setup_folder()
        self._run_compute(n_arrays=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_data", f.data)

    def test_results_to_numpy_data_array_length(self):
        self._setup_folder()
        self._run_compute(n_arrays=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_data")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_creates_mean_y_array(self):
        self._setup_folder()
        self._run_compute(func="mean", n_arrays=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_mean_y", f.data)

    def test_results_to_numpy_mean_y_array_length(self):
        self._setup_folder()
        self._run_compute(func="mean", n_arrays=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_mean_y")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_compound_mean_std_splits(self):
        # mean+std → ST_w0_mean_y AND ST_w0_std_y
        self._setup_folder()
        self._run_compute(func="mean+std", n_arrays=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_mean_y", f.data)
        self.assertIn("ST_w0_std_y", f.data)
        self.assertNotIn("ST_w0_mean_std_y", f.data)

    def test_results_to_numpy_bsln_uses_bsln_prefix(self):
        # baseline result → ST_w0_bsln_y
        self._setup_folder()
        self._run_compute_with_bsln()
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_bsln_y", f.data)

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

    def test_results_to_numpy_no_folder_returns_none(self):
        from pyneuromatic.analysis.nm_tool import HIERARCHY_SELECT_KEYS
        self.tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=arr)

    # --- save_to_numpy ---

    def test_histogram_saves_counts_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertIn("H_ST_w0_mean_y_counts", self.tf.data)

    def test_histogram_saves_edges_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertIn("H_ST_w0_mean_y_edges", self.tf.data)

    def test_histogram_xscale_start(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        d = self.tf.data.get("H_ST_w0_mean_y_counts")
        self.assertAlmostEqual(d.xscale.start, 1.0)

    def test_histogram_xscale_delta(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        d = self.tf.data.get("H_ST_w0_mean_y_counts")
        self.assertAlmostEqual(d.xscale.delta, 1.0)  # (5 - 1) / 4 = 1.0

    def test_histogram_no_save_skips_arrays(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4, save_to_numpy=False)
        self.assertNotIn("H_ST_w0_mean_y_counts", self.tf.data)
        self.assertNotIn("H_ST_w0_mean_y_edges", self.tf.data)

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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
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

    def test_nan_in_array_counts_as_failure(self):
        self.tf.data.new(
            "ST_w0_nan_y",
            nparray=np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        )
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_nan_y", ">", 0, save_to_numpy=False
        )
        # NaN > 0 is False
        self.assertEqual(r["successes"], 4)
        self.assertEqual(r["failures"], 1)

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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder

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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder

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
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder

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

    def test_xclip_default(self):
        self.assertTrue(self.cfg.xclip)

    def test_results_to_history_default(self):
        self.assertFalse(self.cfg.results_to_history)

    def test_results_to_cache_default(self):
        self.assertTrue(self.cfg.results_to_cache)

    def test_results_to_numpy_default(self):
        self.assertFalse(self.cfg.results_to_numpy)

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
        self.assertTrue(t.config.xclip)

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
        for key in ("ignore_nans", "xclip", "overwrite",
                    "results_to_history", "results_to_cache", "results_to_numpy"):
            self.assertIn(key, d)


class TestNMToolStatsOverwrite(unittest.TestCase):
    """overwrite flag controls subfolder reuse vs. new numbered subfolders."""

    def _setup(self, overwrite=False):
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.analysis.nm_tool import HIERARCHY_SELECT_KEYS
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
        tool._results_to_numpy()
        self.assertIn("stats_0", folder.toolfolder)

    def test_overwrite_false_creates_stats_1_on_second_run(self):
        tool, folder = self._setup(overwrite=False)
        self._fill_results(tool)
        tool._results_to_numpy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool)
        tool._results_to_numpy()
        self.assertIn("stats_0", folder.toolfolder)
        self.assertIn("stats_1", folder.toolfolder)

    def test_overwrite_true_creates_stats_0(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool)
        tool._results_to_numpy()
        self.assertIn("stats_0", folder.toolfolder)

    def test_overwrite_true_reuses_stats_0_on_second_run(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool)
        tool._results_to_numpy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool)
        tool._results_to_numpy()
        self.assertIn("stats_0", folder.toolfolder)
        self.assertNotIn("stats_1", folder.toolfolder)

    def test_overwrite_true_replaces_st_arrays(self):
        tool, folder = self._setup(overwrite=True)
        self._fill_results(tool, n=3)
        f = tool._results_to_numpy()
        arr_first = f.data.get("ST_w0_mean_y").nparray.copy()
        tool._NMToolStats__results.clear()
        self._fill_results(tool, n=5)
        f2 = tool._results_to_numpy()
        self.assertIs(f, f2)
        arr_second = f2.data.get("ST_w0_mean_y").nparray
        self.assertEqual(len(arr_second), 5)
        self.assertEqual(len(arr_first), 3)


class TestNMToolStatsNotes(unittest.TestCase):
    """Notes are attached to numeric ST_ arrays in _results_to_numpy()."""

    def setUp(self):
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.analysis.nm_tool import HIERARCHY_SELECT_KEYS
        self.tool = nms.NMToolStats()
        folder = NMFolder(name="F")
        self.tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        self.tool._select["folder"] = folder
        w = list(self.tool.windows)[0]
        w.func = "mean"
        w.x0 = 10.0
        w.x1 = 50.0
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
        self.f = self.tool._results_to_numpy()
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

    def test_note_contains_x0(self):
        self.assertIn("x0=10.0", self._note_text())

    def test_note_contains_x1(self):
        self.assertIn("x1=50.0", self._note_text())

    def test_note_contains_n(self):
        self.assertIn("n=%d" % self.n_arrays, self._note_text())

    def test_st_data_array_has_no_note(self):
        st_data = self.f.data.get("ST_w0_data")
        if st_data is None:
            self.skipTest("ST_w0_data not created for this func")
        notes = getattr(st_data, "notes", None)
        if notes is not None:
            self.assertEqual(len(list(notes)), 0)

    def test_bsln_array_note_uses_bsln_x_range(self):
        # Set up a run with baseline enabled; ST_w0_bsln_y note should
        # record the baseline x-range (bsln_x0/bsln_x1), not the main x0/x1.
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.analysis.nm_tool import HIERARCHY_SELECT_KEYS
        tool = nms.NMToolStats()
        folder = NMFolder(name="F")
        tool._select = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        tool._select["folder"] = folder
        w = list(tool.windows)[0]
        w.func = "mean"
        w.x0 = 10.0
        w.x1 = 50.0
        w._bsln_on_set(True, quiet=True)
        w._bsln_func_set({"name": "mean"}, quiet=True)
        w._x_set("bsln_x0", 0.0, quiet=True)
        w._x_set("bsln_x1", 20.0, quiet=True)
        for k in range(3):
            data = _make_data(name="recordA%d" % k)
            w.compute(data)
            wname = w.name
            results = w.results
            if wname in tool._NMToolStats__results:
                tool._NMToolStats__results[wname].append(results)
            else:
                tool._NMToolStats__results[wname] = [results]
        f = tool._results_to_numpy()
        bsln_arr = f.data.get("ST_w0_bsln_y")
        self.assertIsNotNone(bsln_arr)
        notes = getattr(bsln_arr, "notes", None)
        self.assertIsNotNone(notes)
        note_text = " ".join(str(n) for n in notes)
        self.assertIn("id=bsln", note_text)
        self.assertIn("x0=0.0", note_text)
        self.assertIn("x1=20.0", note_text)
        # Main x0/x1 should NOT appear in the bsln note
        self.assertNotIn("x0=10.0", note_text)


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

    def test_func_setter_logs_basic(self):
        self._ch.clear()
        self.w0.func = 'mean'
        self.assertEqual(len(self._ch.buffer), 1)
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].func", cmd)
        self.assertIn("NMStatFuncBasic", cmd)
        self.assertIn("'mean'", cmd)

    def test_func_setter_logs_maxmin_no_n_mean(self):
        self._ch.clear()
        self.w0.func = 'max'
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncMaxMin", cmd)
        self.assertIn("'max'", cmd)
        self.assertNotIn("n_mean", cmd)

    def test_func_setter_logs_maxmin_with_n_mean(self):
        self._ch.clear()
        self.w0.func = {'name': 'mean@max', 'n_mean': 3}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncMaxMin", cmd)
        self.assertIn("'mean@max'", cmd)
        self.assertIn("n_mean=3", cmd)

    def test_func_setter_logs_level(self):
        self._ch.clear()
        self.w0.func = {'name': 'level', 'ylevel': 10.0}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncLevel", cmd)
        self.assertIn("ylevel=10.0", cmd)

    def test_func_setter_logs_levelnstd(self):
        self._ch.clear()
        self.w0.func = {'name': 'level', 'n_std': 2.0}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncLevelNstd", cmd)
        self.assertIn("n_std=2.0", cmd)

    def test_func_setter_logs_risetime(self):
        self._ch.clear()
        self.w0.func = {'name': 'risetime+', 'p0': 10.0, 'p1': 90.0}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncRiseTime", cmd)
        self.assertIn("p0=10.0", cmd)
        self.assertIn("p1=90.0", cmd)

    def test_func_setter_logs_decaytime_default_p0(self):
        self._ch.clear()
        self.w0.func = 'decaytime+'
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncDecayTime", cmd)
        self.assertIn("p0=", cmd)

    def test_func_setter_logs_decaytime_custom_p0(self):
        self._ch.clear()
        self.w0.func = {'name': 'decaytime+', 'p0': 50.0}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncDecayTime", cmd)
        self.assertIn("p0=50.0", cmd)

    def test_func_setter_logs_fwhm_default_p0_p1(self):
        self._ch.clear()
        self.w0.func = 'fwhm+'
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncFWHM", cmd)
        self.assertIn("p0=", cmd)
        self.assertIn("p1=", cmd)

    def test_func_setter_logs_fwhm_custom_p0_p1(self):
        self._ch.clear()
        self.w0.func = {'name': 'fwhm+', 'p0': 30.0, 'p1': 70.0}
        cmd = self._ch.buffer[0]['command']
        self.assertIn("NMStatFuncFWHM", cmd)
        self.assertIn("p0=30.0", cmd)
        self.assertIn("p1=70.0", cmd)

    def test_func_setter_none_does_not_log(self):
        self._ch.clear()
        self.w0.func = None
        self.assertEqual(len(self._ch.buffer), 0)

    # ------------------------------------------------------------------
    # NMStatWin scalar setters

    def test_x0_setter_logs(self):
        self._ch.clear()
        self.w0.x0 = 100.0
        self.assertEqual(len(self._ch.buffer), 1)
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].x0", cmd)
        self.assertIn("100.0", cmd)

    def test_x1_setter_logs(self):
        self._ch.clear()
        self.w0.x1 = 200.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].x1", cmd)
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

    def test_bsln_x0_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_x0 = -10.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_x0", cmd)
        self.assertIn("-10.0", cmd)

    def test_bsln_x1_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_x1 = 0.0
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_x1", cmd)
        self.assertIn("0.0", cmd)

    def test_bsln_func_setter_logs(self):
        self._ch.clear()
        self.w0.bsln_func = 'mean'
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows['w0'].bsln_func", cmd)
        self.assertIn("'mean'", cmd)

    def test_bsln_func_setter_none_does_not_log(self):
        self._ch.clear()
        self.w0.bsln_func = None
        self.assertEqual(len(self._ch.buffer), 0)

    # ------------------------------------------------------------------
    # NMStatWinContainer.new()

    def test_windows_new_logs(self):
        self._ch.clear()
        self.tool.windows.new()
        self.assertEqual(len(self._ch.buffer), 1)
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.windows.new(", cmd)
        self.assertIn("'w1'", cmd)

    # ------------------------------------------------------------------
    # NMToolStats tool-level flag setters

    def test_xclip_setter_logs(self):
        self._ch.clear()
        self.tool.xclip = True
        cmd = self._ch.buffer[0]['command']
        self.assertIn("stats.xclip", cmd)
        self.assertIn("True", cmd)

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
