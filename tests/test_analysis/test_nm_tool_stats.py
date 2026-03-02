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

    def _run_compute_with_bsln(self, func="mean", bsln_func="mean", n_waves=3):
        """Compute stat with baseline enabled for n_waves."""
        w = list(self.tool.windows)[0]
        w.func = func
        w._bsln_on_set(True, quiet=True)
        w._bsln_func_set({"name": bsln_func}, quiet=True)
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
        self._run_compute(n_waves=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_data", f.data)

    def test_results_to_numpy_data_array_length(self):
        self._setup_folder()
        self._run_compute(n_waves=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_data")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_creates_mean_y_array(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._results_to_numpy()
        self.assertIn("ST_w0_mean_y", f.data)

    def test_results_to_numpy_mean_y_array_length(self):
        self._setup_folder()
        self._run_compute(func="mean", n_waves=3)
        f = self.tool._results_to_numpy()
        d = f.data.get("ST_w0_mean_y")
        self.assertEqual(len(d.nparray), 3)

    def test_results_to_numpy_compound_mean_std_splits(self):
        # mean+std → ST_w0_mean_y AND ST_w0_std_y
        self._setup_folder()
        self._run_compute(func="mean+std", n_waves=3)
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
    """Tests for NMToolStats2.histogram()."""

    def setUp(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        self.tf = NMToolFolder(name="stats0")
        arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=arr)

    # --- return value ---

    def test_histogram_returns_dict(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y")
        self.assertIsInstance(r, dict)
        self.assertIn("counts", r)
        self.assertIn("edges", r)

    def test_histogram_counts_length(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertEqual(len(r["counts"]), 4)

    def test_histogram_edges_length(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertEqual(len(r["edges"]), 5)  # bins + 1

    def test_histogram_counts_sum(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertEqual(sum(r["counts"]), 9)  # all 9 values counted

    # --- saved arrays (save_to_numpy=True, default) ---

    def test_histogram_saves_counts_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertIn("HIST_ST_w0_mean_y_counts", self.tf.data)

    def test_histogram_saves_edges_array(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        self.assertIn("HIST_ST_w0_mean_y_edges", self.tf.data)

    def test_histogram_xscale_start(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        d = self.tf.data.get("HIST_ST_w0_mean_y_counts")
        self.assertAlmostEqual(d.xscale.start, 1.0)

    def test_histogram_xscale_delta(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4)
        d = self.tf.data.get("HIST_ST_w0_mean_y_counts")
        self.assertAlmostEqual(d.xscale.delta, 1.0)  # (5 - 1) / 4 = 1.0

    # --- save_to_numpy=False ---

    def test_histogram_no_save_skips_arrays(self):
        nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4, save_to_numpy=False)
        self.assertNotIn("HIST_ST_w0_mean_y_counts", self.tf.data)
        self.assertNotIn("HIST_ST_w0_mean_y_edges", self.tf.data)

    def test_histogram_no_save_still_returns_dict(self):
        r = nms.NMToolStats2.histogram(self.tf, "ST_w0_mean_y", bins=4,
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
        from pyneuromatic.core.nm_folder import NMFolder
        from pyneuromatic.core.nm_dataseries import NMDataSeries

        self.tf = NMToolFolder(name="stats0")
        # ST_w0_mean_y: values 1..5
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tf.data.new("ST_w0_mean_y", nparray=self.arr.copy())
        # ST_w0_data: wave names for epoch set creation
        self.wave_names = ["RecordA0", "RecordA1", "RecordA2",
                           "RecordA3", "RecordA4"]
        self.tf.data.new(
            "ST_w0_data",
            nparray=np.array(self.wave_names, dtype=object),
        )

        # Build a real NMDataSeries with 5 epochs for set-creation tests
        self.nm = NMManager(quiet=True)
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("f0")
        self.ds = self.folder.dataseries.new("Record")
        for i in range(5):
            self.ds.epochs.new("E%d" % i)

    # --- condition string ---

    def test_condition_str_greater_than(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y > 3")

    def test_condition_str_greater_than_or_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y",
            greater_than_or_equal=3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y >= 3")

    def test_condition_str_less_than(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", less_than=4, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y < 4")

    def test_condition_str_less_than_or_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", less_than_or_equal=4, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y <= 4")

    def test_condition_str_range(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y",
            greater_than=2, less_than=5, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 < y < 5")

    def test_condition_str_range_inclusive(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y",
            greater_than_or_equal=2, less_than_or_equal=4,
            save_to_numpy=False
        )
        self.assertEqual(r["condition"], "2 <= y <= 4")

    def test_condition_str_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", equal=3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y == 3")

    def test_condition_str_not_equal(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", not_equal=3, save_to_numpy=False
        )
        self.assertEqual(r["condition"], "y != 3")

    # --- mask and counts ---

    def test_greater_than_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3, save_to_numpy=False
        )
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)
        self.assertEqual(r["failures"], 3)

    def test_less_than_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", less_than=3, save_to_numpy=False
        )
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)
        self.assertEqual(r["failures"], 3)

    def test_range_mask(self):
        # 2 < y < 5 → [2, 3, 4] pass
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y",
            greater_than=2, less_than=5, save_to_numpy=False
        )
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 2)

    def test_equal_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", equal=3, save_to_numpy=False
        )
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(r["mask"], expected)
        self.assertEqual(r["successes"], 1)

    def test_not_equal_mask(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", not_equal=3, save_to_numpy=False
        )
        self.assertEqual(r["successes"], 4)

    def test_nan_in_array_counts_as_failure(self):
        self.tf.data.new(
            "ST_w0_nan_y",
            nparray=np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        )
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_nan_y", greater_than=0, save_to_numpy=False
        )
        # NaN > 0 is False
        self.assertEqual(r["successes"], 4)
        self.assertEqual(r["failures"], 1)

    # --- result array ---

    def test_binary_output_true(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3, save_to_numpy=False
        )
        np.testing.assert_array_equal(
            r["result"], np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        )

    def test_binary_output_false(self):
        r = nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3,
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
            self.tf, "ST_w0_mean_y", greater_than=3, save_to_numpy=True
        )
        self.assertIn("IQ_ST_w0_mean_y", self.tf.data)

    def test_save_to_numpy_false_does_not_create_array(self):
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3, save_to_numpy=False
        )
        self.assertNotIn("IQ_ST_w0_mean_y", self.tf.data)

    # --- epoch sets ---

    def test_set_name_success_creates_epoch_set(self):
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3,
            dataseries=self.ds,
            set_name_success="Successes",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get("Successes")
        self.assertIsNotNone(set_epochs)

    def test_set_name_success_contains_correct_epochs(self):
        # greater_than=3 → E3, E4 pass
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3,
            dataseries=self.ds,
            set_name_success="Pass",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get("Pass")
        epoch_names = [ep.name for ep in set_epochs]
        self.assertIn("E3", epoch_names)
        self.assertIn("E4", epoch_names)
        self.assertNotIn("E0", epoch_names)

    def test_set_name_failure_contains_failing_epochs(self):
        # greater_than=3 → E0, E1, E2 fail
        nms.NMToolStats2.inequality(
            self.tf, "ST_w0_mean_y", greater_than=3,
            dataseries=self.ds,
            set_name_failure="Fail",
            save_to_numpy=False,
        )
        set_epochs = self.ds.epochs.sets.get("Fail")
        epoch_names = [ep.name for ep in set_epochs]
        self.assertIn("E0", epoch_names)
        self.assertIn("E1", epoch_names)
        self.assertIn("E2", epoch_names)
        self.assertNotIn("E3", epoch_names)

    # --- type validation ---

    def test_bad_toolfolder_raises_typeerror(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.inequality("not_a_folder", "ST_w0_mean_y",
                                        greater_than=1)

    def test_bad_name_raises_typeerror(self):
        with self.assertRaises(TypeError):
            nms.NMToolStats2.inequality(self.tf, 123, greater_than=1)

    def test_unknown_name_raises_keyerror(self):
        with self.assertRaises(KeyError):
            nms.NMToolStats2.inequality(self.tf, "ST_w0_missing",
                                        greater_than=1)

    def test_no_condition_raises_valueerror(self):
        with self.assertRaises(ValueError):
            nms.NMToolStats2.inequality(self.tf, "ST_w0_mean_y",
                                        save_to_numpy=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
