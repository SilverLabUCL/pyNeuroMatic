#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_spike: NMToolSpike.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

import numpy as np

from pyneuromatic.analysis.nm_tool_spike import NMToolSpike, NMToolSpikeConfig
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager, HIERARCHY_SELECT_KEYS

NM = NMManager(quiet=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SR = 10000.0   # sample rate (Hz) used in test signals
_DELTA = 1.0 / _SR


def _sine_data(name="recordA0", freq=200.0, n=1000, ylevel=0.0):
    """NMData containing one full second of a sine wave at *freq* Hz.

    The number of upward crossings of *ylevel* = 0 equals *freq* (one per
    cycle when the signal starts at 0 and rises first).
    """
    t = np.arange(n) * _DELTA
    y = np.sin(2 * np.pi * freq * t)
    return NMData(
        NM,
        name=name,
        nparray=y,
        xscale={"start": 0.0, "delta": _DELTA, "label": "Time", "units": "s"},
        yscale={"label": "Voltage", "units": "mV"},
    )


def _make_targets(data, folder=None):
    """Build run targets list from a list of NMData objects."""
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    return [
        {k: None for k in HIERARCHY_SELECT_KEYS} | {"folder": folder, "data": d}
        for d in data
    ]


def _run(tool, data, folder=None):
    """Run detection on *data* and return (tool, folder)."""
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    targets = _make_targets(data, folder=folder)
    tool.run_all(targets)
    return folder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNMToolSpikeDefaults(unittest.TestCase):
    """Constructor defaults."""

    def setUp(self):
        self.tool = NMToolSpike()

    def test_ylevel_default(self):
        self.assertEqual(self.tool.ylevel, 0.0)

    def test_func_name_default(self):
        self.assertEqual(self.tool.func_name, "level+")

    def test_x0_default(self):
        import math
        self.assertEqual(self.tool.x0, -math.inf)

    def test_x1_default(self):
        import math
        self.assertEqual(self.tool.x1, math.inf)

    def test_ignore_nans_default(self):
        self.assertTrue(self.tool.ignore_nans)

    def test_results_to_history_default(self):
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_cache_default(self):
        self.assertTrue(self.tool.results_to_cache)

    def test_results_to_numpy_default(self):
        self.assertTrue(self.tool.results_to_numpy)


class TestNMToolSpikeProperties(unittest.TestCase):
    """Property validation."""

    def setUp(self):
        self.tool = NMToolSpike()

    # ylevel
    def test_ylevel_accepts_float(self):
        self.tool.ylevel = -10.5
        self.assertEqual(self.tool.ylevel, -10.5)

    def test_ylevel_accepts_int(self):
        self.tool.ylevel = 5
        self.assertEqual(self.tool.ylevel, 5.0)

    def test_ylevel_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.ylevel = True

    def test_ylevel_rejects_string(self):
        with self.assertRaises(TypeError):
            self.tool.ylevel = "0"

    # func_name
    def test_func_name_accepts_level(self):
        self.tool.func_name = "level"
        self.assertEqual(self.tool.func_name, "level")

    def test_func_name_accepts_level_minus(self):
        self.tool.func_name = "level-"
        self.assertEqual(self.tool.func_name, "level-")

    def test_func_name_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self.tool.func_name = "peak"

    def test_func_name_rejects_non_string(self):
        with self.assertRaises(TypeError):
            self.tool.func_name = 1

    # x0 / x1
    def test_x0_accepts_float(self):
        self.tool.x0 = 0.01
        self.assertAlmostEqual(self.tool.x0, 0.01)

    def test_x0_accepts_inf(self):
        import math
        self.tool.x0 = -math.inf
        self.assertEqual(self.tool.x0, -math.inf)

    def test_x0_rejects_nan(self):
        import math
        with self.assertRaises(ValueError):
            self.tool.x0 = math.nan

    def test_x0_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.x0 = True

    def test_x1_accepts_float(self):
        self.tool.x1 = 0.05
        self.assertAlmostEqual(self.tool.x1, 0.05)

    def test_x1_rejects_nan(self):
        import math
        with self.assertRaises(ValueError):
            self.tool.x1 = math.nan

    def test_x1_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.x1 = False

    # ignore_nans
    def test_ignore_nans_set_false(self):
        self.tool.ignore_nans = False
        self.assertFalse(self.tool.ignore_nans)

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.ignore_nans = 1

    # results_to_* flags
    def test_results_to_history_set(self):
        self.tool.results_to_history = True
        self.assertTrue(self.tool.results_to_history)

    def test_results_to_history_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_history = 1

    def test_results_to_cache_set(self):
        self.tool.results_to_cache = False
        self.assertFalse(self.tool.results_to_cache)

    def test_results_to_cache_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_cache = "yes"

    def test_results_to_numpy_set(self):
        self.tool.results_to_numpy = False
        self.assertFalse(self.tool.results_to_numpy)

    def test_results_to_numpy_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.tool.results_to_numpy = None


class TestNMToolSpikeDetection(unittest.TestCase):
    """run_all detection behaviour."""

    def setUp(self):
        self.tool = NMToolSpike()

    def test_rising_crossings_count(self):
        # 200 Hz sine, 1000 samples at 10 kHz → 20 cycles → 20 rising crossings
        # func_name="level+" counts upward crossings of ylevel=0.0
        d = _sine_data(freq=200.0, n=1000)
        folder = _run(self.tool, [d])
        tf = folder.toolfolder
        f = tf.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), 20)

    def test_falling_crossings(self):
        self.tool.func_name = "level-"
        d = _sine_data(freq=200.0, n=1000)
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), 20)

    def test_both_directions_double_count(self):
        self.tool.func_name = "level"
        d = _sine_data(freq=200.0, n=1000)
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), 40)

    def test_x0_x1_window_restricts_detection(self):
        # 200 Hz sine → 20 rising crossings over 0.1 s (1000 samples at 10 kHz).
        # Restrict to first half (0–0.05 s) → ~10 crossings.
        d = _sine_data(freq=200.0, n=1000)
        self.tool.x0 = 0.0
        self.tool.x1 = 0.05
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        count_full = 20
        count_win = int(f.data.get("SP_count").nparray[0])
        self.assertLess(count_win, count_full)
        self.assertGreater(count_win, 0)

    def test_ignore_nans_true_detects_crossing_across_nan_gap(self):
        # y=0 at index 0, NaN at index 1, y=1 at index 2 → crossing exists
        y = np.array([0.0, float("nan"), 1.0])
        d = NMData(NM, name="recA0", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        self.tool.ylevel = 0.5
        self.tool.ignore_nans = True
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 1)

    def test_ignore_nans_false_blocks_crossing_across_nan_gap(self):
        # Same data; with ignore_nans=False the NaN blocks the crossing
        y = np.array([0.0, float("nan"), 1.0])
        d = NMData(NM, name="recA0", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        self.tool.ylevel = 0.5
        self.tool.ignore_nans = False
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 0)

    def test_per_epoch_sp_arrays_created(self):
        data = [_sine_data(name="recA%d" % i) for i in range(3)]
        folder = _run(self.tool, data)
        f = folder.toolfolder.get("Spike_0")
        for i in range(3):
            self.assertIn("SP_recA%d" % i, f.data)

    def test_sp_count_length_equals_epoch_count(self):
        data = [_sine_data(name="recA%d" % i) for i in range(5)]
        folder = _run(self.tool, data)
        f = folder.toolfolder.get("Spike_0")
        self.assertEqual(len(f.data.get("SP_count").nparray), 5)

    def test_sp_count_matches_sp_array_lengths(self):
        data = [_sine_data(name="recA%d" % i, freq=100.0 * (i + 1)) for i in range(3)]
        folder = _run(self.tool, data)
        f = folder.toolfolder.get("Spike_0")
        counts = f.data.get("SP_count").nparray
        for i, d in enumerate(data):
            times = f.data.get("SP_recA%d" % i).nparray
            self.assertEqual(int(counts[i]), len(times))

    def test_epoch_with_no_crossings(self):
        # Constant signal — no crossings
        y = np.ones(500) * 5.0
        d = NMData(NM, name="flat", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 0)
        self.assertEqual(len(f.data.get("SP_flat").nparray), 0)

    def test_none_nparray_skipped(self):
        d = NMData(NM, name="empty",
                   xscale={"start": 0.0, "delta": _DELTA})
        # nparray is None — should not appear in epoch_names
        folder = _run(self.tool, [d])
        # run_finish skips when no epoch_names
        self.assertIsNone(self.tool._toolfolder)

    def test_sp_times_are_floats(self):
        d = _sine_data(freq=100.0)
        folder = _run(self.tool, [d])
        f = folder.toolfolder.get("Spike_0")
        times = f.data.get("SP_recordA0").nparray
        self.assertEqual(times.dtype.kind, "f")

    def test_xscale_units_captured(self):
        d = _sine_data()
        _run(self.tool, [d])
        self.assertEqual(self.tool._detected_xunits, "s")


class TestNMToolSpikeSubfolderNaming(unittest.TestCase):
    """Subfolder naming follows spike_{dataseries}_{channel}_N pattern."""

    def setUp(self):
        self.tool = NMToolSpike()

    def _run_with_selection(self, folder, dataseries=None, channel=None):
        d = _sine_data()
        target = {k: None for k in HIERARCHY_SELECT_KEYS}
        target["folder"] = folder
        target["data"] = d
        if dataseries is not None:
            target["dataseries"] = dataseries
        if channel is not None:
            target["channel"] = channel
        self.tool.run_all([target])

    def test_no_dataseries_no_channel(self):
        folder = NMFolder(NM, name="F")
        self._run_with_selection(folder)
        self.assertIn("Spike_0", folder.toolfolder)

    def test_with_dataseries(self):
        folder = NMFolder(NM, name="F")
        ds = NMDataSeries(NM, name="Record")
        self._run_with_selection(folder, dataseries=ds)
        self.assertIn("Spike_Record_0", folder.toolfolder)

    def test_with_dataseries_and_channel(self):
        folder = NMFolder(NM, name="F")
        ds = NMDataSeries(NM, name="Record")
        ch = NMChannel(NM, name="A")
        self._run_with_selection(folder, dataseries=ds, channel=ch)
        self.assertIn("Spike_Record_A_0", folder.toolfolder)

    def test_second_run_increments_suffix_when_overwrite_false(self):
        folder = NMFolder(NM, name="F")
        self.tool.overwrite = False
        self._run_with_selection(folder)
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self._run_with_selection(folder)
        self.assertIn("Spike_0", folder.toolfolder)
        self.assertIn("Spike_1", folder.toolfolder)


class TestNMToolSpikePST(unittest.TestCase):
    """pst() histogram method."""

    def setUp(self):
        self.tool = NMToolSpike()
        data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        self.folder = _run(self.tool, data)

    def test_pst_returns_nmdata(self):
        result = self.tool.pst(bins=50)
        self.assertIsInstance(result, NMData)

    def test_pst_length_equals_bins(self):
        result = self.tool.pst(bins=50)
        self.assertEqual(len(result.nparray), 50)

    def test_pst_total_counts_equals_total_spikes(self):
        f = self.folder.toolfolder.get("Spike_0")
        total_spikes = int(f.data.get("SP_count").nparray.sum())
        result = self.tool.pst(bins=50)
        self.assertEqual(int(result.nparray.sum()), total_spikes)

    def test_pst_xscale_delta_equals_bin_width(self):
        result = self.tool.pst(bins=100)
        all_times = np.concatenate(self.tool._spike_times)
        expected_delta = (all_times.max() - all_times.min()) / 100
        self.assertAlmostEqual(result.xscale.delta, expected_delta, places=10)

    def test_pst_x0_x1_restricts_range(self):
        result = self.tool.pst(bins=50, x0=0.0, x1=0.05)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.xscale.start, 0.0, places=10)

    def test_pst_output_mode_count_matches_total_spikes(self):
        f = self.folder.toolfolder.get("Spike_0")
        total_spikes = int(f.data.get("SP_count").nparray.sum())
        result = self.tool.pst(bins=50, output_mode="count")
        self.assertEqual(int(result.nparray.sum()), total_spikes)

    def test_pst_output_mode_rate_yscale_label(self):
        result = self.tool.pst(bins=50, output_mode="rate")
        self.assertIn("Hz", result.yscale.label)

    def test_pst_output_mode_rate_values(self):
        # rate = count / (n_epochs * bin_width); sum over bins * bin_width * n_epochs = total spikes
        n_epochs = len(self.tool._spike_times)
        result = self.tool.pst(bins=50, output_mode="rate")
        bin_width = result.xscale.delta
        reconstructed = result.nparray.sum() * bin_width * n_epochs
        f = self.folder.toolfolder.get("Spike_0")
        total_spikes = float(f.data.get("SP_count").nparray.sum())
        self.assertAlmostEqual(reconstructed, total_spikes, places=6)

    def test_pst_output_mode_probability_range(self):
        result = self.tool.pst(bins=50, output_mode="probability")
        self.assertTrue(np.all(result.nparray >= 0.0))
        self.assertTrue(np.all(result.nparray <= 1.0))

    def test_pst_output_mode_probability_yscale_label(self):
        result = self.tool.pst(bins=50, output_mode="probability")
        self.assertIn("probability", result.yscale.label.lower())

    def test_pst_output_mode_case_insensitive(self):
        # "Rate" should behave identically to "rate" — yscale label contains "Hz"
        result = self.tool.pst(bins=50, output_mode="Rate")
        self.assertIn("Hz", result.yscale.label)

    def test_pst_output_mode_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.tool.pst(bins=50, output_mode="density")

    def test_pst_returns_none_when_no_spikes(self):
        tool2 = NMToolSpike()
        y = np.ones(500) * 5.0
        d = NMData(NM, name="flat", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        _run(tool2, [d])
        result = tool2.pst()
        self.assertIsNone(result)

    def test_pst_raises_before_run(self):
        tool2 = NMToolSpike()
        with self.assertRaises(RuntimeError):
            tool2.pst()

    def test_pst_written_to_subfolder(self):
        self.tool.pst(bins=50)
        f = self.folder.toolfolder.get("Spike_0")
        self.assertIn("SP_PST", f.data)


class TestNMToolSpikeISI(unittest.TestCase):
    """isi() histogram method."""

    def setUp(self):
        self.tool = NMToolSpike()
        # Three epochs, each with ~10 spikes (100 Hz, 0.1 s)
        data = [_sine_data(name="recA%d" % i, freq=100.0, n=1000) for i in range(3)]
        self.folder = _run(self.tool, data)

    def test_isi_returns_nmdata(self):
        result = self.tool.isi(bins=50)
        self.assertIsInstance(result, NMData)

    def test_isi_length_equals_bins(self):
        result = self.tool.isi(bins=50)
        self.assertEqual(len(result.nparray), 50)

    def test_isi_xscale_start_nonnegative(self):
        result = self.tool.isi(bins=50)
        self.assertGreaterEqual(result.xscale.start, 0.0)

    def test_isi_max_isi_sets_upper_bound(self):
        result = self.tool.isi(bins=50, max_isi=0.02)
        self.assertIsNotNone(result)

    def test_isi_returns_none_when_fewer_than_2_spikes(self):
        # Signal with exactly one crossing
        y = np.zeros(100)
        y[50] = 1.0  # single upward crossing
        d = NMData(NM, name="single", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        tool2 = NMToolSpike()
        _run(tool2, [d])
        result = tool2.isi()
        self.assertIsNone(result)

    def test_isi_raises_before_run(self):
        tool2 = NMToolSpike()
        with self.assertRaises(RuntimeError):
            tool2.isi()

    def test_isi_written_to_subfolder(self):
        self.tool.isi(bins=50)
        f = self.folder.toolfolder.get("Spike_0")
        self.assertIn("SP_ISI", f.data)

    def test_isi_total_intervals_correct(self):
        # Each epoch: 10 spikes → 9 ISIs; 3 epochs → 27 total ISIs
        result = self.tool.isi(bins=50)
        total_intervals = sum(
            len(np.diff(t)) for t in self.tool._spike_times if len(t) >= 2
        )
        self.assertEqual(int(result.nparray.sum()), total_intervals)

    def test_isi_x0_x1_filters_spike_times(self):
        # Restrict to first half of recording; fewer intervals than full range.
        # Use separate tools/folders so SP_ISI name does not collide.
        data = [_sine_data(name="recA%d" % i, freq=100.0, n=1000) for i in range(3)]
        tool_full = NMToolSpike()
        folder_full = _run(tool_full, data)
        result_full = tool_full.isi(bins=50)
        tool_win = NMToolSpike()
        folder_win = _run(tool_win, data)
        result_win = tool_win.isi(bins=50, x0=0.0, x1=0.05)
        self.assertLess(int(result_win.nparray.sum()), int(result_full.nparray.sum()))

    def test_isi_min_isi_sets_lower_bound(self):
        result = self.tool.isi(bins=50, min_isi=0.001)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.xscale.start, 0.001)

    def test_isi_output_mode_probability_sums_to_one(self):
        result = self.tool.isi(bins=50, output_mode="probability")
        self.assertAlmostEqual(float(result.nparray.sum()), 1.0, places=6)

    def test_isi_output_mode_probability_yscale_label(self):
        result = self.tool.isi(bins=50, output_mode="probability")
        self.assertIn("probability", result.yscale.label.lower())

    def test_isi_output_mode_case_insensitive(self):
        result = self.tool.isi(bins=50, output_mode="Probability")
        self.assertIn("probability", result.yscale.label.lower())

    def test_isi_output_mode_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.tool.isi(bins=50, output_mode="rate")

    def test_isi_note_contains_new_params(self):
        self.tool.isi(bins=50, x0=0.0, x1=0.08, min_isi=0.001, max_isi=0.05,
                      output_mode="probability")
        f = self.folder.toolfolder.get("Spike_0")
        note = f.data.get("SP_ISI").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)
        self.assertIn("min_isi=", note)
        self.assertIn("max_isi=", note)
        self.assertIn("output_mode=", note)


class TestNMToolSpikeOverwrite(unittest.TestCase):
    """overwrite flag controls subfolder reuse vs. new numbered subfolders."""

    def _run_once(self, tool, folder, freq=100.0):
        d = _sine_data(freq=freq)
        targets = _make_targets([d], folder=folder)
        tool.run_all(targets)

    def test_overwrite_true_default(self):
        self.assertTrue(NMToolSpike().overwrite)

    def test_overwrite_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMToolSpike().overwrite = 1

    def test_overwrite_true_creates_Spike_0(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder)
        self.assertIn("Spike_0", folder.toolfolder)

    def test_overwrite_true_reuses_Spike_0_on_second_run(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder)
        self._run_once(tool, folder)
        self.assertIn("Spike_0", folder.toolfolder)
        self.assertNotIn("Spike_1", folder.toolfolder)

    def test_overwrite_true_replaces_sp_arrays(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder, freq=100.0)
        f = folder.toolfolder.get("Spike_0")
        count_first = int(f.data.get("SP_count").nparray[0])
        # Run again with different freq — different spike count
        tool2 = NMToolSpike()
        self._run_once(tool2, folder, freq=200.0)
        count_second = int(f.data.get("SP_count").nparray[0])
        self.assertNotEqual(count_first, count_second)

    def test_overwrite_false_creates_new_subfolder_each_run(self):
        tool = NMToolSpike()
        tool.overwrite = False
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder)
        tool2 = NMToolSpike()
        tool2.overwrite = False
        self._run_once(tool2, folder)
        self.assertIn("Spike_0", folder.toolfolder)
        self.assertIn("Spike_1", folder.toolfolder)

    def test_config_overwrite_default(self):
        cfg = NMToolSpikeConfig()
        self.assertTrue(cfg.overwrite)

    def test_config_overwrite_set_false(self):
        cfg = NMToolSpikeConfig()
        cfg.overwrite = False
        self.assertFalse(cfg.overwrite)

    def test_config_overwrite_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMToolSpikeConfig().overwrite = 1


class TestNMToolSpikeResultsToCache(unittest.TestCase):
    """results_to_cache saves spike times to folder.toolresults."""

    def test_cache_saved(self):
        tool = NMToolSpike()
        d = _sine_data()
        folder = _run(tool, [d])
        self.assertIn("spike", folder.toolresults)

    def test_cache_disabled(self):
        tool = NMToolSpike()
        tool.results_to_cache = False
        d = _sine_data()
        folder = _run(tool, [d])
        self.assertNotIn("spike", folder.toolresults)


class TestNMToolSpikeNotes(unittest.TestCase):
    """Notes are written to SP_ output arrays."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.tool.ylevel = 0.0
        self.tool.func_name = "level+"
        data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(2)]
        self.folder = _run(self.tool, data)
        self.f = self.folder.toolfolder.get("Spike_0")

    def test_sp_epoch_note_contains_nmspike(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("NMSpike", note)

    def test_sp_epoch_note_contains_source(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("recA0", note)

    def test_sp_epoch_note_contains_ylevel(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("ylevel=", note)

    def test_sp_epoch_note_contains_func_name(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("func_name=", note)

    def test_sp_epoch_note_contains_spike_count(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("n=", note)

    def test_sp_epoch_note_contains_x0_x1(self):
        note = self.f.data.get("SP_recA0").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)

    def test_sp_count_note_contains_x0_x1(self):
        note = self.f.data.get("SP_count").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)

    def test_sp_count_note_contains_n_epochs(self):
        note = self.f.data.get("SP_count").notes.note
        self.assertIn("n_epochs=2", note)

    def test_pst_note_contains_nmspike_pst(self):
        self.tool.pst(bins=50)
        note = self.f.data.get("SP_PST").notes.note
        self.assertIn("NMSpike.pst", note)

    def test_pst_note_contains_bins(self):
        self.tool.pst(bins=50)
        note = self.f.data.get("SP_PST").notes.note
        self.assertIn("bins=50", note)

    def test_pst_note_contains_n_spikes(self):
        self.tool.pst(bins=50)
        note = self.f.data.get("SP_PST").notes.note
        self.assertIn("n_spikes=", note)

    def test_pst_note_contains_output_mode(self):
        self.tool.pst(bins=50, output_mode="rate")
        note = self.f.data.get("SP_PST").notes.note
        self.assertIn("output_mode=", note)
        self.assertIn("rate", note)

    def test_pst_note_contains_x0_x1(self):
        self.tool.pst(bins=50, x0=0.0, x1=0.05)
        note = self.f.data.get("SP_PST").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)

    def test_isi_note_contains_nmspike_isi(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI").notes.note
        self.assertIn("NMSpike.isi", note)

    def test_isi_note_contains_bins(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI").notes.note
        self.assertIn("bins=50", note)

    def test_isi_note_contains_n_intervals(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI").notes.note
        self.assertIn("n_intervals=", note)


class TestNMToolSpikeRaster(unittest.TestCase):
    """raster() convenience method."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        _run(self.tool, self.data)

    def test_returns_tuple_of_two_lists(self):
        result = self.tool.raster()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_times_list_length_equals_epoch_count(self):
        times_list, _ = self.tool.raster()
        self.assertEqual(len(times_list), 3)

    def test_labels_list_length_equals_epoch_count(self):
        _, labels = self.tool.raster()
        self.assertEqual(len(labels), 3)

    def test_labels_match_epoch_names(self):
        _, labels = self.tool.raster()
        self.assertEqual(labels, ["recA0", "recA1", "recA2"])

    def test_times_are_numpy_arrays(self):
        times_list, _ = self.tool.raster()
        for times in times_list:
            self.assertIsInstance(times, np.ndarray)

    def test_times_match_internal_spike_times(self):
        times_list, _ = self.tool.raster()
        for a, b in zip(times_list, self.tool._spike_times):
            np.testing.assert_array_equal(a, b)

    def test_raises_before_run(self):
        tool2 = NMToolSpike()
        with self.assertRaises(RuntimeError):
            tool2.raster()


class TestNMToolSpikeConfig(unittest.TestCase):
    """NMToolSpikeConfig schema, defaults, validation, and TOML round-trip."""

    def setUp(self):
        self.cfg = NMToolSpikeConfig()

    def test_defaults(self):
        self.assertEqual(self.cfg.ylevel, 0.0)
        self.assertEqual(self.cfg.func_name, "level+")
        self.assertFalse(self.cfg.results_to_history)
        self.assertTrue(self.cfg.results_to_cache)
        self.assertTrue(self.cfg.results_to_numpy)

    def test_ylevel_set(self):
        self.cfg.ylevel = -10.0
        self.assertEqual(self.cfg.ylevel, -10.0)

    def test_ylevel_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.cfg.ylevel = True

    def test_func_name_accepts_valid(self):
        for v in ("level", "level+", "level-"):
            self.cfg.func_name = v
            self.assertEqual(self.cfg.func_name, v)

    def test_func_name_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self.cfg.func_name = "peak"

    def test_results_to_history_set(self):
        self.cfg.results_to_history = True
        self.assertTrue(self.cfg.results_to_history)

    def test_results_to_history_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.cfg.results_to_history = 1

    def test_results_to_cache_set(self):
        self.cfg.results_to_cache = False
        self.assertFalse(self.cfg.results_to_cache)

    def test_results_to_numpy_set(self):
        self.cfg.results_to_numpy = False
        self.assertFalse(self.cfg.results_to_numpy)

    def test_unknown_key_raises(self):
        with self.assertRaises(AttributeError):
            self.cfg.nonexistent = 1

    def test_to_dict_contains_all_keys(self):
        d = self.cfg.to_dict()
        for key in ("ylevel", "func_name", "overwrite", "results_to_history",
                    "results_to_cache", "results_to_numpy"):
            self.assertIn(key, d)

    def test_to_dict_type_header(self):
        d = self.cfg.to_dict()
        self.assertEqual(d["pyneuromatic"]["type"], "spike_config")

    def test_from_dict_round_trip(self):
        self.cfg.ylevel = -5.0
        self.cfg.func_name = "level-"
        self.cfg.results_to_history = True
        d = self.cfg.to_dict()
        cfg2 = NMToolSpikeConfig.from_dict(d)
        self.assertEqual(cfg2, self.cfg)

    def test_tool_has_config(self):
        tool = NMToolSpike()
        self.assertIsInstance(tool.config, NMToolSpikeConfig)


class TestNMToolSpikeRegistry(unittest.TestCase):
    """spike is registered in the tool registry."""

    def test_registry_returns_nmtoolspike(self):
        from pyneuromatic.core.nm_tool_registry import get_global_registry, reset_global_registry
        reset_global_registry()
        registry = get_global_registry()
        tool = registry.load("spike")
        self.assertIsInstance(tool, NMToolSpike)


if __name__ == "__main__":
    unittest.main()
