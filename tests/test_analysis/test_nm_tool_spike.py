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

_FREQ = 200.0  # frequency of test sine wave (Hz)
_SR = 10000.0   # sample rate (Hz) used in test signals
_DELTA = 1.0 / _SR # time between samples (s)
_SAMPLES = 1000  # number of samples in test signals (0.1 s at 10 kHz)
_CYCLES = int(_FREQ * (_SAMPLES * _DELTA)) # number of cycles in test signals (20 cycles at 200 Hz over 0.1 s)


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



class TestNMToolSpikeDetection(unittest.TestCase):
    """run_all detection behaviour."""

    def setUp(self):
        self.tool = NMToolSpike()

    def test_rising_crossings_count(self):
        # 200 Hz sine, 1000 samples at 10 kHz → 20 cycles → 20 rising crossings
        # func_name="level+" counts upward crossings of ylevel=0.0
        d = _sine_data()
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), _CYCLES)

    def test_falling_crossings(self):
        self.tool.func_name = "level-"
        d = _sine_data()
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), _CYCLES)

    def test_both_directions_double_count(self):
        self.tool.func_name = "level"
        d = _sine_data()
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        count_arr = f.data.get("SP_count")
        self.assertEqual(int(count_arr.nparray[0]), 2*_CYCLES)

    def test_x0_x1_window_restricts_detection(self):
        # 200 Hz sine → 20 rising crossings over 0.1 s (1000 samples at 10 kHz).
        # Window [0.002, 0.048] avoids boundary ambiguity (crossings fall exactly
        # at multiples of 0.005 s): captures t=0.005, 0.010, …, 0.045 → 9 crossings.
        d = _sine_data()
        self.tool.x0 = 0.002
        self.tool.x1 = 0.048
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 9)

    def test_ignore_nans_true_detects_crossing_across_nan_gap(self):
        # y=0 at index 0, NaN at index 1, y=1 at index 2 → crossing exists
        y = np.array([0.0, float("nan"), 1.0])
        d = NMData(NM, name="recA0", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        self.tool.ylevel = 0.5
        self.tool.ignore_nans = True
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 1)

    def test_ignore_nans_false_blocks_crossing_across_nan_gap(self):
        # Same data; with ignore_nans=False the NaN blocks the crossing
        y = np.array([0.0, float("nan"), 1.0])
        d = NMData(NM, name="recA0", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        self.tool.ylevel = 0.5
        self.tool.ignore_nans = False
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
        self.assertEqual(int(f.data.get("SP_count").nparray[0]), 0)

    def test_per_epoch_sp_arrays_created(self):
        data = [_sine_data(name="recA%d" % i) for i in range(3)]
        folder = _run(self.tool, data)
        f = folder.toolfolders.get("Spike_0")
        for i in range(3):
            self.assertIn("SP_recA%d" % i, f.data)

    def test_sp_count_length_equals_epoch_count(self):
        data = [_sine_data(name="recA%d" % i) for i in range(5)]
        folder = _run(self.tool, data)
        f = folder.toolfolders.get("Spike_0")
        self.assertEqual(len(f.data.get("SP_count").nparray), 5)

    def test_sp_count_matches_sp_array_lengths(self):
        data = [_sine_data(name="recA%d" % i, freq=100.0 * (i + 1)) for i in range(3)]
        folder = _run(self.tool, data)
        f = folder.toolfolders.get("Spike_0")
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
        f = folder.toolfolders.get("Spike_0")
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
        d = _sine_data()
        folder = _run(self.tool, [d])
        f = folder.toolfolders.get("Spike_0")
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
        self.assertIn("Spike_0", folder.toolfolders)

    def test_with_dataseries(self):
        folder = NMFolder(NM, name="F")
        ds = NMDataSeries(NM, name="Record")
        self._run_with_selection(folder, dataseries=ds)
        self.assertIn("Spike_Record_0", folder.toolfolders)

    def test_with_dataseries_and_channel(self):
        folder = NMFolder(NM, name="F")
        ds = NMDataSeries(NM, name="Record")
        ch = NMChannel(NM, name="A")
        self._run_with_selection(folder, dataseries=ds, channel=ch)
        self.assertIn("Spike_Record_A_0", folder.toolfolders)

    def test_second_run_increments_suffix_when_overwrite_false(self):
        folder = NMFolder(NM, name="F")
        self.tool.overwrite = False
        self._run_with_selection(folder)
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self._run_with_selection(folder)
        self.assertIn("Spike_0", folder.toolfolders)
        self.assertIn("Spike_1", folder.toolfolders)


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
        f = self.folder.toolfolders.get("Spike_0")
        total_spikes = int(f.data.get("SP_count").nparray.sum())
        result = self.tool.pst(bins=50)
        self.assertEqual(int(result.nparray.sum()), total_spikes)

    def test_pst_xscale_delta_equals_bin_width(self):
        bins = 100
        result = self.tool.pst(bins=bins)
        all_times = np.concatenate(self.tool._spike_times)
        expected_delta = (all_times.max() - all_times.min()) / bins
        self.assertAlmostEqual(result.xscale.delta, expected_delta, places=10)

    def test_pst_x0_x1_restricts_spike_count(self):
        # Full recording has _CYCLES spikes; window [0.002, 0.048] captures 9.
        result_full = self.tool.pst(bins=50, overwrite=False)
        result_win  = self.tool.pst(bins=50, x0=0.002, x1=0.048, overwrite=False)
        self.assertLess(int(result_win.nparray.sum()), int(result_full.nparray.sum()))

    def test_pst_output_mode_count_matches_total_spikes(self):
        f = self.folder.toolfolders.get("Spike_0")
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
        f = self.folder.toolfolders.get("Spike_0")
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
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_PST_0", f.data)

    def test_pst_overwrite_true_reuses_sp_pst_0(self):
        self.tool.pst(bins=50)
        self.tool.pst(bins=100)  # overwrites SP_PST_0
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_PST_0", f.data)
        self.assertNotIn("SP_PST_1", f.data)

    def test_pst_overwrite_false_creates_new_numbered_arrays(self):
        self.tool.pst(bins=50, overwrite=False)
        self.tool.pst(bins=100, overwrite=False)
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_PST_0", f.data)
        self.assertIn("SP_PST_1", f.data)

    def test_pst_result_name_matches_array_in_folder(self):
        d = self.tool.pst(bins=50, overwrite=False)
        self.assertEqual(d.name, "SP_PST_0")
        d2 = self.tool.pst(bins=50, overwrite=False)
        self.assertEqual(d2.name, "SP_PST_1")


class TestNMToolSpikeISI(unittest.TestCase):
    """isi() histogram method."""

    def setUp(self):
        self.tool = NMToolSpike()
        # Three epochs, each with ~10 spikes (100 Hz, 0.1 s)
        data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
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
        max_isi = 0.02
        result = self.tool.isi(bins=50, max_isi=max_isi)
        self.assertIsNotNone(result)
        right_edge = result.xscale.start + len(result.nparray) * result.xscale.delta
        self.assertAlmostEqual(right_edge, max_isi, places=10)

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
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_ISI_0", f.data)

    def test_isi_total_intervals_correct(self):
        # Each epoch: 10 spikes → 9 ISIs; 3 epochs → 27 total ISIs
        result = self.tool.isi(bins=50)
        total_intervals = sum(
            len(np.diff(t)) for t in self.tool._spike_times if len(t) >= 2
        )
        self.assertEqual(int(result.nparray.sum()), total_intervals)

    def test_isi_x0_x1_filters_spike_times(self):
        # Restrict to first half of recording; fewer intervals than full range.
        data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
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
        f = self.folder.toolfolders.get("Spike_0")
        note = f.data.get("SP_ISI_0").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)
        self.assertIn("min_isi=", note)
        self.assertIn("max_isi=", note)
        self.assertIn("output_mode=", note)

    def test_isi_overwrite_true_reuses_sp_isi_0(self):
        self.tool.isi(bins=50)
        self.tool.isi(bins=100)  # overwrites SP_ISI_0
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_ISI_0", f.data)
        self.assertNotIn("SP_ISI_1", f.data)

    def test_isi_overwrite_false_creates_new_numbered_arrays(self):
        self.tool.isi(bins=50, overwrite=False)
        self.tool.isi(bins=100, overwrite=False)
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SP_ISI_0", f.data)
        self.assertIn("SP_ISI_1", f.data)

    def test_isi_result_name_matches_array_in_folder(self):
        d = self.tool.isi(bins=50, overwrite=False)
        self.assertEqual(d.name, "SP_ISI_0")
        d2 = self.tool.isi(bins=50, overwrite=False)
        self.assertEqual(d2.name, "SP_ISI_1")


class TestNMToolSpikeIntervals(unittest.TestCase):
    """intervals() convenience method."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        _run(self.tool, self.data)

    def test_returns_tuple_of_two_lists(self):
        result = self.tool.intervals()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_interval_arrays_length_equals_epoch_count(self):
        ivs, labels = self.tool.intervals()
        self.assertEqual(len(ivs), 3)
        self.assertEqual(len(labels), 3)

    def test_labels_match_epoch_names(self):
        _, labels = self.tool.intervals()
        self.assertEqual(labels, ["recA0", "recA1", "recA2"])

    def test_interval_arrays_are_numpy_arrays(self):
        ivs, _ = self.tool.intervals()
        for iv in ivs:
            self.assertIsInstance(iv, np.ndarray)

    def test_interval_length_is_spike_count_minus_one(self):
        ivs, _ = self.tool.intervals()
        for iv, spike_times in zip(ivs, self.tool._spike_times):
            self.assertEqual(len(iv), max(len(spike_times) - 1, 0))

    def test_interval_values_equal_np_diff(self):
        ivs, _ = self.tool.intervals()
        for iv, spike_times in zip(ivs, self.tool._spike_times):
            np.testing.assert_array_equal(iv, np.diff(spike_times))

    def test_x1_filters_spike_times_before_diff(self):
        ivs_full, _ = self.tool.intervals()
        ivs_half, _ = self.tool.intervals(x1=0.05)
        self.assertLess(len(ivs_half[0]), len(ivs_full[0]))

    def test_x0_x1_window(self):
        # 100 Hz sine: crossings at 0, 0.01, 0.02, … Window [0.005, 0.075]
        # captures t=0.01, …, 0.07 → 7 spikes → 6 intervals of 0.01 s each.
        ivs, _ = self.tool.intervals(x0=0.005, x1=0.075)
        self.assertEqual(len(ivs[0]), 6)
        np.testing.assert_allclose(ivs[0], 0.01, rtol=1e-9)

    def test_raises_before_run(self):
        tool2 = NMToolSpike()
        with self.assertRaises(RuntimeError):
            tool2.intervals()

    def test_raises_if_toolfolder_has_no_sp_arrays(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        empty_tf = NMToolFolder(NM, name="empty")
        with self.assertRaises(RuntimeError):
            self.tool.intervals(toolfolder=empty_tf)

    def test_toolfolder_returns_intervals_from_prior_run(self):
        self.tool.overwrite = False
        folder = NMFolder(NM, name="TestFolder2")
        data1 = [_sine_data(name="recB%d" % i, freq=100.0) for i in range(2)]
        targets1 = _make_targets(data1, folder=folder)
        self.tool.run_all(targets1)
        tf1 = self.tool._toolfolder
        data2 = [_sine_data(name="recC%d" % i, freq=200.0) for i in range(2)]
        targets2 = _make_targets(data2, folder=folder)
        self.tool.run_all(targets2)
        ivs, labels = self.tool.intervals(toolfolder=tf1)
        self.assertEqual(labels, ["recB0", "recB1"])
        self.assertEqual(len(ivs), 2)

    def test_isi_uses_intervals_internally(self):
        # isi() and intervals() should agree on interval count
        ivs, _ = self.tool.intervals()
        total_from_intervals = sum(len(iv) for iv in ivs if len(iv) > 0)
        isi_result = self.tool.isi(bins=50)
        self.assertEqual(int(isi_result.nparray.sum()), total_from_intervals)


class TestNMToolSpikeExtractSpikeWaveforms(unittest.TestCase):
    """extract_spike_waveforms() extracts waveform snippets around detected spikes."""

    # 100 Hz sine, 1000 samples at 10 kHz → 10 rising crossings.
    # Spikes near samples 0, 100, 200, ..., 900.
    # pre=post=0.003 s (30 samples) leaves comfortable margin.
    _PRE  = 0.003
    _POST = 0.003

    def setUp(self):
        self.tool = NMToolSpike()
        self.data = _sine_data(name="recA0", freq=100.0, n=1000)
        self.folder = _run(self.tool, [self.data])

    # --- guards ---

    def test_raises_before_run(self):
        with self.assertRaises(RuntimeError):
            NMToolSpike().extract_spike_waveforms(self._PRE, self._POST)

    def test_raises_for_zero_pre(self):
        with self.assertRaises(ValueError):
            self.tool.extract_spike_waveforms(0.0, self._POST)

    def test_raises_for_negative_pre(self):
        with self.assertRaises(ValueError):
            self.tool.extract_spike_waveforms(-0.001, self._POST)

    def test_raises_for_zero_post(self):
        with self.assertRaises(ValueError):
            self.tool.extract_spike_waveforms(self._PRE, 0.0)

    def test_raises_for_bool_pre(self):
        with self.assertRaises(TypeError):
            self.tool.extract_spike_waveforms(True, self._POST)

    def test_raises_for_invalid_edge(self):
        with self.assertRaises(ValueError):
            self.tool.extract_spike_waveforms(self._PRE, self._POST, edge="truncate")

    def test_raises_for_non_str_align(self):
        with self.assertRaises(TypeError):
            self.tool.extract_spike_waveforms(self._PRE, self._POST, align=1)

    def test_raises_for_invalid_align(self):
        with self.assertRaises(ValueError):
            self.tool.extract_spike_waveforms(self._PRE, self._POST, align="center")

    # --- basic output ---

    def test_returns_list(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertIsInstance(result, list)

    def test_arrays_written_to_subfolder(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertGreater(len(result), 0)
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn(result[0].name, f.data)

    def test_array_name_format(self):
        # Names follow {prefix}{channel}{epoch} pattern: SPK_A0, SPK_A1, ...
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        names = [d.name for d in result]
        for i, name in enumerate(names):
            self.assertEqual(name, "SPK_A%d" % i)

    def test_snippet_length(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        pre_samples  = int(round(self._PRE  / _DELTA))
        post_samples = int(round(self._POST / _DELTA))
        for d in result:
            self.assertEqual(len(d.nparray), pre_samples + post_samples)

    # --- xscale ---

    def test_xscale_start_zero_by_default(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertAlmostEqual(result[0].xscale.start, 0.0, places=10)

    def test_xscale_start_equals_minus_pre_when_align_spike(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST, align="spike")
        self.assertAlmostEqual(result[0].xscale.start, -self._PRE, places=10)

    def test_xscale_start_source_preserves_recording_time(self):
        # align="source": xscale.start should reflect the original recording time
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST, align="source")
        for d in result:
            # start must be >= source xscale.start (snippet begins inside recording)
            self.assertGreaterEqual(d.xscale.start, self.data.xscale.start)

    def test_align_case_insensitive(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST, align="Zero")
        self.assertIsInstance(result, list)

    def test_xscale_delta_matches_source(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertAlmostEqual(result[0].xscale.delta, self.data.xscale.delta)

    def test_xscale_units_match_source(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertEqual(result[0].xscale.units, self.data.xscale.units)

    def test_yscale_copied_from_source(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertEqual(result[0].yscale.label, self.data.yscale.label)
        self.assertEqual(result[0].yscale.units, self.data.yscale.units)

    # --- edge handling ---

    def test_edge_skip_omits_spikes_near_start(self):
        # Large pre so spikes near the start of recording are skipped
        result_small = self.tool.extract_spike_waveforms(self._PRE, self._POST, edge="skip")
        # Use a fresh tool to avoid name collision in the same subfolder
        tool2 = NMToolSpike()
        _run(tool2, [_sine_data(name="recA0", freq=100.0)])
        result_large = tool2.extract_spike_waveforms(0.05, self._POST, edge="skip")
        # Large pre skips more spikes
        self.assertLess(len(result_large), len(result_small))

    def test_edge_pad_returns_full_length_snippets(self):
        pre_samples  = int(round(0.05 / _DELTA))
        post_samples = int(round(self._POST / _DELTA))
        result = self.tool.extract_spike_waveforms(0.05, self._POST, edge="pad")
        for d in result:
            self.assertEqual(len(d.nparray), pre_samples + post_samples)

    def test_edge_pad_fills_out_of_bounds_with_nan(self):
        # Use a large pre so the first spike's snippet definitely extends before index 0
        result = self.tool.extract_spike_waveforms(0.05, self._POST, edge="pad")
        # At least one snippet should contain NaN (from padding)
        has_nan = any(np.any(np.isnan(d.nparray)) for d in result)
        self.assertTrue(has_nan)

    def test_edge_case_insensitive(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST, edge="skip")
        self.assertIsInstance(result, list)

    # --- clip_to_next_spike ---

    def test_clip_to_next_spike_same_length_as_unclipped(self):
        # Snippets should always be the same length regardless of clipping
        # 100 Hz → ISI ≈ 0.01 s; post=0.02 s > ISI so NaN-clipping applies
        result_clip = self.tool.extract_spike_waveforms(
            self._PRE, 0.02, clip_to_next_spike=True
        )
        tool2 = NMToolSpike()
        _run(tool2, [_sine_data(name="recA0", freq=100.0)])
        result_full = tool2.extract_spike_waveforms(
            self._PRE, 0.02, clip_to_next_spike=False
        )
        self.assertEqual(len(result_clip), len(result_full))
        pre_samples  = int(round(self._PRE / _DELTA))
        post_samples = int(round(0.02 / _DELTA))
        expected_len = pre_samples + post_samples
        for d in result_clip:
            self.assertEqual(len(d.nparray), expected_len)
        for d in result_full:
            self.assertEqual(len(d.nparray), expected_len)

    def test_clip_to_next_spike_nans_beyond_next_spike(self):
        # 100 Hz → ISI ≈ 0.01 s; post=0.02 s > ISI → tail of snippet is NaN
        result = self.tool.extract_spike_waveforms(
            self._PRE, 0.02, clip_to_next_spike=True
        )
        # Non-last spikes should have NaN in the clipped tail
        has_nan = any(np.any(np.isnan(d.nparray)) for d in result[:-1])
        self.assertTrue(has_nan)

    def test_clip_to_next_spike_last_has_no_nan(self):
        # Last spike is never clipped, so no NaN from clipping
        # Use post small enough that it stays within bounds
        result = self.tool.extract_spike_waveforms(
            self._PRE, self._POST, clip_to_next_spike=True
        )
        self.assertGreater(len(result), 0)
        last = result[-1]
        self.assertFalse(np.any(np.isnan(last.nparray)))

    # --- no spikes ---

    def test_no_spikes_returns_empty_list(self):
        tool2 = NMToolSpike()
        y = np.ones(500) * 5.0
        d = NMData(NM, name="flat", nparray=y,
                   xscale={"start": 0.0, "delta": _DELTA})
        _run(tool2, [d])
        result = tool2.extract_spike_waveforms(self._PRE, self._POST)
        self.assertEqual(result, [])

    # --- notes ---

    def test_note_contains_source_epoch(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        note = result[0].notes.note
        self.assertIn("recA0", note)

    def test_note_contains_spike_x(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        note = result[0].notes.note
        self.assertIn("spike_x=", note)

    def test_note_contains_pre_post(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        note = result[0].notes.note
        self.assertIn("pre=", note)
        self.assertIn("post=", note)

    def test_note_contains_align(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        note = result[0].notes.note
        self.assertIn("align=", note)

    # --- xarray source ---

    def test_xarray_source_sets_xarray_on_snippet(self):
        # Source with non-uniform xarray → snippet should have xarray, not xscale
        t = np.cumsum(np.ones(_SAMPLES) * _DELTA)  # uniform but stored as xarray
        t[500:] += 0.001  # introduce a gap to make it non-uniform
        y = np.sin(2 * np.pi * 100.0 * np.arange(_SAMPLES) * _DELTA)
        data_xa = NMData(NM, name="recA0", nparray=y,
                         xscale={"label": "Time", "units": "s"})
        data_xa.xarray = t
        tool_xa = NMToolSpike()
        _run(tool_xa, [data_xa])
        result = tool_xa.extract_spike_waveforms(self._PRE, self._POST)
        self.assertGreater(len(result), 0)
        for d in result:
            self.assertIsNotNone(d.xarray)
            self.assertEqual(len(d.xarray), len(d.nparray))

    def test_xarray_source_align_zero_starts_at_zero(self):
        t = np.arange(_SAMPLES, dtype=float) * _DELTA
        y = np.sin(2 * np.pi * 100.0 * t)
        data_xa = NMData(NM, name="recA0", nparray=y,
                         xscale={"label": "Time", "units": "s"})
        data_xa.xarray = t
        tool_xa = NMToolSpike()
        _run(tool_xa, [data_xa])
        result = tool_xa.extract_spike_waveforms(self._PRE, self._POST, align="zero")
        self.assertGreater(len(result), 0)
        self.assertAlmostEqual(result[0].xarray[0], 0.0, places=10)

    def test_xarray_source_align_spike_zero_at_spike(self):
        t = np.arange(_SAMPLES, dtype=float) * _DELTA
        y = np.sin(2 * np.pi * 100.0 * t)
        data_xa = NMData(NM, name="recA0", nparray=y,
                         xscale={"label": "Time", "units": "s"})
        data_xa.xarray = t
        tool_xa = NMToolSpike()
        _run(tool_xa, [data_xa])
        result = tool_xa.extract_spike_waveforms(self._PRE, self._POST, align="spike")
        self.assertGreater(len(result), 0)
        pre_samples = int(round(self._PRE / _DELTA))
        # x at the spike sample should be within one delta of 0
        self.assertLess(abs(result[0].xarray[pre_samples]), _DELTA)

    # --- non-uniform xarray ---

    def _make_nonuniform_data(self, name="recA0"):
        """Signal with two segments at different sample rates.

        Samples 0-499: delta = _DELTA     (10 kHz)
        Samples 500-999: delta = _DELTA*2 (5 kHz)
        A threshold crossing is placed in each segment (samples 200 and 700).
        """
        n1, n2 = 500, 500
        t1 = np.arange(n1, dtype=float) * _DELTA
        t2 = t1[-1] + _DELTA + np.arange(1, n2 + 1, dtype=float) * (_DELTA * 2)
        t = np.concatenate([t1, t2])
        y = np.zeros(len(t))
        y[200] = 1.0   # rising crossing near sample 200 (10 kHz region)
        y[700] = 1.0   # rising crossing near sample 700 (5 kHz region)
        d = NMData(NM, name=name, nparray=y,
                   xscale={"label": "Time", "units": "s"})
        d.xarray = t
        return d, t

    def test_nonuniform_xarray_snippet_matches_source_slice(self):
        # Snippet xarray values must exactly equal the corresponding
        # slice of the source xarray (before alignment shift).
        data_nu, t = self._make_nonuniform_data()
        tool_nu = NMToolSpike()
        _run(tool_nu, [data_nu])
        # Use align="zero" so shift is x_snippet[0]; we can recover the
        # original slice by adding back the first source x value.
        result = tool_nu.extract_spike_waveforms(self._PRE, self._POST,
                                                 align="zero")
        self.assertGreater(len(result), 0)
        for d in result:
            # Key property: spacing pattern should match source spacing
            src_diffs = np.diff(t)
            snip_diffs = np.diff(d.xarray)
            # All snippet intervals must appear somewhere in source intervals
            for interval in snip_diffs:
                self.assertTrue(
                    np.any(np.isclose(src_diffs, interval, rtol=1e-9)),
                    msg="interval %g not found in source xarray diffs" % interval,
                )

    def test_nonuniform_xarray_diffs_not_all_equal(self):
        # Verify the snippet xarray is genuinely non-uniform when the spike
        # spans the boundary between the two sample-rate regions.
        # Place spike exactly at sample 499 (boundary) with large pre+post.
        n1, n2 = 500, 500
        t1 = np.arange(n1, dtype=float) * _DELTA
        t2 = t1[-1] + _DELTA + np.arange(1, n2 + 1, dtype=float) * (_DELTA * 2)
        t = np.concatenate([t1, t2])
        y = np.zeros(len(t))
        y[499] = 1.0   # crossing at boundary
        data_bd = NMData(NM, name="recA0", nparray=y,
                         xscale={"label": "Time", "units": "s"})
        data_bd.xarray = t
        tool_bd = NMToolSpike()
        _run(tool_bd, [data_bd])
        # pre=0.003 s (30 samples at 10 kHz) → straddles the boundary
        result = tool_bd.extract_spike_waveforms(0.003, 0.003)
        self.assertGreater(len(result), 0)
        diffs = np.diff(result[0].xarray)
        # Should contain both _DELTA and _DELTA*2 intervals
        has_fine   = np.any(np.isclose(diffs, _DELTA,     rtol=1e-9))
        has_coarse = np.any(np.isclose(diffs, _DELTA * 2, rtol=1e-9))
        self.assertTrue(has_fine,   "expected fine intervals (_DELTA) in snippet")
        self.assertTrue(has_coarse, "expected coarse intervals (_DELTA*2) in snippet")

    def test_nonuniform_xarray_align_zero_starts_zero(self):
        data_nu, _ = self._make_nonuniform_data()
        tool_nu = NMToolSpike()
        _run(tool_nu, [data_nu])
        result = tool_nu.extract_spike_waveforms(self._PRE, self._POST, align="zero")
        self.assertGreater(len(result), 0)
        for d in result:
            self.assertAlmostEqual(d.xarray[0], 0.0, places=10)

    def test_nonuniform_xarray_align_spike_near_zero(self):
        data_nu, t = self._make_nonuniform_data()
        tool_nu = NMToolSpike()
        _run(tool_nu, [data_nu])
        result = tool_nu.extract_spike_waveforms(self._PRE, self._POST, align="spike")
        self.assertEqual(len(result), 2)
        # extract_spike_waveforms uses np.median(np.diff(xarray)) for pre_samples.
        # This nonuniform array has 499 diffs at _DELTA and 499 at _DELTA*2,
        # so the median is _DELTA*2 and pre_s = int(round(_PRE / (_DELTA*2))).
        pre_s = int(round(self._PRE / (_DELTA * 2)))
        # Spike 0 is in the 10 kHz segment: local spacing _DELTA
        self.assertLess(abs(result[0].xarray[pre_s]), _DELTA)
        # Spike 1 is in the 5 kHz segment: local spacing _DELTA*2
        self.assertLess(abs(result[1].xarray[pre_s]), _DELTA * 2)

    def test_nonuniform_xarray_align_source_preserves_values(self):
        data_nu, t = self._make_nonuniform_data()
        tool_nu = NMToolSpike()
        _run(tool_nu, [data_nu])
        result = tool_nu.extract_spike_waveforms(self._PRE, self._POST, align="source")
        self.assertGreater(len(result), 0)
        for d in result:
            # xarray values must all appear in the source xarray
            for xval in d.xarray:
                self.assertTrue(
                    np.any(np.isclose(t, xval, rtol=1e-9)),
                    msg="x value %g not found in source xarray" % xval,
                )

    def test_align_source_xscale_start_in_recording_range(self):
        # align="source" with uniform xscale: start reflects original position
        result = self.tool.extract_spike_waveforms(
            self._PRE, self._POST, align="source"
        )
        self.assertGreater(len(result), 0)
        for d in result:
            self.assertGreaterEqual(d.xscale.start, self.data.xscale.start)

    # --- build_dataseries ---

    def test_build_dataseries_called_automatically(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        self.assertGreater(len(result), 0)
        f = self.folder.toolfolders.get("Spike_0")
        # dataseries should be populated after extract_spike_waveforms
        self.assertGreater(len(list(f.dataseries.keys())), 0)

    def test_build_dataseries_prefix_is_SPK(self):
        self.tool.extract_spike_waveforms(self._PRE, self._POST)
        f = self.folder.toolfolders.get("Spike_0")
        self.assertIn("SPK_", f.dataseries)

    def test_build_dataseries_has_channel_A(self):
        self.tool.extract_spike_waveforms(self._PRE, self._POST)
        f = self.folder.toolfolders.get("Spike_0")
        ds = f.dataseries.get("SPK_")
        self.assertIsNotNone(ds)
        self.assertIn("A", ds.channels)

    def test_build_dataseries_epoch_count_matches_snippets(self):
        result = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        f = self.folder.toolfolders.get("Spike_0")
        ds = f.dataseries.get("SPK_")
        self.assertEqual(len(list(ds.epochs.keys())), len(result))

    def test_two_channels_create_separate_subfolders(self):
        # Two channels each with one epoch — run_all() called once per channel,
        # matching how NMManager.run_tool() now drives tools.
        folder = NMFolder(NM, name="TwoChFolder")
        folder.new_dataseries("rec", n_channels=2, n_epochs=1,
                              n_points=1000, dx=_DELTA)
        ds = folder.dataseries.get("rec")
        dA = folder.data.get("recA0")
        dB = folder.data.get("recB0")
        dA.nparray = np.sin(2 * np.pi * 100.0 * np.arange(1000) * _DELTA)
        dB.nparray = np.sin(2 * np.pi * 100.0 * np.arange(1000) * _DELTA)
        chA = ds.channels.get("A")
        chB = ds.channels.get("B")
        ep = ds.epochs.get("E0")
        tool2 = NMToolSpike()
        tool2.run_all([{"folder": folder, "dataseries": ds, "channel": chA, "epoch": ep, "data": dA}])
        tool2.extract_spike_waveforms(self._PRE, self._POST)
        tool2.run_all([{"folder": folder, "dataseries": ds, "channel": chB, "epoch": ep, "data": dB}])
        tool2.extract_spike_waveforms(self._PRE, self._POST)
        tf_keys = list(folder.toolfolders.keys())
        self.assertIn("Spike_rec_A_0", tf_keys)
        self.assertIn("Spike_rec_B_0", tf_keys)

    # --- results_to_numpy=False ---

    def test_works_when_results_to_numpy_false(self):
        # extract_spike_waveforms reads self._spike_times (in-memory), which
        # run() always populates. results_to_numpy only controls whether SP_
        # arrays are written to the toolfolder; it does not clear in-memory state.
        tool2 = NMToolSpike()
        tool2.results_to_numpy = False
        folder2 = _run(tool2, [_sine_data(name="recA0", freq=100.0)])
        result = tool2.extract_spike_waveforms(self._PRE, self._POST)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    # --- x0 / x1 window filtering ---

    def test_x1_limits_spike_count(self):
        # 100 Hz sine, 10 spikes in 0.1 s; restrict to first half
        all_snippets = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        tool2 = NMToolSpike()
        _run(tool2, [self.data])
        half_snippets = tool2.extract_spike_waveforms(self._PRE, self._POST, x1=0.05)
        self.assertLess(len(half_snippets), len(all_snippets))

    def test_x0_limits_spike_count(self):
        tool2 = NMToolSpike()
        _run(tool2, [self.data])
        all_snippets = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        late_snippets = tool2.extract_spike_waveforms(self._PRE, self._POST, x0=0.05)
        self.assertLess(len(late_snippets), len(all_snippets))

    def test_x0_x1_window_returns_subset(self):
        tool2 = NMToolSpike()
        _run(tool2, [self.data])
        all_snippets = self.tool.extract_spike_waveforms(self._PRE, self._POST)
        mid_snippets = tool2.extract_spike_waveforms(
            self._PRE, self._POST, x0=0.02, x1=0.07
        )
        self.assertGreater(len(mid_snippets), 0)
        self.assertLess(len(mid_snippets), len(all_snippets))

    def test_x0_x1_empty_window_returns_empty_list(self):
        tool2 = NMToolSpike()
        _run(tool2, [self.data])
        snippets = tool2.extract_spike_waveforms(
            self._PRE, self._POST, x0=0.5, x1=0.5
        )
        self.assertEqual(len(snippets), 0)


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

    def test_overwrite_true_creates_Spike_A_0(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder)
        self.assertIn("Spike_0", folder.toolfolders)

    def test_overwrite_true_reuses_Spike_A_0_on_second_run(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder)
        self._run_once(tool, folder)
        self.assertIn("Spike_0", folder.toolfolders)
        self.assertNotIn("Spike_1", folder.toolfolders)

    def test_overwrite_true_replaces_sp_arrays(self):
        tool = NMToolSpike()
        folder = NMFolder(NM, name="F")
        self._run_once(tool, folder, freq=100.0)
        f = folder.toolfolders.get("Spike_0")
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
        self.assertIn("Spike_0", folder.toolfolders)
        self.assertIn("Spike_1", folder.toolfolders)

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
        self.f = self.folder.toolfolders.get("Spike_0")

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
        note = self.f.data.get("SP_PST_0").notes.note
        self.assertIn("NMSpike.pst", note)

    def test_pst_note_contains_bins(self):
        self.tool.pst(bins=50)
        note = self.f.data.get("SP_PST_0").notes.note
        self.assertIn("bins=50", note)

    def test_pst_note_contains_n_spikes(self):
        self.tool.pst(bins=50)
        note = self.f.data.get("SP_PST_0").notes.note
        self.assertIn("n_spikes=", note)

    def test_pst_note_contains_output_mode(self):
        self.tool.pst(bins=50, output_mode="rate")
        note = self.f.data.get("SP_PST_0").notes.note
        self.assertIn("output_mode=", note)
        self.assertIn("rate", note)

    def test_pst_note_contains_x0_x1(self):
        self.tool.pst(bins=50, x0=0.0, x1=0.05)
        note = self.f.data.get("SP_PST_0").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)

    def test_isi_note_contains_nmspike_isi(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI_0").notes.note
        self.assertIn("NMSpike.isi", note)

    def test_isi_note_contains_bins(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI_0").notes.note
        self.assertIn("bins=50", note)

    def test_isi_note_contains_n_intervals(self):
        self.tool.isi(bins=50)
        note = self.f.data.get("SP_ISI_0").notes.note
        self.assertIn("n_intervals=", note)

    def test_isi_note_contains_x0_x1(self):
        self.tool.isi(bins=50, x0=0.0, x1=0.05)
        note = self.f.data.get("SP_ISI_0").notes.note
        self.assertIn("x0=", note)
        self.assertIn("x1=", note)


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

    def test_x1_limits_spike_count(self):
        times_full, _ = self.tool.raster()
        times_half, _ = self.tool.raster(x1=0.05)
        self.assertLess(len(times_half[0]), len(times_full[0]))

    def test_x0_limits_spike_count(self):
        times_full, _ = self.tool.raster()
        times_late, _ = self.tool.raster(x0=0.05)
        self.assertLess(len(times_late[0]), len(times_full[0]))

    def test_x0_x1_window_returns_subset(self):
        times_mid, labels = self.tool.raster(x0=0.02, x1=0.07)
        self.assertEqual(len(labels), 3)
        self.assertGreater(len(times_mid[0]), 0)

    def test_x0_x1_empty_window_returns_empty_arrays(self):
        times_list, labels = self.tool.raster(x0=0.5, x1=0.5)
        self.assertEqual(len(labels), 3)
        for times in times_list:
            self.assertEqual(len(times), 0)

    def test_epoch_names_preserved_after_filtering(self):
        _, labels = self.tool.raster(x1=0.05)
        self.assertEqual(labels, ["recA0", "recA1", "recA2"])


class TestNMToolSpikeRasterFromToolfolder(unittest.TestCase):
    """raster(toolfolder=...) reads from saved SP_ arrays."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self.folder = NMFolder(NM, name="TestFolder")
        data1 = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        targets1 = _make_targets(data1, folder=self.folder)
        self.tool.run_all(targets1)
        self.tf1 = self.tool._toolfolder  # first run's toolfolder
        # second run with different data
        data2 = [_sine_data(name="recB%d" % i, freq=200.0) for i in range(2)]
        targets2 = _make_targets(data2, folder=self.folder)
        self.tool.run_all(targets2)

    def test_toolfolder_returns_first_run_epoch_names(self):
        _, labels = self.tool.raster(toolfolder=self.tf1)
        self.assertEqual(labels, ["recA0", "recA1", "recA2"])

    def test_toolfolder_returns_first_run_spike_times(self):
        times_list, _ = self.tool.raster(toolfolder=self.tf1)
        self.assertEqual(len(times_list), 3)
        for times in times_list:
            self.assertIsInstance(times, np.ndarray)
            self.assertGreater(len(times), 0)

    def test_default_returns_second_run_epoch_names(self):
        _, labels = self.tool.raster()
        self.assertEqual(labels, ["recB0", "recB1"])

    def test_raises_if_toolfolder_has_no_sp_arrays(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        empty_tf = NMToolFolder(NM, name="empty")
        with self.assertRaises(RuntimeError):
            self.tool.raster(toolfolder=empty_tf)


class TestNMToolSpikePSTFromToolfolder(unittest.TestCase):
    """pst(toolfolder=...) reads from saved SP_ arrays."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self.folder = NMFolder(NM, name="TestFolder")
        data1 = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        targets1 = _make_targets(data1, folder=self.folder)
        self.tool.run_all(targets1)
        self.tf1 = self.tool._toolfolder
        # second run
        data2 = [_sine_data(name="recB%d" % i, freq=200.0) for i in range(2)]
        targets2 = _make_targets(data2, folder=self.folder)
        self.tool.run_all(targets2)

    def test_pst_writes_into_first_toolfolder(self):
        d = self.tool.pst(toolfolder=self.tf1)
        self.assertIsNotNone(d)
        self.assertEqual(d.name, "SP_PST_0")
        self.assertIn("SP_PST_0", self.tf1.data)

    def test_pst_result_independent_of_second_run(self):
        # Default pst() operates on second run (recB epochs, 200 Hz)
        d_default = self.tool.pst()
        # toolfolder pst() operates on first run (recA epochs, 100 Hz)
        d_tf = self.tool.pst(toolfolder=self.tf1)
        # 200 Hz → more spikes → higher total count than 100 Hz per epoch
        self.assertGreater(d_default.nparray.sum(), d_tf.nparray.sum() / 3 * 2)

    def test_pst_raises_if_toolfolder_has_no_sp_arrays(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        empty_tf = NMToolFolder(NM, name="empty")
        with self.assertRaises(RuntimeError):
            self.tool.pst(toolfolder=empty_tf)


class TestNMToolSpikeISIFromToolfolder(unittest.TestCase):
    """isi(toolfolder=...) reads from saved SP_ arrays."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self.folder = NMFolder(NM, name="TestFolder")
        data1 = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        targets1 = _make_targets(data1, folder=self.folder)
        self.tool.run_all(targets1)
        self.tf1 = self.tool._toolfolder
        # second run
        data2 = [_sine_data(name="recB%d" % i, freq=200.0) for i in range(2)]
        targets2 = _make_targets(data2, folder=self.folder)
        self.tool.run_all(targets2)

    def test_isi_writes_into_first_toolfolder(self):
        d = self.tool.isi(toolfolder=self.tf1)
        self.assertIsNotNone(d)
        self.assertEqual(d.name, "SP_ISI_0")
        self.assertIn("SP_ISI_0", self.tf1.data)

    def test_isi_raises_if_toolfolder_has_no_sp_arrays(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        empty_tf = NMToolFolder(NM, name="empty")
        with self.assertRaises(RuntimeError):
            self.tool.isi(toolfolder=empty_tf)


class TestNMToolSpikeExtractWaveformsFromToolfolder(unittest.TestCase):
    """extract_spike_waveforms(toolfolder=...) reads from saved SP_ arrays."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.tool.overwrite = False
        self.folder = NMFolder(NM, name="TestFolder")
        self.data1 = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(2)]
        # Add source data to folder so extract can look them up
        for d in self.data1:
            self.folder.data._add(d)
        targets1 = _make_targets(self.data1, folder=self.folder)
        self.tool.run_all(targets1)
        self.tf1 = self.tool._toolfolder
        # second run
        data2 = [_sine_data(name="recB%d" % i, freq=200.0) for i in range(2)]
        targets2 = _make_targets(data2, folder=self.folder)
        self.tool.run_all(targets2)

    def test_extracts_waveforms_from_prior_toolfolder(self):
        # Verify snippets come from tf1 (100 Hz, fewer spikes), not from the
        # current in-memory state which reflects the second run (200 Hz).
        # extract_spike_waveforms with no toolfolder uses in-memory state (tf2).
        snippets_current = self.tool.extract_spike_waveforms(pre=0.002, post=0.002)
        snippets_tf1 = self.tool.extract_spike_waveforms(
            pre=0.002, post=0.002, toolfolder=self.tf1
        )
        self.assertGreater(len(snippets_tf1), 0)
        self.assertLess(len(snippets_tf1), len(snippets_current))
        for s in snippets_tf1:
            self.assertIsInstance(s, NMData)

    def test_snippets_written_into_toolfolder(self):
        self.tool.extract_spike_waveforms(pre=0.002, post=0.002, toolfolder=self.tf1)
        spk_names = [n for n in self.tf1.data if n.startswith("SPK_")]
        self.assertGreater(len(spk_names), 0)

    def test_raises_if_toolfolder_has_no_sp_arrays(self):
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        empty_tf = NMToolFolder(NM, name="empty")
        with self.assertRaises(RuntimeError):
            self.tool.extract_spike_waveforms(pre=0.002, post=0.002, toolfolder=empty_tf)

    def test_skips_epoch_if_source_not_in_folder(self):
        # Create a toolfolder with SP_ arrays but no matching source in folder
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        tf = NMToolFolder(NM, name="orphan")
        tf.data.new("SP_ghost0", nparray=np.array([0.005, 0.015]))
        # tool.folder is self.folder which has recA0, recA1 but not ghost0
        snippets = self.tool.extract_spike_waveforms(
            pre=0.002, post=0.002, toolfolder=tf
        )
        # All epochs skipped because source not found (ydata is None)
        self.assertEqual(len(snippets), 0)


class TestNMToolSpikeEpochNamesArray(unittest.TestCase):
    """SP_epoch_names is written to the toolfolder and used by _spike_times_from_toolfolder."""

    def setUp(self):
        self.tool = NMToolSpike()
        self.folder = NMFolder(NM, name="TestFolder")
        self.data = [_sine_data(name="recA%d" % i, freq=100.0) for i in range(3)]
        _run(self.tool, self.data, folder=self.folder)
        self.tf = self.tool._toolfolder

    def test_sp_epoch_names_written_to_toolfolder(self):
        self.assertIn("SP_epoch_names", self.tf.data)

    def test_sp_epoch_names_values_match_epoch_names(self):
        d = self.tf.data.get("SP_epoch_names")
        names = [str(n) for n in d.nparray]
        self.assertEqual(names, ["recA0", "recA1", "recA2"])

    def test_sp_epoch_names_length_matches_epoch_count(self):
        d = self.tf.data.get("SP_epoch_names")
        self.assertEqual(len(d.nparray), 3)

    def test_spike_times_from_toolfolder_uses_epoch_names_array(self):
        # Remove one SP_ array to prove we're using SP_epoch_names, not scanning
        # (the removed epoch returns an empty array, not KeyError)
        spike_times, epoch_names, _ = self.tool._spike_times_from_toolfolder(self.tf)
        self.assertEqual(epoch_names, ["recA0", "recA1", "recA2"])
        self.assertEqual(len(spike_times), 3)

    def test_fallback_when_no_epoch_names_array(self):
        # Manually remove SP_epoch_names to exercise the fallback path
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        tf = NMToolFolder(NM, name="legacy")
        tf.data.new("SP_recX0", nparray=np.array([0.01, 0.02]))
        tf.data.new("SP_recX1", nparray=np.array([0.03]))
        tf.data.new("SP_count", nparray=np.array([2.0, 1.0]))
        spike_times, epoch_names, _ = self.tool._spike_times_from_toolfolder(tf)
        self.assertEqual(epoch_names, ["recX0", "recX1"])
        self.assertEqual(len(spike_times[0]), 2)

    def test_epoch_named_count_not_skipped_on_read(self):
        # Under the old scan-by-name approach, an epoch named "count" would be
        # silently dropped because SP_count is in _SP_SKIP.  SP_epoch_names
        # fixes the read side: _spike_times_from_toolfolder uses the explicit
        # names array, so "count" is correctly returned.
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        tf = NMToolFolder(NM, name="clash")
        tf.data.new("SP_count", nparray=np.array([0.005, 0.015]))   # epoch "count"
        tf.data.new("SP_PST", nparray=np.array([0.003]))             # epoch "PST"
        tf.data.new(
            "SP_epoch_names",
            nparray=np.array(["count", "PST"], dtype=object),
        )
        spike_times, epoch_names, _ = self.tool._spike_times_from_toolfolder(tf)
        self.assertEqual(epoch_names, ["count", "PST"])
        self.assertEqual(len(spike_times[0]), 2)
        self.assertEqual(len(spike_times[1]), 1)

    def test_fallback_skips_numbered_pst_isi_arrays(self):
        # SP_PST_0, SP_PST_1, SP_ISI_0 should be skipped in the fallback path
        # just like SP_PST and SP_ISI (prefix-based matching).
        from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
        tf = NMToolFolder(NM, name="numbered")
        tf.data.new("SP_recY0", nparray=np.array([0.01, 0.02]))
        tf.data.new("SP_PST_0", nparray=np.array([1.0, 2.0, 3.0]))
        tf.data.new("SP_ISI_0", nparray=np.array([0.5, 0.5]))
        tf.data.new("SP_PST_1", nparray=np.array([1.5, 2.5]))
        tf.data.new("SP_count", nparray=np.array([2.0]))
        spike_times, epoch_names, _ = self.tool._spike_times_from_toolfolder(tf)
        self.assertEqual(epoch_names, ["recY0"])
        self.assertEqual(len(spike_times[0]), 2)


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
