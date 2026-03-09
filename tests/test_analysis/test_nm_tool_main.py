#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_main_op and NMToolMain.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import unittest

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.analysis.nm_main_op import (
    NMMainOp,
    NMMainOpAverage,
    NMMainOpBaseline,
    NMMainOpDeleteNaNs,
    NMMainOpDeletePoints,
    NMMainOpDifferentiate,
    NMMainOpInsertPoints,
    NMMainOpIntegrate,
    NMMainOpRedimension,
    NMMainOpReplaceValues,
    NMMainOpReverse,
    NMMainOpRotate,
    NMMainOpScale,
    op_from_name,
)
from pyneuromatic.analysis.nm_tool_main import NMToolMain
import pyneuromatic.core.nm_history as nmh

NM = NMManager(quiet=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(name, values, xstart=0.0, xdelta=1.0):
    """Return NMData with the given array values."""
    return NMData(
        NM,
        name=name,
        nparray=np.array(values, dtype=float),
        xscale={"start": xstart, "delta": xdelta, "label": "Time", "units": "ms"},
        yscale={"label": "Vm", "units": "mV"},
    )


def _make_folder_with_data(arrays_by_name):
    """Build NMFolder + direct-data target list from {name: array} dict.

    Returns (folder, targets) where targets is a list of
    {'folder': folder, 'data': nmdata} dicts.
    """
    folder = NMFolder(name="folder0")
    targets = []
    for name, arr in arrays_by_name.items():
        d = folder.data.new(
            name, nparray=np.array(arr, dtype=float)
        )
        targets.append({"folder": folder, "data": d})
    return folder, targets


def _run_op_directly(op, arrays_by_name):
    """Build folder + data_items, call op.run_all(), return folder."""
    folder = NMFolder(name="folder0")
    data_items = []
    for name, arr in arrays_by_name.items():
        d = folder.data.new(name, nparray=np.array(arr, dtype=float))
        data_items.append((d, None))   # channel_name=None → parsed from name
    op.run_all(data_items, folder)
    return folder


# ===========================================================================
# TestNMMainOpAverage
# ===========================================================================

class TestNMMainOpAverage(unittest.TestCase):
    """Test NMMainOpAverage directly (no NMToolMain machinery)."""

    def setUp(self):
        self.op = NMMainOpAverage()
        # Three A-channel waves; expected average = [4, 4, 4]
        self.arrays = {
            "RecordA0": [2.0, 2.0, 2.0],
            "RecordA1": [4.0, 4.0, 4.0],
            "RecordA2": [6.0, 6.0, 6.0],
        }

    def _run(self, arrays=None):
        if arrays is None:
            arrays = self.arrays
        return _run_op_directly(self.op, arrays)

    # --- correct values ---

    def test_correct_values(self):
        folder = self._run()
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [4.0, 4.0, 4.0])

    # --- output naming and results dict ---

    def test_output_in_folder(self):
        self._run()
        self.assertIsNotNone(self.op.results)
        self.assertIn("A", self.op.results)

    def test_output_name_in_results(self):
        self._run()
        self.assertEqual(self.op.results["A"], "Avg_RecordA")

    def test_results_populated_after_run(self):
        self._run()
        self.assertTrue(len(self.op.results) > 0)

    # --- NaN handling ---

    def test_nanmean_ignores_nan(self):
        arrays = {
            "RecordA0": [2.0, math.nan, 2.0],
            "RecordA1": [4.0, 4.0,      4.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        # nanmean of [2, nan] and [4, 4] at index 1 = mean([4]) = 4.0
        self.assertTrue(np.isfinite(out.nparray[1]))

    def test_mean_nan_propagates(self):
        self.op.ignore_nans = False
        arrays = {
            "RecordA0": [2.0, math.nan, 2.0],
            "RecordA1": [4.0, 4.0,      4.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        self.assertTrue(math.isnan(out.nparray[1]))

    # --- unequal lengths ---

    def test_unequal_lengths_truncates(self):
        arrays = {
            "RecordA0": [1.0, 2.0, 3.0],
            "RecordA1": [1.0, 2.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        self.assertEqual(len(out.nparray), 2)

    # --- two channels ---

    def test_two_channels(self):
        arrays = {
            "RecordA0": [1.0, 2.0],
            "RecordA1": [3.0, 4.0],
            "RecordB0": [5.0, 6.0],
            "RecordB1": [7.0, 8.0],
        }
        folder = self._run(arrays)
        self.assertIsNotNone(folder.data.get("Avg_RecordA"))
        self.assertIsNotNone(folder.data.get("Avg_RecordB"))

    # --- state reset ---

    def test_run_all_clears_previous_results(self):
        self._run()   # Avg_RecordA
        # Second run with different prefix — results should reflect new output name
        arrays2 = {"StimulusA0": [10.0], "StimulusA1": [20.0]}
        _run_op_directly(self.op, arrays2)
        # Channel is still "A" but prefix changed → output name differs
        self.assertEqual(self.op.results.get("A"), "Avg_StimulusA")

    # --- ignore_nans property ---

    def test_ignore_nans_default_true(self):
        self.assertTrue(self.op.ignore_nans)

    def test_ignore_nans_setter(self):
        self.op.ignore_nans = False
        self.assertFalse(self.op.ignore_nans)

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.op.ignore_nans = 1

    def test_ignore_nans_rejects_in_init(self):
        with self.assertRaises(TypeError):
            NMMainOpAverage(ignore_nans="yes")

    # --- no folder / no data ---

    def test_no_folder_is_graceful(self):
        # run_all with folder=None → no crash, no results
        data_items = [(_make_data("RecordA0", [1.0, 2.0]), None)]
        self.op.run_all(data_items, None)
        self.assertEqual(self.op.results, {})

    def test_data_without_nparray_is_skipped(self):
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0")   # no nparray
        self.op.run_all([(d, None)], folder)
        self.assertEqual(self.op.results, {})

    # --- notes ---

    def test_note_on_output_wave(self):
        # consecutive epochs → range notation
        folder = self._run({"RecordA0": [1.0, 2.0], "RecordA1": [3.0, 4.0], "RecordA2": [5.0, 6.0]})
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        self.assertEqual(len(out.notes), 1)
        note = out.notes[0]["note"]
        self.assertIn("folder=folder0", note)
        self.assertIn("channel=A", note)
        self.assertIn("epochs=0-2", note)
        self.assertIn("n_epochs=3", note)

    def test_note_non_consecutive_epochs(self):
        # non-consecutive epochs → list notation
        folder = self._run({"RecordA0": [1.0], "RecordA2": [2.0], "RecordA5": [3.0]})
        out = folder.data.get("Avg_RecordA")
        note = out.notes[0]["note"]
        self.assertIn("epochs=[0,2,5]", note)


# ===========================================================================
# TestNMMainOpScale
# ===========================================================================

class TestNMMainOpScale(unittest.TestCase):
    """Test NMMainOpScale directly."""

    def setUp(self):
        self.op = NMMainOpScale()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0])

    def _run_single(self, data=None):
        if data is None:
            data = self.data
        self.op.run(data)

    # --- factor property ---

    def test_factor_default(self):
        self.assertEqual(self.op.factor, 1.0)

    def test_factor_setter(self):
        self.op.factor = 2.5
        self.assertEqual(self.op.factor, 2.5)

    def test_factor_accepts_int(self):
        self.op.factor = 3
        self.assertEqual(self.op.factor, 3.0)
        self.assertIsInstance(self.op.factor, float)

    def test_factor_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.factor = True

    def test_factor_rejects_str(self):
        with self.assertRaises(TypeError):
            self.op.factor = "2.0"

    # --- scaling values ---

    def test_scale_by_2(self):
        self.op.factor = 2.0
        self._run_single()
        np.testing.assert_array_almost_equal(self.data.nparray, [2.0, 4.0, 6.0])

    def test_scale_by_1_no_change(self):
        self.op.factor = 1.0
        self._run_single()
        np.testing.assert_array_almost_equal(self.data.nparray, [1.0, 2.0, 3.0])

    def test_scale_by_0(self):
        self.op.factor = 0.0
        self._run_single()
        np.testing.assert_array_almost_equal(self.data.nparray, [0.0, 0.0, 0.0])

    def test_scale_by_negative(self):
        self.op.factor = -1.0
        self._run_single()
        np.testing.assert_array_almost_equal(self.data.nparray, [-1.0, -2.0, -3.0])

    def test_scale_modifies_nparray_in_place(self):
        original_obj = self.data
        self.op.factor = 2.0
        self._run_single()
        # Same NMData object, modified array
        self.assertIs(self.data, original_obj)
        self.assertEqual(self.data.nparray[0], 2.0)

    # --- skip if no nparray ---

    def test_data_without_nparray_is_skipped(self):
        d = NMData(NM, name="RecordA0")   # nparray=None
        self.op.factor = 2.0
        self.op.run(d)   # should not raise

    # --- notes ---

    def test_note_written(self):
        self.op.factor = 2.0
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        self.assertIn("NMScale(factor=2)", self.data.notes[0]["note"])


# ===========================================================================
# TestNMMainOpRedimension
# ===========================================================================

class TestNMMainOpRedimension(unittest.TestCase):
    """Test NMMainOpRedimension directly."""

    def setUp(self):
        self.op = NMMainOpRedimension()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0, 5.0])

    def _run(self):
        self.op.run(self.data)

    # --- truncate ---

    def test_truncate(self):
        self.op.n_points = 3
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0])

    # --- extend ---

    def test_extend_with_zeros(self):
        self.op.n_points = 7
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0])

    def test_extend_with_fill(self):
        self.op.n_points = 7
        self.op.fill = 9.0
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0])

    # --- edge cases ---

    def test_noop_when_n_points_zero(self):
        self.op.n_points = 0
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_same_length_no_change(self):
        self.op.n_points = 5
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        self.op.n_points = 3
        self.op.run(d)   # should not raise

    # --- validation ---

    def test_n_points_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.n_points = True

    def test_n_points_rejects_float(self):
        with self.assertRaises(TypeError):
            self.op.n_points = 3.0

    def test_n_points_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.op.n_points = -1

    def test_fill_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.fill = True

    def test_fill_accepts_int(self):
        self.op.fill = 2
        self.assertEqual(self.op.fill, 2.0)
        self.assertIsInstance(self.op.fill, float)

    # --- notes ---

    def test_note_truncate(self):
        self.op.n_points = 3
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        note = self.data.notes[0]["note"]
        self.assertIn("NMRedimension(n_points=3)", note)

    def test_note_pad(self):
        self.op.n_points = 7
        self.op.fill = 9.0
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        note = self.data.notes[0]["note"]
        self.assertIn("NMRedimension(n_points=7,fill=9)", note)


# ===========================================================================
# TestNMMainOpInsertPoints
# ===========================================================================

class TestNMMainOpInsertPoints(unittest.TestCase):
    """Test NMMainOpInsertPoints directly."""

    def setUp(self):
        self.op = NMMainOpInsertPoints()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0])

    def _run(self):
        self.op.run(self.data)

    # --- insertion ---

    def test_insert_at_start(self):
        self.op.index = 0
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [0.0, 1.0, 2.0, 3.0])

    def test_insert_at_end(self):
        self.op.index = 3
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 0.0])

    def test_insert_in_middle(self):
        self.op.index = 1
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 0.0, 2.0, 3.0])

    def test_insert_multiple(self):
        self.op.index = 1
        self.op.n_points = 2
        self.op.fill = 9.0
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 9.0, 9.0, 2.0, 3.0])

    def test_insert_with_fill(self):
        self.op.fill = 7.0
        self._run()
        self.assertEqual(self.data.nparray[0], 7.0)

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        self.op.run(d)   # should not raise

    # --- validation ---

    def test_index_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.index = True

    def test_index_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.op.index = -1

    def test_n_points_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.op.n_points = 0

    def test_fill_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.fill = False

    # --- notes ---

    def test_note_written(self):
        self.op.index = 1
        self.op.n_points = 2
        self.op.fill = 0.0
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        note = self.data.notes[0]["note"]
        self.assertIn("NMInsertPoints(index=1,n_points=2,fill=0)", note)


# ===========================================================================
# TestNMMainOpDeletePoints
# ===========================================================================

class TestNMMainOpDeletePoints(unittest.TestCase):
    """Test NMMainOpDeletePoints directly."""

    def setUp(self):
        self.op = NMMainOpDeletePoints()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0, 5.0])

    def _run(self):
        self.op.run(self.data)

    # --- deletion ---

    def test_delete_at_start(self):
        self.op.index = 0
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [2.0, 3.0, 4.0, 5.0])

    def test_delete_at_end(self):
        self.op.index = 4
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0])

    def test_delete_in_middle(self):
        self.op.index = 2
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 4.0, 5.0])

    def test_delete_multiple(self):
        self.op.index = 1
        self.op.n_points = 3
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 5.0])

    def test_index_out_of_range_no_change(self):
        self.op.index = 10
        self._run()
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0, 5.0])

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        self.op.run(d)   # should not raise

    # --- validation ---

    def test_index_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.index = True

    def test_index_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.op.index = -1

    def test_n_points_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.op.n_points = 0

    def test_n_points_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.n_points = True

    # --- notes ---

    def test_note_written(self):
        self.op.index = 2
        self.op.n_points = 3
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        note = self.data.notes[0]["note"]
        self.assertIn("NMDeletePoints(index=2,n_points=3)", note)


# ===========================================================================
# TestOpFromName (registry)
# ===========================================================================

class TestOpFromName(unittest.TestCase):

    def test_average_by_name(self):
        op = op_from_name("average")
        self.assertIsInstance(op, NMMainOpAverage)

    def test_scale_by_name(self):
        op = op_from_name("scale")
        self.assertIsInstance(op, NMMainOpScale)

    def test_redimension_by_name(self):
        op = op_from_name("redimension")
        self.assertIsInstance(op, NMMainOpRedimension)

    def test_insert_points_by_name(self):
        op = op_from_name("insert_points")
        self.assertIsInstance(op, NMMainOpInsertPoints)

    def test_delete_points_by_name(self):
        op = op_from_name("delete_points")
        self.assertIsInstance(op, NMMainOpDeletePoints)

    def test_baseline_by_name(self):
        op = op_from_name("baseline")
        self.assertIsInstance(op, NMMainOpBaseline)

    def test_reverse_by_name(self):
        op = op_from_name("reverse")
        self.assertIsInstance(op, NMMainOpReverse)

    def test_rotate_by_name(self):
        op = op_from_name("rotate")
        self.assertIsInstance(op, NMMainOpRotate)

    def test_integrate_by_name(self):
        op = op_from_name("integrate")
        self.assertIsInstance(op, NMMainOpIntegrate)

    def test_differentiate_by_name(self):
        op = op_from_name("differentiate")
        self.assertIsInstance(op, NMMainOpDifferentiate)

    def test_replace_values_by_name(self):
        op = op_from_name("replace_values")
        self.assertIsInstance(op, NMMainOpReplaceValues)

    def test_delete_nans_by_name(self):
        op = op_from_name("delete_nans")
        self.assertIsInstance(op, NMMainOpDeleteNaNs)

    def test_case_insensitive(self):
        op = op_from_name("AVERAGE")
        self.assertIsInstance(op, NMMainOpAverage)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            op_from_name("badop")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            op_from_name(42)


# ===========================================================================
# TestNMMainOpBaseline
# ===========================================================================


class TestNMMainOpBaseline(unittest.TestCase):
    """Tests for NMMainOpBaseline (per_wave and average modes)."""

    # ------------------------------------------------------------------
    # per_wave mode

    def test_per_wave_subtracts_correct_baseline(self):
        # window [0,1] covers first 2 points [2,2] → baseline=2
        op = NMMainOpBaseline(t_begin=0.0, t_end=1.0, mode="per_wave")
        d = _make_data("RecordA0", [2.0, 2.0, 4.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.0, 0.0, 2.0, 2.0])

    def test_per_wave_different_baselines(self):
        # Two waves with different values in baseline window → independent shifts
        op = NMMainOpBaseline(t_begin=0.0, t_end=0.0, mode="per_wave")
        d1 = _make_data("RecordA0", [1.0, 5.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [3.0, 7.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1)
        op.run(d2)
        np.testing.assert_array_almost_equal(d1.nparray, [0.0, 4.0])
        np.testing.assert_array_almost_equal(d2.nparray, [0.0, 4.0])

    def test_per_wave_nan_ignored(self):
        op = NMMainOpBaseline(t_begin=0.0, t_end=1.0, mode="per_wave", ignore_nans=True)
        d = _make_data("RecordA0", [np.nan, 2.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # nanmean([nan, 2.0]) = 2.0 → subtracted
        self.assertTrue(math.isfinite(float(d.nparray[2])))
        self.assertAlmostEqual(float(d.nparray[2]), 3.0)

    def test_per_wave_nan_propagates(self):
        op = NMMainOpBaseline(t_begin=0.0, t_end=1.0, mode="per_wave", ignore_nans=False)
        d = _make_data("RecordA0", [np.nan, 2.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # mean([nan, 2.0]) = nan → all points become nan
        self.assertTrue(np.all(np.isnan(d.nparray)))

    def test_window_out_of_range_no_subtraction(self):
        # Window is past end of array → empty slice → baseline=0 → no change
        op = NMMainOpBaseline(t_begin=100.0, t_end=200.0, mode="per_wave")
        d = _make_data("RecordA0", [5.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [5.0, 6.0])

    def test_window_partial_clip(self):
        # Window extends beyond array end — only existing samples used
        op = NMMainOpBaseline(t_begin=1.0, t_end=10.0, mode="per_wave")
        # xdelta=1, array=[0,1,2,3]; window 1..10 clips to indices 1..4
        d = _make_data("RecordA0", [0.0, 2.0, 4.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # baseline = mean([2,4,6]) = 4.0
        np.testing.assert_array_almost_equal(d.nparray, [-4.0, -2.0, 0.0, 2.0])

    # ------------------------------------------------------------------
    # average mode

    def test_average_mode_shared_baseline(self):
        # 3 waves [2,2], [4,4], [6,6]; window covers full wave → baselines 2,4,6
        # avg baseline = 4; all shifted by -4
        op = NMMainOpBaseline(t_begin=0.0, t_end=1.0, mode="average")
        d1 = _make_data("RecordA0", [2.0, 2.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [4.0, 4.0], xstart=0.0, xdelta=1.0)
        d3 = _make_data("RecordA2", [6.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1, "A")
        op.run(d2, "A")
        op.run(d3, "A")
        op.run_finish()
        np.testing.assert_array_almost_equal(d1.nparray, [-2.0, -2.0])
        np.testing.assert_array_almost_equal(d2.nparray, [0.0, 0.0])
        np.testing.assert_array_almost_equal(d3.nparray, [2.0, 2.0])

    def test_average_mode_per_channel(self):
        # Channel A: baselines 2,4 → avg=3; Channel B: baseline 10 → avg=10
        op = NMMainOpBaseline(t_begin=0.0, t_end=0.0, mode="average")
        a0 = _make_data("RecordA0", [2.0, 8.0], xstart=0.0, xdelta=1.0)
        a1 = _make_data("RecordA1", [4.0, 8.0], xstart=0.0, xdelta=1.0)
        b0 = _make_data("RecordB0", [10.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(a0, "A")
        op.run(a1, "A")
        op.run(b0, "B")
        op.run_finish()
        # A: avg baseline = (2+4)/2 = 3
        np.testing.assert_array_almost_equal(a0.nparray, [-1.0, 5.0])
        np.testing.assert_array_almost_equal(a1.nparray, [1.0, 5.0])
        # B: avg baseline = 10
        np.testing.assert_array_almost_equal(b0.nparray, [0.0, -5.0])

    def test_average_mode_nan_ignored(self):
        # NaN in baseline window → nanmean used, result finite
        op = NMMainOpBaseline(t_begin=0.0, t_end=1.0, mode="average", ignore_nans=True)
        d = _make_data("RecordA0", [np.nan, 4.0, 10.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d, "A")
        op.run_finish()
        # nanmean([nan, 4]) = 4 → subtract 4
        self.assertAlmostEqual(float(d.nparray[2]), 6.0)

    # ------------------------------------------------------------------
    # validation

    def test_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpBaseline(mode="median")

    def test_mode_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(mode=1)

    def test_t_end_before_t_begin_raises(self):
        op = NMMainOpBaseline(t_begin=5.0, t_end=2.0)
        with self.assertRaises(ValueError):
            op.run_init()  # calls _validate_window → raises

    def test_t_begin_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(t_begin=True)

    def test_t_end_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(t_end=True)

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(ignore_nans=1)

    def test_skips_non_ndarray(self):
        op = NMMainOpBaseline()
        d = NMData(NM, name="RecordA0")  # no nparray
        op.run_init()
        op.run(d)  # should not raise

    # --- notes ---

    def test_per_wave_note_written(self):
        op = NMMainOpBaseline(t_begin=0.0, t_end=0.0, mode="per_wave")
        d = _make_data("RecordA0", [3.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertEqual(len(d.notes), 1)
        note = d.notes[0]["note"]
        self.assertIn("NMBaseline(t_begin=0,t_end=0,mode=per_wave,baseline=3)", note)

    def test_average_note_written(self):
        op = NMMainOpBaseline(t_begin=0.0, t_end=0.0, mode="average")
        d1 = _make_data("RecordA0", [2.0, 8.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [4.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1, "A")
        op.run(d2, "A")
        op.run_finish()
        # avg baseline = (2+4)/2 = 3
        for d in (d1, d2):
            self.assertEqual(len(d.notes), 1)
            note = d.notes[0]["note"]
            self.assertIn("NMBaseline(t_begin=0,t_end=0,mode=average,channel=A,baseline=3)", note)


# ===========================================================================
# TestNMMainOpReverse
# ===========================================================================


class TestNMMainOpReverse(unittest.TestCase):
    """Test NMMainOpReverse directly."""

    def setUp(self):
        self.op = NMMainOpReverse()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0])

    # --- correct values ---

    def test_reverse(self):
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [4.0, 3.0, 2.0, 1.0])

    def test_reverse_twice_is_identity(self):
        self.op.run(self.data)
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0])

    def test_single_element(self):
        d = _make_data("RecordA0", [5.0])
        self.op.run(d)
        np.testing.assert_array_equal(d.nparray, [5.0])

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        self.op.run(d)  # should not raise

    # --- notes ---

    def test_note_written(self):
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        self.assertEqual(self.data.notes[0]["note"], "NMReverse()")


# ===========================================================================
# TestNMMainOpRotate
# ===========================================================================


class TestNMMainOpRotate(unittest.TestCase):
    """Test NMMainOpRotate directly."""

    def setUp(self):
        self.op = NMMainOpRotate()
        self.data = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0])

    # --- correct values ---

    def test_rotate_right(self):
        self.op.n_points = 1
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [4.0, 1.0, 2.0, 3.0])

    def test_rotate_left(self):
        self.op.n_points = -1
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [2.0, 3.0, 4.0, 1.0])

    def test_rotate_by_length_is_identity(self):
        self.op.n_points = 4
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0])

    def test_rotate_zero(self):
        self.op.n_points = 0
        self.op.run(self.data)
        np.testing.assert_array_equal(self.data.nparray, [1.0, 2.0, 3.0, 4.0])

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        self.op.run(d)  # should not raise

    # --- validation ---

    def test_n_points_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.op.n_points = True

    def test_n_points_rejects_float(self):
        with self.assertRaises(TypeError):
            self.op.n_points = 1.0

    # --- notes ---

    def test_note_written(self):
        self.op.n_points = 3
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        self.assertEqual(self.data.notes[0]["note"], "NMRotate(n_points=3)")

    def test_note_negative(self):
        self.op.n_points = -2
        self.op.run(self.data)
        self.assertIn("n_points=-2", self.data.notes[0]["note"])


# ===========================================================================
# TestNMMainOpIntegrate
# ===========================================================================


class TestNMMainOpIntegrate(unittest.TestCase):
    """Test NMMainOpIntegrate directly."""

    # --- rectangular ---

    def test_rectangular_correct_values(self):
        # [1,1,1,1] with delta=1 → cumsum=[1,2,3,4] * 1 = [1,2,3,4]
        d = _make_data("RecordA0", [1.0, 1.0, 1.0, 1.0], xdelta=1.0)
        NMMainOpIntegrate(method="rectangular").run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1.0, 2.0, 3.0, 4.0])

    def test_rectangular_uses_delta(self):
        # [1,1,1,1] with delta=0.5 → cumsum=[1,2,3,4] * 0.5 = [0.5,1.0,1.5,2.0]
        d = _make_data("RecordA0", [1.0, 1.0, 1.0, 1.0], xdelta=0.5)
        NMMainOpIntegrate(method="rectangular").run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.5, 1.0, 1.5, 2.0])

    # --- trapezoid ---

    def test_trapezoid_correct_values(self):
        # [0,2,4] with delta=1:
        #   step0 = 0.5*(0+2)*1 = 1
        #   step1 = 0.5*(2+4)*1 = 3
        # result = [0, 1, 4]
        d = _make_data("RecordA0", [0.0, 2.0, 4.0], xdelta=1.0)
        NMMainOpIntegrate(method="trapezoid").run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.0, 1.0, 4.0])

    def test_trapezoid_uses_delta(self):
        # [0,2,4] with delta=2:
        #   step0 = 0.5*(0+2)*2 = 2
        #   step1 = 0.5*(2+4)*2 = 6
        # result = [0, 2, 8]
        d = _make_data("RecordA0", [0.0, 2.0, 4.0], xdelta=2.0)
        NMMainOpIntegrate(method="trapezoid").run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.0, 2.0, 8.0])

    def test_trapezoid_preserves_length(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = _make_data("RecordA0", arr, xdelta=1.0)
        NMMainOpIntegrate(method="trapezoid").run(d)
        self.assertEqual(len(d.nparray), len(arr))

    # --- defaults and validation ---

    def test_method_default_is_rectangular(self):
        op = NMMainOpIntegrate()
        self.assertEqual(op.method, "rectangular")

    def test_method_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpIntegrate(method="simpson")

    def test_method_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpIntegrate(method=1)

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        NMMainOpIntegrate().run(d)  # should not raise

    # --- notes ---

    def test_note_rectangular(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        NMMainOpIntegrate(method="rectangular").run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("NMIntegrate(method=rectangular)", d.notes[0]["note"])

    def test_note_trapezoid(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        NMMainOpIntegrate(method="trapezoid").run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("NMIntegrate(method=trapezoid)", d.notes[0]["note"])


# ===========================================================================
# TestNMMainOpDifferentiate
# ===========================================================================


class TestNMMainOpDifferentiate(unittest.TestCase):
    """Test NMMainOpDifferentiate directly."""

    # --- correct values ---

    def test_correct_values(self):
        # [0,1,4,9] with delta=1 → np.gradient([0,1,4,9], 1) = [1,2,4,5]
        arr = [0.0, 1.0, 4.0, 9.0]
        d = _make_data("RecordA0", arr, xdelta=1.0)
        NMMainOpDifferentiate().run(d)
        expected = np.gradient(arr, 1.0)
        np.testing.assert_array_almost_equal(d.nparray, expected)

    def test_uses_delta(self):
        # same wave with delta=0.5 → result scaled by 1/0.5
        arr = [0.0, 1.0, 4.0, 9.0]
        d = _make_data("RecordA0", arr, xdelta=0.5)
        NMMainOpDifferentiate().run(d)
        expected = np.gradient(np.array(arr), 0.5)
        np.testing.assert_array_almost_equal(d.nparray, expected)

    def test_constant_wave_is_zero(self):
        d = _make_data("RecordA0", [5.0, 5.0, 5.0, 5.0], xdelta=1.0)
        NMMainOpDifferentiate().run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.0, 0.0, 0.0, 0.0])

    def test_preserves_length(self):
        arr = [1.0, 3.0, 6.0, 10.0, 15.0]
        d = _make_data("RecordA0", arr, xdelta=1.0)
        NMMainOpDifferentiate().run(d)
        self.assertEqual(len(d.nparray), len(arr))

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        NMMainOpDifferentiate().run(d)  # should not raise

    # --- notes ---

    def test_note_written(self):
        d = _make_data("RecordA0", [0.0, 1.0, 4.0])
        NMMainOpDifferentiate().run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertEqual(d.notes[0]["note"], "NMDifferentiate()")


# ===========================================================================
# TestNMMainOpReplaceValues
# ===========================================================================


class TestNMMainOpReplaceValues(unittest.TestCase):
    """Test NMMainOpReplaceValues directly."""

    # --- correct values ---

    def test_replaces_exact_value(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0, 2.0, 1.0])
        NMMainOpReplaceValues(old_value=2.0, new_value=99.0).run(d)
        np.testing.assert_array_equal(d.nparray, [1.0, 99.0, 3.0, 99.0, 1.0])

    def test_replaces_nan(self):
        d = _make_data("RecordA0", [1.0, float("nan"), 3.0])
        NMMainOpReplaceValues(old_value=float("nan"), new_value=0.0).run(d)
        np.testing.assert_array_equal(d.nparray, [1.0, 0.0, 3.0])

    def test_no_match_unchanged(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        NMMainOpReplaceValues(old_value=9.0, new_value=0.0).run(d)
        np.testing.assert_array_equal(d.nparray, [1.0, 2.0, 3.0])

    # --- defaults ---

    def test_old_value_default_is_zero(self):
        self.assertEqual(NMMainOpReplaceValues().old_value, 0.0)

    def test_new_value_default_is_zero(self):
        self.assertEqual(NMMainOpReplaceValues().new_value, 0.0)

    # --- validation ---

    def test_old_value_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpReplaceValues(old_value=True)

    def test_new_value_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpReplaceValues(new_value=True)

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        NMMainOpReplaceValues().run(d)  # should not raise

    # --- notes ---

    def test_note_written(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0, 2.0, 1.0])
        NMMainOpReplaceValues(old_value=2.0, new_value=99.0).run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("NMReplaceValues(old=2,new=99,n=2)", d.notes[0]["note"])

    def test_note_nan_old(self):
        d = _make_data("RecordA0", [1.0, float("nan"), 3.0])
        NMMainOpReplaceValues(old_value=float("nan"), new_value=0.0).run(d)
        self.assertIn("NMReplaceValues(old=nan,new=0,n=1)", d.notes[0]["note"])

    def test_note_written_when_no_match(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        NMMainOpReplaceValues(old_value=9.0, new_value=0.0).run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("n=0", d.notes[0]["note"])


# ===========================================================================
# TestNMMainOpDeleteNaNs
# ===========================================================================


class TestNMMainOpDeleteNaNs(unittest.TestCase):
    """Test NMMainOpDeleteNaNs directly."""

    # --- correct values ---

    def test_deletes_nan_only(self):
        # default: delete_nans=True, delete_infs=False → Inf is kept
        d = _make_data("RecordA0", [1.0, float("nan"), float("inf"), 3.0])
        NMMainOpDeleteNaNs().run(d)
        self.assertEqual(len(d.nparray), 3)
        self.assertTrue(math.isinf(d.nparray[1]))

    def test_deletes_inf_only(self):
        d = _make_data("RecordA0", [1.0, float("nan"), float("inf"), 3.0])
        NMMainOpDeleteNaNs(delete_nans=False, delete_infs=True).run(d)
        self.assertEqual(len(d.nparray), 3)
        self.assertTrue(math.isnan(d.nparray[1]))

    def test_deletes_both(self):
        d = _make_data("RecordA0", [float("nan"), 1.0, float("inf"), 2.0, float("-inf")])
        NMMainOpDeleteNaNs(delete_nans=True, delete_infs=True).run(d)
        np.testing.assert_array_equal(d.nparray, [1.0, 2.0])

    def test_no_nans_unchanged(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        NMMainOpDeleteNaNs().run(d)
        np.testing.assert_array_equal(d.nparray, [1.0, 2.0, 3.0])

    def test_all_nans_empty_array(self):
        d = _make_data("RecordA0", [float("nan"), float("nan")])
        NMMainOpDeleteNaNs().run(d)
        self.assertEqual(len(d.nparray), 0)

    # --- defaults ---

    def test_default_delete_nans_true(self):
        self.assertTrue(NMMainOpDeleteNaNs().delete_nans)

    def test_default_delete_infs_false(self):
        self.assertFalse(NMMainOpDeleteNaNs().delete_infs)

    # --- validation ---

    def test_both_false_raises(self):
        with self.assertRaises(ValueError):
            NMMainOpDeleteNaNs(delete_nans=False, delete_infs=False)

    def test_delete_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpDeleteNaNs(delete_nans=1)

    def test_delete_infs_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpDeleteNaNs(delete_infs=1)

    # --- edge cases ---

    def test_skips_non_ndarray(self):
        d = NMData(NM, name="RecordA0")
        NMMainOpDeleteNaNs().run(d)  # should not raise

    # --- notes ---

    def test_note_written(self):
        d = _make_data("RecordA0", [1.0, float("nan"), 3.0])
        NMMainOpDeleteNaNs().run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("NMDeleteNaNs(delete_nans=True,delete_infs=False,n=1)", d.notes[0]["note"])

    def test_note_no_deletions(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        NMMainOpDeleteNaNs().run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("n=0", d.notes[0]["note"])


# ===========================================================================
# TestNMToolMain
# ===========================================================================

class TestNMToolMain(unittest.TestCase):
    """Test NMToolMain.op property and end-to-end run_all()."""

    def setUp(self):
        self.tool = NMToolMain()

    # --- op property defaults ---

    def test_op_default_is_average(self):
        self.assertIsInstance(self.tool.op, NMMainOpAverage)

    # --- op setter ---

    def test_op_setter_accepts_instance(self):
        self.tool.op = NMMainOpScale()
        self.assertIsInstance(self.tool.op, NMMainOpScale)

    def test_op_setter_accepts_string_average(self):
        self.tool.op = "average"
        self.assertIsInstance(self.tool.op, NMMainOpAverage)

    def test_op_setter_accepts_string_scale(self):
        self.tool.op = "scale"
        self.assertIsInstance(self.tool.op, NMMainOpScale)

    def test_op_setter_rejects_unknown_string(self):
        with self.assertRaises(ValueError):
            self.tool.op = "normalize"

    def test_op_setter_rejects_bad_type(self):
        with self.assertRaises(TypeError):
            self.tool.op = 42

    # --- end-to-end: average ---

    def test_run_all_average_end_to_end(self):
        folder, targets = _make_folder_with_data({
            "RecordA0": [2.0, 4.0],
            "RecordA1": [4.0, 8.0],
        })
        self.tool.op = NMMainOpAverage()
        self.tool.run_all(targets)
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [3.0, 6.0])

    # --- end-to-end: scale ---

    def test_run_all_scale_end_to_end(self):
        folder, targets = _make_folder_with_data({
            "RecordA0": [1.0, 2.0, 3.0],
        })
        self.tool.op = NMMainOpScale(factor=3.0)
        self.tool.run_all(targets)
        d = folder.data.get("RecordA0")
        np.testing.assert_array_almost_equal(d.nparray, [3.0, 6.0, 9.0])

    # --- run_meta populated ---

    def test_run_meta_populated_after_run(self):
        _, targets = _make_folder_with_data({"RecordA0": [1.0]})
        self.tool.run_all(targets)
        meta = self.tool.run_meta
        self.assertIn("date", meta)
        self.assertIn("folders", meta)

    # --- NMHistory ---

    def test_history_logged_after_run(self):
        # Create a fresh NMHistory and register it so we own the buffer handler
        # on the shared logger (other test modules may have replaced it).
        fresh_history = nmh.NMHistory(quiet=True)
        nmh.set_history(fresh_history)
        before = len(fresh_history.buffer)
        _, targets = _make_folder_with_data({"RecordA0": [1.0, 2.0]})
        self.tool.op = NMMainOpScale(factor=2.0)
        self.tool.run_all(targets)
        after = len(fresh_history.buffer)
        self.assertGreater(after, before)
        # Most recent entry should contain the op class name
        last_msg = fresh_history.buffer[-1]["message"]
        self.assertIn("NMMainOpScale", last_msg)


if __name__ == "__main__":
    unittest.main()
