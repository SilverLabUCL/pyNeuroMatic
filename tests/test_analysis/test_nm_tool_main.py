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
    NMMainOpAccumulate,
    _epochs_repr,
    NMMainOpArithmetic,
    NMMainOpArithmeticByArray,
    NMMainOpAverage,
    NMMainOpBaseline,
    NMMainOpDeleteNaNs,
    NMMainOpDeletePoints,
    NMMainOpDFOF,
    NMMainOpDifferentiate,
    NMMainOpHistogram,
    NMMainOpInequality,
    NMMainOpInsertPoints,
    NMMainOpIntegrate,
    NMMainOpMax,
    NMMainOpMin,
    NMMainOpNormalize,
    NMMainOpRedimension,
    NMMainOpReplaceValues,
    NMMainOpConcatenate,
    NMMainOpRescale,
    NMMainOpRescaleX,
    NMMainOpReverse,
    NMMainOpRotate,
    NMMainOpSum,
    NMMainOpSumSqr,
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


def _op_run_all(op, arrays_by_name):
    """Build folder + data_items, call op.run_all(), return folder."""
    folder = NMFolder(name="folder0")
    data_items = []
    for name, arr in arrays_by_name.items():
        d = folder.data.new(name, nparray=np.array(arr, dtype=float))
        data_items.append((d, None))   # channel_name=None → parsed from name
    op.run_all(data_items, folder)
    return folder


# ===========================================================================
# TestEpochsRepr
# ===========================================================================

class TestEpochsRepr(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_epochs_repr([]), "[]")

    def test_single(self):
        self.assertEqual(_epochs_repr([3]), "[3]")

    def test_two_elements(self):
        self.assertEqual(_epochs_repr([2, 5]), "[2, 5]")

    def test_contiguous_from_zero(self):
        self.assertEqual(_epochs_repr(list(range(500))), "list(range(0, 500))")

    def test_contiguous_offset(self):
        self.assertEqual(_epochs_repr([5, 6, 7, 8]), "list(range(5, 9))")

    def test_step_two(self):
        self.assertEqual(_epochs_repr([0, 2, 4, 6]), "list(range(0, 8, 2))")

    def test_non_arithmetic(self):
        self.assertEqual(_epochs_repr([0, 2, 5]), "[0, 2, 5]")

    def test_three_contiguous(self):
        # minimum length that triggers range compression
        self.assertEqual(_epochs_repr([1, 2, 3]), "list(range(1, 4))")


# ===========================================================================
# TestNMMainOpAverage
# ===========================================================================

class TestNMMainOpAverage(unittest.TestCase):
    """Test NMMainOpAverage directly (no NMToolMain machinery)."""

    def setUp(self):
        self.op = NMMainOpAverage()
        # Three A-channel arrays; expected average = [4, 4, 4]
        self.arrays = {
            "RecordA0": [2.0, 2.0, 2.0],
            "RecordA1": [4.0, 4.0, 4.0],
            "RecordA2": [6.0, 6.0, 6.0],
        }

    def _run(self, arrays=None):
        if arrays is None:
            arrays = self.arrays
        return _op_run_all(self.op, arrays)

    # --- correct values ---

    def test_correct_values(self):
        folder = self._run()
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [4.0, 4.0, 4.0])

    # --- output naming and results dict ---

    def test_output_name_in_results(self):
        self._run()
        self.assertIn("A", self.op.results)
        self.assertEqual(self.op.results["A"], "Avg_RecordA")

    def test_results_populated_after_run(self):
        self._run()
        self.assertTrue(len(self.op.results) > 0)

    # --- NaN handling ---

    def test_nanmean_ignores_nan(self):
        # self.op.ignore_nans = True by default
        arrays = {
            "RecordA0": [2.0, math.nan, 2.0],
            "RecordA1": [4.0, 4.0,      4.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Avg_RecordA")
        self.assertIsNotNone(out)
        # nanmean of [2, nan, 2] and [4, 4, 4] at index 1 = mean([nan, 4]) = 4.0
        self.assertTrue(np.isfinite(out.nparray[1]))
        np.testing.assert_array_almost_equal(out.nparray, [3.0, 4.0, 3.0])

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
        _op_run_all(self.op, arrays2)
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

    def test_note_on_output_array(self):
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

    # --- compute_stdv / compute_var / compute_sem ---

    def test_compute_stdv_default_false(self):
        self.assertFalse(self.op.compute_stdv)

    def test_compute_var_default_false(self):
        self.assertFalse(self.op.compute_var)

    def test_compute_sem_default_false(self):
        self.assertFalse(self.op.compute_sem)

    def test_compute_stdv_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.op.compute_stdv = 1

    def test_compute_var_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.op.compute_var = "yes"

    def test_compute_sem_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            self.op.compute_sem = 0

    def test_compute_stdv_writes_array(self):
        self.op.compute_stdv = True
        folder = self._run()
        self.assertIsNotNone(folder.data.get("Stdv_RecordA"))

    def test_compute_stdv_values(self):
        # [2,2,2], [4,4,4], [6,6,6] → stdv (ddof=1) = 2.0 at each point
        self.op.compute_stdv = True
        folder = self._run()
        out = folder.data.get("Stdv_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [2.0, 2.0, 2.0])

    def test_compute_var_values(self):
        self.op.compute_var = True
        folder = self._run()
        out = folder.data.get("Var_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [4.0, 4.0, 4.0])

    def test_compute_sem_values(self):
        import math
        self.op.compute_sem = True
        folder = self._run()
        out = folder.data.get("SEM_RecordA")
        self.assertIsNotNone(out)
        # stdv=2.0, n=3, sem=2/sqrt(3)
        expected = 2.0 / math.sqrt(3)
        np.testing.assert_array_almost_equal(out.nparray, [expected, expected, expected])

    def test_all_three_writes_three_extra_arrays(self):
        self.op.compute_stdv = True
        self.op.compute_var = True
        self.op.compute_sem = True
        folder = self._run()
        self.assertIsNotNone(folder.data.get("Stdv_RecordA"))
        self.assertIsNotNone(folder.data.get("Var_RecordA"))
        self.assertIsNotNone(folder.data.get("SEM_RecordA"))

    def test_stdv_note_on_output_array(self):
        self.op.compute_stdv = True
        folder = self._run({"RecordA0": [2.0, 2.0], "RecordA1": [4.0, 4.0], "RecordA2": [6.0, 6.0]})
        out = folder.data.get("Stdv_RecordA")
        self.assertIsNotNone(out)
        self.assertEqual(len(out.notes), 1)
        note = out.notes[0]["note"]
        self.assertIn("NMStdv(folder=folder0", note)
        self.assertIn("channel=A", note)
        self.assertIn("n_epochs=3", note)


# ===========================================================================
# TestNMMainOpSum
# ===========================================================================

class TestNMMainOpSum(unittest.TestCase):
    """Test NMMainOpSum directly."""

    def setUp(self):
        self.op = NMMainOpSum()
        self.arrays = {
            "RecordA0": [1.0, 2.0, 3.0],
            "RecordA1": [4.0, 5.0, 6.0],
        }

    def _run(self, arrays=None):
        if arrays is None:
            arrays = self.arrays
        return _op_run_all(self.op, arrays)

    def test_correct_values(self):
        folder = self._run()
        out = folder.data.get("Sum_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [5.0, 7.0, 9.0])

    def test_output_name(self):
        folder = self._run()
        self.assertIsNotNone(folder.data.get("Sum_RecordA"))

    def test_nan_handling_ignore_nans(self):
        # self.op.ignore_nans = True by default
        arrays = {
            "RecordA0": [math.nan, 2.0],
            "RecordA1": [4.0, 5.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Sum_RecordA")
        self.assertIsNotNone(out)
        # nansum treats nan as 0, so [nan+4, 2+5] = [4, 7]
        np.testing.assert_array_almost_equal(out.nparray, [4.0, 7.0])

    def test_nan_propagates_when_ignore_nans_false(self):
        self.op.ignore_nans = False
        arrays = {
            "RecordA0": [math.nan, 2.0],
            "RecordA1": [4.0, 5.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Sum_RecordA")
        self.assertTrue(math.isnan(out.nparray[0]))
        self.assertEqual(out.nparray[1], 7.0)

    def test_two_channels(self):
        arrays = {
            "RecordA0": [1.0, 2.0],
            "RecordA1": [3.0, 4.0],
            "RecordB0": [5.0, 6.0],
            "RecordB1": [7.0, 8.0],
        }
        folder = self._run(arrays)
        self.assertIsNotNone(folder.data.get("Sum_RecordA"))
        self.assertIsNotNone(folder.data.get("Sum_RecordB"))

    def test_no_folder_graceful(self):
        data_items = [(_make_data("RecordA0", [1.0, 2.0]), None)]
        self.op.run_all(data_items, None)
        self.assertEqual(self.op.results, {})

    def test_note_written(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        out = folder.data.get("Sum_RecordA")
        self.assertIsNotNone(out)
        self.assertEqual(len(out.notes), 1)
        note = out.notes[0]["note"]
        self.assertIn("NMSum(folder=folder0", note)
        self.assertIn("channel=A", note)
        self.assertIn("n_epochs=2", note)


# ===========================================================================
# TestNMMainOpSumSqr
# ===========================================================================

class TestNMMainOpSumSqr(unittest.TestCase):
    """Test NMMainOpSumSqr directly."""

    def setUp(self):
        self.op = NMMainOpSumSqr()

    def _run(self, arrays):
        return _op_run_all(self.op, arrays)

    def test_correct_values(self):
        # [1,2] and [3,4] → [1²+3², 2²+4²] = [10, 20]
        folder = self._run({"RecordA0": [1.0, 2.0], "RecordA1": [3.0, 4.0]})
        out = folder.data.get("SumSqr_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [10.0, 20.0])

    def test_output_name(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        self.assertIsNotNone(folder.data.get("SumSqr_RecordA"))

    def test_nan_handling_ignore_nans(self):
        arrays = {
            "RecordA0": [math.nan, 2.0],
            "RecordA1": [3.0, 4.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("SumSqr_RecordA")
        self.assertIsNotNone(out)
        # nansum: [nan²+9, 4+16] = [9, 20]
        np.testing.assert_array_almost_equal(out.nparray, [9.0, 20.0])

    def test_note_written(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        out = folder.data.get("SumSqr_RecordA")
        self.assertIsNotNone(out)
        note = out.notes[0]["note"]
        self.assertIn("NMSumSqr(folder=folder0", note)
        self.assertIn("channel=A", note)


# ===========================================================================
# TestNMMainOpMin
# ===========================================================================

class TestNMMainOpMin(unittest.TestCase):
    """Test NMMainOpMin directly."""

    def setUp(self):
        self.op = NMMainOpMin()

    def _run(self, arrays):
        return _op_run_all(self.op, arrays)

    def test_correct_values(self):
        # [1,5] and [3,2] → point-by-point min = [1, 2]
        folder = self._run({"RecordA0": [1.0, 5.0], "RecordA1": [3.0, 2.0]})
        out = folder.data.get("Min_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [1.0, 2.0])

    def test_output_name(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        self.assertIsNotNone(folder.data.get("Min_RecordA"))

    def test_nan_handling_ignore_nans(self):
        arrays = {
            "RecordA0": [math.nan, 5.0],
            "RecordA1": [3.0, 2.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Min_RecordA")
        self.assertIsNotNone(out)
        # nanmin ignores nan: [3.0, 2.0]
        np.testing.assert_array_almost_equal(out.nparray, [3.0, 2.0])

    def test_note_written(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        out = folder.data.get("Min_RecordA")
        self.assertIsNotNone(out)
        note = out.notes[0]["note"]
        self.assertIn("NMMin(folder=folder0", note)
        self.assertIn("channel=A", note)


# ===========================================================================
# TestNMMainOpMax
# ===========================================================================

class TestNMMainOpMax(unittest.TestCase):
    """Test NMMainOpMax directly."""

    def setUp(self):
        self.op = NMMainOpMax()

    def _run(self, arrays):
        return _op_run_all(self.op, arrays)

    def test_correct_values(self):
        # [1,5] and [3,2] → point-by-point max = [3, 5]
        folder = self._run({"RecordA0": [1.0, 5.0], "RecordA1": [3.0, 2.0]})
        out = folder.data.get("Max_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [3.0, 5.0])

    def test_output_name(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        self.assertIsNotNone(folder.data.get("Max_RecordA"))

    def test_nan_handling_ignore_nans(self):
        arrays = {
            "RecordA0": [math.nan, 5.0],
            "RecordA1": [3.0, 2.0],
        }
        folder = self._run(arrays)
        out = folder.data.get("Max_RecordA")
        self.assertIsNotNone(out)
        # nanmax ignores nan: [3.0, 5.0]
        np.testing.assert_array_almost_equal(out.nparray, [3.0, 5.0])

    def test_note_written(self):
        folder = self._run({"RecordA0": [1.0], "RecordA1": [2.0]})
        out = folder.data.get("Max_RecordA")
        self.assertIsNotNone(out)
        note = out.notes[0]["note"]
        self.assertIn("NMMax(folder=folder0", note)
        self.assertIn("channel=A", note)


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
        self.assertIn("NMRedimension(n_points=3, fill=0.0)", note)

    def test_note_pad(self):
        self.op.n_points = 7
        self.op.fill = 9.0
        self.op.run(self.data)
        self.assertEqual(len(self.data.notes), 1)
        note = self.data.notes[0]["note"]
        self.assertIn("NMRedimension(n_points=7, fill=9.0)", note)


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

    def test_insert_ax1(self):
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
        self.assertIn("NMInsertPoints(index=1, n_points=2, fill=0.0)", note)


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

    def test_delete_ax1(self):
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
        self.assertIn("NMDeletePoints(index=2, n_points=3)", note)


# ===========================================================================
# TestOpFromName (registry)
# ===========================================================================

class TestOpFromName(unittest.TestCase):

    def test_average_by_name(self):
        op = op_from_name("average")
        self.assertIsInstance(op, NMMainOpAverage)

    def test_scale_name_raises(self):
        with self.assertRaises(ValueError):
            op_from_name("scale")

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

    def test_normalize_by_name(self):
        op = op_from_name("normalize")
        self.assertIsInstance(op, NMMainOpNormalize)

    def test_sum_by_name(self):
        op = op_from_name("sum")
        self.assertIsInstance(op, NMMainOpSum)

    def test_sum_sqr_by_name(self):
        op = op_from_name("sum_sqr")
        self.assertIsInstance(op, NMMainOpSumSqr)

    def test_min_by_name(self):
        op = op_from_name("min")
        self.assertIsInstance(op, NMMainOpMin)

    def test_max_by_name(self):
        op = op_from_name("max")
        self.assertIsInstance(op, NMMainOpMax)

    def test_arithmetic_by_name(self):
        op = op_from_name("arithmetic")
        self.assertIsInstance(op, NMMainOpArithmetic)

    def test_arithmetic_by_array_by_name(self):
        op = op_from_name("arithmetic_by_array")
        self.assertIsInstance(op, NMMainOpArithmeticByArray)

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
    """Tests for NMMainOpBaseline (per_array and average modes)."""

    # ------------------------------------------------------------------
    # per_array mode

    def test_per_array_subtracts_correct_baseline(self):
        # window [0,1] covers first 2 points [2,2] → baseline=2
        op = NMMainOpBaseline(x0=0.0, x1=1.0, mode="per_array")
        d = _make_data("RecordA0", [2.0, 2.0, 4.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [0.0, 0.0, 2.0, 2.0])

    def test_per_array_different_baselines(self):
        # Two arrays with different values in baseline window → independent shifts
        op = NMMainOpBaseline(x0=0.0, x1=0.0, mode="per_array")
        d1 = _make_data("RecordA0", [1.0, 5.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [3.0, 7.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1)
        op.run(d2)
        np.testing.assert_array_almost_equal(d1.nparray, [0.0, 4.0])
        np.testing.assert_array_almost_equal(d2.nparray, [0.0, 4.0])

    def test_per_array_nan_ignored(self):
        op = NMMainOpBaseline(x0=0.0, x1=1.0, mode="per_array", ignore_nans=True)
        d = _make_data("RecordA0", [np.nan, 2.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # nanmean([nan, 2.0]) = 2.0 → subtracted
        self.assertTrue(math.isfinite(float(d.nparray[2])))
        self.assertAlmostEqual(float(d.nparray[2]), 3.0)

    def test_per_array_nan_propagates(self):
        op = NMMainOpBaseline(x0=0.0, x1=1.0, mode="per_array", ignore_nans=False)
        d = _make_data("RecordA0", [np.nan, 2.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # mean([nan, 2.0]) = nan → all points become nan
        self.assertTrue(np.all(np.isnan(d.nparray)))

    def test_window_out_of_range_no_subtraction(self):
        # Window is past end of array → empty slice → baseline=0 → no change
        op = NMMainOpBaseline(x0=100.0, x1=200.0, mode="per_array")
        d = _make_data("RecordA0", [5.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [5.0, 6.0])

    def test_window_partial_clip(self):
        # Window extends beyond array end — only existing samples used
        op = NMMainOpBaseline(x0=1.0, x1=10.0, mode="per_array")
        # xdelta=1, array=[0,1,2,3]; window 1..10 clips to indices 1..4
        d = _make_data("RecordA0", [0.0, 2.0, 4.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # baseline = mean([2,4,6]) = 4.0
        np.testing.assert_array_almost_equal(d.nparray, [-4.0, -2.0, 0.0, 2.0])

    # ------------------------------------------------------------------
    # average mode

    def test_average_mode_shared_baseline(self):
        # 3 arrays [2,2], [4,4], [6,6]; window covers full array → baselines 2,4,6
        # avg baseline = 4; all shifted by -4
        op = NMMainOpBaseline(x0=0.0, x1=1.0, mode="average")
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
        op = NMMainOpBaseline(x0=0.0, x1=0.0, mode="average")
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
        op = NMMainOpBaseline(x0=0.0, x1=1.0, mode="average", ignore_nans=True)
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

    def test_x1_before_x0_raises(self):
        op = NMMainOpBaseline(x0=5.0, x1=2.0)
        with self.assertRaises(ValueError):
            op.run_init()  # calls _validate_window → raises

    def test_x0_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(x0=True)

    def test_x1_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(x1=True)

    def test_x0_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpBaseline(x0=float("nan"))

    def test_x1_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpBaseline(x1=float("nan"))

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpBaseline(ignore_nans=1)

    def test_skips_non_ndarray(self):
        op = NMMainOpBaseline()
        d = NMData(NM, name="RecordA0")  # no nparray
        op.run_init()
        op.run(d)  # should not raise

    # --- notes ---

    def test_per_array_note_written(self):
        op = NMMainOpBaseline(x0=0.0, x1=0.0, mode="per_array")
        d = _make_data("RecordA0", [3.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertEqual(len(d.notes), 1)
        note = d.notes[0]["note"]
        self.assertIn("NMBaseline(", note)
        self.assertIn("x0=0.0", note)
        self.assertIn("mode='per_array'", note)
        self.assertIn("baseline=3", note)

    def test_average_note_written(self):
        op = NMMainOpBaseline(x0=0.0, x1=0.0, mode="average")
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
            self.assertIn("NMBaseline(", note)
            self.assertIn("x0=0.0", note)
            self.assertIn("mode='average'", note)
            self.assertIn("channel=A", note)
            self.assertIn("baseline=3", note)


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
        self.assertIn("NMIntegrate(method='rectangular')", d.notes[0]["note"])

    def test_note_trapezoid(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        NMMainOpIntegrate(method="trapezoid").run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("NMIntegrate(method='trapezoid')", d.notes[0]["note"])


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
        # same array with delta=0.5 → result scaled by 1/0.5
        arr = [0.0, 1.0, 4.0, 9.0]
        d = _make_data("RecordA0", arr, xdelta=0.5)
        NMMainOpDifferentiate().run(d)
        expected = np.gradient(np.array(arr), 0.5)
        np.testing.assert_array_almost_equal(d.nparray, expected)

    def test_constant_array_is_zero(self):
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
        self.assertIn("NMReplaceValues(old_value=2.0, new_value=99.0, n=2)", d.notes[0]["note"])

    def test_note_nan_old(self):
        d = _make_data("RecordA0", [1.0, float("nan"), 3.0])
        NMMainOpReplaceValues(old_value=float("nan"), new_value=0.0).run(d)
        self.assertIn("NMReplaceValues(old_value=nan, new_value=0.0, n=1)", d.notes[0]["note"])

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
        self.assertIn("NMDeleteNaNs(delete_nans=True, delete_infs=False, n=1)", d.notes[0]["note"])

    def test_note_no_deletions(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        NMMainOpDeleteNaNs().run(d)
        self.assertEqual(len(d.notes), 1)
        self.assertIn("n=0", d.notes[0]["note"])


# ===========================================================================
# TestNMMainOpNormalize
# ===========================================================================


class TestNMMainOpNormalize(unittest.TestCase):
    """Tests for NMMainOpNormalize (per_array and average modes)."""

    # ------------------------------------------------------------------
    # per_array mode — correct values

    def test_per_array_min_max(self):
        # [0,5,10], fxn1=min→0, fxn2=max→10, range=10 → [0.0, 0.5, 1.0]
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=2.0, fxn1="min",
            x0_max=0.0, x1_max=2.0, fxn2="max",
        )
        data = _make_data("RecordA0", [0.0, 5.0, 10.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_almost_equal(data.nparray, [0.0, 0.5, 1.0])

    def test_per_array_mean_zero_range(self):
        # fxn1=mean and fxn2=mean over same window → ref_min==ref_max → norm_min everywhere
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=4.0, fxn1="mean",
            x0_max=0.0, x1_max=4.0, fxn2="mean",
            norm_min=-99.0,
        )
        data = _make_data("RecordA0", [0.0, 2.0, 4.0, 6.0, 8.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_equal(data.nparray, [-99.0] * 5)

    def test_per_array_uses_windows(self):
        # [0,1,2,3,4] xdelta=1; window1=[0,0]→first point(0), window2=[4,4]→last point(4)
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=0.0, fxn1="mean",
            x0_max=4.0, x1_max=4.0, fxn2="mean",
        )
        data = _make_data("RecordA0", [0.0, 1.0, 2.0, 3.0, 4.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_almost_equal(data.nparray, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_per_array_custom_norm_range(self):
        # norm_min=-1, norm_max=1; [0,5,10]→ref_min=0,ref_max=10 → [-1,0,1]
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=2.0, fxn1="min",
            x0_max=0.0, x1_max=2.0, fxn2="max",
            norm_min=-1.0, norm_max=1.0,
        )
        data = _make_data("RecordA0", [0.0, 5.0, 10.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_almost_equal(data.nparray, [-1.0, 0.0, 1.0])

    def test_per_array_mean_at_min(self):
        # [3,1,2], fxn1=mean@min, n_mean1=3; min at i=1, mean of [3,1,2]=2.0 → ref_min=2.0
        # fxn2=max → ref_max=3.0; range=1; normalized: arr-2.0 → [1,-1,0]
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=2.0, fxn1="mean@min", n_mean1=3,
            x0_max=0.0, x1_max=2.0, fxn2="max",
        )
        data = _make_data("RecordA0", [3.0, 1.0, 2.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_almost_equal(data.nparray, [1.0, -1.0, 0.0])

    def test_per_array_mean_at_max(self):
        # [1,5,3], fxn2=mean@max, n_mean2=3; max at i=1, mean of [1,5,3]=3.0 → ref_max=3.0
        # fxn1=min → ref_min=1.0; range=2; normalized: (arr-1)/2 → [0,2,1]
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=2.0, fxn1="min",
            x0_max=0.0, x1_max=2.0, fxn2="mean@max", n_mean2=3,
        )
        data = _make_data("RecordA0", [1.0, 5.0, 3.0])
        op.run_init()
        op.run(data, "A")
        np.testing.assert_array_almost_equal(data.nparray, [0.0, 2.0, 1.0])

    def test_per_array_preserves_length(self):
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=99.0, fxn1="min",
            x0_max=0.0, x1_max=99.0, fxn2="max",
        )
        data = _make_data("RecordA0", list(range(100)))
        op.run_init()
        op.run(data, "A")
        self.assertEqual(len(data.nparray), 100)

    # ------------------------------------------------------------------
    # average mode

    def test_average_mode_shared_refs(self):
        # 2 arrays same channel; ref_mins=[0,2]→avg=1, ref_maxes=[4,6]→avg=5, range=4
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=1.0, fxn1="min",
            x0_max=0.0, x1_max=1.0, fxn2="max",
            mode="average",
        )
        d0 = _make_data("RecordA0", [0.0, 4.0])
        d1 = _make_data("RecordA1", [2.0, 6.0])
        op.run_all([(d0, "A"), (d1, "A")], folder=None)
        np.testing.assert_array_almost_equal(d0.nparray, [-0.25, 0.75])
        np.testing.assert_array_almost_equal(d1.nparray, [0.25, 1.25])

    def test_average_mode_per_channel(self):
        # Channel A ref_max=10, channel B ref_max=5 — independent
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=1.0, fxn1="min",
            x0_max=0.0, x1_max=1.0, fxn2="max",
            mode="average",
        )
        dA = _make_data("RecordA0", [0.0, 10.0])
        dB = _make_data("RecordB0", [0.0, 5.0])
        op.run_all([(dA, "A"), (dB, "B")], folder=None)
        np.testing.assert_array_almost_equal(dA.nparray, [0.0, 1.0])
        np.testing.assert_array_almost_equal(dB.nparray, [0.0, 1.0])
        # Verify each channel's note has its own ref_max
        note_A = dA.notes[0]["note"]
        note_B = dB.notes[0]["note"]
        self.assertIn("ref_max=10", note_A)
        self.assertIn("ref_max=5", note_B)

    # ------------------------------------------------------------------
    # validation

    def test_fxn1_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(fxn1="bad")

    def test_fxn1_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpNormalize(fxn1=42)

    def test_fxn2_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(fxn2="bad")

    def test_n_mean1_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpNormalize(n_mean1=True)

    def test_n_mean1_rejects_zero(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(n_mean1=0)

    def test_norm_min_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpNormalize(norm_min=True)

    def test_norm_max_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpNormalize(norm_max=False)

    def test_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(mode="bad")

    def test_x1_min_before_x0_min_raises(self):
        op = NMMainOpNormalize(x0_min=5.0, x1_min=0.0)
        with self.assertRaises(ValueError):
            op.run_init()

    def test_x0_min_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(x0_min=float("nan"))

    def test_x1_min_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(x1_min=float("nan"))

    def test_x0_max_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(x0_max=float("nan"))

    def test_x1_max_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpNormalize(x1_max=float("nan"))

    # ------------------------------------------------------------------
    # edge cases / notes

    def test_skips_non_ndarray(self):
        op = NMMainOpNormalize()
        d = NMData(NM, name="RecordA0")  # no nparray
        op.run_init()
        op.run(d, "A")  # should not raise
        self.assertEqual(len(d.notes), 0)

    def test_note_per_array(self):
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=2.0, fxn1="min",
            x0_max=0.0, x1_max=2.0, fxn2="max",
            mode="per_array",
        )
        data = _make_data("RecordA0", [0.0, 5.0, 10.0])
        op.run_init()
        op.run(data, "A")
        self.assertEqual(len(data.notes), 1)
        note = data.notes[0]["note"]
        self.assertIn("NMNormalize", note)
        self.assertIn("mode='per_array'", note)
        self.assertIn("ref_min=", note)
        self.assertIn("ref_max=", note)

    def test_note_average(self):
        op = NMMainOpNormalize(
            x0_min=0.0, x1_min=1.0, fxn1="min",
            x0_max=0.0, x1_max=1.0, fxn2="max",
            mode="average",
        )
        data = _make_data("RecordA0", [0.0, 10.0])
        op.run_all([(data, "A")], folder=None)
        self.assertEqual(len(data.notes), 1)
        note = data.notes[0]["note"]
        self.assertIn("NMNormalize", note)
        self.assertIn("mode='average'", note)
        self.assertIn("channel=A", note)
        self.assertIn("ref_min=", note)
        self.assertIn("ref_max=", note)


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
        self.tool.op = NMMainOpArithmetic()
        self.assertIsInstance(self.tool.op, NMMainOpArithmetic)

    def test_op_setter_accepts_string_average(self):
        self.tool.op = "average"
        self.assertIsInstance(self.tool.op, NMMainOpAverage)

    def test_op_setter_accepts_string_arithmetic(self):
        self.tool.op = "arithmetic"
        self.assertIsInstance(self.tool.op, NMMainOpArithmetic)

    def test_op_setter_rejects_unknown_string(self):
        with self.assertRaises(ValueError):
            self.tool.op = "badopname"

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

    # --- end-to-end: arithmetic ---

    def test_run_all_arithmetic_end_to_end(self):
        folder, targets = _make_folder_with_data({
            "RecordA0": [1.0, 2.0, 3.0],
        })
        self.tool.op = NMMainOpArithmetic(factor=3.0)
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
        self.tool.op = NMMainOpArithmetic(factor=2.0)
        self.tool.run_all(targets)
        after = len(fresh_history.buffer)
        self.assertGreater(after, before)
        # Most recent entry should contain the op class name
        last_msg = fresh_history.buffer[-1]["message"]
        self.assertIn("NMMainOpArithmetic", last_msg)


# ===========================================================================
# TestNMMainOpArithmetic
# ===========================================================================


class TestNMMainOpArithmetic(unittest.TestCase):
    """Tests for NMMainOpArithmetic (scalar, list, and dict factor modes)."""

    def _make_items(self, names, arrays):
        return [(_make_data(n, a), None) for n, a in zip(names, arrays)]

    # ------------------------------------------------------------------
    # scalar mode

    def _run_scalar(self, arr, factor, op):
        op_obj = NMMainOpArithmetic(factor=factor, op=op)
        d = _make_data("RecordA0", arr)
        op_obj.run_all([(d, None)], folder=None)
        return d.nparray

    def test_scalar_multiply(self):
        result = self._run_scalar([1.0, 2.0, 3.0], factor=2.0, op="x")
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    def test_scalar_add(self):
        result = self._run_scalar([1.0, 2.0, 3.0], factor=10.0, op="+")
        np.testing.assert_array_almost_equal(result, [11.0, 12.0, 13.0])

    def test_scalar_subtract(self):
        result = self._run_scalar([5.0, 6.0, 7.0], factor=2.0, op="-")
        np.testing.assert_array_almost_equal(result, [3.0, 4.0, 5.0])

    def test_scalar_divide(self):
        result = self._run_scalar([4.0, 6.0, 8.0], factor=2.0, op="/")
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_scalar_assign(self):
        result = self._run_scalar([1.0, 2.0, 3.0], factor=99.0, op="=")
        np.testing.assert_array_almost_equal(result, [99.0, 99.0, 99.0])

    def test_scalar_exponentiate(self):
        result = self._run_scalar([2.0, 3.0, 4.0], factor=2.0, op="**")
        np.testing.assert_array_almost_equal(result, [4.0, 9.0, 16.0])

    # ------------------------------------------------------------------
    # list mode

    def test_list_multiply(self):
        items = self._make_items(
            ["RecordA0", "RecordA1", "RecordA2"],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        )
        op = NMMainOpArithmetic(factor=[2.0, 3.0, 4.0], op="x")
        op.run_all(items, folder=None)
        np.testing.assert_array_almost_equal(items[0][0].nparray, [2.0, 4.0])
        np.testing.assert_array_almost_equal(items[1][0].nparray, [9.0, 12.0])
        np.testing.assert_array_almost_equal(items[2][0].nparray, [20.0, 24.0])

    def test_list_add(self):
        items = self._make_items(["RecordA0", "RecordA1"], [[1.0], [2.0]])
        op = NMMainOpArithmetic(factor=[10.0, 20.0], op="+")
        op.run_all(items, folder=None)
        np.testing.assert_array_almost_equal(items[0][0].nparray, [11.0])
        np.testing.assert_array_almost_equal(items[1][0].nparray, [22.0])

    def test_list_length_mismatch_raises_before_any_run(self):
        items = self._make_items(["RecordA0", "RecordA1"], [[1.0], [2.0]])
        op = NMMainOpArithmetic(factor=[2.0])  # 1 factor for 2 items
        with self.assertRaises(IndexError):
            op.run_all(items, folder=None)
        # error raised in run_init — no items should have been modified
        np.testing.assert_array_almost_equal(items[0][0].nparray, [1.0])
        np.testing.assert_array_almost_equal(items[1][0].nparray, [2.0])

    def test_list_too_long_raises(self):
        items = self._make_items(["RecordA0"], [[1.0]])
        op = NMMainOpArithmetic(factor=[2.0, 3.0])  # 2 factors for 1 item
        with self.assertRaises(IndexError):
            op.run_all(items, folder=None)

    def test_list_state_reset_between_runs(self):
        items = self._make_items(["RecordA0"], [[1.0]])
        op = NMMainOpArithmetic(factor=[5.0], op="x")
        op.run_all(items, folder=None)
        np.testing.assert_array_almost_equal(items[0][0].nparray, [5.0])
        items2 = self._make_items(["RecordA0"], [[2.0]])
        op.run_all(items2, folder=None)
        np.testing.assert_array_almost_equal(items2[0][0].nparray, [10.0])

    def test_list_element_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmetic(factor=[1.0, True])

    # ------------------------------------------------------------------
    # dict mode

    def test_dict_multiply(self):
        items = self._make_items(
            ["RecordA0", "RecordA1"], [[2.0, 3.0], [4.0, 5.0]]
        )
        op = NMMainOpArithmetic(factor={"RecordA0": 10.0, "RecordA1": 0.5}, op="x")
        op.run_all(items, folder=None)
        np.testing.assert_array_almost_equal(items[0][0].nparray, [20.0, 30.0])
        np.testing.assert_array_almost_equal(items[1][0].nparray, [2.0, 2.5])

    def test_dict_missing_key_skips(self):
        d = _make_data("RecordA0", [3.0, 4.0])
        op = NMMainOpArithmetic(factor={"RecordA1": 2.0}, op="x")
        op.run_all([(d, None)], folder=None)
        np.testing.assert_array_almost_equal(d.nparray, [3.0, 4.0])

    def test_factor_rejects_tuple(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmetic(factor=(1.0, 2.0))

    # ------------------------------------------------------------------
    # validation

    def test_op_default_is_multiply(self):
        op = NMMainOpArithmetic(factor=3.0)
        self.assertEqual(op.op, "x")

    def test_op_rejects_unknown(self):
        with self.assertRaises(ValueError):
            NMMainOpArithmetic(factor=1.0, op="^")

    def test_op_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmetic(factor=1.0, op=2)

    def test_factor_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmetic(factor=True)

    def test_factor_rejects_str(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmetic(factor="2")

    def test_skips_non_ndarray(self):
        op = NMMainOpArithmetic(factor=2.0)
        d = NMData("RecordA0")
        op.run(d)

    def test_note_written(self):
        op = NMMainOpArithmetic(factor=2.0, op="x")
        d = _make_data("RecordA0", [1.0, 2.0])
        op.run(d)
        notes = [e["note"] for e in d.notes._entries]
        self.assertTrue(any("NMArithmetic(factor=2, op='x')" in n for n in notes))


# ===========================================================================
# TestNMMainOpArithmeticByArray
# ===========================================================================


def _make_folder_with_ref(folder_name, data_name, arr):
    """Return (folder, NMData) with a data item pre-loaded."""
    folder = NMFolder(name=folder_name)
    folder.data.new(data_name, nparray=np.array(arr, dtype=float))
    return folder, folder.data.get(data_name)


class TestNMMainOpArithmeticByArray(unittest.TestCase):
    """Tests for NMMainOpArithmeticByArray (element-wise reference arithmetic)."""

    def _run(self, data_arr, ref_arr, op):
        d = _make_data("RecordA0", data_arr)
        ref = np.array(ref_arr, dtype=float)
        op_obj = NMMainOpArithmeticByArray(ref=ref, op=op)
        op_obj.run_all([(d, None)], folder=None)
        return d.nparray

    def test_multiply_by_array(self):
        result = self._run([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], "x")
        np.testing.assert_array_almost_equal(result, [2.0, 6.0, 12.0])

    def test_subtract_array(self):
        result = self._run([5.0, 6.0, 7.0], [1.0, 2.0, 3.0], "-")
        np.testing.assert_array_almost_equal(result, [4.0, 4.0, 4.0])

    def test_assign_array(self):
        result = self._run([1.0, 2.0, 3.0], [9.0, 8.0, 7.0], "=")
        np.testing.assert_array_almost_equal(result, [9.0, 8.0, 7.0])

    def test_ref_by_name(self):
        folder, _ = _make_folder_with_ref("Folder1", "RefArray", [2.0, 4.0, 6.0])
        d = _make_data("RecordA0", [1.0, 2.0, 3.0])
        items = [(d, None)]
        op = NMMainOpArithmeticByArray(ref="RefArray", op="x")
        op.run_all(items, folder=folder)
        np.testing.assert_array_almost_equal(d.nparray, [2.0, 8.0, 18.0])

    def test_ref_array_not_found_raises(self):
        folder = NMFolder(name="Folder1")
        d = _make_data("RecordA0", [1.0, 2.0])
        op = NMMainOpArithmeticByArray(ref="NoSuchArray", op="x")
        with self.assertRaises(ValueError):
            op.run_all([(d, None)], folder=folder)

    def test_no_folder_with_string_ref_raises(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        op = NMMainOpArithmeticByArray(ref="RefArray", op="x")
        with self.assertRaises(ValueError):
            op.run_all([(d, None)], folder=None)

    def test_length_mismatch_short_ref_raises_before_any_run(self):
        d = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0])
        ref = np.array([10.0, 10.0], dtype=float)
        op = NMMainOpArithmeticByArray(ref=ref, op="x")
        with self.assertRaises(ValueError):
            op.run_all([(d, None)], folder=None)
        # error raised in run_init — data should be unchanged
        np.testing.assert_array_almost_equal(d.nparray, [1.0, 2.0, 3.0, 4.0])

    def test_length_mismatch_long_ref_raises_before_any_run(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        ref = np.array([3.0, 4.0, 5.0, 6.0], dtype=float)
        op = NMMainOpArithmeticByArray(ref=ref, op="x")
        with self.assertRaises(ValueError):
            op.run_all([(d, None)], folder=None)
        # error raised in run_init — data should be unchanged
        np.testing.assert_array_almost_equal(d.nparray, [1.0, 2.0])

    def test_ref_rejects_non_array_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpArithmeticByArray(ref=42)

    def test_note_with_name_ref(self):
        folder, _ = _make_folder_with_ref("Folder1", "RefArray", [1.0, 1.0])
        d = _make_data("RecordA0", [2.0, 3.0])
        op = NMMainOpArithmeticByArray(ref="RefArray", op="x")
        op.run_all([(d, None)], folder=folder)
        notes = [e["note"] for e in d.notes._entries]
        self.assertTrue(any("NMArithmeticByArray(ref='RefArray', op='x')" in n for n in notes))

    def test_note_with_array_ref(self):
        d = _make_data("RecordA0", [2.0, 3.0])
        ref = np.array([1.0, 1.0])
        op = NMMainOpArithmeticByArray(ref=ref, op="+")
        op.run_all([(d, None)], folder=None)
        notes = [e["note"] for e in d.notes._entries]
        self.assertTrue(any("NMArithmeticByArray(ref=np.array([...]), op='+')" in n for n in notes))


# ===========================================================================
# TestNMMainOpInequality
# ===========================================================================


class TestNMMainOpInequality(unittest.TestCase):

    def _run(self, op, arrays_by_name):
        """Run op and return the folder."""
        return _op_run_all(op, arrays_by_name)

    # --- basic output ---

    def test_gt_binary(self):
        op = NMMainOpInequality(op=">", a=2.0, binary_output=True)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = folder.data.get("IQ_RecordA0")
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out.nparray, [0.0, 0.0, 1.0, 1.0, 1.0])

    def test_lt_non_binary(self):
        op = NMMainOpInequality(op="<", a=4.0, binary_output=False)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = folder.data.get("IQ_RecordA0")
        self.assertIsNotNone(out)
        arr = out.nparray
        np.testing.assert_array_equal(arr[:3], [1.0, 2.0, 3.0])
        self.assertTrue(math.isnan(arr[3]))
        self.assertTrue(math.isnan(arr[4]))

    def test_range_op(self):
        op = NMMainOpInequality(op="<=<=", a=2.0, b=4.0, binary_output=True)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = folder.data.get("IQ_RecordA0")
        np.testing.assert_array_equal(out.nparray, [0.0, 1.0, 1.0, 1.0, 0.0])

    # --- output array in folder ---

    def test_output_array_created_in_folder(self):
        op = NMMainOpInequality(op=">", a=0.0)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        self.assertIsNotNone(folder.data.get("IQ_RecordA0"))

    def test_output_array_values(self):
        op = NMMainOpInequality(op=">=", a=3.0, binary_output=True)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0]})
        out = folder.data.get("IQ_RecordA0")
        np.testing.assert_array_equal(out.nparray, [0.0, 0.0, 1.0, 1.0])

    def test_xscale_copied(self):
        folder = NMFolder(name="folder0")
        folder.data.new(
            "RecordA0",
            nparray=np.array([1.0, 2.0, 3.0], dtype=float),
            xscale={"start": 0.5, "delta": 0.1, "label": "Time", "units": "ms"},
            yscale={"label": "Vm", "units": "mV"},
        )
        src = folder.data.get("RecordA0")
        op = NMMainOpInequality(op=">", a=1.5)
        op.run_all([(src, None)], folder=folder)
        out = folder.data.get("IQ_RecordA0")
        self.assertAlmostEqual(out.xscale.start, 0.5)
        self.assertAlmostEqual(out.xscale.delta, 0.1)

    # --- results dict ---

    def test_results_populated(self):
        op = NMMainOpInequality(op=">", a=2.0)
        self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        r = op.results["IQ_RecordA0"]
        self.assertEqual(r["successes"], 3)
        self.assertEqual(r["failures"], 2)
        self.assertEqual(r["condition"], "y > 2")

    def test_multiple_arrays(self):
        op = NMMainOpInequality(op=">", a=0.0)
        folder = self._run(op, {
            "RecordA0": [1.0],
            "RecordA1": [2.0],
            "RecordA2": [3.0],
        })
        self.assertIsNotNone(folder.data.get("IQ_RecordA0"))
        self.assertIsNotNone(folder.data.get("IQ_RecordA1"))
        self.assertIsNotNone(folder.data.get("IQ_RecordA2"))
        self.assertEqual(len(op.results), 3)

    # --- validation ---

    def test_range_op_without_b_raises_in_run_init(self):
        op = NMMainOpInequality(op="<<", a=1.0)   # b=None
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            op.run_all([(d, None)], folder=folder)

    def test_unknown_op_raises_in_setter(self):
        with self.assertRaises(ValueError):
            NMMainOpInequality(op="^", a=1.0)

    def test_a_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpInequality(op=">", a=True)

    def test_b_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpInequality(op="<<", a=1.0, b=True)

    def test_b_none_valid(self):
        op = NMMainOpInequality(op=">", a=1.0, b=None)
        self.assertIsNone(op.b)

    def test_skips_non_ndarray(self):
        op = NMMainOpInequality(op=">", a=0.0)
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0")   # no nparray
        op.run_all([(d, None)], folder=folder)
        # no output array created, no error
        self.assertIsNone(folder.data.get("IQ_RecordA0"))

    # --- note ---

    def test_note_written(self):
        op = NMMainOpInequality(op=">", a=2.0)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        out = folder.data.get("IQ_RecordA0")
        notes = [e["note"] for e in out.notes._entries]
        self.assertTrue(any("NMInequality(y > 2)" in n for n in notes))

    # --- registry ---

    def test_inequality_by_name(self):
        op = op_from_name("inequality")
        self.assertIsInstance(op, NMMainOpInequality)


# ===========================================================================
# TestNMMainOpHistogram
# ===========================================================================


class TestNMMainOpHistogram(unittest.TestCase):
    """Tests for NMMainOpHistogram."""

    def _run(self, op, arrays_by_name):
        return _op_run_all(op, arrays_by_name)

    def _run_with_yscale(self, op, arrays_by_name, yscale):
        """Like _run_op_directly but sets yscale on each NMData array."""
        folder = NMFolder(name="folder0")
        data_items = []
        for name, arr in arrays_by_name.items():
            d = folder.data.new(name, nparray=np.array(arr, dtype=float),
                                yscale=yscale)
            data_items.append((d, None))
        op.run_all(data_items, folder)
        return folder

    # --- output array created ---

    def test_output_array_created(self):
        op = NMMainOpHistogram()
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        self.assertIsNotNone(folder.data.get("H_RecordA0"))

    def test_counts_length_equals_bins(self):
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": list(range(20))})
        out = folder.data.get("H_RecordA0")
        self.assertEqual(len(out.nparray), 5)

    def test_counts_sum_equals_n_finite(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": arr})
        out = folder.data.get("H_RecordA0")
        self.assertEqual(int(out.nparray.sum()), len(arr))

    # --- default and custom bins ---

    def test_default_bins_is_10(self):
        self.assertEqual(NMMainOpHistogram().bins, 10)

    def test_custom_bins(self):
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": list(range(10))})
        out = folder.data.get("H_RecordA0")
        self.assertEqual(len(out.nparray), 5)

    def test_list_bins(self):
        op = NMMainOpHistogram(bins=[0.0, 2.0, 4.0, 6.0])
        folder = self._run(op, {"RecordA0": [1.0, 3.0, 5.0]})
        out = folder.data.get("H_RecordA0")
        # 3 bin edges → 3 bins, all values land in one of them
        self.assertEqual(len(out.nparray), 3)

    # --- xrange ---

    def test_xrange(self):
        op = NMMainOpHistogram(bins=3, xrange=(1.0, 4.0))
        folder = self._run(op, {"RecordA0": [0.0, 1.5, 2.5, 3.5, 5.0]})
        out = folder.data.get("H_RecordA0")
        # values outside [1,4] are not counted
        self.assertEqual(int(out.nparray.sum()), 3)

    # --- density ---

    def test_density_output(self):
        arr = list(range(100))
        op = NMMainOpHistogram(bins=10, density=True)
        folder = self._run(op, {"RecordA0": arr})
        out = folder.data.get("H_RecordA0")
        counts = out.nparray
        bin_width = out.xscale.delta
        self.assertAlmostEqual(float((counts * bin_width).sum()), 1.0, places=10)

    # --- xscale ---

    def test_xscale_start(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": arr})
        out = folder.data.get("H_RecordA0")
        import numpy as _np
        _, edges = _np.histogram(arr, bins=5)
        self.assertAlmostEqual(out.xscale.start, float(edges[0]))

    def test_xscale_delta(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": arr})
        out = folder.data.get("H_RecordA0")
        import numpy as _np
        _, edges = _np.histogram(arr, bins=5)
        self.assertAlmostEqual(out.xscale.delta, float(edges[1] - edges[0]))

    def test_xscale_label_from_yscale(self):
        op = NMMainOpHistogram(bins=5)
        yscale = {"label": "Vm", "units": "mV"}
        folder = self._run_with_yscale(op, {"RecordA0": [1.0, 2.0, 3.0]}, yscale)
        out = folder.data.get("H_RecordA0")
        self.assertEqual(out.xscale.label, "Vm")
        self.assertEqual(out.xscale.units, "mV")

    # --- yscale of output ---

    def test_yscale_counts_label(self):
        op = NMMainOpHistogram(density=False)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        out = folder.data.get("H_RecordA0")
        self.assertEqual(out.yscale.label, "counts")

    def test_yscale_density_label(self):
        op = NMMainOpHistogram(density=True)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        out = folder.data.get("H_RecordA0")
        self.assertEqual(out.yscale.label, "density")

    # --- NaN / Inf exclusion ---

    def test_nan_inf_excluded(self):
        arr = [1.0, float("nan"), 3.0, float("inf"), 5.0]
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": arr})
        out = folder.data.get("H_RecordA0")
        # 3 finite values should be counted
        self.assertEqual(int(out.nparray.sum()), 3)

    def test_n_excluded_in_results(self):
        arr = [1.0, float("nan"), 3.0, float("inf")]
        op = NMMainOpHistogram(bins=3)
        folder = self._run(op, {"RecordA0": arr})
        r = op.results["H_RecordA0"]
        self.assertEqual(r["n_excluded"], 2)

    # --- results dict ---

    def test_results_populated(self):
        op = NMMainOpHistogram(bins=5)
        self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        r = op.results["H_RecordA0"]
        self.assertIn("counts", r)
        self.assertIn("edges", r)
        self.assertIn("n_excluded", r)

    def test_multiple_arrays(self):
        op = NMMainOpHistogram(bins=5)
        arrays = {"RecordA0": [1.0, 2.0], "RecordA1": [3.0, 4.0],
                  "RecordA2": [5.0, 6.0]}
        folder = self._run(op, arrays)
        self.assertEqual(len(op.results), 3)
        for name in ("RecordA0", "RecordA1", "RecordA2"):
            self.assertIsNotNone(folder.data.get("H_" + name))

    # --- skips non-nparray ---

    def test_skips_non_nparray(self):
        op = NMMainOpHistogram()
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0")  # no nparray
        op.run_all([(d, None)], folder)
        self.assertIsNone(folder.data.get("H_RecordA0"))

    # --- note ---

    def test_note_written(self):
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        out = folder.data.get("H_RecordA0")
        notes = [e["note"] for e in out.notes._entries]
        self.assertTrue(any("NMHistogram" in n for n in notes))

    # --- time window (x0 / x1) ---

    def test_default_x0_is_neg_inf(self):
        self.assertTrue(math.isinf(NMMainOpHistogram().x0))
        self.assertLess(NMMainOpHistogram().x0, 0)

    def test_default_x1_is_pos_inf(self):
        self.assertTrue(math.isinf(NMMainOpHistogram().x1))
        self.assertGreater(NMMainOpHistogram().x1, 0)

    def test_x0_x1_window(self):
        # 10 samples at x=0..9; window x0=2, x1=4 → 3 samples (indices 2,3,4)
        op = NMMainOpHistogram(bins=3, x0=2.0, x1=4.0)
        folder = NMFolder(name="folder0")
        d = folder.data.new(
            "RecordA0",
            nparray=np.arange(10, dtype=float),
            xscale={"start": 0.0, "delta": 1.0},
        )
        op.run_all([(d, None)], folder)
        out = folder.data.get("H_RecordA0")
        self.assertEqual(int(out.nparray.sum()), 3)

    def test_x1_only(self):
        # window = -inf..4 → samples at x=0..4 = 5 samples
        op = NMMainOpHistogram(bins=5, x1=4.0)
        folder = NMFolder(name="folder0")
        d = folder.data.new(
            "RecordA0",
            nparray=np.arange(10, dtype=float),
            xscale={"start": 0.0, "delta": 1.0},
        )
        op.run_all([(d, None)], folder)
        out = folder.data.get("H_RecordA0")
        self.assertEqual(int(out.nparray.sum()), 5)

    def test_x0_only(self):
        # window = 7.0..+inf → samples at x=7,8,9 = 3 samples
        op = NMMainOpHistogram(bins=3, x0=7.0)
        folder = NMFolder(name="folder0")
        d = folder.data.new(
            "RecordA0",
            nparray=np.arange(10, dtype=float),
            xscale={"start": 0.0, "delta": 1.0},
        )
        op.run_all([(d, None)], folder)
        out = folder.data.get("H_RecordA0")
        self.assertEqual(int(out.nparray.sum()), 3)

    def test_x0_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpHistogram(x0=True)

    def test_x1_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpHistogram(x1=True)

    def test_x0_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpHistogram(x0=float("nan"))

    def test_x1_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpHistogram(x1=float("nan"))

    def test_x1_before_x0_raises(self):
        op = NMMainOpHistogram(x0=5.0, x1=2.0)
        with self.assertRaises(ValueError):
            op.run_init()

    # --- validation ---

    def test_bins_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpHistogram(bins=True)

    def test_bins_rejects_zero(self):
        with self.assertRaises(ValueError):
            NMMainOpHistogram(bins=0)

    def test_density_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpHistogram(density=1)

    # --- registry ---

    def test_histogram_by_name(self):
        op = op_from_name("histogram")
        self.assertIsInstance(op, NMMainOpHistogram)

    # --- out_prefix ---

    def test_default_out_prefix(self):
        self.assertEqual(NMMainOpHistogram().out_prefix, "H_")

    def test_custom_out_prefix(self):
        op = NMMainOpHistogram(bins=5)
        op.out_prefix = "Histo_"
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0]})
        self.assertIsNotNone(folder.data.get("Histo_RecordA0"))
        self.assertIsNone(folder.data.get("H_RecordA0"))

    def test_out_prefix_rejects_non_string(self):
        op = NMMainOpHistogram()
        with self.assertRaises(TypeError):
            op.out_prefix = 123

    # --- overwrite ---

    def test_overwrite_default_is_true(self):
        self.assertTrue(NMMainOpHistogram().overwrite)

    def test_overwrite_true_replaces_existing(self):
        op = NMMainOpHistogram(bins=5)
        folder = self._run(op, {"RecordA0": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out_first = folder.data.get("H_RecordA0")
        first_sum = float(out_first.nparray.sum())
        # run again with different data → same name, updated content
        d2 = folder.data.new("RecordA1", nparray=np.array([10.0, 20.0, 30.0]))
        d2.name  # confirm created
        op2 = NMMainOpHistogram(bins=5)
        op2.overwrite = True
        folder2 = NMFolder(name="folder0")
        d_new = folder2.data.new("RecordA0",
                                 nparray=np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        op2.run_all([(d_new, None)], folder2)
        op2.run_all([(d_new, None)], folder2)  # second run: same name
        out = folder2.data.get("H_RecordA0")
        self.assertIsNotNone(out)  # still one array, not two

    def test_overwrite_false_sequences_from_zero(self):
        op = NMMainOpHistogram(bins=5)
        op.overwrite = False
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0",
                            nparray=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        op.run_all([(d, None)], folder)  # → H_RecordA0_0
        op.run_all([(d, None)], folder)  # → H_RecordA0_1
        op.run_all([(d, None)], folder)  # → H_RecordA0_2
        self.assertIsNotNone(folder.data.get("H_RecordA0_0"))
        self.assertIsNotNone(folder.data.get("H_RecordA0_1"))
        self.assertIsNotNone(folder.data.get("H_RecordA0_2"))
        self.assertIsNone(folder.data.get("H_RecordA0"))  # bare name never created

    def test_overwrite_false_sequence_starts_at_zero(self):
        op = NMMainOpHistogram(bins=5)
        op.overwrite = False
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0",
                            nparray=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        op.run_all([(d, None)], folder)  # first run → H_RecordA0_0
        self.assertIsNotNone(folder.data.get("H_RecordA0_0"))
        self.assertIsNone(folder.data.get("H_RecordA0"))   # bare name not used
        self.assertIsNone(folder.data.get("H_RecordA0_1"))

    def test_overwrite_rejects_non_bool(self):
        op = NMMainOpHistogram()
        with self.assertRaises(TypeError):
            op.overwrite = 1


# ===========================================================================
# TestOverwriteAndPrefixInequality
# ===========================================================================


class TestOverwriteAndPrefixInequality(unittest.TestCase):
    """overwrite / out_prefix behaviour for NMMainOpInequality."""

    def test_default_out_prefix(self):
        self.assertEqual(NMMainOpInequality(op=">", a=0.0).out_prefix, "IQ_")

    def test_custom_out_prefix(self):
        op = NMMainOpInequality(op=">", a=2.0)
        op.out_prefix = "MyIQ_"
        folder = _op_run_all(op, {"RecordA0": [1.0, 3.0]})
        self.assertIsNotNone(folder.data.get("MyIQ_RecordA0"))
        self.assertIsNone(folder.data.get("IQ_RecordA0"))

    def test_overwrite_false_sequences(self):
        op = NMMainOpInequality(op=">", a=0.0)
        op.overwrite = False
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
        op.run_all([(d, None)], folder)  # → IQ_RecordA0_0
        op.run_all([(d, None)], folder)  # → IQ_RecordA0_1
        self.assertIsNotNone(folder.data.get("IQ_RecordA0_0"))
        self.assertIsNotNone(folder.data.get("IQ_RecordA0_1"))
        self.assertIsNone(folder.data.get("IQ_RecordA0"))  # bare name never created


# ===========================================================================
# TestOverwriteAndPrefixAccumulate
# ===========================================================================


class TestOverwriteAndPrefixAccumulate(unittest.TestCase):
    """overwrite / out_prefix behaviour for NMMainOpAccumulate subclasses."""

    def _run_average(self, op, arrays_by_name):
        return _op_run_all(op, arrays_by_name)

    def test_default_out_prefix_average(self):
        self.assertEqual(NMMainOpAverage().out_prefix, "Avg_")

    def test_custom_out_prefix_average(self):
        op = NMMainOpAverage()
        op.out_prefix = "Mean_"
        folder = _op_run_all(op, {"RecordA0": [1.0, 2.0],
                                       "RecordA1": [3.0, 4.0]})
        self.assertIsNotNone(folder.data.get("Mean_RecordA"))
        self.assertIsNone(folder.data.get("Avg_RecordA"))

    def test_overwrite_false_sequences_accumulate(self):
        op = NMMainOpSum()
        op.overwrite = False
        folder = NMFolder(name="folder0")
        d = folder.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
        op.run_all([(d, None)], folder)  # → Sum_RecordA_0
        op.run_all([(d, None)], folder)  # → Sum_RecordA_1
        self.assertIsNotNone(folder.data.get("Sum_RecordA_0"))
        self.assertIsNotNone(folder.data.get("Sum_RecordA_1"))
        self.assertIsNone(folder.data.get("Sum_RecordA"))  # bare name never created

    def test_companion_arrays_share_suffix(self):
        # overwrite=False: companions (Stdv_) always get same suffix as Avg_
        op = NMMainOpAverage(compute_stdv=True)
        op.overwrite = False
        folder = NMFolder(name="folder0")
        d0 = folder.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
        d1 = folder.data.new("RecordA1", nparray=np.array([3.0, 4.0]))
        op.run_all([(d0, None), (d1, None)], folder)  # → Avg_RecordA_0, Stdv_RecordA_0
        op.run_all([(d0, None), (d1, None)], folder)  # → Avg_RecordA_1, Stdv_RecordA_1
        self.assertIsNotNone(folder.data.get("Avg_RecordA_0"))
        self.assertIsNotNone(folder.data.get("Stdv_RecordA_0"))
        self.assertIsNotNone(folder.data.get("Avg_RecordA_1"))
        self.assertIsNotNone(folder.data.get("Stdv_RecordA_1"))
        self.assertIsNone(folder.data.get("Avg_RecordA"))  # bare name never created


class TestNMMainOpDFOF(unittest.TestCase):
    """Tests for NMMainOpDFOF (per_array and average modes)."""

    # ------------------------------------------------------------------
    # per_array mode — basic behaviour

    def test_in_place_modification(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [2.0, 2.0, 4.0], xstart=0.0, xdelta=1.0)
        original_id = id(d)
        op.run_init()
        op.run(d)
        self.assertEqual(id(d), original_id)  # same NMData object
        self.assertFalse(np.array_equal(d.nparray, [2.0, 2.0, 4.0]))

    def test_basic_computation(self):
        # window [0,1] → F0 = mean([1.0, 3.0]) = 2.0
        # dF/F = ([1,3,5] - 2) / 2 = [-0.5, 0.5, 1.5]
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [1.0, 3.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [-0.5, 0.5, 1.5])

    def test_f0_zero_sets_nan(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [0.0, 0.0, 1.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertTrue(np.all(np.isnan(d.nparray)))

    def test_default_window_uses_full_array(self):
        # x0=-inf, x1=+inf → F0 = mean of entire array = 3.0
        op = NMMainOpDFOF()
        d = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # (arr - 3) / 3
        f0 = 3.0
        np.testing.assert_array_almost_equal(
            d.nparray, [(1-f0)/f0, (2-f0)/f0, (3-f0)/f0, (4-f0)/f0, (5-f0)/f0]
        )

    def test_x0_x1_window(self):
        # window [2,3] → indices 2,3 → values [3,4] → F0 = 3.5
        op = NMMainOpDFOF(x0=2.0, x1=3.0)
        d = _make_data("RecordA0", [1.0, 2.0, 3.0, 4.0, 5.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        f0 = 3.5
        np.testing.assert_array_almost_equal(
            d.nparray, [(1-f0)/f0, (2-f0)/f0, (3-f0)/f0, (4-f0)/f0, (5-f0)/f0]
        )

    def test_mode_per_array_independent_f0(self):
        # Two arrays with different baselines → independent F0 per array
        op = NMMainOpDFOF(x0=0.0, x1=0.0)
        d1 = _make_data("RecordA0", [2.0, 4.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [4.0, 8.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1)
        op.run(d2)
        # d1: F0=2 → [0, 1]; d2: F0=4 → [0, 1]
        np.testing.assert_array_almost_equal(d1.nparray, [0.0, 1.0])
        np.testing.assert_array_almost_equal(d2.nparray, [0.0, 1.0])

    # ------------------------------------------------------------------
    # average mode

    def test_mode_average_shared_f0(self):
        # F0 values: 2, 4, 6 → mean F0 = 4 per channel
        op = NMMainOpDFOF(x0=0.0, x1=1.0, mode="average")
        d1 = _make_data("RecordA0", [2.0, 2.0], xstart=0.0, xdelta=1.0)
        d2 = _make_data("RecordA1", [4.0, 4.0], xstart=0.0, xdelta=1.0)
        d3 = _make_data("RecordA2", [6.0, 6.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d1, "A")
        op.run(d2, "A")
        op.run(d3, "A")
        op.run_finish()
        # mean_f0 = 4; dF/F = (val - 4) / 4
        np.testing.assert_array_almost_equal(d1.nparray, [-0.5, -0.5])
        np.testing.assert_array_almost_equal(d2.nparray, [0.0, 0.0])
        np.testing.assert_array_almost_equal(d3.nparray, [0.5, 0.5])

    def test_mode_average_per_channel(self):
        # Channel A: F0 values 2, 4 → mean=3; Channel B: F0=10 → mean=10
        op = NMMainOpDFOF(x0=0.0, x1=0.0, mode="average")
        a0 = _make_data("RecordA0", [2.0, 6.0], xstart=0.0, xdelta=1.0)
        a1 = _make_data("RecordA1", [4.0, 6.0], xstart=0.0, xdelta=1.0)
        b0 = _make_data("RecordB0", [10.0, 20.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(a0, "A")
        op.run(a1, "A")
        op.run(b0, "B")
        op.run_finish()
        # A: mean_f0=3 → (2-3)/3=-1/3, (6-3)/3=1
        np.testing.assert_array_almost_equal(a0.nparray, [-1/3, 1.0])
        np.testing.assert_array_almost_equal(a1.nparray, [1/3, 1.0])
        # B: mean_f0=10 → (10-10)/10=0, (20-10)/10=1
        np.testing.assert_array_almost_equal(b0.nparray, [0.0, 1.0])

    # ------------------------------------------------------------------
    # NaN handling

    def test_ignore_nans_true(self):
        # NaN in baseline window excluded → nanmean([nan, 4]) = 4
        op = NMMainOpDFOF(x0=0.0, x1=1.0, ignore_nans=True)
        d = _make_data("RecordA0", [np.nan, 4.0, 8.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        # F0=4 → (8-4)/4 = 1.0
        self.assertAlmostEqual(float(d.nparray[2]), 1.0)

    def test_ignore_nans_false(self):
        # NaN in window → F0=nan → all-NaN result
        op = NMMainOpDFOF(x0=0.0, x1=1.0, ignore_nans=False)
        d = _make_data("RecordA0", [np.nan, 4.0, 8.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertTrue(np.all(np.isnan(d.nparray)))

    # ------------------------------------------------------------------
    # yscale update

    def test_yscale_label_updated(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [1.0, 2.0, 3.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertEqual(d.yscale.label, "dF/F")

    def test_yscale_units_cleared(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [1.0, 2.0, 3.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertEqual(d.yscale.units, "")

    # ------------------------------------------------------------------
    # Notes

    def test_note_written(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [2.0, 2.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertGreater(len(d.notes), 0)
        self.assertIn("NMDFOF", d.notes[0]["note"])

    def test_note_contains_f0(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0)
        d = _make_data("RecordA0", [2.0, 2.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertIn("F0=", d.notes[0]["note"])

    def test_note_mode_per_array(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0, mode="per_array")
        d = _make_data("RecordA0", [2.0, 2.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d)
        self.assertIn("per_array", d.notes[0]["note"])

    def test_note_mode_average(self):
        op = NMMainOpDFOF(x0=0.0, x1=1.0, mode="average")
        d = _make_data("RecordA0", [2.0, 2.0, 4.0], xstart=0.0, xdelta=1.0)
        op.run_init()
        op.run(d, "A")
        op.run_finish()
        self.assertGreater(len(d.notes), 0)
        self.assertIn("average", d.notes[0]["note"])

    # ------------------------------------------------------------------
    # Edge cases

    def test_skips_non_ndarray(self):
        op = NMMainOpDFOF()
        d = NMData(NM, name="RecordA0")  # no nparray
        op.run_init()
        op.run(d)  # should not raise

    def test_multiple_waves(self):
        op = NMMainOpDFOF(x0=0.0, x1=0.0)
        waves = [
            _make_data("RecordA0", [1.0, 5.0], xstart=0.0, xdelta=1.0),
            _make_data("RecordA1", [2.0, 6.0], xstart=0.0, xdelta=1.0),
            _make_data("RecordA2", [4.0, 8.0], xstart=0.0, xdelta=1.0),
        ]
        op.run_init()
        for w in waves:
            op.run(w)
        # Each wave transformed independently
        np.testing.assert_array_almost_equal(waves[0].nparray, [0.0, 4.0])
        np.testing.assert_array_almost_equal(waves[1].nparray, [0.0, 2.0])
        np.testing.assert_array_almost_equal(waves[2].nparray, [0.0, 1.0])

    # ------------------------------------------------------------------
    # Validation

    def test_x0_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpDFOF(x0=True)

    def test_x1_rejects_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpDFOF(x1=True)

    def test_x0_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpDFOF(x0=float("nan"))

    def test_x1_rejects_nan(self):
        with self.assertRaises(ValueError):
            NMMainOpDFOF(x1=float("nan"))

    def test_x1_before_x0_raises(self):
        op = NMMainOpDFOF(x0=5.0, x1=2.0)
        with self.assertRaises(ValueError):
            op.run_init()

    def test_mode_rejects_invalid(self):
        with self.assertRaises(ValueError):
            NMMainOpDFOF(mode="median")

    def test_mode_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpDFOF(mode=1)

    def test_ignore_nans_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            NMMainOpDFOF(ignore_nans=1)

    def test_dfof_by_name(self):
        op = op_from_name("dfof")
        self.assertIsInstance(op, NMMainOpDFOF)


# ---------------------------------------------------------------------------
# TestNMMainOpRescale
# ---------------------------------------------------------------------------


class TestNMMainOpRescale(unittest.TestCase):
    """Tests for NMMainOpRescale."""

    # ------------------------------------------------------------------
    # Basic rescaling

    def test_basic_rescale(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="nA")
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1e-3, 2e-3])

    def test_scale_factor_up(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        d.yscale.units = "V"
        op = NMMainOpRescale(to_units="mV")
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1e3, 2e3])

    def test_same_units_factor_one(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="pA")
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1.0, 2.0])

    # ------------------------------------------------------------------
    # yscale update

    def test_yscale_units_updated(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="nA")
        op.run_init()
        op.run(d)
        self.assertEqual(d.yscale.units, "nA")

    # ------------------------------------------------------------------
    # from_units

    def test_from_units_auto_detected(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        d.yscale.units = "mV"
        op = NMMainOpRescale(to_units="V")
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1e-3, 2e-3])

    def test_from_units_explicit(self):
        d = _make_data("RecordA0", [1.0, 2.0])
        d.yscale.units = ""   # yscale empty — explicit from_units overrides
        op = NMMainOpRescale(to_units="nA", from_units="pA")
        op.run_init()
        op.run(d)
        np.testing.assert_array_almost_equal(d.nparray, [1e-3, 2e-3])

    def test_from_units_empty_raises(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = ""
        op = NMMainOpRescale(to_units="nA")  # from_units=None, yscale empty
        op.run_init()
        with self.assertRaises(ValueError):
            op.run(d)

    # ------------------------------------------------------------------
    # Validation

    def test_to_units_empty_raises(self):
        op = NMMainOpRescale(to_units="")
        with self.assertRaises(ValueError):
            op.run_init()

    def test_base_mismatch_raises(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="mV")
        op.run_init()
        with self.assertRaises(ValueError):
            op.run(d)

    def test_to_units_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpRescale(to_units=123)

    def test_from_units_rejects_non_string_non_none(self):
        with self.assertRaises(TypeError):
            NMMainOpRescale(to_units="nA", from_units=123)

    # ------------------------------------------------------------------
    # Notes

    def test_note_written(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="nA")
        op.run_init()
        op.run(d)
        self.assertGreater(len(d.notes), 0)
        note = d.notes[0]["note"]
        self.assertIn("NMRescale", note)

    def test_note_contains_factor(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = "pA"
        op = NMMainOpRescale(to_units="nA")
        op.run_init()
        op.run(d)
        note = d.notes[0]["note"]
        self.assertIn("factor=", note)

    # ------------------------------------------------------------------
    # Edge cases

    def test_skips_non_ndarray(self):
        d = _make_data("RecordA0", [1.0])
        d.yscale.units = "pA"
        d.nparray = None   # not an ndarray
        op = NMMainOpRescale(to_units="nA")
        op.run_init()
        op.run(d)   # should not raise

    def test_multiple_waves(self):
        waves = [
            _make_data("RecordA0", [1.0, 2.0]),
            _make_data("RecordA1", [3.0, 4.0]),
            _make_data("RecordA2", [5.0, 6.0]),
        ]
        for w in waves:
            w.yscale.units = "mV"
        op = NMMainOpRescale(to_units="V")
        op.run_init()
        for w in waves:
            op.run(w)
        np.testing.assert_array_almost_equal(waves[0].nparray, [1e-3, 2e-3])
        np.testing.assert_array_almost_equal(waves[1].nparray, [3e-3, 4e-3])
        np.testing.assert_array_almost_equal(waves[2].nparray, [5e-3, 6e-3])
        for w in waves:
            self.assertEqual(w.yscale.units, "V")

    def test_rescale_by_name(self):
        op = op_from_name("rescale")
        self.assertIsInstance(op, NMMainOpRescale)


# ---------------------------------------------------------------------------
# TestNMMainOpRescaleX
# ---------------------------------------------------------------------------


class TestNMMainOpRescaleX(unittest.TestCase):
    """Tests for NMMainOpRescaleX."""

    # ------------------------------------------------------------------
    # Basic rescaling

    def test_basic_rescale(self):
        d = _make_data("RecordA0", [1.0, 2.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.start, 0.0)
        self.assertAlmostEqual(d.xscale.delta, 1e-3)

    def test_scale_factor_up(self):
        d = _make_data("RecordA0", [1.0, 2.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "s"
        op = NMMainOpRescaleX(to_units="ms")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.start, 0.0)
        self.assertAlmostEqual(d.xscale.delta, 1e3)

    def test_same_units_factor_one(self):
        d = _make_data("RecordA0", [1.0, 2.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="ms")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.start, 0.0)
        self.assertAlmostEqual(d.xscale.delta, 1.0)

    # ------------------------------------------------------------------
    # xscale field updates

    def test_xscale_start_updated(self):
        d = _make_data("RecordA0", [1.0], xstart=10.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.start, 10e-3)

    def test_xscale_delta_updated(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=0.1)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.delta, 0.1e-3)

    def test_xscale_units_updated(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertEqual(d.xscale.units, "s")

    # ------------------------------------------------------------------
    # from_units

    def test_from_units_auto_detected(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.delta, 1e-3)

    def test_from_units_explicit(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = ""   # empty — explicit from_units overrides
        op = NMMainOpRescaleX(to_units="s", from_units="ms")
        op.run_init()
        op.run(d)
        self.assertAlmostEqual(d.xscale.delta, 1e-3)

    def test_from_units_empty_raises(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = ""
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        with self.assertRaises(ValueError):
            op.run(d)

    # ------------------------------------------------------------------
    # Validation

    def test_to_units_empty_raises(self):
        op = NMMainOpRescaleX(to_units="")
        with self.assertRaises(ValueError):
            op.run_init()

    def test_base_mismatch_raises(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="V")
        op.run_init()
        with self.assertRaises(ValueError):
            op.run(d)

    def test_to_units_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpRescaleX(to_units=123)

    def test_from_units_rejects_non_string_non_none(self):
        with self.assertRaises(TypeError):
            NMMainOpRescaleX(to_units="s", from_units=123)

    # ------------------------------------------------------------------
    # Notes

    def test_note_written(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        self.assertGreater(len(d.notes), 0)
        note = d.notes[0]["note"]
        self.assertIn("NMRescaleX", note)

    def test_note_contains_factor(self):
        d = _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0)
        d.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        op.run(d)
        note = d.notes[0]["note"]
        self.assertIn("factor=", note)

    # ------------------------------------------------------------------
    # Other

    def test_multiple_waves(self):
        waves = [
            _make_data("RecordA0", [1.0], xstart=0.0, xdelta=1.0),
            _make_data("RecordA1", [2.0], xstart=0.0, xdelta=2.0),
            _make_data("RecordA2", [3.0], xstart=10.0, xdelta=0.5),
        ]
        for w in waves:
            w.xscale.units = "ms"
        op = NMMainOpRescaleX(to_units="s")
        op.run_init()
        for w in waves:
            op.run(w)
        self.assertAlmostEqual(waves[0].xscale.delta, 1e-3)
        self.assertAlmostEqual(waves[1].xscale.delta, 2e-3)
        self.assertAlmostEqual(waves[2].xscale.start, 10e-3)
        self.assertAlmostEqual(waves[2].xscale.delta, 0.5e-3)
        for w in waves:
            self.assertEqual(w.xscale.units, "s")

    def test_rescale_x_by_name(self):
        op = op_from_name("rescale_x")
        self.assertIsInstance(op, NMMainOpRescaleX)


# ---------------------------------------------------------------------------
# TestNMMainOpConcatenate
# ---------------------------------------------------------------------------


class TestNMMainOpConcatenate(unittest.TestCase):
    """Tests for NMMainOpConcatenate (1d and 2d modes)."""

    def setUp(self):
        self.op = NMMainOpConcatenate()
        self.arrays = {
            "RecordA0": [1.0, 2.0, 3.0],
            "RecordA1": [4.0, 5.0, 6.0],
        }

    def _run(self, arrays=None, op=None):
        if arrays is None:
            arrays = self.arrays
        if op is None:
            op = self.op
        return _op_run_all(op, arrays)

    # ------------------------------------------------------------------
    # 1D mode — basic

    def test_1d_values(self):
        folder = self._run()
        out = folder.data.get("Cat_RecordA")
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out.nparray, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_1d_unequal_lengths(self):
        arrays = {"RecordA0": [1.0, 2.0, 3.0], "RecordA1": [4.0, 5.0]}
        folder = self._run(arrays)
        out = folder.data.get("Cat_RecordA")
        np.testing.assert_array_almost_equal(out.nparray, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_1d_output_length(self):
        folder = self._run()
        out = folder.data.get("Cat_RecordA")
        self.assertEqual(len(out.nparray), 6)  # 3 + 3

    # ------------------------------------------------------------------
    # 2D mode — basic

    def test_2d_shape(self):
        op = NMMainOpConcatenate(mode="2d")
        arrays = {
            "RecordA0": [1.0, 2.0, 3.0, 4.0],
            "RecordA1": [5.0, 6.0, 7.0, 8.0],
            "RecordA2": [9.0, 10.0, 11.0, 12.0],
        }
        folder = self._run(arrays, op)
        out = folder.data.get("Cat_RecordA")
        self.assertIsNotNone(out)
        self.assertEqual(out.nparray.shape, (3, 4))

    def test_2d_values(self):
        op = NMMainOpConcatenate(mode="2d")
        arrays = {"RecordA0": [1.0, 2.0], "RecordA1": [3.0, 4.0]}
        folder = self._run(arrays, op)
        out = folder.data.get("Cat_RecordA")
        np.testing.assert_array_almost_equal(out.nparray[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(out.nparray[1], [3.0, 4.0])

    def test_2d_unequal_lengths_pads_nan(self):
        op = NMMainOpConcatenate(mode="2d")
        arrays = {"RecordA0": [1.0, 2.0, 3.0], "RecordA1": [4.0, 5.0]}
        folder = self._run(arrays, op)
        out = folder.data.get("Cat_RecordA")
        self.assertEqual(out.nparray.shape, (2, 3))
        np.testing.assert_array_almost_equal(out.nparray[0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(out.nparray[1, 0], 4.0)
        self.assertAlmostEqual(out.nparray[1, 1], 5.0)
        self.assertTrue(np.isnan(out.nparray[1, 2]))

    def test_2d_equal_lengths_no_nan(self):
        op = NMMainOpConcatenate(mode="2d")
        arrays = {"RecordA0": [1.0, 2.0], "RecordA1": [3.0, 4.0]}
        folder = self._run(arrays, op)
        out = folder.data.get("Cat_RecordA")
        self.assertFalse(np.any(np.isnan(out.nparray)))

    # ------------------------------------------------------------------
    # Output naming and metadata

    def test_output_name(self):
        folder = self._run()
        self.assertIsNotNone(folder.data.get("Cat_RecordA"))

    def test_output_name_in_results(self):
        self._run()
        self.assertIn("A", self.op.results)
        self.assertEqual(self.op.results["A"], "Cat_RecordA")

    def test_yscale_copied_from_first(self):
        folder = self._run()
        out = folder.data.get("Cat_RecordA")
        self.assertIsNotNone(out.yscale)

    def test_xscale_copied_from_first(self):
        folder = self._run()
        out = folder.data.get("Cat_RecordA")
        self.assertIsNotNone(out.xscale)

    # ------------------------------------------------------------------
    # Multi-channel

    def test_two_channels(self):
        arrays = {
            "RecordA0": [1.0, 2.0],
            "RecordA1": [3.0, 4.0],
            "RecordB0": [5.0, 6.0],
            "RecordB1": [7.0, 8.0],
        }
        folder = self._run(arrays)
        self.assertIsNotNone(folder.data.get("Cat_RecordA"))
        self.assertIsNotNone(folder.data.get("Cat_RecordB"))

    # ------------------------------------------------------------------
    # Notes

    def test_note_written_1d(self):
        folder = self._run()
        out = folder.data.get("Cat_RecordA")
        note = out.notes[0]["note"]
        self.assertIn("NMConcatenate_1d", note)

    def test_note_written_2d(self):
        op = NMMainOpConcatenate(mode="2d")
        folder = self._run(op=op)
        out = folder.data.get("Cat_RecordA")
        note = out.notes[0]["note"]
        self.assertIn("NMConcatenate_2d", note)

    # ------------------------------------------------------------------
    # Validation

    def test_mode_rejects_invalid(self):
        with self.assertRaises(ValueError):
            NMMainOpConcatenate(mode="3d")

    def test_mode_rejects_non_string(self):
        with self.assertRaises(TypeError):
            NMMainOpConcatenate(mode=1)

    def test_default_mode_is_1d(self):
        self.assertEqual(NMMainOpConcatenate().mode, "1d")

    # ------------------------------------------------------------------
    # Registry

    def test_concatenate_by_name(self):
        op = op_from_name("concatenate")
        self.assertIsInstance(op, NMMainOpConcatenate)

    # ------------------------------------------------------------------
    # Overwrite

    def test_overwrite_false_creates_new(self):
        op = NMMainOpConcatenate(mode="1d")
        op.overwrite = False
        _op_run_all(op, self.arrays)   # creates Cat_RecordA_0
        _op_run_all(op, self.arrays)   # creates Cat_RecordA_1
        # Both _0 and _1 must exist (fresh folder each call, but op tracks state)
        # Verify second run produced a _0-suffixed name
        self.assertIn("A", op.results)
        self.assertTrue(op.results["A"].startswith("Cat_RecordA"))


# ===========================================================================
# TestCommandHistory — to_command_str() and run_all() history hook
# ===========================================================================


class TestCommandHistory(unittest.TestCase):
    """Test that each NMMainOp subclass records an executable command string."""

    def setUp(self):
        from pyneuromatic.core.nm_command_history import (
            NMCommandHistory,
            set_command_history,
        )
        self._cmd_history = NMCommandHistory(quiet=True, log_to_nm_history=False)
        set_command_history(self._cmd_history)

    def tearDown(self):
        from pyneuromatic.core.nm_command_history import (
            NMCommandHistory,
            set_command_history,
        )
        set_command_history(NMCommandHistory(quiet=True, log_to_nm_history=False))

    def _arrays(self):
        return {
            "RecordA0": [1.0, 2.0, 3.0],
            "RecordA1": [4.0, 5.0, 6.0],
        }

    def _run(self, op, arrays=None):
        return _op_run_all(op, arrays or self._arrays())

    # ------------------------------------------------------------------
    # base: to_command_str returns None by default

    def test_base_to_command_str_returns_none(self):
        class _NopOp(NMMainOp):
            def run(self, data, channel_name=None):
                pass
        op = _NopOp()
        result = op.to_command_str("f", "R", ["A"], [0])
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # run_all hook: each op logs to history

    def test_arithmetic_logs_command(self):
        self._run(NMMainOpArithmetic(factor=2.0, op="x"))
        self.assertEqual(len(self._cmd_history), 1)

    def test_arithmetic_by_array_logs_command(self):
        self._run(NMMainOpArithmeticByArray(ref=np.ones(3), op="+"))
        self.assertEqual(len(self._cmd_history), 1)

    def test_baseline_logs_command(self):
        self._run(NMMainOpBaseline(x0=0.0, x1=1.0))
        self.assertEqual(len(self._cmd_history), 1)

    def test_delete_nans_logs_command(self):
        self._run(NMMainOpDeleteNaNs())
        self.assertEqual(len(self._cmd_history), 1)

    def test_delete_points_logs_command(self):
        self._run(NMMainOpDeletePoints(index=0, n_points=1))
        self.assertEqual(len(self._cmd_history), 1)

    def test_differentiate_logs_command(self):
        self._run(NMMainOpDifferentiate())
        self.assertEqual(len(self._cmd_history), 1)

    def test_integrate_logs_command(self):
        self._run(NMMainOpIntegrate())
        self.assertEqual(len(self._cmd_history), 1)

    def test_insert_points_logs_command(self):
        self._run(NMMainOpInsertPoints(index=0, n_points=1))
        self.assertEqual(len(self._cmd_history), 1)

    def test_redimension_logs_command(self):
        self._run(NMMainOpRedimension(n_points=2))
        self.assertEqual(len(self._cmd_history), 1)

    def test_replace_values_logs_command(self):
        self._run(NMMainOpReplaceValues(old_value=0.0, new_value=1.0))
        self.assertEqual(len(self._cmd_history), 1)

    def test_reverse_logs_command(self):
        self._run(NMMainOpReverse())
        self.assertEqual(len(self._cmd_history), 1)

    def test_rotate_logs_command(self):
        self._run(NMMainOpRotate(n_points=1))
        self.assertEqual(len(self._cmd_history), 1)

    def test_average_logs_command(self):
        self._run(NMMainOpAverage())
        self.assertEqual(len(self._cmd_history), 1)

    def test_sum_logs_command(self):
        self._run(NMMainOpSum())
        self.assertEqual(len(self._cmd_history), 1)

    def test_sum_sqr_logs_command(self):
        self._run(NMMainOpSumSqr())
        self.assertEqual(len(self._cmd_history), 1)

    def test_min_logs_command(self):
        self._run(NMMainOpMin())
        self.assertEqual(len(self._cmd_history), 1)

    def test_max_logs_command(self):
        self._run(NMMainOpMax())
        self.assertEqual(len(self._cmd_history), 1)

    def test_concatenate_logs_command(self):
        self._run(NMMainOpConcatenate())
        self.assertEqual(len(self._cmd_history), 1)

    def test_inequality_logs_command(self):
        self._run(NMMainOpInequality(op=">", a=0.0))
        self.assertEqual(len(self._cmd_history), 1)

    def test_histogram_logs_command(self):
        self._run(NMMainOpHistogram())
        self.assertEqual(len(self._cmd_history), 1)

    def test_normalize_logs_command(self):
        self._run(NMMainOpNormalize(x0_min=0.0, x1_min=1.0, x0_max=1.0, x1_max=3.0))
        self.assertEqual(len(self._cmd_history), 1)

    def test_rescale_logs_command(self):
        arrays = {"RecordA0": [1.0, 2.0, 3.0]}
        folder = NMFolder(name="folder0")
        data_items = []
        for name, arr in arrays.items():
            d = folder.data.new(name, nparray=np.array(arr, dtype=float))
            d.yscale.units = "mV"
            data_items.append((d, None))
        op = NMMainOpRescale(to_units="V", from_units="mV")
        op.run_all(data_items, folder)
        self.assertEqual(len(self._cmd_history), 1)

    def test_rescale_x_logs_command(self):
        arrays = {"RecordA0": [1.0, 2.0, 3.0]}
        folder = NMFolder(name="folder0")
        data_items = []
        for name, arr in arrays.items():
            d = folder.data.new(name, nparray=np.array(arr, dtype=float))
            d.xscale.units = "ms"
            data_items.append((d, None))
        op = NMMainOpRescaleX(to_units="s", from_units="ms")
        op.run_all(data_items, folder)
        self.assertEqual(len(self._cmd_history), 1)

    def test_dfof_logs_command(self):
        self._run(NMMainOpDFOF())
        self.assertEqual(len(self._cmd_history), 1)

    # ------------------------------------------------------------------
    # command string content

    def test_command_str_contains_class_name(self):
        self._run(NMMainOpBaseline(x0=0.0, x1=1.0))
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("NMMainOpBaseline", cmd)

    def test_command_str_contains_op_params(self):
        self._run(NMMainOpBaseline(x0=0.5, x1=2.5, mode="per_array"))
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("0.5", cmd)
        self.assertIn("2.5", cmd)
        self.assertIn("per_array", cmd)

    def test_command_str_contains_folder_name(self):
        self._run(NMMainOpBaseline())
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("folder0", cmd)

    def test_command_str_contains_prefix(self):
        # run_all() is called with an explicit prefix so it appears in the command
        folder = NMFolder(name="folder0")
        data_items = []
        for name, arr in self._arrays().items():
            d = folder.data.new(name, nparray=np.array(arr, dtype=float))
            data_items.append((d, None))
        NMMainOpBaseline().run_all(data_items, folder, prefix="Record")
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("Record", cmd)

    def test_command_str_contains_channel(self):
        self._run(NMMainOpBaseline())
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("A", cmd)

    def test_command_str_contains_epochs(self):
        self._run(NMMainOpBaseline())
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("0", cmd)
        self.assertIn("1", cmd)

    def test_command_str_for_average(self):
        op = NMMainOpAverage(ignore_nans=True, compute_stdv=True)
        self._run(op)
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("NMMainOpAverage", cmd)
        self.assertIn("ignore_nans=True", cmd)
        self.assertIn("compute_stdv=True", cmd)

    def test_command_str_for_arithmetic(self):
        op = NMMainOpArithmetic(factor=3.0, op="/")
        self._run(op)
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("NMMainOpArithmetic", cmd)
        self.assertIn("3.0", cmd)
        self.assertIn("/", cmd)

    def test_command_str_for_concatenate(self):
        op = NMMainOpConcatenate(mode="2d")
        self._run(op)
        cmd = self._cmd_history.buffer[0]["command"]
        self.assertIn("NMMainOpConcatenate", cmd)
        self.assertIn("2d", cmd)

    # ------------------------------------------------------------------
    # disabled history

    def test_history_disabled_no_log(self):
        from pyneuromatic.core.nm_command_history import disable_command_history
        disable_command_history()
        self._run(NMMainOpBaseline())
        self.assertEqual(len(self._cmd_history), 0)

    # ------------------------------------------------------------------
    # multiple runs accumulate

    def test_two_runs_log_two_entries(self):
        self._run(NMMainOpBaseline(x0=0.0, x1=1.0))
        self._run(NMMainOpAverage())
        self.assertEqual(len(self._cmd_history), 2)


# ===========================================================================
# TestToolMainOpLogging
# ===========================================================================

class TestToolMainOpLogging(unittest.TestCase):
    """Tests for command history logging of NMToolMain.op setter."""

    def setUp(self):
        from pyneuromatic.core.nm_command_history import (
            NMCommandHistory, set_command_history, get_command_history,
        )
        self._ch = NMCommandHistory(quiet=True, log_to_nm_history=False)
        set_command_history(self._ch)

    def test_op_setter_logs_command(self):
        tool = NMToolMain()
        self._ch.clear()
        tool.op = NMMainOpBaseline(x0=0.0, x1=10.0)
        buf = self._ch.buffer
        self.assertEqual(len(buf), 1)
        self.assertIn("NMMainOpBaseline", buf[0]["command"])
        self.assertIn("tool_main.op", buf[0]["command"])

    def test_op_setter_includes_params(self):
        tool = NMToolMain()
        self._ch.clear()
        tool.op = NMMainOpBaseline(x0=0.0, x1=10.0, mode='per_array')
        cmd = self._ch.buffer[0]["command"]
        self.assertIn("x0=0.0", cmd)
        self.assertIn("x1=10.0", cmd)
        self.assertIn("mode='per_array'", cmd)

    def test_op_setter_string_name_logs_resolved_class(self):
        tool = NMToolMain()
        self._ch.clear()
        tool.op = "average"
        cmd = self._ch.buffer[0]["command"]
        self.assertIn("NMMainOpAverage", cmd)


if __name__ == "__main__":
    unittest.main()
