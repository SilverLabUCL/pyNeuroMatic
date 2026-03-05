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
    NMMainOpScale,
    op_from_name,
)
from pyneuromatic.analysis.nm_tool_main import NMToolMain

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


if __name__ == "__main__":
    unittest.main()
