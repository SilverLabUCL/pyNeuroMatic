#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMFolder and NMFolderContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeriesContainer
from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_utilities as nmu


class NMFolderTestBase(unittest.TestCase):
    """Base class with common setUp for NMFolder tests."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.folder = NMFolder(parent=self.nm, name="TestFolder")


class TestNMFolderInit(NMFolderTestBase):
    """Tests for NMFolder initialization."""

    def test_init_with_defaults(self):
        folder = NMFolder()
        self.assertEqual(folder.name, "NMFolder0")
        self.assertIsInstance(folder.data, NMDataContainer)
        self.assertIsInstance(folder.dataseries, NMDataSeriesContainer)

    def test_init_with_name(self):
        folder = NMFolder(parent=self.nm, name="MyFolder")
        self.assertEqual(folder.name, "MyFolder")
        self.assertEqual(folder._parent, self.nm)

    def test_init_rejects_invalid_copy_arg(self):
        obj = NMObject()
        with self.assertRaises(TypeError):
            NMFolder(copy=obj)

    def test_data_property(self):
        self.assertIsInstance(self.folder.data, NMDataContainer)

    def test_dataseries_property(self):
        self.assertIsInstance(self.folder.dataseries, NMDataSeriesContainer)

    def test_data_container_works(self):
        names = ["data0", "data1", "data2"]
        for n in names:
            self.folder.data.new(n)
        self.assertEqual(len(self.folder.data), 3)
        self.assertEqual(list(self.folder.data.keys()), names)

    def test_data_sets_work(self):
        for i in range(4):
            self.folder.data.new(f"data{i}")
        self.folder.data.sets.add("even", ["data0", "data2"])
        self.folder.data.sets.add("odd", ["data1", "data3"])
        self.folder.data.sets.define_or("all", "even", "odd")

        result = self.folder.data.sets.get_items("all", get_keys=True)
        self.assertEqual(sorted(result), ["data0", "data1", "data2", "data3"])


class TestNMFolderEquality(NMFolderTestBase):
    """Tests for NMFolder equality comparison."""

    def setUp(self):
        super().setUp()
        for i in range(3):
            self.folder.data.new(f"data{i}")

    def test_not_equal_to_bad_types(self):
        for bad in nmu.BADTYPES:
            self.assertFalse(self.folder == bad)

    def test_same_instance_is_same(self):
        self.assertTrue(self.folder is self.folder)

    def test_different_folders_not_equal(self):
        other = NMFolder(parent=self.nm, name="OtherFolder")
        self.assertFalse(self.folder == other)

    def test_copy_equals_original(self):
        c = self.folder.copy()
        self.assertFalse(c is self.folder)
        self.assertTrue(c == self.folder)

    def test_modified_copy_not_equal(self):
        self.folder.data.sets.add("set0", ["data0"])
        c = self.folder.copy()
        self.assertTrue(c == self.folder)

        c.data.sets.remove("set0", "data0")
        self.assertFalse(c == self.folder)

        c.data.sets.add("set0", "data0")
        self.assertTrue(c == self.folder)


class TestNMFolderNotes(NMFolderTestBase):
    """Tests for NMFolder notes integration with NMNotes."""

    def test_notes_returns_nmnotes(self):
        self.assertIsInstance(self.folder.notes, NMNotes)

    def test_notes_initially_empty(self):
        self.assertEqual(len(self.folder.notes), 0)

    def test_notes_add(self):
        self.folder.notes.add("first note")
        self.folder.notes.add("second note")
        self.assertEqual(len(self.folder.notes), 2)
        self.assertEqual(self.folder.notes[0].get("note"), "first note")

    def test_notes_affect_equality(self):
        self.folder.notes.add("test note")
        c = self.folder.copy()
        self.assertTrue(c == self.folder)
        c.notes.add("another note")
        self.assertFalse(c == self.folder)


class TestNMFolderMetadata(NMFolderTestBase):
    """Tests for NMFolder metadata functionality."""

    def test_metadata_initially_empty(self):
        self.assertIsInstance(self.folder.metadata, dict)
        self.assertEqual(len(self.folder.metadata), 0)

    def test_metadata_populate(self):
        self.folder.metadata["root"] = {"AcqMode": "episodic", "NumWaves": 19}
        self.assertEqual(self.folder.metadata["root"]["AcqMode"], "episodic")
        self.assertEqual(self.folder.metadata["root"]["NumWaves"], 19)

    def test_metadata_nested(self):
        self.folder.metadata["root"] = {"FileFormat": 1.72}
        self.folder.metadata["Notes"] = {"H_Name": "cell1", "F_Temp": 34.0}
        self.assertEqual(len(self.folder.metadata), 2)
        self.assertIn("root", self.folder.metadata)
        self.assertIn("Notes", self.folder.metadata)

    def test_metadata_affects_equality(self):
        self.folder.metadata["root"] = {"key": "value"}
        c = self.folder.copy()
        self.assertTrue(c == self.folder)
        c.metadata["root"]["key"] = "different"
        self.assertFalse(c == self.folder)

    def test_metadata_deepcopy_independent(self):
        self.folder.metadata["root"] = {"key": "value"}
        c = copy.deepcopy(self.folder)
        c.metadata["root"]["key"] = "changed"
        self.assertEqual(self.folder.metadata["root"]["key"], "value")


class TestNMFolderDeepCopy(NMFolderTestBase):
    """Tests for NMFolder deep copy."""

    def setUp(self):
        super().setUp()
        for i in range(3):
            self.folder.data.new(f"data{i}")
        self.folder.notes.add("test note")

    def test_deepcopy_creates_new_instance(self):
        c = copy.deepcopy(self.folder)
        self.assertIsNot(c, self.folder)

    def test_deepcopy_preserves_name(self):
        c = copy.deepcopy(self.folder)
        self.assertEqual(c.name, self.folder.name)

    def test_deepcopy_creates_new_data_container(self):
        c = copy.deepcopy(self.folder)
        self.assertIsNot(c.data, self.folder.data)

    def test_deepcopy_preserves_data_count(self):
        c = copy.deepcopy(self.folder)
        self.assertEqual(len(c.data), len(self.folder.data))

    def test_deepcopy_preserves_notes(self):
        c = copy.deepcopy(self.folder)
        self.assertEqual(len(c.notes), len(self.folder.notes))


class TestNMFolderContainer(unittest.TestCase):
    """Tests for NMFolderContainer."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.container = NMFolderContainer(parent=self.nm)

    def test_content_type(self):
        self.assertEqual(self.container.content_type(), "NMFolder")

    def test_new_creates_folder(self):
        folder = self.container.new(name="folder0")
        self.assertIsInstance(folder, NMFolder)
        self.assertEqual(folder.name, "folder0")

    def test_new_with_select(self):
        folder = self.container.new(name="folder0", select=True)
        self.assertEqual(self.container.selected_name, "folder0")

    def test_new_sets_correct_parent(self):
        # Parent should be container's parent (NMManager), not container itself
        folder = self.container.new(name="folder0")
        self.assertEqual(folder._parent, self.nm)

    def test_auto_naming(self):
        f0 = self.container.new()
        f1 = self.container.new()
        self.assertEqual(f0.name, "folder0")
        self.assertEqual(f1.name, "folder1")

    def test_multiple_folders(self):
        self.container.new(name="folder0")
        self.container.new(name="folder1")
        self.assertEqual(len(self.container), 2)
        self.assertIn("folder0", self.container)
        self.assertIn("folder1", self.container)


# =============================================================================
# Tests for detect_prefixes method
# =============================================================================

class TestNMFolderDetectPrefixes(NMFolderTestBase):
    """Tests for NMFolder.detect_prefixes() method."""

    def test_empty_folder(self):
        result = self.folder.detect_prefixes()
        self.assertEqual(result, [])

    def test_single_prefix(self):
        for name in ["RecordA0", "RecordA1", "RecordB0", "RecordB1"]:
            self.folder.data.new(name)
        result = self.folder.detect_prefixes()
        self.assertEqual(result, ["Record"])

    def test_multiple_prefixes(self):
        for name in ["RecordA0", "RecordB0", "avgA0", "avgB0"]:
            self.folder.data.new(name)
        result = self.folder.detect_prefixes()
        self.assertEqual(result, ["Record", "avg"])

    def test_ignores_non_matching_names(self):
        self.folder.data.new("RecordA0")
        self.folder.data.new("SomeOtherData")  # no channel/epoch pattern
        self.folder.data.new("avgB1")
        result = self.folder.detect_prefixes()
        self.assertEqual(result, ["Record", "avg"])


# =============================================================================
# Tests for assemble_dataseries method
# =============================================================================

class TestNMFolderAssembleDataSeries(NMFolderTestBase):
    """Tests for NMFolder.assemble_dataseries() method."""

    def setUp(self):
        super().setUp()
        # Create data with standard naming pattern
        for ch in ["A", "B"]:
            for ep in range(3):
                name = f"Record{ch}{ep}"
                self.folder.data.new(name)

    def test_creates_dataseries(self):
        ds = self.folder.assemble_dataseries("Record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_creates_channels(self):
        ds = self.folder.assemble_dataseries("Record")
        self.assertEqual(len(ds.channels), 2)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)

    def test_creates_epochs(self):
        ds = self.folder.assemble_dataseries("Record")
        self.assertEqual(len(ds.epochs), 3)
        self.assertIn("E0", ds.epochs)
        self.assertIn("E1", ds.epochs)
        self.assertIn("E2", ds.epochs)

    def test_links_data_to_channels(self):
        ds = self.folder.assemble_dataseries("Record")
        chan_a = ds.channels.get("A")
        self.assertEqual(len(chan_a.data), 3)
        data_names = [d.name for d in chan_a.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordA1", data_names)
        self.assertIn("RecordA2", data_names)

    def test_links_data_to_epochs(self):
        ds = self.folder.assemble_dataseries("Record")
        epoch_0 = ds.epochs.get("E0")
        self.assertEqual(len(epoch_0.data), 2)
        data_names = [d.name for d in epoch_0.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordB0", data_names)

    def test_partial_prefix_match(self):
        # User enters "Rec" but full prefix is "Record"
        ds = self.folder.assemble_dataseries("Rec")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_case_insensitive_prefix(self):
        ds = self.folder.assemble_dataseries("record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_no_match_returns_none(self):
        ds = self.folder.assemble_dataseries("NoMatch")
        self.assertIsNone(ds)

    def test_empty_prefix_returns_none(self):
        ds = self.folder.assemble_dataseries("")
        self.assertIsNone(ds)

    def test_none_prefix_returns_none(self):
        ds = self.folder.assemble_dataseries(None)
        self.assertIsNone(ds)

    def test_select_parameter(self):
        self.folder.assemble_dataseries("Record", select=True)
        self.assertEqual(self.folder.dataseries.selected_name, "Record")

    def test_adds_to_dataseries_container(self):
        self.folder.assemble_dataseries("Record")
        self.assertIn("Record", self.folder.dataseries)

    def test_returns_existing_dataseries_on_second_call(self):
        ds1 = self.folder.assemble_dataseries("Record")
        ds2 = self.folder.assemble_dataseries("Record")
        self.assertIsNotNone(ds1)
        self.assertIs(ds1, ds2)  # Same object, not duplicated

    def test_get_data_works_after_make(self):
        ds = self.folder.assemble_dataseries("Record")
        ds.channels.selected_name = "A"
        ds.epochs.selected_name = "E1"
        data = ds.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.name, "RecordA1")


class TestNMFolderAssembleDataSeriesMultiplePrefixes(NMFolderTestBase):
    """Tests for assemble_dataseries with multiple prefixes in folder."""

    def setUp(self):
        super().setUp()
        # Create data with two different prefixes
        for ch in ["A", "B"]:
            for ep in range(2):
                self.folder.data.new(f"Record{ch}{ep}")
                self.folder.data.new(f"avg{ch}{ep}")

    def test_creates_correct_dataseries(self):
        ds_record = self.folder.assemble_dataseries("Record")
        ds_avg = self.folder.assemble_dataseries("avg")

        self.assertEqual(ds_record.name, "Record")
        self.assertEqual(ds_avg.name, "avg")

    def test_dataseries_have_correct_data(self):
        ds_record = self.folder.assemble_dataseries("Record")
        ds_avg = self.folder.assemble_dataseries("avg")

        # Check Record dataseries
        chan_a = ds_record.channels.get("A")
        data_names = [d.name for d in chan_a.data]
        self.assertIn("RecordA0", data_names)
        self.assertNotIn("avgA0", data_names)

        # Check avg dataseries
        chan_a = ds_avg.channels.get("A")
        data_names = [d.name for d in chan_a.data]
        self.assertIn("avgA0", data_names)
        self.assertNotIn("RecordA0", data_names)

    def test_record_and_avg_record_prefixes(self):
        """Avg_Record prefix must not cross-contaminate Record dataseries.

        NMMainOpAverage produces output named Avg_{prefix}, e.g. Avg_RecordA0.
        When the folder contains both RecordA0 and Avg_RecordA0, assembling
        "Record" should include only Record* data and assembling "Avg_Record"
        should include only Avg_Record* data.

        setUp already creates RecordA0/A1/B0/B1; we add Avg_Record* here.
        """
        # setUp created RecordA0, RecordA1, RecordB0, RecordB1
        # Add the averaged output that NMMainOpAverage would produce
        for ch in ["A", "B"]:
            for ep in range(2):
                self.folder.data.new("Avg_Record%s%d" % (ch, ep))

        ds_rec = self.folder.assemble_dataseries("Record")
        ds_avg = self.folder.assemble_dataseries("Avg_Record")

        self.assertEqual(ds_rec.name, "Record")
        self.assertEqual(ds_avg.name, "Avg_Record")

        # Record dataseries contains only Record* data
        rec_chan_a = ds_rec.channels.get("A")
        rec_names = [d.name for d in rec_chan_a.data]
        self.assertIn("RecordA0", rec_names)
        self.assertNotIn("Avg_RecordA0", rec_names)
        self.assertEqual(len(rec_chan_a.data), 2)  # A0, A1

        # Avg_Record dataseries contains only Avg_Record* data
        avg_chan_a = ds_avg.channels.get("A")
        avg_names = [d.name for d in avg_chan_a.data]
        self.assertIn("Avg_RecordA0", avg_names)
        self.assertNotIn("RecordA0", avg_names)
        self.assertEqual(len(avg_chan_a.data), 2)  # Avg_RecordA0, Avg_RecordA1


# =============================================================================
# Tests for new_dataseries method
# =============================================================================

class TestNMFolderNewDataseries(NMFolderTestBase):
    """Tests for NMFolder.new_dataseries()."""

    def test_returns_dataseries(self):
        ds = self.folder.new_dataseries("Record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_default_single_channel_single_epoch(self):
        ds = self.folder.new_dataseries("Record")
        self.assertEqual(len(ds.channels), 1)
        self.assertEqual(len(ds.epochs), 1)

    def test_n_channels_and_n_epochs(self):
        ds = self.folder.new_dataseries("Record", n_channels=3, n_epochs=5)
        self.assertEqual(len(ds.channels), 3)
        self.assertEqual(len(ds.epochs), 5)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)
        self.assertIn("C", ds.channels)

    def test_data_objects_created_in_folder(self):
        self.folder.new_dataseries("Record", n_channels=2, n_epochs=3)
        for ch in ["A", "B"]:
            for ep in range(3):
                self.assertIn("Record%s%d" % (ch, ep), self.folder.data)

    def test_fill_default_zeros(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=50)
        data = self.folder.data.get("RecordA0")
        self.assertTrue(np.all(data.nparray == 0.0))
        self.assertEqual(len(data.nparray), 50)

    def test_fill_scalar_zero(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=50, fill=0.0)
        data = self.folder.data.get("RecordA0")
        self.assertTrue(np.all(data.nparray == 0.0))

    def test_fill_scalar_nan(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=50, fill=np.nan)
        data = self.folder.data.get("RecordA0")
        self.assertTrue(np.all(np.isnan(data.nparray)))

    def test_fill_scalar_sentinel(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=10, fill=-9999.0)
        data = self.folder.data.get("RecordA0")
        self.assertTrue(np.all(data.nparray == -9999.0))

    def test_fill_callable_numpy_ones(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=50, fill=np.ones)
        data = self.folder.data.get("RecordA0")
        self.assertTrue(np.all(data.nparray == 1.0))

    def test_fill_callable_numpy_random(self):
        import numpy as np
        ds = self.folder.new_dataseries("Record", n_points=50, fill=np.random.random)
        data = self.folder.data.get("RecordA0")
        self.assertEqual(len(data.nparray), 50)
        self.assertFalse(np.all(data.nparray == 0.0))

    def test_fill_callable_lambda_noise(self):
        import numpy as np
        noise_std = 2.0
        ds = self.folder.new_dataseries(
            "Record", n_points=500,
            fill=lambda n: np.random.normal(0, noise_std, n),
        )
        data = self.folder.data.get("RecordA0")
        self.assertEqual(len(data.nparray), 500)
        # SE of sample std ≈ sigma/sqrt(2*n) ≈ 0.063 for n=500;
        # delta=0.3 covers > 99.9 % of random draws without a fixed seed.
        self.assertAlmostEqual(np.std(data.nparray), noise_std, delta=0.3)

    def test_xscale(self):
        self.folder.new_dataseries(
            "Record", n_points=100, dx=0.1, x_start=5.0,
            x_label="Time", x_units="ms",
        )
        data = self.folder.data.get("RecordA0")
        self.assertAlmostEqual(data.xscale.delta, 0.1)
        self.assertAlmostEqual(data.xscale.start, 5.0)
        self.assertEqual(data.xscale.label, "Time")
        self.assertEqual(data.xscale.units, "ms")

    def test_yscale(self):
        self.folder.new_dataseries("Record", y_label="Voltage", y_units="mV")
        data = self.folder.data.get("RecordA0")
        self.assertEqual(data.yscale.label, "Voltage")
        self.assertEqual(data.yscale.units, "mV")

    def test_select_parameter(self):
        self.folder.new_dataseries("Record", select=True)
        self.assertEqual(self.folder.dataseries.selected_name, "Record")

    def test_invalid_prefix_type(self):
        with self.assertRaises(TypeError):
            self.folder.new_dataseries(123)

    def test_invalid_prefix_value(self):
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("")
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("bad name!")

    def test_invalid_n_channels_type(self):
        with self.assertRaises(TypeError):
            self.folder.new_dataseries("Record", n_channels=1.0)

    def test_invalid_n_channels_zero(self):
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("Record", n_channels=0)

    def test_invalid_n_epochs_zero(self):
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("Record", n_epochs=0)

    def test_invalid_n_points_negative(self):
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("Record", n_points=-1)

    def test_n_points_zero_creates_empty_arrays(self):
        ds = self.folder.new_dataseries("Record", n_points=0)
        self.assertIsNotNone(ds)
        data = self.folder.data.get("RecordA0")
        self.assertIsNotNone(data)
        self.assertEqual(len(data.nparray), 0)

    def test_invalid_fill_type(self):
        with self.assertRaises(TypeError):
            self.folder.new_dataseries("Record", fill="zeros")

    def test_append_epochs(self):
        self.folder.new_dataseries("Record", n_channels=2, n_epochs=3)
        ds = self.folder.new_dataseries("Record", n_channels=2, n_epochs=2, ep_start=3)
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds.epochs), 5)
        self.assertEqual(len(ds.channels), 2)
        chan_a = ds.channels.get("A")
        self.assertEqual(len(chan_a.data), 5)

    def test_append_channel(self):
        self.folder.new_dataseries("Record", n_channels=1, n_epochs=3)
        ds = self.folder.new_dataseries("Record", n_channels=1, n_epochs=3, ch_start=1)
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds.channels), 2)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)

    def test_append_does_not_duplicate_existing_data_links(self):
        self.folder.new_dataseries("Record", n_channels=2, n_epochs=3)
        self.folder.new_dataseries("Record", n_channels=2, n_epochs=2, ep_start=3)
        ds = self.folder.dataseries.get("Record")
        chan_a = ds.channels.get("A")
        # RecordA0..RecordA4 — each should appear exactly once
        names = [d.name for d in chan_a.data]
        self.assertEqual(len(names), len(set(names)))

    def test_raises_if_data_name_collision(self):
        self.folder.data.new("RecordA0")
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("Record")

    def test_no_partial_state_on_data_collision(self):
        # RecordA1 exists but RecordA0 does not — collision on second iteration
        self.folder.data.new("RecordA1")
        with self.assertRaises(ValueError):
            self.folder.new_dataseries("Record", n_epochs=3)
        # RecordA0 should NOT have been created
        self.assertNotIn("RecordA0", self.folder.data)


class TestNMFolderToolResults(NMFolderTestBase):
    """Tests for NMFolder.toolresults_save() and toolresults_clear()."""

    def test_toolresults_initially_empty(self):
        self.assertEqual(self.folder.toolresults, {})

    def test_save_returns_index(self):
        idx = self.folder.toolresults_save("stats", {"s": 1.0})
        self.assertEqual(idx, 0)

    def test_save_second_entry_returns_incremented_index(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        idx = self.folder.toolresults_save("stats", {"s": 2.0})
        self.assertEqual(idx, 1)

    def test_save_stores_results_under_tool_key(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.assertIn("stats", self.folder.toolresults)

    def test_save_entry_contains_date_and_results(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        entry = self.folder.toolresults["stats"][0]
        self.assertIn("date", entry)
        self.assertIn("results", entry)
        self.assertNotIn("tool", entry)

    def test_save_entry_results_correct(self):
        self.folder.toolresults_save("stats", {"s": 42.0})
        entry = self.folder.toolresults["stats"][0]
        self.assertEqual(entry["results"], {"s": 42.0})

    def test_save_multiple_tools(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_save("main", {"v": 2.0})
        self.assertIn("stats", self.folder.toolresults)
        self.assertIn("main", self.folder.toolresults)

    def test_save_multiple_tools_independent(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_save("main", {"v": 2.0})
        self.assertEqual(len(self.folder.toolresults["stats"]), 1)
        self.assertEqual(len(self.folder.toolresults["main"]), 1)

    def test_save_rejects_non_string_tool(self):
        with self.assertRaises(TypeError):
            self.folder.toolresults_save(123, {"s": 1.0})

    def test_clear_all(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_save("main", {"v": 2.0})
        self.folder.toolresults_clear()
        self.assertEqual(self.folder.toolresults, {})

    def test_clear_by_tool(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_save("main", {"v": 2.0})
        self.folder.toolresults_clear("stats")
        self.assertNotIn("stats", self.folder.toolresults)
        self.assertIn("main", self.folder.toolresults)

    def test_clear_by_index(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_save("stats", {"s": 2.0})
        self.folder.toolresults_save("stats", {"s": 3.0})
        self.folder.toolresults_clear("stats", 1)
        entries = self.folder.toolresults["stats"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["results"], {"s": 1.0})
        self.assertEqual(entries[1]["results"], {"s": 3.0})

    def test_clear_last_entry_removes_tool_key(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        self.folder.toolresults_clear("stats", 0)
        self.assertNotIn("stats", self.folder.toolresults)

    def test_clear_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            self.folder.toolresults_clear("stats")

    def test_clear_index_out_of_range_raises(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        with self.assertRaises(IndexError):
            self.folder.toolresults_clear("stats", 5)

    def test_clear_non_string_tool_raises(self):
        with self.assertRaises(TypeError):
            self.folder.toolresults_clear(123)

    def test_clear_non_int_index_raises(self):
        self.folder.toolresults_save("stats", {"s": 1.0})
        with self.assertRaises(TypeError):
            self.folder.toolresults_clear("stats", "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
