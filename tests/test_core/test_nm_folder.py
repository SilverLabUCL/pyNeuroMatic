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
from pyneuromatic.core.nm_project import NMProject
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
        p = NMProject()
        with self.assertRaises(TypeError):
            NMFolder(copy=p)

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

        result = self.folder.data.sets.get("all", get_keys=True)
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
        self.project = NMProject(parent=self.nm)
        self.container = NMFolderContainer(parent=self.project)

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
        # Parent should be container's parent (NMProject), not container itself
        folder = self.container.new(name="folder0")
        self.assertEqual(folder._parent, self.project)

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

class TestDetectPrefixes(NMFolderTestBase):
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
# Tests for make_dataseries method
# =============================================================================

class TestMakeDataSeries(NMFolderTestBase):
    """Tests for NMFolder.make_dataseries() method."""

    def setUp(self):
        super().setUp()
        # Create data with standard naming pattern
        for ch in ["A", "B"]:
            for ep in range(3):
                name = f"Record{ch}{ep}"
                self.folder.data.new(name)

    def test_creates_dataseries(self):
        ds = self.folder.make_dataseries("Record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_creates_channels(self):
        ds = self.folder.make_dataseries("Record")
        self.assertEqual(len(ds.channels), 2)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)

    def test_creates_epochs(self):
        ds = self.folder.make_dataseries("Record")
        self.assertEqual(len(ds.epochs), 3)
        self.assertIn("E0", ds.epochs)
        self.assertIn("E1", ds.epochs)
        self.assertIn("E2", ds.epochs)

    def test_links_data_to_channels(self):
        ds = self.folder.make_dataseries("Record")
        chan_a = ds.channels.get("A")
        self.assertEqual(len(chan_a.data), 3)
        data_names = [d.name for d in chan_a.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordA1", data_names)
        self.assertIn("RecordA2", data_names)

    def test_links_data_to_epochs(self):
        ds = self.folder.make_dataseries("Record")
        epoch_0 = ds.epochs.get("E0")
        self.assertEqual(len(epoch_0.data), 2)
        data_names = [d.name for d in epoch_0.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordB0", data_names)

    def test_partial_prefix_match(self):
        # User enters "Rec" but full prefix is "Record"
        ds = self.folder.make_dataseries("Rec")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_case_insensitive_prefix(self):
        ds = self.folder.make_dataseries("record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_no_match_returns_none(self):
        ds = self.folder.make_dataseries("NoMatch")
        self.assertIsNone(ds)

    def test_empty_prefix_returns_none(self):
        ds = self.folder.make_dataseries("")
        self.assertIsNone(ds)

    def test_none_prefix_returns_none(self):
        ds = self.folder.make_dataseries(None)
        self.assertIsNone(ds)

    def test_select_parameter(self):
        self.folder.make_dataseries("Record", select=True)
        self.assertEqual(self.folder.dataseries.selected_name, "Record")

    def test_adds_to_dataseries_container(self):
        self.folder.make_dataseries("Record")
        self.assertIn("Record", self.folder.dataseries)

    def test_does_not_duplicate_dataseries(self):
        ds1 = self.folder.make_dataseries("Record")
        ds2 = self.folder.make_dataseries("Record")
        self.assertIsNotNone(ds1)
        self.assertIsNone(ds2)  # Already exists

    def test_get_data_works_after_make(self):
        ds = self.folder.make_dataseries("Record")
        ds.channels.selected_name = "A"
        ds.epochs.selected_name = "E1"
        data = ds.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.name, "RecordA1")


class TestMakeDataSeriesMultiplePrefixes(NMFolderTestBase):
    """Tests for make_dataseries with multiple prefixes in folder."""

    def setUp(self):
        super().setUp()
        # Create data with two different prefixes
        for ch in ["A", "B"]:
            for ep in range(2):
                self.folder.data.new(f"Record{ch}{ep}")
                self.folder.data.new(f"avg{ch}{ep}")

    def test_creates_correct_dataseries(self):
        ds_record = self.folder.make_dataseries("Record")
        ds_avg = self.folder.make_dataseries("avg")

        self.assertEqual(ds_record.name, "Record")
        self.assertEqual(ds_avg.name, "avg")

    def test_dataseries_have_correct_data(self):
        ds_record = self.folder.make_dataseries("Record")
        ds_avg = self.folder.make_dataseries("avg")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
