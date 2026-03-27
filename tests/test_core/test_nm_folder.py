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
# Tests for detect_data_prefixes method
# =============================================================================

class TestNMFolderDetectDataPrefixes(NMFolderTestBase):
    """Tests for NMFolder.detect_data_prefixes() method."""

    def test_empty_folder(self):
        result = self.folder.detect_data_prefixes()
        self.assertEqual(result, [])

    def test_single_prefix(self):
        for name in ["RecordA0", "RecordA1", "RecordB0", "RecordB1"]:
            self.folder.data.new(name)
        result = self.folder.detect_data_prefixes()
        self.assertEqual(result, ["Record"])

    def test_multiple_prefixes(self):
        for name in ["RecordA0", "RecordB0", "avgA0", "avgB0"]:
            self.folder.data.new(name)
        result = self.folder.detect_data_prefixes()
        self.assertEqual(result, ["Record", "avg"])

    def test_ignores_non_matching_names(self):
        self.folder.data.new("RecordA0")
        self.folder.data.new("SomeOtherData")  # no channel/epoch pattern
        self.folder.data.new("avgB1")
        result = self.folder.detect_data_prefixes()
        self.assertEqual(result, ["Record", "avg"])


# =============================================================================
# Tests for build_dataseries method
# =============================================================================

class TestNMFolderBuildDataseries(NMFolderTestBase):
    """Tests for NMFolder.build_dataseries() method."""

    def _make_data(self, names):
        """Helper: create NMData objects and return a dict by name."""
        result = {}
        for name in names:
            result[name] = self.folder.data.new(name, quiet=True)
        return result

    def test_empty_matches_returns_none(self):
        ds = self.folder.build_dataseries("Record", matches={})
        self.assertIsNone(ds)

    def test_creates_dataseries(self):
        d = self._make_data(["RecordA0"])
        ds = self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]})
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_dataseries_added_to_container(self):
        d = self._make_data(["RecordA0"])
        self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]})
        self.assertIn("Record", self.folder.dataseries)

    def test_creates_single_channel_and_epoch(self):
        d = self._make_data(["RecordA0"])
        ds = self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]})
        self.assertEqual(len(ds.channels), 1)
        self.assertIn("A", ds.channels)
        self.assertEqual(len(ds.epochs), 1)
        self.assertIn("E0", ds.epochs)

    def test_creates_multiple_channels(self):
        d = self._make_data(["RecordA0", "RecordB0", "RecordC0"])
        matches = {
            ("A", 0): d["RecordA0"],
            ("B", 0): d["RecordB0"],
            ("C", 0): d["RecordC0"],
        }
        ds = self.folder.build_dataseries("Record", matches)
        self.assertEqual(len(ds.channels), 3)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)
        self.assertIn("C", ds.channels)

    def test_creates_multiple_epochs(self):
        d = self._make_data(["RecordA0", "RecordA1", "RecordA2"])
        matches = {
            ("A", 0): d["RecordA0"],
            ("A", 1): d["RecordA1"],
            ("A", 2): d["RecordA2"],
        }
        ds = self.folder.build_dataseries("Record", matches)
        self.assertEqual(len(ds.epochs), 3)
        self.assertIn("E0", ds.epochs)
        self.assertIn("E1", ds.epochs)
        self.assertIn("E2", ds.epochs)

    def test_links_data_to_channel(self):
        d = self._make_data(["RecordA0", "RecordA1", "RecordB0", "RecordB1"])
        matches = {
            ("A", 0): d["RecordA0"],
            ("A", 1): d["RecordA1"],
            ("B", 0): d["RecordB0"],
            ("B", 1): d["RecordB1"],
        }
        ds = self.folder.build_dataseries("Record", matches)
        chan_a = ds.channels.get("A")
        chan_b = ds.channels.get("B")
        a_names = [x.name for x in chan_a.data]
        b_names = [x.name for x in chan_b.data]
        self.assertIn("RecordA0", a_names)
        self.assertIn("RecordA1", a_names)
        self.assertNotIn("RecordB0", a_names)
        self.assertIn("RecordB0", b_names)
        self.assertIn("RecordB1", b_names)
        self.assertNotIn("RecordA0", b_names)

    def test_links_data_to_epoch(self):
        d = self._make_data(["RecordA0", "RecordB0"])
        matches = {
            ("A", 0): d["RecordA0"],
            ("B", 0): d["RecordB0"],
        }
        ds = self.folder.build_dataseries("Record", matches)
        epoch_0 = ds.epochs.get("E0")
        ep_names = [x.name for x in epoch_0.data]
        self.assertIn("RecordA0", ep_names)
        self.assertIn("RecordB0", ep_names)

    def test_select_parameter(self):
        d = self._make_data(["RecordA0"])
        self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]}, select=True)
        self.assertEqual(self.folder.dataseries.selected_name, "Record")

    def test_select_false_does_not_change_existing_selection(self):
        # Pre-select "Other" then build "Record" with select=False — selection stays on "Other"
        d_other = self._make_data(["OtherA0"])
        self.folder.build_dataseries("Other", {("A", 0): d_other["OtherA0"]}, select=True)
        d = self._make_data(["RecordA0"])
        self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]}, select=False)
        self.assertEqual(self.folder.dataseries.selected_name, "Other")

    def test_idempotent_second_call_same_matches(self):
        d = self._make_data(["RecordA0", "RecordA1"])
        matches = {("A", 0): d["RecordA0"], ("A", 1): d["RecordA1"]}
        ds1 = self.folder.build_dataseries("Record", matches)
        ds2 = self.folder.build_dataseries("Record", matches)
        self.assertIs(ds1, ds2)
        self.assertEqual(len(ds1.channels), 1)
        self.assertEqual(len(ds1.epochs), 2)
        # Data should not be duplicated in channel
        chan_a = ds1.channels.get("A")
        self.assertEqual(len(chan_a.data), 2)

    def test_extends_existing_dataseries_with_new_epoch(self):
        d = self._make_data(["RecordA0", "RecordA1"])
        self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]})
        ds = self.folder.build_dataseries("Record", {("A", 1): d["RecordA1"]})
        self.assertEqual(len(ds.epochs), 2)
        chan_a = ds.channels.get("A")
        names = [x.name for x in chan_a.data]
        self.assertIn("RecordA0", names)
        self.assertIn("RecordA1", names)

    def test_extends_existing_dataseries_with_new_channel(self):
        d = self._make_data(["RecordA0", "RecordB0"])
        self.folder.build_dataseries("Record", {("A", 0): d["RecordA0"]})
        ds = self.folder.build_dataseries("Record", {("B", 0): d["RecordB0"]})
        self.assertEqual(len(ds.channels), 2)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)

    def test_non_zero_epoch_keys_create_correct_count(self):
        # ep_num keys in matches are internal identifiers; epochs are named
        # sequentially (E0, E1, ...) by the epoch container regardless of key value.
        d = self._make_data(["RecordA5", "RecordA6"])
        matches = {("A", 5): d["RecordA5"], ("A", 6): d["RecordA6"]}
        ds = self.folder.build_dataseries("Record", matches)
        self.assertEqual(len(ds.epochs), 2)
        self.assertIn("E0", ds.epochs)
        self.assertIn("E1", ds.epochs)

    def test_get_data_works_after_build(self):
        d = self._make_data(["RecordA0", "RecordA1", "RecordB0", "RecordB1"])
        matches = {
            ("A", 0): d["RecordA0"],
            ("A", 1): d["RecordA1"],
            ("B", 0): d["RecordB0"],
            ("B", 1): d["RecordB1"],
        }
        ds = self.folder.build_dataseries("Record", matches)
        ds.channels.selected_name = "B"
        ds.epochs.selected_name = "E1"
        data = ds.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.name, "RecordB1")


# =============================================================================
# Tests for scan_dataseries method
# =============================================================================

class TestNMFolderScanDataseries(NMFolderTestBase):
    """Tests for NMFolder.scan_dataseries() — preview without side effects."""

    def setUp(self):
        super().setUp()
        for ch in ["A", "B"]:
            for ep in range(3):
                self.folder.data.new(f"Record{ch}{ep}")

    def test_returns_correct_prefix(self):
        info = self.folder.scan_dataseries("Record")
        self.assertIsNotNone(info)
        self.assertEqual(info.prefix, "Record")

    def test_returns_correct_channels(self):
        info = self.folder.scan_dataseries("Record")
        self.assertEqual(info.channels, ["A", "B"])

    def test_returns_correct_epochs(self):
        info = self.folder.scan_dataseries("Record")
        self.assertEqual(info.epochs, [0, 1, 2])

    def test_returns_correct_n_data(self):
        info = self.folder.scan_dataseries("Record")
        self.assertEqual(info.n_data, 6)

    def test_no_side_effects(self):
        self.folder.scan_dataseries("Record")
        self.assertEqual(len(self.folder.dataseries), 0)

    def test_no_match_returns_none(self):
        self.assertIsNone(self.folder.scan_dataseries("NoMatch"))

    def test_empty_prefix_returns_none(self):
        self.assertIsNone(self.folder.scan_dataseries(""))

    def test_none_prefix_returns_none(self):
        self.assertIsNone(self.folder.scan_dataseries(None))

    def test_wildcard_pattern(self):
        info = self.folder.scan_dataseries("Rec*")
        self.assertIsNotNone(info)
        self.assertEqual(info.prefix, "Record")

    def test_partial_prefix(self):
        info = self.folder.scan_dataseries("Rec")
        self.assertIsNotNone(info)
        self.assertEqual(info.prefix, "Record")


# =============================================================================
# Tests for sync_dataseries method
# =============================================================================

class TestNMFolderSyncDataSeries(NMFolderTestBase):
    """Tests for NMFolder.sync_dataseries() method."""

    def setUp(self):
        super().setUp()
        # Create data with standard naming pattern
        for ch in ["A", "B"]:
            for ep in range(3):
                name = f"Record{ch}{ep}"
                self.folder.data.new(name)

    def test_creates_dataseries(self):
        ds = self.folder.sync_dataseries("Record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_creates_channels(self):
        ds = self.folder.sync_dataseries("Record")
        self.assertEqual(len(ds.channels), 2)
        self.assertIn("A", ds.channels)
        self.assertIn("B", ds.channels)

    def test_creates_epochs(self):
        ds = self.folder.sync_dataseries("Record")
        self.assertEqual(len(ds.epochs), 3)
        self.assertIn("E0", ds.epochs)
        self.assertIn("E1", ds.epochs)
        self.assertIn("E2", ds.epochs)

    def test_links_data_to_channels(self):
        ds = self.folder.sync_dataseries("Record")
        chan_a = ds.channels.get("A")
        self.assertEqual(len(chan_a.data), 3)
        data_names = [d.name for d in chan_a.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordA1", data_names)
        self.assertIn("RecordA2", data_names)

    def test_links_data_to_epochs(self):
        ds = self.folder.sync_dataseries("Record")
        epoch_0 = ds.epochs.get("E0")
        self.assertEqual(len(epoch_0.data), 2)
        data_names = [d.name for d in epoch_0.data]
        self.assertIn("RecordA0", data_names)
        self.assertIn("RecordB0", data_names)

    def test_partial_prefix_match(self):
        # User enters "Rec" but full prefix is "Record"
        ds = self.folder.sync_dataseries("Rec")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_case_insensitive_prefix(self):
        ds = self.folder.sync_dataseries("record")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_no_match_returns_none(self):
        ds = self.folder.sync_dataseries("NoMatch")
        self.assertIsNone(ds)

    def test_empty_prefix_returns_none(self):
        ds = self.folder.sync_dataseries("")
        self.assertIsNone(ds)

    def test_none_prefix_returns_none(self):
        ds = self.folder.sync_dataseries(None)
        self.assertIsNone(ds)

    def test_select_parameter(self):
        self.folder.sync_dataseries("Record", select=True)
        self.assertEqual(self.folder.dataseries.selected_name, "Record")

    def test_adds_to_dataseries_container(self):
        self.folder.sync_dataseries("Record")
        self.assertIn("Record", self.folder.dataseries)

    def test_returns_existing_dataseries_on_second_call(self):
        ds1 = self.folder.sync_dataseries("Record")
        ds2 = self.folder.sync_dataseries("Record")
        self.assertIsNotNone(ds1)
        self.assertIs(ds1, ds2)  # Same object, not duplicated

    def test_get_data_works_after_make(self):
        ds = self.folder.sync_dataseries("Record")
        ds.channels.selected_name = "A"
        ds.epochs.selected_name = "E1"
        data = ds.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.name, "RecordA1")


class TestNMFolderSyncDataSeriesMultiplePrefixes(NMFolderTestBase):
    """Tests for sync_dataseries with multiple prefixes in folder."""

    def setUp(self):
        super().setUp()
        # Create data with two different prefixes
        for ch in ["A", "B"]:
            for ep in range(2):
                self.folder.data.new(f"Record{ch}{ep}")
                self.folder.data.new(f"avg{ch}{ep}")

    def test_creates_correct_dataseries(self):
        ds_record = self.folder.sync_dataseries("Record")
        ds_avg = self.folder.sync_dataseries("avg")

        self.assertEqual(ds_record.name, "Record")
        self.assertEqual(ds_avg.name, "avg")

    def test_dataseries_have_correct_data(self):
        ds_record = self.folder.sync_dataseries("Record")
        ds_avg = self.folder.sync_dataseries("avg")

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

        ds_rec = self.folder.sync_dataseries("Record")
        ds_avg = self.folder.sync_dataseries("Avg_Record")

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
# Tests for sync_dataseries wildcard support
# =============================================================================

class TestNMFolderSyncDataSeriesWildcard(NMFolderTestBase):
    """Tests for wildcard pattern support in NMFolder.sync_dataseries()."""

    def setUp(self):
        super().setUp()
        # Two prefixes starting with "Rec": "Recent" sorts before "Record"
        for ch in ["A", "B"]:
            self.folder.data.new(f"Record{ch}0")
            self.folder.data.new(f"Recent{ch}0")
        self.folder.data.new("avgA0")

    def test_star_matches_full_prefix(self):
        ds = self.folder.sync_dataseries("Record*")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_star_matches_partial_prefix_alphabetically_first(self):
        # "Rec*" matches both "Recent" and "Record"; "Recent" < "Record" alphabetically
        ds = self.folder.sync_dataseries("Rec*")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Recent")

    def test_question_mark_wildcard(self):
        # "Rec?rd" matches "Record" (? = exactly one character)
        ds = self.folder.sync_dataseries("Rec?rd")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Record")

    def test_star_only_returns_alphabetically_first(self):
        # "*" matches all prefixes; "Recent" < "Record" < "avg" alphabetically
        ds = self.folder.sync_dataseries("*")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Recent")

    def test_wildcard_no_match_returns_none(self):
        ds = self.folder.sync_dataseries("Xyz*")
        self.assertIsNone(ds)

    def test_wildcard_case_insensitive(self):
        ds = self.folder.sync_dataseries("rec*")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Recent")

    def test_non_wildcard_partial_still_alphabetically_first(self):
        # No wildcard: falls back to startswith — "Rec" matches "Recent" and "Record"
        ds = self.folder.sync_dataseries("Rec")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.name, "Recent")

    def test_wildcard_links_correct_data(self):
        ds = self.folder.sync_dataseries("Record*")
        chan_a = ds.channels.get("A")
        names = [d.name for d in chan_a.data]
        self.assertIn("RecordA0", names)
        self.assertNotIn("RecentA0", names)
        self.assertNotIn("avgA0", names)


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


class TestNMFolderCopyDataseries(NMFolderTestBase):
    """Tests for NMFolder.copy_dataseries()."""

    def setUp(self):
        super().setUp()
        self.folder.new_dataseries(
            "Record", n_channels=2, n_epochs=3, n_points=10,
            x_start=0.0, dx=0.1, x_units="ms", y_units="mV", quiet=True,
        )
        self.ds = self.folder.dataseries.get("Record")

    # --- auto prefix ---

    def test_auto_prefix_same_folder(self):
        ds2 = self.folder.copy_dataseries("Record", quiet=True)
        self.assertIsNotNone(ds2)
        self.assertIsNotNone(self.folder.dataseries.get("C_Record"))

    def test_explicit_new_prefix(self):
        self.folder.copy_dataseries("Record", "Avg_Record", quiet=True)
        self.assertIsNotNone(self.folder.dataseries.get("Avg_Record"))

    def test_copy_to_other_folder_reuses_prefix(self):
        other = NMFolder(parent=self.nm, name="Other")
        self.folder.copy_dataseries("Record", folder=other, quiet=True)
        self.assertIsNotNone(other.dataseries.get("Record"))

    def test_copy_to_other_folder_explicit_prefix(self):
        other = NMFolder(parent=self.nm, name="Other")
        self.folder.copy_dataseries("Record", "Stim", folder=other, quiet=True)
        self.assertIsNotNone(other.dataseries.get("Stim"))

    # --- data integrity ---

    def test_nparray_is_deep_copied(self):
        import numpy as np
        for name in list(self.folder.data):
            self.folder.data.get(name).nparray = np.ones(10) * 42.0
        self.folder.copy_dataseries("Record", quiet=True)
        # Mutate source; copy must be unaffected
        for name in list(self.folder.data):
            if not name.startswith("C_"):
                self.folder.data.get(name).nparray[:] = 0.0
        copy_ds = self.folder.dataseries.get("C_Record")
        ch_a = copy_ds.channels.get("A")
        self.assertTrue(all(d.nparray[0] == 42.0 for d in ch_a.data))

    def test_xscale_preserved(self):
        copy_ds = self.folder.copy_dataseries("Record", quiet=True)
        ch_a = copy_ds.channels.get("A")
        # xscale lives on individual NMData objects (channel xscale is separate)
        d = ch_a.data[0]
        self.assertAlmostEqual(d.xscale.start, 0.0)
        self.assertAlmostEqual(d.xscale.delta, 0.1)
        self.assertEqual(d.xscale.units, "ms")

    def test_yscale_preserved(self):
        copy_ds = self.folder.copy_dataseries("Record", quiet=True)
        ch_a = copy_ds.channels.get("A")
        d = ch_a.data[0]
        self.assertEqual(d.yscale.units, "mV")

    def test_original_dataseries_unchanged(self):
        self.folder.copy_dataseries("Record", quiet=True)
        self.assertIsNotNone(self.folder.dataseries.get("Record"))
        self.assertEqual(len(list(self.ds.channels)), 2)
        self.assertEqual(len(list(self.ds.epochs)), 3)

    # --- channel/epoch subsetting ---

    def test_copy_subset_channels(self):
        ds2 = self.folder.copy_dataseries("Record", channel="A", quiet=True)
        self.assertIsNotNone(ds2.channels.get("A"))
        self.assertIsNone(ds2.channels.get("B"))

    def test_copy_subset_channels_by_int(self):
        ds2 = self.folder.copy_dataseries("Record", channel=0, quiet=True)
        self.assertIsNotNone(ds2.channels.get("A"))
        self.assertIsNone(ds2.channels.get("B"))

    def test_copy_subset_epochs(self):
        ds2 = self.folder.copy_dataseries("Record", epoch=range(0, 2), quiet=True)
        self.assertIsNotNone(ds2.epochs.get("E0"))
        self.assertIsNotNone(ds2.epochs.get("E1"))
        self.assertIsNone(ds2.epochs.get("E2"))

    def test_copy_subset_channel_and_epoch(self):
        ds2 = self.folder.copy_dataseries(
            "Record", channel="A", epoch=[0, 2], quiet=True
        )
        self.assertEqual(len(list(ds2.channels)), 1)
        self.assertEqual(len(list(ds2.epochs)), 2)

    # --- error cases ---

    def test_raises_for_unknown_source(self):
        with self.assertRaises(ValueError):
            self.folder.copy_dataseries("NoSuch", quiet=True)

    def test_raises_if_new_prefix_already_exists(self):
        self.folder.copy_dataseries("Record", quiet=True)  # creates C_Record
        with self.assertRaises(ValueError):
            self.folder.copy_dataseries("Record", quiet=True)  # C_Record exists

    def test_raises_for_unknown_channel(self):
        with self.assertRaises(ValueError):
            self.folder.copy_dataseries("Record", channel="Z", quiet=True)

    def test_raises_for_unknown_epoch(self):
        with self.assertRaises(ValueError):
            self.folder.copy_dataseries("Record", epoch=99, quiet=True)


class TestNMFolderRemoveDataseries(NMFolderTestBase):
    """Tests for NMFolder.remove_dataseries()."""

    def setUp(self):
        super().setUp()
        self.folder.new_dataseries(
            "Record", n_channels=2, n_epochs=3, quiet=True
        )

    def test_removes_dataseries(self):
        self.folder.remove_dataseries("Record", quiet=True)
        self.assertIsNone(self.folder.dataseries.get("Record"))

    def test_data_not_deleted_by_default(self):
        self.folder.remove_dataseries("Record", quiet=True)
        self.assertIn("RecordA0", self.folder.data)

    def test_delete_data_true_removes_data(self):
        self.folder.remove_dataseries("Record", delete_data=True, quiet=True)
        self.assertNotIn("RecordA0", self.folder.data)
        self.assertNotIn("RecordB2", self.folder.data)

    def test_raises_for_unknown_prefix(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries("NoSuch", quiet=True)

    def test_folder_data_unaffected_after_removal_without_delete(self):
        # 2 channels × 3 epochs = 6 NMData objects
        self.folder.remove_dataseries("Record", quiet=True)
        self.assertEqual(len(self.folder.data), 6)


class TestNMFolderRemoveDataseriesChannel(NMFolderTestBase):
    """Tests for NMFolder.remove_dataseries_channel()."""

    def setUp(self):
        super().setUp()
        self.folder.new_dataseries(
            "Record", n_channels=3, n_epochs=2, quiet=True
        )
        self.ds = self.folder.dataseries.get("Record")

    def test_remove_dataseries_channel_by_char(self):
        self.folder.remove_dataseries_channel("Record", "B", quiet=True)
        self.assertIsNone(self.ds.channels.get("B"))

    def test_remove_dataseries_channel_by_int(self):
        self.folder.remove_dataseries_channel("Record", 0, quiet=True)
        self.assertIsNone(self.ds.channels.get("A"))

    def test_remove_dataseries_channel_list(self):
        self.folder.remove_dataseries_channel("Record", ["A", "C"], quiet=True)
        self.assertIsNone(self.ds.channels.get("A"))
        self.assertIsNone(self.ds.channels.get("C"))
        self.assertIsNotNone(self.ds.channels.get("B"))

    def test_remove_dataseries_channel_range(self):
        self.folder.remove_dataseries_channel("Record", range(2), quiet=True)
        self.assertIsNone(self.ds.channels.get("A"))
        self.assertIsNone(self.ds.channels.get("B"))
        self.assertIsNotNone(self.ds.channels.get("C"))

    def test_data_not_deleted_by_default(self):
        self.folder.remove_dataseries_channel("Record", "A", quiet=True)
        self.assertIn("RecordA0", self.folder.data)
        self.assertIn("RecordA1", self.folder.data)

    def test_delete_data_true_removes_data(self):
        self.folder.remove_dataseries_channel("Record", "A", delete_data=True, quiet=True)
        self.assertNotIn("RecordA0", self.folder.data)
        self.assertNotIn("RecordA1", self.folder.data)
        self.assertIn("RecordB0", self.folder.data)

    def test_removed_channel_data_no_longer_in_epochs(self):
        ch_a_data = list(self.ds.channels.get("A").data)
        self.folder.remove_dataseries_channel("Record", "A", quiet=True)
        for ep_name in self.ds.epochs:
            ep = self.ds.epochs.get(ep_name)
            for d in ch_a_data:
                self.assertNotIn(d, ep.data)

    def test_raises_for_unknown_channel(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries_channel("Record", "Z", quiet=True)

    def test_raises_for_unknown_prefix(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries_channel("NoSuch", "A", quiet=True)

    def test_raises_for_bool_channel(self):
        with self.assertRaises(TypeError):
            self.folder.remove_dataseries_channel("Record", True, quiet=True)


class TestNMFolderRemoveDataseriesEpoch(NMFolderTestBase):
    """Tests for NMFolder.remove_dataseries_epoch()."""

    def setUp(self):
        super().setUp()
        self.folder.new_dataseries(
            "Record", n_channels=2, n_epochs=4, quiet=True
        )
        self.ds = self.folder.dataseries.get("Record")

    def test_remove_dataseries_epoch_by_int(self):
        self.folder.remove_dataseries_epoch("Record", 0, quiet=True)
        self.assertIsNone(self.ds.epochs.get("E0"))

    def test_remove_dataseries_epoch_by_str(self):
        self.folder.remove_dataseries_epoch("Record", "E1", quiet=True)
        self.assertIsNone(self.ds.epochs.get("E1"))

    def test_remove_dataseries_epoch_list(self):
        self.folder.remove_dataseries_epoch("Record", [0, 2], quiet=True)
        self.assertIsNone(self.ds.epochs.get("E0"))
        self.assertIsNone(self.ds.epochs.get("E2"))
        self.assertIsNotNone(self.ds.epochs.get("E1"))
        self.assertIsNotNone(self.ds.epochs.get("E3"))

    def test_remove_dataseries_epoch_range(self):
        self.folder.remove_dataseries_epoch("Record", range(1, 3), quiet=True)
        self.assertIsNone(self.ds.epochs.get("E1"))
        self.assertIsNone(self.ds.epochs.get("E2"))
        self.assertIsNotNone(self.ds.epochs.get("E0"))
        self.assertIsNotNone(self.ds.epochs.get("E3"))

    def test_data_not_deleted_by_default(self):
        self.folder.remove_dataseries_epoch("Record", 0, quiet=True)
        self.assertIn("RecordA0", self.folder.data)

    def test_delete_data_true_removes_data(self):
        self.folder.remove_dataseries_epoch("Record", 0, delete_data=True, quiet=True)
        self.assertNotIn("RecordA0", self.folder.data)
        self.assertNotIn("RecordB0", self.folder.data)
        self.assertIn("RecordA1", self.folder.data)

    def test_removed_epoch_data_no_longer_in_channels(self):
        ep0 = self.ds.epochs.get("E0")
        ep0_data = list(ep0.data)
        self.folder.remove_dataseries_epoch("Record", 0, quiet=True)
        for ch_name in self.ds.channels:
            ch = self.ds.channels.get(ch_name)
            for d in ep0_data:
                self.assertNotIn(d, ch.data)

    def test_raises_for_unknown_epoch_int(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries_epoch("Record", 99, quiet=True)

    def test_raises_for_unknown_epoch_str(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries_epoch("Record", "E99", quiet=True)

    def test_raises_for_unknown_prefix(self):
        with self.assertRaises(ValueError):
            self.folder.remove_dataseries_epoch("NoSuch", 0, quiet=True)

    def test_raises_for_bool_epoch(self):
        with self.assertRaises(TypeError):
            self.folder.remove_dataseries_epoch("Record", True, quiet=True)

    def test_remove_range_200_epochs(self):
        # Simulate use-case: 200 epochs, remove the last 100
        self.folder.new_dataseries(
            "Big", n_channels=1, n_epochs=200, quiet=True
        )
        ds = self.folder.dataseries.get("Big")
        self.folder.remove_dataseries_epoch("Big", range(100, 200), quiet=True)
        self.assertEqual(len(list(ds.epochs)), 100)
        self.assertIsNotNone(ds.epochs.get("E99"))
        self.assertIsNone(ds.epochs.get("E100"))


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
