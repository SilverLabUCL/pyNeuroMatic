#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMManager.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_project import NMProject

QUIET = True
NUMFOLDERS = 5
DATASERIES = ["data", "avg", "stim"]
NUMDATA = [8, 9, 10]
NUMCHANNELS = [2, 3, 4]
NUMEPOCHS = [5, 6, 7]
ISELECT = 0


class NMManagerTestBase(unittest.TestCase):
    """Base class with shared setUp for NMManager tests."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        ilast = ISELECT == -1
        self.select_values = {}
        self.select_keys = {}

        p = self.nm.project

        for ifolder in range(NUMFOLDERS):
            fselect = False
            if ilast or ifolder == ISELECT:
                f = p.folders.new(select=True)
                self.select_values["folder"] = f
                self.select_keys["folder"] = f.name
                fselect = True
            else:
                f = p.folders.new(select=False)
            jdata = 0
            for idataseries in range(len(DATASERIES)):
                prefix = DATASERIES[idataseries]
                # data
                for idata in range(NUMDATA[idataseries]):
                    n = prefix + str(idata)
                    if ilast or jdata == ISELECT:
                        d = f.data.new(n, select=True)
                        if fselect:
                            self.select_values["data"] = d
                            self.select_keys["data"] = d.name
                    else:
                        d = f.data.new(n, select=False)
                    jdata += 1
                # dataseries
                ds_select = False
                if ilast or idataseries == ISELECT:
                    ds = f.dataseries.new(prefix, select=True)
                    if fselect:
                        self.select_values["dataseries"] = ds
                        self.select_keys["dataseries"] = ds.name
                        ds_select = True
                else:
                    ds = f.dataseries.new(prefix, select=False)
                for ichannel in range(NUMCHANNELS[idataseries]):
                    if ilast or ichannel == ISELECT:
                        c = ds.channels.new(select=True)
                        if ds_select:
                            self.select_values["channel"] = c
                            self.select_keys["channel"] = c.name
                    else:
                        c = ds.channels.new(select=False)
                for iepoch in range(NUMEPOCHS[idataseries]):
                    if ilast or iepoch == ISELECT:
                        e = ds.epochs.new(select=True)
                        if ds_select:
                            self.select_values["epoch"] = e
                            self.select_keys["epoch"] = e.name
                    else:
                        e = ds.epochs.new(select=False)
                ds.channels.sets.add("set0", ["A", "B"])
                ds.epochs.sets.add("set0", ["E0", "E1"])
            self.data_set0 = ["data0", "avg0", "stim0"]
            f.data.sets.add("set0", self.data_set0)
            f.dataseries.sets.add("set0", ["data", "avg"])
        p.folders.sets.add("set0", ["folder0", "folder1"])


class TestNMManagerInit(NMManagerTestBase):
    """Tests for NMManager initialization."""

    def test_creates_project(self):
        self.assertIsInstance(self.nm.project, NMProject)

    def test_project_named_root(self):
        self.assertEqual(self.nm.project.name, "root")

    def test_project_not_none(self):
        self.assertIsNotNone(self.nm.project)


class TestNMManagerSelect(NMManagerTestBase):
    """Tests for NMManager selection methods."""

    def test_select_values_returns_current_selection(self):
        self.assertEqual(self.nm.select_values, self.select_values)

    def test_select_keys_returns_names(self):
        self.assertEqual(self.nm.select_keys, self.select_keys)

    def test_select_values_is_readonly(self):
        with self.assertRaises(AttributeError):
            self.nm.select_values = {}

    def test_select_keys_rejects_non_dict(self):
        bad_types = (None, 3, 3.14, True, [], (), set(), "string")
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.select_keys = b

    def test_select_keys_set_updates_hierarchy(self):
        s1 = {
            "folder": "folder1",
            "data": "data3",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E3",
        }
        self.nm.select_keys = s1
        self.assertEqual(self.nm.select_keys, s1)

    def test_select_keys_set_partial_update(self):
        s1 = {
            "folder": "folder1",
            "data": "data3",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E3",
        }
        self.nm.select_keys = s1

        # Change just dataseries - channel/epoch revert to selected in new dataseries
        self.nm.select_keys = {"dataseries": "avg"}
        expected = {
            "folder": "folder1",
            "data": "data3",
            "dataseries": "avg",
            "channel": "A",  # First channel in avg dataseries
            "epoch": "E0",   # First epoch in avg dataseries
        }
        self.assertEqual(self.nm.select_keys, expected)

    def test_select_keys_rejects_invalid_project(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"project": "test"}

    def test_select_keys_rejects_invalid_folder(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"folder": "test"}

    def test_select_keys_rejects_invalid_data(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"data": "test"}

    def test_select_keys_rejects_invalid_dataseries(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"dataseries": "test"}

    def test_select_keys_rejects_invalid_channel(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"channel": "test"}

    def test_select_keys_rejects_invalid_epoch(self):
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"epoch": "test"}

    def test_select_keys_case_insensitive(self):
        # Test with uppercase keys
        s1 = {
            "FOLDER": "folder1",
            "DATA": "data3",
            "DATASERIES": "data",
            "CHANNEL": "A",
            "EPOCH": "E3",
        }
        self.nm.select_keys = s1
        expected = {
            "folder": "folder1",
            "data": "data3",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E3",
        }
        self.assertEqual(self.nm.select_keys, expected)

    def test_select_keys_mixed_case(self):
        # Test with mixed case keys
        s1 = {"Folder": "folder1", "DataSeries": "data", "Channel": "A", "Epoch": "E0"}
        self.nm.select_keys = s1
        self.assertEqual(self.nm.select_keys["folder"], "folder1")
        self.assertEqual(self.nm.select_keys["dataseries"], "data")


class TestNMManagerExecuteKeys(NMManagerTestBase):
    """Tests for execute_keys and execute_values methods."""

    def test_execute_keys_returns_selected_dataseries(self):
        s = self.nm.select_keys.copy()
        s.pop("data")
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])

    def test_execute_keys_returns_selected_data(self):
        s = self.nm.select_keys.copy()
        s.pop("dataseries")
        s.pop("channel")
        s.pop("epoch")
        elist = self.nm.execute_keys(dataseries_priority=False)
        self.assertEqual(elist, [s])

    def test_execute_reset_all_restores_selected(self):
        self.nm.execute_reset_all()
        s = self.nm.select_keys.copy()
        s.pop("data")
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])

    def test_execute_keys_with_folder_set(self):
        p = self.nm.project
        p.folders.execute_target = "set0"
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(len(elist), 2)  # set0 has folder0 and folder1

    def test_execute_keys_with_folder_all(self):
        p = self.nm.project
        p.folders.execute_target = "all"
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(len(elist), NUMFOLDERS)

    def test_execute_count_returns_target_count(self):
        self.nm.execute_reset_all()
        count = self.nm.execute_count(dataseries_priority=True)
        self.assertEqual(count, 1)

    def test_execute_count_with_all_epochs(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        self.nm.execute_keys_set(e0)
        count = self.nm.execute_count(dataseries_priority=True)
        self.assertEqual(count, NUMEPOCHS[0])


class TestNMManagerExecuteKeysSet(NMManagerTestBase):
    """Tests for execute_keys_set method."""

    def test_rejects_non_dict(self):
        bad_types = (None, 3, 3.14, True, [], (), set(), "string")
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set(b)

    def test_rejects_non_string_key(self):
        bad_types = (None, 3, 3.14, True, [], (), {}, set())
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set({b: ""})

    def test_rejects_non_string_value(self):
        bad_types = (None, 3, 3.14, True, [], (), {}, set())
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set({"folder": b})

    def test_rejects_unknown_key(self):
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set({"test": ""})

    def test_rejects_both_data_and_dataseries(self):
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set({"data": "", "dataseries": ""})

    def test_rejects_missing_folder(self):
        e0 = {"dataseries": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

    def test_rejects_missing_data_or_dataseries(self):
        e0 = {"folder": "folder1", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

    def test_rejects_data_with_channel(self):
        e0 = {"folder": "folder1", "data": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

    def test_rejects_missing_channel(self):
        e0 = {"folder": "folder1", "dataseries": "stim", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

    def test_rejects_missing_epoch(self):
        e0 = {"folder": "folder1", "dataseries": "stim", "channel": "A"}
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

    def test_rejects_invalid_folder_name(self):
        e0 = {"folder": "test", "dataseries": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)

    def test_rejects_invalid_dataseries_name(self):
        e0 = {"folder": "folder1", "dataseries": "test", "channel": "A", "epoch": "E0"}
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)

    def test_sets_specific_targets(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(elist, [e0])

    def test_selected_uses_current_selection(self):
        self.nm.execute_reset_all()
        e1 = {
            "folder": "selected",
            "dataseries": "selected",
            "channel": "selected",
            "epoch": "selected",
        }
        elist = self.nm.execute_keys_set(e1)
        select = self.nm.select_keys.copy()
        select.pop("data")
        self.assertEqual(elist, [select])

    def test_channel_all_expands_to_all_channels(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "all",
            "epoch": "E0",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), NUMCHANNELS[2])

    def test_epoch_all_expands_to_all_epochs(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), NUMEPOCHS[2])

    def test_data_set_expands_to_set_members(self):
        e0 = {"folder": "folder1", "data": "set0"}
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), len(self.data_set0))

    def test_folder_all_is_valid(self):
        e0 = {
            "folder": "all",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), NUMFOLDERS)

    def test_folder_set_is_valid(self):
        e0 = {
            "folder": "set0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), 2)  # set0 has folder0 and folder1

    def test_execute_keys_set_case_insensitive(self):
        # Test with uppercase keys
        e0 = {
            "FOLDER": "folder1",
            "DATASERIES": "stim",
            "CHANNEL": "A",
            "EPOCH": "E0",
        }
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), 1)
        self.assertEqual(elist[0]["folder"], "folder1")
        self.assertEqual(elist[0]["dataseries"], "stim")

    def test_execute_keys_set_mixed_case(self):
        # Test with mixed case keys
        e0 = {"Folder": "folder1", "Data": "data0"}
        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(len(elist), 1)
        self.assertEqual(elist[0]["folder"], "folder1")
        self.assertEqual(elist[0]["data"], "data0")


class TestNMManagerExecuteMaxTargets(NMManagerTestBase):
    """Tests for max_targets parameter in execute_keys_set."""

    def test_succeeds_within_limit(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.execute_keys_set(e0, max_targets=100)
        self.assertEqual(len(elist), NUMEPOCHS[0])

    def test_raises_when_exceeding_limit(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0, max_targets=1)

    def test_none_allows_unlimited(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.execute_keys_set(e0, max_targets=None)
        self.assertEqual(len(elist), NUMEPOCHS[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
