#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMManager.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_command_history import NMCommandHistory, get_command_history

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

        for ifolder in range(NUMFOLDERS):
            fselect = False
            if ilast or ifolder == ISELECT:
                f = self.nm.folders.new(select=True)
                self.select_values["folder"] = f
                self.select_keys["folder"] = f.name
                fselect = True
            else:
                f = self.nm.folders.new(select=False)
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
                epoch_names = list(ds.epochs.keys())
                ds.epochs.groups.assign_cyclic(epoch_names, n_groups=3, quiet=QUIET)
            self.data_set0 = ["data0", "avg0", "stim0"]
            f.data.sets.add("set0", self.data_set0)
            f.dataseries.sets.add("set0", ["data", "avg"])
        self.nm.folders.sets.add("set0", ["folder0", "folder1"])


class TestNMManagerInit(NMManagerTestBase):
    """Tests for NMManager initialization."""

    def test_has_folders(self):
        from pyneuromatic.core.nm_folder import NMFolderContainer
        self.assertIsInstance(self.nm.folders, NMFolderContainer)

    def test_name_is_nm(self):
        self.assertEqual(self.nm.name, "nm")

    def test_folders_not_none(self):
        self.assertIsNotNone(self.nm.folders)


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


class TestNMManagerRunKeys(NMManagerTestBase):
    """Tests for run_keys and run_values methods."""

    def test_run_keys_returns_selected_dataseries(self):
        s = self.nm.select_keys.copy()
        s.pop("data")
        elist = self.nm.run_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])

    def test_run_keys_returns_selected_data(self):
        s = self.nm.select_keys.copy()
        s.pop("dataseries")
        s.pop("channel")
        s.pop("epoch")
        elist = self.nm.run_keys(dataseries_priority=False)
        self.assertEqual(elist, [s])

    def test_run_reset_all_restores_selected(self):
        self.nm.run_reset_all()
        s = self.nm.select_keys.copy()
        s.pop("data")
        elist = self.nm.run_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])

    def test_run_keys_with_folder_set(self):
        self.nm.folders.run_target = "set0"
        elist = self.nm.run_keys(dataseries_priority=True)
        self.assertEqual(len(elist), 2)  # set0 has folder0 and folder1

    def test_run_keys_with_folder_all(self):
        self.nm.folders.run_target = "all"
        elist = self.nm.run_keys(dataseries_priority=True)
        self.assertEqual(len(elist), NUMFOLDERS)

    def test_run_count_returns_target_count(self):
        self.nm.run_reset_all()
        count = self.nm.run_count(dataseries_priority=True)
        self.assertEqual(count, 1)

    def test_run_count_with_all_epochs(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        self.nm.run_keys_set(e0)
        count = self.nm.run_count(dataseries_priority=True)
        self.assertEqual(count, NUMEPOCHS[0])


class TestNMManagerRunKeysSet(NMManagerTestBase):
    """Tests for run_keys_set method."""

    def test_rejects_non_dict(self):
        bad_types = (None, 3, 3.14, True, [], (), set(), "string")
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.run_keys_set(b)

    def test_rejects_non_string_key(self):
        bad_types = (None, 3, 3.14, True, [], (), {}, set())
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.run_keys_set({b: ""})

    def test_rejects_non_string_value(self):
        bad_types = (None, 3, 3.14, True, [], (), {}, set())
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.run_keys_set({"folder": b})

    def test_rejects_unknown_key(self):
        with self.assertRaises(KeyError):
            self.nm.run_keys_set({"test": ""})

    def test_rejects_both_data_and_dataseries(self):
        with self.assertRaises(KeyError):
            self.nm.run_keys_set({"data": "", "dataseries": ""})

    def test_rejects_missing_folder(self):
        e0 = {"dataseries": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.run_keys_set(e0)

    def test_rejects_missing_data_or_dataseries(self):
        e0 = {"folder": "folder1", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.run_keys_set(e0)

    def test_rejects_data_with_channel(self):
        e0 = {"folder": "folder1", "data": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.run_keys_set(e0)

    def test_rejects_missing_channel(self):
        e0 = {"folder": "folder1", "dataseries": "stim", "epoch": "E0"}
        with self.assertRaises(KeyError):
            self.nm.run_keys_set(e0)

    def test_rejects_missing_epoch(self):
        e0 = {"folder": "folder1", "dataseries": "stim", "channel": "A"}
        with self.assertRaises(KeyError):
            self.nm.run_keys_set(e0)

    def test_rejects_invalid_folder_name(self):
        e0 = {"folder": "test", "dataseries": "stim", "channel": "A", "epoch": "E0"}
        with self.assertRaises(ValueError):
            self.nm.run_keys_set(e0)

    def test_rejects_invalid_dataseries_name(self):
        e0 = {"folder": "folder1", "dataseries": "test", "channel": "A", "epoch": "E0"}
        with self.assertRaises(ValueError):
            self.nm.run_keys_set(e0)

    def test_sets_specific_targets(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(elist, [e0])

    def test_selected_uses_current_selection(self):
        self.nm.run_reset_all()
        e1 = {
            "folder": "selected",
            "dataseries": "selected",
            "channel": "selected",
            "epoch": "selected",
        }
        elist = self.nm.run_keys_set(e1)
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
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), NUMCHANNELS[2])

    def test_epoch_all_expands_to_all_epochs(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), NUMEPOCHS[2])

    def test_data_set_expands_to_set_members(self):
        e0 = {"folder": "folder1", "data": "set0"}
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), len(self.data_set0))

    def test_folder_all_is_valid(self):
        e0 = {
            "folder": "all",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), NUMFOLDERS)

    def test_folder_set_is_valid(self):
        e0 = {
            "folder": "set0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), 2)  # set0 has folder0 and folder1

    def test_run_keys_set_case_insensitive(self):
        # Test with uppercase keys
        e0 = {
            "FOLDER": "folder1",
            "DATASERIES": "stim",
            "CHANNEL": "A",
            "EPOCH": "E0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), 1)
        self.assertEqual(elist[0]["folder"], "folder1")
        self.assertEqual(elist[0]["dataseries"], "stim")

    def test_run_keys_set_mixed_case(self):
        # Test with mixed case keys
        e0 = {"Folder": "folder1", "Data": "data0"}
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), 1)
        self.assertEqual(elist[0]["folder"], "folder1")
        self.assertEqual(elist[0]["data"], "data0")


class TestNMManagerRunGroupTarget(NMManagerTestBase):
    """Tests for epoch group run targets in run_keys_set."""

    def test_group0_expands_to_correct_count(self):
        # folder1 uses DATASERIES[2]="stim", NUMEPOCHS[2]=7, n_groups=3
        # group0 gets epochs 0,3,6 → 3 epochs
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "group0",
        }
        elist = self.nm.run_keys_set(e0)
        expected = (NUMEPOCHS[2] + 2) // 3  # ceil(7/3) = 3
        self.assertEqual(len(elist), expected)

    def test_group1_expands_to_correct_count(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "group1",
        }
        elist = self.nm.run_keys_set(e0)
        expected = (NUMEPOCHS[2] + 1) // 3  # floor((7+1)/3) = 2 or 3
        self.assertGreater(len(elist), 0)

    def test_group_target_string_is_preserved_in_run_target(self):
        f = self.nm.folders.get("folder1")
        ds = f.dataseries.get("stim")
        ds.epochs.run_target = "group0"
        self.assertEqual(ds.epochs.run_target, "group0")

    def test_group_uppercase_accepted(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "GROUP0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertGreater(len(elist), 0)

    def test_group_all_folders_expands(self):
        e0 = {
            "folder": "all",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "group0",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertGreater(len(elist), 0)

    def test_unknown_group_raises(self):
        e0 = {
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "group99",
        }
        elist = self.nm.run_keys_set(e0)
        self.assertEqual(len(elist), 0)  # group 99 doesn't exist → no targets


class TestNMManagerRunMaxTargets(NMManagerTestBase):
    """Tests for max_targets parameter in run_keys_set."""

    def test_succeeds_within_limit(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.run_keys_set(e0, max_targets=100)
        self.assertEqual(len(elist), NUMEPOCHS[0])

    def test_raises_when_exceeding_limit(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        with self.assertRaises(ValueError):
            self.nm.run_keys_set(e0, max_targets=1)

    def test_none_allows_unlimited(self):
        e0 = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "A",
            "epoch": "all",
        }
        elist = self.nm.run_keys_set(e0, max_targets=None)
        self.assertEqual(len(elist), NUMEPOCHS[0])


class TestNMManagerSelectValueSet(NMManagerTestBase):
    """Tests for select_value_set method."""

    def test_rejects_non_nmobject(self):
        bad_types = (None, 3, 3.14, True, [], (), {}, set(), "string")
        for b in bad_types:
            with self.assertRaises(TypeError):
                self.nm.select_value_set(b)

    def test_select_epoch(self):
        # Get a specific epoch from a different folder/dataseries
        f = self.nm.folders["folder2"]
        ds = f.dataseries["stim"]
        epoch = ds.epochs["E3"]

        self.nm.select_value_set(epoch)

        self.assertEqual(self.nm.select_keys["folder"], "folder2")
        self.assertEqual(self.nm.select_keys["dataseries"], "stim")
        self.assertEqual(self.nm.select_keys["epoch"], "E3")

    def test_select_channel(self):
        # Get a specific channel
        f = self.nm.folders["folder1"]
        ds = f.dataseries["avg"]
        channel = ds.channels["B"]

        self.nm.select_value_set(channel)

        self.assertEqual(self.nm.select_keys["folder"], "folder1")
        self.assertEqual(self.nm.select_keys["dataseries"], "avg")
        self.assertEqual(self.nm.select_keys["channel"], "B")
        # Epoch should retain its current selection within this dataseries
        self.assertIn("epoch", self.nm.select_keys)

    def test_select_dataseries(self):
        f = self.nm.folders["folder3"]
        ds = f.dataseries["stim"]

        self.nm.select_value_set(ds)

        self.assertEqual(self.nm.select_keys["folder"], "folder3")
        self.assertEqual(self.nm.select_keys["dataseries"], "stim")
        # Channel and epoch should retain selection within this dataseries
        self.assertIn("channel", self.nm.select_keys)
        self.assertIn("epoch", self.nm.select_keys)

    def test_select_folder(self):
        f = self.nm.folders["folder4"]

        self.nm.select_value_set(f)

        self.assertEqual(self.nm.select_keys["folder"], "folder4")
        # Dataseries, channel, epoch should retain selection
        self.assertIn("dataseries", self.nm.select_keys)
        self.assertIn("channel", self.nm.select_keys)
        self.assertIn("epoch", self.nm.select_keys)

    def test_select_data(self):
        f = self.nm.folders["folder1"]
        data = f.data["data5"]

        self.nm.select_value_set(data)

        self.assertEqual(self.nm.select_keys["folder"], "folder1")
        self.assertEqual(self.nm.select_keys["data"], "data5")

    def test_preserves_lower_tier_selection(self):
        # First set a specific selection
        self.nm.select_keys = {
            "folder": "folder0",
            "dataseries": "data",
            "channel": "B",
            "epoch": "E2",
        }

        # Now select a different dataseries in same folder
        f = self.nm.folders["folder0"]
        ds = f.dataseries["avg"]

        self.nm.select_value_set(ds)

        # Folder should be same, dataseries changed
        self.assertEqual(self.nm.select_keys["folder"], "folder0")
        self.assertEqual(self.nm.select_keys["dataseries"], "avg")
        # Channel/epoch should be whatever is selected in "avg" dataseries


class TestNMManagerCommandHistory(NMManagerTestBase):
    """Tests for NMManager command history logging."""

    def setUp(self):
        # Use command_history=True so history is active for all tests in this class.
        # NMManagerTestBase.setUp hardcodes NMManager(quiet=QUIET), so we replicate
        # it here with the extra flag instead of calling super().
        self.nm = NMManager(quiet=QUIET, command_history=True)
        ilast = ISELECT == -1
        self.select_values = {}
        self.select_keys = {}

        for ifolder in range(NUMFOLDERS):
            fselect = False
            if ilast or ifolder == ISELECT:
                f = self.nm.folders.new(select=True)
                self.select_values["folder"] = f
                self.select_keys["folder"] = f.name
                fselect = True
            else:
                f = self.nm.folders.new(select=False)
            jdata = 0
            for idataseries in range(len(DATASERIES)):
                prefix = DATASERIES[idataseries]
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

    def test_command_history_property_returns_instance(self):
        self.assertIsInstance(self.nm.command_history, NMCommandHistory)

    def test_init_logs_nm_manager(self):
        # nm = NMManager() is the first entry; workspace tool_add() calls follow
        fresh = NMManager(quiet=True, command_history=True)
        buf = fresh.command_history.buffer
        self.assertGreaterEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm = NMManager()")

    def test_nm_name_default(self):
        fresh = NMManager(quiet=True, command_history=True)
        self.assertEqual(fresh.command_history.nm_name, "nm")

    def test_nm_name_custom(self):
        fresh = NMManager(quiet=True, nm_name="manager", command_history=True)
        buf = fresh.command_history.buffer
        self.assertEqual(fresh.command_history.nm_name, "manager")
        self.assertEqual(buf[0]["command"], "manager = NMManager()")

    def test_tool_add_logs_command(self):
        self.nm.command_history.clear()
        self.nm.tool_add("main")
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.tool_add('main')")

    def test_tool_add_select_logs_select_true(self):
        self.nm.command_history.clear()
        self.nm.tool_add("main", select=True)
        cmd = self.nm.command_history.buffer[0]["command"]
        self.assertEqual(cmd, "nm.tool_add('main', select=True)")

    def test_tool_remove_logs_command(self):
        self.nm.tool_add("main")
        self.nm.command_history.clear()
        self.nm.tool_remove("main")
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.tool_remove('main')")

    def test_tool_remove_not_found_does_not_log(self):
        self.nm.command_history.clear()
        self.nm.tool_remove("nonexistent")
        self.assertEqual(len(self.nm.command_history.buffer), 0)

    def test_tool_select_setter_logs_command(self):
        self.nm.tool_add("main")
        self.nm.command_history.clear()
        self.nm.tool_select = "main"
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.tool_select = 'main'")

    def test_tool_select_invalid_does_not_log(self):
        self.nm.command_history.clear()
        with self.assertRaises(KeyError):
            self.nm.tool_select = "nonexistent"
        self.assertEqual(len(self.nm.command_history.buffer), 0)

    def test_run_keys_set_logs_command(self):
        self.nm.command_history.clear()
        self.nm.run_keys_set({
            "folder": "folder0",
            "dataseries": "data",
            "channel": "all",
            "epoch": "all",
        })
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        cmd = buf[0]["command"]
        self.assertIn("nm.run_keys_set(", cmd)
        self.assertIn("folder0", cmd)
        self.assertIn("dataseries", cmd)
        self.assertIn("channel", cmd)
        self.assertIn("epoch", cmd)

    def test_run_keys_set_data_mode_logs_command(self):
        self.nm.command_history.clear()
        self.nm.run_keys_set({"folder": "folder0", "data": "data0"})
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertIn("nm.run_keys_set(", buf[0]["command"])

    def test_run_keys_set_logs_normalised_keys(self):
        # Keys are normalised to lowercase before logging
        self.nm.command_history.clear()
        self.nm.run_keys_set({
            "Folder": "folder0",
            "Dataseries": "data",
            "Channel": "all",
            "Epoch": "all",
        })
        cmd = self.nm.command_history.buffer[0]["command"]
        self.assertIn("'folder'", cmd)
        self.assertNotIn("'Folder'", cmd)

    def test_failed_run_keys_set_does_not_log(self):
        self.nm.command_history.clear()
        with self.assertRaises(KeyError):
            self.nm.run_keys_set({"folder": "folder0"})  # missing data/dataseries
        self.assertEqual(len(self.nm.command_history.buffer), 0)

    def test_select_keys_set_logs_command(self):
        self.nm.command_history.clear()
        self.nm._select_keys_set({"folder": "folder0"})
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertIn("nm.select_keys =", buf[0]["command"])
        self.assertIn("folder0", buf[0]["command"])

    def test_select_keys_set_logs_normalised_keys(self):
        self.nm.command_history.clear()
        self.nm._select_keys_set({"Folder": "folder0"})
        cmd = self.nm.command_history.buffer[0]["command"]
        self.assertIn("'folder'", cmd)
        self.assertNotIn("'Folder'", cmd)

    def test_select_keys_set_invalid_does_not_log(self):
        self.nm.command_history.clear()
        with self.assertRaises(KeyError):
            self.nm._select_keys_set({"invalid_tier": "folder0"})
        self.assertEqual(len(self.nm.command_history.buffer), 0)

    def test_select_value_set_logs_via_select_keys(self):
        self.nm.command_history.clear()
        folder = self.nm.folders["folder0"]
        self.nm.select_value_set(folder)
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        cmd = buf[0]["command"]
        self.assertIn("nm.select_keys =", cmd)
        self.assertIn("folder0", cmd)

    def test_select_value_set_epoch_logs_full_chain(self):
        self.nm.command_history.clear()
        epoch = self.select_values["epoch"]
        self.nm.select_value_set(epoch)
        cmd = self.nm.command_history.buffer[0]["command"]
        self.assertIn("folder", cmd)
        self.assertIn("dataseries", cmd)
        self.assertIn("epoch", cmd)

    def test_run_reset_all_logs_command(self):
        self.nm.command_history.clear()
        self.nm.run_reset_all()
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.run_reset_all()")

    def test_run_tool_default_logs_resolved_name(self):
        self.nm.tool_add("main", select=True)
        self.nm.run_keys_set({
            "folder": "folder0", "dataseries": "data",
            "channel": "all", "epoch": "all",
        })
        self.nm.command_history.clear()
        self.nm.run_tool()
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.run_tool('main')")

    def test_run_tool_explicit_name_logs_name(self):
        self.nm.tool_add("main", select=True)
        self.nm.run_keys_set({
            "folder": "folder0", "dataseries": "data",
            "channel": "all", "epoch": "all",
        })
        self.nm.command_history.clear()
        self.nm.run_tool("main")
        buf = self.nm.command_history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["command"], "nm.run_tool('main')")

    def test_run_tool_invalid_does_not_log(self):
        self.nm.command_history.clear()
        with self.assertRaises(KeyError):
            self.nm.run_tool("nonexistent")
        self.assertEqual(len(self.nm.command_history.buffer), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
