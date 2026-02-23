# -*- coding: utf-8 -*-
"""
Tests for NMTool base class.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

from pyneuromatic.core.nm_manager import NMManager, SELECT_LEVELS
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.analysis.nm_tool import NMTool

QUIET = True


class TestNMToolInit(unittest.TestCase):
    """Tests for NMTool initialization."""

    def test_init_creates_select_dict(self):
        tool = NMTool()
        self.assertIsInstance(tool._select, dict)
        self.assertEqual(set(tool._select.keys()), set(SELECT_LEVELS))

    def test_init_all_values_none(self):
        tool = NMTool()
        for level in SELECT_LEVELS:
            self.assertIsNone(tool._select[level])


class TestNMToolProperties(unittest.TestCase):
    """Tests for NMTool read-only properties."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        self.tool = NMTool()

        # Create test objects
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)

        self.data = self.folder.data.new("test_data")
        assert isinstance(self.data, NMData)

        self.dataseries = self.folder.dataseries.new("test_ds")
        assert isinstance(self.dataseries, NMDataSeries)

        self.channel = self.dataseries.channels.new("A")
        assert isinstance(self.channel, NMChannel)

        self.epoch = self.dataseries.epochs.new("E0")
        assert isinstance(self.epoch, NMEpoch)

    def test_folder_property_returns_none_when_empty(self):
        self.assertIsNone(self.tool.folder)

    def test_folder_property_returns_folder(self):
        self.tool._select["folder"] = self.folder
        self.assertIs(self.tool.folder, self.folder)

    def test_folder_property_returns_none_for_wrong_type(self):
        self.tool._select["folder"] = self.data  # Wrong type
        self.assertIsNone(self.tool.folder)

    def test_data_property_returns_none_when_empty(self):
        self.assertIsNone(self.tool.data)

    def test_data_property_returns_data(self):
        self.tool._select["data"] = self.data
        self.assertIs(self.tool.data, self.data)

    def test_data_property_returns_none_for_wrong_type(self):
        self.tool._select["data"] = self.folder  # Wrong type
        self.assertIsNone(self.tool.data)

    def test_dataseries_property_returns_none_when_empty(self):
        self.assertIsNone(self.tool.dataseries)

    def test_dataseries_property_returns_dataseries(self):
        self.tool._select["dataseries"] = self.dataseries
        self.assertIs(self.tool.dataseries, self.dataseries)

    def test_dataseries_property_returns_none_for_wrong_type(self):
        self.tool._select["dataseries"] = self.folder  # Wrong type
        self.assertIsNone(self.tool.dataseries)

    def test_channel_property_returns_none_when_empty(self):
        self.assertIsNone(self.tool.channel)

    def test_channel_property_returns_channel(self):
        self.tool._select["channel"] = self.channel
        self.assertIs(self.tool.channel, self.channel)

    def test_channel_property_returns_none_for_wrong_type(self):
        self.tool._select["channel"] = self.folder  # Wrong type
        self.assertIsNone(self.tool.channel)

    def test_epoch_property_returns_none_when_empty(self):
        self.assertIsNone(self.tool.epoch)

    def test_epoch_property_returns_epoch(self):
        self.tool._select["epoch"] = self.epoch
        self.assertIs(self.tool.epoch, self.epoch)

    def test_epoch_property_returns_none_for_wrong_type(self):
        self.tool._select["epoch"] = self.folder  # Wrong type
        self.assertIsNone(self.tool.epoch)


class TestNMToolSelectValues(unittest.TestCase):
    """Tests for NMTool select_values property."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        self.tool = NMTool()

        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)

        self.dataseries = self.folder.dataseries.new("test_ds")
        assert isinstance(self.dataseries, NMDataSeries)

        self.channel = self.dataseries.channels.new("A")
        self.epoch = self.dataseries.epochs.new("E0")

    def test_select_values_getter_returns_copy(self):
        values = self.tool.select_values
        self.assertIsNot(values, self.tool._select)

    def test_select_values_getter_has_all_levels(self):
        values = self.tool.select_values
        self.assertEqual(set(values.keys()), set(SELECT_LEVELS))

    def test_select_values_getter_returns_current_values(self):
        self.tool._select["folder"] = self.folder
        self.tool._select["dataseries"] = self.dataseries

        values = self.tool.select_values
        self.assertIs(values["folder"], self.folder)
        self.assertIs(values["dataseries"], self.dataseries)
        self.assertIsNone(values["data"])

    def test_select_values_setter_updates_values(self):
        values = {
            "folder": self.folder,
            "dataseries": self.dataseries,
            "channel": self.channel,
            "epoch": self.epoch,
        }
        self.tool.select_values = values

        self.assertIs(self.tool._select["folder"], self.folder)
        self.assertIs(self.tool._select["dataseries"], self.dataseries)
        self.assertIs(self.tool._select["channel"], self.channel)
        self.assertIs(self.tool._select["epoch"], self.epoch)

    def test_select_values_setter_partial_update(self):
        # Set initial value
        self.tool._select["folder"] = self.folder

        # Partial update (only dataseries)
        self.tool.select_values = {"dataseries": self.dataseries}

        # folder should remain, dataseries should be updated
        self.assertIs(self.tool._select["folder"], self.folder)
        self.assertIs(self.tool._select["dataseries"], self.dataseries)

    def test_select_values_setter_ignores_unknown_keys(self):
        values = {
            "folder": self.folder,
            "unknown_key": "ignored",
        }
        self.tool.select_values = values

        self.assertIs(self.tool._select["folder"], self.folder)
        self.assertNotIn("unknown_key", self.tool._select)

    def test_select_values_setter_allows_none(self):
        self.tool._select["folder"] = self.folder

        self.tool.select_values = {"folder": None}

        self.assertIsNone(self.tool._select["folder"])


class TestNMToolSelectKeys(unittest.TestCase):
    """Tests for NMTool select_keys property."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        self.tool = NMTool()

        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)

        self.dataseries = self.folder.dataseries.new("test_ds")

    def test_select_keys_returns_none_for_empty(self):
        keys = self.tool.select_keys
        for level in SELECT_LEVELS:
            self.assertIsNone(keys[level])

    def test_select_keys_returns_names(self):
        self.tool._select["folder"] = self.folder
        self.tool._select["dataseries"] = self.dataseries

        keys = self.tool.select_keys
        self.assertEqual(keys["folder"], "test_folder")
        self.assertEqual(keys["dataseries"], "test_ds")
        self.assertIsNone(keys["data"])
        self.assertIsNone(keys["channel"])
        self.assertIsNone(keys["epoch"])

    def test_select_keys_has_all_levels(self):
        keys = self.tool.select_keys
        self.assertEqual(set(keys.keys()), set(SELECT_LEVELS))


class TestNMToolRunMethods(unittest.TestCase):
    """Tests for NMTool run methods."""

    def test_run_init_returns_true(self):
        tool = NMTool()
        self.assertTrue(tool.run_init())

    def test_run_returns_true(self):
        tool = NMTool()
        self.assertTrue(tool.run())

    def test_run_finish_returns_true(self):
        tool = NMTool()
        self.assertTrue(tool.run_finish())


class TestNMToolSubclass(unittest.TestCase):
    """Tests for NMTool subclassing."""

    def test_subclass_can_override_run(self):
        class CustomTool(NMTool):
            def __init__(self):
                super().__init__()
                self.run_finished = False

            def run(self) -> bool:
                self.run_finished = True
                return True

        tool = CustomTool()
        self.assertFalse(tool.run_finished)
        result = tool.run()
        self.assertTrue(result)
        self.assertTrue(tool.run_finished)

    def test_subclass_can_access_selection(self):
        nm = NMManager(quiet=QUIET)
        assert nm.project.folders is not None
        folder = nm.project.folders.new("test_folder")
        assert isinstance(folder, NMFolder)

        class CustomTool(NMTool):
            def run(self) -> bool:
                return self.folder is not None

        tool = CustomTool()
        tool._select["folder"] = folder

        self.assertTrue(tool.run())
        self.assertEqual(tool.folder.name, "test_folder")

    def test_subclass_run_init_called_before_run(self):
        call_order = []

        class CustomTool(NMTool):
            def run_init(self) -> bool:
                call_order.append("init")
                return True

            def run(self) -> bool:
                call_order.append("run")
                return True

            def run_finish(self) -> bool:
                call_order.append("finish")
                return True

        tool = CustomTool()
        tool.run_init()
        tool.run()
        tool.run_finish()

        self.assertEqual(call_order, ["init", "run", "finish"])


class TestNMToolIntegration(unittest.TestCase):
    """Integration tests with NMManager."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)

        self.dataseries = self.folder.dataseries.new("test_ds")
        assert isinstance(self.dataseries, NMDataSeries)

        self.channel = self.dataseries.channels.new("A")
        self.epoch = self.dataseries.epochs.new("E0")

    def test_tool_receives_manager_selection(self):
        tool = NMTool()

        # Get selection from manager and set on tool
        tool.select_values = self.nm.select_values

        # Tool should have the same selection as manager
        self.assertEqual(
            tool.select_keys["folder"],
            self.nm.select_keys["folder"]
        )

    def test_select_values_roundtrip(self):
        tool = NMTool()

        # Set values
        original = {
            "folder": self.folder,
            "dataseries": self.dataseries,
            "channel": self.channel,
            "epoch": self.epoch,
        }
        tool.select_values = original

        # Get values back
        retrieved = tool.select_values

        # Verify all set values are correct
        self.assertIs(retrieved["folder"], self.folder)
        self.assertIs(retrieved["dataseries"], self.dataseries)
        self.assertIs(retrieved["channel"], self.channel)
        self.assertIs(retrieved["epoch"], self.epoch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
