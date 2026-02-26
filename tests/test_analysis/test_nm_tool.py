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


class TestNMToolRunAll(unittest.TestCase):
    """Tests for NMTool.run_all()."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)
        self.dataseries = self.folder.dataseries.new("test_ds")
        assert isinstance(self.dataseries, NMDataSeries)
        self.channel = self.dataseries.channels.new("A")
        self.epoch = self.dataseries.epochs.new("E0")

    def _make_target(self):
        return {
            "folder": self.folder,
            "dataseries": self.dataseries,
            "channel": self.channel,
            "epoch": self.epoch,
        }

    def test_run_all_empty_targets_returns_true(self):
        tool = NMTool()
        result = tool.run_all([])
        self.assertTrue(result)

    def test_run_all_empty_targets_calls_init_and_finish(self):
        calls = []

        class TrackTool(NMTool):
            def run_init(self):
                calls.append("init")
                return True

            def run(self):
                calls.append("run")
                return True

            def run_finish(self):
                calls.append("finish")
                return True

        tool = TrackTool()
        tool.run_all([])
        self.assertEqual(calls, ["init", "finish"])

    def test_run_all_calls_run_for_each_target(self):
        run_count = []

        class CountTool(NMTool):
            def run(self):
                run_count.append(1)
                return True

        tool = CountTool()
        targets = [self._make_target(), self._make_target(), self._make_target()]
        tool.run_all(targets)
        self.assertEqual(len(run_count), 3)

    def test_run_all_sets_selection_before_each_run(self):
        seen_folders = []

        class CaptureTool(NMTool):
            def run(self):
                seen_folders.append(self.folder)
                return True

        tool = CaptureTool()
        tool.run_all([self._make_target()])
        self.assertEqual(len(seen_folders), 1)
        self.assertIs(seen_folders[0], self.folder)

    def test_run_all_stops_early_if_run_returns_false(self):
        run_count = []

        class StopAfterOneTool(NMTool):
            def run(self):
                run_count.append(1)
                return False  # stop after first

        tool = StopAfterOneTool()
        targets = [self._make_target(), self._make_target(), self._make_target()]
        tool.run_all(targets)
        self.assertEqual(len(run_count), 1)

    def test_run_all_still_calls_run_finish_after_early_stop(self):
        calls = []

        class StopTool(NMTool):
            def run(self):
                return False

            def run_finish(self):
                calls.append("finish")
                return True

        tool = StopTool()
        tool.run_all([self._make_target()])
        self.assertIn("finish", calls)

    def test_run_all_returns_false_if_run_init_returns_false(self):
        class BadInitTool(NMTool):
            def run_init(self):
                return False

        tool = BadInitTool()
        result = tool.run_all([self._make_target()])
        self.assertFalse(result)

    def test_run_all_skips_run_if_run_init_returns_false(self):
        run_count = []

        class BadInitTool(NMTool):
            def run_init(self):
                return False

            def run(self):
                run_count.append(1)
                return True

        tool = BadInitTool()
        tool.run_all([self._make_target()])
        self.assertEqual(len(run_count), 0)

    def test_run_all_returns_run_finish_result(self):
        class FinishFalseTool(NMTool):
            def run_finish(self):
                return False

        tool = FinishFalseTool()
        result = tool.run_all([])
        self.assertFalse(result)

    def test_run_all_order_init_run_finish(self):
        calls = []

        class OrderTool(NMTool):
            def run_init(self):
                calls.append("init")
                return True

            def run(self):
                calls.append("run")
                return True

            def run_finish(self):
                calls.append("finish")
                return True

        tool = OrderTool()
        tool.run_all([self._make_target()])
        self.assertEqual(calls, ["init", "run", "finish"])

    def test_run_all_multiple_targets_order(self):
        calls = []

        class OrderTool(NMTool):
            def run_init(self):
                calls.append("init")
                return True

            def run(self):
                calls.append("run")
                return True

            def run_finish(self):
                calls.append("finish")
                return True

        tool = OrderTool()
        tool.run_all([self._make_target(), self._make_target()])
        self.assertEqual(calls, ["init", "run", "run", "finish"])


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


class TestNMToolRunMeta(unittest.TestCase):
    """Tests for NMTool.run_meta populated by run_all()."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("test_folder")
        assert isinstance(self.folder, NMFolder)
        self.dataseries = self.folder.dataseries.new("test_ds")
        assert isinstance(self.dataseries, NMDataSeries)
        self.channel = self.dataseries.channels.new("A")
        self.epoch0 = self.dataseries.epochs.new("E0")
        self.epoch1 = self.dataseries.epochs.new("E1")

    def _make_target(self, epoch=None):
        return {
            "folder": self.folder,
            "dataseries": self.dataseries,
            "channel": self.channel,
            "epoch": epoch if epoch is not None else self.epoch0,
        }

    def test_run_meta_empty_before_run_all(self):
        tool = NMTool()
        self.assertEqual(tool.run_meta, {})

    def test_run_meta_returns_copy(self):
        tool = NMTool()
        tool.run_all([])
        meta1 = tool.run_meta
        meta1["injected"] = True
        self.assertNotIn("injected", tool.run_meta)

    def test_run_meta_has_date_after_run_all(self):
        tool = NMTool()
        tool.run_all([])
        self.assertIn("date", tool.run_meta)
        self.assertIsInstance(tool.run_meta["date"], str)
        self.assertGreater(len(tool.run_meta["date"]), 0)

    def test_run_meta_run_keys_empty_without_run_keys(self):
        tool = NMTool()
        tool.run_all([self._make_target()])
        self.assertEqual(tool.run_meta["run_keys"], {})

    def test_run_meta_run_keys_stored_from_run_keys(self):
        tool = NMTool()
        run_keys = {"folder": "test_folder", "dataseries": "test_ds",
                    "channel": "A", "epoch": "Set1"}
        tool.run_all([self._make_target()], run_keys=run_keys)
        self.assertEqual(tool.run_meta["run_keys"]["epoch"], "Set1")
        self.assertEqual(tool.run_meta["run_keys"]["channel"], "A")

    def test_run_meta_run_keys_epoch_all(self):
        tool = NMTool()
        run_keys = {"folder": "test_folder", "dataseries": "test_ds",
                    "channel": "A", "epoch": "all"}
        tool.run_all([self._make_target()], run_keys=run_keys)
        self.assertEqual(tool.run_meta["run_keys"]["epoch"], "all")

    def test_run_meta_run_keys_channel_set(self):
        tool = NMTool()
        run_keys = {"folder": "test_folder", "dataseries": "test_ds",
                    "channel": "ChanSet1", "epoch": "all"}
        tool.run_all([self._make_target()], run_keys=run_keys)
        self.assertEqual(tool.run_meta["run_keys"]["channel"], "ChanSet1")

    def test_run_meta_run_keys_is_copy(self):
        tool = NMTool()
        run_keys = {"folder": "f", "dataseries": "ds", "channel": "A", "epoch": "all"}
        tool.run_all([self._make_target()], run_keys=run_keys)
        tool.run_meta["run_keys"]["injected"] = True  # mutate the returned copy
        self.assertNotIn("injected", tool.run_meta["run_keys"])

    def test_run_meta_epochs_empty_for_no_targets(self):
        tool = NMTool()
        tool.run_all([])
        self.assertEqual(tool.run_meta["epochs"], [])

    def test_run_meta_epochs_accumulates_from_targets(self):
        tool = NMTool()
        targets = [self._make_target(self.epoch0), self._make_target(self.epoch1)]
        tool.run_all(targets)
        self.assertEqual(tool.run_meta["epochs"], ["E0", "E1"])

    def test_run_meta_epochs_no_duplicates(self):
        tool = NMTool()
        # Same epoch repeated twice (e.g. multiple channels)
        targets = [self._make_target(self.epoch0), self._make_target(self.epoch0)]
        tool.run_all(targets)
        self.assertEqual(tool.run_meta["epochs"], ["E0"])

    def test_run_meta_epochs_preserves_order(self):
        tool = NMTool()
        targets = [self._make_target(self.epoch1), self._make_target(self.epoch0)]
        tool.run_all(targets)
        self.assertEqual(tool.run_meta["epochs"], ["E1", "E0"])

    def test_run_meta_channels_accumulates_from_targets(self):
        tool = NMTool()
        channel_b = self.dataseries.channels.new("B")
        t0 = {**self._make_target(self.epoch0), "channel": self.channel}
        t1 = {**self._make_target(self.epoch0), "channel": channel_b}
        tool.run_all([t0, t1])
        self.assertEqual(tool.run_meta["channels"], ["A", "B"])

    def test_run_meta_folders_accumulates_from_targets(self):
        tool = NMTool()
        targets = [self._make_target(self.epoch0)]
        tool.run_all(targets)
        self.assertIn("test_folder", tool.run_meta["folders"])

    def test_run_meta_dataseries_accumulates_from_targets(self):
        tool = NMTool()
        targets = [self._make_target(self.epoch0)]
        tool.run_all(targets)
        self.assertIn("test_ds", tool.run_meta["dataseries"])

    def test_run_meta_accessible_from_run_finish(self):
        captured = {}

        class CaptureTool(NMTool):
            def run_finish(self):
                captured.update(self.run_meta)
                return True

        tool = CaptureTool()
        run_keys = {"folder": "f", "dataseries": "ds", "channel": "A", "epoch": "Set1"}
        targets = [self._make_target(self.epoch0), self._make_target(self.epoch1)]
        tool.run_all(targets, run_keys=run_keys)

        self.assertEqual(captured["run_keys"]["epoch"], "Set1")
        self.assertEqual(captured["epochs"], ["E0", "E1"])
        self.assertIn("date", captured)

    def test_run_meta_accessible_from_run_init(self):
        captured = {}

        class CaptureTool(NMTool):
            def run_init(self):
                captured.update(self.run_meta)
                return True

        tool = CaptureTool()
        run_keys = {"folder": "f", "dataseries": "ds", "channel": "A", "epoch": "Set1"}
        tool.run_all([self._make_target()], run_keys=run_keys)

        self.assertEqual(captured["run_keys"]["epoch"], "Set1")
        self.assertIn("date", captured)

    def test_run_meta_reset_on_each_run_all(self):
        tool = NMTool()
        run_keys1 = {"folder": "f", "dataseries": "ds", "channel": "A", "epoch": "Set1"}
        run_keys2 = {"folder": "f", "dataseries": "ds", "channel": "A", "epoch": "Set2"}
        tool.run_all([self._make_target(self.epoch0)], run_keys=run_keys1)
        tool.run_all([self._make_target(self.epoch1)], run_keys=run_keys2)
        self.assertEqual(tool.run_meta["run_keys"]["epoch"], "Set2")
        self.assertEqual(tool.run_meta["epochs"], ["E1"])

    def test_run_meta_epochs_not_accumulated_across_runs(self):
        tool = NMTool()
        tool.run_all([self._make_target(self.epoch0)])
        tool.run_all([self._make_target(self.epoch1)])
        # Second run should only have epoch1, not epoch0
        self.assertEqual(tool.run_meta["epochs"], ["E1"])


class TestNMManagerRunConfig(unittest.TestCase):
    """Tests for NMManager.__run_config passed to run_all via run_tool."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        assert self.nm.project.folders is not None
        self.folder = self.nm.project.folders.new("folder0")
        assert isinstance(self.folder, NMFolder)
        self.dataseries = self.folder.dataseries.new("Record")
        assert isinstance(self.dataseries, NMDataSeries)
        self.channel = self.dataseries.channels.new("A")
        self.epoch0 = self.dataseries.epochs.new("E0")
        self.epoch1 = self.dataseries.epochs.new("E1")

    def test_run_tool_passes_epoch_set_in_run_meta(self):
        captured = {}

        class CaptureTool(NMTool):
            def run_finish(self):
                captured.update(self.run_meta)
                return True

        self.nm.tool_add("mytool", CaptureTool(), select=True)
        self.nm.run_keys_set({
            "folder": "selected",
            "dataseries": "selected",
            "channel": "selected",
            "epoch": "all",
        })
        self.nm.run_tool()
        self.assertEqual(captured["run_keys"]["epoch"], "all")

    def test_run_tool_run_meta_epochs_match_targets(self):
        captured = {}

        class CaptureTool(NMTool):
            def run_finish(self):
                captured["epochs"] = list(self.run_meta["epochs"])
                return True

        self.nm.tool_add("mytool", CaptureTool(), select=True)
        self.nm.run_keys_set({
            "folder": "selected",
            "dataseries": "selected",
            "channel": "selected",
            "epoch": "all",
        })
        self.nm.run_tool()
        self.assertIn("E0", captured["epochs"])
        self.assertIn("E1", captured["epochs"])

    def test_run_reset_all_clears_run_meta_epoch_set(self):
        captured = {}

        class CaptureTool(NMTool):
            def run_finish(self):
                captured.update(self.run_meta)
                return True

        self.nm.tool_add("mytool", CaptureTool(), select=True)
        self.nm.run_keys_set({
            "folder": "selected",
            "dataseries": "selected",
            "channel": "selected",
            "epoch": "all",
        })
        self.nm.run_reset_all()
        # After reset, run_config is None so epoch_set should be None
        self.nm.run_tool()
        self.assertEqual(captured.get("run_keys"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
