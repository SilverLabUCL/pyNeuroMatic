#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMDataSeries and NMDataSeriesContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

from pyneuromatic.core.nm_channel import NMChannel, NMChannelContainer
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_epoch import NMEpoch, NMEpochContainer
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.core.nm_utilities as nmu


class NMDataSeriesTestBase(unittest.TestCase):
    """Base class with common setUp for NMDataSeries tests."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")


class TestNMDataSeriesInit(NMDataSeriesTestBase):
    """Tests for NMDataSeries initialization."""

    def test_init_with_defaults(self):
        ds = NMDataSeries()
        self.assertEqual(ds.name, "NMDataSeries0")
        self.assertIsInstance(ds.channels, NMChannelContainer)
        self.assertIsInstance(ds.epochs, NMEpochContainer)

    def test_init_with_name(self):
        ds = NMDataSeries(parent=self.nm, name="Record")
        self.assertEqual(ds.name, "Record")
        self.assertEqual(ds._parent, self.nm)

    def test_init_rejects_invalid_copy_arg(self):
        with self.assertRaises(TypeError):
            NMDataSeries(copy=self.nm)

    def test_channels_property(self):
        self.assertIsInstance(self.ds.channels, NMChannelContainer)

    def test_epochs_property(self):
        self.assertIsInstance(self.ds.epochs, NMEpochContainer)


class TestNMDataSeriesEquality(NMDataSeriesTestBase):
    """Tests for NMDataSeries equality comparison."""

    def test_equal_empty(self):
        ds1 = NMDataSeries(parent=self.nm, name="Record")
        ds2 = NMDataSeries(parent=self.nm, name="Record")
        self.assertEqual(ds1, ds2)

    def test_not_equal_different_name(self):
        ds1 = NMDataSeries(parent=self.nm, name="Record")
        ds2 = NMDataSeries(parent=self.nm, name="Avg")
        self.assertNotEqual(ds1, ds2)

    def test_not_equal_different_channels(self):
        ds1 = NMDataSeries(parent=self.nm, name="Record")
        ds2 = NMDataSeries(parent=self.nm, name="Record")
        ds1.channels.new()  # Add channel to ds1
        self.assertNotEqual(ds1, ds2)

    def test_not_equal_different_epochs(self):
        ds1 = NMDataSeries(parent=self.nm, name="Record")
        ds2 = NMDataSeries(parent=self.nm, name="Record")
        ds1.epochs.new()  # Add epoch to ds1
        self.assertNotEqual(ds1, ds2)

    def test_equal_with_same_channels_epochs(self):
        ds1 = NMDataSeries(parent=self.nm, name="Record")
        ds2 = NMDataSeries(parent=self.nm, name="Record")
        ds1.channels.new()
        ds1.epochs.new()
        ds2.channels.new()
        ds2.epochs.new()
        self.assertEqual(ds1, ds2)

    def test_not_equal_to_non_dataseries(self):
        result = self.ds.__eq__("not a dataseries")
        self.assertEqual(result, NotImplemented)


class TestNMDataSeriesWithData(unittest.TestCase):
    """Tests for NMDataSeries with actual data populated."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")

        # Create channels A and B
        self.chan_a = self.ds.channels.new()
        self.chan_b = self.ds.channels.new()

        # Create epochs E0, E1, E2
        self.epoch_0 = self.ds.epochs.new()
        self.epoch_1 = self.ds.epochs.new()
        self.epoch_2 = self.ds.epochs.new()

        # Create data and link to channels/epochs
        self.data = {}
        for ch_name, channel in [("A", self.chan_a), ("B", self.chan_b)]:
            for i, epoch in enumerate([self.epoch_0, self.epoch_1, self.epoch_2]):
                name = f"Record{ch_name}{i}"
                d = NMData(parent=self.nm, name=name)
                self.data[name] = d
                channel.data.append(d)
                epoch.data.append(d)

        # Select channel B, epoch E1
        self.ds.channels.selected_name = "B"
        self.ds.epochs.selected_name = "E1"

    def test_get_selected_returns_data(self):
        result = self.ds.get_selected()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "RecordB1")

    def test_get_selected_with_keys(self):
        result = self.ds.get_selected(get_keys=True)
        self.assertEqual(result, ["RecordB1"])

    def test_get_selected_no_channel_selected(self):
        self.ds.channels.selected_name = None
        result = self.ds.get_selected()
        self.assertEqual(result, [])

    def test_get_selected_no_epoch_selected(self):
        self.ds.epochs.selected_name = None
        result = self.ds.get_selected()
        self.assertEqual(result, [])

    def test_get_data_with_explicit_channel_epoch(self):
        result = self.ds.get_data(channel="A", epoch="E2")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "RecordA2")

    def test_get_data_uses_selected_when_none(self):
        result = self.ds.get_data()
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "RecordB1")

    def test_get_data_invalid_channel(self):
        result = self.ds.get_data(channel="Z", epoch="E0")
        self.assertIsNone(result)

    def test_get_data_invalid_epoch(self):
        result = self.ds.get_data(channel="A", epoch="E99")
        self.assertIsNone(result)


class TestNMDataSeriesBulkDimensions(unittest.TestCase):
    """Tests for bulk dimension setting methods."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")

        # Create channels with data that has dimensions
        self.chan_a = self.ds.channels.new()
        self.chan_b = self.ds.channels.new()

        xscale = {"start": 0, "delta": 0.01, "label": "time", "units": "ms"}
        yscale_a = {"label": "current", "units": "pA"}
        yscale_b = {"label": "voltage", "units": "mV"}

        # Create data with scale dicts
        for i in range(3):
            d_a = NMData(
                parent=self.nm, name=f"RecordA{i}",
                xscale=dict(xscale), yscale=dict(yscale_a)
            )
            self.chan_a.data.append(d_a)

            d_b = NMData(
                parent=self.nm, name=f"RecordB{i}",
                xscale=dict(xscale), yscale=dict(yscale_b)
            )
            self.chan_b.data.append(d_b)

    def test_set_xstart_all_channels(self):
        count = self.ds.set_xstart(0.5)
        self.assertEqual(count, 6)  # 3 data per channel * 2 channels
        for d in self.chan_a.data:
            self.assertEqual(d.xscale["start"], 0.5)
        for d in self.chan_b.data:
            self.assertEqual(d.xscale["start"], 0.5)

    def test_set_xstart_single_channel(self):
        count = self.ds.set_xstart(0.5, channel="A")
        self.assertEqual(count, 3)
        for d in self.chan_a.data:
            self.assertEqual(d.xscale["start"], 0.5)
        # Channel B unchanged
        for d in self.chan_b.data:
            self.assertEqual(d.xscale["start"], 0)

    def test_set_xdelta_all_channels(self):
        count = self.ds.set_xdelta(0.02)
        self.assertEqual(count, 6)
        for d in self.chan_a.data:
            self.assertEqual(d.xscale["delta"], 0.02)

    def test_set_xlabel_all_channels(self):
        count = self.ds.set_xlabel("Time")
        self.assertEqual(count, 6)
        for d in self.chan_a.data:
            self.assertEqual(d.xscale["label"], "Time")

    def test_set_xunits_single_channel(self):
        count = self.ds.set_xunits("s", channel="B")
        self.assertEqual(count, 3)
        for d in self.chan_b.data:
            self.assertEqual(d.xscale["units"], "s")
        # Channel A unchanged
        for d in self.chan_a.data:
            self.assertEqual(d.xscale["units"], "ms")

    def test_set_ylabel_single_channel(self):
        count = self.ds.set_ylabel("Current", channel="A")
        self.assertEqual(count, 3)
        for d in self.chan_a.data:
            self.assertEqual(d.yscale["label"], "Current")

    def test_set_yunits_all_channels(self):
        count = self.ds.set_yunits("nA")
        self.assertEqual(count, 6)
        for d in self.chan_a.data:
            self.assertEqual(d.yscale["units"], "nA")

    def test_set_xstart_invalid_channel(self):
        with self.assertRaises(KeyError):
            self.ds.set_xstart(0.5, channel="Z")

    def test_set_xstart_rejects_non_number(self):
        with self.assertRaises(TypeError):
            self.ds.set_xstart("invalid")

    def test_set_xlabel_rejects_non_string(self):
        with self.assertRaises(TypeError):
            self.ds.set_xlabel(123)


class TestNMDataSeriesScalesSummary(unittest.TestCase):
    """Tests for scale summary diagnostic methods."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")

        self.chan_a = self.ds.channels.new()
        self.chan_b = self.ds.channels.new()

        # Create data with uniform scales for channel A
        for i in range(3):
            d = NMData(
                parent=self.nm, name=f"RecordA{i}",
                xscale={"start": 0, "delta": 0.01},
                yscale={"label": "current", "units": "pA"},
            )
            self.chan_a.data.append(d)

        # Create data with non-uniform scales for channel B
        for i in range(3):
            d = NMData(
                parent=self.nm, name=f"RecordB{i}",
                xscale={"start": i * 0.1, "delta": 0.01 + i * 0.001},
                yscale={"label": "voltage", "units": "mV"},
            )
            self.chan_b.data.append(d)

    def test_get_xscales_summary_uniform(self):
        summary = self.ds.get_xscales_summary()
        self.assertIn("A", summary)
        self.assertTrue(summary["A"]["uniform"])
        self.assertEqual(summary["A"]["start"], {0})
        self.assertEqual(summary["A"]["delta"], {0.01})

    def test_get_xscales_summary_non_uniform(self):
        summary = self.ds.get_xscales_summary()
        self.assertIn("B", summary)
        self.assertFalse(summary["B"]["uniform"])
        self.assertEqual(len(summary["B"]["start"]), 3)  # 3 different values
        self.assertEqual(len(summary["B"]["delta"]), 3)

    def test_get_yscales_summary(self):
        summary = self.ds.get_yscales_summary()
        self.assertIn("A", summary)
        self.assertEqual(summary["A"]["label"], {"current"})
        self.assertEqual(summary["A"]["units"], {"pA"})
        self.assertIn("B", summary)
        self.assertEqual(summary["B"]["label"], {"voltage"})
        self.assertEqual(summary["B"]["units"], {"mV"})


class TestNMDataSeriesDeepCopy(unittest.TestCase):
    """Tests for NMDataSeries deep copy."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")
        self.ds.channels.new()
        self.ds.epochs.new()

    def test_deepcopy_creates_new_instance(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertIsNot(ds_copy, self.ds)

    def test_deepcopy_preserves_name(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertEqual(ds_copy.name, self.ds.name)

    def test_deepcopy_creates_new_containers(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertIsNot(ds_copy.channels, self.ds.channels)
        self.assertIsNot(ds_copy.epochs, self.ds.epochs)

    def test_deepcopy_preserves_channel_count(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertEqual(len(ds_copy.channels), len(self.ds.channels))

    def test_deepcopy_preserves_epoch_count(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertEqual(len(ds_copy.epochs), len(self.ds.epochs))

    def test_deepcopy_sets_copy_of(self):
        ds_copy = copy.deepcopy(self.ds)
        self.assertIs(ds_copy._NMObject__copy_of, self.ds)


class TestNMDataSeriesContent(NMDataSeriesTestBase):
    """Tests for NMDataSeries content and parameters."""

    def test_content_includes_channels(self):
        self.ds.channels.new()
        content = self.ds.content
        self.assertIn("nmchannelcontainer", content)

    def test_content_includes_epochs(self):
        self.ds.epochs.new()
        content = self.ds.content
        self.assertIn("nmepochcontainer", content)

    def test_parameters_returns_dict(self):
        params = self.ds.parameters
        self.assertIsInstance(params, dict)


class TestNMDataSeriesContainer(unittest.TestCase):
    """Tests for NMDataSeriesContainer."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.container = NMDataSeriesContainer(parent=self.nm)

    def test_content_type(self):
        self.assertEqual(self.container.content_type(), "NMDataSeries")

    def test_new_creates_dataseries(self):
        ds = self.container.new(name="Record")
        self.assertIsInstance(ds, NMDataSeries)
        self.assertEqual(ds.name, "Record")

    def test_new_with_select(self):
        ds = self.container.new(name="Record", select=True)
        self.assertEqual(self.container.selected_name, "Record")

    def test_new_sets_correct_parent(self):
        # Parent should be container's parent (NMManager), not container itself
        ds = self.container.new(name="Record")
        self.assertEqual(ds._parent, self.nm)

    def test_duplicate_raises_error(self):
        with self.assertRaises(RuntimeError):
            self.container.duplicate()

    def test_multiple_dataseries(self):
        ds1 = self.container.new(name="Record")
        ds2 = self.container.new(name="Avg")
        self.assertEqual(len(self.container), 2)
        self.assertIn("Record", self.container)
        self.assertIn("Avg", self.container)


class TestNMDataSeriesSets(unittest.TestCase):
    """Tests for NMDataSeries with channel and epoch sets."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.ds = NMDataSeries(parent=self.nm, name="Record")

        # Create channels A, B, C, D
        for _ in range(4):
            self.ds.channels.new()

        # Create epochs E0-E9
        for _ in range(10):
            self.ds.epochs.new()

    def test_channel_sets(self):
        self.ds.channels.sets.add("set0", ["A", "B"])
        self.ds.channels.sets.add("set1", ["C", "D"])
        self.ds.channels.sets.define_or("set2", "set0", "set1")

        result = self.ds.channels.sets.get("set2", get_keys=True)
        self.assertEqual(set(result), {"A", "B", "C", "D"})

    def test_epoch_sets(self):
        # Even epochs
        for i in range(0, 10, 2):
            self.ds.epochs.sets.add("even", f"E{i}")
        # Odd epochs
        for i in range(1, 10, 2):
            self.ds.epochs.sets.add("odd", f"E{i}")

        even_result = self.ds.epochs.sets.get("even", get_keys=True)
        self.assertEqual(len(even_result), 5)

        odd_result = self.ds.epochs.sets.get("odd", get_keys=True)
        self.assertEqual(len(odd_result), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
