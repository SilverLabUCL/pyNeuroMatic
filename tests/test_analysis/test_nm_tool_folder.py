#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_folder: NMToolFolder and NMToolFolderContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

import numpy as np

from pyneuromatic.analysis.nm_tool_folder import NMToolFolder, NMToolFolderContainer
from pyneuromatic.core.nm_manager import NMManager

NM = NMManager(quiet=True)


def _make_container() -> NMToolFolderContainer:
    return NMToolFolderContainer(parent=NM)


class TestNMToolFolderContainerNew(unittest.TestCase):
    """Tests for NMToolFolderContainer.new()."""

    def test_first_toolfolder_not_auto_selected(self):
        # Creating the first toolfolder must NOT auto-select it — a selected
        # toolfolder switches the data/dataseries/channel/epoch context, so
        # auto-selection would silently redirect normal folder-level workflows.
        tf = _make_container()
        tf.new("Spike_0")
        self.assertIsNone(tf.selected_value)

    def test_new_with_select_true_selects(self):
        tf = _make_container()
        f = tf.new("Spike_0", select=True)
        self.assertIs(tf.selected_value, f)

    def test_second_new_without_select_leaves_selection_unchanged(self):
        tf = _make_container()
        tf.new("Spike_0", select=True)
        tf.new("Spike_1")
        self.assertEqual(tf.selected_name, "Spike_0")


class TestNMToolFolderContainerGetOrCreate(unittest.TestCase):
    """Tests for NMToolFolderContainer.get_or_create()."""

    def test_overwrite_true_creates_base_0_when_absent(self):
        tf = _make_container()
        f = tf.get_or_create("spike", overwrite=True)
        self.assertIsInstance(f, NMToolFolder)
        self.assertEqual(f.name, "spike_0")

    def test_overwrite_true_reuses_existing_base_0(self):
        tf = _make_container()
        f1 = tf.get_or_create("spike", overwrite=True)
        f2 = tf.get_or_create("spike", overwrite=True)
        self.assertIs(f1, f2)

    def test_overwrite_true_clears_data(self):
        tf = _make_container()
        f = tf.get_or_create("spike", overwrite=True)
        f.data.new("SP_rec0", nparray=np.zeros(5))
        self.assertEqual(len(list(f.data)), 1)
        f2 = tf.get_or_create("spike", overwrite=True)
        self.assertIs(f, f2)
        self.assertEqual(len(list(f2.data)), 0)

    def test_overwrite_false_creates_base_0_when_absent(self):
        tf = _make_container()
        f = tf.get_or_create("spike", overwrite=False)
        self.assertEqual(f.name, "spike_0")

    def test_overwrite_false_increments_to_1(self):
        tf = _make_container()
        tf.get_or_create("spike", overwrite=False)
        f1 = tf.get_or_create("spike", overwrite=False)
        self.assertEqual(f1.name, "spike_1")

    def test_overwrite_false_increments_to_2(self):
        tf = _make_container()
        tf.get_or_create("spike", overwrite=False)
        tf.get_or_create("spike", overwrite=False)
        f2 = tf.get_or_create("spike", overwrite=False)
        self.assertEqual(f2.name, "spike_2")

    def test_overwrite_false_does_not_clear_existing_data(self):
        tf = _make_container()
        f0 = tf.get_or_create("spike", overwrite=False)
        f0.data.new("SP_rec0", nparray=np.zeros(3))
        tf.get_or_create("spike", overwrite=False)
        self.assertEqual(len(list(f0.data)), 1)

    def test_default_overwrite_is_true(self):
        tf = _make_container()
        f1 = tf.get_or_create("spike")
        f2 = tf.get_or_create("spike")
        self.assertIs(f1, f2)
        self.assertEqual(f1.name, "spike_0")

    def test_different_bases_are_independent(self):
        tf = _make_container()
        fs = tf.get_or_create("spike", overwrite=False)
        fst = tf.get_or_create("stats", overwrite=False)
        self.assertEqual(fs.name, "spike_0")
        self.assertEqual(fst.name, "stats_0")
        self.assertIsNot(fs, fst)

    def test_overwrite_false_does_not_affect_other_base(self):
        tf = _make_container()
        tf.get_or_create("spike", overwrite=False)
        tf.get_or_create("spike", overwrite=False)
        # stats base starts fresh at _0
        f = tf.get_or_create("stats", overwrite=False)
        self.assertEqual(f.name, "stats_0")

    def test_overwrite_true_targets_0_after_false_increments(self):
        # Build spike_0, spike_1, spike_2 with overwrite=False,
        # each containing one data array.
        tf = _make_container()
        f0 = tf.get_or_create("spike", overwrite=False)
        f0.data.new("SP_rec0", nparray=np.zeros(3))
        f1 = tf.get_or_create("spike", overwrite=False)
        f1.data.new("SP_rec1", nparray=np.ones(3))
        f2 = tf.get_or_create("spike", overwrite=False)
        f2.data.new("SP_rec2", nparray=np.full(3, 2.0))

        # Now call with overwrite=True — must reuse spike_0 (cleared)
        fw = tf.get_or_create("spike", overwrite=True)
        self.assertIs(fw, f0)
        self.assertEqual(fw.name, "spike_0")
        self.assertEqual(len(list(fw.data)), 0)  # data cleared

        # spike_1 and spike_2 are untouched
        self.assertEqual(len(list(f1.data)), 1)
        self.assertEqual(len(list(f2.data)), 1)

    def test_returned_folder_is_in_container(self):
        tf = _make_container()
        f = tf.get_or_create("mybase", overwrite=True)
        self.assertIn("mybase_0", tf)

    def test_overwrite_false_all_returned_folders_in_container(self):
        tf = _make_container()
        f0 = tf.get_or_create("mybase", overwrite=False)
        f1 = tf.get_or_create("mybase", overwrite=False)
        f2 = tf.get_or_create("mybase", overwrite=False)
        self.assertIn("mybase_0", tf)
        self.assertIn("mybase_1", tf)
        self.assertIn("mybase_2", tf)
        self.assertIsNot(f0, f1)
        self.assertIsNot(f1, f2)


if __name__ == "__main__":
    unittest.main()
