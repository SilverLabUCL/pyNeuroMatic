"""Tests for pyneuromatic.gui.folder_model.FolderTreeModel."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.folder_model import FolderTreeModel

pytestmark = pytest.mark.gui


@pytest.fixture
def nm():
    return NMManager(quiet=True)


@pytest.fixture
def folder(nm):
    f = nm.folders.new("folder0")
    for chan in ("A", "B"):
        for epoch in range(3):
            f.data.new(
                "Record%s%d" % (chan, epoch),
                nparray=np.array([1.0, 2.0, 3.0]),
            )
    f.sync_dataseries("Record")
    f.data.new("Standalone0", nparray=np.array([9.0]))  # no dataseries
    f.toolfolders.get_or_create("Spike_Record_A_0")
    return f


@pytest.fixture
def model(nm, folder):
    return FolderTreeModel(nm)


def _find_child(model, parent, label):
    for row in range(model.rowCount(parent)):
        idx = model.index(row, 0, parent)
        if model.data(idx) == label:
            return idx
    return None


class TestTopLevel:
    def test_rowcount_is_folder_count(self, model):
        assert model.rowCount() == 1

    def test_folder_name(self, model):
        root = model.index(0, 0)
        assert model.data(root) == "folder0"

    def test_folder_parent_is_invisible_root(self, model):
        root = model.index(0, 0)
        assert model.parent(root) == QtCore.QModelIndex()


class TestGroups:
    def test_folder_has_three_groups(self, model):
        root = model.index(0, 0)
        labels = {model.data(model.index(r, 0, root)) for r in range(model.rowCount(root))}
        assert labels == {"Data", "Data Series", "Tool Folders"}

    def test_empty_group_is_hidden(self, nm):
        nm.folders.new("empty_folder")
        model = FolderTreeModel(nm)
        root = model.index(0, 0)
        assert model.rowCount(root) == 0

    def test_data_group_children(self, model):
        root = model.index(0, 0)
        data_group = _find_child(model, root, "Data")
        names = {model.data(model.index(r, 0, data_group))
                  for r in range(model.rowCount(data_group))}
        assert names == {"RecordA0", "RecordA1", "RecordA2",
                          "RecordB0", "RecordB1", "RecordB2", "Standalone0"}

    def test_dataseries_group_children(self, model):
        root = model.index(0, 0)
        ds_group = _find_child(model, root, "Data Series")
        assert model.rowCount(ds_group) == 1
        ds_idx = model.index(0, 0, ds_group)
        assert model.data(ds_idx) == "Record"

    def test_dataseries_has_channels_and_epochs_groups(self, model):
        root = model.index(0, 0)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = model.index(0, 0, ds_group)
        labels = {model.data(model.index(r, 0, ds_idx))
                  for r in range(model.rowCount(ds_idx))}
        assert labels == {"Channels", "Epochs"}

    def test_channels_leaves(self, model):
        root = model.index(0, 0)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = model.index(0, 0, ds_group)
        ch_group = _find_child(model, ds_idx, "Channels")
        names = [model.data(model.index(r, 0, ch_group))
                  for r in range(model.rowCount(ch_group))]
        assert names == ["A", "B"]

    def test_epochs_leaves(self, model):
        root = model.index(0, 0)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = model.index(0, 0, ds_group)
        ep_group = _find_child(model, ds_idx, "Epochs")
        names = [model.data(model.index(r, 0, ep_group))
                  for r in range(model.rowCount(ep_group))]
        assert names == ["E0", "E1", "E2"]

    def test_toolfolders_group_children(self, model):
        root = model.index(0, 0)
        tf_group = _find_child(model, root, "Tool Folders")
        assert model.rowCount(tf_group) == 1
        name = model.data(model.index(0, 0, tf_group))
        assert name.startswith("Spike_Record_A_0")

    def test_leaf_has_no_children(self, model):
        root = model.index(0, 0)
        data_group = _find_child(model, root, "Data")
        leaf = model.index(0, 0, data_group)
        assert model.rowCount(leaf) == 0

    def test_group_row_not_selectable(self, model):
        root = model.index(0, 0)
        data_group = _find_child(model, root, "Data")
        flags = model.flags(data_group)
        assert not (flags & QtCore.Qt.ItemFlag.ItemIsSelectable)

    def test_leaf_row_selectable(self, model):
        root = model.index(0, 0)
        data_group = _find_child(model, root, "Data")
        leaf = model.index(0, 0, data_group)
        flags = model.flags(leaf)
        assert flags & QtCore.Qt.ItemFlag.ItemIsSelectable


class TestParentRoundTrip:
    def test_every_node_parent_matches(self, model):
        """Recursively verify parent(index(child)) == parent_index for the whole tree."""
        def walk(index):
            for row in range(model.rowCount(index)):
                child = model.index(row, 0, index)
                assert model.parent(child) == index
                walk(child)

        walk(QtCore.QModelIndex())


class TestRefresh:
    def test_refresh_picks_up_new_data(self, nm, folder, model):
        assert model.rowCount() == 1
        nm.folders.new("folder1")
        model.refresh()
        assert model.rowCount() == 2

    def test_refresh_does_not_crash_with_no_folders(self, nm):
        model = FolderTreeModel(nm)
        model.refresh()
        assert model.rowCount() == 0
