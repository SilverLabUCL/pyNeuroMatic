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
    """Find a child of *parent* by its display label (position-independent)."""
    for row in range(model.rowCount(parent)):
        idx = model.index(row, 0, parent)
        if model.data(idx) == label:
            return idx
    return None


def _folders_group(model):
    return model.index(0, 0)


def _folder0(model):
    return _find_child(model, _folders_group(model), "folder0")


class TestRootAndFoldersGroup:
    def test_root_has_one_child_the_folders_group(self, model):
        assert model.rowCount() == 1
        assert model.data(_folders_group(model)) == "Folders"

    def test_folders_group_parent_is_invisible_root(self, model):
        assert model.parent(_folders_group(model)) == QtCore.QModelIndex()

    def test_folders_group_is_selectable(self, model):
        # Selectable so clicking it in the tree gives normal visual
        # feedback, since it's what drives the detail pane.
        flags = model.flags(_folders_group(model))
        assert flags & QtCore.Qt.ItemFlag.ItemIsSelectable

    def test_folders_group_shown_even_with_no_folders(self):
        # So there's always a row to right-click "New..." on to create
        # the first folder, even in a brand-new, completely empty manager.
        # "Sets" itself stays hidden here too (no folder-level sets
        # defined) - see TestSets.
        nm = NMManager(quiet=True)
        model = FolderTreeModel(nm)
        assert model.rowCount() == 1
        assert model.data(_folders_group(model)) == "Folders"
        assert model.rowCount(_folders_group(model)) == 0

    def test_folder_name(self, model):
        assert model.data(_folder0(model)) == "folder0"

    def test_folder_parent_is_folders_group(self, model):
        assert model.parent(_folder0(model)) == _folders_group(model)

    def test_folder_selectable(self, model):
        flags = model.flags(_folder0(model))
        assert flags & QtCore.Qt.ItemFlag.ItemIsSelectable


class TestGroups:
    def test_folder_has_data_dataseries_and_toolfolders_groups(self, model):
        # NMFolder itself has no .sets (only its sub-containers do), so
        # "Sets" is never a sibling of Data/Data Series/Tool Folders here
        # - it's a child *within* each of them, and only once populated
        # (see TestSets).
        root = _folder0(model)
        labels = {model.data(model.index(r, 0, root)) for r in range(model.rowCount(root))}
        assert labels == {"Data", "Data Series", "Tool Folders"}

    def test_empty_folder_still_shows_data_and_dataseries(self, nm):
        # So there's always a row to right-click "New..." on, even for
        # a brand-new folder with nothing in it yet.
        nm.folders.new("empty_folder")
        model = FolderTreeModel(nm)
        root = _find_child(model, _folders_group(model), "empty_folder")
        labels = {model.data(model.index(r, 0, root)) for r in range(model.rowCount(root))}
        assert labels == {"Data", "Data Series"}

    def test_empty_data_and_dataseries_groups_have_no_children(self, nm):
        # No sets defined either, so "Sets" stays hidden too - these
        # groups are genuinely empty, not just "empty except Sets".
        nm.folders.new("empty_folder")
        model = FolderTreeModel(nm)
        root = _find_child(model, _folders_group(model), "empty_folder")
        data_group = _find_child(model, root, "Data")
        ds_group = _find_child(model, root, "Data Series")
        assert model.rowCount(data_group) == 0
        assert model.rowCount(ds_group) == 0

    def test_empty_toolfolders_group_is_hidden(self, nm):
        # Unlike Data/Data Series, Tool Folders (the group itself) stays
        # hidden when empty.
        nm.folders.new("empty_folder")
        model = FolderTreeModel(nm)
        root = _find_child(model, _folders_group(model), "empty_folder")
        assert _find_child(model, root, "Tool Folders") is None

    def test_data_group_children(self, model):
        # No sets defined on folder.data - "Sets" stays hidden.
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        names = {model.data(model.index(r, 0, data_group))
                  for r in range(model.rowCount(data_group))}
        assert names == {"RecordA0", "RecordA1", "RecordA2",
                          "RecordB0", "RecordB1", "RecordB2", "Standalone0"}

    def test_dataseries_group_children(self, model):
        root = _folder0(model)
        ds_group = _find_child(model, root, "Data Series")
        assert model.rowCount(ds_group) == 1  # just "Record" - no dataseries-level sets
        ds_idx = _find_child(model, ds_group, "Record")
        assert ds_idx is not None
        assert model.data(ds_idx) == "Record"

    def test_dataseries_has_channels_and_epochs_groups(self, model):
        root = _folder0(model)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = _find_child(model, ds_group, "Record")
        labels = {model.data(model.index(r, 0, ds_idx))
                  for r in range(model.rowCount(ds_idx))}
        assert labels == {"Channels", "Epochs"}

    def test_channels_leaves(self, model):
        root = _folder0(model)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = _find_child(model, ds_group, "Record")
        ch_group = _find_child(model, ds_idx, "Channels")
        names = {model.data(model.index(r, 0, ch_group))
                  for r in range(model.rowCount(ch_group))}
        assert names == {"A", "B"}

    def test_epochs_leaves(self, model):
        root = _folder0(model)
        ds_group = _find_child(model, root, "Data Series")
        ds_idx = _find_child(model, ds_group, "Record")
        ep_group = _find_child(model, ds_idx, "Epochs")
        names = {model.data(model.index(r, 0, ep_group))
                  for r in range(model.rowCount(ep_group))}
        assert names == {"E0", "E1", "E2"}

    def test_toolfolders_group_children(self, model):
        root = _folder0(model)
        tf_group = _find_child(model, root, "Tool Folders")
        assert model.rowCount(tf_group) == 1  # just the tool folder - no toolfolder-level sets
        name = model.data(model.index(0, 0, tf_group))
        assert name.startswith("Spike_Record_A_0")

    def test_leaf_has_no_children(self, model):
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        leaf = _find_child(model, data_group, "RecordA0")
        assert model.rowCount(leaf) == 0

    def test_group_row_is_selectable(self, model):
        # Selectable so clicking it in the tree gives normal visual
        # feedback, since it's what drives the detail pane.
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        flags = model.flags(data_group)
        assert flags & QtCore.Qt.ItemFlag.ItemIsSelectable

    def test_leaf_row_selectable(self, model):
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        leaf = _find_child(model, data_group, "RecordA0")
        flags = model.flags(leaf)
        assert flags & QtCore.Qt.ItemFlag.ItemIsSelectable


class TestSets:
    def test_sets_hidden_when_no_sets_defined(self, model):
        # Unlike Data/Data Series/Folders, "Sets" is hidden (not shown
        # empty) until the container actually has >=1 set - it appears
        # under every group, so always-showing it was too much clutter.
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        assert _find_child(model, data_group, "Sets") is None

    def test_sets_appears_once_a_set_is_defined(self, nm, folder, model):
        folder.data.sets.add("Set1", ["RecordA0", "RecordA1"])
        model.refresh()
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        sets_node = _find_child(model, data_group, "Sets")
        assert sets_node is not None
        names = [model.data(model.index(r, 0, sets_node)) for r in range(model.rowCount(sets_node))]
        assert names == ["Set1"]

    def test_set_node_lists_members(self, nm, folder, model):
        folder.data.sets.add("Set1", ["RecordA0", "RecordA1"])
        model.refresh()
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        sets_node = _find_child(model, data_group, "Sets")
        set1 = _find_child(model, sets_node, "Set1")
        members = {model.data(model.index(r, 0, set1)) for r in range(model.rowCount(set1))}
        assert members == {"RecordA0", "RecordA1"}

    def test_set_member_is_distinct_from_direct_leaf(self, nm, folder, model):
        # Same underlying NMData, but a different node identity - see
        # _SetMemberNode's docstring for why (parent-uniqueness).
        folder.data.sets.add("Set1", ["RecordA0"])
        model.refresh()
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        direct = _find_child(model, data_group, "RecordA0")
        sets_node = _find_child(model, data_group, "Sets")
        set1 = _find_child(model, sets_node, "Set1")
        via_set = _find_child(model, set1, "RecordA0")
        assert direct.internalPointer() is not via_set.internalPointer()
        assert direct.internalPointer() is via_set.internalPointer().obj

    def test_folders_group_sets_hidden_by_default(self, model):
        # The top-level "Folders" group follows the same hidden-when-
        # empty rule as every other group's "Sets" child.
        assert _find_child(model, _folders_group(model), "Sets") is None

    def test_folders_group_sets_appears_once_defined(self, nm, model):
        nm.folders.sets.add("SetOfFolders", ["folder0"])
        model.refresh()
        sets_node = _find_child(model, _folders_group(model), "Sets")
        assert sets_node is not None
        names = [model.data(model.index(r, 0, sets_node)) for r in range(model.rowCount(sets_node))]
        assert names == ["SetOfFolders"]

    def test_sets_hidden_again_after_last_set_deleted(self, nm, folder, model):
        folder.data.sets.add("Set1")
        model.refresh()
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        assert _find_child(model, data_group, "Sets") is not None

        del folder.data.sets["Set1"]
        model.refresh()
        root = _folder0(model)
        data_group = _find_child(model, root, "Data")
        assert _find_child(model, data_group, "Sets") is None


class TestParentRoundTrip:
    def test_every_node_parent_matches(self, nm, folder, model):
        """Recursively verify parent(index(child)) == parent_index for the whole tree."""
        folder.data.sets.add("Set1", ["RecordA0", "RecordA1"])
        model.refresh()

        def walk(index):
            for row in range(model.rowCount(index)):
                child = model.index(row, 0, index)
                assert model.parent(child) == index
                walk(child)

        walk(QtCore.QModelIndex())


class TestRefresh:
    def test_refresh_picks_up_new_data(self, nm, folder, model):
        assert model.rowCount(_folders_group(model)) == 1  # just folder0
        nm.folders.new("folder1")
        model.refresh()
        assert model.rowCount(_folders_group(model)) == 2  # folder0 + folder1

    def test_refresh_does_not_crash_with_no_folders(self, nm):
        model = FolderTreeModel(nm)
        model.refresh()
        assert model.rowCount() == 1  # the always-shown, empty "Folders" group
