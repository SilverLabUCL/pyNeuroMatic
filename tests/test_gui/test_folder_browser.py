"""Tests for pyneuromatic.gui.folder_browser.FolderBrowserWidget."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.folder_browser import FolderBrowserWidget

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
    return f


@pytest.fixture
def widget(qtbot, nm, folder):
    w = FolderBrowserWidget(nm)
    qtbot.addWidget(w)
    w.resize(400, 600)
    w.show()  # needed for reliable visualRect()/coordinate-based clicks
    w.tree.expandAll()
    return w


def _find(model, parent, label):
    for row in range(model.rowCount(parent)):
        idx = model.index(row, 0, parent)
        if model.data(idx) == label:
            return idx
    return None


def _folders_group(model):
    return model.index(0, 0)


def _folder0(model):
    # Position-independent: "Sets" is always folder0's group's first
    # child too now, so row 0 under "Folders" isn't reliably folder0.
    return _find(model, _folders_group(model), "folder0")


def _select_row(widget, group_index, label):
    """Add the row labelled *label* under group_index to the detail selection."""
    model = widget.model
    for row in range(model.rowCount(group_index)):
        idx = model.index(row, 0, group_index)
        if model.data(idx) == label:
            widget.detail.selectionModel().select(
                idx,
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
            return idx
    raise AssertionError("row not found: %r" % label)


class TestSelectionIsPureNavigation:
    """Tree clicks never touch NMManager selection."""

    def test_selecting_leaf_does_not_touch_manager_selection(self, nm, widget):
        before = nm.select_keys
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.setCurrentIndex(rec_a0)
        assert nm.select_keys == before

    def test_selecting_leaf_updates_qt_current_index(self, widget):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.setCurrentIndex(rec_a0)
        assert widget.tree.currentIndex() == rec_a0

    def test_navigation_does_not_request_global_selection(self, widget, folder):
        requests = []
        widget.selection_requested.connect(requests.append)
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        rec_a0 = _find(model, data_group, "RecordA0")

        widget.tree.setCurrentIndex(rec_a0)

        assert requests == []


class TestActivated:
    def test_activated_does_not_raise(self, widget):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        rec_a0 = _find(model, data_group, "RecordA0")
        widget.tree.activated.emit(rec_a0)  # placeholder handler: no-op


class TestRefresh:
    def test_refresh_reflects_new_folder(self, nm, widget):
        assert widget.model.rowCount(_folders_group(widget.model)) == 1  # folder0
        nm.folders.new("folder1")
        widget.refresh()
        assert widget.model.rowCount(_folders_group(widget.model)) == 2  # folder0 + folder1

    def test_refresh_preserves_current_selection_and_detail_root(self, widget):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        widget.refresh()
        assert model.data(widget.tree.currentIndex()) == "Data"
        assert model.data(widget.detail.rootIndex()) == "Data"

    def test_refresh_preserves_expand_state(self, widget):
        model = widget.model
        ds_group = _find(model, _folder0(model), "Data Series")
        ds_idx = model.index(0, 0, ds_group)
        widget.tree.expandAll()
        assert widget.tree.isExpanded(ds_idx)
        widget.refresh()
        # Re-fetch: old indices don't survive the underlying model reset.
        ds_group2 = _find(model, _folder0(model), "Data Series")
        ds_idx2 = model.index(0, 0, ds_group2)
        assert widget.tree.isExpanded(ds_idx2)

    def test_refresh_falls_back_to_ancestor_when_selected_row_is_gone(
        self, widget, folder
    ):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        import unittest.mock as mock

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ):
            widget._delete_selected()  # deletes RecordA0, then calls refresh()

        # Selection falls back to the surviving "Data" group, not lost entirely.
        assert model.data(widget.tree.currentIndex()) == "Data"
        assert model.data(widget.detail.rootIndex()) == "Data"


class TestDetailPaneTracksTreeSelection:
    def test_selecting_group_row_sets_detail_root(self, widget):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        assert widget.detail.rootIndex() == data_group

    def test_selecting_object_row_shows_its_children(self, widget, folder):
        model = widget.model
        folder_idx = _folder0(model)
        widget.tree.setCurrentIndex(folder_idx)
        assert widget.detail.rootIndex() == folder_idx
        labels = {
            model.data(model.index(r, 0, folder_idx))
            for r in range(model.rowCount(folder_idx))
        }
        assert labels == {"Data", "Data Series"}


class TestCurrentContainer:
    def test_group_root_returns_container(self, widget, folder):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        assert widget._current_container() is folder.data

    def test_object_root_returns_none(self, widget):
        widget.tree.setCurrentIndex(_folder0(widget.model))
        assert widget._current_container() is None

    def test_no_selection_returns_none(self, widget):
        # .show() gives the tree a default current index (row 0); clear it
        # explicitly to test the true "nothing selected" state (this also
        # resets the detail root, via _on_tree_current_changed).
        widget.tree.selectionModel().clearCurrentIndex()
        assert widget._current_container() is None


class TestAddItem:
    def test_creates_new_object(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("NewData", True)),
        )
        widget._add_item()
        assert "NewData" in folder.data

    def test_cancelled_dialog_does_nothing(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        before = set(folder.data.keys())
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("", False)),
        )
        widget._add_item()
        assert set(folder.data.keys()) == before

    def test_no_container_does_nothing(self, widget, monkeypatch):
        widget.tree.selectionModel().clearCurrentIndex()  # see test_no_selection_returns_none
        calls = []
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: calls.append(1) or ("X", True)),
        )
        widget._add_item()  # nothing selected in the tree
        assert not calls


class TestRenameSelected:
    def test_renames_object(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("Renamed0", True)),
        )
        widget._rename_selected()
        assert "Renamed0" in folder.data
        assert "RecordA0" not in folder.data

    def test_requires_exactly_one_selected(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        _select_row(widget, data_group, "RecordA1")
        calls = []
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: calls.append(1) or ("X", True)),
        )
        widget._rename_selected()
        assert not calls
        assert "RecordA0" in folder.data and "RecordA1" in folder.data

    def test_duplicate_name_shows_warning_not_crash(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("RecordA1", True)),  # already exists
        )
        warnings = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: warnings.append(1)),
        )
        widget._rename_selected()  # must not raise
        assert warnings


class TestDeleteSelected:
    def test_confirmed_delete_removes_items(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        _select_row(widget, data_group, "RecordA1")
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes),
        )
        widget._delete_selected()
        assert "RecordA0" not in folder.data
        assert "RecordA1" not in folder.data

    def test_cancelled_delete_keeps_items(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No),
        )
        widget._delete_selected()
        assert "RecordA0" in folder.data

    def test_nothing_selected_does_nothing(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        calls = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: calls.append(1)),
        )
        widget._delete_selected()
        assert not calls


class TestAddToSet:
    def test_add_to_set_by_name(self, widget, folder):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        _select_row(widget, data_group, "RecordA1")
        widget._add_selected_to_set("Set1")
        items = folder.data.sets.get_items("Set1", get_keys=True)
        assert set(items) == {"RecordA0", "RecordA1"}

    def test_add_to_new_set_prompts_for_name(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("MySet", True)),
        )
        widget._add_selected_to_new_set()
        assert "RecordA0" in folder.data.sets.get_items("MySet", get_keys=True)

    def test_add_to_new_set_cancelled_does_nothing(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        widget.tree.setCurrentIndex(data_group)
        _select_row(widget, data_group, "RecordA0")
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("", False)),
        )
        widget._add_selected_to_new_set()
        assert list(folder.data.sets.keys()) == []


class TestTreeContextMenu:
    """Right-click a group row in the tree -> "New..." on its container.

    QMenu.exec() is mocked to capture the built menu instead of actually
    popping it (which would block waiting for user interaction) -
    consistent with how the detail pane's context menu is tested (its
    actions are called directly rather than exercising a real popup).
    """

    def _captured_menu(self, monkeypatch):
        captured = {}

        def fake_exec(menu_self, *a, **k):
            captured["menu"] = menu_self
            return None

        monkeypatch.setattr(QtWidgets.QMenu, "exec", fake_exec)
        return captured

    def test_new_action_present_for_group_row(self, widget, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        pos = widget.tree.visualRect(data_group).center()
        captured = self._captured_menu(monkeypatch)

        widget._show_tree_context_menu(pos)

        menu = captured["menu"]
        assert [a.text() for a in menu.actions()] == ["New...", "New Set..."]

    def test_new_action_adds_item_to_correct_container(self, widget, folder, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        pos = widget.tree.visualRect(data_group).center()
        captured = self._captured_menu(monkeypatch)
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("FromMenu", True)),
        )

        widget._show_tree_context_menu(pos)
        captured["menu"].actions()[0].trigger()

        assert "FromMenu" in folder.data

    def test_new_set_action_bootstraps_first_set_on_group(self, widget, folder, monkeypatch):
        # The whole point of offering "New Set..." on the group itself:
        # creating the *first* set works even though "Sets" isn't a
        # visible row yet (there's nothing to right-click otherwise).
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        assert _find(model, data_group, "Sets") is None
        pos = widget.tree.visualRect(data_group).center()
        captured = self._captured_menu(monkeypatch)
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("FirstSet", True)),
        )

        widget._show_tree_context_menu(pos)
        new_set_action = next(a for a in captured["menu"].actions() if a.text() == "New Set...")
        new_set_action.trigger()

        assert "FirstSet" in folder.data.sets.keys()

    def test_new_group_creates_empty_numbered_group(self, widget, folder, monkeypatch):
        dataseries_group = _find(widget.model, _folder0(widget.model), "Data Series")
        record = _find(widget.model, dataseries_group, "Record")
        epochs_group = _find(widget.model, record, "Epochs")
        widget.tree.setCurrentIndex(epochs_group)
        pos = widget.tree.visualRect(epochs_group).center()
        captured = self._captured_menu(monkeypatch)
        content_changes = []
        widget.content_changed.connect(lambda: content_changes.append(True))
        dialog_arguments = {}

        def get_group_count(*args, **kwargs):
            dialog_arguments.update(kwargs)
            return 3, True

        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getInt",
            staticmethod(get_group_count),
        )

        widget._show_tree_context_menu(pos)

        actions = captured["menu"].actions()
        assert [action.text() for action in actions] == [
            "New...",
            "New Set...",
            "New Group...",
        ]
        actions[-1].trigger()

        groups = folder.dataseries["Record"].epochs.groups
        assert dialog_arguments["min"] == 1
        assert groups.group_numbers == [0, 1, 2]
        assert all(groups.get_items(number) == [] for number in range(3))
        assert content_changes == [True]

    def test_add_selected_epochs_to_group(self, widget, folder, monkeypatch):
        dataseries_group = _find(widget.model, _folder0(widget.model), "Data Series")
        record = _find(widget.model, dataseries_group, "Record")
        epochs_group = _find(widget.model, record, "Epochs")
        epoch_container = folder.dataseries["Record"].epochs
        epoch_container.groups.add_group(3, quiet=True)
        widget.tree.setCurrentIndex(epochs_group)
        _select_row(widget, epochs_group, "E0")
        captured = self._captured_menu(monkeypatch)

        widget._show_detail_context_menu(QtCore.QPoint(0, 0))

        add_to_group = next(
            action.menu()
            for action in captured["menu"].actions()
            if action.text() == "Add to Group"
        )
        assert [action.text() for action in add_to_group.actions()] == ["Group 3"]
        add_to_group.actions()[0].trigger()

        assert epoch_container.groups.get_group("E0") == 3

    def test_assigning_epoch_moves_it_between_groups(self, widget, folder, monkeypatch):
        dataseries_group = _find(widget.model, _folder0(widget.model), "Data Series")
        record = _find(widget.model, dataseries_group, "Record")
        epochs_group = _find(widget.model, record, "Epochs")
        epoch_container = folder.dataseries["Record"].epochs
        epoch_container.groups.add_group(0, quiet=True)
        epoch_container.groups.add_group(1, quiet=True)
        epoch_container.groups.assign("E0", 0, quiet=True)
        widget.tree.setCurrentIndex(epochs_group)
        _select_row(widget, epochs_group, "E0")
        captured = self._captured_menu(monkeypatch)

        widget._show_detail_context_menu(QtCore.QPoint(0, 0))

        add_to_group = next(
            action.menu()
            for action in captured["menu"].actions()
            if action.text() == "Add to Group"
        )
        next(action for action in add_to_group.actions() if action.text() == "Group 1").trigger()

        assert epoch_container.groups.get_group("E0") == 1
        assert epoch_container.groups.get_items(0) == []
        assert epoch_container.groups.get_items(1) == ["E0"]

    def test_select_menu_for_object_row(self, widget, monkeypatch):
        # NMFolder rows aren't editable groups, but can be explicitly selected.
        pos = widget.tree.visualRect(_folder0(widget.model)).center()
        captured = self._captured_menu(monkeypatch)

        widget._show_tree_context_menu(pos)

        assert [action.text() for action in captured["menu"].actions()] == ["Select"]

    def test_select_action_for_object_row(self, widget, monkeypatch):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        record = _find(model, data_group, "RecordA0")
        pos = widget.tree.visualRect(record).center()
        captured = self._captured_menu(monkeypatch)
        requests = []
        widget.selection_requested.connect(requests.append)

        widget._show_tree_context_menu(pos)

        assert [action.text() for action in captured["menu"].actions()] == ["Select"]
        captured["menu"].actions()[0].trigger()
        assert requests == [{"object": record.internalPointer(), "set": None}]

    def test_no_menu_for_empty_area(self, widget, monkeypatch):
        captured = self._captured_menu(monkeypatch)

        widget._show_tree_context_menu(QtCore.QPoint(-10, -10))

        assert "menu" not in captured

    def test_no_new_action_for_toolfolders_group(self, widget, folder, monkeypatch):
        # Tool folders are created by NM Tools when they run, not by
        # users - no "New...". "New Set..." is still offered though:
        # grouping *existing* tool folders into a set is unrelated to
        # who creates them.
        folder.toolfolders.get_or_create("Spike_0")
        widget.refresh()
        widget.tree.expandAll()
        tf_group = _find(widget.model, _folder0(widget.model), "Tool Folders")
        pos = widget.tree.visualRect(tf_group).center()
        captured = self._captured_menu(monkeypatch)

        widget._show_tree_context_menu(pos)

        assert [a.text() for a in captured["menu"].actions()] == ["New Set..."]

    def test_context_menu_signal_is_wired(self, widget, monkeypatch):
        # Verifies customContextMenuRequested is actually connected to
        # _show_tree_context_menu (rather than relying on a raw right
        # -click, since deriving a QContextMenuEvent from a synthetic
        # QTest mouse click isn't reliable under the offscreen QPA
        # platform used in CI).
        captured = self._captured_menu(monkeypatch)
        data_group = _find(widget.model, _folder0(widget.model), "Data")
        pos = widget.tree.visualRect(data_group).center()
        widget.tree.customContextMenuRequested.emit(pos)
        assert "menu" in captured


class TestSetsTreeContextMenu:
    """Right-click "Sets" in the tree -> New Set... / Rename Set / Delete Set."""

    def _captured_menu(self, monkeypatch):
        captured = {}

        def fake_exec(menu_self, *a, **k):
            captured["menu"] = menu_self
            return None

        monkeypatch.setattr(QtWidgets.QMenu, "exec", fake_exec)
        return captured

    def _sets_node_pos(self, widget):
        # "Sets" is only present once >=1 set already exists - see
        # TestTreeContextMenu.test_new_action_present_for_group_row for
        # creating the *first* set, which goes through the parent
        # group's own menu instead (no "Sets" row to click yet).
        data_group = _find(widget.model, _folder0(widget.model), "Data")
        sets_idx = _find(widget.model, data_group, "Sets")
        return widget.tree.visualRect(sets_idx).center()

    def test_new_set_creates_additional_set(self, widget, folder, monkeypatch):
        folder.data.sets.add("Existing")
        widget.refresh()
        captured = self._captured_menu(monkeypatch)
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("Set1", True)),
        )
        widget._show_tree_context_menu(self._sets_node_pos(widget))
        menu = captured["menu"]
        new_set_action = next(a for a in menu.actions() if a.text() == "New Set...")
        new_set_action.trigger()
        assert "Set1" in folder.data.sets.keys()
        assert folder.data.sets.get_items("Set1", get_keys=True) == []

    def test_rename_and_delete_submenus_appear_once_a_set_exists(
        self, widget, folder, monkeypatch
    ):
        folder.data.sets.add("Set1")
        widget.refresh()
        captured = self._captured_menu(monkeypatch)
        widget._show_tree_context_menu(self._sets_node_pos(widget))
        menu = captured["menu"]
        titles = [a.text() for a in menu.actions()]
        assert "New Set..." in titles
        rename_menu = next(a.menu() for a in menu.actions() if a.text() == "Rename Set")
        delete_menu = next(a.menu() for a in menu.actions() if a.text() == "Delete Set")
        assert [a.text() for a in rename_menu.actions()] == ["Set1"]
        assert [a.text() for a in delete_menu.actions()] == ["Set1"]

    def test_select_menu_for_leaf_row(self, widget, monkeypatch):
        data_group = _find(widget.model, _folder0(widget.model), "Data")
        rec_a0 = _find(widget.model, data_group, "RecordA0")
        pos = widget.tree.visualRect(rec_a0).center()
        captured = self._captured_menu(monkeypatch)
        widget._show_tree_context_menu(pos)
        assert [action.text() for action in captured["menu"].actions()] == ["Select"]


class TestSetActions:
    """_add_set / _rename_set / _delete_sets / _remove_from_set directly."""

    def _sets_node(self, widget):
        # Built via the model's own cache accessor rather than found in
        # the tree - "Sets" isn't visible there until >=1 set exists,
        # but the underlying node (and the container it wraps) is
        # perfectly usable before that, exactly like the real "New
        # Set..." action on the parent group's own context menu does.
        data_group = _find(widget.model, _folder0(widget.model), "Data")
        return widget.model._sets_node(data_group.internalPointer())

    def test_add_set_creates_empty_set(self, widget, folder, monkeypatch):
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("Set1", True)),
        )
        widget._add_set(self._sets_node(widget))
        assert "Set1" in folder.data.sets.keys()

    def test_add_set_cancelled_does_nothing(self, widget, folder, monkeypatch):
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("", False)),
        )
        widget._add_set(self._sets_node(widget))
        assert list(folder.data.sets.keys()) == []

    def test_rename_set(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1", ["RecordA0"])
        widget.refresh()
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("Renamed", True)),
        )
        widget._rename_set(self._sets_node(widget), "Set1")
        assert "Renamed" in folder.data.sets.keys()
        assert "Set1" not in folder.data.sets.keys()
        assert folder.data.sets.get_items("Renamed", get_keys=True) == ["RecordA0"]

    def test_rename_set_missing_name_shows_warning_not_hang(
        self, widget, folder, monkeypatch
    ):
        # NMSets.rename() raises AttributeError (not KeyError) for a
        # name that doesn't exist - regression test for that mismatch.
        # Needs at least one *other* real set present: with an empty
        # NMSets._map, rename()'s internal loop never executes at all,
        # so the AttributeError (on `None.lower()`) never triggers -
        # the bug only shows up once there's something to iterate over.
        folder.data.sets.add("Set1")
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("New", True)),
        )
        warnings = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: warnings.append(1)),
        )
        widget._rename_set(self._sets_node(widget), "DoesNotExist")
        assert warnings

    def test_delete_sets(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1")
        folder.data.sets.add("Set2")
        widget.refresh()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes),
        )
        widget._delete_sets(self._sets_node(widget), ["Set1", "Set2"])
        assert list(folder.data.sets.keys()) == []

    def test_delete_sets_cancelled_keeps_sets(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1")
        widget.refresh()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No),
        )
        widget._delete_sets(self._sets_node(widget), ["Set1"])
        assert "Set1" in folder.data.sets.keys()

    def test_remove_from_set_does_not_delete_object(self, widget, folder):
        folder.data.sets.add("Set1", ["RecordA0", "RecordA1"])
        widget.refresh()
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        sets_idx = _find(model, data_group, "Sets")
        set1_idx = _find(model, sets_idx, "Set1")
        set1_node = set1_idx.internalPointer()

        rec_a0 = folder.data["RecordA0"]
        widget._remove_from_set(set1_node, [rec_a0])

        assert folder.data.sets.get_items("Set1", get_keys=True) == ["RecordA1"]
        assert "RecordA0" in folder.data  # object itself untouched


class TestSetsDetailPaneContextMenu:
    """Detail pane's context menu when its root is "Sets" or a specific set."""

    def _captured_menu(self, monkeypatch):
        captured = {}

        def fake_exec(menu_self, *a, **k):
            captured["menu"] = menu_self
            return None

        monkeypatch.setattr(QtWidgets.QMenu, "exec", fake_exec)
        return captured

    def _navigate_to_sets(self, widget):
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        sets_idx = _find(model, data_group, "Sets")
        widget.tree.setCurrentIndex(sets_idx)
        return sets_idx

    def test_no_new_set_action_in_detail_pane(self, widget, folder, monkeypatch):
        # "New Set..." is tree-only, same convention as "New..." for items.
        folder.data.sets.add("Set1")
        widget.refresh()
        sets_idx = self._navigate_to_sets(widget)
        set1_row = _find(widget.model, sets_idx, "Set1")
        widget.detail.selectionModel().select(
            set1_row,
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        titles = [a.text() for a in captured["menu"].actions()]
        assert "New Set..." not in titles
        assert titles == ["Rename Set...", "Delete Set"]

    def test_no_menu_when_nothing_selected_in_sets_list(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1")
        widget.refresh()
        self._navigate_to_sets(widget)
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        assert "menu" not in captured

    def test_rename_set_from_detail_pane(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1")
        widget.refresh()
        sets_idx = self._navigate_to_sets(widget)
        set1_row = _find(widget.model, sets_idx, "Set1")
        widget.detail.selectionModel().select(
            set1_row,
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        monkeypatch.setattr(
            QtWidgets.QInputDialog, "getText",
            staticmethod(lambda *a, **k: ("Renamed", True)),
        )
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        captured["menu"].actions()[0].trigger()  # "Rename Set..."
        assert "Renamed" in folder.data.sets.keys()

    def test_delete_sets_from_detail_pane_multi_select(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1")
        folder.data.sets.add("Set2")
        widget.refresh()
        sets_idx = self._navigate_to_sets(widget)
        for name in ("Set1", "Set2"):
            row = _find(widget.model, sets_idx, name)
            widget.detail.selectionModel().select(
                row,
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes),
        )
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        titles = [a.text() for a in captured["menu"].actions()]
        assert titles == ["Delete Set"]  # no Rename... with 2 selected
        captured["menu"].actions()[0].trigger()
        assert list(folder.data.sets.keys()) == []

    def test_remove_from_set_via_detail_pane(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1", ["RecordA0", "RecordA1"])
        widget.refresh()
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        sets_idx = _find(model, data_group, "Sets")
        set1_idx = _find(model, sets_idx, "Set1")
        widget.tree.setCurrentIndex(set1_idx)
        member_row = _find(model, set1_idx, "RecordA0")
        widget.detail.selectionModel().select(
            member_row,
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        assert [a.text() for a in captured["menu"].actions()] == ["Remove from Set"]
        captured["menu"].actions()[0].trigger()
        assert folder.data.sets.get_items("Set1", get_keys=True) == ["RecordA1"]
        assert "RecordA0" in folder.data

    def test_no_menu_when_nothing_selected_in_set_members(self, widget, folder, monkeypatch):
        folder.data.sets.add("Set1", ["RecordA0"])
        widget.refresh()
        model = widget.model
        data_group = _find(model, _folder0(model), "Data")
        sets_idx = _find(model, data_group, "Sets")
        set1_idx = _find(model, sets_idx, "Set1")
        widget.tree.setCurrentIndex(set1_idx)
        captured = self._captured_menu(monkeypatch)
        widget._show_detail_context_menu(QtCore.QPoint(0, 0))
        assert "menu" not in captured
