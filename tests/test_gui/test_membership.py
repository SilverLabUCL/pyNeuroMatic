"""Tests for editing set and group membership with checkboxes."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.membership import MembershipModel

pytestmark = pytest.mark.gui

CHECKED = QtCore.Qt.CheckState.Checked.value
UNCHECKED = QtCore.Qt.CheckState.Unchecked.value
CHECK_ROLE = QtCore.Qt.ItemDataRole.CheckStateRole


@pytest.fixture
def series():
    """Series Record of 10 epochs; Set1 = [E1, E3]; E0-E3 in groups 0/1/0/1."""
    folder = NMManager(quiet=True).folders.new("Demo")
    series = folder.new_dataseries("Record", n_epochs=10, n_points=3)
    epochs = series.epochs
    epochs.sets.add("Set1", [epochs["E1"], epochs["E3"]])
    epochs.groups.assign_cyclic(["E0", "E1", "E2", "E3"], n_groups=2)
    return series


def _checked(model):
    return [row for row in range(model.rowCount())
            if model.data(model.index(row, 0), CHECK_ROLE) == QtCore.Qt.CheckState.Checked]


def test_set_membership_model(series):
    epochs = series.epochs
    model = MembershipModel(epochs, set_name="Set1")
    assert model.rowCount() == 10 and model.columnCount() == 1
    assert _checked(model) == [1, 3]
    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemFlag.ItemIsUserCheckable

    changes = []
    model.membership_changed.connect(changes.append)
    assert model.setData(model.index(5, 0), CHECKED, CHECK_ROLE)
    assert epochs.sets.get_items("Set1", get_keys=True) == ["E1", "E3", "E5"]
    model.setData(model.index(1, 0), UNCHECKED, CHECK_ROLE)
    assert model.members() == ["E3", "E5"]
    assert changes == [1, 1]

    assert model.check_all() == 8 and len(model.members()) == 10
    assert model.check_none() == 10 and model.members() == []
    model.set_checked([0, 2], True)
    model.invert()
    assert model.members() == ["E1", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]


def test_group_membership_model_moves_epochs(series):
    groups = series.epochs.groups
    model = MembershipModel(series.epochs, group=1)
    assert model.columnCount() == 2 and model.target_label == "Group 1"
    assert _checked(model) == [1, 3]
    assert model.data(model.index(0, 1)) == "0"
    assert model.data(model.index(5, 1)) == ""

    model.setData(model.index(0, 0), CHECKED, CHECK_ROLE)  # E0: group 0 -> 1
    assert groups.get_group("E0") == 1
    assert model.data(model.index(0, 1)) == "1"
    model.setData(model.index(3, 0), UNCHECKED, CHECK_ROLE)  # E3: out of groups
    assert groups.get_group("E3") is None
    assert groups.get_items(1) == ["E0", "E1"]


def _app_for(qtbot, series):
    window = NMAppWindow(series._manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series)
    return window


def test_browser_edit_mode_for_a_set(qtbot, series):
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    assert browser.edit_checkbox.isVisibleTo(browser)
    assert not browser.is_editing  # view mode: the set's members

    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    assert browser.is_editing and model.target_label == "Set1"
    assert _checked(model) == [1, 3]

    # Ticking updates the set, the strip's epoch count and keeps edit mode
    window.selection_model.update(epoch_set="Set1")
    model.setData(model.index(6, 0), CHECKED, CHECK_ROLE)
    assert series.epochs.sets.get_items("Set1", get_keys=True) == ["E1", "E3", "E6"]
    assert window.selection_strip.epoch_count_label.text() == "3 epochs"
    assert browser.is_editing and browser.membership_model is model

    # The tree's member rows follow
    current = browser.tree.currentIndex()
    tree = browser.tree.model()
    assert [tree.data(tree.index(r, 0, current)) for r in range(tree.rowCount(current))] == [
        "E1", "E3", "E6",
    ]

    # All / None buttons
    browser._all_button.click()
    assert len(series.epochs.sets.get_items("Set1", get_keys=True)) == 10
    browser._none_button.click()
    assert series.epochs.sets.get_items("Set1", get_keys=True) == []


def test_shift_click_ticks_a_range(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    model.setData(model.index(4, 0), CHECKED, CHECK_ROLE)
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                        staticmethod(lambda: QtCore.Qt.KeyboardModifier.ShiftModifier))
    model.setData(model.index(8, 0), CHECKED, CHECK_ROLE)
    assert series.epochs.sets.get_items("Set1", get_keys=True) == [
        "E1", "E3", "E4", "E5", "E6", "E7", "E8",
    ]


def test_edit_mode_moves_between_sets_and_groups(qtbot, series):
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    browser.edit_checkbox.setChecked(True)

    browser.show_groups_of(series, 1)
    model = browser.membership_model
    assert browser.is_editing and model.target_label == "Group 1"
    model.setData(model.index(9, 0), CHECKED, CHECK_ROLE)
    assert series.epochs.groups.get_group("E9") == 1

    # A row that is not a set or group hides the bar and leaves edit mode
    browser.show_object(series)
    assert not browser.is_editing
    assert not browser.edit_checkbox.isVisibleTo(browser)


def test_combined_set_cannot_be_edited(qtbot, series):
    series.epochs.sets.add("Set2", [series.epochs["E2"]])
    series.epochs.sets.define_or("Both", "Set1", "Set2")
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Both", kind="epochs")
    assert browser.edit_checkbox.isVisibleTo(browser)
    assert not browser.edit_checkbox.isEnabled()
    assert "combined" in browser.edit_checkbox.toolTip()


def test_data_set_edit_mode(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for name in ("A0", "B0", "C0"):
        folder.data.new(name, nparray=np.array([1.0]))
    folder.data.sets.add("Mine", [folder.data["B0"]])
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    browser = window.browser_widget
    browser.show_data_of(folder, sets=True, select="Mine")
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    assert model.columnCount() == 1 and _checked(model) == [1]
    model.setData(model.index(2, 0), CHECKED, CHECK_ROLE)
    assert folder.data.sets.get_items("Mine", get_keys=True) == ["B0", "C0"]


def _menu_for(monkeypatch, show):
    """Run *show* with QMenu.exec stubbed; returns the menu's actions by text."""
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    show()
    menu = captured.get("menu")
    return {} if menu is None else {a.text(): a for a in menu.actions()}


def _child(browser, parent, label):
    tree = browser.tree.model()
    for row in range(tree.rowCount(parent)):
        index = tree.index(row, 0, parent)
        if tree.data(index) == label:
            return index
    raise AssertionError(label)


def _tree_menu(browser, monkeypatch, index):
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    return _menu_for(monkeypatch,
                     lambda: browser._show_tree_context_menu(QtCore.QPoint(0, 0)))


def _detail_menu(browser, monkeypatch, labels):
    view = browser.detail
    tree = view.model()
    root = view.rootIndex()
    for label in labels:
        view.selectionModel().select(
            _child(browser, root, label),
            QtCore.QItemSelectionModel.SelectionFlag.Select
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
    return _menu_for(monkeypatch,
                     lambda: browser._show_detail_context_menu(QtCore.QPoint(0, 0)))


def test_remove_from_set_in_tree(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    member = _child(browser, browser.tree.currentIndex(), "E3")
    actions = _tree_menu(browser, monkeypatch, member)
    assert list(actions) == ["Select", "Remove from Set"]
    actions["Remove from Set"].trigger()
    assert series.epochs.sets.get_items("Set1", get_keys=True) == ["E1"]
    assert "E3" in series.epochs


def test_remove_from_group_in_tree(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_groups_of(series, 1)
    member = _child(browser, browser.tree.currentIndex(), "E1")
    actions = _tree_menu(browser, monkeypatch, member)
    assert list(actions) == ["Select", "Remove from Group"]
    actions["Remove from Group"].trigger()
    groups = series.epochs.groups
    assert groups.get_items(1) == ["E3"] and groups.get_group("E1") is None
    assert "E1" in series.epochs


def test_remove_from_group_in_detail_pane(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_groups_of(series, 0)
    actions = _detail_menu(browser, monkeypatch, ["E0", "E2"])
    assert list(actions) == ["Remove from Group"]
    actions["Remove from Group"].trigger()
    assert series.epochs.groups.get_items(0) == []
    assert series.epochs.groups.get_items(1) == ["E1", "E3"]


def test_no_remove_from_group_menu_without_selection(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_groups_of(series, 0)
    assert _detail_menu(browser, monkeypatch, []) == {}


def test_emptied_groups_are_kept(series):
    groups = series.epochs.groups
    model = MembershipModel(series.epochs, group=1)
    model.check_none()  # Group 1 was made by assigning epochs
    assert groups.group_numbers == [0, 1] and groups.get_items(1) == []
    model.check_all()  # moves Group 0's epochs here; Group 0 stays
    assert groups.group_numbers == [0, 1] and groups.get_items(0) == []


def test_ticking_inside_a_selection_ticks_every_selected_row(qtbot, series):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    browser.edit_checkbox.setChecked(True)
    view, model = browser.membership_view, browser.membership_model
    browser._select_member_rows([5, 6, 7])

    # A real click on a box inside the selection
    rect = view.visualRect(model.index(6, 0))
    box = QtCore.QPoint(rect.left() + 10, rect.center().y())
    qtbot.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=box)
    assert series.epochs.sets.get_items("Set1", get_keys=True) == [
        "E1", "E3", "E5", "E6", "E7",
    ]
    qtbot.waitUntil(lambda: browser._selected_member_rows() == [5, 6, 7])

    # Unticking one of them unticks all three
    qtbot.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=box)
    assert series.epochs.sets.get_items("Set1", get_keys=True) == ["E1", "E3"]

    # A box outside the selection ticks just its own row
    model.setData(model.index(9, 0), CHECKED, CHECK_ROLE)
    assert series.epochs.sets.get_items("Set1", get_keys=True) == ["E1", "E3", "E9"]


def test_group_edit_buttons(qtbot, series):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_groups_of(series, 1)
    browser.edit_checkbox.setChecked(True)
    assert browser._all_button.text() == "Ungrouped"
    assert not browser._invert_button.isVisibleTo(browser)

    # Ungrouped takes E4-E9 and leaves Group 0 alone
    browser._all_button.click()
    groups = series.epochs.groups
    assert groups.get_items(0) == ["E0", "E2"]
    assert groups.get_items(1) == ["E1", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]

    # Back on a set: All and Invert
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    assert browser._all_button.text() == "All"
    assert browser._invert_button.isVisibleTo(browser)


def test_clear_set_in_tree(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    actions = _tree_menu(browser, monkeypatch, browser.tree.currentIndex())
    assert list(actions) == ["Select Set", "Rename Set...", "Delete Set", "Clear Set"]
    actions["Clear Set"].trigger()
    assert series.epochs.sets.get_items("Set1", get_keys=True) == []
    assert "Set1" in series.epochs.sets and len(series.epochs) == 10


def test_clear_set_from_sets_row(qtbot, series, monkeypatch):
    series.epochs.sets.add("Set2", [series.epochs["E2"]])
    series.epochs.sets.define_or("Both", "Set1", "Set2")
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    browser.show_data_of(series, sets=True, kind="epochs")  # the Sets row
    actions = _tree_menu(browser, monkeypatch, browser.tree.currentIndex())
    clear_menu = actions["Clear Set"].menu()
    # A combined set is defined by others, so it is not offered
    assert [a.text() for a in clear_menu.actions()] == ["Set1", "Set2", "SetX"]
    next(a for a in clear_menu.actions() if a.text() == "Set2").trigger()
    assert series.epochs.sets.get_items("Set2", get_keys=True) == []

    # Detail pane: selected set rows
    actions = _detail_menu(browser, monkeypatch, ["Set1", "Both"])
    assert "Clear Set" in actions
    actions["Clear Set"].trigger()
    assert series.epochs.sets.get_items("Set1", get_keys=True) == []
    assert series.epochs.sets.is_equation("Both")


def test_clear_group_and_groups(qtbot, series, monkeypatch):
    groups = series.epochs.groups
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)

    browser.show_groups_of(series, 1)
    actions = _tree_menu(browser, monkeypatch, browser.tree.currentIndex())
    assert list(actions) == ["Select Group", "Clear Group", "Delete Group"]
    actions["Clear Group"].trigger()
    assert groups.group_numbers == [0, 1] and groups.get_items(1) == []

    groups.assign("E5", 1, quiet=True)
    browser.refresh()
    browser.show_groups_of(series)  # the Groups row
    actions = _tree_menu(browser, monkeypatch, browser.tree.currentIndex())
    assert list(actions) == ["New Groups...", "Clear Groups", "Delete Group"]
    actions["Clear Groups"].trigger()
    assert groups.group_numbers == [0, 1]
    assert groups.get_items(0) == [] and groups.get_items(1) == []


def test_clear_group_in_detail_pane(qtbot, series, monkeypatch):
    groups = series.epochs.groups
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    browser.show_groups_of(series)
    actions = _detail_menu(browser, monkeypatch, ["Group 0"])
    assert list(actions) == ["Select Group", "Clear Group", "Delete Group"]
    actions["Clear Group"].trigger()
    assert groups.get_items(0) == [] and groups.get_items(1) == ["E1", "E3"]


def test_clear_is_cancelled_without_confirmation(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: False)
    browser.show_groups_of(series, 0)
    _tree_menu(browser, monkeypatch, browser.tree.currentIndex())["Clear Group"].trigger()
    assert series.epochs.groups.get_items(0) == ["E0", "E2"]


def test_rename_set_in_tree(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Mine", True)))
    browser.show_data_of(series, sets=True, select="Set1", kind="epochs")
    _tree_menu(browser, monkeypatch, browser.tree.currentIndex())["Rename Set..."].trigger()
    assert "Mine" in series.epochs.sets and "Set1" not in series.epochs.sets
    assert series.epochs.sets.get_items("Mine", get_keys=True) == ["E1", "E3"]


def test_select_group_in_detail_pane(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    browser.show_groups_of(series)
    _detail_menu(browser, monkeypatch, ["Group 1"])["Select Group"].trigger()
    assert window.selection_model.selection["group"] == 1
    # No Select Group with two groups selected
    assert "Select Group" not in _detail_menu(browser, monkeypatch, ["Group 0"])


def test_delete_group(qtbot, series, monkeypatch):
    groups = series.epochs.groups
    window = _app_for(qtbot, series)
    browser = window.browser_widget
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    window.selection_model.update(group=1)

    browser.show_groups_of(series, 1)
    _tree_menu(browser, monkeypatch, browser.tree.currentIndex())["Delete Group"].trigger()
    assert groups.group_numbers == [0] and groups.get_group("E1") is None
    assert len(series.epochs) == 10
    assert window.selection_model.selection["group"] is None  # strip follows

    # Groups row: Delete Group > a number, or All Groups
    groups.add_group(1)
    groups.add_group(2)
    browser.refresh()
    browser.show_groups_of(series)
    delete_menu = _tree_menu(browser, monkeypatch, browser.tree.currentIndex())[
        "Delete Group"].menu()
    texts = [a.text() for a in delete_menu.actions() if a.text()]
    assert texts == ["0", "1", "2", "All Groups"]
    next(a for a in delete_menu.actions() if a.text() == "2").trigger()
    assert groups.group_numbers == [0, 1]

    # Detail pane, several at once
    _detail_menu(browser, monkeypatch, ["Group 0", "Group 1"])["Delete Group"].trigger()
    assert groups.group_numbers == []
    assert browser.tree.model().rowCount(browser.tree.currentIndex()) == 0


def test_epochs_rows_sets_and_groups_have_tree_menus(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_data_of(series, kind="epochs")  # column 2 lists Sets, Groups, E0...
    view = browser.detail
    root = view.rootIndex()

    def menu_at(label):
        index = _child(browser, root, label)
        view.selectionModel().clearSelection()
        monkeypatch.setattr(view, "indexAt", lambda pos: index)
        return _menu_for(monkeypatch,
                         lambda: browser._show_detail_context_menu(QtCore.QPoint(0, 0)))

    assert list(menu_at("Sets")) == ["New Set...", "Rename Set", "Delete Set", "Clear Set"]
    assert list(menu_at("Groups")) == ["New Groups...", "Clear Groups", "Delete Group"]


def test_empty_space_under_groups_list(qtbot, series, monkeypatch):
    window = _app_for(qtbot, series)  # keep the window alive
    browser = window.browser_widget
    browser.show_groups_of(series)
    monkeypatch.setattr(browser.detail, "indexAt", lambda pos: QtCore.QModelIndex())
    actions = _menu_for(monkeypatch,
                        lambda: browser._show_detail_context_menu(QtCore.QPoint(0, 0)))
    assert list(actions) == ["New Groups...", "Clear Groups", "Delete Group"]
