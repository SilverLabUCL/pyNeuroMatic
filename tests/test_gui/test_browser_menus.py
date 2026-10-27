"""Tests for the browser's right-click menus: the same in both columns."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Folders Demo (series Record, 2 epochs, groups; subfolder Stats0) and Other."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_epochs=2, n_points=3)
    series.epochs.groups.assign("E0", 0)
    folder.toolfolders.new("Stats0")
    manager.folders.new("Other")
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.folder, window.series = folder, series
    return window


def _actions(monkeypatch, show):
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    show()
    menu = captured.get("menu")
    return {} if menu is None else {a.text(): a for a in menu.actions() if a.text()}


def _find(browser, obj):
    """The tree row of *obj* (expanding its parents)."""
    browser.show_object(obj)
    return browser.tree.currentIndex()


def _tree_menu(app, monkeypatch, index):
    browser = app.browser_widget
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    return _actions(monkeypatch, lambda: browser._show_tree_context_menu(QtCore.QPoint(0, 0)))


def _list_menu(app, monkeypatch, index):
    """The menu for *index*'s row in the second column (its parent listed)."""
    browser = app.browser_widget
    browser.tree.setCurrentIndex(index.parent())
    view = browser.detail
    view.selectionModel().select(
        index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QtCore.QItemSelectionModel.SelectionFlag.Rows)
    monkeypatch.setattr(view, "indexAt", lambda pos: index)
    return _actions(monkeypatch, lambda: browser._show_detail_context_menu(QtCore.QPoint(0, 0)))


def _both(app, monkeypatch, obj):
    index = _find(app.browser_widget, obj)
    tree = list(_tree_menu(app, monkeypatch, index))
    listed = list(_list_menu(app, monkeypatch, index))
    assert tree == listed  # the same menu in both columns
    return tree


def test_folder_menus_match_the_strip(app, monkeypatch):
    assert _both(app, monkeypatch, app.folder) == [
        "Select", "Duplicate...", "Rename...", "Delete...",
    ]
    menu = app.folder_actions.folder_menu(app)  # kept, so its actions live
    strip = [a.text() for a in menu.actions() if a.text()]
    assert strip[-4:] == ["Duplicate…", "Rename…", "Delete…",  # Delete, not Close
                          "Show in Browser"]


def test_folder_menu_uses_the_strip_actions_on_that_folder(app, monkeypatch):
    asked = []
    monkeypatch.setattr(app.folder_actions, "_choose",
                        lambda title, **options: asked.append((title, options["current"])))
    other = app.folder._container["Other"]
    actions = _tree_menu(app, monkeypatch, _find(app.browser_widget, other))
    actions["Delete..."].trigger()
    actions["Rename..."].trigger()
    assert asked == [("Delete Folder", "Other"), ("Rename Folder", "Other")]


def test_subfolder_menu(app, monkeypatch):
    subfolder = app.folder.toolfolders["Stats0"]
    assert _both(app, monkeypatch, subfolder) == [
        "Select", "Duplicate...", "Rename...", "Delete...",
    ]
    asked = []
    monkeypatch.setattr(app.folder_actions, "_choose",
                        lambda title, **options: asked.append((title, options["current"])))
    app.selection_model.update(folder=app.folder._container["Other"])  # another folder
    _tree_menu(app, monkeypatch, _find(app.browser_widget, subfolder))["Rename..."].trigger()
    assert asked == [("Rename Subfolder", "Stats0")]


def test_data_array_menu(app, monkeypatch):
    assert _both(app, monkeypatch, app.folder.data["RecordA0"]) == [
        "Select", "Rename...", "Duplicate", "Copy to Folder", "Move to Folder", "Add to Set",
        "Delete...",
    ]


def test_data_series_channel_and_epoch_menus(app, monkeypatch):
    # No sets for data series or channels; channels and epochs can't be renamed
    assert _both(app, monkeypatch, app.series) == ["Select", "Rename...", "Delete..."]
    assert _both(app, monkeypatch, app.series.channels["A"]) == [
        "Select", "Filter && Transform...", "Delete..."]  # shows "&"
    assert _both(app, monkeypatch, app.series.epochs["E1"]) == [
        "Select", "Add to Set", "Add to Group", "Delete...",
    ]


def test_add_to_set_new(app, monkeypatch):
    monkeypatch.setattr(app.browser_widget, "_ask_set_number", lambda *a: (4, True))
    epoch = app.series.epochs["E1"]
    actions = _tree_menu(app, monkeypatch, _find(app.browser_widget, epoch))
    submenu = actions["Add to Set"].menu()
    assert [a.text() for a in submenu.actions() if a.text()][-1] == "New..."
    next(a for a in submenu.actions() if a.text() == "New...").trigger()
    assert app.series.epochs.sets.get_items("Set4", get_keys=True) == ["E1"]


def test_group_rows_have_only_new(app, monkeypatch):
    browser = app.browser_widget
    browser.show_data_of(app.folder)  # the Data row
    assert list(_tree_menu(app, monkeypatch, browser.tree.currentIndex())) == ["New..."]
    browser.show_data_of(app.series, kind="epochs")  # the Epochs row
    assert list(_tree_menu(app, monkeypatch, browser.tree.currentIndex())) == ["New..."]


def test_folders_row_new_uses_the_strip_action(app, monkeypatch):
    browser = app.browser_widget
    folders_row = browser.tree.model().index(0, 0)
    called = []
    monkeypatch.setattr(app.folder_actions, "_choose",
                        lambda title, **options: called.append(title))
    _tree_menu(app, monkeypatch, folders_row)["New..."].trigger()
    assert called == ["New Folder"]


def test_sets_rows_only_for_data_and_epochs(app, monkeypatch):
    browser = app.browser_widget
    model = browser.tree.model()
    browser.show_data_of(app.folder)
    data_row = browser.tree.currentIndex()
    labels = [model.data(model.index(r, 0, data_row)) for r in range(model.rowCount(data_row))]
    assert labels[0] == "Sets"  # shown even with no data sets yet
    folder_row = data_row.parent()
    series_row = next(model.index(r, 0, folder_row) for r in range(model.rowCount(folder_row))
                      if model.data(model.index(r, 0, folder_row)) == "Data Series")
    assert "Sets" not in [model.data(model.index(r, 0, series_row))
                          for r in range(model.rowCount(series_row))]


def test_no_set_options_for_data_series(app, monkeypatch):
    # The second column listing data series: no Add to Set / New Set
    actions = _list_menu(app, monkeypatch, _find(app.browser_widget, app.series))
    assert "Add to Set" not in actions and "New Set..." not in actions


def test_data_list_shows_only_arrays(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder)  # the Data row
    view, model = browser.detail, browser.tree.model()
    root = view.rootIndex()
    shown = [model.data(model.index(r, 0, root)) for r in range(model.rowCount(root))
             if not view.isRowHidden(r)]
    assert shown == list(app.folder.data.keys())  # no "Sets" row
    assert model.data(model.index(0, 0, root)) == "Sets"  # still in the tree


def test_new_set_prompt_starts_with_the_next_free_name(app, monkeypatch):
    seen = []

    def ask(_title, _label, value):
        seen.append(value)
        return value, False  # cancelled

    browser = app.browser_widget
    monkeypatch.setattr(browser, "_ask_set_number", ask)
    browser.show_data_of(app.series, sets=True, kind="epochs")  # Set0, Set1, SetX
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New..."].trigger()
    browser.show_data_of(app.folder, sets=True)  # no data sets yet
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New..."].trigger()
    epoch = app.series.epochs["E1"]
    actions = _tree_menu(app, monkeypatch, _find(browser, epoch))
    next(a for a in actions["Add to Set"].menu().actions() if a.text() == "New...").trigger()
    # The next free set number: 2 for epochs (Set0, Set1 exist), 0 for data
    assert seen == [2, 0, 2]


def test_new_set_refuses_a_number_in_use(app, monkeypatch):
    browser = app.browser_widget
    monkeypatch.setattr(browser, "_ask_set_number", lambda *a: (1, True))  # Set1 exists
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[2])))
    browser.show_data_of(app.series, sets=True, kind="epochs")
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New..."].trigger()
    assert warnings == ["'Set1' already exists"]
    assert list(app.series.epochs.sets.keys()) == ["Set0", "Set1", "SetX"]


def test_data_series_row_new_opens_the_new_series_dialog(app, monkeypatch):
    other = app.folder._container["Other"]
    asked = []

    def ask(defaults, validate):
        asked.append(defaults)
        return dict(defaults, name="Stim", n_channels=2, n_epochs=3, points=5)

    monkeypatch.setattr(app.series_actions, "_ask_new_series", ask)
    browser = app.browser_widget
    # The Data Series row of Other, while Demo is selected in the strip
    other_row = _find(browser, other)
    model = browser.tree.model()
    series_row = next(model.index(r, 0, other_row) for r in range(model.rowCount(other_row))
                      if model.data(model.index(r, 0, other_row)) == "Data Series")
    actions = _tree_menu(app, monkeypatch, series_row)
    assert list(actions) == ["New..."]
    actions["New..."].trigger()
    assert asked[0]["name"] == "Record"
    stim = other.dataseries["Stim"]
    assert (len(stim.channels), len(stim.epochs)) == (2, 3)
    assert "Stim" not in app.folder.dataseries
    assert app.selection_model.selection["dataseries"] is app.series  # unchanged


def test_delete_is_last(app, monkeypatch):
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser = app.browser_widget
    index = _find(browser, app.folder.data["RecordA0"])
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    browser._show_tree_context_menu(QtCore.QPoint(0, 0))
    actions = captured["menu"].actions()
    assert actions[-1].text() == "Delete..."
    assert not any(action.isSeparator() for action in actions)
