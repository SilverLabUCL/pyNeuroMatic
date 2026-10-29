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
        "Select", "Rename...", "Copy...", "Save...", "Delete...",
    ]
    menu = app.folder_actions.folder_menu(app)  # kept, so its actions live
    strip = [a.text() for a in menu.actions() if a.text()]
    assert strip[-5:] == ["Rename…", "Copy…", "Save…", "Delete…",  # Delete, not Close
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
        "Select", "Rename...", "Copy...", "Delete...",
    ]
    asked = []
    monkeypatch.setattr(app.folder_actions, "_choose",
                        lambda title, **options: asked.append((title, options["current"])))
    app.selection_model.update(folder=app.folder._container["Other"])  # another folder
    _tree_menu(app, monkeypatch, _find(app.browser_widget, subfolder))["Rename..."].trigger()
    assert asked == [("Rename Subfolder", "Stats0")]


def test_data_array_menu(app, monkeypatch):
    assert _both(app, monkeypatch, app.folder.data["RecordA0"]) == [
        "Select", "Edit Scales...", "Rename...", "Copy...", "Move...", "Delete...",
        "Add to Set",
    ]


def test_data_series_channel_and_epoch_menus(app, monkeypatch):
    # No sets for data series or channels; channels and epochs can't be renamed
    assert _both(app, monkeypatch, app.series) == [
        "Select", "Edit Scales...", "Rename...", "Copy...", "Move...", "Append...", "Delete..."]
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


def test_sets_row_new_uses_the_strip_prompt(app, monkeypatch):
    asked = []

    def ask(title, **options):
        asked.append((title, options))
        return None  # cancelled

    monkeypatch.setattr(app.epoch_set_actions, "_choose", ask)
    monkeypatch.setattr(app.data_actions, "_choose", ask)
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, kind="epochs")  # Set0, Set1, SetX
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New..."].trigger()
    browser.show_data_of(app.folder, sets=True)  # no data sets yet
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New..."].trigger()
    # The New Epoch Set / New Data Set prompts, at the next free number
    assert [title for title, _o in asked] == ["New Epoch Set", "New Data Set"]
    assert [o["extras"][0][1].value for _t, o in asked] == [2, 0]
    assert asked[1][1]["validate"](None, None, 0) is None


def test_epoch_quick_new_set_starts_with_the_next_free_name(app, monkeypatch):
    seen = []
    browser = app.browser_widget
    monkeypatch.setattr(browser, "_ask_set_number",
                        lambda _title, _label, value: seen.append(value) or (value, False))
    epoch = app.series.epochs["E1"]
    actions = _tree_menu(app, monkeypatch, _find(browser, epoch))
    next(a for a in actions["Add to Set"].menu().actions() if a.text() == "New...").trigger()
    assert seen == [2]  # Set0, Set1 exist


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


def test_add_to_set_is_last_for_data_arrays(app, monkeypatch):
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser = app.browser_widget
    index = _find(browser, app.folder.data["RecordA0"])
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    browser._show_tree_context_menu(QtCore.QPoint(0, 0))
    actions = captured["menu"].actions()
    assert [a.text() for a in actions[-2:]] == ["Delete...", "Add to Set"]
    assert not any(action.isSeparator() for action in actions)


def test_one_array_uses_the_data_prompts(app, monkeypatch):
    asked = []
    monkeypatch.setattr(app.data_actions, "_choose",
                        lambda title, **options: asked.append((title, options)) or None)
    actions = _tree_menu(app, monkeypatch, _find(app.browser_widget, app.folder.data["RecordA0"]))
    for text, title in (("Rename...", "Rename Data"), ("Copy...", "Copy Data"),
                        ("Move...", "Move Data to Folder"), ("Delete...", "Delete Data")):
        actions[text].trigger()
        got_title, options = asked[-1]
        assert got_title == title  # the same prompt as Data ⋯
        assert options["current"] == "RecordA0"  # starting on the array clicked


def test_an_array_of_another_folder_uses_its_folder(app, monkeypatch):
    folders = app.browser_widget._nm.folders
    elsewhere = folders["Other"].data.new("Far", nparray=np.zeros(2))
    asked = []
    monkeypatch.setattr(app.data_actions, "_choose",
                        lambda title, **options: asked.append(options) or None)
    app.data_actions.copy_data(elsewhere)
    options = asked[-1]
    assert options["context"] == ("From folder", "Other")
    assert options["items"] == ["Far"] and options["current"] == "Far"
    assert options["extras"][0][1][0] == "Other"  # this folder first: the array's own


def test_several_arrays_keep_the_quick_submenus(app, monkeypatch):
    browser = app.browser_widget
    data = app.folder.data
    menu = QtWidgets.QMenu()
    browser._populate_object_menu(menu, [data["RecordA0"], data["RecordA1"]], data)
    actions = {a.text(): a for a in menu.actions() if a.text()}
    assert list(actions) == ["Copy", "Move", "Delete...", "Add to Set"]
    copy_items = [a.text() for a in actions["Copy"].menu().actions() if a.text()]
    assert copy_items == ["Demo (this folder)", "Other"]
    actions["Copy"].menu().actions()[0].trigger()
    assert {"C_RecordA0", "C_RecordA1"} <= set(data.keys())


def test_series_items_use_the_prefix_prompts(app, monkeypatch):
    stim = app.folder.new_dataseries("Stim", n_channels=1, n_epochs=1, n_points=4)
    app.selection_model.update(dataseries=app.series)  # the strip shows another series
    asked = []
    monkeypatch.setattr(app.series_actions, "_choose",
                        lambda title, **options: asked.append((title, options)) or None)
    actions = _tree_menu(app, monkeypatch, _find(app.browser_widget, stim))
    for text, title in (("Rename...", "Rename Series"), ("Copy...", "Copy Series"),
                        ("Move...", "Move Series"), ("Append...", "Append Series")):
        actions[text].trigger()
        got_title, options = asked[-1]
        assert got_title == title  # the same prompt as Prefix ⋯
        if "current" in options and text != "Append...":
            assert options["current"] == "Stim"  # starting on the series clicked
        else:
            assert options["context"] == ("Series", "Demo › Stim")
    assert app.selection_model.selection["dataseries"] is app.series  # strip unchanged


def test_data_set_rows_use_the_data_set_prompts(app, monkeypatch):
    app.folder.data.sets.add("Set0", [app.folder.data["RecordA0"]])
    app.folder.data.sets.add("Set3")
    asked = []
    monkeypatch.setattr(app.data_actions, "_choose",
                        lambda title, **options: asked.append((title, options)) or None)
    browser = app.browser_widget
    browser.refresh()
    browser.show_data_of(app.folder, sets=True, select="Set3")
    actions = _tree_menu(app, monkeypatch, browser.tree.currentIndex())
    for text, title in (("Renumber...", "Renumber Data Set"), ("Clear...", "Clear Data Set"),
                        ("Delete...", "Delete Data Set")):
        actions[text].trigger()
        got_title, options = asked[-1]
        assert got_title == title and options["current"] == "Set3"  # Data Set ⋯'s prompt


def test_groups_row_uses_the_group_prompts(app, monkeypatch):
    asked = []
    monkeypatch.setattr(app.group_actions, "_choose",
                        lambda title, **options: asked.append((title, options)) or None)
    browser = app.browser_widget
    browser.show_groups_of(app.series)
    _tree_menu(app, monkeypatch, browser.tree.currentIndex())["New Groups..."].trigger()
    title, options = asked[-1]
    assert title == "New Groups" and options["context"] == ("Data Series", "Demo › Record")


def test_set_rows_copy_and_invert(app, monkeypatch):
    asked = []
    monkeypatch.setattr(app.epoch_set_actions, "_choose",
                        lambda title, **options: asked.append((title, options)) or None)
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set1", kind="epochs")
    actions = _tree_menu(app, monkeypatch, browser.tree.currentIndex())
    assert list(actions)[:4] == ["Select", "Renumber...", "Copy...", "Invert..."]
    for text, title in (("Copy...", "Copy Epoch Set"), ("Invert...", "Invert Epoch Set")):
        actions[text].trigger()
        assert asked[-1][0] == title and asked[-1][1]["current"] == "Set1"
    browser.show_data_of(app.series, sets=True, select="SetX", kind="epochs")
    actions = _tree_menu(app, monkeypatch, browser.tree.currentIndex())
    assert "Copy..." in actions and "Invert..." not in actions  # SetX: no Invert
