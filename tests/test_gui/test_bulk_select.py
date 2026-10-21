"""Tests for choosing many folders, data arrays or data series in the browser
(Edit mode on their lists), with a name filter and Delete..."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.folder_browser import name_matches
from pyneuromatic.gui.membership import ChoiceModel

pytestmark = pytest.mark.gui

CHECKED = QtCore.Qt.CheckState.Checked.value
CHECK_ROLE = QtCore.Qt.ItemDataRole.CheckStateRole


@pytest.fixture
def app(qtbot):
    """Folder Demo: data RecordA0-2, RecordB0-2, Copy_copy; folders Old1, Old2."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for name in ("RecordA0", "RecordA1", "RecordA2", "RecordB0", "RecordB1",
                 "RecordB2", "Copy_copy"):
        folder.data.new(name, nparray=np.array([1.0]))
    manager.folders.new("Old1")
    manager.folders.new("Old2")
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.manager, window.folder = manager, folder
    return window


def _edit_data(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder)  # the Data row
    browser.edit_checkbox.setChecked(True)
    return browser, browser.membership_model


def _delete_ticked(browser, monkeypatch):
    """Right-click the list and choose Delete... (for the ticked items)."""
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser._show_chosen_context_menu(QtCore.QPoint(0, 0))
    next(a for a in captured["menu"].actions() if a.text() == "Delete...").trigger()


def _shown(browser):
    model = browser.membership_model
    return [model.name_at(r) for r in range(model.rowCount())
            if not browser.membership_view.isRowHidden(r)]


@pytest.mark.parametrize("pattern, name, match", [
    ("", "RecordA0", True),
    ("RecordB*", "RecordB2", True),
    ("recordb*", "RecordB2", True),  # case-insensitive
    ("RecordB*", "RecordA0", False),
    ("*_copy", "Copy_copy", True),
    ("Record?0", "RecordA0", True),
    ("ordA", "RecordA1", True),  # no wildcards: part of the name
    ("ordC", "RecordA1", False),
])
def test_name_matches(pattern, name, match):
    assert name_matches(name, pattern) is match


def test_edit_on_the_data_list(app):
    browser, model = _edit_data(app)
    assert isinstance(model, ChoiceModel) and browser.is_editing
    assert _shown(browser) == list(app.folder.data.keys())  # no Sets row
    assert browser.filter_edit.isVisibleTo(browser)
    model.setData(model.index(0, 0), CHECKED, CHECK_ROLE)
    assert [d.name for d in model.chosen()] == ["RecordA0"]
    assert len(app.folder.data) == 7  # ticking changes nothing


def test_filter_then_all_then_delete(app, monkeypatch):
    browser, model = _edit_data(app)
    browser.filter_edit.setText("RecordB*")
    assert _shown(browser) == ["RecordB0", "RecordB1", "RecordB2"]
    browser._all_button.click()  # only what the filter shows
    assert [d.name for d in model.chosen()] == ["RecordB0", "RecordB1", "RecordB2"]

    asked = []
    monkeypatch.setattr(browser, "_confirm", lambda title, text: asked.append(text) or True)
    _delete_ticked(browser, monkeypatch)
    assert asked == ["Delete 3 data arrays: RecordB0, RecordB1, RecordB2?"]
    assert list(app.folder.data.keys()) == ["RecordA0", "RecordA1", "RecordA2", "Copy_copy"]
    # Still editing the (reloaded) list, with nothing ticked
    assert browser.is_editing and browser.membership_model.chosen() == []


def test_long_confirmations_list_the_first_names(app, monkeypatch):
    browser, model = _edit_data(app)
    browser._all_button.click()
    asked = []
    monkeypatch.setattr(browser, "_confirm", lambda title, text: asked.append(text) or False)
    _delete_ticked(browser, monkeypatch)
    assert asked == ["Delete 7 data arrays: RecordA0, RecordA1, RecordA2, "
                     "RecordB0, RecordB1, ...?"]
    assert len(app.folder.data) == 7  # cancelled


def test_clear_and_invert_follow_the_filter(app):
    browser, model = _edit_data(app)
    browser._all_button.click()
    browser.filter_edit.setText("RecordA*")
    browser._none_button.click()  # unticks only RecordA*
    assert [d.name for d in model.chosen()] == ["RecordB0", "RecordB1", "RecordB2", "Copy_copy"]
    browser._invert_button.click()  # RecordA* ticked again; the rest untouched
    assert len(model.chosen()) == 7


def test_shift_click_skips_filtered_rows(app, monkeypatch):
    browser, model = _edit_data(app)
    browser.filter_edit.setText("*0")  # RecordA0, RecordB0
    model.setData(model.index(0, 0), CHECKED, CHECK_ROLE)  # RecordA0
    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                        staticmethod(lambda: QtCore.Qt.KeyboardModifier.ShiftModifier))
    model.setData(model.index(3, 0), CHECKED, CHECK_ROLE)  # RecordB0
    assert [d.name for d in model.chosen()] == ["RecordA0", "RecordB0"]


def test_right_click_menu_for_the_ticked_items(app, monkeypatch):
    browser, model = _edit_data(app)
    model.set_checked([0, 1], True)
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser._show_chosen_context_menu(QtCore.QPoint(0, 0))
    texts = [a.text() for a in captured["menu"].actions() if a.text()]
    assert texts == ["Duplicate", "Copy to Folder", "Add to Set", "Delete..."]
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Pair", True)))
    add = next(a for a in captured["menu"].actions() if a.text() == "Add to Set").menu()
    next(a for a in add.actions() if a.text() == "New...").trigger()
    assert app.folder.data.sets.get_items("Pair", get_keys=True) == ["RecordA0", "RecordA1"]


def test_delete_folders_keeps_a_folder_selected(app, monkeypatch):
    browser = app.browser_widget
    folders_row = browser.tree.model().index(0, 0)
    browser.tree.setCurrentIndex(folders_row)
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    assert [model.name_at(r) for r in range(model.rowCount())] == ["Demo", "Old1", "Old2"]
    model.set_checked([0, 1], True)  # Demo (selected in the strip) and Old1
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    _delete_ticked(browser, monkeypatch)
    assert list(app.manager.folders.keys()) == ["Old2"]
    assert app.selection_model.selection["folder"].name == "Old2"


def test_data_series_and_subfolder_lists(app):
    app.folder.new_dataseries("Stim", n_epochs=1, n_points=2)
    app.folder.toolfolders.new("Stats0")
    browser = app.browser_widget
    browser.refresh()
    browser.show_object(app.folder.dataseries["Stim"])
    browser.tree.setCurrentIndex(browser.tree.currentIndex().parent())  # Data Series
    browser.edit_checkbox.setChecked(True)
    assert isinstance(browser.membership_model, ChoiceModel)
    assert browser.membership_model.name_at(0) == "Stim"
    browser.show_object(app.folder.toolfolders["Stats0"])  # an object: Edit goes off
    browser.tree.setCurrentIndex(browser.tree.currentIndex().parent())  # Subfolders
    browser.edit_checkbox.setChecked(True)
    assert browser.membership_model.name_at(0) == "Stats0"


def test_locked_series_data_are_kept(app, monkeypatch):
    series = app.folder.new_dataseries("Rec", n_epochs=1, n_points=2)
    series.locked = True
    browser, model = _edit_data(app)
    browser.filter_edit.setText("Rec*")
    browser._all_button.click()
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[2])))
    _delete_ticked(browser, monkeypatch)
    assert "RecA0" in app.folder.data  # in the locked series
    assert "RecordA0" not in app.folder.data
    assert "locked" in warnings[0]


def test_filter_starts_empty_on_another_list(app):
    browser, _model = _edit_data(app)
    browser.filter_edit.setText("RecordB*")
    browser.tree.setCurrentIndex(browser.tree.model().index(0, 0))  # Folders
    assert browser.filter_edit.text() == ""
    assert len(_shown(browser)) == 3


def test_filter_for_set_edit_mode_too(app):
    app.folder.data.sets.add("Mine")
    browser = app.browser_widget
    browser.refresh()
    browser.show_data_of(app.folder, sets=True, select="Mine")
    browser.edit_checkbox.setChecked(True)
    assert browser.filter_edit.isVisibleTo(browser)
    browser.filter_edit.setText("*A*")
    browser._all_button.click()
    assert app.folder.data.sets.get_items("Mine", get_keys=True) == [
        "RecordA0", "RecordA1", "RecordA2",
    ]


def test_edit_off_shows_the_normal_list(app):
    browser, _model = _edit_data(app)
    browser.edit_checkbox.setChecked(False)
    assert not browser.is_editing
    assert not browser.filter_edit.isVisibleTo(browser)


def test_no_menu_with_nothing_ticked(app, monkeypatch):
    browser, _model = _edit_data(app)
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser._show_chosen_context_menu(QtCore.QPoint(0, 0))
    assert "menu" not in captured


def test_delete_a_data_series_uses_the_strips_dialog(app, monkeypatch):
    app.folder.new_dataseries("Stim", n_epochs=2, n_points=2)
    other = app.manager.folders["Old1"]
    other.new_dataseries("Rec", n_epochs=1, n_points=2)
    browser = app.browser_widget
    browser.refresh()
    asked = []

    def choose(title, **options):
        asked.append((title, options.get("current"), options["context"], options["extras"]))
        return ("Rec", None, "Delete")  # and delete its arrays

    monkeypatch.setattr(app.series_actions, "_choose", choose)
    # A series in another folder than the strip's, right-clicked in the tree
    browser.show_object(other.dataseries["Rec"])
    index = browser.tree.currentIndex()
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    browser._show_tree_context_menu(QtCore.QPoint(0, 0))
    next(a for a in captured["menu"].actions() if a.text() == "Delete...").trigger()
    title, current, context, extras = asked[0]
    assert (title, current, context) == ("Delete Series", "Rec", ("Folder", "Old1"))
    assert extras == [("Arrays", ["Keep", "Delete"], "Keep")]
    assert "Rec" not in other.dataseries and "RecA0" not in other.data  # arrays too
    assert "Stim" in app.folder.dataseries


def test_delete_ticked_data_series_keeps_their_arrays(app, monkeypatch):
    for prefix in ("Stim", "Puff", "Rec"):
        app.folder.new_dataseries(prefix, n_epochs=1, n_points=2)
    browser = app.browser_widget
    browser.refresh()
    browser.show_object(app.folder.dataseries["Stim"])
    browser.tree.setCurrentIndex(browser.tree.currentIndex().parent())  # Data Series
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    model.set_checked([0, 1], True)  # Stim, Puff
    asked = []

    def choose(title, **options):
        asked.append((title, options["message"], options["validate"](None, None, "Keep")))
        return (None, None, "Keep")

    monkeypatch.setattr(app.series_actions, "_choose", choose)
    _delete_ticked(browser, monkeypatch)
    title, message, error = asked[0]
    assert title == "Delete Series" and message.startswith("Delete 2 data series: Stim, Puff?")
    assert error is None
    assert list(app.folder.dataseries.keys()) == ["Rec"]
    assert "StimA0" in app.folder.data  # arrays kept


def test_locked_data_series_is_not_deleted(app, monkeypatch):
    app.folder.new_dataseries("Stim", n_epochs=1, n_points=2).locked = True
    browser = app.browser_widget
    browser.refresh()
    errors = []
    monkeypatch.setattr(app.series_actions, "_choose",
                        lambda title, **o: errors.append(o["validate"]("Stim", None, "Keep")))
    app.series_actions.delete_series_list(app.folder, ["Stim"])
    assert "locked" in errors[0]
    assert "Stim" in app.folder.dataseries
