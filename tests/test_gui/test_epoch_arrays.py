"""Tests for epochs in the browser: their data arrays, and the Edit mode."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.membership import ChoiceModel

pytestmark = pytest.mark.gui

DISPLAY = QtCore.Qt.ItemDataRole.DisplayRole


@pytest.fixture
def app(qtbot):
    """Series Record: 2 channels x 4 epochs; Set0 = E1, E3; groups 0 / 1."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=4, n_points=3)
    series.epochs.sets.add("Set0", [series.epochs["E1"], series.epochs["E3"]])
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), n_groups=2)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.folder, window.series = folder, series
    return window


def _rows(browser):
    """(name, data) of the second column's shown rows."""
    view, model = browser.detail, browser.tree.model()
    root = view.rootIndex()
    return [(model.data(model.index(r, 0, root)), model.data(model.index(r, 1, root)))
            for r in range(model.rowCount(root)) if not view.isRowHidden(r)]


def _menu(monkeypatch, show):
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    show()
    menu = captured.get("menu")
    return {} if menu is None else {a.text(): a for a in menu.actions() if a.text()}


# --- the Data column -------------------------------------------------------

def test_epochs_list_shows_their_arrays(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, kind="epochs")
    assert not browser.detail.isColumnHidden(1)
    model = browser.tree.model()
    assert model.headerData(1, QtCore.Qt.Orientation.Horizontal, DISPLAY) == "Data"
    assert _rows(browser)[:2] == [("E0", "RecordA0, RecordB0"), ("E1", "RecordA1, RecordB1")]


def test_set_and_group_members_show_their_arrays(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set0", kind="epochs")
    assert _rows(browser) == [("E1", "RecordA1, RecordB1"), ("E3", "RecordA3, RecordB3")]
    browser.show_groups_of(app.series, 0)
    assert _rows(browser) == [("E0", "RecordA0, RecordB0"), ("E2", "RecordA2, RecordB2")]


def test_data_column_only_for_epochs(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder)  # the Data list
    assert browser.detail.isColumnHidden(1)
    assert browser.tree.isColumnHidden(1)  # the tree stays narrow


def test_tree_tooltip_names_the_arrays(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set0", kind="epochs")
    model = browser.tree.model()
    member = model.index(0, 0, browser.tree.currentIndex())  # E1 under Set0
    assert model.data(member, QtCore.Qt.ItemDataRole.ToolTipRole) == "RecordA1, RecordB1"


def test_edit_lists_show_the_arrays(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set0", kind="epochs")
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    assert model.data(model.index(2, 1)) == "RecordA2, RecordB2"


# --- Edit mode for the Epochs list ---------------------------------------

def _edit_epochs(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, kind="epochs")
    browser.edit_checkbox.setChecked(True)
    return browser, browser.membership_model


def test_edit_on_the_epochs_list(app):
    browser, model = _edit_epochs(app)
    assert isinstance(model, ChoiceModel) and browser.is_editing
    assert model.columnCount() == 2 and model.data(model.index(0, 1)) == "RecordA0, RecordB0"
    assert browser.filter_edit.placeholderText() == "Filter, e.g. E1*"
    browser.filter_edit.setText("E1*")
    shown = [model.name_at(r) for r in range(model.rowCount())
             if not browser.membership_view.isRowHidden(r)]
    assert shown == ["E1"]


def test_right_click_ticked_epochs(app, monkeypatch):
    browser, model = _edit_epochs(app)
    model.set_checked([0, 2], True)  # E0, E2
    actions = _menu(monkeypatch, lambda: browser._show_chosen_context_menu(QtCore.QPoint(0, 0)))
    assert list(actions) == ["Add to Set", "Add to Group", "Delete..."]
    monkeypatch.setattr(browser, "_ask_set_number", lambda *a: (4, True))
    add = actions["Add to Set"].menu()
    next(a for a in add.actions() if a.text() == "New...").trigger()
    assert app.series.epochs.sets.get_items("Set4", get_keys=True) == ["E0", "E2"]


def test_delete_ticked_epochs_keeps_or_deletes_arrays(app, monkeypatch):
    browser, model = _edit_epochs(app)
    model.set_checked([1, 3], True)  # E1, E3
    asked = []

    def choose(title, **options):
        asked.append((title, options["message"], options["extras"]))
        return (None, None, "Delete")

    monkeypatch.setattr(app.series_actions, "_choose", choose)
    actions = _menu(monkeypatch, lambda: browser._show_chosen_context_menu(QtCore.QPoint(0, 0)))
    actions["Delete..."].trigger()
    title, message, extras = asked[0]
    assert title == "Remove Epochs" and message.startswith("Remove 2 epochs: E1, E3?")
    assert extras == [("Arrays", ["Keep", "Delete"], "Keep")]
    assert list(app.series.epochs.keys()) == ["E0", "E2"]
    assert "RecordA1" not in app.folder.data  # deleted, as chosen
    # The series stays in step: its channels no longer list them
    assert [d.name for d in app.series.channels["A"].data] == ["RecordA0", "RecordA2"]


def test_delete_one_epoch_keeping_its_arrays(app, monkeypatch):
    browser = app.browser_widget
    browser.show_data_of(app.series, kind="epochs")
    monkeypatch.setattr(app.series_actions, "_choose", lambda title, **o: (None, None, "Keep"))
    browser._delete_objects(app.series.epochs, [app.series.epochs["E0"]])
    assert "E0" not in app.series.epochs
    assert "RecordA0" in app.folder.data  # kept in the folder


def test_locked_series_epochs_are_not_deleted(app, monkeypatch):
    app.series.locked = True
    errors = []
    monkeypatch.setattr(app.series_actions, "_choose",
                        lambda title, **o: errors.append(o["validate"](None, None, "Keep")))
    app.browser_widget._delete_objects(app.series.epochs, [app.series.epochs["E0"]])
    assert "locked" in errors[0]
    assert "E0" in app.series.epochs


def test_delete_a_channel_through_the_series(app, monkeypatch):
    monkeypatch.setattr(app.series_actions, "_choose", lambda title, **o: (None, None, "Keep"))
    app.browser_widget._delete_objects(app.series.channels, [app.series.channels["B"]])
    assert list(app.series.channels.keys()) == ["A"]
    assert "RecordB0" in app.folder.data
