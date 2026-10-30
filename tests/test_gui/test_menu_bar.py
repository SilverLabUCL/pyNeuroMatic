"""Tests for the menu bar's Edit and View menus (#425)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record (channel A, 2 epochs x 3 points), shown in the Table."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(2):
        folder.data.new(f"RecordA{epoch}", nparray=np.array([0.0, 1.0, 2.0]) + epoch)
    series = folder.sync_dataseries("Record", select=True)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(epoch_set="All")
    window.folder = folder
    return window


def _texts(menu):
    return [a.text() for a in menu.actions() if a.text()]


def _focus(monkeypatch, widget):
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", staticmethod(lambda: widget))


def test_menus(app):
    titles = [a.text() for a in app.menuBar().actions()]
    assert titles == ["File", "Folder", "Data", "Series", "Edit", "View"]  # no Tools
    assert _texts(app.edit_menu) == ["Copy", "Paste"]
    assert app.copy_action.shortcut() == QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy)
    assert app.paste_action.shortcut() == QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Paste)
    assert _texts(app.view_menu) == ["Browser", "Main", "Stats", "Spike",
                                     "Plot", "Table", "History"]
    assert app.tool_view_actions["Stats"].shortcut() == QtGui.QKeySequence("Ctrl+3")


def test_view_switches_the_tool_as_the_rail_does(app):
    actions = app.tool_view_actions
    assert actions["Browser"].isChecked()
    actions["Stats"].trigger()
    assert app.current_tool_name == "Stats" and app.tool_rail.current_tool_name == "Stats"
    assert app.tool_workspace.currentWidget() is app.stats_tab
    assert actions["Stats"].isChecked() and not actions["Browser"].isChecked()
    app.tool_rail.select_tool("Main")  # a rail click: the menu follows
    assert actions["Main"].isChecked() and not actions["Stats"].isChecked()


def test_view_switches_the_panel(app):
    actions = app.panel_view_actions
    actions["Table"].trigger()
    assert app.context_panel.currentWidget() is app.table_panel
    assert actions["Table"].isChecked() and not actions["Plot"].isChecked()
    app.context_panel.setCurrentWidget(app.plot_widget)  # a tab click: the menu follows
    assert actions["Plot"].isChecked() and not actions["Table"].isChecked()


def test_view_history_shows_and_hides_the_dock(app):
    app.show()
    action = app.history_view_action
    assert action.text() == "History" and action.isCheckable() and action.isChecked()
    action.trigger()
    assert not app.history_dock.isVisible() and not action.isChecked()
    action.trigger()
    assert app.history_dock.isVisible() and action.isChecked()


def test_edit_copy_and_paste_in_the_table(app, monkeypatch):
    view = app.table_panel.channel_table.view
    _focus(monkeypatch, view)
    app.table_panel.x_action.setChecked(False)
    view.clearSelection()
    view.selectionModel().setCurrentIndex(
        view.model().index(1, 1), QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
    app.copy_action.trigger()
    assert QtWidgets.QApplication.clipboard().text() == "1"
    QtWidgets.QApplication.clipboard().setText("5\t6")
    app.paste_action.trigger()
    assert app.folder.data["RecordA0"].nparray[1] == 5.0
    assert app.folder.data["RecordA1"].nparray[1] == 6.0


def test_edit_copy_and_paste_in_a_text_box(app, qtbot, monkeypatch):
    box = QtWidgets.QLineEdit("hello")
    qtbot.addWidget(box)
    _focus(monkeypatch, box)
    box.selectAll()
    app.copy_action.trigger()
    assert QtWidgets.QApplication.clipboard().text() == "hello"
    box.setText("")
    app.paste_action.trigger()
    assert box.text() == "hello"


def test_edit_items_are_greyed_out_when_there_is_nothing_to_do(app, monkeypatch):
    _focus(monkeypatch, None)  # e.g. the plot
    app.edit_menu.aboutToShow.emit()
    assert not app.copy_action.isEnabled() and not app.paste_action.isEnabled()
    app.edit_menu.aboutToHide.emit()  # enabled again, for ⌘C / ⌘V
    assert app.copy_action.isEnabled() and app.paste_action.isEnabled()
    # The history: copy, not paste (read only)
    app.history_panel.view.selectAll()
    _focus(monkeypatch, app.history_panel.view)
    QtWidgets.QApplication.clipboard().setText("x")
    app.edit_menu.aboutToShow.emit()
    assert app.copy_action.isEnabled() and not app.paste_action.isEnabled()
    # The table with Show Transformed on: copy, not paste
    view = app.table_panel.channel_table.view
    view.selectAll()
    _focus(monkeypatch, view)
    app.table_panel.channel_table.model.set_transform(object())
    app.edit_menu.aboutToShow.emit()
    assert app.copy_action.isEnabled() and not app.paste_action.isEnabled()


def test_about(app, monkeypatch):
    import pyneuromatic

    action = app.about_action
    assert action.text() == "About pyNeuroMatic"
    assert action.menuRole() == QtGui.QAction.MenuRole.AboutRole  # macOS: the app menu
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "about",
                        lambda parent, title, text: shown.append((title, text)))
    action.trigger()
    title, text = shown[0]
    assert title == "About pyNeuroMatic"
    assert f"pyNeuroMatic {pyneuromatic.__version__}" in text
    assert "https://github.com/SilverLabUCL/pyNeuroMatic" in text
    assert '<a href="https://silverlab.org/">The Silver Lab</a>' in text
