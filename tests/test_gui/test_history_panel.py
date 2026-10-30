"""Tests for the History panel: entries by kind, shown kinds, time, copy (#417)."""

import logging

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

import pyneuromatic.core.nm_history as nmh
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import history_panel
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Folder Demo with data RecordA0 = [1, 3], selected."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.array([1.0, 3.0]))
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.manager, window.folder = manager, folder
    return window


def _entries(app, kind=None):
    app.history_panel.flush()
    return [e.text for e in app.history_panel.model.entries if kind is None or e.kind == kind]


def test_kinds_of_entry(app):
    panel = app.history_panel
    assert [b.text() for b in panel.kind_buttons.values()] == [
        "Commands", "Messages", "Warnings", "Errors", "Selection", "Tool Runs"]
    app.folder.data.new("Extra", nparray=np.zeros(2))  # a command and messages
    nmh.alert("careful", path="NONE")
    nmh.error("broken", path="NONE")
    app.selection_model.update(data=app.folder.data["Extra"])
    assert "nm.folders[\"Demo\"].data.new('Extra')" in _entries(app, "command")
    assert any("new 'Extra'" in text for text in _entries(app, "message"))
    assert _entries(app, "warning") == ["careful"] and _entries(app, "error") == ["broken"]
    assert _entries(app, "selection")[-1].startswith("Selection: ")
    # Warnings and Errors say how many
    assert panel.kind_buttons["warning"].text() == "Warnings (1)"
    assert panel.kind_buttons["error"].text() == "Errors (1)"


def test_commands_are_marked_in_the_core_history(app):
    app.folder.data.new("Extra", nparray=np.zeros(2))
    buffer = app.manager.history.buffer
    commands = [e["message"] for e in buffer if e["kind"] == "command"]
    assert "nm.folders[\"Demo\"].data.new('Extra')" in commands
    assert all(e["kind"] == "" for e in buffer if e["message"].startswith("nm.Demo"))


def test_show_only_some_kinds(app):
    panel = app.history_panel
    app.folder.data.new("Extra", nparray=np.zeros(2))
    nmh.error("broken", path="NONE")
    panel.set_shown_kinds({"error"})  # only errors
    assert panel.text() == "broken"
    panel.set_shown_kinds({"command"})  # only commands
    assert panel.text().splitlines() == [
        line for line in _entries(app, "command")]
    panel.set_shown_kinds(history_panel.KINDS)
    assert len(panel.text().splitlines()) == len(_entries(app))


def test_time_stamps_on_and_off(app):
    panel = app.history_panel
    panel.set_shown_kinds({"error"})
    nmh.error("broken", path="NONE")
    assert panel.text() == "broken"
    panel.time_button.setChecked(True)
    stamp, text = panel.text().split("  ", 1)
    assert text == "broken" and len(stamp) == 8 and stamp.count(":") == 2  # HH:MM:SS
    panel.time_button.setChecked(False)
    assert panel.text() == "broken"


def test_tool_runs(app):
    tab = app.stats_tab
    tab.run_button.click()  # no measurement chosen: fails
    assert _entries(app, "error")[-1] == (
        "Stats: Failed: Choose a measurement in an enabled Stats window first")
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    folder = tab.stats_tool.toolfolder.name
    assert _entries(app, "tool")[-1] == f"Stats: Complete → {folder}"


def test_copy_and_clear(app):
    panel = app.history_panel
    panel.set_shown_kinds({"warning", "error"})
    nmh.alert("careful", path="NONE")
    nmh.error("broken", path="NONE")
    panel.copy_all_action.trigger()
    assert QtWidgets.QApplication.clipboard().text() == "careful\nbroken"
    panel.view.selectionModel().select(
        panel.filter.index(1, 0), QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
    panel.copy_action.trigger()
    assert QtWidgets.QApplication.clipboard().text() == "broken"
    # Clear asks first: No keeps the entries
    asked = []
    panel._confirm = lambda title, text: asked.append(text) or False
    panel.clear_action.trigger()
    count = len(panel.model.entries)  # all, shown or not (e.g. the welcome line)
    assert asked == [f"Remove all {count} entries from the History panel? This can't be undone."]
    assert panel.text() == "careful\nbroken"
    panel._confirm = lambda title, text: True
    panel.clear_action.trigger()
    assert panel.text() == "" and panel.kind_buttons["error"].text() == "Errors"
    asked.clear()
    panel._confirm = lambda title, text: asked.append(text) or True
    panel.clear_action.trigger()  # nothing to clear: not asked
    assert asked == []


def test_edit_copy_copies_the_selected_entries(app, monkeypatch):
    panel = app.history_panel
    panel.set_shown_kinds({"error"})
    nmh.error("broken", path="NONE")
    panel.flush()
    panel.view.selectAll()
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", staticmethod(lambda: panel.view))
    app.copy_action.trigger()
    assert QtWidgets.QApplication.clipboard().text() == "broken"


def test_keeps_the_most_recent_entries(app, monkeypatch):
    monkeypatch.setattr(history_panel, "MAX_ENTRIES", 3)
    panel = app.history_panel
    panel.clear()
    for word in ("one", "two"):
        nmh.error(word, path="NONE")
    panel.flush()
    for word in ("three", "four"):
        nmh.error(word, path="NONE")
    panel.set_shown_kinds({"error"})
    assert panel.text().splitlines() == ["two", "three", "four"]
    assert panel.kind_buttons["error"].text() == "Errors (3)"


def test_closing_the_window_stops_the_history_writing_to_it(qtbot):
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    handler = window.history_panel.handler
    assert handler in logging.getLogger("pyneuromatic").handlers
    window.close()
    assert handler not in logging.getLogger("pyneuromatic").handlers
