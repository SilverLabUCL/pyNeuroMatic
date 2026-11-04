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
        "Commands", "Messages", "Warnings", "Errors", "Selection", "Tool Runs", "Inner Calls"]
    app.folder.data.new("Extra", nparray=np.zeros(2))  # a command (no message: it says it)
    app.folder.new_dataseries("Rec", n_points=2)  # a command and a message
    nmh.alert("careful", path="NONE")
    nmh.error("broken", path="NONE")
    app.selection_model.update(data=app.folder.data["Extra"])
    assert "nm.folders['Demo'].data.new('Extra', n_points=2, fill=0.0)" in _entries(app, "command")
    assert _entries(app, "message")[-1] == "Demo: new data series Rec: channels A; epochs E0..E0"
    assert _entries(app, "warning") == ["careful"] and _entries(app, "error") == ["broken"]
    assert _entries(app, "selection")[-1].startswith("Selection: ")
    # Warnings and Errors say how many
    assert panel.kind_buttons["warning"].text() == "Warnings (1)"
    assert panel.kind_buttons["error"].text() == "Errors (1)"


def test_commands_are_marked_in_the_core_history(app):
    app.folder.data.new("Extra", nparray=np.zeros(2))
    buffer = app.manager.history.buffer
    commands = [e["message"] for e in buffer if e["kind"] == "command"]
    assert "nm.folders['Demo'].data.new('Extra', n_points=2, fill=0.0)" in commands
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
    tab = app.tool_tabs["Stats"]
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


def test_inner_calls(app):
    """Calls made inside a command (#437): off by default, not kept while off."""
    panel = app.history_panel
    assert not panel.kind_buttons["inner"].isChecked()
    folder = app.folder
    folder.new_dataseries("Rec", n_channels=1, n_epochs=3, n_points=2)
    assert _entries(app, "inner") == []  # not kept while hidden
    commands = _entries(app, "command")
    assert commands[-1].startswith("nm.folders['Demo'].new_dataseries('Rec'")
    assert not any("data.new('RecA0')" in c for c in commands)  # nested: not a command
    panel.kind_buttons["inner"].setChecked(True)
    folder.new_dataseries("Two", n_channels=1, n_epochs=2, n_points=2)
    inner = _entries(app, "inner")
    assert "nm.folders['Demo'].data.new('TwoA0')" in inner
    panel.set_shown_kinds({"inner"})
    assert panel.text().splitlines()[0].startswith("    ")  # indented, under its command


def test_a_group_action_is_one_command(app, monkeypatch):
    """Add Epochs to Group: one command for the range, not one per epoch."""
    folder = app.folder
    series = folder.new_dataseries("Rec", n_channels=1, n_epochs=20, n_points=2)
    app.selection_model.update(dataseries=series, epoch=series.epochs["E0"])
    actions = app.group_actions
    monkeypatch.setattr(actions, "_choose",
                        lambda title, **options: (None, None, "0", "E0", "E19", "1"))
    before = len(_entries(app, "command"))
    actions.assign_epochs()
    commands = _entries(app, "command")[before:]
    assert commands == [
        "nm.folders['Demo'].dataseries['Rec'].epochs.groups.assign("
        "[f'E{i}' for i in range(20)], 0)"]  # a range, not 20 names


def test_clear_group_is_one_command(app, monkeypatch):
    """Clear Group: clear_group(n), with no add_group to keep the group (#437)."""
    series = app.folder.new_dataseries("Rec", n_channels=1, n_epochs=8, n_points=2)
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), 2)
    app.selection_model.update(dataseries=series, epoch=series.epochs["E0"], group=0)
    actions = app.group_actions
    monkeypatch.setattr(actions, "_choose_groups", lambda title, describe: [0])
    before = len(_entries(app, "command"))
    actions.clear_group()
    assert _entries(app, "command")[before:] == [
        "nm.folders['Demo'].dataseries['Rec'].epochs.groups.clear_group(0)"]
    assert series.epochs.groups.group_numbers == [0, 1]  # kept, empty
    assert series.epochs.groups.get_items(0) == []


def test_history_starts_with_the_manager(qtbot):
    """The first command makes the manager: "nm = NMManager()" (#437)."""
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.history_panel.flush()
    commands = [e.text for e in window.history_panel.model.entries if e.kind == "command"]
    assert commands[0] == "nm = NMManager()"
    assert manager.command_history.buffer[0]["command"] == "nm = NMManager()"
    # Recording already on (NMManager made it): not added again
    manager = NMManager(quiet=True, command_history=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    assert [c["command"] for c in manager.command_history.buffer].count("nm = NMManager()") == 1


def test_an_exported_script_runs(qtbot, tmp_path):
    """The command history, saved as a script, re-creates the session (made
    after the window opens, so all of it is recorded)."""
    manager = NMManager(quiet=True)
    app = NMAppWindow(manager)
    qtbot.addWidget(app)
    app.manager = manager
    folder = manager.folders.new("Demo")
    series = folder.new_dataseries("Rec", n_channels=2, n_epochs=6, n_points=3, fill=1.0)
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), 2)
    series.epochs.sets.add("Set0", ["E0", "E1"])
    app.selection_model.update(dataseries=series, epoch=series.epochs["E2"])
    path = tmp_path / "session.py"
    app.manager.command_history.to_script(str(path))
    namespace = {}
    exec(path.read_text(), namespace)
    replayed = namespace["nm"]
    again = replayed.folders["Demo"]
    assert sorted(again.dataseries["Rec"].epochs.keys()) == sorted(series.epochs.keys())
    assert again.dataseries["Rec"].epochs.groups == series.epochs.groups
    assert again.dataseries["Rec"].epochs.sets.get_items("Set0", get_keys=True) == ["E0", "E1"]


def test_selection_says_when_and_or_is_not_chosen(app):
    """Epoch Set and Group without AND/OR: "?" and a hint, not "+" (#441)."""
    series = app.folder.new_dataseries("Rec", n_epochs=4, n_points=2)
    series.epochs.sets.add("Set0", ["E0", "E1"])
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), 2)
    app.selection_model.update(dataseries=series, epoch_set="Set0", group=0)
    assert _entries(app, "selection")[-1] == (
        "Selection: Demo / Rec / A / Epoch Set0 ? Group 0 (choose AND or OR)")
    app.selection_model.update(group_operator="AND")
    assert _entries(app, "selection")[-1] == "Selection: Demo / Rec / A / Epoch Set0 AND Group 0"


def test_select_commands_are_not_shown(app):
    """A click: the Selection entry, not also "nm.select_keys = ..." (kept in
    the command history, for scripts) (#441)."""
    series = app.folder.new_dataseries("Rec", n_epochs=4, n_points=2)
    for epoch in ("E1", "E2"):
        app.selection_model.update(dataseries=series, epoch=series.epochs[epoch])
    assert not any("select_keys" in text for text in _entries(app, "command"))
    assert _entries(app, "selection")[-2:] == [
        "Selection: Demo / Rec / A / E1", "Selection: Demo / Rec / A / E2"]
    commands = [c["command"] for c in app.manager.command_history.buffer]
    assert commands[-1] == "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Rec', 'epoch': 'E2'}"
    assert sum("select_keys" in c for c in commands) == 1  # E1 replaced by E2


def test_edit_scales_is_one_command(app, monkeypatch):
    """Data › Edit Scales: one set_scales command, no message per value (#441)."""
    data = app.folder.data["RecordA0"]
    actions = app.data_actions
    monkeypatch.setattr(actions, "_ask_scales", lambda *a, **k: {
        "xstart": -1.0, "xdelta": 0.1, "xlabel": "Time", "xunits": "ms",
        "y": {"Y": {"label": "Voltage", "units": "mV"}}})
    before = len(_entries(app))
    actions.edit_scales(data)
    assert _entries(app)[before:] == [
        "nm.folders['Demo'].data['RecordA0'].set_scales(xstart=-1.0, xdelta=0.1, "
        "xlabel='Time', xunits='ms', ylabel='Voltage', yunits='mV')"]
    assert (data.xscale.start, data.yscale.units) == (-1.0, "mV")
