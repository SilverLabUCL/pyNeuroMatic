"""Tests for a tool run's progress bar and Cancel (#450)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtTest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.selection_model import ALL_EPOCHS, ALL_SETS

pytestmark = pytest.mark.gui

N = 6


@pytest.fixture
def app(qtbot):
    """Channel A x 6 epochs (y = epoch number); Set0 E0-E2, Set1 E3-E5;
    Stats mean, on All epochs."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=N, n_points=5)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = np.full(5, float(i))
    series.epochs.sets.add("Set0", ["E0", "E1", "E2"])
    series.epochs.sets.add("Set1", ["E3", "E4", "E5"])
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(epoch_set=ALL_EPOCHS)
    tab = window.tool_tabs["Stats"]
    tab.output_arrays_checkbox.setChecked(True)
    tab.measurement_combo.setCurrentText("mean")
    window.tool_rail.select_tool("Stats")
    window.stats, window.manager = tab, manager
    return window


def _show_at_once(tab):
    tab.SHOW_PROGRESS_AFTER = 0.0
    tab.PROGRESS_INTERVAL = 0.0


def _means(app):
    values = app.stats.stats_tool.toolfolder.data.get("ST_w0_mean_y").nparray
    return list(values[~np.isnan(values)])


def test_a_quick_run_shows_no_progress(app):
    shown = []
    app.stats.progress_changed.connect(lambda done, total: shown.append(done))
    app.stats.run_button.click()
    assert shown == []  # under half a second
    assert app.stats.status_label.text() == "Complete"
    assert _means(app) == [float(i) for i in range(N)]


def test_progress_while_running(app):
    tab = app.stats
    _show_at_once(tab)
    seen = []

    def during(done, total):
        seen.append((done, total, tab.run_button.text(), tab.run_info_label.text(),
                     tab.progress_bar.isVisible() or not tab.isVisible()))

    tab.progress_changed.connect(during)
    info = tab.run_info_label.text()
    tab.run_button.click()
    assert [s[:2] for s in seen] == [(i, N) for i in range(1, N + 1)]
    assert all(s[2] == "Cancel" for s in seen)  # Run becomes Cancel
    assert seen[1][3] == f"2 / {N}"
    # Afterwards: Run again, no bar, the label as it was
    assert tab.run_button.text() == "Run"
    assert not tab.progress_bar.isVisible()
    assert tab.run_info_label.text() == info


def test_stop_keeps_what_was_done(app):
    tab = app.stats
    _show_at_once(tab)
    tab.progress_changed.connect(lambda done, total: done == 2 and tab.run_button.click())
    history = app.manager.command_history
    count = len(history.buffer)
    tab.run_button.click()
    assert tab.status_label.text() == f"Cancelled after 2 of {N}"
    assert _means(app) == [0.0, 1.0]
    folder = tab.stats_tool.toolfolder
    assert "cancelled after 2 of 6: partial results" in [n["note"] for n in folder.notes]
    commands = [e["command"] for e in history.buffer][count:]
    assert sum(c.startswith("nm.stats.run_selection(") for c in commands) == 1  # kept
    assert f"Stats: Cancelled after 2 of {N}" in app.history_panel.text()
    assert tab.run_button.text() == "Run" and not tab.running


def test_esc_stops(app):
    tab = app.stats
    _show_at_once(tab)
    tab.progress_changed.connect(
        lambda done, total: done == 3 and QtTest.QTest.keyClick(app, QtCore.Qt.Key.Key_Escape))
    tab.run_button.click()
    assert tab.status_label.text() == f"Cancelled after 3 of {N}"


def test_the_rest_of_the_window_is_blocked(app):
    tab = app.stats
    _show_at_once(tab)
    on = tab.window_enabled.isChecked()

    def during(done, total):
        if done == 2:
            QtTest.QTest.mouseClick(tab.window_enabled, QtCore.Qt.MouseButton.LeftButton)
            QtTest.QTest.mouseClick(app.tool_rail._buttons["Browser"],
                                    QtCore.Qt.MouseButton.LeftButton)

    tab.progress_changed.connect(during)
    tab.run_button.click()
    assert tab.window_enabled.isChecked() == on  # On didn't change
    assert app.current_tool_name == "Stats"  # the rail didn't change
    assert tab.status_label.text() == "Complete"
    # ... and works again afterwards
    QtTest.QTest.mouseClick(app.tool_rail._buttons["Browser"], QtCore.Qt.MouseButton.LeftButton)
    assert app.current_tool_name == "Browser"


def test_all_sets_count_together_and_stop_ends_them(app):
    tab = app.stats
    _show_at_once(tab)
    app.selection_model.update(epoch_set=ALL_SETS)
    seen = []
    tab.progress_changed.connect(lambda done, total: seen.append((done, total)))
    tab.run_button.click()
    assert seen == [(i, N) for i in range(1, N + 1)]  # Set0 then Set1, as one count
    # Stop in Set0: Set1 isn't run
    seen.clear()
    tab.progress_changed.connect(lambda done, total: done == 2 and tab.stop())
    history = app.manager.command_history
    count = len(history.buffer)
    tab.run_button.click()
    assert tab.status_label.text() == f"Cancelled after 2 of {N}"
    commands = [e["command"] for e in history.buffer][count:]
    assert sum(c.startswith("nm.stats.run_selection(") for c in commands) == 1  # Set0 only


def test_run_is_green_and_cancel_red(app):
    tab = app.stats
    _show_at_once(tab)
    assert tab.run_button.text() == "Run" and tab.RUN_COLOR in tab.run_button.styleSheet()
    during = []
    tab.progress_changed.connect(
        lambda done, total: during.append((tab.run_button.text(), tab.run_button.styleSheet())))
    tab.run_button.click()
    assert during[0][0] == "Cancel" and tab.CANCEL_COLOR in during[0][1]
    assert tab.run_button.text() == "Run" and tab.RUN_COLOR in tab.run_button.styleSheet()
