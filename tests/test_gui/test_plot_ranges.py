"""Tests for the chosen tool's X ranges on the Plot, with draggable edges (#436)."""

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6 import QtCore

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.plot_ranges import snap

pytestmark = pytest.mark.gui

DASH = QtCore.Qt.PenStyle.DashLine
SOLID = QtCore.Qt.PenStyle.SolidLine


def test_snap_to_the_nearest_sample():
    x = np.array([0.0, 0.5, 1.0, 1.5])
    assert snap(x, 0.7) == 0.5 and snap(x, 0.8) == 1.0
    assert snap(x, -3.0) == 0.0 and snap(x, 9.0) == 1.5
    assert snap(np.arange(10) * 0.1, 0.31) == 0.3  # not 0.30000000000000004
    assert snap(np.array([]), 2.2) == 2.2


@pytest.fixture
def app(qtbot):
    """Series Record: channel A x 2 epochs of 10 points (x = 0..9); Stats chosen."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=10)
    for d in series.channels["A"].data:
        d.nparray = np.arange(10.0)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.tool_rail.select_tool("Stats")
    window.stats = window.tool_tabs["Stats"]
    window.stats.measurement_combo.setCurrentText("mean")
    window.series = series
    return window


def _regions(app):
    return app.plot_widget.range_overlay.regions


def _styles(region):
    return [line.pen.style() for line in region.lines]


def _type(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _move(region, low, high):
    """Dragging, not yet released (setRegion would also finish the drag)."""
    for line, value in zip(region.lines, (low, high)):
        if line.value() != value:
            line.setValue(value)


def _drag(region, low, high):
    _move(region, low, high)
    region.sigRegionChangeFinished.emit(region)  # released


def test_ranges_shown_only_for_the_chosen_tool(app):
    assert list(_regions(app)) == ["window"]
    app.tool_rail.select_tool("Browser")
    assert _regions(app) == {}
    app.tool_rail.select_tool("Stats")
    assert list(_regions(app)) == ["window"]


def test_unbounded_edges_sit_at_the_trace_ends_dashed(app):
    region = _regions(app)["window"]
    assert region.getRegion() == (0.0, 9.0)
    assert _styles(region) == [DASH, DASH]


def test_typing_moves_the_edge(app):
    _type(app.stats.xbgn_edit, "2")
    region = _regions(app)["window"]
    assert region.getRegion() == (2.0, 9.0)
    assert _styles(region) == [SOLID, DASH]
    _type(app.stats.xend_edit, "6")
    assert region.getRegion() == (2.0, 6.0)


def test_dragging_updates_the_fields_then_saves(app):
    window = app.stats._selected_window()
    _type(app.stats.xbgn_edit, "2")
    region = _regions(app)["window"]
    _move(region, 3.4, 9.0)  # dragging the begin edge
    assert app.stats.xbgn_edit.text() == "3"  # snapped, shown live
    assert window.xbgn == 2.0  # saved only when released
    region.sigRegionChangeFinished.emit(region)
    assert window.xbgn == 3.0
    assert window.xend == math.inf  # the end edge wasn't dragged
    assert region.getRegion() == (3.0, 9.0)


def test_a_drag_is_one_command_when_released(app):
    history = app.selection_model.manager.command_history
    count = len(history.buffer)
    region = _regions(app)["window"]
    for value in (1.2, 2.4, 3.1):  # dragging
        _move(region, value, 9.0)
    assert len(history.buffer) == count
    region.sigRegionChangeFinished.emit(region)
    assert [e["command"] for e in history.buffer][count:] == [
        "nm.stats.windows['w0'].xbgn = 3.0"]


def test_dragging_the_middle_moves_both(app):
    window = app.stats._selected_window()
    _type(app.stats.xbgn_edit, "2")
    _type(app.stats.xend_edit, "5")
    _drag(_regions(app)["window"], 4.2, 7.2)
    assert (window.xbgn, window.xend) == (4.0, 7.0)
    assert _styles(_regions(app)["window"]) == [SOLID, SOLID]


def test_dragging_an_unbounded_edge_sets_it_and_clearing_returns_it(app):
    window = app.stats._selected_window()
    _drag(_regions(app)["window"], 0.0, 7.6)
    assert (window.xbgn, window.xend) == (-math.inf, 8.0)
    _type(app.stats.xend_edit, "")
    assert window.xend == math.inf
    assert _regions(app)["window"].getRegion() == (0.0, 9.0)
    assert _styles(_regions(app)["window"]) == [DASH, DASH]


def test_baseline_range_follows_bsln(app):
    _type(app.stats.xbgn_edit, "5")
    app.stats.bsln_checkbox.setChecked(True)
    assert list(_regions(app)) == ["window", "bsln"]
    # Its range is prefilled: trace start to the window's X begin
    assert _regions(app)["bsln"].getRegion() == (0.0, 5.0)
    _drag(_regions(app)["bsln"], 1.0, 5.0)
    window = app.stats._selected_window()
    assert (window.bsln_xbgn, window.bsln_xend) == (1.0, 5.0)
    assert app.stats.bsln_xbgn_edit.text() == "1"
    app.stats.bsln_checkbox.setChecked(False)
    assert list(_regions(app)) == ["window"]


def test_another_window_shows_its_range(app):
    _type(app.stats.xbgn_edit, "2")
    app.stats._add_window()
    assert _regions(app)["window"].getRegion() == (0.0, 9.0)  # w1: -inf..inf


def test_ranges_stay_when_stepping_traces(app):
    _type(app.stats.xbgn_edit, "2")
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert _regions(app)["window"].getRegion() == (2.0, 9.0)
    assert app.plot_widget.rendered_trace_count == 1


def test_tool_ranges_on_frozen_or_off(app):
    panel = app.plot_widget
    assert panel.range_mode_actions["on"].isChecked()  # on by default
    assert [a.text() for a in panel.ranges_menu.actions()] == ["On", "Frozen", "Off"]
    assert _regions(app)["window"].movable

    # Frozen: shown, but can't be dragged
    panel.range_mode_actions["frozen"].trigger()
    region = _regions(app)["window"]
    assert region.getRegion() == (0.0, 9.0)
    assert not region.movable and not any(line.movable for line in region.lines)
    _type(app.stats.xbgn_edit, "2")  # typing still moves it
    assert _regions(app)["window"].getRegion() == (2.0, 9.0)

    # Off: hidden, also on another trace or after choosing the tool again
    panel.range_mode_actions["off"].trigger()
    assert _regions(app) == {}
    assert not any(isinstance(item, type(region))
                   for item in panel.plot_widget.getPlotItem().items)
    app.selection_model.update(epoch=app.series.epochs["E1"])
    app.tool_rail.select_tool("Browser")
    app.tool_rail.select_tool("Stats")
    app.stats.bsln_checkbox.setChecked(True)
    assert _regions(app) == {}

    panel.range_mode_actions["on"].trigger()
    assert list(_regions(app)) == ["window", "bsln"]
    assert all(r.movable for r in _regions(app).values())


def test_edges_are_wide_enough_to_grab(app):
    region = _regions(app)["window"]
    for line in region.lines:
        assert line.pen.width() == 2
        assert line.hoverPen.width() == 4  # wider under the pointer
