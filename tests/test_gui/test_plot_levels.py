"""Tests for the chosen tool's Y levels on the Plot, as draggable lines (#527)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.plot_levels import round_level

pytestmark = pytest.mark.gui


def test_round_level_to_the_axis_shown():
    assert round_level(-41.23678, 80.0) == -41.24
    assert round_level(0.0123456, 0.05) == 0.01235
    assert round_level(1234.5678, 2000.0) == 1235.0
    assert round_level(2.5, float("nan")) == 2.5  # no axis: as dragged
    assert round_level(2.5, 0.0) == 2.5


@pytest.fixture
def app(qtbot):
    """Series Record: channel A x 2 epochs of 10 points; Stats chosen, w0 a
    level+ at 4.5 (crossed between x = 4 and 5)."""
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
    tab = window.tool_tabs["Stats"]
    tab.set_measurement("level+")
    _type(tab.param_edits["ylevel"], "4.5")
    window.stats = tab
    window.manager = manager
    return window


def _lines(app):
    return app.plot_widget.level_overlay.lines


def _type(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _move(line, y):
    """Dragging, not yet released."""
    line.setValue(y)


def _drag(line, y):
    _move(line, y)
    line.sigPositionChangeFinished.emit(line)  # released


def _commands(app):
    return [e["command"] for e in app.manager.command_history.buffer]


def test_a_level_measure_shows_its_level(app):
    line = _lines(app)["level"]
    assert line.value() == 4.5 and line.movable
    assert "w0 Y level" in line.toolTip()
    # Typed: the line follows
    _type(app.stats.param_edits["ylevel"], "6")
    assert _lines(app)["level"].value() == 6.0
    # Not a level measure: no line
    app.stats.set_measurement("max")
    assert _lines(app) == {}
    app.stats.set_measurement("level-")
    assert _lines(app)["level"].value() == 6.0


def test_levels_shown_only_for_the_chosen_tool(app):
    assert list(_lines(app)) == ["level"]
    app.tool_rail.select_tool("Browser")
    assert _lines(app) == {}
    app.tool_rail.select_tool("Stats")
    assert list(_lines(app)) == ["level"]


def test_drag_shows_the_level_and_saves_it_once_when_released(app):
    line = _lines(app)["level"]
    count = len(_commands(app))
    _move(line, 3.21234)
    edit = app.stats.param_edits["ylevel"]
    assert float(edit.text()) == pytest.approx(3.21234, abs=0.01)  # rounded to the axis
    assert len(_commands(app)) == count  # not saved while dragging
    _drag(line, 3.21234)
    shown = float(edit.text())
    assert app.manager.stats.windows["w0"].func["ylevel"] == shown
    new = _commands(app)[count:]
    assert len(new) == 1 and "ylevel" in new[0]
    assert _lines(app)["level"].value() == shown


def test_frozen_and_off(app):
    app.plot_widget.set_range_mode("frozen")
    line = _lines(app)["level"]
    assert not line.movable and "drag" not in line.toolTip()
    app.plot_widget.set_range_mode("off")
    assert _lines(app) == {}
    app.plot_widget.set_range_mode("on")
    assert _lines(app)["level"].movable


def test_n_std_level_from_the_traces_baseline(app):
    """baseline mean + n std: drawn at the trace's level, and dragging sets n std."""
    tab = app.stats
    tab.bsln_checkbox.setChecked(True)
    tab.bsln_combo.setCurrentIndex(tab.bsln_combo.findData("mean+std"))
    _type(tab.bsln_xbgn_edit, "0")
    _type(tab.bsln_xend_edit, "3")
    tab.level_mode_combo.setCurrentIndex(tab.level_mode_combo.findData("n_std"))
    _type(tab.param_edits["n_std"], "2")
    window = app.manager.stats.windows["w0"]
    assert window.func.get("n_std") == 2
    bsln = next(r for r in tab._measure(window, app.plot_widget.shown_trace)
                if r.get("id") == "bsln")
    mean, std = bsln["s"], bsln["std"]  # as the core measures it
    line = _lines(app)["level"]
    assert line.value() == pytest.approx(mean + 2 * std)
    assert "mean + 2 std" in line.toolTip()
    # Dragged to mean + 1 std: n std = 1
    _drag(line, mean + std)
    assert window.func["n_std"] == pytest.approx(1.0, abs=0.01)
    assert float(tab.param_edits["n_std"].text()) == window.func["n_std"]
    # Dragged to the mean: n std 0 isn't allowed, so it stays
    before = window.func["n_std"]
    _drag(_lines(app)["level"], mean)
    assert window.func["n_std"] == before
    assert _lines(app)["level"].value() == pytest.approx(mean + before * std)


def test_a_tool_without_levels_draws_none(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.tool_rail.select_tool("Stats")
    assert _lines(window) == {}
