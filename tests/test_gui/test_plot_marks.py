"""Tests for the chosen tool's measurements drawn over the trace (#463)."""

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_transform import NMTransformInvert
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.plot_marks import PlotMark
from pyneuromatic.gui.stats_tab import result_marks

pytestmark = pytest.mark.gui

RED, GREEN = "red", "green"


def _marks(results, first=0.0, last=99.0):
    return [(m.kind, m.x, m.y, m.color, m.dashed)
            for m in result_marks(results, first, last, RED, GREEN)]


# --- what is drawn for each kind of result -------------------------------------

def test_a_level_is_a_line_across_its_range():
    results = [{"id": "main", "func": {"name": "mean"}, "xbgn": 10.0, "xend": 20.0, "s": -65.0}]
    assert _marks(results) == [("line", [10.0, 20.0], [-65.0, -65.0], RED, False)]


def test_a_point_found_is_a_marker():
    results = [{"id": "main", "func": {"name": "max"}, "xbgn": 10.0, "xend": 20.0,
                "s": -55.0, "x": 14.0}]
    assert _marks(results) == [("points", [14.0], [-55.0], RED, False)]


def test_a_level_crossing_is_a_marker_on_a_dashed_level():
    """e.g. FWHM: the max, then a crossing either side at half its height."""
    results = [
        {"id": "fwhm+", "func": {"name": "max"}, "xbgn": 60.0, "xend": 120.0,
         "s": -55.0, "x": 80.0},
        {"id": "fwhm+", "func": {"name": "level+", "ylevel": -62.5}, "xbgn": 60.0,
         "xend": 80.0, "x": 70.0},
        {"id": "fwhm+", "func": {"name": "level-", "ylevel": -62.5}, "xbgn": 80.0,
         "xend": 120.0, "x": 90.0},
    ]
    assert _marks(results) == [
        ("points", [80.0], [-55.0], RED, False),
        ("points", [70.0], [-62.5], RED, False),
        ("line", [60.0, 80.0], [-62.5, -62.5], RED, True),
        ("points", [90.0], [-62.5], RED, False),
        ("line", [80.0, 120.0], [-62.5, -62.5], RED, True),
    ]


def test_slope_value_and_baseline():
    results = [
        {"id": "bsln", "func": {"name": "mean"}, "xbgn": 0.0, "xend": 40.0, "s": -70.0},
        {"id": "main", "func": {"name": "slope"}, "xbgn": 10.0, "xend": 20.0, "s": 2.0, "b": 1.0},
        {"id": "main", "func": {"name": "value@xend"}, "xbgn": 10.0, "xend": 20.0, "s": 4.0},
    ]
    assert _marks(results) == [
        ("line", [0.0, 40.0], [-70.0, -70.0], GREEN, False),
        ("line", [10.0, 20.0], [21.0, 41.0], RED, False),
        ("points", [20.0], [4.0], RED, False),
    ]


def test_not_drawn_not_in_the_trace_units_nan_or_failed():
    results = [
        {"id": "main", "func": {"name": "std"}, "xbgn": 10.0, "xend": 20.0, "s": 2.0},
        {"id": "main", "func": {"name": "area"}, "xbgn": 10.0, "xend": 20.0, "s": 300.0},
        {"id": "main", "func": {"name": "mean"}, "xbgn": 10.0, "xend": 20.0, "s": math.nan},
        {"id": "main", "func": {"name": "max"}, "xbgn": 10.0, "xend": 20.0},  # none found
    ]
    assert _marks(results) == []


def test_an_unbounded_range_is_drawn_to_the_trace_ends():
    results = [{"id": "main", "func": {"name": "mean"}, "xbgn": -math.inf,
                "xend": math.inf, "s": 1.0}]
    assert _marks(results, first=5.0, last=50.0)[0][1] == [5.0, 50.0]


def test_plot_mark_kinds():
    with pytest.raises(ValueError):
        PlotMark("curve", [0.0], [0.0], RED)


# --- in the app ------------------------------------------------------------------

PEAK_X = (80.0, 90.0)  # epoch 0 and 1


@pytest.fixture
def app(qtbot):
    """Channel A x 2 epochs: a peak of 15 above -70 (at x 80, then 90); Stats
    chosen, its window max over 60..120."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=200)
    x = np.arange(200.0)
    for d, peak in zip(series.channels["A"].data, PEAK_X):
        d.nparray = -70.0 + 15.0 * np.exp(-((x - peak) / 12.0) ** 2)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.tool_rail.select_tool("Stats")
    stats = window.tool_tabs["Stats"]
    stats.measurement_combo.setCurrentText("max")
    for edit, text in ((stats.xbgn_edit, "60"), (stats.xend_edit, "120")):
        edit.setText(text)
        edit.editingFinished.emit()
    window.stats, window.series = stats, series
    return window


def _shown(app):
    overlay = app.plot_widget.mark_overlay
    return [(m.kind, m.x, m.y) for m in overlay.marks]


def test_the_window_measurement_is_drawn(app):
    assert _shown(app) == [("points", [80.0], [-55.0])]
    assert len(app.plot_widget.mark_overlay.items) == 1


def test_shown_only_for_the_chosen_tool(app):
    app.tool_rail.select_tool("Browser")
    assert _shown(app) == []
    app.tool_rail.select_tool("Stats")
    assert _shown(app) == [("points", [80.0], [-55.0])]


def test_follows_the_trace(app):
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert _shown(app) == [("points", [90.0], [-55.0])]


def test_follows_the_function_range_and_baseline(app):
    app.stats.measurement_combo.setCurrentText("mean")
    ((kind, x, y),) = _shown(app)
    assert kind == "line" and x == [60.0, 120.0] and y[0] == y[1] > -70.0
    app.stats.xend_edit.setText("70")
    app.stats.xend_edit.editingFinished.emit()
    assert _shown(app)[0][1] == [60.0, 70.0]
    app.stats.bsln_checkbox.setChecked(True)  # prefilled 0..60
    colors = [m.color for m in app.plot_widget.mark_overlay.marks]
    assert colors == [app.stats.BSLN_MARK_COLOR, app.stats.MARK_COLOR]


def test_measures_the_trace_as_plotted(app):
    app.series.channels["A"].transform = [NMTransformInvert()]
    app._on_transforms_changed()
    app.stats.measurement_combo.setCurrentText("min")
    assert _shown(app) == [("points", [80.0], [55.0])]  # the inverted peak


def test_drawing_writes_no_command_and_keeps_the_results(app):
    history = app.selection_model.manager.command_history
    window = app.stats._selected_window()
    count, results = len(history.buffer), list(window.results)
    app.selection_model.update(epoch=app.series.epochs["E1"])  # measured again
    commands = [e["command"] for e in history.buffer][count:]
    assert [c for c in commands if not c.startswith("nm.select_keys")] == []
    count = len(history.buffer)
    app.plot_widget.mark_overlay.redraw()
    assert len(history.buffer) == count
    assert window.results == results


def test_tool_measures_off_and_on(app):
    action = app.plot_widget.measures_action
    assert action.isChecked()
    action.setChecked(False)
    assert _shown(app) == []
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert _shown(app) == []
    action.setChecked(True)
    assert _shown(app) == [("points", [90.0], [-55.0])]


def test_a_measurement_that_fails_draws_nothing(app):
    for edit, text in ((app.stats.xend_edit, "600"), (app.stats.xbgn_edit, "500")):
        edit.setText(text)  # outside the trace (x 0..199)
        edit.editingFinished.emit()
    assert app.stats._selected_window().xbgn == 500.0
    assert _shown(app) == []


def test_the_preview_writes_nothing_to_the_history(app):
    """Changing a setting redraws the measurement: no "compute" notes."""
    history = app.selection_model.manager.history
    count = len(history.buffer)
    app.stats.bsln_checkbox.setChecked(True)
    for edit, text in ((app.stats.bsln_xbgn_edit, "0.5"), (app.stats.bsln_xend_edit, "2.6")):
        edit.setText(text)
        edit.editingFinished.emit()
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert app.plot_widget.mark_overlay.marks  # measured
    messages = [e["message"] for e in history.buffer[count:]]
    assert not any("compute" in m for m in messages)
