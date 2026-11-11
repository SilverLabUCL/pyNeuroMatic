"""Tests for Stats1's Details: the current window's measurement as text (#471)."""

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.results_view import StatsTables
from pyneuromatic.gui.stats_tab import result_lines

pytestmark = pytest.mark.gui

BSLN = {"id": "bsln", "func": {"name": "mean"}, "xbgn": 0.0, "xend": 40.0,
        "s": -70.0, "sunits": "mV"}


# --- the text of each kind of result -------------------------------------------

def test_a_point_found_with_baseline_and_difference():
    max_ = {"id": "main", "func": {"name": "max"}, "s": -55.0, "sunits": "mV",
            "x": 80.0, "xunits": "ms", "Δs": 15.0}
    assert result_lines([BSLN, max_]) == [
        "bsln mean = -70 mV  (x 0 to 40)",
        "max = -55 mV at 80 ms  (Δ = 15 mV)",
    ]


def test_a_pipeline_a_line_per_step_then_its_measure():
    steps = [
        BSLN,
        {"id": "risetime+", "func": {"name": "max"}, "s": -55.0, "sunits": "mV",
         "x": 80.0, "xunits": "ms", "Δs": 15.0},
        {"id": "risetime+", "p0": 10.0, "func": {"name": "level+", "ylevel": -68.5,
                                                  "yunits": "mV"}, "x": 61.774, "xunits": "ms"},
        {"id": "risetime+", "p1": 90.0, "func": {"name": "level+", "ylevel": -56.5,
                                                  "yunits": "mV"}, "x": 76.116, "xunits": "ms",
         "dx": 14.342},
    ]
    assert result_lines(steps)[1:] == [
        "max = -55 mV at 80 ms  (Δ = 15 mV)",
        "level+ 10% = -68.5 mV at 61.774 ms",
        "level+ 90% = -56.5 mV at 76.116 ms",
        "risetime+ = 14.342 ms",
    ]
    assert result_lines(steps, digits=3)[-1] == "risetime+ = 14.3 ms"
    # The window's X range: on the measurement's result line, not each step's (#498)
    assert result_lines(steps, window_x=(60, 120))[1:] == [
        "max = -55 mV at 80 ms  (Δ = 15 mV)",
        "level+ 10% = -68.5 mV at 61.774 ms",
        "level+ 90% = -56.5 mV at 76.116 ms",
        "risetime+ = 14.342 ms  (x 60 to 120)",
    ]


def test_other_results():
    slope = {"id": "main", "func": {"name": "slope"}, "s": -0.164, "sunits": "mV/ms",
             "b": -50.022, "bunits": "mV", "Δs": 69.8}
    spread = {"id": "main", "func": {"name": "mean+std"}, "s": -64.811, "sunits": "mV",
              "std": 5.341}
    missing = {"id": "main", "func": {"name": "max"}, "Δs": math.nan}
    nan = {"id": "main", "func": {"name": "mean"}, "s": math.nan, "sunits": "mV"}
    assert result_lines([BSLN, slope, spread, missing, nan])[1:] == [
        "slope = -0.164 mV/ms, intercept -50.022 mV",  # no Δ: not the baseline's units
        "mean = -64.811 mV, ±std 5.341 mV",  # mean+std: its value is the mean
        "max: no value",
        "mean = NaN mV",
    ]


def test_the_number_of_points():  # #498
    from pyneuromatic.gui.results_view import points_text
    assert points_text(15) == "n = 15"
    assert points_text(15, 1) == "n = 15, 1 NaN"
    assert points_text(15, 2) == "n = 15, 2 NaNs"
    bsln = dict(BSLN, n=41, nans=0)
    mean = {"id": "main", "func": {"name": "mean+std"}, "s": -64.8, "sunits": "mV",
            "std": 5.3, "Δs": 5.2, "n": 15, "nans": 2}
    max_ = {"id": "main", "func": {"name": "max"}, "s": -55.0, "sunits": "mV",
            "x": 80.0, "xunits": "ms", "n": 61}
    assert result_lines([bsln, mean, max_]) == [
        "bsln mean = -70 mV  (x 0 to 40, n = 41)",
        "mean = -64.8 mV, ±std 5.3 mV  (Δ = 5.2 mV)  (n = 15, 2 NaNs)",
        "max = -55 mV at 80 ms",  # a point found: no n
    ]
    # mean@max: the points averaged around the peak
    mean_at_max = {"id": "main", "func": {"name": "mean@max", "n_mean": 5}, "s": -55.2,
                   "sunits": "mV", "x": 80.0, "xunits": "ms", "n": 5, "nans": 0}
    assert result_lines([mean_at_max], window_x=(60, 120)) == [
        "mean@max = -55.2 mV at 80 ms  (x 60 to 120, n = 5)"]


# --- in the app ------------------------------------------------------------------

@pytest.fixture
def app(qtbot):
    """Channel A x 2 epochs: a peak of 15 above -70, at x 80 then 90."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=200)
    x = np.arange(200.0)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = -70.0 + 15.0 * np.exp(-((x - 80 - 10 * i) / 12.0) ** 2)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.stats, window.manager, window.series = window.tool_tabs["Stats"], manager, series
    return window


def _type(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _lines(app):
    return app.stats.details.toPlainText().splitlines()


def test_details_follow_the_window_and_the_trace(app):
    tab = app.stats
    assert _lines(app) == ["RecordA0  w0", "Choose a measurement"]
    tab.measurement_combo.setCurrentText("max")
    # Its X range on the measurement's line, as the baseline's (#498): here
    # the trace's (-inf to inf)
    assert _lines(app) == ["RecordA0  w0 max", "max = -55 at 80  (x 0 to 199)"]
    _type(tab.xbgn_edit, "60")
    _type(tab.xend_edit, "120")
    assert _lines(app) == ["RecordA0  w0 max", "max = -55 at 80  (x 60 to 120)"]
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert _lines(app) == ["RecordA1  w0 max", "max = -55 at 90  (x 60 to 120)"]
    tab.bsln_checkbox.setChecked(True)
    _type(tab.bsln_xbgn_edit, "0")
    _type(tab.bsln_xend_edit, "40")
    assert _lines(app) == ["RecordA1  w0 max",
                           "bsln mean = -70  (x 0 to 40, n = 41)",
                           "max = -55 at 90  (Δ = 15)  (x 60 to 120)"]  # a point: no n
    tab.window_enabled.setChecked(False)
    assert _lines(app) == ["RecordA1  w0 max", "Window off"]


def test_details_with_tool_measures_off_and_another_tool(app):
    tab = app.stats
    tab.measurement_combo.setCurrentText("max")
    app.plot_widget.measures_action.setChecked(False)
    app.tool_rail.select_tool("Browser")
    app.selection_model.update(epoch=app.series.epochs["E1"])
    assert _lines(app)[1] == "max = -55 at 90  (x 0 to 199)"


def test_a_measurement_that_fails_says_so(app):
    tab = app.stats
    tab.measurement_combo.setCurrentText("max")
    _type(tab.xend_edit, "600")
    _type(tab.xbgn_edit, "500")  # outside the trace (x 0..199)
    assert _lines(app)[1] == "max: no value  (x 500 to 600)"


def test_details_write_no_commands_or_notes(app):
    tab = app.stats
    tab.measurement_combo.setCurrentText("max")
    commands = len(app.manager.command_history.buffer)
    notes = len(app.manager.history.buffer)
    tab._show_details()
    app.plot_widget.trace_drawn.emit(app.plot_widget.shown_trace)
    assert len(app.manager.command_history.buffer) == commands
    assert len(app.manager.history.buffer) == notes


def test_details_are_selectable_and_short(app):
    tab = app.stats
    assert tab.details.isReadOnly()
    assert tab.details.height() < 10 * tab.details.fontMetrics().lineSpacing()


def test_digits_shown_in_settings(app, qtbot):
    """Settings › Stats › Digits displayed (the last option): Stats1's Details and Stats2's tables
    and Details; saved with the settings."""
    tab = app.stats
    tab.measurement_combo.setCurrentText("mean")
    _type(tab.xbgn_edit, "60")
    _type(tab.xend_edit, "100")
    mean = _lines(app)[1]
    assert mean.startswith("mean = -6") and len(mean.split()[2]) == len("-61.2345")  # 6 digits
    dialog = app.settings_dialog()
    qtbot.addWidget(dialog)
    spin = dialog.tool_spinboxes["Stats"]["digits"]
    assert (spin.value(), spin.minimum(), spin.maximum()) == (6, 2, 10)
    spin.setValue(3)
    assert app.manager.settings.stats_digits == 3
    assert _lines(app)[1] == f"mean = {float(mean.split()[2]):.3g}  (x 60 to 100, n = 41)"
    # Stats2: its tables, after a run
    tab.run_button.click()
    tables = tab.stats2.tables
    assert StatsTables.digits == 3
    text = tables.arrays_table.item(0, 0).text()
    assert len(text.lstrip("-").replace(".", "")) <= 3


def test_index_and_count_arrays_in_settings(app, qtbot):
    """Settings › Stats › Show index and count ST_ arrays, before Digits displayed,
    off by default: Stats2's tables and Array list; saved with the settings;
    no All columns checkbox in Stats2 (#496)."""
    tab = app.stats
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    stats2 = tab.stats2
    assert not hasattr(stats2, "all_columns_checkbox")
    keys = [option[0] for option in tab.config_options()]
    assert keys[-2:] == ["detail_arrays", "digits"]  # Digits displayed last
    dialog = app.settings_dialog()
    qtbot.addWidget(dialog)
    box = dialog.tool_checkboxes["Stats"]["detail_arrays"]
    assert box.text() == "Show index and count ST_ arrays" and not box.isChecked()
    names = [stats2.array_combo.itemText(i) for i in range(stats2.array_combo.count())]
    assert "ST_w0_mean_y" in names
    assert not [n for n in names if n.rsplit("_", 1)[-1] in ("i", "n", "nans") or
                n == "ST_epoch_num"]
    box.setChecked(True)
    assert app.manager.settings.stats_detail_arrays is True
    names = [stats2.array_combo.itemText(i) for i in range(stats2.array_combo.count())]
    assert "ST_epoch_num" in names and "ST_w0_mean_n" in names
    assert "ST_epoch_num" in stats2.tables.columns
