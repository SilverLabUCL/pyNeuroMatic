"""Spike1: the Spike tool's detection tab (#528)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.spike_tab import SpikeToolTab

pytestmark = pytest.mark.gui

# Each epoch: 100 points (x = 0..99 ms), pulses to 1 at x = 10, 40 (and 70 in E1)
PULSES = {0: (10, 40), 1: (10, 40, 70), 2: (40,)}


@pytest.fixture
def app(qtbot):
    """Series Record: channel A x 3 epochs; Spike chosen, threshold 0.5."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=3, n_points=100)
    for number, data in enumerate(series.channels["A"].data):
        y = np.zeros(100)
        for x in PULSES[number]:
            y[x:x + 5] = 1.0
        data.nparray = y
        data.xscale.units = "ms"
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.tool_rail.select_tool("Spike")
    tab = window.tool_tabs["Spike"]
    _type(tab.ylevel_edit, "0.5")
    window.spike = tab
    window.manager = manager
    return window


def _type(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _commands(app):
    return [e["command"] for e in app.manager.command_history.buffer]


def _overlays(app):
    plot = app.plot_widget
    return plot.range_overlay.regions, plot.level_overlay.lines, plot.mark_overlay.marks


def test_the_tab_and_its_defaults(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Spike"]
    assert isinstance(tab, SpikeToolTab)
    assert tab.mode_tabs.tabText(0) == "Spike1"
    assert tab.ylevel_edit.text() == "0"
    assert tab.crossing_combo.currentText() == "rising"
    assert (tab.xbgn_edit.text(), tab.xend_edit.text()) == ("-inf", "inf")
    assert [tab.crossing_combo.itemText(i) for i in range(3)] == ["rising", "falling", "any"]


def test_typed_values_are_saved_one_command_each(app):
    tab, tool = app.spike, app.spike.spike_tool
    assert tool.ylevel == 0.5
    count = len(_commands(app))
    _type(tab.xbgn_edit, "5")
    tab.crossing_combo.setCurrentIndex(tab.crossing_combo.findData("level-"))
    assert (tool.xbgn, tool.func_name) == (5.0, "level-")
    assert _commands(app)[count:] == ["nm.spike.xbgn = 5.0", "nm.spike.func_name = 'level-'"]
    _type(tab.xend_edit, "5")  # not after X begin: a warning, not saved
    assert "X begin must be less than X end" in tab.status_label.text()
    assert tool.xend == float("inf")
    _type(tab.ylevel_edit, "high")
    assert "Threshold must be a number" in tab.status_label.text()
    assert tool.ylevel == 0.5


def test_plot_shows_range_threshold_and_crossings(app):
    regions, lines, marks = _overlays(app)
    assert list(regions) == ["range"] and list(lines) == ["threshold"]
    assert lines["threshold"].value() == 0.5
    assert marks[0].x == pytest.approx([9.5, 39.5]) and marks[0].y == [0.5, 0.5]
    # Another tool: none of them
    app.tool_rail.select_tool("Stats")
    regions, lines, _marks = _overlays(app)
    assert "range" not in regions and "threshold" not in lines


def test_crossings_follow_the_settings(app):
    tab = app.spike
    _type(tab.xbgn_edit, "20")
    assert _overlays(app)[2][0].x == pytest.approx([39.5])
    tab.crossing_combo.setCurrentIndex(tab.crossing_combo.findData("level"))
    assert _overlays(app)[2][0].x == pytest.approx([39.5, 44.5])
    _type(tab.ylevel_edit, "2")  # above the pulses: none
    assert _overlays(app)[2] == []


def test_drag_threshold_and_range(app):
    tab, tool = app.spike, app.spike.spike_tool
    _regions, lines, _marks = _overlays(app)
    count = len(_commands(app))
    line = lines["threshold"]
    line.setValue(0.25)
    assert float(tab.ylevel_edit.text()) == pytest.approx(0.25, abs=0.01)
    assert len(_commands(app)) == count  # not saved while dragging
    line.sigPositionChangeFinished.emit(line)
    assert tool.ylevel == float(tab.ylevel_edit.text())
    assert len(_commands(app)) == count + 1
    region = _overlays(app)[0]["range"]
    region.lines[0].setValue(20.0)
    region.sigRegionChangeFinished.emit(region)
    assert tool.xbgn == 20.0 and tab.xbgn_edit.text() == "20"


def test_results_box_lists_the_spikes(app):
    text = app.spike.details.toPlainText()
    assert text.splitlines() == ["RecordA0  rising 0.5  (x in ms)", "2 spikes: 9.500, 39.500"]
    _type(app.spike.ylevel_edit, "2")
    assert app.spike.details.toPlainText().splitlines()[1] == "No spikes"


def test_run_all_epochs(app):
    tab, tool = app.spike, app.spike.spike_tool
    app.selection_model.update(epoch_set="All")
    assert tab.run_info_label.text() == "n=3"
    tab.run_button.click()
    assert tab.status_label.text() == "Complete"
    folder = tool.toolfolder
    assert folder.name == "Spike_Record_A_All_0"
    assert list(folder.data.get("SP_count").nparray) == [2, 3, 1]
    assert list(folder.data.get("SP_RY").nparray) == [0, 0, 1, 1, 1, 2]
    assert _commands(app)[-1] == (
        "nm.spike.run_selection(nm.folders['Demo'].dataseries['Record'], "
        "channel='A', epochs='all')")
    assert tab.run_summary() == "Complete → Spike_Record_A_All_0"


def test_run_with_no_spikes_warns(app):
    tab = app.spike
    _type(tab.ylevel_edit, "2")
    tab.run_button.click()
    assert tab.status_label.text() == "Complete with warnings"
    assert tab.warnings == ["No spikes found"]


def test_settings_page(app):
    tab, tool = app.spike, app.spike.spike_tool
    options = {option[0]: option for option in tab.config_options()}
    assert list(options) == ["ignore_nans", "overwrite", "results_to_numpy",
                             "results_all_epochs", "results_format", "decimals"]
    assert options["results_format"][2] == "raster"
    assert list(options["results_format"][4]) == ["raster", "epochs"]
    dialog = app.settings_dialog()
    combo = dialog.tool_combos["Spike"]["results_format"]
    combo.setCurrentIndex(combo.findData("epochs"))
    assert tool.results_format == "epochs"
    assert tool.config.results_format == "epochs"
    dialog.tool_checkboxes["Spike"]["overwrite"].setChecked(False)
    assert tool.overwrite is False


def test_decimals_displayed(app):
    """Settings › Spike › Decimals displayed: the spike times all to the same
    decimal places (a setting of the app, as Stats's)."""
    dialog = app.settings_dialog()
    spin = dialog.tool_spinboxes["Spike"]["decimals"]
    assert (spin.value(), spin.minimum(), spin.maximum()) == (3, 0, 10)
    spin.setValue(1)
    assert app.manager.settings.spike_decimals == 1
    assert app.spike.details.toPlainText().splitlines()[1] == "2 spikes: 9.5, 39.5"
    spin.setValue(0)
    assert app.spike.details.toPlainText().splitlines()[1] == "2 spikes: 10, 40"
