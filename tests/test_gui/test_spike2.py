"""Spike2: the Spike tool's saved results (#529)."""

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.spike2_view import Spike2Panel, SpikeTables

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
    tab.ylevel_edit.setText("0.5")
    tab.ylevel_edit.editingFinished.emit()
    window.spike, window.manager, window.series = tab, manager, series
    return window


def _run_all(app):
    app.selection_model.update(epoch_set="All")
    app.spike.run_button.click()
    assert app.spike.status_label.text() == "Complete"


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _plotted(app):
    """The Spike2 plot: (title, x, y) of its one item."""
    widget = app.spike.plot_view.plot_widget
    items = widget.listDataItems()
    x, y = items[0].getData() if items else (None, None)
    return widget.plotItem.titleLabel.text, x, y


def test_spike2_tab_without_results(app):
    tab = app.spike
    assert isinstance(tab.spike2, Spike2Panel)
    assert _items_tabs(tab) == ["Spike1", "Spike2"]
    assert not tab.has_results() and tab.lists_view()
    assert "No saved Spike results in Demo" in tab.spike2.message_label.text()
    assert _plotted(app)[0] == tab.NO_RESULTS_TITLE
    # Spike2 chosen: no Run; its tab in the Plot panel
    tab.mode_tabs.setCurrentWidget(tab.spike2)
    assert tab.run_row.isHidden() and tab.preferred_view() == "tool"
    tab.mode_tabs.setCurrentIndex(0)
    assert not tab.run_row.isHidden() and tab.preferred_view() == "channel"


def _items_tabs(tab):
    return [tab.mode_tabs.tabText(i) for i in range(tab.mode_tabs.count())]


def test_a_run_shows_its_raster(app):
    _run_all(app)
    tab, panel = app.spike, app.spike.spike2
    assert tab.mode_tabs.currentWidget() is panel
    assert panel.toolfolder().name == "Spike_Record_A_All_0"
    assert _items(panel.array_combo) == ["SP_RX", "SP_count", "SP_rate", "SP_first", "SP_isi"]
    assert panel.selected_array() == "SP_RX"
    title, x, y = _plotted(app)
    assert title == "Spike_Record_A_All_0: raster"
    assert list(y) == [0, 0, 1, 1, 1, 2]
    assert list(x) == pytest.approx([9.5, 39.5, 9.5, 39.5, 69.5, 39.5])
    assert tab.plot_view.plot_widget.getAxis("bottom").labelText == "Time (ms)"
    assert app.context_panel.currentWidget() is app.plot_widget


def test_plot_a_row_array_against_its_epochs(app):
    _run_all(app)
    panel = app.spike.spike2
    panel.select_array("SP_count")
    title, x, y = _plotted(app)
    assert title == "Spike_Record_A_All_0: SP_count"
    assert list(x) == [0, 1, 2] and list(y) == [2, 3, 1]
    # Raster: whichever array is chosen; SP_RX chosen with it
    panel.raster_action.trigger()
    assert _plotted(app)[0].endswith(": raster")
    assert panel.selected_array() == "SP_RX"
    assert panel.details.toPlainText().startswith("SP_RX")


def test_the_table(app):
    _run_all(app)
    panel, tables = app.spike.spike2, app.spike.spike2.tables
    assert app.spike.table_view is tables
    assert tables.columns == ["SP_count", "SP_rate", "SP_first", "SP_isi"]
    headers = [tables.arrays_table.horizontalHeaderItem(c).text() for c in range(4)]
    assert headers == ["count", "rate", "first", "isi"]
    rows = [tables.arrays_table.verticalHeaderItem(r).text() for r in range(3)]
    assert rows == ["RecordA0", "RecordA1", "RecordA2"]
    assert tables.arrays_table.item(1, 0).text() == "3"  # a count, as it is
    assert tables.arrays_table.item(0, 2).text() == "9.500"  # first spike, 3 decimals
    # Table (Array ⋯): the Table panel, the chosen array's column
    panel.select_array("SP_first")
    panel.table_action.trigger()
    assert app.context_panel.currentWidget() is app.table_panel
    assert tables.selected_array() == "SP_first"
    # A column clicked picks the array
    tables.arrays_table.setCurrentCell(0, 1)
    assert panel.selected_array() == "SP_rate"


def test_results_box(app):
    _run_all(app)
    panel = app.spike.spike2
    panel.select_array("SP_count")
    lines = panel.details.toPlainText().splitlines()
    assert lines[0] == "SP_count  (Spikes)"
    assert lines[1].startswith("NMSpike: Record, channel A")
    assert "mean  2.000" in lines and "N     3" in lines


def test_epochs_format_still_shows_the_raster(app):
    app.spike.spike_tool.results_format = "epochs"
    _run_all(app)
    panel = app.spike.spike2
    assert "SP_RX" not in _items(panel.array_combo)
    assert panel.selected_array() == "SP_count"
    title, x, y = _plotted(app)
    assert title.endswith(": raster") and list(y) == [0, 0, 1, 1, 1, 2]
    # Raster (no SP_RX to choose): plotted, SP_count still chosen
    panel.plot_selected()
    panel.raster_action.trigger()
    assert _plotted(app)[0].endswith(": raster")
    assert panel.selected_array() == "SP_count"


def test_one_point_per_epoch_in_settings(app):
    tab, tool = app.spike, app.spike.spike_tool
    dialog = app.settings_dialog()
    box = dialog.tool_checkboxes["Spike"]["results_all_epochs"]
    assert box.text() == "SP_ arrays: one point per epoch" and not box.isChecked()
    box.setChecked(True)
    assert tool.results_all_epochs and tool.config.results_all_epochs
    tool.run_selection(app.series, channel="A", epochs=["E0", "E2"])  # E1 left out
    tab.spike2.show_toolfolder(tool.toolfolder)
    tables = tab.spike2.tables
    rows = [tables.arrays_table.verticalHeaderItem(r).text() for r in range(3)]
    assert rows == ["RecordA0", "E1", "RecordA2"]  # E1 not analysed
    count = tool.toolfolder.data.get("SP_count").nparray
    assert count[0] == 2 and math.isnan(count[1]) and count[2] == 1
    assert tables.arrays_table.item(1, 0).text() == "NaN"


def test_decimals_apply_to_the_tables(app):
    _run_all(app)
    app.spike.set_config_option("decimals", 1)
    assert SpikeTables.decimals == 1
    assert app.spike.spike2.tables.arrays_table.item(0, 2).text() == "9.5"
    app.spike.set_config_option("decimals", 3)


def test_plot_x_epoch(app):
    """Plot x: epoch (as Stats2): a row array against SP_epoch_num, or off,
    the row index; not for the raster."""
    app.spike.spike_tool.results_all_epochs = False
    app.selection_model.update(epoch_set="All")
    app.spike.spike_tool.run_selection(app.series, channel="A", epochs=["E1", "E2"])
    panel = app.spike.spike2
    panel.show_toolfolder(app.spike.spike_tool.toolfolder)
    texts = [a.text() for a in panel.plot_menu.actions() if not a.isSeparator()]
    assert texts == ["Plot", "Plot x: epoch", "Table", "Raster", "Show in Browser"]
    action = panel.epoch_x_action
    panel.select_array("SP_count")
    panel.plot_selected()
    _title, x, y = _plotted(app)
    assert action.isEnabled() and action.isChecked()
    assert list(x) == [1, 2] and list(y) == [3, 1]
    assert app.spike.plot_view.plot_widget.getAxis("bottom").labelText == "epoch"
    action.setChecked(False)  # the plot follows
    _title, x, y = _plotted(app)
    assert list(x) == [0, 1] and list(y) == [3, 1]
    assert app.spike.plot_view.plot_widget.getAxis("bottom").labelText == "data index"
    panel.select_array("SP_RX")
    assert not action.isEnabled()
    action.setChecked(True)


def test_each_subfolder_remembers_its_array(app):
    _run_all(app)
    _run_all(app)  # Overwrite off: a second subfolder
    panel = app.spike.spike2
    first, second = "Spike_Record_A_All_0", "Spike_Record_A_All_1"
    assert panel.toolfolder().name == second
    panel.select_array("SP_rate")
    panel.folder_combo.setCurrentIndex(panel.folder_combo.findData(first))
    assert panel.selected_array() == "SP_RX"  # as it was after its run
    panel.select_array("SP_first")
    panel.folder_combo.setCurrentIndex(panel.folder_combo.findData(second))
    assert panel.selected_array() == "SP_rate"
    assert _plotted(app)[0] == f"{second}: SP_rate"
    panel.folder_combo.setCurrentIndex(panel.folder_combo.findData(first))
    assert panel.selected_array() == "SP_first"
    # A run shows its default (the raster), as before
    _run_all(app)
    assert panel.selected_array() == "SP_RX"
