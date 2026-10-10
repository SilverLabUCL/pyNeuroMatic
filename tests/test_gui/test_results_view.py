"""Tests for the Table context tab and the Stats2 panel."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.context_views import ValuesPlot
from pyneuromatic.gui.results_view import StatsTables

pytestmark = pytest.mark.gui

PEAKS = (20.0, 22.0, 24.0, 26.0)


def _stats_window(qtbot, *, arrays=True):
    """App window with four step traces (peak above -70 from PEAKS) selected."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch, peak in enumerate(PEAKS):
        y = np.r_[np.full(10, -70.0), np.full(5, -70.0 + peak), np.full(5, -70.0)]
        folder.data.new(
            f"RecordA{epoch}",
            nparray=y,
            xscale={"start": 0.0, "delta": 1.0},
            yscale={"units": "mV"},
        )
    dataseries = folder.sync_dataseries("Record", select=True)
    dataseries.epochs.sets.add("Set1", [f"E{i}" for i in range(len(PEAKS))])
    manager.select_keys = {
        "folder": folder.name,
        "dataseries": dataseries.name,
        "channel": "A",
        "epoch": "E0",
    }
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(set="Set1")

    tab = window.stats_tab
    tab.output_arrays_checkbox.setChecked(arrays)
    tab.measurement_combo.setCurrentText("max")
    tab.xbgn_edit.setText("10")
    tab.xbgn_edit.editingFinished.emit()
    tab.bsln_checkbox.setChecked(True)
    # Prefill ends the baseline at X begin (10), the first step sample
    tab.bsln_xend_edit.setText("9")
    tab.bsln_xend_edit.editingFinished.emit()
    return manager, folder, window


def _headers(table):
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


def _row_labels(table):
    return [table.verticalHeaderItem(r).text() for r in range(table.rowCount())]


def _column(table, label):
    column = _headers(table).index(label)
    return [table.item(r, column).text() for r in range(table.rowCount())]


def _tab_texts(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_stats_tabs_appear_in_plot_and_table_with_results(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    panel = window.context_panel
    assert _tab_texts(panel) == ["Plot", "Table"]
    plot, table = window.plot_widget, window.table_panel
    assert _tab_texts(plot.view_tabs) == ["A"]
    assert _tab_texts(table.view_tabs) == ["A"]

    window.stats_tab.run_button.click()
    assert _tab_texts(plot.view_tabs) == ["A", "Stats"]
    assert _tab_texts(table.view_tabs) == ["A", "Stats"]
    assert table.view_tabs.current_view() == ("tool", "Stats")
    assert isinstance(table.stack.currentWidget(), StatsTables)
    assert plot.view_tabs.current_view() == ("channel", "A")

    # Stepping through traces keeps the Stats view selected
    window.selection_model.update(epoch=folder.dataseries["Record"].epochs["E1"])
    assert table.view_tabs.current_view() == ("tool", "Stats")

    table.view_tabs.select_channel("A")
    assert table.stack.currentWidget() is table.channel_table

    # Deleting the only Stats folder removes the tabs
    stats2 = window.stats_tab.stats2
    stats2._confirm_delete = lambda name: True
    stats2.delete_button.click()
    assert _tab_texts(plot.view_tabs) == ["A"]
    assert _tab_texts(table.view_tabs) == ["A"]
    assert table.stack.currentWidget() is table.channel_table


def test_stats_tool_has_stats1_and_stats2_tabs(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.stats_tab
    modes = tab.mode_tabs
    assert [modes.tabText(i) for i in range(modes.count())] == ["Stats1", "Stats2"]

    modes.setCurrentWidget(tab.stats2)
    assert tab.results_group.isHidden()  # results are in Stats2 and the Table
    assert tab.run_row.isHidden()
    modes.setCurrentIndex(0)
    assert not tab.run_row.isHidden()


def test_run_selects_stats2_when_arrays_saved(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.stats_tab
    assert tab.mode_tabs.currentIndex() == 0
    tab.run_button.click()
    assert tab.mode_tabs.currentWidget() is tab.stats2
    assert tab.stats2.folder_combo.currentText() == "Stats_Record_A_0"


def test_run_stays_on_stats1_without_arrays(qtbot):
    _manager, _folder, window = _stats_window(qtbot, arrays=False)
    tab = window.stats_tab
    tab.run_button.click()
    assert tab.mode_tabs.currentIndex() == 0


def test_run_fills_arrays_and_summary_tables(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2
    tables = stats2.tables

    assert window.context_panel.currentWidget() is window.table_panel
    assert stats2.folder_combo.currentText() == "Stats_Record_A_0"
    assert stats2.window_combo.currentText() == "w0"
    assert tables.title_label.text() == "Stats_Record_A_0  w0"

    labels = ["bsln_y", "max_y", "max_x", "max_ds"]
    assert _headers(tables.arrays_table) == labels
    assert _headers(tables.summary_table) == labels
    assert _row_labels(tables.arrays_table) == [
        "RecordA0", "RecordA1", "RecordA2", "RecordA3",
    ]
    assert _row_labels(tables.summary_table) == [
        "mean", "std", "sem", "N", "NaNs", "INFs", "min", "max",
    ]
    assert _column(tables.arrays_table, "max_ds") == ["20", "22", "24", "26"]
    assert _column(tables.summary_table, "max_ds") == [
        "23", "2.58199", "1.29099", "4", "0", "0", "20", "26",
    ]
    assert "mV" in tables.arrays_table.horizontalHeaderItem(3).toolTip()

    # Columns stay aligned between the two tables
    for column in range(len(labels)):
        assert (
            tables.arrays_table.columnWidth(column)
            == tables.summary_table.columnWidth(column)
        )

    # A second run is shown as the newest folder
    window.stats_tab.run_button.click()
    assert stats2.folder_combo.currentText() == "Stats_Record_A_1"


def test_column_selection_is_shared_between_tables(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2
    tables = stats2.tables

    tables.summary_table.selectColumn(3)
    assert tables.selected_array() == "ST_w0_max_ds"
    assert {i.column() for i in tables.arrays_table.selectedIndexes()} == {3}
    assert stats2.plot_button.isEnabled()
    assert stats2.array_combo.currentText() == "ST_w0_max_ds"


def test_array_dropdown_selects_column_and_shows_details(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2
    tables = stats2.tables

    names = [stats2.array_combo.itemText(i) for i in range(stats2.array_combo.count())]
    assert names == ["ST_w0_bsln_y", "ST_w0_max_y", "ST_w0_max_x", "ST_w0_max_ds"]
    # The first array is chosen and highlighted in the tables
    assert tables.selected_array() == "ST_w0_bsln_y"
    assert stats2.details.toPlainText().startswith("ST_w0_bsln_y")

    stats2.array_combo.setCurrentText("ST_w0_max_ds")
    assert tables.selected_array() == "ST_w0_max_ds"
    details = stats2.details.toPlainText().splitlines()
    assert details[0] == "ST_w0_max_ds (mV)"
    assert details[1].startswith("NMStats(win=w0, func=max")
    assert "mean  23" in details
    assert "N     4" in details

    stats2.all_columns_checkbox.setChecked(True)
    assert stats2.array_combo.currentText() == "ST_w0_max_ds"
    assert stats2.array_combo.findText("ST_w0_max_n") >= 0


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_details_list_nan_rows(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    folder.data["RecordA2"].nparray[:10] = np.nan  # no baseline for this trace
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2
    stats2.array_combo.setCurrentText("ST_w0_max_ds")
    assert stats2.details.toPlainText().splitlines()[-1] == "NaN for: RecordA2"


def test_all_columns_and_window_switch(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.stats_tab
    tab.window_combo.setCurrentIndex(tab.window_combo.findData(tab.NEW_WINDOW_ACTION))
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    stats2 = tab.stats2
    tables = stats2.tables

    assert [stats2.window_combo.itemText(i) for i in range(stats2.window_combo.count())] == [
        "w0", "w1",
    ]
    assert "max_n" not in _headers(tables.arrays_table)
    stats2.all_columns_checkbox.setChecked(True)
    assert {"max_i", "max_n", "bsln_n"} <= set(_headers(tables.arrays_table))

    stats2.window_combo.setCurrentText("w1")
    assert "mean_y" in _headers(tables.arrays_table)


def test_plot_and_histogram_draw_in_plot_tab(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2
    stats2.array_combo.setCurrentText("ST_w0_max_ds")

    plots = []
    stats2.plot_requested.connect(lambda *args: plots.append(args))
    stats2.plot_button.click()
    title, y, x, _xlabel, ylabel, style = plots[-1]
    assert title == "Stats_Record_A_0: ST_w0_max_ds"
    assert list(y) == list(PEAKS) and x is None and style == "points"
    assert ylabel == "ST_w0_max_ds (mV)"
    assert window.context_panel.currentWidget() is window.plot_widget
    plot = window.plot_widget
    assert plot.view_tabs.current_view() == ("tool", "Stats")
    assert isinstance(plot.stack.currentWidget(), ValuesPlot)
    assert window.stats_plot.plot_widget.getPlotItem().listDataItems()
    # The trace plot is untouched
    assert plot.plot_widget.getPlotItem().listDataItems()

    stats2.histogram_button.click()
    _title, counts, edges, _xlabel, _ylabel, style = plots[-1]
    assert style == "histogram"
    assert sum(counts) == len(PEAKS)
    assert len(edges) == len(counts) + 1


def test_copy_tables_as_tab_separated_text(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()

    window.stats_tab.stats2.copy_button.click()
    lines = QtWidgets.QApplication.clipboard().text().splitlines()
    assert lines[0] == "\tbsln_y\tmax_y\tmax_x\tmax_ds"
    assert lines[1] == "RecordA0\t-70\t-50\t10\t20"
    assert lines[5] == ""
    assert lines[6] == lines[0]
    assert lines[7].startswith("mean\t-70\t-47")


def test_delete_folder_after_confirmation(qtbot, monkeypatch):
    _manager, folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2

    monkeypatch.setattr(stats2, "_confirm_delete", lambda name: False)
    stats2.delete_button.click()
    assert "Stats_Record_A_0" in folder.toolfolders

    monkeypatch.setattr(stats2, "_confirm_delete", lambda name: True)
    stats2.delete_button.click()
    assert "Stats_Record_A_0" not in folder.toolfolders
    assert stats2.folder_combo.count() == 0
    assert stats2.tables.arrays_table.columnCount() == 0
    assert "No saved Stats results in Demo" in stats2.tables.message_label.text()


def test_without_stats_arrays_table_explains(qtbot):
    _manager, _folder, window = _stats_window(qtbot, arrays=False)
    window.stats_tab.run_button.click()
    stats2 = window.stats_tab.stats2

    assert window.stats_tab.status_label.text() == "Complete"
    assert window.context_panel.currentWidget() is window.plot_widget
    assert stats2.folder_combo.count() == 0
    assert "Run Stats1 with 'Stats arrays' on" in stats2.message_label.text()
