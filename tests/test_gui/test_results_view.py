"""Tests for the Table context tab and the Stats2 panel."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.context_views import ValuesPlot
from pyneuromatic.gui.results_view import Stats2Panel, StatsTables

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
    window.selection_model.update(epoch_set="Set1")

    tab = window.tool_tabs["Stats"]
    tab.set_config_option("results_to_numpy", arrays)  # Settings › Stats (#480)
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


def _delete(stats2):
    """Folder menu > Delete <the subfolder shown>."""
    stats2.folder_combo.setCurrentIndex(
        stats2.folder_combo.findData(stats2.DELETE_FOLDER_ACTION))


def _tab_texts(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_stats_tabs_appear_in_plot_and_table_with_results(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    plot, table = window.plot_widget, window.table_panel
    assert _tab_texts(plot.view_tabs) == ["A"]
    assert _tab_texts(table.view_tabs) == ["A"]

    window.tool_tabs["Stats"].run_button.click()
    assert _tab_texts(plot.view_tabs) == ["A", "Stats"]
    assert _tab_texts(table.view_tabs) == ["A", "Stats"]
    assert table.view_tabs.current_view() == ("tool", "Stats")
    assert isinstance(table.stack.currentWidget(), StatsTables)
    # One bar (#460): the Plot panel shows Stats too, its plot
    assert plot.view_tabs.current_view() == ("tool", "Stats")
    assert plot.stack.currentWidget() is window.tool_tabs["Stats"].plot_view

    # Stepping through traces keeps the Stats view selected
    window.selection_model.update(epoch=folder.dataseries["Record"].epochs["E1"])
    assert table.view_tabs.current_view() == ("tool", "Stats")

    table.view_tabs.select_channel("A")
    assert table.stack.currentWidget() is table.channel_table

    # Deleting the only Stats folder removes the tabs
    stats2 = window.tool_tabs["Stats"].stats2
    stats2._confirm_delete = lambda name: True
    _delete(stats2)
    assert _tab_texts(plot.view_tabs) == ["A"]
    assert _tab_texts(table.view_tabs) == ["A"]
    assert table.stack.currentWidget() is table.channel_table


def test_stats_tool_has_stats1_and_stats2_tabs(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    modes = tab.mode_tabs
    assert [modes.tabText(i) for i in range(modes.count())] == ["Stats1", "Stats2"]

    modes.setCurrentWidget(tab.stats2)
    assert tab.results_group.isHidden()  # results are in Stats2 and the Table
    assert tab.run_row.isHidden()
    modes.setCurrentIndex(0)
    assert not tab.run_row.isHidden()


def test_run_selects_stats2_when_arrays_saved(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    assert tab.mode_tabs.currentIndex() == 0
    tab.run_button.click()
    assert tab.mode_tabs.currentWidget() is tab.stats2
    assert tab.stats2.folder_combo.currentText() == "Stats_Record_A_S1_0"


def test_run_adds_subfolder_to_selection_strip(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    sub_menu = window.selection_strip.combo("toolfolder")
    assert [sub_menu.itemText(i) for i in range(sub_menu.count())] == ["—"]
    window.tool_tabs["Stats"].run_button.click()
    assert [sub_menu.itemText(i) for i in range(sub_menu.count())] == [
        "—", "Stats_Record_A_S1_0",
    ]
    assert sub_menu.isEnabled()


def test_run_stays_on_stats1_without_arrays(qtbot):
    _manager, _folder, window = _stats_window(qtbot, arrays=False)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    assert tab.mode_tabs.currentIndex() == 0


def test_run_fills_arrays_and_summary_tables(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    tables = stats2.tables

    # After a run: the Stats plot (its tables a click away, on Table)
    assert window.context_panel.currentWidget() is window.plot_widget
    assert window.view_tabs.current_view() == ("tool", "Stats")
    assert stats2.folder_combo.currentText() == "Stats_Record_A_S1_0"
    assert stats2.window_combo.currentText() == "w0"
    assert tables.title_label.text() == "Stats_Record_A_S1_0  w0"

    labels = ["bsln_y", "max_y", "max_x", "max_dy"]
    assert _headers(tables.arrays_table) == labels
    assert _headers(tables.summary_table) == labels
    assert _row_labels(tables.arrays_table) == [
        "RecordA0", "RecordA1", "RecordA2", "RecordA3",
    ]
    assert _row_labels(tables.summary_table) == [
        "mean", "std", "sem", "N", "NaNs", "INFs", "min", "max",
    ]
    assert _column(tables.arrays_table, "max_dy") == ["20", "22", "24", "26"]
    assert _column(tables.summary_table, "max_dy") == [
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
    window.tool_tabs["Stats"].run_button.click()
    assert stats2.folder_combo.currentText() == "Stats_Record_A_S1_1"


def test_column_selection_is_shared_between_tables(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    tables = stats2.tables

    tables.summary_table.setCurrentCell(1, 3)  # a click on one summary cell
    assert tables.selected_array() == "ST_w0_max_dy"
    assert tables.arrays_table.currentColumn() == 3
    # Only the clicked cell is highlighted, not its column
    assert [(i.row(), i.column()) for i in tables.summary_table.selectedIndexes()] == [(1, 3)]
    assert tables.arrays_table.selectedIndexes() == []
    assert stats2.plot_button.isEnabled()
    assert stats2.array_combo.currentText() == "ST_w0_max_dy"


def test_array_dropdown_selects_column_and_shows_details(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    tables = stats2.tables

    names = [stats2.array_combo.itemText(i) for i in range(stats2.array_combo.count())]
    assert names == ["ST_w0_bsln_y", "ST_w0_max_y", "ST_w0_max_x", "ST_w0_max_dy"]
    # The measurement is chosen (not the baseline) and highlighted in the tables
    assert tables.selected_array() == "ST_w0_max_y"
    assert stats2.details.toPlainText().startswith("ST_w0_max_y")

    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    assert tables.selected_array() == "ST_w0_max_dy"
    details = stats2.details.toPlainText().splitlines()
    assert details[0] == "ST_w0_max_dy  (values in mV)"  # no x units in this data
    assert details[1].startswith("NMStats(win=w0, func=max")
    assert "mean  23" in details
    assert "N     4" in details

    stats2.all_columns_checkbox.setChecked(True)
    assert stats2.array_combo.currentText() == "ST_w0_max_dy"
    assert stats2.array_combo.findText("ST_w0_max_n") >= 0


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_details_list_nan_rows(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    folder.data["RecordA2"].nparray[:10] = np.nan  # no baseline for this trace
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    assert stats2.details.toPlainText().splitlines()[-1] == "NaN for: RecordA2"


def test_all_columns_and_window_switch(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
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


def test_stats_plot_follows_the_stats2_array(qtbot):
    """After a run, the Stats plot already shows the array chosen in Stats2
    (the Table panel stays shown), and follows a new choice."""
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    plot_item = tab.plot_view.plot_widget.getPlotItem()
    assert not plot_item.listDataItems()
    assert plot_item.titleLabel.text == ValuesPlot.EMPTY_TITLE
    tab.run_button.click()
    assert window.context_panel.currentWidget() is window.plot_widget  # shown after a run
    assert window.plot_widget.stack.currentWidget() is tab.plot_view
    chosen = tab.stats2.selected_array()
    assert chosen is not None
    assert plot_item.titleLabel.text == f"Stats_Record_A_S1_0: {chosen}"
    assert plot_item.listDataItems()

    tab.stats2.array_combo.setCurrentText("ST_w0_max_dy")
    assert plot_item.titleLabel.text == "Stats_Record_A_S1_0: ST_w0_max_dy"
    assert list(plot_item.listDataItems()[0].yData) == list(PEAKS)
    window.context_panel.setCurrentWidget(window.table_panel)
    tab.stats2.array_combo.setCurrentText("ST_w0_max_y")
    assert window.context_panel.currentWidget() is window.table_panel  # a choice doesn't switch

    # Deleting the folder: nothing chosen, the hint again
    tab.stats2._confirm_delete = lambda name: True
    _delete(tab.stats2)
    assert not plot_item.listDataItems()
    assert plot_item.titleLabel.text == ValuesPlot.EMPTY_TITLE


def test_plot_and_histogram_draw_in_plot_tab(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")

    plots = []
    stats2.plot_requested.connect(lambda *args: plots.append(args))
    stats2.plot_button.click()
    title, y, x, _xlabel, ylabel, style = plots[-1]
    assert title == "Stats_Record_A_S1_0: ST_w0_max_dy"
    assert list(y) == list(PEAKS) and style == "points"
    assert list(x) == [0, 1, 2, 3] and _xlabel == "epoch"  # x: epoch (#476)
    assert ylabel == "ST_w0_max_dy (mV)"
    assert window.context_panel.currentWidget() is window.plot_widget
    plot = window.plot_widget
    assert plot.view_tabs.current_view() == ("tool", "Stats")
    assert isinstance(plot.stack.currentWidget(), ValuesPlot)
    assert window.tool_tabs["Stats"].plot_view.plot_widget.getPlotItem().listDataItems()
    # The trace plot is untouched
    assert plot.plot_widget.getPlotItem().listDataItems()

    # Histogram (⋯ menu, #485): Plot draws it, a preview (nothing saved)
    stats2.histogram_action.setChecked(True)
    assert not stats2.epoch_x_action.isEnabled()  # its x is the bin edges
    stats2.plot_button.click()
    _title, counts, edges, _xlabel, _ylabel, style = plots[-1]
    assert style == "histogram"
    assert sum(counts) == len(PEAKS)
    assert len(edges) == len(counts) + 1
    subfolder = window.tool_tabs["Stats"].stats_tool.toolfolder
    assert "H_ST_w0_max_dy_counts" not in subfolder.data
    # Save Histogram saves what's shown, and replaces it when saved again
    stats2.save_histogram_action.trigger()
    assert list(subfolder.data.get("H_ST_w0_max_dy_counts").nparray) == list(counts)
    assert list(subfolder.data.get("H_ST_w0_max_dy_edges").nparray) == list(edges)
    stats2.save_histogram_action.trigger()
    assert "H_ST_w0_max_dy_counts" in subfolder.data
    # Off: points again, x: epoch back
    stats2.histogram_action.setChecked(False)
    assert stats2.epoch_x_action.isEnabled()
    stats2.plot_button.click()
    assert plots[-1][5] == "points"


def _field_widgets(panel, label: str) -> list:
    """The widgets of form row *label* in *panel*."""
    form = panel.findChild(QtWidgets.QFormLayout)
    role = QtWidgets.QFormLayout.ItemRole
    for row in range(form.rowCount()):
        if form.itemAt(row, role.LabelRole).widget().text() == label:
            field = form.itemAt(row, role.FieldRole).layout()
            return [field.itemAt(i).widget() for i in range(field.count())
                    if field.itemAt(i).widget() is not None]
    return []


def test_plot_options_are_in_a_menu_beside_plot(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    assert stats2.plot_menu_button.menu() is stats2.plot_menu
    texts = [a.text() for a in stats2.plot_menu.actions() if a.text()]
    assert texts == ["Histogram", "x: epoch", "Save Histogram"]
    assert not stats2.histogram_action.isChecked()  # points, until turned on
    # The Array row: the dropdown (its width back), Plot and ⋯ only (#485)
    assert _field_widgets(stats2, "Array") == [
        stats2.array_combo, stats2.plot_button, stats2.plot_menu_button]


@pytest.mark.parametrize("style", ["macos", "fusion"])
def test_array_list_spans_its_row(qtbot, style):
    """The rows span the panel, so array names aren't cut off, also with
    macOS's style (its forms keep fields at their size hint)."""
    if style not in [key.lower() for key in QtWidgets.QStyleFactory.keys()]:
        pytest.skip(f"no {style} style here")
    panel = Stats2Panel(NMManager(quiet=True))
    qtbot.addWidget(panel)
    qt_style = QtWidgets.QStyleFactory.create(style)
    panel.setStyle(qt_style)
    panel.resize(420, 600)
    panel.show()
    qtbot.waitExposed(panel)
    combo, plot, menu = panel.array_combo, panel.plot_button, panel.plot_menu_button
    assert combo.width() > panel.width() // 2
    # Up to Plot and ⋯, at the row's end
    assert combo.geometry().right() < plot.geometry().left()
    assert menu.geometry().right() >= panel.width() - 20


def test_histogram_follows_the_chosen_array_without_saving(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    stats2 = tab.stats2
    stats2.histogram_action.setChecked(True)
    drawn = []
    stats2.array_changed.connect(lambda: drawn.append(stats2.selected_plot()[5]))
    for i in range(stats2.array_combo.count()):  # browsing
        stats2.array_combo.setCurrentIndex(i)
    assert drawn and set(drawn) == {"histogram"}
    assert not any(name.startswith("H_") for name in tab.stats_tool.toolfolder.data)
    stats2.refresh()
    assert stats2.histogram_action.isChecked()  # on until turned off


def test_copy_all_stats_tables_from_the_table_panel(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    assert not hasattr(window.tool_tabs["Stats"].stats2, "copy_button")  # moved to the Table panel

    window.table_panel.show_tool("Stats")
    window.table_panel.copy_all_action.trigger()
    lines = QtWidgets.QApplication.clipboard().text().splitlines()
    assert lines[0] == "\tbsln_y\tmax_y\tmax_x\tmax_dy"
    assert lines[1] == "RecordA0\t-70\t-50\t10\t20"
    assert lines[5] == ""
    assert lines[6] == lines[0]
    assert lines[7].startswith("mean\t-70\t-47")


def test_delete_folder_after_confirmation(qtbot, monkeypatch):
    _manager, folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2

    monkeypatch.setattr(stats2, "_confirm_delete", lambda name: False)
    _delete(stats2)
    assert "Stats_Record_A_S1_0" in folder.toolfolders

    monkeypatch.setattr(stats2, "_confirm_delete", lambda name: True)
    _delete(stats2)
    assert "Stats_Record_A_S1_0" not in folder.toolfolders
    assert stats2.folder_combo.count() == 0
    assert stats2.tables.arrays_table.columnCount() == 0
    assert "No saved Stats results in Demo" in stats2.tables.message_label.text()


def test_without_stats_arrays_table_explains(qtbot):
    _manager, _folder, window = _stats_window(qtbot, arrays=False)
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2

    assert window.tool_tabs["Stats"].status_label.text() == "Complete"
    assert window.context_panel.currentWidget() is window.plot_widget
    assert stats2.folder_combo.count() == 0
    assert "Run Stats1 with 'Stats arrays' on" in stats2.message_label.text()


def test_browser_shows_new_results_folder_after_run(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    browser = window.browser_widget
    model = browser.tree.model()

    def tool_folders():
        root = model.index(0, 0)
        demo = next(model.index(r, 0, root) for r in range(model.rowCount(root))
                    if model.data(model.index(r, 0, root)) == folder.name)
        for r in range(model.rowCount(demo)):
            index = model.index(r, 0, demo)
            if model.data(index) == "Tool Folders":
                return [model.data(model.index(i, 0, index)) for i in range(model.rowCount(index))]
        return []

    assert tool_folders() == []
    window.tool_tabs["Stats"].run_button.click()
    assert tool_folders() == ["Stats_Record_A_S1_0"]


@pytest.mark.parametrize("names, measure", [
    (["ST_w0_bsln_y", "ST_w0_max_y", "ST_w0_max_x", "ST_w0_max_dy"], "ST_w0_max_y"),
    (["ST_w0_bsln_y", "ST_w0_rt_p_y", "ST_w0_rt_p_x", "ST_w0_rt_p_dx"], "ST_w0_rt_p_dx"),
    (["ST_w0_bsln_y", "ST_w0_level_p_x", "ST_w0_level_p_dy"], "ST_w0_level_p_x"),
    (["ST_w0_mean_y"], "ST_w0_mean_y"),  # no baseline
    ([], None),
])
def test_the_measurement_array(qtbot, monkeypatch, names, measure):
    _manager, _folder, window = _stats_window(qtbot)
    stats2 = window.tool_tabs["Stats"].stats2
    monkeypatch.setattr(type(stats2.tables), "columns", property(lambda self: names))
    assert stats2.measure_array() == measure


def test_a_run_chooses_its_measurement_again(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    tab.stats2.array_combo.setCurrentText("ST_w0_bsln_y")  # looked at the baseline
    tab.run_button.click()
    assert tab.stats2.selected_array() == "ST_w0_max_y"
    plot = tab.plot_view.plot_widget.getPlotItem()
    assert plot.titleLabel.text.endswith(": ST_w0_max_y")  # the Stats plot shows it



def test_delete_is_in_the_folder_menu(qtbot):
    """No Delete folder button: the Folder menu ends with Delete...; with
    several subfolders, choose which; with one, confirm it."""
    _manager, folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    tab.run_button.click()
    stats2 = tab.stats2
    combo = stats2.folder_combo
    assert not hasattr(stats2, "delete_button")
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Stats_Record_A_S1_0", "Stats_Record_A_S1_1", "", "Delete..."]
    combo.setCurrentIndex(0)  # shown: _0
    asked = []
    stats2._confirm_delete = lambda name: asked.append(name) or True  # not used: several

    stats2._choose_delete = lambda names, shown: asked.append((names, shown)) and None
    _delete(stats2)  # cancelled
    assert asked == [(["Stats_Record_A_S1_0", "Stats_Record_A_S1_1"], "Stats_Record_A_S1_0")]
    assert combo.currentText() == "Stats_Record_A_S1_0"  # back to it
    assert len(folder.toolfolders) == 2

    stats2._choose_delete = lambda names, shown: "Stats_Record_A_S1_1"  # not the one shown
    _delete(stats2)
    assert "Stats_Record_A_S1_1" not in folder.toolfolders
    assert combo.currentText() == "Stats_Record_A_S1_0"  # still shown

    stats2._choose_delete = lambda names, shown: pytest.fail("one: confirm, not choose")
    asked.clear()
    _delete(stats2)  # only one left: confirm it
    assert asked == ["Stats_Record_A_S1_0"]
    assert combo.count() == 0  # none left: no Delete... either
