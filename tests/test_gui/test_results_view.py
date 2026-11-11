"""Tests for the Table context tab and the Stats2 panel."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets
from PyQt6.QtTest import QTest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.context_views import ValuesPlot
from pyneuromatic.gui.histogram_dialog import HistogramDialog
from pyneuromatic.gui.inequality_dialog import InequalityDialog
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


def _delete(stats2, confirm=True):
    """Folder ⋯ > Delete… the subfolder shown (the strip's Delete Subfolder
    prompt, #494), confirmed or cancelled."""
    stats2._folder_actions._choose = (
        lambda title, **options: (options["current"], None) if confirm else None)
    stats2.delete_folder_action.trigger()


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

    assert stats2.array_combo.findText("ST_w0_max_n") < 0
    window.tool_tabs["Stats"].set_config_option("detail_arrays", True)  # Settings › Stats (#496)
    assert stats2.array_combo.currentText() == "ST_w0_max_dy"  # still chosen
    assert stats2.array_combo.findText("ST_w0_max_n") >= 0
    window.tool_tabs["Stats"].set_config_option("detail_arrays", False)
    assert stats2.array_combo.findText("ST_w0_max_n") < 0


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_details_list_nan_rows(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    folder.data["RecordA2"].nparray[:10] = np.nan  # no baseline for this trace
    window.tool_tabs["Stats"].run_button.click()
    stats2 = window.tool_tabs["Stats"].stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    assert stats2.details.toPlainText().splitlines()[-1] == "NaN for: RecordA2"


def test_index_and_count_arrays_and_window_switch(qtbot):
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
    tab.set_config_option("detail_arrays", True)
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

    # Histogram... (⋯ menu, #492): saved as H_<array>, chosen and drawn as bars
    stats2._ask_histogram = lambda name, values, units: {"binwidth": 2}
    stats2.histogram_dialog_action.trigger()
    subfolder = window.tool_tabs["Stats"].stats_tool.toolfolder
    assert "H_ST_w0_max_dy" in subfolder.data and "H_ST_w0_max_dy_edges" not in subfolder.data
    assert stats2.selected_array() == "H_ST_w0_max_dy"
    title, counts, edges, xlabel, ylabel, style = plots[-1]
    assert title == "Stats_Record_A_S1_0: H_ST_w0_max_dy" and style == "histogram"
    assert sum(counts) == len(PEAKS) and counts[0] == 0 and counts[-1] == 0
    assert len(edges) == len(counts) + 1 and list(np.diff(edges)) == [2.0] * len(counts)
    assert (xlabel, ylabel) == ("ST_w0_max_dy (mV)", "count")
    assert not stats2.epoch_x_action.isEnabled()  # its x is the bin edges
    assert not stats2.histogram_dialog_action.isEnabled()  # not of a histogram
    # Back to an ST_ array: points, x: epoch
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
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
    assert texts == ["x: epoch", "Histogram...", "Inequality...",  # #492, #499
                     "Show in Browser"]  # #494
    assert stats2.plot_menu.actions()[1].isSeparator()
    assert stats2.plot_menu.actions()[4].isSeparator()
    # Show in Browser: the chosen array, in the Browser
    stats2.array_combo.setCurrentText("ST_w0_max_y")
    shown = []
    stats2._folder_actions.browse_requested.connect(shown.append)
    stats2.browse_array_action.trigger()
    assert shown == [window.tool_tabs["Stats"].stats_tool.toolfolder.data["ST_w0_max_y"]]
    assert window.tool_rail.current_tool_name == "Browser"
    # The Array row: the dropdown (its width back), Plot and ⋯ only (#485)
    assert _field_widgets(stats2, "Array") == [
        stats2.array_combo, stats2.plot_button, stats2.plot_menu_button]


@pytest.mark.parametrize("style", ["macos", "fusion"])
def test_array_list_fits_its_names(qtbot, style):
    """The Array list is as wide as its longest array name (not cut off, also
    with macOS's style), then Plot and ⋯; the row's space is after them."""
    if style not in [key.lower() for key in QtWidgets.QStyleFactory.keys()]:
        pytest.skip(f"no {style} style here")
    _manager, _folder, window = _stats_window(qtbot)
    window.tool_tabs["Stats"].run_button.click()
    panel = window.tool_tabs["Stats"].stats2
    panel.setStyle(QtWidgets.QStyleFactory.create(style))
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.resize(600, 600)
    panel.show()
    qtbot.waitExposed(panel)
    combo, plot, menu = panel.array_combo, panel.plot_button, panel.plot_menu_button
    names = [combo.itemText(i) for i in range(combo.count())]
    longest = max(combo.fontMetrics().horizontalAdvance(n) for n in names)
    assert longest < combo.width() < longest + 60  # fits its names, no more
    assert combo.geometry().right() < plot.geometry().left() < menu.geometry().left()
    assert menu.geometry().right() < panel.width() - 100  # not stretched to the end


def test_histograms_are_listed_after_the_windows_arrays(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    stats2 = tab.stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    stats2._ask_histogram = lambda name, values, units: {"bins": 3, "output": "probability"}
    stats2.histogram_dialog_action.trigger()
    names = [stats2.array_combo.itemText(i) for i in range(stats2.array_combo.count())]
    assert names[-1] == "H_ST_w0_max_dy" and names[:-1] == stats2.tables.columns
    assert "H_ST_w0_max_dy" not in stats2.tables.columns  # not in the Stats tables
    assert tab.stats_tool.toolfolder.data["H_ST_w0_max_dy"].nparray.sum() == pytest.approx(1)
    # Summary: its notes (source, bins, output)
    text = stats2.details.toPlainText()
    assert text.startswith("H_ST_w0_max_dy  (probability, x in mV)")
    assert "NMHistogram(array=ST_w0_max_dy, bins=3, nbins=5" in text
    # Listed again after a refresh, and chosen again
    stats2.refresh()
    assert stats2.array_combo.findText("H_ST_w0_max_dy") >= 0


def test_histogram_cancelled_saves_nothing(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    tab.stats2._ask_histogram = lambda name, values, units: None
    tab.stats2.histogram_dialog_action.trigger()
    assert not [n for n in tab.stats_tool.toolfolder.data if n.startswith("H_")]


def test_histogram_dialog(qtbot):
    values = np.array([-55.3, -52.0, -50.0, -48.0, -44.1, np.nan])
    dialog = HistogramDialog("ST_w0_max_y", values, "mV")
    qtbot.addWidget(dialog)
    assert dialog.auto_radio.isChecked() and dialog.kwargs() == {"pad": 1, "output": "count"}
    assert dialog.pad_box.value() == 1 and not dialog.width_edit.isEnabled()
    dialog.width_radio.setChecked(True)
    dialog.width_edit.setText("2")
    dialog.output_combo.setCurrentIndex(dialog.output_combo.findData("density"))
    assert dialog.kwargs() == {"binwidth": 2.0, "pad": 1, "output": "density"}
    # A range is used as given: no empty bins
    dialog.range_checkbox.setChecked(True)
    assert not dialog.pad_box.isEnabled()
    dialog.min_edit.setText("-60")
    dialog.max_edit.setText("-40")
    assert dialog.kwargs() == {"binwidth": 2.0, "xrange": (-60.0, -40.0), "output": "density"}
    # Bad values keep it open, saying why
    dialog.width_edit.setText("0")
    dialog.accept()
    assert dialog.result() == 0 and "more than 0" in dialog.error_label.text()
    # Number of bins: at least 2 (one bin isn't a histogram)
    dialog.number_radio.setChecked(True)
    assert dialog.number_box.minimum() == 2
    dialog.number_box.setValue(1)
    assert dialog.number_box.value() == 2
    assert dialog.kwargs()["bins"] == 2
    # Typed 0: an error, not the box's last good value
    dialog.number_box.lineEdit().selectAll()
    QTest.keyClicks(dialog.number_box, "0")
    assert dialog.number_box.text() == "0"
    dialog.accept()
    assert dialog.result() == 0 and "at least 2" in dialog.error_label.text()
    dialog.number_box.setValue(3)
    # Shown again next time
    again = HistogramDialog("ST_w0_max_y", values, "mV", dialog.settings())
    qtbot.addWidget(again)
    assert again.number_radio.isChecked() and again.number_box.value() == 3
    assert again.range_checkbox.isChecked()
    assert again.output_combo.currentData() == "density"


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

    _delete(stats2, confirm=False)
    assert "Stats_Record_A_S1_0" in folder.toolfolders

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
    assert "Run Stats1 with Settings › Stats › 'Save ST_ arrays' on" in stats2.message_label.text()


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



def test_folder_menu(qtbot):
    """A ⋯ menu beside the Folder list, as the strip's Subfolder ⋯ (#494):
    its prompts start on the subfolder shown; Stats2 then shows the result."""
    _manager, folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    tab.run_button.click()
    stats2 = tab.stats2
    combo = stats2.folder_combo
    assert stats2.folder_menu_button.menu() is stats2.folder_menu
    texts = [a.text() for a in stats2.folder_menu.actions()]
    assert texts == ["Rename…", "Copy…", "Delete…", "", "Show in Browser"]
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Stats_Record_A_S1_0", "Stats_Record_A_S1_1"]  # no Delete... item
    combo.setCurrentIndex(0)

    asked = []
    actions = stats2._folder_actions

    def answer(new):
        def choose(title, **options):
            asked.append((title, options["current"], options))
            return (options["current"], new) if new else None
        actions._choose = choose

    answer("Stats_Control")
    stats2.rename_folder_action.trigger()
    assert asked[-1][:2] == ("Rename Subfolder", "Stats_Record_A_S1_0")
    assert combo.currentText() == "Stats_Control"
    # The name must keep Stats_ (Stats2 lists only those)
    validate = asked[-1][2]["validate"]
    assert "must start with Stats_" in validate("Stats_Control", "Control")
    assert validate("Stats_Control", "Stats_Drug") is None

    answer(None)  # cancelled
    stats2.copy_folder_action.trigger()
    assert asked[-1][:2] == ("Copy Subfolder", "Stats_Control")
    assert combo.currentText() == "Stats_Control"
    answer("Stats_Control_copy")
    stats2.copy_folder_action.trigger()
    assert combo.currentText() == "Stats_Control_copy"
    assert "must start with Stats_" in asked[-1][2]["validate"]("Stats_Control", "Copy1")

    _delete(stats2)  # the one shown: then the newest
    assert "Stats_Control_copy" not in folder.toolfolders
    assert combo.currentText() == "Stats_Record_A_S1_1"

    shown = []
    actions.browse_requested.connect(shown.append)
    stats2.browse_folder_action.trigger()
    assert shown == [folder.toolfolders["Stats_Record_A_S1_1"]]


def test_folder_menu_greyed_out_without_a_subfolder(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    stats2 = window.tool_tabs["Stats"].stats2
    assert stats2.folder_combo.count() == 0
    assert not any(a.isEnabled() for a in stats2.folder_menu.actions() if a.text())


# --- Inequality... (#499) ---------------------------------------------------------

def test_inequality_saves_an_iq_array_and_epoch_sets(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    stats2 = tab.stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")  # 20, 22, 24, 26
    asked = []

    def ask(name, set_names):
        asked.append((name, set_names))
        return {"op": ">", "a": 23.0, "binary_output": True,
                "set_name_success": set_names[0], "set_name_failure": set_names[1]}
    stats2._ask_inequality = ask
    stats2.inequality_dialog_action.trigger()
    # The next free Epoch Sets (the fixture has Set0 and Set1)
    assert asked == [("ST_w0_max_dy", ("Set2", "Set3"))]
    subfolder = tab.stats_tool.toolfolder
    assert list(subfolder.data["IQ_ST_w0_max_dy"].nparray) == [0, 0, 1, 1]
    series = folder.dataseries["Record"]
    items = lambda key: [e.name for e in series.epochs.sets.get_items(key)]
    assert (items("Set2"), items("Set3")) == (["E2", "E3"], ["E0", "E1"])
    # Chosen and plotted; a column of the Stats tables, after the ST_ arrays
    assert stats2.selected_array() == "IQ_ST_w0_max_dy"
    assert stats2.selected_plot()[5] == "points"
    assert stats2.tables.columns[-1] == "IQ_ST_w0_max_dy"
    assert _headers(stats2.tables.arrays_table)[-1] == "IQ_max_dy"
    assert stats2.measure_array() == "ST_w0_max_y"  # never an IQ_
    # Summary: its condition and counts
    assert stats2.details.toPlainText().splitlines()[1] == (
        "ST_w0_max_dy > 23: 2 pass, 2 fail (of 4)")
    assert not stats2.inequality_dialog_action.isEnabled()  # not of an IQ_
    # Value or NaN: its fails are NaN, counted above, not listed again
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    stats2._ask_inequality = lambda name, set_names: {"op": ">", "a": 23.0,
                                                      "binary_output": False}
    stats2.inequality_dialog_action.trigger()
    assert "NaN for:" not in stats2.details.toPlainText()
    # The strip lists the new Epoch Sets
    combo = window.selection_strip.combo("epoch_set")
    names = [combo.itemText(i) for i in range(combo.count())]
    assert "Set2" in names and "Set3" in names


def test_inequality_asks_before_replacing_an_epoch_set(qtbot):
    _manager, folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    stats2 = tab.stats2
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    stats2._ask_inequality = lambda name, set_names: {
        "op": "<=<=", "a": 21.0, "b": 25.0, "set_name_success": "Set1"}
    asked = []
    stats2._confirm_replace_sets = lambda names, series: asked.append(names) or False
    stats2.inequality_dialog_action.trigger()
    assert asked == [["Set1"]]
    subfolder = tab.stats_tool.toolfolder
    assert "IQ_ST_w0_max_dy" not in subfolder.data  # cancelled: nothing made
    series = folder.dataseries["Record"]
    assert len(series.epochs.sets.get_items("Set1")) == 4
    stats2._confirm_replace_sets = lambda names, series: True
    stats2.inequality_dialog_action.trigger()
    assert [e.name for e in series.epochs.sets.get_items("Set1")] == ["E1", "E2"]  # 22, 24
    assert list(subfolder.data["IQ_ST_w0_max_dy"].nparray) == [0, 1, 1, 0]


def test_inequality_without_epoch_numbers_has_no_epoch_sets(qtbot):
    _manager, _folder, window = _stats_window(qtbot)
    tab = window.tool_tabs["Stats"]
    tab.run_button.click()
    stats2 = tab.stats2
    tab.stats_tool.toolfolder.data.pop("ST_epoch_num")  # e.g. a Data Set run
    stats2.array_combo.setCurrentText("ST_w0_max_dy")
    asked = []
    stats2._ask_inequality = lambda name, set_names: asked.append(set_names) or None
    stats2.inequality_dialog_action.trigger()
    assert asked == [None]  # the dialog's Epoch Sets greyed out


def test_inequality_dialog(qtbot):
    dialog = InequalityDialog("ST_w0_max_y", ("Set0", "Set1"))
    qtbot.addWidget(dialog)
    dialog.show()
    labels = [dialog.op_combo.itemText(i) for i in range(dialog.op_combo.count())]
    assert labels == ["y > a", "y ≥ a", "y < a", "y ≤ a", "y = a", "y ≠ a",
                      "a < y < b", "a ≤ y ≤ b", "a ≤ y < b", "a < y ≤ b"]
    assert not dialog.b_edit.isVisible()  # b: ranges only
    dialog.a_edit.setText("10")
    assert dialog.kwargs() == {"op": ">", "a": 10.0, "binary_output": True}
    dialog.op_combo.setCurrentIndex(dialog.op_combo.findData("<=<"))
    assert dialog.b_edit.isVisible()
    dialog.b_edit.setText("5")
    dialog.accept()  # b must be more than a: stays open
    assert dialog.result() == 0 and "more than a" in dialog.error_label.text()
    dialog.b_edit.setText("20")
    dialog.output_combo.setCurrentIndex(1)  # value or NaN
    dialog.pass_checkbox.setChecked(True)
    dialog.fail_checkbox.setChecked(True)
    assert dialog.kwargs() == {"op": "<=<", "a": 10.0, "b": 20.0, "binary_output": False,
                               "set_name_success": "Set0", "set_name_failure": "Set1"}
    dialog.fail_edit.setText("set0")
    with pytest.raises(ValueError, match="different"):
        dialog.kwargs()
    dialog.fail_edit.setText("no good")
    with pytest.raises(ValueError, match="valid Epoch Set name"):
        dialog.kwargs()
    dialog.a_edit.setText("x")
    with pytest.raises(ValueError, match="a isn't a number"):
        dialog.kwargs()
    # Without epoch numbers: no Epoch Sets
    other = InequalityDialog("ST_w0_max_y", None, dialog.settings())
    qtbot.addWidget(other)
    assert other.op_combo.currentData() == "<=<"  # the last settings
    other.a_edit.setText("1")
    other.b_edit.setText("2")
    assert "set_name_success" not in other.kwargs()


def test_nan_rows_are_shortened():
    """Summary's "NaN for:": consecutive data names as a range, e.g. RecordA0-3."""
    from pyneuromatic.gui.results_view import runs_text
    assert runs_text(["RecordA0", "RecordA1", "RecordA2", "RecordA3"]) == "RecordA0-3"
    assert runs_text(["RecordA0", "RecordA1", "RecordA3", "RecordA5", "RecordA6",
                      "RecordB7", "RecordB8"]) == "RecordA0-1, RecordA3, RecordA5-6, RecordB7-8"
    assert runs_text(["RecordA2"]) == "RecordA2"
    assert runs_text(["notes", "7"]) == "notes, 7"  # not prefix + channel + number
