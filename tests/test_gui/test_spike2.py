"""Spike2: the Spike tool's saved results (#529)."""

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.spike2_dialogs import ISIDialog, PSTDialog, WaveformsDialog
from pyneuromatic.gui.spike2_view import WAVEFORMS, Spike2Panel, SpikeTables
import pyneuromatic.core.nm_command_history as nmch

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
    panel.select_array("SP_count")  # the row arrays' table (SP_RX: the raster's)
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
    tab.spike2.select_array("SP_count")  # the row arrays' table
    tables = tab.spike2.tables
    rows = [tables.arrays_table.verticalHeaderItem(r).text() for r in range(3)]
    assert rows == ["RecordA0", "E1", "RecordA2"]  # E1 not analysed
    count = tool.toolfolder.data.get("SP_count").nparray
    assert count[0] == 2 and math.isnan(count[1]) and count[2] == 1
    assert tables.arrays_table.item(1, 0).text() == "NaN"


def test_decimals_apply_to_the_tables(app):
    _run_all(app)
    app.spike.spike2.select_array("SP_count")  # the row arrays' table
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
    assert texts == ["Plot", "Plot x: epoch", "Table", "Raster", "PST...", "ISI...",
                     "Waveforms...", "Show in Browser"]
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


# --- PST..., ISI... and Waveforms... (#530) ------------------------------------


def _answer(monkeypatch, panel, kwargs):
    """The analysis dialogs answer *kwargs* (None: cancelled); the kinds
    asked are listed."""
    asked = []

    def ask(kind, toolfolder, times, units):
        asked.append((kind, toolfolder.name, list(times), units))
        return kwargs

    monkeypatch.setattr(panel, "_ask_analysis", ask)
    return asked


def _last_command():
    return nmch.get_command_history().buffer[-1]["command"]


def test_pst(app, monkeypatch):
    _run_all(app)
    panel = app.spike.spike2
    asked = _answer(monkeypatch, panel, {"binwidth": 20.0})
    panel.analysis_actions["pst"].trigger()
    assert asked == [("pst", "Spike_Record_A_All_0",
                      pytest.approx([9.5, 39.5, 9.5, 39.5, 69.5, 39.5]), "ms")]
    assert "H_SP_PST" in _items(panel.array_combo)
    assert panel.selected_array() == "H_SP_PST"
    title, x, y = _plotted(app)
    assert title == "Spike_Record_A_All_0: H_SP_PST"
    # Bins of 20 ms on its multiples: 9.5 (x2), 39.5 (x3), 69.5
    assert list(x) == [0, 20, 40, 60, 80] and list(y) == [2, 3, 0, 1]
    assert _last_command() == (
        "nm.spike.pst(binwidth=20.0, "
        "toolfolder=nm.folders['Demo'].toolfolders['Spike_Record_A_All_0'])")


def test_pst_without_limits(app, monkeypatch):
    """-inf to inf (typed as in Spike1's x range): every spike, as no range."""
    _run_all(app)
    panel = app.spike.spike2
    _answer(monkeypatch, panel, {"binwidth": 20.0, "xbgn": -math.inf, "xend": math.inf,
                                 "output_mode": "rate"})
    panel.analysis_actions["pst"].trigger()
    assert panel.selected_array() == "H_SP_PST"
    _title, x, y = _plotted(app)
    # 2, 3, 0 and 1 spikes in 3 epochs' 20 ms bins, in Hz
    assert list(x) == [0, 20, 40, 60, 80]
    assert list(y) == pytest.approx([1000 * n / (3 * 20) for n in (2, 3, 0, 1)])


def test_isi(app, monkeypatch):
    _run_all(app)
    panel = app.spike.spike2
    _answer(monkeypatch, panel, {"binwidth": 10.0, "min_isi": 20.0, "max_isi": 40.0,
                                 "output_mode": "probability"})
    panel.analysis_actions["isi"].trigger()
    assert panel.selected_array() == "H_SP_ISI"
    _title, x, y = _plotted(app)
    assert list(x)[:3] == [20, 30, 40] and list(y)[:2] == [0, 1]  # every ISI is 30 ms
    assert sum(y) == 1
    assert "output_mode='probability'" in _last_command()


def test_waveforms(app, monkeypatch):
    _run_all(app)
    panel = app.spike.spike2
    _answer(monkeypatch, panel, {"pre": 2.0, "post": 3.0, "align": "spike"})
    panel.analysis_actions["waveforms"].trigger()
    assert _items(panel.array_combo)[-1] == WAVEFORMS
    assert panel.selected_array() == WAVEFORMS
    widget = app.spike.plot_view.plot_widget
    items = widget.listDataItems()
    assert len(items) == 2  # the waveforms, as one line, and their mean
    x, y = items[0].getData()  # a NaN after each waveform
    assert len(x) == 6 * (5 + 1) and np.isnan(x[5]) and np.isnan(y[5])
    assert list(x[:5]) == [-2, -1, 0, 1, 2] and list(y[:5]) == [0, 0, 1, 1, 1]
    _x, mean = items[1].getData()
    assert len(mean) == 5
    assert widget.plotItem.titleLabel.text == "Spike_Record_A_All_0: SPK_ (6 waveforms)"
    assert not panel.epoch_x_action.isEnabled()
    lines = panel.details.toPlainText().splitlines()
    assert lines[0] == "SPK_  6 spike waveforms, SPK_A0 to SPK_A5 (the SPK_ data series)"
    toolfolder = panel.toolfolder()
    assert "SP_RX" in toolfolder.data and "SPK_" in toolfolder.dataseries
    assert _last_command().startswith("nm.spike.extract_spike_waveforms(2.0, 3.0, ")
    # Again: replaced, not added to
    _answer(monkeypatch, panel, {"pre": 1.0, "post": 1.0, "align": "spike"})
    panel.analysis_actions["waveforms"].trigger()
    assert len([n for n in toolfolder.data.keys() if n.startswith("SPK_")]) == 6


def test_cancelled_or_failed(app, monkeypatch):
    _run_all(app)
    panel = app.spike.spike2
    _answer(monkeypatch, panel, None)  # cancelled: nothing
    panel.analysis_actions["pst"].trigger()
    assert "H_SP_PST" not in _items(panel.array_combo)
    # A window wider than the recordings: no waveforms
    _answer(monkeypatch, panel, {"pre": 500.0, "post": 500.0})
    panel.analysis_actions["waveforms"].trigger()
    assert WAVEFORMS not in _items(panel.array_combo)
    assert panel.message_label.text() == (
        "No waveforms of Spike_Record_A_All_0: no spike's window fits its recording.")
    assert not panel.message_label.isHidden()


def test_analyses_need_spikes(app):
    panel = app.spike.spike2
    assert not any(a.isEnabled() for a in panel.analysis_actions.values())
    _run_all(app)
    assert all(a.isEnabled() for a in panel.analysis_actions.values())


def test_pst_dialog(qtbot):
    times = np.array([9.5, 39.5, 9.5, 39.5, 69.5, 39.5])
    dialog = PSTDialog("Spike_Record_A_All_0", times, "ms")
    qtbot.addWidget(dialog)
    width = dialog.kwargs()["binwidth"]
    assert width in (10.0, 20.0, 50.0)  # numpy's auto width, rounded
    assert dialog.kwargs() == {"binwidth": width, "output_mode": "count"}
    assert not dialog.xbgn_edit.isEnabled()
    dialog.range_checkbox.setChecked(True)
    assert dialog.xbgn_edit.text() == "9.5" and dialog.xend_edit.text() == "69.5"
    dialog.xend_edit.setText("5")
    dialog.accept()
    assert dialog.error_label.text() == "The x range's max must be more than its min."
    dialog.xbgn_edit.setText("-inf")  # no limit, as Spike1's x range
    dialog.xend_edit.setText("inf")
    assert dialog.kwargs()["xbgn"] == -math.inf and dialog.kwargs()["xend"] == math.inf
    dialog.xbgn_edit.setText("9.5")
    dialog.xend_edit.setText("100")
    dialog.output_combo.setCurrentIndex(dialog.output_combo.findData("rate"))
    dialog.width_edit.setText("0")
    dialog.accept()
    assert dialog.error_label.text() == "The bin width must be more than 0."
    dialog.width_edit.setText("2.5")
    kwargs = dialog.kwargs()
    assert kwargs == {"binwidth": 2.5, "xbgn": 9.5, "xend": 100.0, "output_mode": "rate"}
    # Shown again next time
    again = PSTDialog("Spike_Record_A_All_1", np.array([0.0, 1.0]), "ms", dialog.settings())
    qtbot.addWidget(again)
    assert again.kwargs() == kwargs


def test_isi_dialog(qtbot):
    dialog = ISIDialog("S", np.array([9.5, 39.5, 69.5]), np.array([30.0, 30.0]), "ms")
    qtbot.addWidget(dialog)
    dialog.isi_checkbox.setChecked(True)
    dialog.min_edit.setText("1")
    assert dialog.kwargs() == {"binwidth": 1.0, "min_isi": 1.0, "max_isi": 30.0,
                               "output_mode": "count"}  # all one ISI: width 1


def test_waveforms_dialog(qtbot):
    dialog = WaveformsDialog("S", np.array([9.5, 69.5]), "ms")
    qtbot.addWidget(dialog)
    assert dialog.kwargs() == {"pre": 1.0, "post": 2.0, "clip_to_next_spike": False,
                               "edge": "skip", "align": "spike"}
    assert not dialog.group_checkbox.isChecked()
    dialog.group_checkbox.setChecked(True)
    assert dialog.kwargs()["group_by_epoch"] is True
    again = WaveformsDialog("S", np.array([9.5, 69.5]), "ms", dialog.settings())
    qtbot.addWidget(again)
    assert again.group_checkbox.isChecked()
    dialog.group_checkbox.setChecked(False)
    seconds = WaveformsDialog("S", np.array([0.0, 1.0]), "s")
    qtbot.addWidget(seconds)
    assert (seconds.kwargs()["pre"], seconds.kwargs()["post"]) == (0.001, 0.002)
    dialog.pre_edit.setText("0")
    dialog.accept()
    assert dialog.error_label.text() == "The times before and after must be more than 0."


@pytest.mark.parametrize("values, width", [
    (np.linspace(0, 100, 1000), 10.0),  # auto: 10
    (np.linspace(0, 0.1, 1000), 0.01),
    (np.linspace(0, 60, 1000), 5.0),  # auto: 6
    (np.array([3.0, 3.0]), 1.0),  # no spread
    (np.array([]), 1.0),
])
def test_nice_width(values, width):
    from pyneuromatic.gui.spike2_dialogs import nice_width
    assert nice_width(values) == pytest.approx(width)



@pytest.mark.parametrize("text, spikes", [
    ("", None), ("all", None), ("0-99", (0, 99)), (" 100 - ", (100, None)),
    ("5", (5, 5)), ("0-0", (0, 0)),
])
def test_parse_spikes(text, spikes):
    from pyneuromatic.gui.spike2_dialogs import parse_spikes, spikes_text
    assert parse_spikes(text) == spikes
    assert parse_spikes(spikes_text(spikes)) == spikes  # shown again as typed


@pytest.mark.parametrize("text", ["x", "9-3", "-5", "1-2-3", "1.5"])
def test_parse_spikes_rejects(text):
    from pyneuromatic.gui.spike2_dialogs import parse_spikes
    with pytest.raises(ValueError):
        parse_spikes(text)


def test_waveforms_dialog_spikes(qtbot):
    dialog = WaveformsDialog("S", np.array([9.5, 69.5]), "ms")
    qtbot.addWidget(dialog)
    assert "spikes" not in dialog.kwargs()  # blank: all
    dialog.spikes_edit.setText("0-99")
    assert dialog.kwargs()["spikes"] == (0, 99)
    again = WaveformsDialog("S", np.array([9.5, 69.5]), "ms", dialog.settings())
    qtbot.addWidget(again)
    assert again.spikes_edit.text() == "0-99"
    dialog.spikes_edit.setText("9-3")
    dialog.accept()
    assert "first can't be after the last" in dialog.error_label.text()



def test_waveforms_dialog_counts_the_spikes(qtbot):
    """The numbers there are to choose from, e.g. 0-5 (6 spikes); within
    Spike times when it's on."""
    times = np.array([9.5, 39.5, 9.5, 39.5, 69.5, 39.5])
    dialog = WaveformsDialog("S", times, "ms")
    qtbot.addWidget(dialog)
    assert dialog.spikes_label.text() == "0-5 (6 spikes)"
    dialog.times_checkbox.setChecked(True)
    dialog.xbgn_edit.setText("30")
    dialog.xend_edit.setText("50")
    assert dialog.spikes_label.text() == "0-2 (3 spikes)"
    dialog.xend_edit.setText("35")
    assert dialog.spikes_label.text() == "(0 spikes)"
    dialog.xend_edit.setText("")  # being typed: all
    assert dialog.spikes_label.text() == "0-5 (6 spikes)"
    dialog.times_checkbox.setChecked(False)
    assert dialog.spike_count() == 6


def test_the_raster_table(app):
    """SP_RX (or SP_RY) chosen: the Table shows them, a row per spike."""
    _run_all(app)
    panel, tables = app.spike.spike2, app.spike.spike2.tables
    assert panel.selected_array() == "SP_RX"
    assert tables.columns == ["SP_RX", "SP_RY"]
    headers = [tables.arrays_table.horizontalHeaderItem(c).text() for c in range(2)]
    assert headers == ["RX", "RY"]
    assert tables.arrays_table.rowCount() == 6
    assert [tables.arrays_table.item(r, 1).text() for r in range(6)] == [
        "0.000", "0.000", "1.000", "1.000", "1.000", "2.000"]
    assert tables.title_label.text().endswith("raster (a row per spike)")
    # The Array list is the same whichever table shows
    assert _items(panel.array_combo) == ["SP_RX", "SP_count", "SP_rate", "SP_first", "SP_isi"]
    panel.select_array("SP_rate")
    assert tables.columns[0] == "SP_count" and tables.selected_array() == "SP_rate"


def test_the_waveforms_table(app, monkeypatch):
    """SPK_ chosen: the Table shows the waveforms, a column each, a row per
    point named by its time; a column clicked leaves SPK_ chosen."""
    _run_all(app)
    panel, tables = app.spike.spike2, app.spike.spike2.tables
    _answer(monkeypatch, panel, {"pre": 2.0, "post": 3.0, "align": "spike"})
    panel.analysis_actions["waveforms"].trigger()
    assert panel.selected_array() == WAVEFORMS
    assert tables.columns == [f"SPK_A{i}" for i in range(6)]
    headers = [tables.arrays_table.horizontalHeaderItem(c).text() for c in range(6)]
    assert headers == ["A0", "A1", "A2", "A3", "A4", "A5"]
    rows = [tables.arrays_table.verticalHeaderItem(r).text() for r in range(5)]
    assert rows == ["-2.000", "-1.000", "0.000", "1.000", "2.000"]
    assert [tables.arrays_table.item(r, 0).text() for r in range(5)] == [
        "0.000", "0.000", "1.000", "1.000", "1.000"]
    tables.arrays_table.setCurrentCell(0, 3)
    assert panel.selected_array() == WAVEFORMS  # SPK_A3 isn't in the Array list


def test_many_columns_are_sized_from_a_few(qtbot):
    """Over MANY_COLUMNS columns (e.g. 1921 waveforms), each is as wide as
    the first few's values, or its header."""
    from pyneuromatic.tools.nm_tool_folder import NMToolFolder
    tables = SpikeTables()
    qtbot.addWidget(tables)
    folder = NMToolFolder(name="Spike_X_0")
    for i in range(SpikeTables.MANY_COLUMNS + 10):
        folder.data.new(f"SPK_A{i}", nparray=np.full(4, -60.0 - i % 3))
    tables.show_results(folder, "waveforms")
    widths = {tables.arrays_table.columnWidth(c) for c in range(len(tables.columns))}
    assert len(widths) == 1 and widths.pop() > 0
