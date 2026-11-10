"""Tests for Stats2's Results: Show All epochs, every epoch's saved result (#475)."""

import re

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.results_view import StatsTables, epoch_result_lines
from pyneuromatic.gui.selection_model import ALL_EPOCHS

pytestmark = pytest.mark.gui

N = 3


def _app(qtbot, func, epoch_set=ALL_EPOCHS, bsln=True):
    """Channel A x 3 epochs: a peak of 15, 16, 17 above -70 at x 80 (ms, mV);
    Stats w0 *func* over 60..120, baseline mean 0..40; run; E1 shown."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=N, n_points=200)
    x = np.arange(200.0)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = -70.0 + (15.0 + i) * np.exp(-((x - 80.0) / 12.0) ** 2)
        d.xscale.units, d.yscale.units = "ms", "mV"
    series.epochs.sets.add("Set1", ["E0", "E1"])
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E1"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(epoch_set=epoch_set)
    tab = window.tool_tabs["Stats"]
    manager.stats.windows["w0"].func = func
    tab._load_selected_window()
    for edit, text in ((tab.xbgn_edit, "60"), (tab.xend_edit, "120")):
        edit.setText(text)
        edit.editingFinished.emit()
    if bsln:
        tab.bsln_checkbox.setChecked(True)
        for edit, text in ((tab.bsln_xbgn_edit, "0"), (tab.bsln_xend_edit, "40")):
            edit.setText(text)
            edit.editingFinished.emit()
    tab.run_button.click()
    window.stats, window.series = tab, series
    return window


def _epochs(app):
    stats2 = app.stats.stats2
    stats2.results_combo.setCurrentIndex(stats2.results_combo.findData("epochs"))
    return stats2.details.toPlainText().splitlines()


def test_every_epoch_a_line(qtbot):
    app = _app(qtbot, "max")
    assert _epochs(app) == [
        "Stats_Record_A_All_0  w0 max  (x 60 to 120)  bsln mean (x 0 to 40)"
        "  (values in mV, x in ms)",
        "RecordA0: bsln mean = -70; max = -55 at 80; Δ = 15",
        "RecordA1: bsln mean = -70; max = -54 at 80; Δ = 16",
        "RecordA2: bsln mean = -70; max = -53 at 80; Δ = 17",
    ]


@pytest.mark.parametrize("func, line", [
    ({"name": "risetime+", "p0": 10, "p1": 90},
     "RecordA1: bsln mean = -70; max = -54 at 80; Δ = 16; "
     "level+ 10% = -68.4 at 61.774; level+ 90% = -55.6 at 76.1158; "
     "risetime+ = 14.3418"),
    ("mean+std", "RecordA1: bsln mean = -70; mean = -64.4647, ±std 5.69728; "
                 "Δ = 5.53524"),
    ({"name": "level+", "ylevel": -60},
     "RecordA1: bsln mean = -70; level+ = -60 at 71.7734; Δ = 9.99999"),
])
def test_each_kind_of_result(qtbot, func, line):
    assert _epochs(_app(qtbot, func))[2] == line


def test_slope_without_baseline(qtbot):
    assert _epochs(_app(qtbot, "slope", bsln=False)) [:2] == [
        "Stats_Record_A_All_0  w0 slope  (x 60 to 120)  (values in mV/ms, x in ms)",
        "RecordA0: slope = -0.164321, intercept -50.0218 mV",  # not in the header's units
    ]


def test_only_the_epochs_computed(qtbot):
    app = _app(qtbot, "max", epoch_set="Set1")  # E0, E1
    assert [line.split(":")[0] for line in _epochs(app)[1:]] == ["RecordA0", "RecordA1"]


def test_selectable_to_copy_and_no_wrapping(qtbot):
    app = _app(qtbot, "max")
    _epochs(app)
    details = app.stats.stats2.details
    assert details.isReadOnly()  # selectable
    assert details.lineWrapMode() == details.LineWrapMode.NoWrap  # a line per epoch


def test_summary_is_still_there_and_digits_apply(qtbot):
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    assert stats2.results_combo.currentData() == "summary"  # as before, by default
    assert stats2.details.toPlainText().startswith("ST_w0_max_y")
    app.stats.set_config_option("digits", 3)
    assert StatsTables.digits == 3
    toolfolder = stats2.toolfolder()
    assert epoch_result_lines(toolfolder, "w0", 1, digits=2)[1] == "max = -54 mV at 80 ms   Δ = 16 mV"
    stats2.results_combo.setCurrentIndex(stats2.results_combo.findData("summary"))
    assert stats2.details.lineWrapMode() == stats2.details.LineWrapMode.WidgetWidth


def test_stats1_results_after_a_run_and_no_history_checkbox(qtbot):
    app = _app(qtbot, "max")
    tab = app.stats
    assert tab.details.toPlainText().splitlines()[-1] == "max = -54 at 80  (Δ = 16)"
    assert not hasattr(tab, "output_history_checkbox")
    assert hasattr(app.selection_model.manager.stats, "results_to_history")  # for scripts



@pytest.mark.parametrize("func", [
    "max", "mean+std", {"name": "risetime+", "p0": 10, "p1": 90}, "fwhm+",
    {"name": "decaytime+", "p0": 36.7879}, {"name": "level+", "ylevel": -60},
])
def test_all_epochs_matches_stats1(qtbot, func):
    """From the saved arrays, the same text as Stats1's Results (#476)."""
    app = _app(qtbot, func)
    stats1 = app.stats.details.toPlainText().splitlines()[1:]  # E1 (shown), a preview
    line = _epochs(app)[2]  # RecordA1
    # Both give their units once, on the top line
    # Stats1's "  (Δ = 16)" is All epochs' "; Δ = 16"
    steps = [re.sub(r"  \(Δ = ([^)]*)\)", r"; Δ = \1", step.split("  (x ")[0])
             for step in stats1]
    assert line == "RecordA1: " + "; ".join(steps)  # its bsln range is in the header


def test_plot_x_is_the_epoch_numbers(qtbot):
    app = _app(qtbot, "max", epoch_set="Set1")  # E0, E1
    stats2 = app.stats.stats2
    title, y, x, xlabel, _ylabel, _style = stats2.selected_plot()
    assert list(x) == [0, 1] and xlabel == "epoch"
    stats2.epoch_x_action.setChecked(False)
    assert stats2.selected_plot()[2] is None and stats2.selected_plot()[3] == "data index"


def test_an_older_subfolder_still_reads(qtbot):
    """Saved before #476: ST_w0_data (no ST_data_names or ST_epoch_num)."""
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    folder = stats2._folder
    old = folder.toolfolders.new("Stats_Old_0")
    old.data.new("ST_w0_data", nparray=np.array(["nm.Demo.RecordA0", "nm.Demo.RecordA1"],
                                                 dtype=object))
    peak = old.data.new("ST_w0_max_y", nparray=np.array([-55.0, -54.0]),
                        yscale={"units": "mV"})
    peak.notes.add("NMStats(win=w0, func=max, id=main, xbgn=60.0, xend=120.0, n=2)")
    stats2.refresh(select="Stats_Old_0")
    assert stats2._windows(old) == ["w0"]
    assert _epochs(app) == ["Stats_Old_0  w0 max  (x 60 to 120)  (values in mV, x in ms)",
                            "RecordA0: max = -55", "RecordA1: max = -54"]
    title, y, x, xlabel, _yl, _style = stats2.selected_plot()
    assert x is None and xlabel == "data index"  # no epoch numbers saved
    assert not stats2.epoch_x_action.isEnabled()


def test_epoch_numbers_with_the_index_and_count_arrays(qtbot):
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    assert "ST_epoch_num" not in stats2.tables.columns
    app.stats.set_config_option("detail_arrays", True)  # Settings › Stats (#496)
    assert stats2.tables.columns[0] == "ST_epoch_num"
    assert stats2.measure_array() == "ST_w0_max_y"  # never the rows' epochs


def test_the_epoch_column_is_headed_epoch(qtbot):
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    app.stats.set_config_option("detail_arrays", True)  # Settings › Stats (#496)
    table = stats2.tables.arrays_table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers[0] == "epoch"  # not "ch_num" (ST_epoch_num cut as a window's array)
    assert table.horizontalHeaderItem(0).toolTip() == "ST_epoch_num"
    assert [table.item(r, 0).text() for r in range(table.rowCount())] == ["0", "1", "2"]
    assert "max_y" in headers


def test_the_chosen_arrays_header_is_bold_in_the_summary_too(qtbot):
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    summary = stats2.tables.summary_table

    def bold():
        return [summary.horizontalHeaderItem(c).text()
                for c in range(summary.columnCount())
                if summary.horizontalHeaderItem(c).font().bold()]

    assert bold() == ["max_y"]  # the measurement, after the run
    stats2.array_combo.setCurrentText("ST_w0_max_x")
    assert bold() == ["max_x"]
    stats2.tables.arrays_table.setCurrentCell(0, 0)  # a cell in the bsln column
    assert bold() == ["bsln_y"]


def test_plot_x_axis_shows_whole_numbers(qtbot):
    """Points against a row index or epoch number: whole-number ticks, no
    minor ticks; a histogram's bin edges keep the usual axis."""
    from pyneuromatic.gui.context_views import ValuesPlot
    plot = ValuesPlot()
    qtbot.addWidget(plot)
    plot.resize(600, 400)
    plot.show()
    axis = plot.x_axis
    for n, step in ((3, 1.0), (9, 2.0), (40, 10.0)):
        plot.show_values("t", np.arange(float(n)), None, "data index", "y", "points")
        low, high = axis.range
        assert axis.whole_numbers
        assert axis.tickSpacing(low, high, 500) == [(step, 0.0)]  # one level: no minor ticks
        values = axis.tickValues(low, high, 500)[0][1]
        assert all(v == int(v) for v in values)
    plot.show_values("e", np.arange(3.0), np.array([0, 2, 4]), "epoch", "y", "points")
    assert axis.whole_numbers
    plot.show_values("h", np.arange(3.0), np.array([0.0, 0.5, 1.0, 1.5]), "x", "count",
                     "histogram")
    assert not axis.whole_numbers


def test_stats1_x_ranges_have_x_units(qtbot):
    app = _app(qtbot, "max")
    lines = app.stats.details.toPlainText().splitlines()
    # Units once, on the top line, as Stats2's
    assert lines[0] == "RecordA1  w0 max  (x 60 to 120)  (values in mV, x in ms)"
    assert lines[1] == "bsln mean = -70  (x 0 to 40)"


def test_an_unbounded_range_shows_the_traces_limits(qtbot):
    """-inf / inf: where the trace starts or ends (x 0 to 199 ms), in Stats1
    and in Stats2's header."""
    app = _app(qtbot, "max", bsln=False)
    tab = app.stats
    for edit in (tab.xbgn_edit, tab.xend_edit):
        edit.setText("")  # -inf, inf
        edit.editingFinished.emit()
    assert tab.details.toPlainText().splitlines()[0] == (
        "RecordA1  w0 max  (x 0 to 199)  (values in mV, x in ms)")
    tab.run_button.click()
    assert _epochs(app)[0] == "Stats_Record_A_All_1  w0 max  (x 0 to 199)  (values in mV, x in ms)"



def test_a_baselines_spread_is_shown(qtbot):
    """bsln mean ± std: its std too, in Stats1 and Stats2."""
    app = _app(qtbot, "max")
    tab = app.stats
    tab.bsln_combo.setCurrentIndex(tab.bsln_combo.findData("mean+std"))
    import re
    stats1 = tab.details.toPlainText().splitlines()[1]
    assert re.match(r"bsln mean = -70, \u00b1std \S+  \(x 0 to 40\)$", stats1)
    tab.run_button.click()
    assert re.match(r"RecordA0: bsln mean = -70, \u00b1std \S+; max", _epochs(app)[1])


def test_summary_gives_values_and_x_units(qtbot):
    app = _app(qtbot, "max")
    stats2 = app.stats.stats2
    assert stats2.details.toPlainText().splitlines()[0] == (
        "ST_w0_max_y  (values in mV, x in ms)")
    stats2.array_combo.setCurrentText("ST_w0_max_x")  # an x array: in ms
    assert stats2.details.toPlainText().splitlines()[0] == (
        "ST_w0_max_x  (values in ms, x in ms)")
