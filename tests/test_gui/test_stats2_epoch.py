"""Tests for Stats2's Results: Show All epochs, every epoch's saved result (#475)."""

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
    tab.output_arrays_checkbox.setChecked(True)
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
        "Stats_Record_A_All_0  w0 max  (x 60 to 120)  bsln mean (x 0 to 40)",
        "RecordA0: bsln mean = -70 mV; max = -55 mV at 80 ms   Δ = 15 mV",
        "RecordA1: bsln mean = -70 mV; max = -54 mV at 80 ms   Δ = 16 mV",
        "RecordA2: bsln mean = -70 mV; max = -53 mV at 80 ms   Δ = 17 mV",
    ]


@pytest.mark.parametrize("func, line", [
    ({"name": "risetime+", "p0": 10, "p1": 90},
     "RecordA1: bsln mean = -70 mV; peak = -54 mV at 80 ms   Δ = 16 mV; p0 at 61.774 ms; "
     "p1 at 76.1158 ms; risetime+ = 14.3418 ms"),
    ("mean+std", "RecordA1: bsln mean = -70 mV; mean+std = -64.4647 mV, std 5.69728 mV   "
                 "Δ = 5.53524 mV"),
    ({"name": "level+", "ylevel": -60},
     "RecordA1: bsln mean = -70 mV; level+ at 71.7734 ms"),  # its level isn't saved
])
def test_each_kind_of_result(qtbot, func, line):
    assert _epochs(_app(qtbot, func))[2] == line


def test_slope_without_baseline(qtbot):
    assert _epochs(_app(qtbot, "slope", bsln=False)) [:2] == [
        "Stats_Record_A_All_0  w0 slope  (x 60 to 120)",
        "RecordA0: slope = -0.164321 mV/ms, intercept -50.0218 mV",
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
    assert tab.details.toPlainText().splitlines()[-1] == "max = -54 mV at 80 ms   Δ = 16 mV"
    assert not hasattr(tab, "output_history_checkbox")
    assert hasattr(app.selection_model.manager.stats, "results_to_history")  # for scripts
