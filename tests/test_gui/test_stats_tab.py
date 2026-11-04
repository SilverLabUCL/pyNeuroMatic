"""Tests for the Stats1 tool tab integration."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.stats_tab import StatsToolTab

pytestmark = pytest.mark.gui


def _main_result(manager, window="w0", index=0):
    """Main (non-baseline) result of the last Stats run for data *index*."""
    results = manager.stats.results[window][index]
    return next(r for r in results if r["id"] != "bsln")


def test_stats_tab_runs_selected_trace_and_shows_backend_results(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new(
        "RecordA0",
        nparray=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
        xscale={"start": 0.0, "delta": 1.0, "label": "Time", "units": "ms"},
        yscale={"label": "Vm", "units": "mV"},
    )
    manager.select_value_set(data)

    window = NMAppWindow(manager)
    qtbot.addWidget(window)

    assert isinstance(window.tool_tabs["Stats"], StatsToolTab)
    assert "stats" in manager.enabled_tools()
    assert window.tool_workspace.widget(window.tool_rail.index_for_name("Stats")) is window.tool_tabs["Stats"]

    tab = window.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    tab.xbgn_edit.setText("1")
    tab.xend_edit.setText("3")
    tab.run_button.click()

    assert tab.status_label.text() == "Complete"
    assert _main_result(manager)["s"] == 20.0
    assert _main_result(manager)["sunits"] == "mV"
    assert manager.stats.windows["w0"].xbgn == 1.0
    assert manager.stats.windows["w0"].xend == 3.0


def test_window_combo_creates_and_selects_new_window(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    new_window_index = tab.window_combo.findData(tab.NEW_WINDOW_ACTION)
    assert new_window_index >= 0
    tab.window_combo.setCurrentIndex(new_window_index)

    assert tab.window_combo.currentText() == "w1"
    assert manager.stats.windows.selected_name == "w1"
    assert tab.window_combo.findData(tab.NEW_WINDOW_ACTION) >= 0


def test_run_uses_selected_epoch_set_instead_of_current_trace(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch, value in enumerate((10.0, 20.0, 30.0)):
        folder.data.new(
            f"RecordA{epoch}",
            nparray=np.array([value, value + 2.0]),
            xscale={"start": 0.0, "delta": 1.0},
        )
    dataseries = folder.sync_dataseries("Record", select=True)
    dataseries.epochs.sets.add("Chosen", ["E1", "E2"])
    manager.select_keys = {
        "folder": folder.name,
        "dataseries": dataseries.name,
        "channel": "A",
        "epoch": "E0",
    }
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    window.selection_model.update(epoch_set="Chosen")
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()

    window_results = tab.stats_tool.results["w0"]
    results = [result for data_results in window_results for result in data_results]
    assert [result["data"] for result in results] == [
        "nm.Demo.RecordA1",
        "nm.Demo.RecordA2",
    ]
    assert [result["s"] for result in results] == [21.0, 31.0]


def test_stats_tab_preserves_multiple_window_settings(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    tab.window_enabled.setChecked(False)
    tab.window_combo.setCurrentIndex(tab.window_combo.findData(tab.NEW_WINDOW_ACTION))
    tab.measurement_combo.setCurrentText("max")
    tab.xbgn_edit.setText("2")
    tab.xbgn_edit.editingFinished.emit()
    tab.xend_edit.setText("8")
    tab.xend_edit.editingFinished.emit()

    assert tab.window_combo.currentText() == "w1"
    assert manager.stats.windows["w1"].func["name"] == "max"
    assert manager.stats.windows["w1"].xbgn == 2.0
    assert manager.stats.windows["w1"].xend == 8.0
    assert manager.stats.windows["w0"].on is False

    tab.window_combo.setCurrentText("w0")

    assert manager.stats.windows.selected_name == "w0"
    assert tab.measurement_combo.currentData() is None
    assert tab.window_enabled.isChecked() is False


def test_stats_tab_uses_analysis_channel_not_plot_channel(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    folder.data.new(
        "RecordA0",
        nparray=np.array([1.0, 3.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder.data.new(
        "RecordB0",
        nparray=np.array([10.0, 30.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    dataseries = folder.sync_dataseries("Record", select=True)
    manager.select_keys = {
        "folder": folder.name,
        "dataseries": dataseries.name,
        "channel": "A",
        "epoch": "E0",
    }

    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    window.plot_widget.view_tabs.setCurrentIndex(1)
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()

    assert manager.select_values["channel"].name == "A"
    assert window.plot_widget.current_trace is dataseries.get_data("B", "E0")
    assert _main_result(manager)["s"] == 2.0


def test_stats_output_options_control_backend_sinks(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.array([1.0, 3.0]))
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    tab.measurement_combo.setCurrentText("mean")
    tab.output_history_checkbox.setChecked(True)
    tab.output_cache_checkbox.setChecked(False)
    tab.output_arrays_checkbox.setChecked(True)
    tab.run_button.click()

    assert manager.stats.results_to_history is True
    assert manager.stats.results_to_cache is False
    assert manager.stats.results_to_numpy is True
    assert "Stats_0" in folder.toolfolders


def test_options_are_in_settings_not_the_panel(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    # No Config control in the panel (#432)
    assert not hasattr(tab, "config_checkbox")
    assert "Config" not in [box.text() for box in tab.findChildren(QtWidgets.QCheckBox)]
    # Settings › Stats edits the tool
    boxes = window.settings_dialog().tool_checkboxes["Stats"]
    boxes["ignore_nans"].setChecked(False)
    assert manager.stats.ignore_nans is False
    assert manager.stats.config.ignore_nans is False

def _param_visible(tab, key):
    return not tab.param_row.isHidden() and not tab.param_edits[key].isHidden()


def test_param_row_shows_only_inputs_for_current_measurement(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    assert tab.param_row.isHidden()
    tab.measurement_combo.setCurrentText("mean@max")
    assert _param_visible(tab, "n_mean")
    assert not _param_visible(tab, "ylevel")
    tab.measurement_combo.setCurrentText("level+")
    assert _param_visible(tab, "ylevel")
    assert not _param_visible(tab, "n_mean")
    tab.measurement_combo.setCurrentText("max")
    assert tab.param_row.isHidden()
    assert manager.stats.windows["w0"].func == {"name": "max"}


def test_mean_at_max_uses_n_mean_parameter(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new(
        "RecordA0",
        nparray=np.array([0.0, 1.0, 9.0, 10.0, 8.0, 2.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    tab.measurement_combo.setCurrentText("mean@max")
    tab.param_edits["n_mean"].setText("3")
    tab.param_edits["n_mean"].editingFinished.emit()
    assert manager.stats.windows["w0"].func == {"name": "mean@max", "n_mean": 3}

    tab.run_button.click()
    assert tab.status_label.text() == "Complete"
    assert _main_result(manager)["s"] == 9.0


def test_level_uses_ylevel_parameter_and_rejects_bad_input(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    tab.measurement_combo.setCurrentText("level+")
    tab.param_edits["ylevel"].setText("-20.5")
    tab.param_edits["ylevel"].editingFinished.emit()
    assert manager.stats.windows["w0"].func == {"name": "level+", "ylevel": -20.5}

    tab.param_edits["ylevel"].setText("abc")
    tab.param_edits["ylevel"].editingFinished.emit()
    assert "Y level must be a number" in tab.status_label.text()
    tab.param_edits["ylevel"].setText("nan")
    tab.param_edits["ylevel"].editingFinished.emit()
    assert "Invalid level+ parameter" in tab.status_label.text()
    assert manager.stats.windows["w0"].func["ylevel"] == -20.5


def test_param_values_restored_per_window(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    tab.measurement_combo.setCurrentText("mean@min")
    tab.param_edits["n_mean"].setText("7")
    tab.param_edits["n_mean"].editingFinished.emit()
    tab.window_combo.setCurrentIndex(tab.window_combo.findData(tab.NEW_WINDOW_ACTION))
    tab.measurement_combo.setCurrentText("mean@max")
    tab.param_edits["n_mean"].setText("2")
    tab.param_edits["n_mean"].editingFinished.emit()

    tab.window_combo.setCurrentText("w0")
    assert tab.measurement_combo.currentText() == "mean@min"
    assert tab.param_edits["n_mean"].text() == "7"


def _step_trace_window(qtbot):
    """App window with a trace resting at -70 that steps to -20 at x=10..14."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new(
        "RecordA0",
        nparray=np.concatenate(
            [np.full(10, -70.0), np.full(5, -20.0), np.full(10, -70.0)]
        ),
        xscale={"start": 0.0, "delta": 1.0},
    )
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    return manager, window, window.tool_tabs["Stats"]


def _set_text(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def test_optional_baseline_adds_delta_and_prefills_range(qtbot):
    manager, _window, tab = _step_trace_window(qtbot)
    tab.measurement_combo.setCurrentText("max")
    _set_text(tab.xbgn_edit, "10")
    _set_text(tab.xend_edit, "20")

    assert tab.bsln_checkbox.isEnabled()
    assert not tab.bsln_combo.isEnabled()
    tab.bsln_checkbox.setChecked(True)

    assert tab.bsln_xbgn_edit.text() == "0"
    assert tab.bsln_xend_edit.text() == "10"
    win = manager.stats.windows["w0"]
    assert win.bsln_on is True
    assert win.bsln_func == {"name": "mean"}
    assert (win.bsln_xbgn, win.bsln_xend) == (0.0, 10.0)

    _set_text(tab.bsln_xend_edit, "9")
    tab.run_button.click()
    assert tab.status_label.text() == "Complete"
    assert _main_result(manager)["Δs"] == 50.0


def test_baseline_disabled_for_non_y_measurements(qtbot):
    manager, _window, tab = _step_trace_window(qtbot)
    tab.measurement_combo.setCurrentText("mean")
    tab.bsln_checkbox.setChecked(True)
    assert manager.stats.windows["w0"].bsln_on is True

    tab.measurement_combo.setCurrentText("count")
    assert not tab.bsln_checkbox.isEnabled()
    assert not tab.bsln_checkbox.isChecked()
    assert manager.stats.windows["w0"].bsln_on is False

    # The user's choice comes back for a measurement that supports Δs
    tab.measurement_combo.setCurrentText("median")
    assert tab.bsln_checkbox.isChecked()
    assert manager.stats.windows["w0"].bsln_on is True


def test_level_n_std_requires_mean_std_baseline(qtbot):
    manager, _window, tab = _step_trace_window(qtbot)
    _set_text(tab.xbgn_edit, "10")
    tab.measurement_combo.setCurrentText("level+")
    assert tab.level_mode_combo.isVisibleTo(tab)
    assert tab.bsln_checkbox.isEnabled()

    tab.level_mode_combo.setCurrentIndex(tab.level_mode_combo.findData("n_std"))
    _set_text(tab.param_edits["n_std"], "3")

    win = manager.stats.windows["w0"]
    assert win.func == {"name": "level+", "n_std": 3.0}
    assert tab.bsln_checkbox.isChecked()
    assert not tab.bsln_checkbox.isEnabled()
    assert win.bsln_on is True
    assert win.bsln_func == {"name": "mean+std"}
    model = tab.bsln_combo.model()
    enabled = [
        tab.bsln_combo.itemData(i)
        for i in range(tab.bsln_combo.count())
        if model.item(i).isEnabled()
    ]
    assert enabled == ["mean+std"]

    # Back to an absolute Y level frees the baseline again
    tab.level_mode_combo.setCurrentIndex(tab.level_mode_combo.findData("ylevel"))
    assert tab.bsln_checkbox.isEnabled()
    assert win.func["name"] == "level+" and "ylevel" in win.func


def test_run_rejects_baseline_without_range(qtbot):
    manager, _window, tab = _step_trace_window(qtbot)
    tab.measurement_combo.setCurrentText("mean")
    tab.bsln_checkbox.setChecked(True)  # main X begin is -inf, so no prefill
    tab.run_button.click()
    assert "w0: set a baseline X range" in tab.status_label.text()


def test_baseline_settings_restored_per_window(qtbot):
    manager, _window, tab = _step_trace_window(qtbot)
    tab.measurement_combo.setCurrentText("mean")
    tab.bsln_checkbox.setChecked(True)
    tab.bsln_combo.setCurrentText("median")
    _set_text(tab.bsln_xbgn_edit, "1")
    _set_text(tab.bsln_xend_edit, "4")

    tab.window_combo.setCurrentIndex(tab.window_combo.findData(tab.NEW_WINDOW_ACTION))
    assert not tab.bsln_checkbox.isChecked()

    tab.window_combo.setCurrentText("w0")
    assert tab.bsln_checkbox.isChecked()
    assert tab.bsln_combo.currentText() == "median"
    assert tab.bsln_xbgn_edit.text() == "1"
    assert tab.bsln_xend_edit.text() == "4"


def _pulse_window(qtbot):
    """App window with a half-sine pulse of height 20 on a -70 baseline."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    t = np.arange(2001) * 0.02
    y = np.full_like(t, -70.0)
    mask = (t >= 10.0) & (t <= 30.0)
    y[mask] += 20.0 * np.sin(np.pi * (t[mask] - 10.0) / 20.0)
    data = folder.data.new(
        "RecordA0", nparray=y, xscale={"start": 0.0, "delta": 0.02}
    )
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    return manager, window, window.tool_tabs["Stats"]


def test_percent_measurements_use_family_defaults(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    expected = {
        "risetime+": {"p0": 10.0, "p1": 90.0},
        "falltime-": {"p0": 90.0, "p1": 10.0},
        "fwhm+": {"p0": 50.0, "p1": 50.0},
    }
    for name, params in expected.items():
        tab.measurement_combo.setCurrentText(name)
        assert _param_visible(tab, "p0") and _param_visible(tab, "p1")
        assert manager.stats.windows["w0"].func == {"name": name, **params}

    tab.measurement_combo.setCurrentText("decaytime+")
    assert _param_visible(tab, "p0") and not _param_visible(tab, "p1")
    assert tab.param_edits["p0"].text() == "36.7879"


def test_percent_values_kept_per_family(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    tab.measurement_combo.setCurrentText("risetime+")
    _set_text(tab.param_edits["p0"], "20")
    _set_text(tab.param_edits["p1"], "80")

    tab.measurement_combo.setCurrentText("fwhm+")
    assert manager.stats.windows["w0"].func == {"name": "fwhm+", "p0": 50.0, "p1": 50.0}

    # Same family, other direction: rise time values carry over
    tab.measurement_combo.setCurrentText("risetimeslope-")
    assert manager.stats.windows["w0"].func == {
        "name": "risetimeslope-", "p0": 20.0, "p1": 80.0
    }


def test_risetime_rejects_p0_above_p1(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    tab.measurement_combo.setCurrentText("risetime+")
    _set_text(tab.param_edits["p0"], "95")
    assert "Invalid risetime+ parameter" in tab.status_label.text()
    assert manager.stats.windows["w0"].func["p0"] == 10.0


def test_risetime_runs_with_required_baseline(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    _set_text(tab.xbgn_edit, "10")
    _set_text(tab.xend_edit, "30")
    tab.measurement_combo.setCurrentText("risetime+")

    assert tab.bsln_checkbox.isChecked()
    assert not tab.bsln_checkbox.isEnabled()
    assert tab.bsln_xbgn_edit.text() == "0"
    assert tab.bsln_xend_edit.text() == "10"
    _set_text(tab.bsln_xend_edit, "9.98")

    tab.run_button.click()
    assert tab.status_label.text() == "Complete"
    w0 = manager.stats.results["w0"][0]
    assert w0[-1]["dx"] == pytest.approx(6.491, abs=1e-3)
    assert [r["func"]["ylevel"] for r in w0 if "ylevel" in r["func"]] == [
        pytest.approx(-68.0, abs=0.01),
        pytest.approx(-52.0, abs=0.01),
    ]


def test_run_info_counts_epochs_and_data_arrays(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(3):
        folder.data.new(
            f"RecordA{epoch}",
            nparray=np.array([1.0, 2.0]),
            xscale={"start": 0.0, "delta": 1.0},
        )
    dataseries = folder.sync_dataseries("Record", select=True)
    dataseries.epochs.sets.add("Pair", ["E0", "E2"])
    manager.select_keys = {
        "folder": folder.name,
        "dataseries": dataseries.name,
        "channel": "A",
        "epoch": "E0",
    }
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]

    assert tab.run_info_label.text() == "n=3"  # All epochs by default
    window.selection_model.update(epoch_set="Current Epoch")
    assert tab.run_info_label.text() == "n=1"
    window.selection_model.update(epoch_set="Pair")
    assert tab.run_info_label.text() == "n=2"

    window.selection_model.update(data=folder.data["RecordA1"])
    assert tab.run_info_label.text() == "n=1"
    assert tab.run_info_label.toolTip() == "Run analyses 1 data array"


def test_run_info_explains_missing_selection(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    assert window.tool_tabs["Stats"].run_info_label.text() == (
        "Select a data array or a Data Series channel and epoch first"
    )


def test_st_arrays_one_point_per_epoch(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch, value in enumerate((10.0, 20.0, 30.0, 40.0)):
        folder.data.new(f"RecordA{epoch}", nparray=np.array([value, value]),
                        xscale={"start": 0.0, "delta": 1.0})
    dataseries = folder.sync_dataseries("Record", select=True)
    dataseries.epochs.sets.add("Chosen", ["E1", "E3"])
    manager.select_keys = {"folder": folder.name, "dataseries": dataseries.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    window.selection_model.update(epoch_set="Chosen")
    tab.measurement_combo.setCurrentText("mean")

    # The option, in Settings › Stats
    box = window.settings_dialog().tool_checkboxes["Stats"]["results_all_epochs"]
    assert box.text() == "ST_ arrays: one point per epoch"
    assert not box.isChecked()  # off by default
    box.setChecked(True)
    assert manager.stats.results_all_epochs and manager.stats.config.results_all_epochs

    tab.run_button.click()
    values = tab.stats_tool.toolfolder.data["ST_w0_mean_y"].nparray
    assert len(values) == 4  # every epoch of the series
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert (values[1], values[3]) == (20.0, 40.0)


def test_subfolder_named_after_the_epoch_selection(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(4):
        folder.data.new(f"RecordA{epoch}", nparray=np.array([1.0, 2.0]),
                        xscale={"start": 0.0, "delta": 1.0})
    dataseries = folder.sync_dataseries("Record", select=True)
    dataseries.epochs.sets.add("Set1", ["E1", "E3"])
    dataseries.epochs.groups.assign_cyclic(list(dataseries.epochs.keys()), n_groups=2)
    manager.select_keys = {"folder": folder.name, "dataseries": dataseries.name,
                           "channel": "A", "epoch": "E2"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")

    def run(**selection):
        window.selection_model.update(**selection)
        tab.run_button.click()
        return tab.stats_tool.toolfolder.name

    assert run(epoch_set="All") == "Stats_Record_A_All_0"
    assert run(epoch_set="Set1") == "Stats_Record_A_S1_0"
    assert run(epoch_set=None, group=0) == "Stats_Record_A_G0_0"
    assert run(epoch_set="Set1", group=1, group_operator="AND") == "Stats_Record_A_S1andG1_0"
    assert run(epoch_set="Set1", group=1, group_operator="OR") == "Stats_Record_A_S1orG1_0"
    assert run(epoch_set="Current Epoch", group=None) == "Stats_Record_A_E2_0"

    # Data arrays: the Data Set's code
    folder.data.sets.add("Set2", ["RecordA0", "RecordA1"])
    assert run(data=folder.data["RecordA0"], source="data", data_set="Set2") == "Stats_S2_0"


def test_choose_measurement_is_red(qtbot):
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    combo = window.tool_tabs["Stats"].measurement_combo
    brush = combo.itemData(0, QtCore.Qt.ItemDataRole.ForegroundRole)
    assert combo.itemText(0) == "Choose measurement" and brush.color().name() == "#c62828"
    combo.setCurrentIndex(0)
    assert "#c62828" in combo.styleSheet()  # the closed combo too
    # The open list keeps the usual colours: only its first item is red
    window.show()
    combo.showPopup()
    assert combo.styleSheet() == ""
    combo.hidePopup()
    assert "#c62828" in combo.styleSheet()
    combo.setCurrentText("mean")
    assert combo.styleSheet() == ""


def _series_manager():
    """Folder Demo, series Record (4 epochs, epoch i = i, i+1, ...); Set0 =
    E1, E2; groups 0/1 cyclic; RecordSet = RecordA0, RecordA3."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for epoch in range(4):
        folder.data.new(f"RecordA{epoch}", nparray=np.arange(5.0) + epoch,
                        xscale={"start": 0.0, "delta": 1.0})
    series = folder.sync_dataseries("Record", select=True)
    series.epochs.sets.add("Set0", ["E1", "E2"])
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), n_groups=2)
    folder.data.sets.add("RecordSet", ["RecordA0", "RecordA3"])
    return manager, folder, series


@pytest.fixture
def series_app(qtbot):
    manager, folder, series = _series_manager()
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series, epoch_set="Set0")
    window.manager, window.folder, window.series = manager, folder, series
    return window


def _history(app, start=0):
    QtWidgets.QApplication.processEvents()
    app.history_panel.flush()
    return [(e.kind, e.text) for e in app.history_panel.model.entries[start:]]


def _edit(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def test_window_changes_are_commands_one_per_change(series_app):
    """Only what changed, as the window's own commands; no echo messages (#448)."""
    tab, start = series_app.tool_tabs["Stats"], len(_history(series_app))
    tab.measurement_combo.setCurrentText("mean")
    _edit(tab.xbgn_edit, "1")
    _edit(tab.xbgn_edit, "1")  # unchanged: nothing
    _edit(tab.xend_edit, "3")
    tab.window_enabled.setChecked(False)
    assert _history(series_app, start) == [
        ("command", "nm.stats.windows['w0'].func = 'mean'"),
        ("command", "nm.stats.windows['w0'].xbgn = 1.0"),
        ("command", "nm.stats.windows['w0'].xend = 3.0"),
        ("command", "nm.stats.windows['w0'].on = False"),
    ]


def test_baseline_fields_are_set_once_it_is_on(series_app):
    tab, start = series_app.tool_tabs["Stats"], len(_history(series_app))
    tab.measurement_combo.setCurrentText("mean")
    _edit(tab.xbgn_edit, "2")
    tab.bsln_checkbox.setChecked(True)  # its range is prefilled: 0 to 2
    commands = [text for kind, text in _history(series_app, start) if kind == "command"]
    assert commands[-4:] == [
        "nm.stats.windows['w0'].bsln_on = True",
        "nm.stats.windows['w0'].bsln_func = 'mean'",
        "nm.stats.windows['w0'].bsln_xbgn = 0.0",
        "nm.stats.windows['w0'].bsln_xend = 2.0",
    ]


def test_new_window_is_one_command(series_app):
    tab, start = series_app.tool_tabs["Stats"], len(_history(series_app))
    tab.window_combo.setCurrentIndex(tab.window_combo.findData(tab.NEW_WINDOW_ACTION))
    tab.window_combo.setCurrentIndex(tab.window_combo.findData("w0"))
    assert [text for kind, text in _history(series_app, start) if kind == "command"] == [
        "nm.stats.windows.new(select=True)",
        "nm.stats.windows.selected_name = 'w0'",
    ]


def test_run_is_one_command_and_one_result(series_app):
    tab = series_app.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    start = len(_history(series_app))
    tab.run_button.click()
    assert _history(series_app, start) == [
        ("command", "nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'], "
                    "channel='A', epochs='Set0')"),
        ("tool", "Stats: Complete → Stats_Record_A_S0_0"),
    ]


@pytest.mark.parametrize("scope, expected", [
    ({"epoch_set": "All"}, "channel='A', epochs='all'"),
    ({"epoch_set": None, "group": 1}, "channel='A', group=1"),
    ({"group_operator": "OR", "group": 1}, "channel='A', epochs='Set0', group=1, "
                                           "group_operator='OR'"),
    ({"epoch_set": None}, "channel='A', epochs='E0'"),  # the current epoch
])
def test_run_command_names_the_epochs(series_app, scope, expected):
    tab = series_app.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    for key, value in scope.items():  # one at a time, as the strip does
        series_app.selection_model.update(**{key: value})
    start = len(_history(series_app))
    tab.run_button.click()
    commands = [text for kind, text in _history(series_app, start) if kind == "command"]
    assert commands == ["nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'], %s)"
                        % expected]


def test_data_set_run_command(series_app):
    tab = series_app.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    series_app.selection_model.update(data=series_app.folder.data["RecordA0"],
                                      data_set="RecordSet")
    start = len(_history(series_app))
    tab.run_button.click()
    assert ("command", "nm.stats.run_selection(nm.folders['Demo'], data='RecordSet')") \
        in _history(series_app, start)
    assert [r[0]["s"] for r in series_app.manager.stats.results["w0"]] == [2.0, 5.0]


def test_all_sets_is_a_command_per_set(series_app):
    tab = series_app.tool_tabs["Stats"]
    series_app.series.epochs.sets.add("Set2", ["E3"])
    tab.measurement_combo.setCurrentText("mean")
    series_app.selection_model.update(epoch_set="All Sets")
    start = len(_history(series_app))
    tab.run_button.click()
    commands = [text for kind, text in _history(series_app, start) if kind == "command"]
    assert [c.split("channel='A', ")[1] for c in commands] == [
        "epochs='Set0')", "epochs='Set2')"]


def test_recorded_stats_commands_repeat_the_run(series_app):
    """A script of the tab's commands gives the same results (#448)."""
    tab, manager = series_app.tool_tabs["Stats"], series_app.manager
    start = len(manager.command_history.buffer)
    tab.measurement_combo.setCurrentText("risetime+")
    _edit(tab.param_edits["p0"], "20")
    _edit(tab.xbgn_edit, "1")
    series_app.selection_model.update(group_operator="AND")
    series_app.selection_model.update(group=0)
    tab.run_button.click()
    expected = manager.stats.results
    commands = [e["command"] for e in manager.command_history.buffer[start:]
                if e["command"].startswith("nm.stats")]

    other, _folder, _series = _series_manager()
    other.tool_add("stats")
    for command in commands:
        exec(command, {"nm": other})
    assert other.stats.windows["w0"].func == manager.stats.windows["w0"].func
    assert other.stats.results == expected
    assert other.stats.toolfolder.name == manager.stats.toolfolder.name


def test_stats_options_are_commands_without_messages(series_app):
    """Settings › Stats and Output: the command alone, no "set ..." echo (#448)."""
    tab, start = series_app.tool_tabs["Stats"], len(_history(series_app))
    tab.set_config_option("overwrite", True)
    tab.set_config_option("results_all_epochs", True)
    tab.output_history_checkbox.setChecked(True)
    assert _history(series_app, start) == [
        ("command", "nm.stats.overwrite = True"),
        ("command", "nm.stats.results_all_epochs = True"),
        ("command", "nm.stats.results_to_history = True"),
    ]


def test_deleting_a_stats_subfolder_is_a_command(series_app):
    """Subfolders and their arrays are named in commands, so deleting one is
    recorded (and repeats), not just echoed (#448)."""
    tab, folder = series_app.tool_tabs["Stats"], series_app.folder
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    subfolder = folder.toolfolders["Stats_Record_A_S0_0"]
    array = next(iter(subfolder.data.keys()))
    start = len(_history(series_app))
    subfolder.data.pop(array)
    folder.toolfolders.pop("Stats_Record_A_S0_0")
    assert _history(series_app, start) == [
        ("command", "nm.folders['Demo'].toolfolders['Stats_Record_A_S0_0'].data.pop(%r)" % array),
        ("command", "nm.folders['Demo'].toolfolders.pop('Stats_Record_A_S0_0')"),
    ]
    other, other_folder, _series = _series_manager()
    other_folder.toolfolders.new("Stats_Record_A_S0_0")
    exec(_history(series_app, start)[-1][1], {"nm": other})
    assert "Stats_Record_A_S0_0" not in other_folder.toolfolders


def test_stats2_histogram_is_one_command_saved_in_the_subfolder(series_app):
    """Saved as H_ arrays in the Stats subfolder; its steps (replacing the
    arrays) aren't commands of their own; the command repeats it (#448)."""
    tab, manager = series_app.tool_tabs["Stats"], series_app.manager
    tab.measurement_combo.setCurrentText("mean")
    series_app.selection_model.update(epoch_set="All")
    tab.run_button.click()
    tab.stats2.array_combo.setCurrentText("ST_w0_mean_y")
    start = len(_history(series_app))
    tab.stats2.histogram_button.click()
    tab.stats2.histogram_button.click()  # again: replaced
    command = ("nm.stats.histogram(nm.folders['Demo'].toolfolders['Stats_Record_A_All_0'], "
               "'ST_w0_mean_y', bins=2)")
    assert _history(series_app, start) == [("command", command)] * 2
    subfolder = manager.stats.toolfolder
    counts = list(subfolder.data["H_ST_w0_mean_y_counts"].nparray)
    assert sum(counts) == 4

    subfolder.data.pop("H_ST_w0_mean_y_counts")
    exec(command, {"nm": manager})
    assert list(subfolder.data["H_ST_w0_mean_y_counts"].nparray) == counts


# --- Reset: the window back to its defaults ------------------------------------

def _reset_tab(qtbot):
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.arange(10.0),
                           xscale={"start": 0.0, "delta": 1.0})
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.tool_rail.select_tool("Stats")
    tab = window.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("max")
    for edit, text in ((tab.xbgn_edit, "2"), (tab.xend_edit, "8")):
        edit.setText(text)
        edit.editingFinished.emit()
    tab.bsln_checkbox.setChecked(True)
    return manager, window, tab


def test_reset_returns_the_window_to_its_defaults(qtbot):
    manager, window, tab = _reset_tab(qtbot)
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or True
    count = len(manager.command_history.buffer)
    tab.reset_button.click()
    assert asked and "w0" in asked[0]
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows['w0'].reset()"]
    w0 = manager.stats.windows["w0"]
    assert w0.func == {} and w0.on and not w0.bsln_on
    # The fields show it
    assert tab.measurement_combo.currentData() is None
    assert (tab.xbgn_edit.text(), tab.xend_edit.text()) == ("-inf", "inf")
    assert not tab.bsln_checkbox.isChecked()
    assert tab.window_enabled.isChecked()
    # ... and the Plot: its range is the whole trace, and nothing is measured
    region = window.plot_widget.range_overlay.regions["window"]
    assert region.getRegion() == (0.0, 9.0)
    assert window.plot_widget.mark_overlay.marks == []


def test_reset_cancelled_or_nothing_to_reset(qtbot):
    manager, _window, tab = _reset_tab(qtbot)
    tab._confirm = lambda title, text: False  # No
    tab.reset_button.click()
    assert manager.stats.windows["w0"].func == {"name": "max"}
    w0 = manager.stats.windows["w0"]
    w0.reset()
    tab._load_selected_window()
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or True
    count = len(manager.command_history.buffer)
    tab.reset_button.click()  # already the defaults: not asked, no command
    assert asked == [] and len(manager.command_history.buffer) == count


def test_reset_restores_parameter_defaults(qtbot):
    _manager, _window, tab = _reset_tab(qtbot)
    tab._confirm = lambda title, text: True
    tab.measurement_combo.setCurrentText("risetime+")
    default = tab.param_edits["p0"].text()
    tab.param_edits["p0"].setText("20")
    tab.param_edits["p0"].editingFinished.emit()
    tab.reset_button.click()
    tab.measurement_combo.setCurrentText("risetime+")
    assert tab.param_edits["p0"].text() == default


# --- Delete Window: the last one ---------------------------------------------

def _items(combo):
    return [combo.itemText(i) for i in range(combo.count()) if combo.itemText(i)]


def test_delete_window_deletes_the_last_one(qtbot):
    manager, _window, tab = _reset_tab(qtbot)
    combo = tab.window_combo
    assert _items(combo) == ["w0", "New Window..."]  # w0 can't be deleted
    tab._add_window()
    tab._add_window()
    assert _items(combo) == ["w0", "w1", "w2", "New Window...", "Delete Window w2"]
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or True
    count = len(manager.command_history.buffer)
    combo.setCurrentIndex(combo.findData(tab.DELETE_WINDOW_ACTION))
    assert asked == ["Delete w2? Its settings will be lost."]
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows.delete_last()"]
    assert _items(combo) == ["w0", "w1", "New Window...", "Delete Window w1"]
    assert combo.currentData() == "w1"  # w2 was shown
    tab._add_window()
    assert combo.currentData() == "w2"  # its name again


def test_delete_window_cancelled(qtbot):
    manager, _window, tab = _reset_tab(qtbot)
    combo = tab.window_combo
    tab._add_window()
    combo.setCurrentIndex(combo.findData("w0"))
    tab._confirm = lambda title, text: False
    combo.setCurrentIndex(combo.findData(tab.DELETE_WINDOW_ACTION))
    assert [w.name for w in manager.stats.windows] == ["w0", "w1"]
    assert combo.currentData() == "w0"  # back to the window shown
    assert manager.stats.windows.selected_name == "w0"
