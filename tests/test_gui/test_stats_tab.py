"""Tests for the Stats1 tool tab integration."""

import html

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


def test_new_window_creates_and_selects_it(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    asked = []

    def choose(title, **options):
        asked.append((title, options))
        return ("", None, options["extras"][0][1].prefix + str(options["extras"][0][1].value))

    tab._choose = choose
    tab.new_window_action.trigger()

    title, options = asked[0]
    assert title == "New Stats Window" and options["extras"][0][1].value == 1
    assert tab.window_combo.currentText() == "w1"
    assert manager.stats.windows.selected_name == "w1"


def test_new_window_can_fill_a_gap(qtbot):
    """New Window… starts on the number after the highest (w0, w1, w3 → w4);
    a gap can be filled (w2), shown in order; a used number is refused."""
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    windows = manager.stats.windows
    for _ in range(3):
        tab._add_window()
    windows.delete("w2")
    tab._refresh_window_combo()
    asked = []

    def choose(title, **options):  # its messages, while it's open
        asked.append((options, {n: (options["message"]("", n), options["validate"]("", None, n))
                                for n in ("w2", "w3")}))
        return ("", None, "w2")

    tab._choose = choose
    count = len(manager.command_history.buffer)
    tab.new_window_action.trigger()
    options, shown = asked[0]
    assert options["extras"][0][1].value == 4
    assert shown == {"w2": ("Creates w2.", None), "w3": ("", "w3 already exists.")}
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows.new(number=2, select=True)"]
    combo = tab.window_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["w0", "w1", "w2", "w3"]
    assert combo.currentText() == "w2" and windows.selected_name == "w2"
    assert windows.next_number == 4


def test_new_window_cancelled(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    tab._choose = lambda title, **options: None
    tab.new_window_action.trigger()
    assert [w.name for w in manager.stats.windows] == ["w0"]


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
    tab._add_window()
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


def test_stats_arrays_are_a_setting_and_no_cache(qtbot):
    """No Output group (#480): Stats arrays is in Settings › Stats (on by
    default); the results cache is off by default (memory; for scripts)."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.array([1.0, 3.0]))
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    for name in ("output_history_checkbox", "output_cache_checkbox",
                 "output_arrays_checkbox"):
        assert not hasattr(tab, name)
    options = {option[0]: option for option in tab.config_options()}
    assert options["results_to_numpy"][1:3] == ("Save ST_ arrays", True)

    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    assert manager.stats.results_to_cache is False
    assert "stats" not in folder.toolresults  # nothing kept in memory
    assert "Stats_0" in folder.toolfolders

    tab.set_config_option("results_to_numpy", False)
    tab.run_button.click()
    assert "Stats_1" not in folder.toolfolders  # no arrays saved

    manager.stats.results_to_cache = True  # from a script: still works
    tab.run_button.click()
    assert len(folder.toolresults["stats"]) == 1


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
    tab.set_measurement("level+")
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

    tab.set_measurement("level+")
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
    tab._add_window()
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

    tab.set_measurement("count")
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
    tab.set_measurement("level+")
    assert tab.level_mode_combo.isVisibleTo(tab)
    assert tab.bsln_checkbox.isEnabled()

    tab.level_mode_combo.setCurrentIndex(tab.level_mode_combo.findData("n_std"))
    _set_text(tab.param_edits["n_std"], "3")

    win = manager.stats.windows["w0"]
    assert win.func == {"name": "level+", "n_std": 3.0}
    assert tab.bsln_checkbox.isChecked()
    assert tab.bsln_checkbox.isEnabled()  # not faded, but locked on
    assert tab.bsln_checkbox.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
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

    tab._add_window()
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
        tab.set_measurement(name)
        assert _param_visible(tab, "p0") and _param_visible(tab, "p1")
        assert manager.stats.windows["w0"].func == {"name": name, **params}

    tab.set_measurement("decaytime+")
    assert _param_visible(tab, "p0") and not _param_visible(tab, "p1")
    assert tab.param_edits["p0"].text() == "36.7879"


def test_percent_values_kept_per_family(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    tab.set_measurement("risetime+")
    _set_text(tab.param_edits["p0"], "20")
    _set_text(tab.param_edits["p1"], "80")

    tab.set_measurement("fwhm+")
    assert manager.stats.windows["w0"].func == {"name": "fwhm+", "p0": 50.0, "p1": 50.0}

    # Same family, other direction: rise time values carry over
    tab.set_measurement("risetimeslope-")
    assert manager.stats.windows["w0"].func == {
        "name": "risetimeslope-", "p0": 20.0, "p1": 80.0
    }


def test_risetime_rejects_p0_above_p1(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    tab.set_measurement("risetime+")
    _set_text(tab.param_edits["p0"], "95")
    assert "Invalid risetime+ parameter" in tab.status_label.text()
    assert manager.stats.windows["w0"].func["p0"] == 10.0


def test_risetime_runs_with_required_baseline(qtbot):
    manager, _window, tab = _pulse_window(qtbot)
    _set_text(tab.xbgn_edit, "10")
    _set_text(tab.xend_edit, "30")
    tab.set_measurement("risetime+")

    assert tab.bsln_checkbox.isChecked()
    assert tab.bsln_checkbox.isEnabled()  # not faded, but locked on
    assert tab.bsln_checkbox.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
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


def test_choose_measurement_is_plain(qtbot):
    """"Choose measure" (none yet) looks like the other items: no red."""
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    combo = window.tool_tabs["Stats"].measurement_combo
    assert combo.itemText(0) == "Choose measure" and combo.itemData(0) is None
    assert combo.itemData(0, QtCore.Qt.ItemDataRole.ForegroundRole) is None
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
    tab._add_window()
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
    tab.set_measurement("risetime+")
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
    tab.set_config_option("results_to_numpy", False)
    assert _history(series_app, start) == [
        ("command", "nm.stats.overwrite = True"),
        ("command", "nm.stats.results_all_epochs = True"),
        ("command", "nm.stats.results_to_numpy = False"),
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
    """Saved as H_<array> in the Stats subfolder (#492); its steps (replacing
    the array) aren't commands of their own; the command repeats it (#448)."""
    tab, manager = series_app.tool_tabs["Stats"], series_app.manager
    tab.measurement_combo.setCurrentText("mean")
    series_app.selection_model.update(epoch_set="All")
    tab.run_button.click()
    tab.stats2.array_combo.setCurrentText("ST_w0_mean_y")
    start = len(_history(series_app))
    tab.stats2._ask_histogram = lambda name, values, units: {"binwidth": 0.5,
                                                              "output": "probability"}
    tab.stats2.histogram_dialog_action.trigger()
    tab.stats2.array_combo.setCurrentText("ST_w0_mean_y")
    tab.stats2.histogram_dialog_action.trigger()  # again: replaced
    command = ("nm.stats.histogram(nm.folders['Demo'].toolfolders['Stats_Record_A_All_0'], "
               "'ST_w0_mean_y', binwidth=0.5, output='probability')")
    assert _history(series_app, start) == [("command", command)] * 2
    subfolder = manager.stats.toolfolder
    values = list(subfolder.data["H_ST_w0_mean_y"].nparray)
    assert sum(values) == pytest.approx(1)

    subfolder.data.pop("H_ST_w0_mean_y")
    exec(command, {"nm": manager})
    assert list(subfolder.data["H_ST_w0_mean_y"].nparray) == values


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


def _prompt(tab, answer=None):
    """Replace the Window prompts: answer with the window *answer* (or the
    one it starts on), checking the message and validate as OK would.
    Returns the prompts asked, as (title, options, message, error)."""
    asked = []

    def choose(title, **options):
        item = answer or options["current"]
        message = options["message"](item)
        error = options["validate"](item, None) if options.get("validate") else None
        asked.append((title, options, message, error))
        return None if error else (item, None)

    tab._choose = choose
    return asked


def _choose_reset(tab, answer=None):
    """Window ⋯ › Reset… (OK on *answer*, or the window shown)."""
    asked = _prompt(tab, answer)
    tab.reset_window_action.trigger()
    return asked


def test_reset_returns_the_window_to_its_defaults(qtbot):
    manager, window, tab = _reset_tab(qtbot)
    count = len(manager.command_history.buffer)
    asked = _choose_reset(tab)
    title, options, message, error = asked[0]
    assert title == "Reset Stats Window" and options["item_label"] == "Window"
    assert options["items"] == ["w0"] and options["current"] == "w0"  # the window shown
    assert message == "w0's measurement, X begin / X end and baseline will be cleared."
    assert error is None
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
    tab._choose = lambda title, **options: None  # Cancel
    tab.reset_window_action.trigger()
    assert manager.stats.windows["w0"].func == {"name": "max"}
    w0 = manager.stats.windows["w0"]
    w0.reset()
    tab._load_selected_window()
    count = len(manager.command_history.buffer)
    asked = _choose_reset(tab)  # already the defaults: OK closes, no command
    assert asked[0][2] == "w0 already has its defaults."
    assert asked[0][3] is None  # not an error
    assert len(manager.command_history.buffer) == count


def test_reset_restores_parameter_defaults(qtbot):
    _manager, _window, tab = _reset_tab(qtbot)
    tab.set_measurement("risetime+")
    default = tab.param_edits["p0"].text()
    tab.param_edits["p0"].setText("20")
    tab.param_edits["p0"].editingFinished.emit()
    _choose_reset(tab)
    tab.set_measurement("risetime+")
    assert tab.param_edits["p0"].text() == default


def test_reset_another_window(qtbot):
    """Reset… can choose a window other than the one shown."""
    manager, _window, tab = _reset_tab(qtbot)  # w0: max
    tab._add_window()  # w1, shown
    tab.measurement_combo.setCurrentText("mean")
    _choose_reset(tab, "w0")
    windows = manager.stats.windows
    assert windows["w0"].func == {} and windows["w1"].func == {"name": "mean"}
    assert tab.window_combo.currentData() == "w1"  # still shown
    assert tab.measurement_combo.currentText() == "mean"


# --- The Window ⋯ menu, Delete… any window (#514) ---------------------------------

def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_the_window_list_has_only_windows_and_a_menu(qtbot):
    """New, Reset and Delete are in the ⋯ menu, not the list (#514), so
    moving through the list can't start one."""
    _manager, _window, tab = _reset_tab(qtbot)
    combo = tab.window_combo
    assert _items(combo) == ["w0"]
    assert [a.text() for a in tab.window_actions_button.menu().actions()] == [
        "New Window…", "Copy…", "Reset…", "Delete…"]
    assert not tab.delete_window_action.isEnabled()  # w0, the only one, is kept
    tab._add_window()
    tab._add_window()
    assert _items(combo) == ["w0", "w1", "w2"]
    assert tab.delete_window_action.isEnabled()


def test_delete_any_window(qtbot):
    manager, _window, tab = _reset_tab(qtbot)
    combo = tab.window_combo
    tab._add_window()
    tab._add_window()  # w2, shown
    count = len(manager.command_history.buffer)
    asked = _prompt(tab, "w1")
    tab.delete_window_action.trigger()
    title, options, message, _error = asked[0]
    assert title == "Delete Stats Window" and options["items"] == ["w0", "w1", "w2"]
    assert options["current"] == "w2"  # starts on the window shown
    assert message == ("w1 will be deleted; its settings will be lost. "
                       "The other windows keep their names.")
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows.delete('w1')"]
    assert _items(combo) == ["w0", "w2"]  # w2 keeps its name
    assert combo.currentData() == "w2"  # still shown
    tab._add_window()
    assert _items(combo) == ["w0", "w2", "w3"]  # after the highest
    # Deleting the window shown shows the one before it
    _prompt(tab)  # w3, shown
    tab.delete_window_action.trigger()
    assert _items(combo) == ["w0", "w2"] and combo.currentData() == "w2"
    assert manager.stats.windows.selected_name == "w2"
    tab._add_window()
    assert combo.currentData() == "w3"  # the last's name again


def test_copy_window(qtbot):
    """Copy… (#515): a new window with a window's settings, starting on the
    one shown, numbered as New Window…; the copy is shown."""
    manager, _window, tab = _reset_tab(qtbot)  # w0: max, x 2 to 8, baseline on
    windows = manager.stats.windows
    tab._add_window()  # w1, shown
    tab._add_window()  # w2
    windows.delete("w1")
    tab._refresh_window_combo()
    tab.window_combo.setCurrentIndex(tab.window_combo.findData("w0"))
    asked = []

    def choose(title, **options):  # its messages, while it's open
        asked.append((title, options, options["message"]("w0", "w1", False),
                      options["validate"]("w0", None, "w2", False)))
        return ("w0", None, "w1", False)

    tab._choose = choose
    count = len(manager.command_history.buffer)
    tab.copy_window_action.trigger()
    title, options, message, error = asked[0]
    assert title == "Copy Stats Window" and options["item_label"] == "Window"
    assert options["items"] == ["w0", "w2"] and options["current"] == "w0"
    assert options["extras"][0][0] == "To" and options["extras"][0][1].value == 3
    assert message == "Copies w0's settings to w1."
    assert error == "w2 already exists."
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows.copy('w0', number=1, select=True)"]
    copy = windows["w1"]
    assert copy.to_dict() == {**windows["w0"].to_dict(), "name": "w1"}
    combo = tab.window_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["w0", "w1", "w2"]
    assert combo.currentData() == "w1"  # the copy, shown
    assert tab.measurement_combo.currentText() == "max"
    assert (tab.xbgn_edit.text(), tab.xend_edit.text()) == ("2", "8")
    assert tab.bsln_checkbox.isChecked()
    # Cancelled: nothing
    tab._choose = lambda title, **options: None
    tab.copy_window_action.trigger()
    assert len(windows) == 3


def test_copy_window_over_an_existing_one(qtbot):
    """With Overwrite (only while the number is a window's, as Copy Set),
    the window's settings are replaced, with a warning; not onto itself."""
    manager, _window, tab = _reset_tab(qtbot)  # w0: max, x 2 to 8
    windows = manager.stats.windows
    tab._add_window()  # w1, shown: the defaults
    shown = {}

    def choose(title, **options):  # its messages, while it's open
        check = options["extras"][1][1]
        shown.update(
            enabled={n: check.enabled("w0", n, False) for n in ("w1", "w2")},
            warning=options["message"]("w0", "w1", True),
            without=options["validate"]("w0", None, "w1", False),
            itself=options["validate"]("w0", None, "w0", True))
        return ("w0", None, "w1", True)

    tab._choose = choose
    count = len(manager.command_history.buffer)
    tab.copy_window_action.trigger()
    assert shown["enabled"] == {"w1": True, "w2": False}  # only an existing window
    assert "w1's settings will be replaced with w0's" in html.unescape(shown["warning"])
    assert "color" in shown["warning"]  # in red
    assert shown["without"] == "w1 already exists."
    assert shown["itself"] == "w0 can't be copied onto itself."
    assert [e["command"] for e in manager.command_history.buffer][count:] == [
        "nm.stats.windows.copy('w0', number=1, select=True, overwrite=True)"]
    assert [w.name for w in windows] == ["w0", "w1"]
    assert windows["w1"].to_dict() == {**windows["w0"].to_dict(), "name": "w1"}
    assert tab.window_combo.currentData() == "w1"
    assert tab.measurement_combo.currentText() == "max"


def test_delete_window_cancelled(qtbot):
    manager, _window, tab = _reset_tab(qtbot)
    combo = tab.window_combo
    tab._add_window()
    combo.setCurrentIndex(combo.findData("w0"))
    tab._choose = lambda title, **options: None  # Cancel
    tab.delete_window_action.trigger()
    assert [w.name for w in manager.stats.windows] == ["w0", "w1"]
    assert combo.currentData() == "w0"
    assert manager.stats.windows.selected_name == "w0"


def test_a_required_baseline_is_shown_on_and_cant_be_turned_off(qtbot, click):
    """Not faded (#478 follow-up): on, and a click through the window (as a
    mouse) doesn't turn it off; a measurement that can't use one is faded."""
    manager, window, tab = _pulse_window(qtbot)
    window.resize(1300, 800)
    window.show()
    qtbot.waitExposed(window)
    window.tool_rail.select_tool("Stats")
    tab.set_measurement("risetime+")
    box = tab.bsln_checkbox
    assert box.isChecked() and box.isEnabled()
    assert "requires a baseline" in box.toolTip()
    click(box)
    assert box.isChecked() and manager.stats.windows["w0"].bsln_on
    tab.measurement_combo.setCurrentText("std")  # can't use a baseline
    assert not box.isEnabled() and not box.isChecked()
    tab.measurement_combo.setCurrentText("mean")  # optional: the user's choice
    assert box.isEnabled()
    assert not box.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    before = box.isChecked()
    click(box)
    assert box.isChecked() != before  # takes the click


def test_stats2_inequality_is_one_command(series_app):
    """Saved as IQ_<array> in the Stats subfolder, with Epoch Sets of the
    passes and fails, as one command that repeats it (#499)."""
    tab, manager = series_app.tool_tabs["Stats"], series_app.manager
    tab.measurement_combo.setCurrentText("mean")
    series_app.selection_model.update(epoch_set="All")
    tab.run_button.click()
    tab.stats2.array_combo.setCurrentText("ST_w0_mean_y")
    start = len(_history(series_app))
    values = list(manager.stats.toolfolder.data["ST_w0_mean_y"].nparray)
    middle = float(sorted(values)[len(values) // 2])
    tab.stats2._ask_inequality = lambda name, set_names: {
        "op": ">=", "a": middle, "set_name_success": "Big"}
    tab.stats2.inequality_dialog_action.trigger()
    command = ("nm.stats.inequality(nm.folders['Demo'].toolfolders['Stats_Record_A_All_0'], "
               f"'ST_w0_mean_y', '>=', {middle!r}, "
               "dataseries=nm.folders['Demo'].dataseries['Record'], set_name_success='Big')")
    assert _history(series_app, start) == [("command", command)]
    subfolder = manager.stats.toolfolder
    result = list(subfolder.data["IQ_ST_w0_mean_y"].nparray)
    assert result == [1.0 if v >= middle else 0.0 for v in values]

    subfolder.data.pop("IQ_ST_w0_mean_y")
    exec(command, {"nm": manager})
    assert list(subfolder.data["IQ_ST_w0_mean_y"].nparray) == result
