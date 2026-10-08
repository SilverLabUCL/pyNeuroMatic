"""Tests for the Stats1 tool tab integration."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.stats_tab import StatsToolTab

pytestmark = pytest.mark.gui


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

    assert isinstance(window.stats_tab, StatsToolTab)
    assert "stats" in manager.enabled_tools()
    assert window.tool_workspace.widget(window.tool_rail.index_for_name("Stats")) is window.stats_tab

    tab = window.stats_tab
    tab.measurement_combo.setCurrentText("mean")
    tab.xbgn_edit.setText("1")
    tab.xend_edit.setText("3")
    tab.run_button.click()

    assert tab.status_label.text() == "Complete"
    output = tab.results_panel.output.toPlainText()
    assert '"s": 20.0' in output
    assert '"sunits": "mV"' in output
    assert manager.stats.windows["w0"].xbgn == 1.0
    assert manager.stats.windows["w0"].xend == 3.0


def test_window_combo_creates_and_selects_new_window(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.stats_tab

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
    tab = window.stats_tab
    window.selection_model.update(set="Chosen")
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
    tab = window.stats_tab

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
    tab = window.stats_tab
    window.plot_widget.channel_tabs.setCurrentIndex(1)
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()

    assert manager.select_values["channel"].name == "A"
    assert window.plot_widget.current_trace is dataseries.get_data("B", "E0")
    assert '"s": 2.0' in tab.results_panel.output.toPlainText()


def test_stats_output_options_control_backend_sinks(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.array([1.0, 3.0]))
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.stats_tab

    tab.measurement_combo.setCurrentText("mean")
    tab.output_history_checkbox.setChecked(True)
    tab.output_cache_checkbox.setChecked(False)
    tab.output_arrays_checkbox.setChecked(True)
    tab.run_button.click()

    assert manager.stats.results_to_history is True
    assert manager.stats.results_to_cache is False
    assert manager.stats.results_to_numpy is True
    assert "Stats_0" in folder.toolfolders


def test_config_checkbox_shows_editable_tool_configuration_list(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.stats_tab

    assert tab.tool_config_list.isHidden()
    tab.config_checkbox.setChecked(True)
    assert not tab.tool_config_list.isHidden()

    ignore_item = next(
        tab.tool_config_list.item(index)
        for index in range(tab.tool_config_list.count())
        if tab.tool_config_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
        == "ignore_nans"
    )
    ignore_item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    assert manager.stats.ignore_nans is False
    assert manager.stats.config.ignore_nans is False

def _param_visible(tab, key):
    return not tab.param_row.isHidden() and not tab.param_edits[key].isHidden()


def test_param_row_shows_only_inputs_for_current_measurement(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.stats_tab

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
    tab = window.stats_tab

    tab.measurement_combo.setCurrentText("mean@max")
    tab.param_edits["n_mean"].setText("3")
    tab.param_edits["n_mean"].editingFinished.emit()
    assert manager.stats.windows["w0"].func == {"name": "mean@max", "n_mean": 3}

    tab.run_button.click()
    assert tab.status_label.text() == "Complete"
    assert '"s": 9.0' in tab.results_panel.output.toPlainText()


def test_level_uses_ylevel_parameter_and_rejects_bad_input(qtbot):
    manager = NMManager(quiet=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.stats_tab

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
    tab = window.stats_tab

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
    return manager, window, window.stats_tab


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
    assert '"Δs": 50.0' in tab.results_panel.output.toPlainText()


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
    return manager, window, window.stats_tab


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
