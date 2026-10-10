"""Tests for the application shell and app-level GUI wiring."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow, SelectionStrip, ToolRail
from pyneuromatic.gui.folder_browser import FolderBrowserWidget

pytestmark = pytest.mark.gui


@pytest.fixture
def nm():
    return NMManager(quiet=True)


def test_selection_strip_exposes_expected_selectors():
    strip = SelectionStrip()
    labels = [strip.combo_label(i) for i in range(strip.count())]
    assert labels == [
        "Folder",
        "Data",
        "Data Series",
        "Channel",
        "Set",
        "Operator",
        "Group",
    ]
    strip.set_values({"dataseries": "Record"})
    assert strip.combo_boxes()[2].currentText() == "Record"
    assert [combo.minimumContentsLength() for combo in strip.combo_boxes()] == [
        14, 16, 14, 4, 8, 4, 3
    ]


def test_plot_trace_navigator_steps_data_and_series(qtbot, nm):
    folder = nm.folders.new("Demo", select=False)
    for name, values in (
        ("RecordA0", [1.0, 2.0]),
        ("RecordA1", [3.0, 4.0]),
    ):
        folder.data.new(
            name,
            nparray=np.array(values),
            xscale={"start": 0.0, "delta": 1.0},
        )
    dataseries = folder.sync_dataseries("Record")
    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    win.selection_model.update(folder=folder, data=folder.data["RecordA0"])

    navigator = win.plot_widget
    assert navigator.trace_index.minimum() == 0
    assert navigator.trace_index.maximum() == 1
    assert navigator.trace_name_label.text() == "RecordA0"

    navigator.trace_next_button.click()
    assert nm.select_values["data"] is folder.data["RecordA1"]
    assert navigator.trace_name_label.text() == "RecordA1"

    win.selection_model.update(dataseries=dataseries)
    assert navigator.trace_index.maximum() == 1
    navigator.trace_index.setValue(1)
    assert nm.select_values["epoch"] is dataseries.epochs["E1"]
    assert navigator.trace_name_label.text() == "RecordA1"

    navigator.trace_previous_button.click()
    assert nm.select_values["epoch"] is dataseries.epochs["E0"]
    assert navigator.trace_name_label.text() == "RecordA0"


def test_plot_channel_tabs_are_independent_of_analysis_channel(qtbot, nm):
    folder = nm.folders.new("Demo", select=False)
    for channel in ("A", "B"):
        for epoch in range(2):
            folder.data.new(
                f"Record{channel}{epoch}",
                nparray=np.array([epoch + (0 if channel == "A" else 10), 1.0]),
                xscale={"start": 0.0, "delta": 1.0},
                yscale={"label": "Vm", "units": "mV"},
            )
    folder.data.new(
        "StimA0",
        nparray=np.array([2.0, 3.0]),
        xscale={"start": 0.0, "delta": 1.0},
        yscale={"label": "Current", "units": "pA"},
    )
    dataseries = folder.sync_dataseries("Record", select=True)
    stim = folder.sync_dataseries("Stim")
    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    plot = win.plot_widget
    if plot.plot_widget is None:
        pytest.skip("pyqtgraph is not installed")

    analysis_channel = nm.select_values["channel"]
    assert [plot.view_tabs.tabText(i) for i in range(plot.view_tabs.count())] == [
        "A", "B"
    ]
    assert plot.view_tabs.currentIndex() == 0

    plot.view_tabs.setCurrentIndex(1)

    assert nm.select_values["channel"] is analysis_channel
    assert plot.current_trace is dataseries.get_data(channel="B", epoch="E0")
    assert plot.trace_name_label.text() == "RecordB0"

    plot.trace_index.setValue(1)
    assert nm.select_values["channel"] is analysis_channel
    assert plot.current_trace is dataseries.get_data(channel="B", epoch="E1")
    assert plot.plot_widget.getPlotItem().listDataItems()
    assert "Epoch" not in [
        win.selection_strip.combo_label(index)
        for index in range(win.selection_strip.count())
    ]

    win.selection_model.update(dataseries=stim)
    assert nm.select_values["dataseries"] is stim
    assert [plot.view_tabs.tabText(i) for i in range(plot.view_tabs.count())] == ["A"]
    assert plot.view_tabs.tabText(plot.view_tabs.currentIndex()) == "A"
    assert plot.current_trace is stim.get_data(channel="A", epoch="E0")

    win.selection_model.update(dataseries=dataseries)
    assert [plot.view_tabs.tabText(i) for i in range(plot.view_tabs.count())] == [
        "A", "B"
    ]
    assert plot.view_tabs.tabText(plot.view_tabs.currentIndex()) == "A"
    assert nm.select_values["channel"].name == "A"


def test_plot_overlay_limits_and_downsamples_traces(qtbot, nm):
    folder = nm.folders.new("Overlay", select=False)
    for epoch in range(8):
        folder.data.new(
            f"RecordA{epoch}",
            nparray=np.sin(np.linspace(0, 20 * np.pi, 5000) + epoch),
            xscale={"start": 0.0, "delta": 0.1},
        )
    dataseries = folder.sync_dataseries("Record", select=True)
    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    navigator = win.plot_widget
    if navigator.plot_widget is None:
        pytest.skip("pyqtgraph is not installed")

    assert navigator.overlay_limit.value() == 50
    assert not navigator.overlay_checkbox.isChecked()
    assert len(navigator.plot_widget.getPlotItem().listDataItems()) == 1

    navigator.overlay_limit.setValue(3)
    navigator.overlay_checkbox.setChecked(True)

    curves = navigator.plot_widget.getPlotItem().listDataItems()
    assert navigator.rendered_trace_count == 3
    assert len(curves) == 3
    assert navigator.overlay_status_label.text() == "Showing 3 of 8 traces"
    assert all(curve.opts["autoDownsample"] for curve in curves)
    assert all(curve.opts["clipToView"] for curve in curves)

    navigator.overlay_limit.setValue(6)
    assert navigator.rendered_trace_count == 6
    assert dataseries.epochs.selected_name == "E0"


def test_app_window_builds_shell(qtbot, nm):
    win = NMAppWindow(nm)
    qtbot.addWidget(win)

    assert win.menuBar() is not None
    assert hasattr(win, "selection_strip")
    assert hasattr(win, "tool_rail")
    assert hasattr(win, "tool_workspace")
    assert hasattr(win, "history_panel")
    assert hasattr(win, "selection_model")
    assert hasattr(win, "browser_widget")
    assert hasattr(win, "plot_widget")
    assert win.context_panel.tabText(0) == "Plot"
    assert isinstance(win.browser_widget, FolderBrowserWidget)
    assert win.tool_rail.tool_names() == ["Browser", "Main", "Stats", "Spike"]
    win.show()
    qtbot.waitUntil(lambda: win.workspace_splitter.width() > 0)
    workspace_width, plot_width = win.workspace_splitter.sizes()[1:]
    assert workspace_width < plot_width


def test_browser_selection_updates_shared_gui_state(qtbot, nm):
    folder = nm.folders.new("folder0", select=False)
    data = folder.data.new("Record", nparray=np.array([1.0, 2.0, 3.0]))
    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    win.selection_model.clear()

    model = win.browser_widget.model
    folders_group = model.index(0, 0)
    folder_index = _find_child(model, folders_group, "folder0")
    data_group = _find_child(model, folder_index, "Data")
    data_index = _find_child(model, data_group, "Record")
    win.browser_widget.tree.setCurrentIndex(data_index)

    assert win.selection_model.selection["data"] is None
    win.browser_widget._request_selection(data_index)

    assert win.selection_model.selection["data"] is data
    assert win.selection_strip.combo_boxes()[1].currentText() == "Record"
    assert "Record" in win.plot_widget.selection_label.text()
    assert "Record" in win.history_panel.toPlainText()

    win.selection_model.clear()

    assert all(value is None for value in win.selection_model.selection.values())
    assert win.selection_strip.combo_boxes()[1].currentText() == ""
    assert "No selection" in win.plot_widget.selection_label.text()


def test_selection_menus_follow_folder_and_series(qtbot, nm):
    folder1 = nm.folders.new("Demo1", select=False)
    folder1.data.new(
        "AlphaA0", nparray=np.array([1.0, 2.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder1.data.new(
        "AlphaB1", nparray=np.array([3.0, 4.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder1.data.new(
        "BetaA0", nparray=np.array([5.0, 6.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    alpha = folder1.sync_dataseries("Alpha")
    folder1.sync_dataseries("Beta")
    alpha.epochs.sets.add("Demo1Set", ["E0"])
    alpha.epochs.groups.assign_cyclic(["E0", "E1"], n_groups=2)

    folder2 = nm.folders.new("Demo2", select=False)
    folder2.data.new(
        "GammaA0", nparray=np.array([7.0, 8.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder2.data.new(
        "GammaA1", nparray=np.array([8.0, 9.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder2.data.new(
        "GammaB0", nparray=np.array([9.0, 10.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    folder2.data.new(
        "GammaB1", nparray=np.array([9.0, 10.0]),
        xscale={"start": 0.0, "delta": 1.0},
    )
    gamma = folder2.sync_dataseries("Gamma")
    gamma.epochs.sets.add("Demo2Set", ["E0"])
    gamma.epochs.groups.assign_cyclic(["E0", "E1"], n_groups=2)

    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    strip = win.selection_strip
    (
        folder_menu,
        data_menu,
        series_menu,
        channel_menu,
        set_menu,
        operator_menu,
        group_menu,
    ) = strip.combo_boxes()

    assert _combo_values(folder_menu) == ["", "Demo1", "Demo2"]
    assert _combo_values(data_menu) == ["", "AlphaA0", "AlphaB1", "BetaA0"]
    assert _combo_values(series_menu) == ["", "Alpha", "Beta"]
    assert folder_menu.currentText() == "Demo1"
    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Alpha"
    assert channel_menu.currentText() == "A"
    assert win.plot_widget.trace_name_label.text() == "AlphaA0"
    assert _combo_values(set_menu) == ["", "Demo1Set"]
    assert _combo_values(group_menu) == ["", "0", "1"]
    assert _combo_values(operator_menu) == [""]

    set_menu.setCurrentText("Demo1Set")
    group_menu.setCurrentText("1")
    assert _combo_values(operator_menu) == ["", "AND", "OR"]
    operator_menu.setCurrentText("AND")
    folder_menu.setCurrentText("Demo2")

    assert nm.select_values["folder"] is folder2
    assert nm.select_values["data"] is None
    assert nm.select_values["dataseries"] is folder2.dataseries["Gamma"]
    assert _combo_values(data_menu) == ["", "GammaA0", "GammaA1", "GammaB0", "GammaB1"]
    assert _combo_values(series_menu) == ["", "Gamma"]
    assert channel_menu.currentText() == "A"
    assert win.plot_widget.trace_name_label.text() == "GammaA0"
    assert _combo_values(set_menu) == ["", "Demo2Set"]
    assert set_menu.currentText() == ""
    assert _combo_values(group_menu) == ["", "0", "1"]
    assert group_menu.currentText() == ""
    assert _combo_values(operator_menu) == [""]
    set_menu.setCurrentText("Demo2Set")
    group_menu.setCurrentText("0")
    operator_menu.setCurrentText("OR")

    series_menu.setCurrentText("Gamma")

    assert nm.select_values["dataseries"] is folder2.dataseries["Gamma"]
    channel_menu.setCurrentText("B")
    win.plot_widget.trace_index.setValue(1)
    assert nm.select_values["epoch"] is folder2.dataseries["Gamma"].epochs["E1"]
    data_menu.setCurrentText("GammaB1")
    assert nm.select_values["data"] is folder2.data["GammaB1"]
    assert win.plot_widget.trace_name_label.text() == "GammaB1"

    series_menu.setCurrentText("Gamma")

    assert nm.select_values["channel"] is folder2.dataseries["Gamma"].channels["B"]
    assert nm.select_values["epoch"] is folder2.dataseries["Gamma"].epochs["E1"]
    assert win.plot_widget.trace_name_label.text() == "GammaB1"

    folder_menu.setCurrentIndex(0)
    assert all(value is None for value in win.selection_model.selection.values())
    folder_menu.setCurrentText("Demo1")
    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Alpha"
    assert win.plot_widget.trace_name_label.text() == "AlphaA0"
    assert set_menu.currentText() == "Demo1Set"
    assert group_menu.currentText() == "1"
    assert operator_menu.currentText() == "AND"

    folder_menu.setCurrentText("Demo2")
    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Gamma"
    assert win.plot_widget.view_tabs.tabText(win.plot_widget.view_tabs.currentIndex()) == "A"
    assert win.plot_widget.trace_name_label.text() == "GammaA1"
    assert set_menu.currentText() == "Demo2Set"
    assert group_menu.currentText() == "0"
    assert operator_menu.currentText() == "OR"


def test_selection_memory_is_independent_per_dataseries(qtbot, nm):
    folder = nm.folders.new("Demo", select=False)
    for name, values in (
        ("AlphaA0", [1.0, 2.0]),
        ("AlphaB1", [3.0, 4.0]),
        ("BetaC2", [5.0, 6.0]),
        ("BetaD3", [7.0, 8.0]),
    ):
        folder.data.new(name, nparray=np.array(values))
    alpha = folder.sync_dataseries("Alpha", select=True)
    beta = folder.sync_dataseries("Beta")
    alpha.epochs.sets.add("AlphaSet", ["E0"])
    alpha.epochs.groups.assign_cyclic(["E0", "E1"], n_groups=2)
    beta.epochs.sets.add("BetaSet", ["E1"])
    beta.epochs.groups.assign_cyclic(["E0", "E1"], n_groups=2)

    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    _, _, series_menu, channel_menu, set_menu, operator_menu, group_menu = (
        win.selection_strip.combo_boxes()
    )
    trace_navigator = win.plot_widget

    set_menu.setCurrentText("AlphaSet")
    group_menu.setCurrentText("1")
    operator_menu.setCurrentText("AND")
    channel_menu.setCurrentText("B")
    trace_navigator.trace_index.setValue(1)
    series_menu.setCurrentText("Beta")

    assert nm.select_values["dataseries"] is beta
    assert channel_menu.currentText() == "A"
    assert trace_navigator.trace_index.value() == 0
    assert set_menu.currentText() == ""
    assert group_menu.currentText() == ""
    assert operator_menu.currentText() == ""

    set_menu.setCurrentText("BetaSet")
    group_menu.setCurrentText("0")
    operator_menu.setCurrentText("OR")
    series_menu.setCurrentText("Alpha")

    assert nm.select_values["dataseries"] is alpha
    assert channel_menu.currentText() == "B"
    assert trace_navigator.trace_index.value() == 1
    assert set_menu.currentText() == "AlphaSet"
    assert group_menu.currentText() == "1"
    assert operator_menu.currentText() == "AND"

    series_menu.setCurrentText("Beta")

    assert channel_menu.currentText() == "A"
    assert trace_navigator.trace_index.value() == 0
    assert set_menu.currentText() == "BetaSet"
    assert group_menu.currentText() == "0"
    assert operator_menu.currentText() == "OR"


def test_data_and_dataseries_selectors_toggle_exclusively(qtbot, nm):
    folder = nm.folders.new("Demo", select=False)
    scale = {"start": 0.0, "delta": 1.0, "label": "Time", "units": "ms"}
    yscale = {"label": "Vm", "units": "mV"}
    folder.data.new("Flat0", nparray=np.array([1.0, 2.0]), xscale=scale, yscale=yscale)
    folder.data.new("RecordA0", nparray=np.array([3.0, 4.0]), xscale=scale, yscale=yscale)
    folder.data.new("RecordB1", nparray=np.array([5.0, 6.0]), xscale=scale, yscale=yscale)
    record = folder.sync_dataseries("Record", select=True)
    record.epochs.sets.add("EpochSet", ["E0"])
    record.epochs.groups.assign_cyclic(["E0", "E1"], n_groups=2)
    folder.data.sets.add("FlatSet", ["Flat0"])
    folder.data.sets.add("DataSet", ["RecordA0"])

    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    _, data_menu, series_menu, channel_menu, set_menu, operator_menu, group_menu = (
        win.selection_strip.combo_boxes()
    )
    trace_navigator = win.plot_widget

    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Record"
    assert channel_menu.isEnabled()
    assert trace_navigator.trace_index.isEnabled()
    assert _combo_values(set_menu) == ["", "EpochSet"]
    assert _combo_values(group_menu) == ["", "0", "1"]

    channel_menu.setCurrentText("B")
    trace_navigator.trace_index.setValue(1)
    set_menu.setCurrentText("EpochSet")
    group_menu.setCurrentText("1")
    operator_menu.setCurrentText("AND")
    data_menu.setCurrentText("RecordA0")

    assert nm.select_values["data"] is folder.data["RecordA0"]
    assert nm.select_values["dataseries"] is None
    assert series_menu.currentText() == ""
    assert not channel_menu.isEnabled()
    assert trace_navigator.trace_index.isEnabled()
    assert trace_navigator.trace_name_label.text() == "RecordA0"
    assert _combo_values(set_menu) == ["", "FlatSet", "DataSet"]
    assert _combo_values(group_menu) == ["", "0", "1"]
    set_menu.setCurrentText("DataSet")
    group_menu.setCurrentText("0")
    operator_menu.setCurrentText("OR")

    series_menu.setCurrentText("Record")

    assert nm.select_values["data"] is None
    assert nm.select_values["dataseries"] is record
    assert data_menu.currentText() == ""
    assert channel_menu.currentText() == "B"
    assert trace_navigator.trace_index.value() == 1
    assert channel_menu.isEnabled()
    assert trace_navigator.trace_index.isEnabled()
    assert set_menu.currentText() == "EpochSet"
    assert group_menu.currentText() == "1"
    assert operator_menu.currentText() == "AND"

    data_menu.setCurrentText("RecordA0")
    assert set_menu.currentText() == "DataSet"
    assert group_menu.currentText() == "0"
    assert operator_menu.currentText() == "OR"


def test_new_group_refreshes_group_selector(qtbot, nm, monkeypatch):
    folder = nm.folders.new("Demo", select=False)
    for name in ("RecordA0", "RecordB1"):
        folder.data.new(
            name,
            nparray=np.array([1.0, 2.0]),
            xscale={"start": 0.0, "delta": 1.0},
        )
    dataseries = folder.sync_dataseries("Record", select=True)
    win = NMAppWindow(nm)
    qtbot.addWidget(win)
    group_menu = win.selection_strip.combo_boxes()[6]

    assert _combo_values(group_menu) == [""]

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getInt",
        staticmethod(lambda *a, **k: (2, True)),
    )
    win.browser_widget._new_group(dataseries.epochs)

    assert _combo_values(group_menu) == ["", "0", "1"]
    assert dataseries.epochs.groups.get_items(0) == []
    assert dataseries.epochs.groups.get_items(1) == []


def _find_child(model, parent, label):
    for row in range(model.rowCount(parent)):
        index = model.index(row, 0, parent)
        if model.data(index) == label:
            return index
    raise AssertionError(f"row not found: {label}")


def _combo_values(combo):
    return [combo.itemText(index) for index in range(combo.count())]


def test_tool_rail_switches_active_tool(qtbot, nm):
    win = NMAppWindow(nm)
    qtbot.addWidget(win)

    win.tool_rail.select_tool("Stats")

    assert win.current_tool_name == "Stats"
    assert win.tool_workspace.currentIndex() == win.tool_rail.index_for_name("Stats")
