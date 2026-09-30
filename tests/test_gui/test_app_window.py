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
        "Epoch",
        "Set",
        "Operator",
        "Group",
    ]
    strip.set_values({"dataseries": "Record"})
    assert strip.combo_boxes()[2].currentText() == "Record"
    assert [combo.minimumContentsLength() for combo in strip.combo_boxes()] == [
        14, 16, 14, 4, 4, 8, 4, 3
    ]


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
        epoch_menu,
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
    assert epoch_menu.currentText() == "E0"
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
    assert _combo_values(data_menu) == ["", "GammaA0", "GammaB1"]
    assert _combo_values(series_menu) == ["", "Gamma"]
    assert channel_menu.currentText() == "A"
    assert epoch_menu.currentText() == "E0"
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
    epoch_menu.setCurrentText("E1")
    data_menu.setCurrentText("GammaB1")
    assert nm.select_values["data"] is folder2.data["GammaB1"]
    assert _combo_values(channel_menu) == [""]
    assert _combo_values(epoch_menu) == [""]
    assert not channel_menu.isEnabled()
    assert not epoch_menu.isEnabled()

    series_menu.setCurrentText("Gamma")

    assert nm.select_values["channel"] is folder2.dataseries["Gamma"].channels["B"]
    assert nm.select_values["epoch"] is folder2.dataseries["Gamma"].epochs["E1"]

    folder_menu.setCurrentIndex(0)
    assert all(value is None for value in win.selection_model.selection.values())
    folder_menu.setCurrentText("Demo1")
    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Alpha"
    assert channel_menu.currentText() == "A"
    assert epoch_menu.currentText() == "E0"
    assert set_menu.currentText() == "Demo1Set"
    assert group_menu.currentText() == "1"
    assert operator_menu.currentText() == "AND"

    folder_menu.setCurrentText("Demo2")
    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Gamma"
    assert channel_menu.currentText() == "B"
    assert epoch_menu.currentText() == "E1"
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
    _, _, series_menu, channel_menu, epoch_menu, set_menu, operator_menu, group_menu = (
        win.selection_strip.combo_boxes()
    )

    set_menu.setCurrentText("AlphaSet")
    group_menu.setCurrentText("1")
    operator_menu.setCurrentText("AND")
    channel_menu.setCurrentText("B")
    epoch_menu.setCurrentText("E1")
    series_menu.setCurrentText("Beta")

    assert nm.select_values["dataseries"] is beta
    assert channel_menu.currentText() == "A"
    assert epoch_menu.currentText() == "E0"
    assert set_menu.currentText() == ""
    assert group_menu.currentText() == ""
    assert operator_menu.currentText() == ""

    set_menu.setCurrentText("BetaSet")
    group_menu.setCurrentText("0")
    operator_menu.setCurrentText("OR")
    series_menu.setCurrentText("Alpha")

    assert nm.select_values["dataseries"] is alpha
    assert channel_menu.currentText() == "B"
    assert epoch_menu.currentText() == "E1"
    assert set_menu.currentText() == "AlphaSet"
    assert group_menu.currentText() == "1"
    assert operator_menu.currentText() == "AND"

    series_menu.setCurrentText("Beta")

    assert channel_menu.currentText() == "A"
    assert epoch_menu.currentText() == "E0"
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
    _, data_menu, series_menu, channel_menu, epoch_menu, set_menu, operator_menu, group_menu = (
        win.selection_strip.combo_boxes()
    )

    assert data_menu.currentText() == ""
    assert series_menu.currentText() == "Record"
    assert channel_menu.isEnabled()
    assert epoch_menu.isEnabled()
    assert _combo_values(set_menu) == ["", "EpochSet"]
    assert _combo_values(group_menu) == ["", "0", "1"]

    channel_menu.setCurrentText("B")
    epoch_menu.setCurrentText("E1")
    set_menu.setCurrentText("EpochSet")
    group_menu.setCurrentText("1")
    operator_menu.setCurrentText("AND")
    data_menu.setCurrentText("RecordA0")

    assert nm.select_values["data"] is folder.data["RecordA0"]
    assert nm.select_values["dataseries"] is None
    assert series_menu.currentText() == ""
    assert not channel_menu.isEnabled()
    assert not epoch_menu.isEnabled()
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
    assert epoch_menu.currentText() == "E1"
    assert channel_menu.isEnabled()
    assert epoch_menu.isEnabled()
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
    group_menu = win.selection_strip.combo_boxes()[7]

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
