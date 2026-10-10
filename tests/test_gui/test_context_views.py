"""Tests for the channel/tool tab bar and the channel table."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.context_views import ChannelToolBar, ChannelTableModel

pytestmark = pytest.mark.gui


def _tab_texts(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def _headers(model):
    return [
        model.headerData(c, QtCore.Qt.Orientation.Horizontal)
        for c in range(model.columnCount())
    ]


def _cell(model, row, column):
    return model.data(model.index(row, column))


@pytest.fixture
def two_channel_window(qtbot):
    """Data series "Record" with channels A, B and epochs E0-E3 (A: k*10 + i)."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(4):
        for channel, scale in (("A", 1.0), ("B", 100.0)):
            folder.data.new(
                f"Record{channel}{epoch}",
                nparray=scale * (epoch * 10 + np.arange(5.0)),
                xscale={"start": 2.0, "delta": 0.5, "units": "ms"},
            )
    dataseries = folder.sync_dataseries("Record", select=True)
    manager.select_keys = {
        "folder": folder.name,
        "dataseries": dataseries.name,
        "channel": "A",
        "epoch": "E1",
    }
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    return window, folder, dataseries


def test_channel_table_shows_every_epoch_as_a_column(two_channel_window):
    window, _folder, dataseries = two_channel_window
    table = window.table_panel
    model = table.channel_table.model

    assert _tab_texts(table.view_tabs) == ["A", "B"]
    assert _headers(model) == ["x (ms)", "RecordA0", "RecordA1", "RecordA2", "RecordA3"]
    assert model.rowCount() == 5
    assert [_cell(model, 2, c) for c in range(5)] == ["3", "2", "12", "22", "32"]
    assert table.channel_table.title_label.text() == "Record  channel A: 4 epochs"
    # The current epoch's column is selected
    assert table.channel_table.selected_array() is dataseries.get_data("A", "E1")

    window.selection_model.update(epoch=dataseries.epochs["E3"])
    assert table.channel_table.selected_array() is dataseries.get_data("A", "E3")


def test_channel_table_follows_set_and_channel_tab(two_channel_window):
    window, _folder, dataseries = two_channel_window
    dataseries.epochs.sets.add("Odd", ["E1", "E3"])
    table = window.table_panel
    model = table.channel_table.model

    window.selection_model.update(set="Odd")
    assert _headers(model) == ["x (ms)", "RecordA1", "RecordA3"]
    assert table.channel_table.title_label.text() == "Record  channel A: 2 epochs (Odd)"

    table.view_tabs.select_channel("B")
    assert _headers(model) == ["x (ms)", "RecordB1", "RecordB3"]
    assert _cell(model, 0, 1) == "1000"
    # The analysis channel is not changed by the table's channel tab
    assert window.selection_model.selection["channel"].name == "A"


def test_channel_table_reports_missing_operator(two_channel_window):
    window, _folder, dataseries = two_channel_window
    dataseries.epochs.sets.add("Odd", ["E1", "E3"])
    dataseries.epochs.groups.assign_cyclic(["E0", "E1", "E2", "E3"], n_groups=2)
    window.selection_model.update(set="Odd", group=0, group_operator=None)
    table = window.table_panel
    assert window.selection_model.selection.get("group_operator") is None
    assert table.channel_table.model.columnCount() == 0
    assert "Choose AND or OR" in table.channel_table.title_label.text()


def test_data_selection_shows_data_tab(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("Trace0", nparray=np.array([1.0, 2.0, 3.0]))
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)

    table = window.table_panel
    assert _tab_texts(table.view_tabs) == ["Data"]
    assert _tab_texts(window.plot_widget.view_tabs) == ["Data"]
    assert _headers(table.channel_table.model) == ["x", "Trace0"]
    assert table.channel_table.title_label.text() == "Trace0"


def test_channel_table_model_formats_cells_on_demand():
    from pyneuromatic.core.nm_data import NMData

    NM = NMManager(quiet=True)
    long = NMData(NM, name="Long", nparray=np.arange(200_000.0),
                  xscale={"start": 0.0, "delta": 0.01})
    short = NMData(NM, name="Short", nparray=np.array([np.nan, 1.5]))
    model = ChannelTableModel()
    model.set_arrays([long, short])

    assert model.rowCount() == 200_000
    assert _cell(model, 199_999, 0) == "1999.99"
    assert _cell(model, 199_999, 1) == "199999"
    assert _cell(model, 0, 2) == "NaN"
    assert _cell(model, 5, 2) == ""  # past the end of the shorter array


def test_tool_bar_keeps_tool_selected_when_channels_change(qtbot):
    bar = ChannelToolBar()
    qtbot.addWidget(bar)
    views = []
    bar.view_changed.connect(lambda kind, name: views.append((kind, name)))

    bar.set_views(["A", "B"], [], current_channel="B")
    assert bar.current_view() == ("channel", "B")
    assert views == []  # set_views is quiet

    bar.set_views(["A", "B"], ["Stats"], current_channel="B")
    assert bar.select_tool("Stats")
    assert views[-1] == ("tool", "Stats")
    assert bar.tabTextColor(2) == ChannelToolBar.TOOL_TEXT_COLOR

    bar.set_views(["A", "B", "C"], ["Stats"], current_channel="A")
    assert bar.current_view() == ("tool", "Stats")

    bar.set_views(["A", "B", "C"], [], current_channel="A")
    assert bar.current_view() == ("channel", "A")
    assert not bar.select_tool("Stats")
