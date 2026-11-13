"""Tests for the shared channel/tool bar and the Plot | Table toggle (#460)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import app_window
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tool_tab import ToolTabWidget

pytestmark = pytest.mark.gui


class TableOnlyTab(ToolTabWidget):
    """A tool with tables but no plot."""

    def __init__(self, manager, parent=None):
        super().__init__("TableOnly", parent)
        self.table_view = QtWidgets.QLabel("tables")

    def has_results(self):
        return True


@pytest.fixture
def app(qtbot, monkeypatch):
    """Series Record: channels A, B x 3 epochs; Spike is a tool with only tables."""
    monkeypatch.setitem(app_window.TOOL_TABS, "Spike", TableOnlyTab)
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    for channel in series.channels.values():
        for i, d in enumerate(channel.data):
            d.nparray = np.arange(4.0) + 10 * i
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.series = series
    return window


def _tabs(window):
    bar = window.view_tabs
    return [bar.tabText(i) for i in range(bar.count())]


def _choose(window, name):
    """Click the Plot or Table tab."""
    window.panel_tabs.setCurrentIndex(window.PANEL_NAMES.index(name))


def _shown(window):
    """The Plot | Table tab shown as current."""
    return window.panel_tabs.tabText(window.panel_tabs.currentIndex())


def _enabled(window, name):
    return window.panel_tabs.isTabEnabled(window.PANEL_NAMES.index(name))


def test_one_bar_on_the_left_and_the_toggle_on_the_right(app):
    assert app.plot_widget.view_tabs is app.table_panel.view_tabs is app.view_tabs
    assert _tabs(app) == ["A"]  # the strip's channel; Browser: no tool's tab (#523)
    app.tool_rail.select_tool("Spike")
    assert _tabs(app) == ["A", "Spike"]  # the tool chosen's
    app.tool_rail.select_tool("Stats")
    assert _tabs(app) == ["A", "Stats2"]
    header = app.context_area.layout().itemAt(0).layout()
    widgets = [header.itemAt(i).widget() for i in range(header.count())]
    assert widgets[0] is app.view_tabs
    assert widgets[-1] is app.panel_tabs
    assert isinstance(app.panel_tabs, QtWidgets.QTabBar)  # tabs, as A / B (#517)
    assert [app.panel_tabs.tabText(i) for i in range(app.panel_tabs.count())] == [
        "Plot", "Table"]
    assert _shown(app) == "Plot"


def test_switching_plot_and_table_keeps_the_channel(app):
    app.selection_model.update(channel=app.series.channels["B"])
    _choose(app, "Table")
    assert app.context_panel.currentWidget() is app.table_panel
    assert app.view_tabs.current_view() == ("channel", "B")
    assert app.table_panel.channel_table.model.arrays[0].name == "RecordB0"
    _choose(app, "Plot")
    assert app.context_panel.currentWidget() is app.plot_widget
    assert app.view_tabs.current_view() == ("channel", "B")
    assert app.plot_widget.current_trace.name == "RecordB0"


def test_view_menu_and_toggle_agree(app):
    app.panel_view_actions["Table"].trigger()
    assert app.context_panel.currentWidget() is app.table_panel
    assert _shown(app) == "Table"
    _choose(app, "Plot")
    assert app.panel_view_actions["Plot"].isChecked()
    assert not app.panel_view_actions["Table"].isChecked()


def test_showing_a_view_from_elsewhere_moves_the_toggle(app):
    app._show_data_table()  # e.g. the Data menu's Show in Table
    assert _shown(app) == "Table"
    app.tool_rail.select_tool("Spike")
    app.tool_tabs["Spike"].view_requested.emit("table")
    assert app.view_tabs.current_view() == ("tool", "Spike")
    assert _shown(app) == "Table"


def test_a_tool_without_a_plot(app):
    app.tool_rail.select_tool("Spike")
    app.view_tabs.select_tool("Spike")
    # On Plot, choosing a tool with only tables shows its tables
    assert app.context_panel.currentWidget() is app.table_panel
    assert app.table_panel.stack.currentWidget() is app.tool_tabs["Spike"].table_view
    assert not _enabled(app, "Plot")
    assert not app.panel_view_actions["Plot"].isEnabled()
    # The Plot panel's page for it says so
    assert app.plot_widget.stack.currentWidget() is app.plot_widget.no_view_label
    assert app.plot_widget.no_view_label.text() == "Spike has no plot"
    app.view_tabs.select_channel("A")
    assert _enabled(app, "Plot")
