"""Tests for the application shell and app-level GUI wiring."""

import pytest

pytest.importorskip("PyQt6")

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
        "Group / Set",
    ]


def test_app_window_builds_shell(qtbot, nm):
    win = NMAppWindow(nm)
    qtbot.addWidget(win)

    assert win.menuBar() is not None
    assert hasattr(win, "selection_strip")
    assert hasattr(win, "tool_rail")
    assert hasattr(win, "tool_workspace")
    assert hasattr(win, "history_panel")
    assert hasattr(win, "browser_widget")
    assert hasattr(win, "plot_widget")
    assert win.context_panel.tabText(0) == "Plot"
    assert isinstance(win.browser_widget, FolderBrowserWidget)
    assert win.tool_rail.tool_names() == ["Browser", "Main", "Stats", "Spike"]
    win.show()
    qtbot.waitUntil(lambda: win.workspace_splitter.width() > 0)
    workspace_width, plot_width = win.workspace_splitter.sizes()[1:]
    assert workspace_width < plot_width


def test_tool_rail_switches_active_tool(qtbot, nm):
    win = NMAppWindow(nm)
    qtbot.addWidget(win)

    win.tool_rail.select_tool("Stats")

    assert win.current_tool_name == "Stats"
    assert win.tool_workspace.currentIndex() == win.tool_rail.index_for_name("Stats")
