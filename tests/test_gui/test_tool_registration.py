"""Tests for registering tool tabs with the app window (#454)."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import app_window
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tool_tab import ToolTabWidget

pytestmark = pytest.mark.gui


class BareTab(ToolTabWidget):
    """A tool that provides nothing to the window (e.g. Main, at first)."""

    def __init__(self, manager, parent=None):
        super().__init__("Bare", parent)


class FullTab(ToolTabWidget):
    """A tool that provides every optional part."""

    def __init__(self, manager, parent=None):
        super().__init__("Full", parent)
        self.plot_view = QtWidgets.QLabel("plot")
        self.table_view = QtWidgets.QLabel("table")
        self.results = False
        self.option = False
        self.calls = []

    def has_results(self):
        return self.results

    def config_options(self):
        return [("option", "An option", self.option, "")]

    def set_config_option(self, key, value):
        self.option = value

    def refresh(self):
        self.calls.append("refresh")

    def apply_settings(self):
        self.calls.append("apply_settings")

    def close_views(self):
        self.calls.append("close_views")


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setitem(app_window.TOOL_TABS, "Main", BareTab)
    monkeypatch.setitem(app_window.TOOL_TABS, "Spike", FullTab)
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    return window


def test_every_tool_tab_is_registered_from_the_list(window):
    assert window.tool_names == ["Browser", "Main", "Stats", "Spike"]
    assert list(window.tool_tabs) == ["Main", "Stats", "Spike"]
    for name, tab in window.tool_tabs.items():
        assert window.tool_workspace.indexOf(tab) == window.tool_rail.index_for_name(name)


def test_a_tool_that_provides_nothing(window):
    bare = window.tool_tabs["Main"]
    assert window.plot_widget.tool_page("Main") is None
    assert window.table_panel.tool_page("Main") is None
    assert "Main" not in window.settings_dialog().page_titles()
    assert bare.selection == window.selection_model.selection  # still bound


def test_views_appear_while_the_tool_has_results(window):
    full = window.tool_tabs["Spike"]
    assert window.plot_widget.tool_page("Spike") is full.plot_view
    assert window.table_panel.tool_page("Spike") is full.table_view
    assert "Spike" not in window.plot_widget.view_tabs.tool_names
    full.results = True
    full.results_changed.emit()
    assert "Spike" in window.plot_widget.view_tabs.tool_names
    assert "Spike" in window.table_panel.view_tabs.tool_names


def test_view_requested_shows_the_view(window):
    full = window.tool_tabs["Spike"]
    full.results = True
    full.results_changed.emit()
    full.view_requested.emit("table")
    assert window.context_panel.currentWidget() is window.table_panel
    assert window.table_panel.stack.currentWidget() is full.table_view
    full.view_requested.emit("plot")
    assert window.context_panel.currentWidget() is window.plot_widget
    assert window.plot_widget.stack.currentWidget() is full.plot_view


def test_settings_page_and_button(window):
    full = window.tool_tabs["Spike"]
    dialog = window.settings_dialog()
    assert dialog.page_titles() == ["General", "Plot", "Stats", "Spike"]
    dialog.tool_checkboxes["Spike"]["option"].setChecked(True)
    assert full.option is True


def test_history_refresh_settings_and_close(window):
    full = window.tool_tabs["Spike"]
    full.history_entry.emit("tool", "Full: Complete")
    assert "Full: Complete" in window.history_panel.text()
    window._on_content_changed()
    window._apply_settings()
    window.selection_model.manager.settings.confirm_quit = False
    window.close()
    assert full.calls == ["refresh", "apply_settings", "close_views"]
