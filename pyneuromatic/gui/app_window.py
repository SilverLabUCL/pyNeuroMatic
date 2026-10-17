# -*- coding: utf-8 -*-
"""Application shell for the pyNeuroMatic GUI.

This module provides the initial shell for the desktop UI: a top-level menu bar,
a persistent selection strip, a tool rail, a central tool workspace, a dynamic
right-hand context panel, and a history/results panel.
"""
from __future__ import annotations

from math import sin

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.context_views import (
    DATA_VIEW,
    ChannelToolBar,
    TablePanel,
    ValuesPlot,
)
from pyneuromatic.gui.folder_browser import FolderBrowserWidget
from pyneuromatic.gui.selection_model import (
    SelectionModel,
    selection_summary,
    source_object,
)
from pyneuromatic.gui.selection_strip import SelectionStrip
from pyneuromatic.gui.tier_actions import (
    DataActions,
    EpochSetActions,
    FolderActions,
    SeriesActions,
)
from pyneuromatic.gui.stats_tab import StatsToolTab


class ToolRail(QtWidgets.QWidget):
    """Vertical tool selector used to switch between tool workspaces."""

    tool_selected = QtCore.pyqtSignal(str)

    def __init__(self, tool_names: list[str] | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tool_names = list(tool_names or ["Main", "Stats", "Spike"])
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        self._current_tool: str = self._tool_names[0] if self._tool_names else ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        for name in self._tool_names:
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=name: self.select_tool(value))
            self._buttons[name] = button
            layout.addWidget(button)

        if self._tool_names:
            self.select_tool(self._tool_names[0], emit=False)

    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    def index_for_name(self, name: str) -> int:
        try:
            return self._tool_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Tool '{name}' not found") from exc

    def select_tool(self, name: str, emit: bool = True) -> None:
        if name not in self._buttons:
            raise KeyError(f"Tool '{name}' not found")
        self._current_tool = name
        for tool_name, button in self._buttons.items():
            button.setChecked(tool_name == name)
        if emit:
            self.tool_selected.emit(name)

    @property
    def current_tool_name(self) -> str:
        return self._current_tool


class _PlaceholderToolWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        label = QtWidgets.QLabel(f"{title} tool")
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(label)

        info = QtWidgets.QLabel(
            "This is a placeholder for the active tool workspace. "
            "Tool-specific controls will be added here."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()


class PlotPanel(QtWidgets.QWidget):
    """Plot panel with a trace navigator for data and dataseries modes."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.selection_label = QtWidgets.QLabel("No selection", self)
        layout.addWidget(self.selection_label)

        # Channel tabs, then tabs of tools with results (e.g. Stats)
        self.view_tabs = ChannelToolBar(self)
        self.view_tabs.view_changed.connect(self._on_view_changed)
        layout.addWidget(self.view_tabs)
        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack, stretch=1)
        self.trace_page = QtWidgets.QWidget(self.stack)
        trace_layout = QtWidgets.QVBoxLayout(self.trace_page)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.trace_page)
        self._tool_pages: dict[str, QtWidgets.QWidget] = {}
        self._tools: list[str] = []

        navigator_layout = QtWidgets.QHBoxLayout()
        self.trace_previous_button = QtWidgets.QToolButton(self)
        self.trace_previous_button.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.trace_previous_button.setToolTip("Previous trace")
        self.trace_previous_button.setFixedSize(28, 28)
        navigator_layout.addWidget(self.trace_previous_button)

        self.trace_index = QtWidgets.QSpinBox(self)
        self.trace_index.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trace_index.setFixedWidth(72)
        self.trace_index.setRange(0, 0)
        navigator_layout.addWidget(self.trace_index)

        self.trace_count_label = QtWidgets.QLabel("of 0", self)
        navigator_layout.addWidget(self.trace_count_label)

        self.trace_next_button = QtWidgets.QToolButton(self)
        self.trace_next_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.trace_next_button.setToolTip("Next trace")
        self.trace_next_button.setFixedSize(28, 28)
        navigator_layout.addWidget(self.trace_next_button)

        self.trace_name_label = QtWidgets.QLabel("No trace", self)
        navigator_layout.addWidget(self.trace_name_label, stretch=1)
        trace_layout.addLayout(navigator_layout)

        overlay_layout = QtWidgets.QHBoxLayout()
        self.overlay_checkbox = QtWidgets.QCheckBox("Overlay traces", self)
        self.overlay_checkbox.setToolTip(
            "Plot nearby epochs together for the selected data series and channel"
        )
        overlay_layout.addWidget(self.overlay_checkbox)
        overlay_layout.addWidget(QtWidgets.QLabel("Limit", self))
        self.overlay_limit = QtWidgets.QSpinBox(self)
        self.overlay_limit.setRange(1, 2000)
        self.overlay_limit.setValue(50)
        self.overlay_limit.setFixedWidth(72)
        overlay_layout.addWidget(self.overlay_limit)
        self.overlay_status_label = QtWidgets.QLabel("", self)
        overlay_layout.addWidget(self.overlay_status_label, stretch=1)
        trace_layout.addLayout(overlay_layout)

        self._selection_model: SelectionModel | None = None
        self._trace_entries: list[tuple[object, object | None]] = []
        self._trace_mode: str | None = None
        self._updating_navigator = False
        self._updating_channel_tabs = False
        self._plot_dataseries = None
        self._plot_channel_name: str | None = None
        self._last_selection: dict = {}
        self.current_trace = None
        self.rendered_trace_count = 0
        self.trace_index.valueChanged.connect(self._on_trace_index_changed)
        self.trace_previous_button.clicked.connect(lambda: self._step_trace(-1))
        self.trace_next_button.clicked.connect(lambda: self._step_trace(1))
        self.overlay_checkbox.toggled.connect(self._render_current_traces)
        self.overlay_limit.valueChanged.connect(self._render_current_traces)

        if pg is not None:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setBackground("w")
            self.plot_widget.setTitle("Preview")
            trace_layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            self.plot_fallback_label = QtWidgets.QLabel(
                "Plot view unavailable. Install pyqtgraph to enable preview plots."
            )
            self.plot_fallback_label.setWordWrap(True)
            self.plot_fallback_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            trace_layout.addWidget(self.plot_fallback_label)

    # --- tool views -------------------------------------------------------

    def register_tool(self, name: str, page: QtWidgets.QWidget) -> None:
        self._tool_pages[name] = page
        self.stack.addWidget(page)

    def tool_page(self, name: str) -> QtWidgets.QWidget | None:
        return self._tool_pages.get(name)

    def set_tools(self, names: list[str]) -> None:
        """Show tabs for the registered tools in *names*, in that order."""
        self._tools = [name for name in names if name in self._tool_pages]
        self._update_channel_tabs(self._last_selection)

    def show_tool(self, name: str) -> None:
        self.view_tabs.select_tool(name)

    def _show_current_page(self) -> None:
        view = self.view_tabs.current_view()
        if view is not None and view[0] == "tool":
            self.stack.setCurrentWidget(self._tool_pages[view[1]])
        else:
            self.stack.setCurrentWidget(self.trace_page)

    def bind_selection_model(self, model: SelectionModel) -> None:
        self._selection_model = model
        model.selection_changed.connect(self.update_selection)
        self.update_selection(model.selection)

    def update_selection(self, selection: dict) -> None:
        self._last_selection = selection.copy()
        selected = selection_summary(selection)
        self.selection_label.setText(
            "Selection: " + " / ".join(selected) if selected else "No selection"
        )
        self._update_channel_tabs(selection)
        self._update_trace_entries(selection)

    def _update_channel_tabs(self, selection: dict) -> None:
        dataseries = source_object(selection, "dataseries")
        if dataseries is None:
            self._plot_dataseries = None
            self._plot_channel_name = None
            channels = [DATA_VIEW] if source_object(selection, "data") is not None else []
            self._set_views(channels, DATA_VIEW)
            return

        channel_names = list(dataseries.channels.keys())
        previous_channel = self._plot_channel_name
        if previous_channel not in channel_names:
            selected_channel = selection.get("channel")
            selected_name = selected_channel.name if selected_channel is not None else None
            previous_channel = (
                selected_name if selected_name in channel_names
                else channel_names[0] if channel_names else None
            )

        self._plot_dataseries = dataseries
        self._plot_channel_name = previous_channel
        self._set_views(channel_names, previous_channel)

    def _set_views(self, channels: list[str], current_channel: str | None) -> None:
        self._updating_channel_tabs = True
        self.view_tabs.set_views(channels, self._tools, current_channel=current_channel)
        self._updating_channel_tabs = False
        self._show_current_page()

    def _on_view_changed(self, kind: str, name: str) -> None:
        if self._updating_channel_tabs:
            return
        self._show_current_page()
        if kind == "channel":
            if name != DATA_VIEW:
                self._plot_channel_name = name
            self._update_trace_entries(self._last_selection)

    def _update_trace_entries(self, selection: dict) -> None:
        entries: list[tuple[object, object | None]] = []
        mode = None
        current_item = None

        data = source_object(selection, "data")
        dataseries = source_object(selection, "dataseries")
        if data is not None:
            folder = selection.get("folder")
            context = selection.get("toolfolder") or folder
            if context is not None:
                entries = [(item, item) for item in context.data.values()]
                mode = "data"
                current_item = data
        elif dataseries is not None:
            channel = dataseries.channels.get(self._plot_channel_name)
            if channel is None:
                channel = selection.get("channel")
            if channel is None:
                channel = dataseries.channels.selected_value
            if channel is None and dataseries.channels.values():
                channel = dataseries.channels.values()[0]
            if channel is not None:
                entries = [
                    (epoch, dataseries.get_data(channel=channel.name, epoch=epoch.name))
                    for epoch in dataseries.epochs.values()
                ]
                mode = "dataseries"
                current_item = selection.get("epoch") or dataseries.epochs.selected_value

        self._trace_entries = entries
        self._trace_mode = mode
        if not entries:
            self._sync_trace_controls(0)
            self.trace_name_label.setText("No trace")
            self.current_trace = None
            self._render_current_traces()
            return

        current_index = next(
            (index for index, (item, _) in enumerate(entries) if item is current_item),
            0,
        )
        self._sync_trace_controls(current_index)
        item, self.current_trace = entries[current_index]
        self.trace_name_label.setText(
            self.current_trace.name
            if self.current_trace is not None
            else f"{item.name} (no data)"
        )
        self._render_current_traces()

    def _sync_trace_controls(self, current_value: int) -> None:
        count = len(self._trace_entries)
        self._updating_navigator = True
        self.trace_index.setRange(0, count - 1) if count else self.trace_index.setRange(0, 0)
        if count:
            self.trace_index.setValue(current_value)
        self.trace_index.setEnabled(count > 0)
        self.trace_count_label.setText(f"of {count}")
        self.trace_previous_button.setEnabled(current_value > 0 and count > 0)
        self.trace_next_button.setEnabled(current_value < count - 1 and count > 0)
        can_overlay = self._trace_mode == "dataseries" and self.plot_widget is not None
        self.overlay_checkbox.setEnabled(can_overlay)
        self.overlay_limit.setEnabled(can_overlay and self.overlay_checkbox.isChecked())
        self._updating_navigator = False

    def _step_trace(self, delta: int) -> None:
        if self.trace_index.isEnabled():
            self.trace_index.setValue(self.trace_index.value() + delta)

    def _on_trace_index_changed(self, value: int) -> None:
        if self._updating_navigator or self._selection_model is None:
            return
        index = value
        if not 0 <= index < len(self._trace_entries):
            return
        item, data = self._trace_entries[index]
        if self._trace_mode == "data":
            self._selection_model.update(data=item)
        elif self._trace_mode == "dataseries":
            self._selection_model.update(epoch=item)

    def _render_current_traces(self, *_args) -> None:
        if self.plot_widget is None:
            return
        self.plot_widget.clear()
        self.rendered_trace_count = 0

        if not self._trace_entries:
            self.trace_count_label.setText("of 0")
            self.overlay_status_label.setText("")
            self.plot_widget.setTitle("Preview")
            return

        current_index = max(0, min(self.trace_index.value(), len(self._trace_entries) - 1))
        overlay = self.overlay_checkbox.isChecked() and self._trace_mode == "dataseries"
        if overlay:
            total = len(self._trace_entries)
            limit = min(self.overlay_limit.value(), total)
            first_index = max(0, min(current_index - limit // 2, total - limit))
            last_index = first_index + limit
            entries = self._trace_entries[first_index:last_index]
            self.overlay_status_label.setText(f"Showing {len(entries)} of {total} traces")
        else:
            entries = [self._trace_entries[current_index]]
            self.overlay_status_label.setText("")

        current_data = self._trace_entries[current_index][1]
        if current_data is not None:
            self.plot_widget.setLabel(
                "bottom", current_data.xscale.label, units=current_data.xscale.units
            )
            self.plot_widget.setLabel(
                "left", current_data.yscale.label, units=current_data.yscale.units
            )

        for item, data in entries:
            if data is None or data.nparray is None:
                continue
            x_values = self._x_values(data)
            is_current = item is self._trace_entries[current_index][0]
            color = "#2563eb" if is_current else "#78909c"
            width = 2 if is_current else 1
            curve = self.plot_widget.plot(
                x_values,
                data.nparray,
                pen=pg.mkPen(color=color, width=width, alpha=255 if is_current else 150),
            )
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.rendered_trace_count += 1

        selected_item = self._trace_entries[current_index][0]
        suffix = " (overlay)" if overlay else ""
        self.plot_widget.setTitle(f"{selected_item.name}{suffix}")

    @staticmethod
    def _x_values(data):
        if data.xarray is not None:
            return data.xarray
        x_values = [data.get_xvalue(index) for index in range(data.nparray.size)]
        if any(value is None for value in x_values):
            return list(range(data.nparray.size))
        return x_values


class NMAppWindow(QtWidgets.QMainWindow):
    """Application shell for pyNeuroMatic.

    The layout follows the legacy NeuroMatic panel idea:
    - menu bar at the top
    - selection strip below
    - tool rail on the left
    - central tool/browser workspace
    - dynamic right-panel for plot/inspector
    - history/results dock at the bottom

    """

    def __init__(self, manager: NMManager | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager or NMManager(quiet=True)
        self.selection_model = SelectionModel(self._manager, self)
        self.current_tool_name = "Browser"

        self.setWindowTitle("pyNeuroMatic")
        self.resize(1200, 800)

        # Tier actions, shared by the strip's ⋯ menus and the menu bar
        self.folder_actions = FolderActions(self.selection_model, self)
        self.data_actions = DataActions(self.selection_model, self)
        self.series_actions = SeriesActions(self.selection_model, self)
        self.epoch_set_actions = EpochSetActions(self.selection_model, self)

        self._build_menu()
        self._build_selection_strip()
        self._build_tool_shell()
        self._build_history_panel()
        self.selection_model.selection_changed.connect(self._append_selection_history)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.folder_actions.open_action)
        file_menu.addAction(self.folder_actions.save_action)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit))
        quit_action.triggered.connect(self.close)

        folder_menu = self.folder_actions.folder_menu(self)
        folder_menu.addSeparator()
        folder_menu.addMenu(self.folder_actions.subfolder_menu(folder_menu))
        self.menuBar().addMenu(folder_menu)

        data_menu = self.data_actions.data_menu(self)
        data_menu.addSeparator()
        data_menu.addMenu(self.data_actions.data_set_menu(data_menu))
        self.menuBar().addMenu(data_menu)

        series_menu = self.series_actions.series_menu(self)
        series_menu.addSeparator()
        series_menu.addMenu(self.epoch_set_actions.epoch_set_menu(series_menu))
        self.menuBar().addMenu(series_menu)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction("Rename")
        edit_menu.addAction("Delete")

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Browser")
        view_menu.addAction("Plot")
        view_menu.addAction("History")

        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction("Main")
        tools_menu.addAction("Stats")
        tools_menu.addAction("Spike")

    def _build_selection_strip(self) -> None:
        # Placed above the workspace in _build_tool_shell, leaving the menu
        # bar in its usual place
        self.selection_strip = SelectionStrip(self)
        self.selection_strip.bind_selection_model(self.selection_model)
        strip = self.selection_strip
        strip.set_actions_menu("folder", self.folder_actions.folder_menu(strip))
        strip.set_actions_menu("toolfolder", self.folder_actions.subfolder_menu(strip))
        strip.set_actions_menu("data", self.data_actions.data_menu(strip))
        strip.set_actions_menu("data_set", self.data_actions.data_set_menu(strip))
        strip.set_actions_menu("dataseries", self.series_actions.series_menu(strip))
        strip.set_actions_menu("epoch_set", self.epoch_set_actions.epoch_set_menu(strip))

    def _build_tool_shell(self) -> None:
        self.tool_rail = ToolRail(["Browser", "Main", "Stats", "Spike"], self)
        self.tool_rail.tool_selected.connect(self._on_tool_selected)

        self.tool_workspace = QtWidgets.QStackedWidget(self)
        if self._manager is not None:
            browser_widget = FolderBrowserWidget(self._manager, self.tool_workspace)
            browser_widget.selection_requested.connect(self._on_browser_selection_requested)
        else:
            browser_widget = QtWidgets.QTreeWidget()
            browser_widget.setHeaderLabels(["Browser"])
            browser_widget.addTopLevelItem(QtWidgets.QTreeWidgetItem(["Folders"]))
            browser_widget.addTopLevelItem(QtWidgets.QTreeWidgetItem(["Data"]))
        self.browser_widget = browser_widget
        self.tool_workspace.addWidget(browser_widget)

        self.tool_workspace.addWidget(
            _PlaceholderToolWidget("Main", self.tool_workspace)
        )
        self.stats_tab = StatsToolTab(self._manager, self.tool_workspace)
        self.stats_tab.bind_selection_model(self.selection_model)
        self.tool_workspace.addWidget(self.stats_tab)
        self.tool_workspace.addWidget(
            _PlaceholderToolWidget("Spike", self.tool_workspace)
        )

        self.context_panel = QtWidgets.QTabWidget(self)
        self.context_panel.setTabsClosable(False)
        self.plot_widget = PlotPanel(self.context_panel)
        self.plot_widget.bind_selection_model(self.selection_model)
        self.context_panel.addTab(self.plot_widget, "Plot")
        self.table_panel = TablePanel(self.context_panel)
        self.table_panel.bind_selection_model(self.selection_model)
        self.context_panel.addTab(self.table_panel, "Table")

        # Tool views appear in both panels while the folder has their results
        stats2 = self.stats_tab.stats2
        self.stats_plot = ValuesPlot()
        self.plot_widget.register_tool("Stats", self.stats_plot)
        self.table_panel.register_tool("Stats", stats2.tables)
        stats2.results_changed.connect(self._update_tool_tabs)
        self.folder_actions.content_changed.connect(self._on_content_changed)
        self.data_actions.content_changed.connect(self._on_content_changed)
        self.series_actions.content_changed.connect(self._on_content_changed)
        self.series_actions.browse_requested.connect(self._show_object_in_browser)
        self.epoch_set_actions.content_changed.connect(self._on_content_changed)
        self.epoch_set_actions.browse_requested.connect(self._show_epoch_sets_in_browser)
        self.data_actions.browse_requested.connect(self._show_in_browser)
        self.data_actions.table_requested.connect(self._show_data_table)
        self.table_panel.data_edited.connect(self._on_data_edited)
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.content_changed.connect(self._on_browser_content_changed)
            self.browser_widget.data_created.connect(self._on_browser_data_created)
        # A run or a deleted folder changes the Subfolder list
        stats2.results_changed.connect(self.selection_strip.refresh_options)
        stats2.plot_requested.connect(self._show_plot_values)
        self.stats_tab.run_finished.connect(self._on_stats_run_finished)
        self._update_tool_tabs()
        # Future: Inspector tab reserved for later development.
        self.context_panel.setMinimumWidth(280)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.tool_rail)
        splitter.addWidget(self.tool_workspace)
        splitter.addWidget(self.context_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([120, 350, 700])
        self.workspace_splitter = splitter

        container = QtWidgets.QWidget(self)
        container.setLayout(QtWidgets.QVBoxLayout())
        container.layout().addWidget(self.selection_strip)
        container.layout().addWidget(splitter, stretch=1)
        self.setCentralWidget(container)

    def _build_history_panel(self) -> None:
        history_widget = QtWidgets.QWidget(self)
        history_layout = QtWidgets.QVBoxLayout(history_widget)
        history_layout.setContentsMargins(6, 4, 6, 4)

        label = QtWidgets.QLabel("History")
        label.setStyleSheet("font-weight: bold;")
        history_layout.addWidget(label)

        self.history_panel = QtWidgets.QPlainTextEdit(history_widget)
        self.history_panel.setReadOnly(True)
        self.history_panel.setPlainText("Welcome to pyNeuroMatic\n")
        history_layout.addWidget(self.history_panel)

        dock = QtWidgets.QDockWidget("History", self)
        dock.setWidget(history_widget)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _on_tool_selected(self, tool_name: str) -> None:
        self.current_tool_name = tool_name
        idx = self.tool_rail.index_for_name(tool_name)
        self.tool_workspace.setCurrentIndex(idx)

    def closeEvent(self, event) -> None:
        # Tear pyqtgraph views down before Qt deletes them, so a queued paint
        # event never reaches a half-deleted plot (this crashed the GUI tests)
        for view in (
            getattr(self.plot_widget, "plot_widget", None),
            getattr(getattr(self, "stats_plot", None), "plot_widget", None),
        ):
            if view is not None and not getattr(view, "closed", False):
                view.close()
        super().closeEvent(event)

    def _on_content_changed(self) -> None:
        """Reload views after folders, subfolders, data or sets change."""
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh()
        self._on_browser_content_changed()

    def _on_browser_content_changed(self) -> None:
        self.selection_model.refresh()
        self.selection_strip.refresh_options()
        self.stats_tab.stats2.refresh()
        self.folder_actions.update_enabled()
        self.data_actions.update_enabled()
        self.series_actions.update_enabled()
        self.epoch_set_actions.update_enabled()
        # Names or data series membership may have changed
        selection = self.selection_model.selection
        self.plot_widget.update_selection(selection)
        self.table_panel.update_selection(selection)

    def _show_in_browser(self, owner, sets: bool = False) -> None:
        """Switch to the Browser showing *owner*'s Data list, or its data Sets,
        with the current data array (or Data Set) selected."""
        self.tool_rail.select_tool("Browser")
        if not isinstance(self.browser_widget, FolderBrowserWidget):
            return
        selection = self.selection_model.selection
        if sets:
            name = selection.get("data_set")
        else:
            data = selection.get("data")
            name = data.name if data is not None else None
        self.browser_widget.show_data_of(owner, sets=sets, select=name)

    def _show_object_in_browser(self, obj) -> None:
        """Switch to the Browser with *obj* (e.g. a data series) selected."""
        self.tool_rail.select_tool("Browser")
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.show_object(obj)

    def _show_epoch_sets_in_browser(self, series, set_name) -> None:
        """Switch to the Browser with *series*' epoch set (or Sets) selected."""
        self.tool_rail.select_tool("Browser")
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.show_data_of(series, sets=True, select=set_name, kind="epochs")

    def _show_data_table(self) -> None:
        """Show the current data array in the Table panel, ready to edit."""
        self.table_panel.update_selection(self.selection_model.selection)
        self.table_panel.show_channel_table()
        self.context_panel.setCurrentWidget(self.table_panel)

    def _on_browser_data_created(self, data) -> None:
        self.selection_model.update(data=data)
        self._show_data_table()

    def _on_data_edited(self, _data) -> None:
        # Redraw the trace with the edited values
        self.plot_widget.update_selection(self.selection_model.selection)

    def _update_tool_tabs(self) -> None:
        tools = ["Stats"] if self.stats_tab.stats2.has_results() else []
        self.plot_widget.set_tools(tools)
        self.table_panel.set_tools(tools)

    def _show_plot_values(self, title, y, x, xlabel, ylabel, style) -> None:
        self.stats_plot.show_values(title, y, x, xlabel, ylabel, style)
        self.plot_widget.show_tool("Stats")
        self.context_panel.setCurrentWidget(self.plot_widget)

    def _on_stats_run_finished(self, succeeded: bool) -> None:
        if succeeded and self.stats_tab.stats_tool.toolfolder is not None:
            self.table_panel.show_tool("Stats")
            self.context_panel.setCurrentWidget(self.table_panel)

    def _on_browser_selection_requested(self, selection: dict) -> None:
        obj = selection.get("object")
        set_name = selection.get("set")
        if obj is None:
            self.selection_model.clear()
            if set_name is not None:
                self.selection_model.update(data_set=set_name)
            return
        tier = SelectionModel.tier_for_object(obj)
        # Sets of a data series' epochs are epoch sets; others are data sets
        set_key = "epoch_set" if tier in ("dataseries", "channel", "epoch") else "data_set"
        self.selection_model.update({tier: obj, set_key: set_name})

    def _append_selection_history(self, selection: dict) -> None:
        selected = selection_summary(selection)
        summary = " / ".join(selected) if selected else "cleared"
        self.history_panel.appendPlainText(f"Selection: {summary}")

    @property
    def tool_names(self) -> list[str]:
        return self.tool_rail.tool_names()


__all__ = [
    "SelectionStrip",
    "ToolRail",
    "NMAppWindow",
]
