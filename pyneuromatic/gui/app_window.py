# -*- coding: utf-8 -*-
"""Application shell for the pyNeuroMatic GUI.

This module provides the initial shell for the desktop UI: a top-level menu bar,
a persistent selection strip, a tool rail, a central tool workspace, a dynamic
right-hand context panel, and a history/results panel.
"""
from __future__ import annotations

from math import sin

from PyQt6 import QtCore, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core import nm_utilities
from pyneuromatic.gui.folder_browser import FolderBrowserWidget
from pyneuromatic.gui.selection_model import SelectionModel
from pyneuromatic.gui.stats_tab import StatsToolTab


class SelectionStrip(QtWidgets.QWidget):
    """Persistent selection strip across the top of the main app window."""

    selection_changed = QtCore.pyqtSignal(dict)

    _LABELS = (
        "Folder",
        "Data",
        "Data Series",
        "Channel",
        "Set",
        "Operator",
        "Group",
    )
    _TIERS = {
        "Folder": "folder",
        "Data": "data",
        "Data Series": "dataseries",
        "Channel": "channel",
        "Set": "set",
        "Operator": "group_operator",
        "Group": "group",
    }
    _MIN_CONTENT_LENGTH = {
        "Folder": 14,
        "Data": 16,
        "Data Series": 14,
        "Channel": 4,
        "Set": 8,
        "Operator": 4,
        "Group": 3,
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels = list(self._LABELS)
        self._combos: list[QtWidgets.QComboBox] = []
        self._selection_model: SelectionModel | None = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        for label in self._labels:
            label_widget = QtWidgets.QLabel(f"{label}:")
            combo = QtWidgets.QComboBox()
            combo.setMinimumContentsLength(self._MIN_CONTENT_LENGTH[label])
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.addItem("")
            self._combos.append(combo)

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(label_widget)
            row.addWidget(combo)
            layout.addLayout(row)

        spacer = QtWidgets.QSpacerItem(
            20,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        layout.addItem(spacer)

    def count(self) -> int:
        return len(self._labels)

    def combo_label(self, index: int) -> str:
        if not 0 <= index < len(self._labels):
            raise IndexError("selector index out of range")
        return self._labels[index]

    def combo_boxes(self) -> list[QtWidgets.QComboBox]:
        return list(self._combos)

    def set_values(self, values: dict[str, str]) -> None:
        for index, label in enumerate(self._labels):
            combo = self._combos[index]
            key = self._TIERS[label]
            value = values.get(key, values.get(label.lower(), ""))
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            if value:
                combo.addItem(value)
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def bind_selection_model(self, model: SelectionModel) -> None:
        self._selection_model = model
        model.selection_changed.connect(self._update_from_selection)
        for index, label in enumerate(self._labels):
            key = self._TIERS[label]
            self._combos[index].currentIndexChanged.connect(
                lambda combo_index, tier=key, combo=self._combos[index]:
                self._on_selector_changed(tier, combo.itemData(combo_index))
            )
        self._update_from_selection(model.selection)

    def _on_selector_changed(self, tier: str, value) -> None:
        if self._selection_model is not None:
            self._selection_model.update(**{tier: value})

    def refresh_options(self) -> None:
        if self._selection_model is not None:
            self._update_from_selection(self._selection_model.selection)

    def _update_from_selection(self, selection: dict) -> None:
        if self._selection_model is None:
            return
        options = self._options_for_selection(selection)
        for index, label in enumerate(self._labels):
            tier = self._TIERS[label]
            self._set_combo_options(
                self._combos[index],
                options.get(tier, []),
                selection.get(tier),
            )

    def _options_for_selection(self, selection: dict) -> dict[str, list[tuple[str, object]]]:
        manager = self._selection_model.manager
        options: dict[str, list[tuple[str, object]]] = {
            "folder": [(name, name) for name in manager.folders.keys()],
            "data": [],
            "dataseries": [],
            "channel": [],
            "epoch": [],
            "set": [],
            "group_operator": [],
            "group": [],
        }

        folder = selection.get("folder")
        if folder is None:
            return options
        context = selection.get("toolfolder") or folder
        options["data"] = [(name, name) for name in context.data.keys()]
        options["dataseries"] = [(name, name) for name in context.dataseries.keys()]

        dataseries = selection.get("dataseries")
        data = selection.get("data")
        if dataseries is not None:
            options["channel"] = [(name, name) for name in dataseries.channels.keys()]
            options["epoch"] = [(name, name) for name in dataseries.epochs.keys()]

        if dataseries is not None:
            selection_container = dataseries.epochs
            group_container = dataseries.epochs
        elif data is not None:
            selection_container = context.data
            data_series = data._dataseries
            if data_series is None:
                parsed_name = nm_utilities.parse_data_name(data.name)
                if parsed_name is not None:
                    prefix, _, _ = parsed_name
                    if prefix in context.dataseries:
                        data_series = context.dataseries[prefix]
            group_container = data_series.epochs if data_series is not None else None
        else:
            selection_container = context.data
            group_container = None
        options["set"] = [
            (name, name) for name in selection_container.sets.keys()
        ]
        if group_container is not None:
            options["group"] = [
                (str(number), number)
                for number in group_container.groups.group_numbers
            ]
        if selection.get("set") is not None and selection.get("group") is not None:
            options["group_operator"] = [("AND", "AND"), ("OR", "OR")]
        return options

    @staticmethod
    def _set_combo_options(
        combo: QtWidgets.QComboBox,
        options: list[tuple[str, object]],
        selected,
    ) -> None:
        if hasattr(selected, "name"):
            selected = selected.name
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", None)
        for text, value in options:
            combo.addItem(text, value)
        selected_index = combo.findData(selected)
        combo.setCurrentIndex(max(selected_index, 0))
        combo.setEnabled(bool(options))
        combo.blockSignals(False)


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

        self.channel_tabs = QtWidgets.QTabBar(self)
        self.channel_tabs.setShape(QtWidgets.QTabBar.Shape.RoundedNorth)
        self.channel_tabs.setUsesScrollButtons(True)
        self.channel_tabs.setExpanding(False)
        self.channel_tabs.currentChanged.connect(self._on_plot_channel_changed)
        self.channel_tabs.hide()
        layout.addWidget(self.channel_tabs)

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
        layout.addLayout(navigator_layout)

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
        layout.addLayout(overlay_layout)

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
            layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            self.plot_fallback_label = QtWidgets.QLabel(
                "Plot view unavailable. Install pyqtgraph to enable preview plots."
            )
            self.plot_fallback_label.setWordWrap(True)
            self.plot_fallback_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.plot_fallback_label)

    def bind_selection_model(self, model: SelectionModel) -> None:
        self._selection_model = model
        model.selection_changed.connect(self.update_selection)
        self.update_selection(model.selection)

    def update_selection(self, selection: dict) -> None:
        self._last_selection = selection.copy()
        selected = [
            value.name if hasattr(value, "name") else str(value)
            for value in selection.values()
            if value is not None
        ]
        self.selection_label.setText(
            "Selection: " + " / ".join(selected) if selected else "No selection"
        )
        self._update_channel_tabs(selection)
        self._update_trace_entries(selection)

    def _update_channel_tabs(self, selection: dict) -> None:
        dataseries = selection.get("dataseries")
        if dataseries is None:
            self.channel_tabs.hide()
            self._plot_dataseries = None
            self._plot_channel_name = None
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

        current_names = [
            self.channel_tabs.tabText(index)
            for index in range(self.channel_tabs.count())
        ]
        self._updating_channel_tabs = True
        if current_names != channel_names:
            while self.channel_tabs.count():
                self.channel_tabs.removeTab(0)
            for name in channel_names:
                self.channel_tabs.addTab(name)
        if previous_channel in channel_names:
            self.channel_tabs.setCurrentIndex(channel_names.index(previous_channel))
        self.channel_tabs.setVisible(bool(channel_names))
        self._updating_channel_tabs = False

        self._plot_dataseries = dataseries
        self._plot_channel_name = previous_channel

    def _on_plot_channel_changed(self, index: int) -> None:
        if self._updating_channel_tabs or index < 0:
            return
        self._plot_channel_name = self.channel_tabs.tabText(index)
        self._update_trace_entries(self._last_selection)

    def _update_trace_entries(self, selection: dict) -> None:
        entries: list[tuple[object, object | None]] = []
        mode = None
        current_item = None

        data = selection.get("data")
        dataseries = selection.get("dataseries")
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

        self._build_menu()
        self._build_selection_strip()
        self._build_tool_shell()
        self._build_history_panel()
        self.selection_model.selection_changed.connect(self._append_selection_history)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Quit")

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
        self.selection_strip = SelectionStrip(self)
        self.selection_strip.bind_selection_model(self.selection_model)
        self.setMenuWidget(self.selection_strip)

    def _build_tool_shell(self) -> None:
        self.tool_rail = ToolRail(["Browser", "Main", "Stats", "Spike"], self)
        self.tool_rail.tool_selected.connect(self._on_tool_selected)

        self.tool_workspace = QtWidgets.QStackedWidget(self)
        if self._manager is not None:
            browser_widget = FolderBrowserWidget(self._manager, self.tool_workspace)
            browser_widget.selection_requested.connect(self._on_browser_selection_requested)
            browser_widget.content_changed.connect(self.selection_strip.refresh_options)
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
        container.layout().addWidget(splitter)
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

    def _on_browser_selection_requested(self, selection: dict) -> None:
        obj = selection.get("object")
        set_name = selection.get("set")
        if obj is None:
            self.selection_model.clear()
            if set_name is not None:
                self.selection_model.update(set=set_name)
            return
        changes = {"set": set_name}
        changes[SelectionModel.tier_for_object(obj)] = obj
        self.selection_model.update(changes)

    def _append_selection_history(self, selection: dict) -> None:
        selected = [
            value.name if hasattr(value, "name") else str(value)
            for value in selection.values()
            if value is not None
        ]
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
