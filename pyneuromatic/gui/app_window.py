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


class SelectionStrip(QtWidgets.QWidget):
    """Persistent selection strip across the top of the main app window."""

    selection_changed = QtCore.pyqtSignal(dict)

    _LABELS = (
        "Folder",
        "Data",
        "Data Series",
        "Channel",
        "Epoch",
        "Set",
        "Operator",
        "Group",
    )
    _TIERS = {
        "Folder": "folder",
        "Data": "data",
        "Data Series": "dataseries",
        "Channel": "channel",
        "Epoch": "epoch",
        "Set": "set",
        "Operator": "group_operator",
        "Group": "group",
    }
    _MIN_CONTENT_LENGTH = {
        "Folder": 14,
        "Data": 16,
        "Data Series": 14,
        "Channel": 4,
        "Epoch": 4,
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
    """Simple plot panel for the right-hand context area."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.selection_label = QtWidgets.QLabel("No selection", self)
        layout.addWidget(self.selection_label)

        if pg is not None:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setBackground("w")
            self.plot_widget.setTitle("Preview")
            x = [i / 20.0 for i in range(200)]
            y = [sin(value) for value in x]
            self.plot_widget.plot(x, y, pen={"color": "#3b82f6", "width": 2})
            layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            label = QtWidgets.QLabel("Plot view unavailable. Install pyqtgraph to enable preview plots.")
            label.setWordWrap(True)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def bind_selection_model(self, model: SelectionModel) -> None:
        model.selection_changed.connect(self.update_selection)
        self.update_selection(model.selection)

    def update_selection(self, selection: dict) -> None:
        selected = [
            value.name if hasattr(value, "name") else str(value)
            for value in selection.values()
            if value is not None
        ]
        self.selection_label.setText(
            "Selection: " + " / ".join(selected) if selected else "No selection"
        )


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

        for name in ["Main", "Stats", "Spike"]:
            self.tool_workspace.addWidget(_PlaceholderToolWidget(name, self.tool_workspace))

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
