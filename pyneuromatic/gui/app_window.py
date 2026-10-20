# -*- coding: utf-8 -*-
"""Application shell for the pyNeuroMatic GUI.

This module provides the initial shell for the desktop UI: a top-level menu bar,
a persistent selection strip, a tool rail, a central tool workspace, a dynamic
right-hand context panel, and a history/results panel.
"""
from __future__ import annotations

import bisect
import ctypes
import ctypes.util
import sys
from math import sin

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None

from pyneuromatic.core.nm_epoch import DEFAULT_EPOCH_SETS, EXCLUDE_SET
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.context_views import (
    DATA_VIEW,
    ChannelToolBar,
    TablePanel,
    ValuesPlot,
)
from pyneuromatic.gui.folder_browser import FolderBrowserWidget
from pyneuromatic.gui.membership import set_item_group
from pyneuromatic.gui.selection_model import (
    ALL_EPOCHS,
    SelectionModel,
    is_stored_epoch_set,
    scoped_epoch_names,
    selection_summary,
    source_object,
)
from pyneuromatic.gui.selection_strip import SelectionStrip
from pyneuromatic.gui.tier_actions import (
    DataActions,
    EpochSetActions,
    FolderActions,
    GroupActions,
    SeriesActions,
)
from pyneuromatic.gui.stats_tab import StatsToolTab


APP_NAME = "pyNeuroMatic"


def set_app_name(name: str = APP_NAME) -> None:
    """Name the application, e.g. in the macOS menu bar.

    Call before creating the QApplication. On macOS the first menu is named
    after the bundle, which for a script is "Python"; this renames it by
    setting the main bundle's CFBundleName through the Objective-C runtime.
    """
    QtCore.QCoreApplication.setApplicationName(name)
    if sys.platform != "darwin":
        return
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        void_p = ctypes.c_void_p
        call = ctypes.CFUNCTYPE(void_p, void_p, void_p)(send)
        call_text = ctypes.CFUNCTYPE(void_p, void_p, void_p, ctypes.c_char_p)(send)
        call_set = ctypes.CFUNCTYPE(None, void_p, void_p, void_p, void_p)(send)
        sel, cls = objc.sel_registerName, objc.objc_getClass

        def ns_string(text: str):
            return call_text(cls(b"NSString"), sel(b"stringWithUTF8String:"), text.encode())

        bundle = call(cls(b"NSBundle"), sel(b"mainBundle"))
        info = call(bundle, sel(b"infoDictionary"))
        call_set(info, sel(b"setObject:forKey:"), ns_string(name), ns_string("CFBundleName"))
    except (OSError, AttributeError, TypeError):  # pragma: no cover - cosmetic only
        pass


def _navigator_icon(kind: str, palette: QtGui.QPalette) -> QtGui.QIcon:
    """A chevron (< >), or a chevron with a bar (|< >|), all in one style.

    *kind* is "first", "previous", "next" or "last".
    """
    size, scale = 18, 3  # drawn large, so it stays crisp on Retina screens
    icon = QtGui.QIcon()
    for mode, role_group in ((QtGui.QIcon.Mode.Normal, QtGui.QPalette.ColorGroup.Active),
                             (QtGui.QIcon.Mode.Disabled, QtGui.QPalette.ColorGroup.Disabled)):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(palette.color(role_group, QtGui.QPalette.ColorRole.ButtonText), 2.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        forward = kind in ("next", "last")
        jump = kind in ("first", "last")
        # Chevron, nudged away from the bar on jump buttons
        tip, back = (12.0, 7.0) if forward else (6.0, 11.0)
        if jump:
            shift = -1.5 if forward else 1.5
            tip, back = tip + shift, back + shift
        painter.drawPolyline(QtGui.QPolygonF([
            QtCore.QPointF(back, 3.5), QtCore.QPointF(tip, 9.0), QtCore.QPointF(back, 14.5),
        ]))
        if jump:
            bar = 14.0 if forward else 4.0
            painter.drawLine(QtCore.QPointF(bar, 3.5), QtCore.QPointF(bar, 14.5))
        painter.end()
        icon.addPixmap(pixmap, mode)
    return icon


def _scope_text(selection: dict) -> str:
    """A short name for the epochs selected, e.g. "Set1", "Set1 AND Group 0"."""
    epoch_set = selection.get("epoch_set")
    parts = []
    if is_stored_epoch_set(epoch_set) or epoch_set == ALL_EPOCHS:
        parts.append(epoch_set)
    if selection.get("group") is not None:
        parts.append(f"Group {selection['group']}")
    operator = selection.get("group_operator") or "+"
    return f" {operator} ".join(parts)


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

    # The current epoch or data array was added to (or removed from) a set or group
    membership_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

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
        self.trace_first_button = QtWidgets.QToolButton(self)
        self.trace_first_button.setIcon(_navigator_icon("first", self.palette()))
        self.trace_first_button.setToolTip("First trace")
        self.trace_first_button.setFixedSize(28, 28)
        self.trace_first_button.setIconSize(QtCore.QSize(22, 22))
        navigator_layout.addWidget(self.trace_first_button)

        self.trace_previous_button = QtWidgets.QToolButton(self)
        self.trace_previous_button.setIcon(_navigator_icon("previous", self.palette()))
        self.trace_previous_button.setToolTip("Previous trace")
        self.trace_previous_button.setFixedSize(28, 28)
        self.trace_previous_button.setIconSize(QtCore.QSize(22, 22))
        navigator_layout.addWidget(self.trace_previous_button)

        self.trace_index = QtWidgets.QSpinBox(self)
        self.trace_index.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trace_index.setFixedWidth(72)
        self.trace_index.setRange(0, 0)
        navigator_layout.addWidget(self.trace_index)

        self.trace_count_label = QtWidgets.QLabel("of 0", self)
        navigator_layout.addWidget(self.trace_count_label)

        self.trace_next_button = QtWidgets.QToolButton(self)
        self.trace_next_button.setIcon(_navigator_icon("next", self.palette()))
        self.trace_next_button.setToolTip("Next trace")
        self.trace_next_button.setFixedSize(28, 28)
        self.trace_next_button.setIconSize(QtCore.QSize(22, 22))
        navigator_layout.addWidget(self.trace_next_button)

        self.trace_last_button = QtWidgets.QToolButton(self)
        self.trace_last_button.setIcon(_navigator_icon("last", self.palette()))
        self.trace_last_button.setToolTip("Last trace")
        self.trace_last_button.setFixedSize(28, 28)
        self.trace_last_button.setIconSize(QtCore.QSize(22, 22))
        navigator_layout.addWidget(self.trace_last_button)

        # Stepping options, in a ⋯ menu beside the buttons: how far < > move,
        # and whether they visit only the selected traces
        self.step_menu = QtWidgets.QMenu("Stepping", self)
        step_row = QtWidgets.QWidget(self.step_menu)
        step_layout = QtWidgets.QHBoxLayout(step_row)
        step_layout.setContentsMargins(28, 2, 8, 2)
        step_layout.addWidget(QtWidgets.QLabel("Step", step_row))
        self.step_size = QtWidgets.QSpinBox(step_row)
        self.step_size.setRange(1, 9999)
        self.step_size.setToolTip("How many traces < > move")
        step_layout.addWidget(self.step_size)
        step_action = QtWidgets.QWidgetAction(self.step_menu)
        step_action.setDefaultWidget(step_row)
        self.step_menu.addAction(step_action)
        self.selection_only_action = self.step_menu.addAction("Selection Only")
        self.selection_only_action.setCheckable(True)
        self.selection_only_action.setToolTip(
            "Step only through the traces in the strip's Epoch Set / Group or Data Set"
        )
        self.step_button = QtWidgets.QToolButton(self)
        self.step_button.setText("⋯")
        self.step_button.setToolTip("Stepping options")
        self.step_button.setMenu(self.step_menu)
        self.step_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        # Like the strip's ⋯ buttons: no menu arrow
        self.step_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
        self.step_button.setFixedHeight(28)
        navigator_layout.addWidget(self.step_button)

        self.trace_name_label = QtWidgets.QLabel("No trace", self)
        navigator_layout.addWidget(self.trace_name_label)

        # While stepping through the selection only, e.g. "(Set1: 250)"
        navigator_layout.addSpacing(8)
        self.selection_scope_label = QtWidgets.QLabel("", self)
        self.selection_scope_label.setToolTip("◀ ▶ step through these traces only")
        navigator_layout.addWidget(self.selection_scope_label)
        # e.g. "Showing 3 of 8 traces" while traces are overlaid
        self.overlay_status_label = QtWidgets.QLabel("", self)
        navigator_layout.addSpacing(8)
        navigator_layout.addWidget(self.overlay_status_label, stretch=1)

        # Set checkboxes (and, for epochs, a Group dropdown) for the current
        # epoch or data array, as in NeuroMatic
        self.membership_row = QtWidgets.QWidget(self)
        membership_layout = QtWidgets.QHBoxLayout(self.membership_row)
        membership_layout.setContentsMargins(0, 0, 0, 0)
        self._sets_layout = QtWidgets.QHBoxLayout()
        self._sets_layout.setContentsMargins(0, 0, 0, 0)
        membership_layout.addLayout(self._sets_layout)
        self.set_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        membership_layout.addSpacing(8)
        self.group_label = QtWidgets.QLabel("Group", self.membership_row)
        membership_layout.addWidget(self.group_label)
        self.group_combo = QtWidgets.QComboBox(self.membership_row)
        self.group_combo.setToolTip("The current epoch's group")
        self.group_combo.setMinimumContentsLength(1)
        self.group_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.group_combo.currentIndexChanged.connect(self._on_group_combo_changed)
        membership_layout.addWidget(self.group_combo)
        membership_layout.addStretch(1)
        self._updating_membership = False
        trace_layout.addLayout(navigator_layout)
        # Directly beneath the trace navigator
        trace_layout.addWidget(self.membership_row)

        # Graph functions, in a ⋯ menu at the plot's upper-left corner
        self.graph_menu = QtWidgets.QMenu("Graph", self)
        self.overlay_action = self.graph_menu.addAction("Overlay Traces")
        self.overlay_action.setCheckable(True)
        self.overlay_action.setToolTip(
            "Plot nearby epochs together for the selected data series and channel"
        )
        limit_row = QtWidgets.QWidget(self.graph_menu)
        limit_layout = QtWidgets.QHBoxLayout(limit_row)
        limit_layout.setContentsMargins(28, 2, 8, 2)
        limit_layout.addWidget(QtWidgets.QLabel("Limit", limit_row))
        self.overlay_limit = QtWidgets.QSpinBox(limit_row)
        self.overlay_limit.setRange(1, 2000)
        self.overlay_limit.setValue(50)
        self.overlay_limit.setToolTip("Most traces to overlay")
        limit_layout.addWidget(self.overlay_limit)
        limit_action = QtWidgets.QWidgetAction(self.graph_menu)
        limit_action.setDefaultWidget(limit_row)
        self.graph_menu.addAction(limit_action)
        self.graph_menu.addSeparator()
        self.grid_action = self.graph_menu.addAction("Grid")
        self.grid_action.setCheckable(True)
        self.rescale_action = self.graph_menu.addAction("Rescale")
        self.rescale_action.setToolTip("Fit the axes to the data")

        self._selection_model: SelectionModel | None = None
        self._trace_entries: list[tuple[object, object | None]] = []
        self._trace_mode: str | None = None
        # The data or epoch container being stepped through (for its sets)
        self._trace_container = None
        self._updating_navigator = False
        self._updating_channel_tabs = False
        self._plot_dataseries = None
        self._plot_channel_name: str | None = None
        self._last_selection: dict = {}
        self.current_trace = None
        self.rendered_trace_count = 0
        self.trace_index.valueChanged.connect(self._on_trace_index_changed)
        self.trace_first_button.clicked.connect(lambda: self._go_to("first"))
        self.trace_previous_button.clicked.connect(lambda: self._go_to("previous"))
        self.trace_next_button.clicked.connect(lambda: self._go_to("next"))
        self.trace_last_button.clicked.connect(lambda: self._go_to("last"))
        self.step_size.valueChanged.connect(self._update_step_buttons)
        self.selection_only_action.toggled.connect(self._update_step_buttons)
        # Rows of the traces in the strip's selection (None: no selection)
        self._scope_rows: list[int] | None = None
        self._scope_text = ""
        self.overlay_action.toggled.connect(self._render_current_traces)
        self.overlay_action.toggled.connect(self._sync_overlay_limit)
        self.overlay_limit.valueChanged.connect(self._render_current_traces)
        self.grid_action.toggled.connect(self._show_grid)
        self.rescale_action.triggered.connect(self.rescale)

        if pg is not None:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setBackground("w")
            # No title (the navigator names the trace); a small top margin
            # keeps the corner ⋯ button clear of the axis labels
            self.plot_widget.getPlotItem().setContentsMargins(0, 22, 4, 0)
            trace_layout.addWidget(self.plot_widget)
            self.graph_button = QtWidgets.QToolButton(self.plot_widget)
            self.graph_button.setText("⋯")
            self.graph_button.setToolTip("Graph functions")
            self.graph_button.setAutoRaise(True)
            # Like the strip's ⋯ buttons: no menu arrow
            self.graph_button.setStyleSheet(
                "QToolButton::menu-indicator { image: none; width: 0px; }"
            )
            self.graph_button.setMenu(self.graph_menu)
            self.graph_button.setPopupMode(
                QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
            )
            self.graph_button.move(2, 2)
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
        container = None

        data = source_object(selection, "data")
        dataseries = source_object(selection, "dataseries")
        if data is not None:
            folder = selection.get("folder")
            context = selection.get("toolfolder") or folder
            if context is not None:
                entries = [(item, item) for item in context.data.values()]
                mode = "data"
                current_item = data
                container = context.data
        elif dataseries is not None:
            channel = dataseries.channels.get(self._plot_channel_name)
            if channel is None:
                channel = selection.get("channel")
            if channel is None:
                channel = dataseries.channels.selected_value
            if channel is None and dataseries.channels.values():
                channel = dataseries.channels.values()[0]
            if channel is not None:
                # One pass over the series, not a lookup per epoch
                found = dataseries.epoch_data_map(channel.name)
                entries = [
                    (epoch, found.get(name)) for name, epoch in dataseries.epochs.items()
                ]
                mode = "dataseries"
                current_item = selection.get("epoch") or dataseries.epochs.selected_value
                container = dataseries.epochs

        self._trace_entries = entries
        self._trace_container = container
        self._trace_mode = mode
        if not entries:
            self._sync_trace_controls(0)
            self.trace_name_label.setText("No trace")
            self.current_trace = None
            self._render_current_traces()
            self._update_membership()
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
        self._update_membership()

    def _sync_trace_controls(self, current_value: int) -> None:
        count = len(self._trace_entries)
        self._updating_navigator = True
        self.trace_index.setRange(0, count - 1) if count else self.trace_index.setRange(0, 0)
        if count:
            self.trace_index.setValue(current_value)
        self.trace_index.setEnabled(count > 0)
        self.trace_count_label.setText(f"of {count}")
        self._update_scope()
        self._update_step_buttons()
        can_overlay = self._trace_mode == "dataseries" and self.plot_widget is not None
        self.overlay_action.setEnabled(can_overlay)
        self._sync_overlay_limit()
        self._updating_navigator = False

    # --- the current item's sets and group -------------------------------

    # Epochs show Set1/Set2/SetX; data has no default sets, so the first
    # few. Either way the strip's set is added if it isn't shown.
    MAX_DATA_SET_BOXES = 3

    def current_item(self):
        """The epoch or data array shown (the blue trace when overlaid)."""
        if self._trace_mode is None or not self._trace_entries:
            return None
        index = max(0, min(self.trace_index.value(), len(self._trace_entries) - 1))
        return self._trace_entries[index][0]

    def current_epoch(self):
        """The epoch shown, when stepping through a data series' epochs."""
        return self.current_item() if self._trace_mode == "dataseries" else None

    def _membership_set_names(self) -> list[str]:
        container = self._trace_container
        if self._trace_mode == "dataseries":
            names = list(DEFAULT_EPOCH_SETS)
            chosen = self._last_selection.get("epoch_set")  # may be All, Current Epoch
        else:
            names = list(container.sets.keys())[:self.MAX_DATA_SET_BOXES]
            chosen = self._last_selection.get("data_set")
        # Plus the strip's set, so any set can be ticked while stepping
        if chosen is not None and chosen not in names and chosen in container.sets:
            names.append(chosen)
        return names

    def _rebuild_set_checkboxes(self, names: list[str]) -> None:
        if list(self.set_checkboxes) == names:
            return
        for box in self.set_checkboxes.values():
            # Gone at once: deleteLater alone leaves them drawn until the
            # event loop runs
            self._sets_layout.removeWidget(box)
            box.hide()
            box.setParent(None)
            box.deleteLater()
        self.set_checkboxes = {}
        noun = "epoch" if self._trace_mode == "dataseries" else "data array"
        for set_name in names:
            box = QtWidgets.QCheckBox(set_name, self.membership_row)
            box.setToolTip(f"Add the current {noun} to {set_name}")
            if self._trace_mode == "dataseries" and set_name == EXCLUDE_SET:
                box.setToolTip("Exclude the current epoch from analysis (SetX)")
                box.setStyleSheet("QCheckBox { color: #c62828; }")
            box.toggled.connect(
                lambda checked, name=set_name: self._on_set_box_toggled(name, checked)
            )
            self.set_checkboxes[set_name] = box
            self._sets_layout.addWidget(box)

    def _update_membership(self, *_args) -> None:
        item = self.current_item()
        container = self._trace_container
        if item is None or container is None:
            self.membership_row.hide()
            return
        names = self._membership_set_names()
        # Shown only when there is a set (or group) to put the item in
        has_groups = (self._trace_mode == "dataseries"
                      and bool(container.groups.group_numbers))
        self.membership_row.setVisible(bool(names) or has_groups)
        if not names and not has_groups:
            return
        self._updating_membership = True
        self._rebuild_set_checkboxes(names)
        for set_name, box in self.set_checkboxes.items():
            box.setEnabled(not container.sets.is_equation(set_name))
            box.setChecked(container.sets.contains(set_name, item.name))
        self.group_label.setVisible(has_groups)
        self.group_combo.setVisible(has_groups)
        if has_groups:
            self.group_combo.clear()
            self.group_combo.addItem("", None)
            for number in container.groups.group_numbers:
                self.group_combo.addItem(str(number), number)
            self.group_combo.setCurrentIndex(
                max(self.group_combo.findData(container.groups.get_group(item.name)), 0)
            )
        self._updating_membership = False

    def _on_set_box_toggled(self, set_name: str, checked: bool) -> None:
        item = self.current_item()
        if self._updating_membership or item is None:
            return
        sets = self._trace_container.sets
        if checked:
            sets.add(set_name, [item])  # creates the set if it was deleted
        else:
            sets.remove(set_name, [item], error=False)
        self.membership_changed.emit()

    def _on_group_combo_changed(self, index: int) -> None:
        epoch = self.current_epoch()
        if self._updating_membership or epoch is None:
            return
        set_item_group(self._trace_container.groups, epoch.name,
                       self.group_combo.itemData(index))
        self.membership_changed.emit()

    # --- graph functions ------------------------------------------------

    def _sync_overlay_limit(self, *_args) -> None:
        self.overlay_limit.setEnabled(
            self.overlay_action.isEnabled() and self.overlay_action.isChecked()
        )

    def _show_grid(self, show: bool) -> None:
        if self.plot_widget is not None:
            self.plot_widget.showGrid(x=show, y=show, alpha=0.3)

    def rescale(self) -> None:
        """Fit the axes to the plotted data again (after zooming or panning)."""
        if self.plot_widget is not None:
            self.plot_widget.enableAutoRange()

    # --- stepping -------------------------------------------------------

    def _update_scope(self) -> None:
        """Find the traces in the strip's selection, for "Selection only"."""
        selection = self._last_selection
        names, label = None, ""
        if self._trace_mode == "dataseries" and self._plot_dataseries is not None:
            # The epochs an analysis would use (SetX left out)
            try:
                names = scoped_epoch_names(self._plot_dataseries, selection)
            except (RuntimeError, KeyError):  # e.g. Set and Group without AND/OR
                names = None
            label = _scope_text(selection)
        elif self._trace_mode == "data" and self._trace_container is not None:
            data_set = selection.get("data_set")
            if data_set is not None and data_set in self._trace_container.sets:
                names = self._trace_container.sets.get_items(data_set, get_keys=True) or []
                label = data_set
        if names is None:
            self._scope_rows = None
            self._scope_text = ""
        else:
            wanted = set(names)
            self._scope_rows = [
                row for row, (item, _) in enumerate(self._trace_entries) if item.name in wanted
            ]
            self._scope_text = f"({label}: {len(self._scope_rows)})"
        self.selection_only_action.setEnabled(self._scope_rows is not None)

    def _step_rows(self) -> list[int]:
        """The rows ◀ ▶ |◀ ▶| visit, in order."""
        if self.selection_only_action.isChecked() and self._scope_rows is not None:
            return self._scope_rows
        return list(range(len(self._trace_entries)))

    def _target_row(self, where: str) -> int | None:
        """The row to go to, or None if there is nowhere to go."""
        rows = self._step_rows()
        if not rows or not self._trace_entries:
            return None
        current = self.trace_index.value()
        step = self.step_size.value()
        if where == "first":
            target = rows[0]
        elif where == "last":
            target = rows[-1]
        elif where == "next":
            after = bisect.bisect_right(rows, current)  # first row after current
            if after >= len(rows):
                return None
            target = rows[min(after + step - 1, len(rows) - 1)]
        else:
            before = bisect.bisect_left(rows, current) - 1  # last row before current
            if before < 0:
                return None
            target = rows[max(before - step + 1, 0)]
        return None if target == current else target

    def _update_step_buttons(self, *_args) -> None:
        # The scope shows beside the trace name only while stepping through it
        stepping_scope = (self.selection_only_action.isChecked()
                          and self._scope_rows is not None)
        self.selection_scope_label.setText(self._scope_text if stepping_scope else "")
        for where, button in (("first", self.trace_first_button),
                              ("previous", self.trace_previous_button),
                              ("next", self.trace_next_button),
                              ("last", self.trace_last_button)):
            button.setEnabled(self._target_row(where) is not None)

    def _go_to(self, where: str) -> None:
        """Go to the first, previous, next or last trace (see :meth:`_target_row`)."""
        target = self._target_row(where)
        if target is not None and self.trace_index.isEnabled():
            self.trace_index.setValue(target)

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
            return

        current_index = max(0, min(self.trace_index.value(), len(self._trace_entries) - 1))
        overlay = self.overlay_action.isChecked() and self._trace_mode == "dataseries"
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
        self.group_actions = GroupActions(self.selection_model, self)

        self._build_menu()
        self._build_selection_strip()
        self._build_tool_shell()
        self._build_history_panel()
        self.selection_model.selection_changed.connect(self._append_selection_history)
        # The selection is shown in the window title, saving plot space
        self.selection_model.selection_changed.connect(self._update_window_title)
        self._update_window_title(self.selection_model.selection)

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
        series_menu.addMenu(self.group_actions.group_menu(series_menu))
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
        strip.set_actions_menu("group", self.group_actions.group_menu(strip))

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
        self.plot_widget.membership_changed.connect(self._on_content_changed)
        self.context_panel.addTab(self.plot_widget, "Plot")
        self.table_panel = TablePanel(self.context_panel)
        self.table_panel.bind_selection_model(self.selection_model)
        self.context_panel.addTab(self.table_panel, "Table")
        self._build_view_tabs_corner()

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
        self.group_actions.content_changed.connect(self._on_content_changed)
        self.group_actions.browse_requested.connect(self._show_epochs_in_browser)
        self.data_actions.browse_requested.connect(self._show_in_browser)
        self.data_actions.table_requested.connect(self._show_data_table)
        self.table_panel.data_edited.connect(self._on_data_edited)
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.content_changed.connect(self._on_browser_content_changed)
            self.browser_widget.data_created.connect(self._on_browser_data_created)
            self.browser_widget.edit_scales_requested.connect(self.data_actions.edit_scales)
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

    def _build_view_tabs_corner(self) -> None:
        """Channel/tool tabs to the right of the Plot/Table tabs, saving a row.

        Each panel keeps its own tab bar; only the open panel's is shown.
        """
        corner = QtWidgets.QWidget(self.context_panel)
        layout = QtWidgets.QHBoxLayout(corner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._view_tabs_holders = {}
        for panel in (self.plot_widget, self.table_panel):
            # A holder, since each tab bar hides itself when it has no tabs
            holder = QtWidgets.QWidget(corner)
            holder_layout = QtWidgets.QHBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            panel.view_tabs.setDrawBase(False)
            holder_layout.addWidget(panel.view_tabs)
            layout.addWidget(holder)
            self._view_tabs_holders[panel] = holder
        self.context_panel.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
        # Plot/Table at the far left (macOS centres them), so the channel and
        # tool tabs have room to grow leftwards. A plain pane frame, since the
        # native one keeps a gap in its edge for centred tabs.
        self.context_panel.setStyleSheet(
            "QTabWidget::tab-bar { alignment: left; }"
            " QTabWidget::pane { border: 1px solid palette(mid); border-radius: 4px;"
            " top: -1px; }"
        )
        self.context_panel.currentChanged.connect(self._show_current_view_tabs)
        self._show_current_view_tabs()

    def _show_current_view_tabs(self, *_args) -> None:
        current = self.context_panel.currentWidget()
        for panel, holder in self._view_tabs_holders.items():
            holder.setVisible(panel is current)

    def _update_window_title(self, selection: dict) -> None:
        """e.g. "pyNeuroMatic — Demo1 / Record / A / E0"."""
        selected = selection_summary(selection)
        self.setWindowTitle(
            "pyNeuroMatic — " + " / ".join(selected) if selected else "pyNeuroMatic"
        )

    def _on_content_changed(self) -> None:
        """Reload views after folders, subfolders, data or sets change."""
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh()
        self._on_browser_content_changed()

    def _on_browser_content_changed(self) -> None:
        selection = self.selection_model.selection
        series, group = selection.get("dataseries"), selection.get("group")
        if (series is not None and group is not None
                and group not in series.epochs.groups.group_numbers):
            self.selection_model.update(group=None)  # the group was deleted
        self.selection_model.refresh()
        self.selection_strip.refresh_options()
        self.stats_tab.stats2.refresh()
        self.folder_actions.update_enabled()
        self.data_actions.update_enabled()
        self.series_actions.update_enabled()
        self.epoch_set_actions.update_enabled()
        self.group_actions.update_enabled()
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

    def _show_epochs_in_browser(self, series) -> None:
        """Switch to the Browser with *series*' current group (or its Groups,
        or Epochs, row) selected."""
        self.tool_rail.select_tool("Browser")
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.show_groups_of(series, self.selection_model.selection.get("group"))

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
        # A run adds a results subfolder: show it in the browser too
        if succeeded and isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh()
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
        changes = {tier: obj}
        if set_name is not None or "group" not in selection:
            changes[set_key] = set_name
        if "group" in selection:  # a group row, or one of its epochs
            changes["group"] = selection["group"]
        self.selection_model.update(changes)

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
