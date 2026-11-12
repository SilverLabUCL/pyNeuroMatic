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

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_epoch import DEFAULT_EPOCH_SETS, EXCLUDE_SET
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_settings import TRACE_LINES, TRACE_SYMBOLS, NMSettings
from pyneuromatic.gui.context_views import (
    DATA_VIEW,
    ChannelToolBar,
    TablePanel,
    no_view_label as _no_view_label,
    show_tool_page as _show_tool_page,
)
from pyneuromatic.gui import table_format
from pyneuromatic.gui.folder_browser import FolderBrowserWidget
from pyneuromatic.gui.history_panel import SEGMENT_STYLE, HistoryPanel
from pyneuromatic.gui.plot_marks import MarkOverlay
from pyneuromatic.gui.plot_ranges import RangeOverlay
from pyneuromatic.gui.settings_dialog import LINE_LABELS, SettingsDialog
from pyneuromatic.gui.membership import set_item_group
from pyneuromatic.gui.selection_model import (
    ALL_EPOCHS,
    ALL_GROUPS,
    ALL_SETS,
    SelectionModel,
    is_stored_epoch_set,
    scoped_epoch_names,
    needs_group_operator,
    selection_summary,
    source_object,
)
from pyneuromatic.gui.selection_strip import SelectionStrip, is_text_array
from pyneuromatic.gui.tier_actions import (
    DataActions,
    EpochSetActions,
    FolderActions,
    ChannelActions,
    GroupActions,
    SeriesActions,
)
from pyneuromatic.gui.stats_tab import StatsToolTab
from pyneuromatic.gui.tool_tab import ToolTabWidget
from pyneuromatic.gui.transform_dialog import (
    MARK,
    MENU_TEXT,
    TransformDialog,
    mark_html,
    steps_tooltip,
)


APP_NAME = "pyNeuroMatic"

# The tool tabs, in the tool rail after Browser (None: a placeholder). The app
# window registers each the same way: see ToolTabWidget and _register_tool_tab.
TOOL_TABS = {"Main": None, "Stats": StatsToolTab, "Spike": None}


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
    if is_stored_epoch_set(epoch_set) or epoch_set in (ALL_EPOCHS, ALL_SETS):
        parts.append(epoch_set)
    group = selection.get("group")
    if group == ALL_GROUPS:
        parts.append(ALL_GROUPS)
    elif group is not None:
        parts.append(f"Group {group}")
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


# The Plot's trace style settings and their defaults (Settings › Plot, #465)
TRACE_STYLE_DEFAULTS = {key: spec["default"] for key, spec in NMSettings._schema.items()
                        if key.startswith(("plot_trace_", "plot_overlay_"))}
_PEN_STYLES = {
    "solid": QtCore.Qt.PenStyle.SolidLine,
    "dash": QtCore.Qt.PenStyle.DashLine,
    "dot": QtCore.Qt.PenStyle.DotLine,
    "dash-dot": QtCore.Qt.PenStyle.DashDotLine,
}
_SYMBOLS = {"circle": "o", "square": "s", "triangle": "t", "diamond": "d",
            "plus": "+", "cross": "x", "star": "star"}


def _style_color(name: str, default: str) -> QtGui.QColor:
    """*name* as a colour, or *default* if it isn't one (e.g. a typo in the
    settings file)."""
    color = QtGui.QColor(name)
    return color if color.isValid() else QtGui.QColor(default)


class PlotPanel(QtWidgets.QWidget):
    """Plot panel with a trace navigator for data and dataseries modes."""

    # The current epoch or data array was added to (or removed from) a set or group
    membership_changed = QtCore.pyqtSignal()
    # A note of the current epoch (or data array) was added, edited or deleted
    notes_changed = QtCore.pyqtSignal()
    # A channel's filters or transforms changed (Filter & Transform...)
    transforms_changed = QtCore.pyqtSignal()
    # Trace Style...: open Settings at its Plot page
    trace_style_requested = QtCore.pyqtSignal()
    # A trace style chosen by right-clicking a trace: {setting: value}
    trace_style_set = QtCore.pyqtSignal(dict)
    # The current trace drawn, as plotted (transformed, if the channel is); None
    trace_drawn = QtCore.pyqtSignal(object)

    # The trace menu's choices (right-click a trace)
    TRACE_WIDTHS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    SYMBOL_SIZES = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None,
                 view_tabs: ChannelToolBar | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel tabs, then tabs of tools with results (e.g. Stats): the
        # window's, shared with the Table panel (#460), or its own
        self.view_tabs = view_tabs if view_tabs is not None else ChannelToolBar(self)
        self.view_tabs.view_changed.connect(self._on_view_changed)
        if view_tabs is None:
            layout.addWidget(self.view_tabs)
        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack, stretch=1)
        self.no_view_label = _no_view_label("plot", self.stack)
        self.stack.addWidget(self.no_view_label)
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
        # whether they visit only the selected traces, and overlaid epochs
        self.step_menu = QtWidgets.QMenu("Stepping", self)
        # Its title: "Epoch" (a data series) or "Data" (a Data Set's arrays)
        self.step_menu_title = self.step_menu.addAction("Epoch")
        self.step_menu_title.setEnabled(False)
        self.step_menu.addSeparator()
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
        self.selection_only_action = self.step_menu.addAction("Selected Only")
        self.selection_only_action.setCheckable(True)
        self.selection_only_action.setToolTip(
            "Step only through the traces in the strip's Epoch Set / Group or Data Set"
        )
        # Which epochs are drawn around the current one (#483)
        self.step_menu.addSeparator()
        overlay_row = QtWidgets.QWidget(self.step_menu)
        overlay_layout = QtWidgets.QHBoxLayout(overlay_row)
        overlay_layout.setContentsMargins(28, 2, 8, 2)
        overlay_layout.addWidget(QtWidgets.QLabel("Overlay", overlay_row))
        self.overlay_count = QtWidgets.QSpinBox(overlay_row)
        self.overlay_count.setRange(0, 1999)  # 0: no overlay
        self.overlay_count.setToolTip(
            "How many nearby epochs to plot with the current one (0: none), "
            "for the selected data series and channel"
        )
        overlay_layout.addWidget(self.overlay_count)
        overlay_action = QtWidgets.QWidgetAction(self.step_menu)
        overlay_action.setDefaultWidget(overlay_row)
        self.step_menu.addAction(overlay_action)
        # The current epoch's notes (or data array's, for a Data Set) (#505)
        self.step_menu.addSeparator()
        self.notes_action = self.step_menu.addAction("Notes…")
        self.notes_action.setToolTip(
            "Show, add, edit and delete the current epoch's notes (a Data Set: "
            "the current data array's)")
        self.notes_action.triggered.connect(self.show_current_notes)
        self.notes_action.setEnabled(False)
        self._current_item = None  # the epoch (or data array) shown
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
        self.group_combo.setFixedWidth(56)  # holds a group number
        self.group_combo.currentIndexChanged.connect(self._on_group_combo_changed)
        membership_layout.addWidget(self.group_combo)
        membership_layout.addStretch(1)
        self._updating_membership = False
        trace_layout.addLayout(navigator_layout)
        # Directly beneath the trace navigator
        trace_layout.addWidget(self.membership_row)

        # Graph functions, in a ⋯ menu at the plot's upper-left corner
        self.graph_menu = QtWidgets.QMenu("Graph", self)
        self.grid_action = self.graph_menu.addAction("Grid")
        self.grid_action.setCheckable(True)
        self.rescale_action = self.graph_menu.addAction("Rescale")
        self.rescale_action.setToolTip("Fit the axes to the data")
        # The chosen tool's X ranges (e.g. a Stats window): on, frozen or off
        self.ranges_menu = self.graph_menu.addMenu("Tool Ranges")
        self.ranges_menu.setToolTipsVisible(True)
        self.range_mode_actions: dict[str, QtGui.QAction] = {}
        modes = QtGui.QActionGroup(self.ranges_menu)
        for mode, text, tip in (
                ("on", "On", "Shown; drag an edge, or the middle to move both"),
                ("frozen", "Frozen", "Shown, but can't be dragged"),
                ("off", "Off", "Hidden")):
            action = self.ranges_menu.addAction(text)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda _checked=False, m=mode: self.set_range_mode(m))
            modes.addAction(action)
            self.range_mode_actions[mode] = action
        self.range_mode_actions["on"].setChecked(True)
        self.measures_action = self.graph_menu.addAction("Tool Measures")
        self.measures_action.setCheckable(True)
        self.measures_action.setChecked(True)
        self.measures_action.setToolTip(
            "Show what the chosen tool measures on the trace (e.g. the Stats "
            "window's mean, or its max point)")
        self.trace_style_action = self.graph_menu.addAction("Trace Style...")
        self.trace_style_action.setToolTip(
            "Colour, line and symbols of the traces (opens Settings at Plot)")
        self.trace_style_action.triggered.connect(self.trace_style_requested)
        self.graph_menu.addSeparator()
        self.transform_action = self.graph_menu.addAction(MENU_TEXT)
        self.transform_action.setToolTip(
            "Filters and transforms of the channel, as plotted and analysed "
            "(e.g. by Stats); the data arrays are not changed"
        )
        self.transform_action.setEnabled(False)

        self._selection_model: SelectionModel | None = None
        self._trace_entries: list[tuple[object, object | None]] = []
        self._trace_mode: str | None = None
        # The data or epoch container being stepped through (for its sets)
        self._trace_container = None
        # The channel plotted (data series mode), and its Filter & Transform window
        self._trace_channel: NMChannel | None = None
        self._transform_dialog: TransformDialog | None = None
        self._updating_navigator = False
        self._updating_channel_tabs = False
        self._plot_dataseries = None
        self._plot_channel_name: str | None = None
        self._strip_channel_name: str | None = None  # the strip's channel, last seen
        self._last_selection: dict = {}
        self.current_trace = None
        self._trace_style = dict(TRACE_STYLE_DEFAULTS)  # until set_trace_style
        self._run_display = False  # showing a tool run's epochs (#468)
        self.shown_trace = None  # the current trace, as plotted (trace_drawn)
        self._drawn_curves: list[tuple[object, bool]] = []
        self.rendered_trace_count = 0
        self.transform_error = ""  # why the plotted traces aren't transformed
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
        self.overlay_count.valueChanged.connect(self._render_current_traces)
        self.grid_action.toggled.connect(self._show_grid)
        self.measures_action.toggled.connect(self._show_measures)
        self.rescale_action.triggered.connect(self.rescale)
        self.transform_action.triggered.connect(self.edit_transforms)

        if pg is not None:
            self.plot_widget = pg.PlotWidget(self)
            # The chosen tool's X ranges, with draggable edges (#436)
            self.range_overlay = RangeOverlay(self.plot_widget, self)
            # ... and its measurements of the trace shown (#463)
            self.mark_overlay = MarkOverlay(self.plot_widget, self)
            self.plot_widget.setBackground("w")
            # No title (the navigator names the trace); a small top margin
            # keeps the corner ⋯ button clear of the axis labels
            self.plot_widget.getPlotItem().setContentsMargins(0, 22, 4, 0)
            trace_layout.addWidget(self.plot_widget)
            self._build_plot_right_click()
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
            self.range_overlay = None
            self.mark_overlay = None
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
        """Show tabs for the tools in *names*, in that order (one without a
        plot shows a note)."""
        self._tools = list(names)
        self._update_channel_tabs(self._last_selection)

    def show_tool(self, name: str) -> None:
        self.view_tabs.select_tool(name)

    def _show_current_page(self) -> None:
        view = self.view_tabs.current_view()
        if view is not None and view[0] == "tool":
            _show_tool_page(self.stack, self._tool_pages, self.no_view_label, view[1])
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
        selected_channel = selection.get("channel")
        selected_name = selected_channel.name if selected_channel is not None else None
        # A channel chosen in the strip is shown (a tab shows another until then)
        strip_changed = selected_name != self._strip_channel_name
        self._strip_channel_name = selected_name
        if strip_changed and selected_name in channel_names:
            previous_channel = selected_name
        if previous_channel not in channel_names:
            previous_channel = (
                selected_name if selected_name in channel_names
                else channel_names[0] if channel_names else None
            )

        self._plot_dataseries = dataseries
        self._plot_channel_name = previous_channel
        self._set_views(channel_names, previous_channel)
        self.view_tabs.mark_channels(
            {name: steps_tooltip(channel) for name, channel in dataseries.channels.items()}
        )

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
        plotted_channel = None

        data = source_object(selection, "data")
        dataseries = source_object(selection, "dataseries")
        if data is not None:
            folder = selection.get("folder")
            context = selection.get("toolfolder") or folder
            if context is not None:
                # Numeric arrays only: text arrays (e.g. ST_w0_data) can't be plotted
                entries = [(item, item) for item in context.data.values()
                           if not is_text_array(item)]
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
                plotted_channel = channel

        self._trace_entries = entries
        self._trace_channel = plotted_channel
        self.transform_action.setEnabled(plotted_channel is not None)
        if self._transform_dialog is not None and self._transform_dialog.isVisible():
            self._transform_dialog.set_channel(plotted_channel)
        self._trace_container = container
        self._trace_mode = mode
        if not entries:
            self._sync_trace_controls(0)
            self.trace_name_label.setText("No trace")
            self.trace_name_label.setToolTip("")
            self.current_trace = None
            self._current_item = None
            self.notes_action.setEnabled(False)
            self._render_current_traces()
            self._update_membership()
            return

        current_index = next(
            (index for index, (item, _) in enumerate(entries) if item is current_item),
            0,
        )
        self._sync_trace_controls(current_index)
        item, self.current_trace = entries[current_index]
        self._current_item = item
        self.notes_action.setEnabled(getattr(item, "notes", None) is not None)
        name = (self.current_trace.name if self.current_trace is not None
                else f"{item.name} (no data)")
        # ƒ: plotted with the channel's filters or transforms
        tip = steps_tooltip(plotted_channel)
        self.trace_name_label.setText(mark_html(name) if tip else name)  # red ƒ
        self.trace_name_label.setToolTip(tip)
        self._render_current_traces()
        self._update_membership()

    def notes_target(self) -> tuple | None:
        """(notes, owner text) of the epoch (or data array) shown, or None."""
        from pyneuromatic.gui.notes_view import notes_owner
        item = self._current_item
        notes = getattr(item, "notes", None)
        return (notes, notes_owner(item)) if notes is not None else None

    def show_current_notes(self) -> None:
        """Notes… (the epoch ⋯ menu): the current epoch's notes (#505)."""
        target = self.notes_target()
        if target is not None:
            self._open_notes(*target)

    def _open_notes(self, notes, owner: str) -> None:
        """The Notes dialog (replaced in tests)."""
        from pyneuromatic.gui.notes_view import NotesDialog
        dialog = NotesDialog(notes, owner, self)
        dialog.editor.notes_changed.connect(self.notes_changed)
        dialog.exec()

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
        self.overlay_count.setEnabled(can_overlay)
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
        # Epochs of a data series, or the arrays of a Data Set
        self.step_menu_title.setText("Data" if self._trace_mode == "data" else "Epoch")

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
        self._drawn_curves = []  # (curve, is current)

        if not self._trace_entries:
            self.trace_count_label.setText("of 0")
            self.overlay_status_label.setText("")
            self.range_overlay.set_x(None)
            self.range_overlay.redraw()
            self.mark_overlay.set_data(None)
            self.mark_overlay.redraw()
            self.shown_trace = None
            self.trace_drawn.emit(None)
            return

        current_index = max(0, min(self.trace_index.value(), len(self._trace_entries) - 1))
        overlay = (self.overlay_count.value() > 0 and self._trace_mode == "dataseries"
                   and not self._run_display)  # a run shows its epoch alone
        if overlay:
            total = len(self._trace_entries)
            limit = min(self.overlay_count.value() + 1, total)  # and the current
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

        # The channel's filters and transforms (copies; the arrays don't change)
        channel = self._trace_channel
        transform = channel if channel is not None and channel.transform_on else None
        error = ""
        shown = None  # the current trace, as plotted
        for item, data in entries:
            if data is None or data.nparray is None:
                continue
            if transform is not None:
                try:
                    data = transform.transformed(data)
                except Exception as e:  # shown raw, with the reason
                    error = error or f"{data.name}: {e}"
            x_values = self._x_values(data)
            is_current = item is self._trace_entries[current_index][0]
            if is_current:
                shown = data
            # As set in Settings › Plot (#465)
            style = self._current_style() if is_current else self._overlay_style()
            curve = self.plot_widget.plot(x_values, data.nparray, **style)
            curve.setDownsampling(auto=True, method="peak")
            curve.setCurveClickable(True, width=8)  # right-click: its style
            self._drawn_curves.append((curve, is_current))
            curve.setClipToView(True)
            self.rendered_trace_count += 1
        self.transform_error = error
        # The tool's ranges, over the current trace's x (cleared above)
        self.range_overlay.set_x(self._x_values(current_data) if current_data is not None
                                 and current_data.nparray is not None else None)
        self.range_overlay.redraw()
        self.mark_overlay.set_data(shown)  # measured as plotted (transformed)
        self.mark_overlay.redraw()
        self.shown_trace = shown
        self.trace_drawn.emit(shown)
        if error:
            shown = self.overlay_status_label.text()
            self.overlay_status_label.setText(
                (shown + "  " if shown else "") + f"{MARK} not applied (see Filter & Transform)"
            )
            self.overlay_status_label.setToolTip(error)
        else:
            self.overlay_status_label.setToolTip("")
        if self._transform_dialog is not None:
            self._transform_dialog.set_status(f"Can't apply to {error}" if error else "")

    def set_range_mode(self, mode: str) -> None:
        """The tool ranges: "on" (draggable), "frozen" or "off" (hidden)."""
        self.range_mode_actions[mode].setChecked(True)
        if self.range_overlay is not None:
            self.range_overlay.set_mode(mode)

    # --- right-click: a trace's style, or the Plot menu ------------------------

    def _build_plot_right_click(self) -> None:
        """Right-click a trace: its style. Elsewhere, the axes too: the Plot
        menu (as ⋯), which gets pyqtgraph's axis controls and Export."""
        view_box = self.plot_widget.getPlotItem().getViewBox()
        self.axes_menu = view_box.menu  # View All, X Axis, Y Axis, Mouse Mode
        self.axes_menu.setTitle("Axes")
        self.graph_menu.addSeparator()
        self.graph_menu.addMenu(self.axes_menu)
        self.export_action = self.graph_menu.addAction("Export...")
        self.export_action.setToolTip("Save the plot as an image (PNG, SVG...) or its data (CSV)")
        self.export_action.triggered.connect(self._export_plot)
        # The axes pass their right-clicks to the view box
        view_box.raiseContextMenu = self._on_right_click

    def _export_plot(self) -> None:
        scene = self.plot_widget.scene()
        scene.contextMenuItem = self.plot_widget.getPlotItem()
        scene.showExportDialog()

    def _on_right_click(self, event) -> None:
        on_trace = self.trace_at(event.scenePos())
        menu = self.graph_menu if on_trace is None else self.trace_menu(current=on_trace)
        self._popup(menu, event.screenPos().toPoint())

    def _popup(self, menu: QtWidgets.QMenu, pos: QtCore.QPoint) -> None:
        """Show *menu* at *pos* (replaced in tests)."""
        menu.popup(pos)

    def trace_at(self, scene_pos) -> bool | None:
        """True if *scene_pos* is on the current trace, False if on an overlay
        trace, None if on neither."""
        for curve, is_current in sorted(self._drawn_curves, key=lambda c: not c[1]):
            line, points = curve.curve, curve.scatter
            if line.isVisible() and line.mouseShape().contains(line.mapFromScene(scene_pos)):
                return is_current
            if points.isVisible() and len(points.pointsAt(points.mapFromScene(scene_pos))):
                return is_current
        return None

    def trace_menu(self, current: bool) -> QtWidgets.QMenu:
        """The style of the current trace, or of the overlay traces."""
        prefix = "plot_trace_" if current else "plot_overlay_"
        menu = QtWidgets.QMenu(self)
        header = menu.addAction("Current Trace" if current else "Overlay Traces")
        header.setEnabled(False)
        menu.addAction("Colour...").triggered.connect(
            lambda: self._pick_trace_color(prefix + "color"))
        self._choice_menu(menu, "Line Width", prefix + "width", self.TRACE_WIDTHS)
        if current:
            self._choice_menu(menu, "Line", "plot_trace_line", TRACE_LINES, LINE_LABELS)
            self._choice_menu(menu, "Symbol", "plot_trace_symbol", TRACE_SYMBOLS,
                              {name: name.capitalize() for name in TRACE_SYMBOLS})
            self._choice_menu(menu, "Symbol Size", "plot_trace_symbol_size", self.SYMBOL_SIZES)
        menu.addSeparator()
        menu.addAction("Default Style").triggered.connect(
            lambda: self.trace_style_set.emit(dict(TRACE_STYLE_DEFAULTS)))
        menu.addAction("Trace Style...").triggered.connect(self.trace_style_requested)
        self._trace_menu = menu  # kept while it shows
        return menu

    def _choice_menu(self, menu: QtWidgets.QMenu, title: str, key: str, values,
                     labels: dict | None = None) -> QtWidgets.QMenu:
        """A submenu of *values* for setting *key*, the current one checked."""
        current = self._trace_style[key]
        values = list(values)
        if labels is None and current not in values:  # e.g. a width typed in Settings
            values = sorted(values + [current])
        submenu = menu.addMenu(title)
        group = QtGui.QActionGroup(submenu)
        for value in values:
            action = submenu.addAction(labels[value] if labels else f"{value:g}")
            action.setCheckable(True)
            action.setChecked(value == current)
            action.triggered.connect(
                lambda _checked=False, v=value: self.trace_style_set.emit({key: v}))
            group.addAction(action)
        return submenu

    def _choose_color(self, initial: str) -> str | None:
        """A colour from the colour picker, or None if cancelled (replaced in tests)."""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(initial), self, "Trace colour")
        return color.name() if color.isValid() else None

    def _pick_trace_color(self, key: str) -> None:
        color = self._choose_color(self._trace_style[key])
        if color is not None:
            self.trace_style_set.emit({key: color})

    # --- trace style (Settings › Plot, #465) ---------------------------------

    def set_trace_style(self, settings) -> None:
        """Draw traces as *settings* say (its plot_trace_* and plot_overlay_*
        values), and redraw."""
        self._trace_style = {key: getattr(settings, key) for key in TRACE_STYLE_DEFAULTS}
        self._render_current_traces()

    @property
    def trace_style(self) -> dict:
        return dict(self._trace_style)

    def _current_style(self) -> dict:
        """pyqtgraph plot() arguments for the current trace."""
        style = self._trace_style
        color = _style_color(style["plot_trace_color"], TRACE_STYLE_DEFAULTS["plot_trace_color"])
        line = style["plot_trace_line"]
        symbol = style["plot_trace_symbol"]
        if line == "none" and symbol == "none":
            symbol = "circle"  # neither: the trace would not be seen
        kwargs: dict = {"pen": None if line == "none" else pg.mkPen(
            color, width=style["plot_trace_width"], style=_PEN_STYLES[line])}
        if symbol != "none":
            kwargs.update(symbol=_SYMBOLS[symbol], symbolSize=style["plot_trace_symbol_size"],
                          symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color))
        return kwargs

    def _overlay_style(self) -> dict:
        """pyqtgraph plot() arguments for an overlay trace (behind the current)."""
        style = self._trace_style
        color = _style_color(style["plot_overlay_color"],
                             TRACE_STYLE_DEFAULTS["plot_overlay_color"])
        color.setAlpha(150)
        return {"pen": pg.mkPen(color, width=style["plot_overlay_width"])}

    # --- a tool run's epochs (#468) --------------------------------------------

    @property
    def run_display(self) -> bool:
        """True while showing a tool run's epochs."""
        return self._run_display

    def show_run_trace(self, data) -> None:
        """Show *data*, the epoch (or data array) a tool run just analysed, as
        the current trace: display only, so the selection doesn't change and
        no command is written. None: the run ended, show the selection again."""
        if data is None:
            if self._run_display:
                self._run_display = False
                self.update_selection(self._last_selection)
            return
        index = next((i for i, (_item, d) in enumerate(self._trace_entries) if d is data),
                     None)
        if index is None:  # not in this plot (e.g. another channel)
            return
        self._run_display = True
        self._sync_trace_controls(index)  # the navigator shows it
        self.current_trace = data
        self.trace_name_label.setText(data.name)
        self._render_current_traces()

    def set_tool(self, tool) -> None:
        """Show the X ranges (#436) and measurements (#463) of *tool* (a tool
        tab), or none."""
        if self.range_overlay is not None:
            self.range_overlay.set_source(tool)
            self.mark_overlay.set_source(tool)

    def _show_measures(self, show: bool) -> None:
        if self.mark_overlay is not None:
            self.mark_overlay.set_visible(show)

    def edit_transforms(self) -> None:
        """Open Filter & Transform for the channel plotted."""
        channel = self._trace_channel
        if channel is None:
            return
        if self._transform_dialog is None:
            self._transform_dialog = TransformDialog(channel, self)
            self._transform_dialog.changed.connect(self.transforms_changed)
        else:  # opened again: Undo Changes goes back to the steps now
            self._transform_dialog.set_channel(
                channel, reopened=not self._transform_dialog.isVisible())
        self._transform_dialog.show()
        self._transform_dialog.raise_()
        self._transform_dialog.activateWindow()


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
        # Tables show NaN as the general settings say
        table_format.set_nan_mode(self._manager.settings.table_nan)
        # Notes as commands, or messages, as the settings say (#510)
        from pyneuromatic.gui.notes_view import NotesEditor
        NotesEditor.settings = self._manager.settings
        self.selection_model = SelectionModel(self._manager, self)
        self.current_tool_name = "Browser"

        self.setWindowTitle("pyNeuroMatic")
        self.resize(1200, 800)

        # Tier actions, shared by the strip's ⋯ menus and the menu bar
        self.folder_actions = FolderActions(self.selection_model, self)
        self.data_actions = DataActions(self.selection_model, self)
        self.series_actions = SeriesActions(self.selection_model, self)
        self.channel_actions = ChannelActions(
            self.selection_model, self, remove_action=self.series_actions.remove_channel_action)
        self.epoch_set_actions = EpochSetActions(self.selection_model, self)
        self.group_actions = GroupActions(self.selection_model, self)

        self._build_menu()
        self._build_selection_strip()
        self._build_tool_shell()
        self._build_history_panel()
        self._build_view_menu()
        self.selection_model.selection_changed.connect(self._append_selection_history)
        # The selection is shown in the window title, saving plot space
        self.selection_model.selection_changed.connect(self._update_window_title)
        self._update_window_title(self.selection_model.selection)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.folder_actions.open_action)
        file_menu.addAction(self.folder_actions.save_action)
        file_menu.addSeparator()
        # About pyNeuroMatic (Qt moves it to the pyNeuroMatic menu on macOS)
        self.about_action = file_menu.addAction(f"About {APP_NAME}")
        self.about_action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
        self.about_action.triggered.connect(self.show_about)
        # Settings... (⌘, ; Qt moves it to the pyNeuroMatic menu on macOS)
        self.settings_action = file_menu.addAction("Settings…")
        self.settings_action.setShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Preferences)
        )
        self.settings_action.setMenuRole(QtGui.QAction.MenuRole.PreferencesRole)
        self.settings_action.triggered.connect(lambda: self.open_settings())
        self.export_history_action = file_menu.addAction("Export History…")
        self.export_history_action.setToolTip("Save the commands as a Python script")
        self.export_history_action.triggered.connect(lambda: self.export_history())
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
        series_menu.addMenu(self.channel_actions.channel_menu(series_menu))
        series_menu.addMenu(self.epoch_set_actions.epoch_set_menu(series_menu))
        series_menu.addMenu(self.group_actions.group_menu(series_menu))
        self.menuBar().addMenu(series_menu)

        # Edit: Copy and Paste, for what has focus (a table, or a text box)
        self.edit_menu = self.menuBar().addMenu("Edit")
        self.copy_action = self.edit_menu.addAction("Copy")
        self.copy_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy))
        self.copy_action.triggered.connect(self.edit_copy)
        self.paste_action = self.edit_menu.addAction("Paste")
        self.paste_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Paste))
        self.paste_action.triggered.connect(self.edit_paste)
        # Greyed out in the open menu when there's nothing to copy or paste;
        # enabled again after, so ⌘C / ⌘V reach a table that gains focus
        self.edit_menu.aboutToShow.connect(self._update_edit_actions)
        self.edit_menu.aboutToHide.connect(self._enable_edit_actions)

        # View: filled once the tools and panels exist (_build_view_menu)
        self.view_menu = self.menuBar().addMenu("View")

    @staticmethod
    def about_text() -> str:
        """The About window's text: version, lab, licence and website."""
        import pyneuromatic

        url = "https://github.com/SilverLabUCL/pyNeuroMatic"
        return (
            f"<h3>{APP_NAME} {pyneuromatic.__version__}</h3>"
            "<p>Python tools for acquiring, analysing and simulating "
            "electrophysiological data, after NeuroMatic.</p>"
            '<p><a href="https://silverlab.org/">The Silver Lab</a> · '
            f"{pyneuromatic.__license__} License<br>"
            f'<a href="{url}">{url}</a></p>'
            f"<p><small>Python {sys.version.split()[0]} · "
            f"Qt {QtCore.qVersion()} · PyQt {QtCore.PYQT_VERSION_STR}</small></p>"
        )

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(self, f"About {APP_NAME}", self.about_text())

    def _build_view_menu(self) -> None:
        """View: the tools (as the tool rail), the Plot or Table panel, and
        the History dock."""
        menu = self.view_menu
        self.tool_view_actions: dict[str, QtGui.QAction] = {}
        tools = QtGui.QActionGroup(self)
        for number, name in enumerate(self.tool_rail.tool_names(), start=1):
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setShortcut(QtGui.QKeySequence(f"Ctrl+{number}"))  # ⌘1... on macOS
            action.triggered.connect(lambda _checked=False, n=name: self.tool_rail.select_tool(n))
            tools.addAction(action)
            self.tool_view_actions[name] = action
        self.tool_view_actions[self.current_tool_name].setChecked(True)
        menu.addSeparator()
        self.panel_view_actions: dict[str, QtGui.QAction] = {}
        panels = QtGui.QActionGroup(self)
        for name, panel in (("Plot", self.plot_widget), ("Table", self.table_panel)):
            action = menu.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, p=panel: self.context_panel.setCurrentWidget(p))
            panels.addAction(action)
            self.panel_view_actions[name] = action
        self._update_panel_buttons()  # checks these too
        menu.addSeparator()
        self.history_view_action = self.history_dock.toggleViewAction()  # checkable
        menu.addAction(self.history_view_action)

    # --- Edit › Copy and Paste ----------------------------------------------

    def _edit_target(self) -> tuple[str | None, QtWidgets.QWidget | None]:
        """What Copy and Paste act on: ("table", view) while the Table panel
        has focus, ("history", view) for the History panel's entries,
        ("text", widget) for a text box, else (None, None)."""
        widget = QtWidgets.QApplication.focusWidget()
        if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
                               QtWidgets.QTextEdit)):
            return "text", widget
        if (isinstance(widget, QtWidgets.QAbstractItemView)
                and self.table_panel.isAncestorOf(widget)):
            return "table", widget
        if widget is not None and widget is self.history_panel.view:
            return "history", widget
        if isinstance(widget, QtWidgets.QAbstractItemView) and callable(
                getattr(widget, "copy_selection", None)):  # e.g. Stats2's notes (#501)
            return "view", widget
        return None, None

    def edit_copy(self) -> None:
        kind, widget = self._edit_target()
        if kind == "table":
            self.table_panel.copy_selection()
        elif kind == "history":
            self.history_panel.copy_selection()
        elif kind == "view":
            widget.copy_selection()
        elif kind == "text":
            widget.copy()

    def edit_paste(self) -> None:
        kind, widget = self._edit_target()
        if kind == "table":
            self.table_panel.paste()
        elif kind == "text" and not widget.isReadOnly():
            widget.paste()

    def _update_edit_actions(self) -> None:
        kind, widget = self._edit_target()
        has_text = bool(QtWidgets.QApplication.clipboard().text())
        if kind == "table":
            selection = widget.selectionModel()
            can_copy = selection is not None and selection.hasSelection()
            can_paste = has_text and self.table_panel.can_paste()
        elif kind in ("history", "view"):
            can_copy = widget.selectionModel().hasSelection()
            can_paste = False
        elif kind == "text":
            can_copy = (widget.hasSelectedText() if isinstance(widget, QtWidgets.QLineEdit)
                        else widget.textCursor().hasSelection())
            can_paste = has_text and not widget.isReadOnly()
        else:
            can_copy = can_paste = False
        self.copy_action.setEnabled(can_copy)
        self.paste_action.setEnabled(can_paste)

    def _enable_edit_actions(self) -> None:
        self.copy_action.setEnabled(True)
        self.paste_action.setEnabled(True)

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
        strip.set_actions_menu("channel", self.channel_actions.channel_menu(strip))
        strip.set_actions_menu("epoch_set", self.epoch_set_actions.epoch_set_menu(strip))
        strip.set_actions_menu("group", self.group_actions.group_menu(strip))

    def _build_tool_shell(self) -> None:
        self.tool_rail = ToolRail(["Browser", *TOOL_TABS], self)
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

        # name -> tool tab (placeholders aren't tool tabs)
        self.tool_tabs: dict[str, ToolTabWidget] = {}
        for name, tab_class in TOOL_TABS.items():
            if tab_class is None:
                self.tool_workspace.addWidget(_PlaceholderToolWidget(name, self.tool_workspace))
            else:
                self.tool_tabs[name] = tab_class(self._manager, self.tool_workspace)
                self.tool_workspace.addWidget(self.tool_tabs[name])

        # What is shown (channel or tool) on the left, then how (Plot | Table)
        # on the right: one channel/tool bar for both panels (#460)
        self.view_tabs = ChannelToolBar(self)
        self.context_panel = QtWidgets.QStackedWidget(self)
        self.plot_widget = PlotPanel(self.context_panel, view_tabs=self.view_tabs)
        self.plot_widget.bind_selection_model(self.selection_model)
        self.plot_widget.membership_changed.connect(self._on_content_changed)
        self.plot_widget.notes_changed.connect(self._on_content_changed)
        self.plot_widget.transforms_changed.connect(self._on_transforms_changed)
        self.plot_widget.trace_style_requested.connect(lambda: self.open_settings("Plot"))
        self.plot_widget.trace_style_set.connect(self._set_trace_style)
        if self._manager is not None:
            self.plot_widget.set_trace_style(self._manager.settings)
        self.context_panel.addWidget(self.plot_widget)
        self.table_panel = TablePanel(self.context_panel, view_tabs=self.view_tabs)
        self.table_panel.bind_selection_model(self.selection_model)
        self.table_panel.show_in_browser_requested.connect(self._show_object_in_browser)
        self.context_panel.addWidget(self.table_panel)
        self._build_context_area()

        self.folder_actions.content_changed.connect(self._on_content_changed)
        self.data_actions.content_changed.connect(self._on_content_changed)
        self.series_actions.content_changed.connect(self._on_content_changed)
        self.series_actions.browse_requested.connect(self._show_object_in_browser)
        self.epoch_set_actions.content_changed.connect(self._on_content_changed)
        self.epoch_set_actions.browse_requested.connect(self._show_epoch_sets_in_browser)
        self.group_actions.content_changed.connect(self._on_content_changed)
        self.group_actions.browse_requested.connect(self._show_epochs_in_browser)
        self.channel_actions.browse_requested.connect(self._show_object_in_browser)
        self.folder_actions.browse_requested.connect(self._show_object_in_browser)
        self.channel_actions.content_changed.connect(self._on_content_changed)
        self.channel_actions.transforms_requested.connect(self._edit_channel_transforms)
        self.data_actions.browse_requested.connect(self._show_in_browser)
        self.data_actions.table_requested.connect(self._show_data_table)
        self.table_panel.data_edited.connect(self._on_data_edited)
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.content_changed.connect(self._on_browser_content_changed)
            self.browser_widget.data_created.connect(self._on_browser_data_created)
            self.browser_widget.edit_scales_requested.connect(self._edit_scales_of)
            self.browser_widget.edit_transforms_requested.connect(
                self._edit_channel_transforms)
            self.browser_widget.set_folder_actions(self.folder_actions)
            self.browser_widget.set_series_actions(self.series_actions)
            self.browser_widget.set_data_actions(self.data_actions)
            self.browser_widget.set_set_actions(self.epoch_set_actions, self.group_actions)
        for name, tab in self.tool_tabs.items():
            self._register_tool_tab(name, tab)
        self._update_tool_tabs()
        # Future: Inspector tab reserved for later development.
        self.context_area.setMinimumWidth(280)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.tool_rail)
        splitter.addWidget(self.tool_workspace)
        splitter.addWidget(self.context_area)
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
        history_layout.setContentsMargins(6, 0, 6, 4)  # the dock's title names it

        # Commands, messages, warnings, errors, selection and tool runs; the
        # core history log's records as they are written
        self.history_panel = HistoryPanel(history_widget)
        self.history_panel.add("message", f"Welcome to {APP_NAME}")
        self.history_panel.attach()
        self._start_command_history()
        history_layout.addWidget(self.history_panel)

        dock = QtWidgets.QDockWidget("History", self)
        dock.setWidget(history_widget)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        # Short to start with (a few lines), so the tools and plot get the
        # height; drag its edge to see more
        self.resizeDocks([dock], [180], QtCore.Qt.Orientation.Vertical)
        self.history_dock = dock

    def _start_command_history(self) -> None:
        """Record Python commands for what is done (the History panel's
        Commands), starting with the manager: "nm = NMManager()"."""
        history = self._manager.command_history
        if history.enabled:
            return  # already recording (e.g. NMManager(command_history=True))
        history.enabled = True
        if len(history) == 0:
            history.add("%s = NMManager()" % history.nm_name)

    def _on_tool_selected(self, tool_name: str) -> None:
        self.current_tool_name = tool_name
        # The Plot shows the X ranges and measurements of the tool chosen only
        self.plot_widget.set_tool(self.tool_tabs.get(tool_name))
        idx = self.tool_rail.index_for_name(tool_name)
        self.tool_workspace.setCurrentIndex(idx)
        action = getattr(self, "tool_view_actions", {}).get(tool_name)
        if action is not None:
            action.setChecked(True)  # View › the tool, as the rail

    def has_session_to_lose(self) -> bool:
        """True when quitting would lose something: a folder, or commands
        recorded since the start (beyond "nm = NMManager()")."""
        if len(self._manager.folders) > 0:
            return True
        history = self._manager.command_history
        return history.enabled and len(history) > 1

    def export_history(self) -> bool:
        """Save the command history as a Python script (asks where); False
        if not saved (cancelled)."""
        path = self._export_history_path()
        if not path:
            return False
        self._manager.command_history.to_script(path)
        return True

    def _export_history_path(self) -> str:
        """Where to save the history script ("" if cancelled; replaced in tests)."""
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export History", "pyneuromatic_history.py", "Python script (*.py)")
        return path

    def _may_quit(self) -> bool:
        """Ask before quitting, if set to and there is something to lose (#445)."""
        settings = self._manager.settings
        if not settings.confirm_quit or not self.has_session_to_lose():
            return True
        choice, dont_ask = self._ask_quit()
        if choice == "cancel" or (choice == "export" and not self.export_history()):
            return False
        if dont_ask:
            settings.confirm_quit = False
            settings.save_user()
        return True

    def _ask_quit(self) -> tuple[str, bool]:
        """The quit question: ("quit", "export" or "cancel", don't ask
        again). Replaced in tests."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(f"Quit {APP_NAME}")
        box.setText(f"Quit {APP_NAME}?")
        box.setInformativeText("Data and history not saved will be lost.")
        quit_button = box.addButton("Quit", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        export_button = box.addButton("Export History…",
                                      QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        export_button.setToolTip("Save the commands as a Python script, then quit")
        cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        dont_ask = QtWidgets.QCheckBox("Don't ask again", box)
        box.setCheckBox(dont_ask)
        box.exec()
        clicked = box.clickedButton()
        choice = ("quit" if clicked is quit_button
                  else "export" if clicked is export_button else "cancel")
        return choice, dont_ask.isChecked()

    def closeEvent(self, event) -> None:
        # The close button, Quit and Cmd+Q all come here
        if not self._may_quit():
            event.ignore()
            return
        self.history_panel.detach()  # the history log no longer writes here
        # Tear pyqtgraph views down before Qt deletes them, so a queued paint
        # event never reaches a half-deleted plot (this crashed the GUI tests)
        view = getattr(self.plot_widget, "plot_widget", None)
        if view is not None and not getattr(view, "closed", False):
            view.close()
        for tab in self.tool_tabs.values():
            tab.close_views()
        super().closeEvent(event)

    def _build_context_area(self) -> None:
        """The channel/tool tabs, Plot | Table, then the panel shown."""
        self.context_area = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(self.context_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(6, 4, 6, 0)
        self.view_tabs.setDrawBase(False)
        # No wider than its tabs, which macOS would otherwise centre
        self.view_tabs.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                     QtWidgets.QSizePolicy.Policy.Fixed)
        header.addWidget(self.view_tabs)
        header.addStretch(1)
        header.addSpacing(8)
        # A toggle, not tabs: Plot and Table are views of the channel or tool
        self.panel_buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(self.context_area)
        for segment, (name, panel) in zip(
                ("first", "last"), (("Plot", self.plot_widget), ("Table", self.table_panel))):
            button = QtWidgets.QToolButton(self.context_area)
            button.setText(name)
            button.setCheckable(True)
            button.setProperty("segment", segment)
            button.setToolTip(f"Show the {name.lower()} of the channel or tool chosen on the left")
            button.clicked.connect(
                lambda _checked=False, p=panel: self.context_panel.setCurrentWidget(p))
            group.addButton(button)
            header.addWidget(button)
            self.panel_buttons[name] = button
        self.context_area.setStyleSheet(SEGMENT_STYLE)
        layout.addLayout(header)
        layout.addWidget(self.context_panel, stretch=1)
        self.context_panel.currentChanged.connect(self._update_panel_buttons)
        self.view_tabs.view_changed.connect(self._update_panel_buttons)
        self.selection_model.selection_changed.connect(self._update_panel_buttons)
        self._update_panel_buttons()

    def _panels_of_view(self) -> dict[str, bool]:
        """Whether the Plot and Table panels have a view of the current
        channel or tool: a tool may have only one (#454)."""
        view = self.view_tabs.current_view()
        if view is None or view[0] != "tool":
            return {"Plot": True, "Table": True}
        return {"Plot": self.plot_widget.tool_page(view[1]) is not None,
                "Table": self.table_panel.tool_page(view[1]) is not None}

    def _update_panel_buttons(self, *_args) -> None:
        """Check the panel shown; a tool's missing view can't be chosen."""
        available = self._panels_of_view()
        panels = {"Plot": self.plot_widget, "Table": self.table_panel}
        current = self.context_panel.currentWidget()
        if not available["Plot" if current is self.plot_widget else "Table"]:
            other = "Table" if current is self.plot_widget else "Plot"
            if available[other]:
                self.context_panel.setCurrentWidget(panels[other])
                return  # back here from currentChanged
        for name, button in self.panel_buttons.items():
            button.setEnabled(available[name])
            button.setChecked(panels[name] is current)
        for name, action in getattr(self, "panel_view_actions", {}).items():
            action.setEnabled(available[name])
            action.setChecked(panels[name] is current)

    def settings_dialog(self) -> SettingsDialog:
        """A Settings window: General, then each tool's options."""
        pages = [(name, tab) for name, tab in self.tool_tabs.items() if tab.config_options()]
        dialog = SettingsDialog(self._manager.settings, pages, parent=self)
        dialog.settings_changed.connect(self._apply_settings)
        return dialog

    def open_settings(self, page: str | None = None) -> None:
        """Open Settings, at *page* (e.g. "Stats", from the Stats tool's ⋯)."""
        dialog = self.settings_dialog()
        if page:
            dialog.show_page(page)
        dialog.exec()

    def _set_trace_style(self, values: dict) -> None:
        """A trace style chosen on the Plot (right-click a trace): saved, as
        if set in Settings › Plot."""
        settings = self._manager.settings
        for key, value in values.items():
            setattr(settings, key, value)
        settings.save_user()
        self._apply_settings()

    def _apply_settings(self) -> None:
        """Show the general settings, e.g. redraw tables after the NaN one."""
        table_format.set_nan_mode(self._manager.settings.table_nan)
        self.plot_widget.set_trace_style(self._manager.settings)
        self.table_panel.channel_table.model.refresh()
        for tab in self.tool_tabs.values():
            tab.apply_settings()

    def _edit_scales_of(self, obj) -> None:
        """The browser's Edit Scales...: a data array's, a channel's or a
        data series'."""
        if isinstance(obj, NMChannel):
            self.series_actions.edit_channel_scales(obj)
        elif isinstance(obj, NMDataSeries):
            self.series_actions.edit_scales(obj)
        else:
            self.data_actions.edit_scales(obj)

    def _update_window_title(self, selection: dict) -> None:
        """e.g. "pyNeuroMatic — Demo1 / Record / A / E0"."""
        selected = selection_summary(selection)
        self.setWindowTitle(
            "pyNeuroMatic — " + " / ".join(selected) if selected else "pyNeuroMatic"
        )

    def _edit_channel_transforms(self, channel) -> None:
        """Filter & Transform for *channel* (from the Browser): plotted, so the
        window and the plot show the same channel."""
        self.selection_model.update({"channel": channel})
        self.plot_widget.view_tabs.select_channel(channel.name)
        self.context_panel.setCurrentWidget(self.plot_widget)
        self.plot_widget.edit_transforms()

    def _on_transforms_changed(self) -> None:
        """Redraw with a channel's new filters or transforms (ƒ marks too)."""
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh_info()  # its Filter & Transform rows
        self.selection_strip.refresh_options()
        selection = self.selection_model.selection
        self.plot_widget.update_selection(selection)
        self.table_panel.update_selection(selection)

    def _on_content_changed(self) -> None:
        """Reload views after folders, subfolders, data or sets change."""
        if isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh()
        self._on_browser_content_changed()

    def _on_browser_content_changed(self) -> None:
        selection = self.selection_model.selection
        series, group = selection.get("dataseries"), selection.get("group")
        groups = series.epochs.groups.group_numbers if series is not None else []
        if series is not None and group is not None and (
            group not in groups if group != ALL_GROUPS else not groups
        ):
            self.selection_model.update(group=None)  # the group(s) were deleted
        self.selection_model.refresh()
        self.selection_strip.refresh_options()
        for tab in self.tool_tabs.values():
            tab.refresh()
        self.folder_actions.update_enabled()
        self.data_actions.update_enabled()
        self.series_actions.update_enabled()
        self.epoch_set_actions.update_enabled()
        self.group_actions.update_enabled()
        self.channel_actions.update_enabled()
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
            # A folder or subfolder opens, showing its contents
            self.browser_widget.show_object(
                obj, expand=isinstance(obj, (NMFolder, NMToolFolder)))

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

    def _register_tool_tab(self, name: str, tab: ToolTabWidget) -> None:
        """Connect a tool tab to the window: what it provides is optional."""
        tab.bind_selection_model(self.selection_model)
        # Display while running (#468): the shared settings, and the Plot
        if self._manager is not None:
            tab.set_settings(self._manager.settings)
        tab.run_display_changed.connect(self._apply_settings)
        tab.run_trace_shown.connect(self._show_run_trace)
        if self.plot_widget.plot_widget is not None:
            tab.allow_during_run(self.plot_widget.plot_widget)  # zoom an epoch shown
        # The trace the Plot shows, e.g. for Stats' Details (#471)
        self.plot_widget.trace_drawn.connect(tab.set_trace)
        tab.set_trace(self.plot_widget.shown_trace)
        tab.settings_requested.connect(self.open_settings)
        tab.set_folder_actions(self.folder_actions)  # e.g. Stats2's Folder ⋯ (#494)
        tab.history_entry.connect(lambda kind, text: self.history_panel.add(kind, text))
        # Its views appear in both panels while the folder has its results
        if tab.plot_view is not None:
            self.plot_widget.register_tool(name, tab.plot_view)
        if tab.table_view is not None:
            self.table_panel.register_tool(name, tab.table_view)
        tab.results_changed.connect(self._update_tool_tabs)
        # A run or a deleted subfolder changes the Subfolder list
        tab.results_changed.connect(self.selection_strip.refresh_options)
        tab.view_requested.connect(lambda view, n=name: self._show_tool_view(n, view))
        tab.run_finished.connect(self._on_tool_run_finished)

    def _show_run_trace(self, data, channel: str) -> None:
        """A tool run's epoch in the Plot (None: the run ended, and the Plot is
        as it was: its panel, channel or tool tab, and epoch)."""
        if data is not None and not self.plot_widget.run_display:  # the run's first
            self._before_run_display = (self.context_panel.currentWidget(),
                                        self.view_tabs.current_view())
            self.context_panel.setCurrentWidget(self.plot_widget)
            if channel in self.view_tabs.channel_names:
                self.view_tabs.select_channel(channel)
        self.plot_widget.show_run_trace(data)
        before = getattr(self, "_before_run_display", None)
        if data is None and before is not None:
            self._before_run_display = None
            panel, view = before
            if view is not None:
                kind, name = view
                (self.view_tabs.select_channel if kind == "channel"
                 else self.view_tabs.select_tool)(name)
            self.context_panel.setCurrentWidget(panel)

    def _update_tool_tabs(self) -> None:
        tools = [name for name, tab in self.tool_tabs.items() if tab.has_results()]
        self.plot_widget.set_tools(tools)
        self.table_panel.set_tools(tools)
        self._update_panel_buttons()

    def _show_tool_view(self, name: str, view: str) -> None:
        """Show tool *name*'s "plot" or "table" view."""
        panel = self.plot_widget if view == "plot" else self.table_panel
        panel.show_tool(name)
        self.context_panel.setCurrentWidget(panel)

    def _on_tool_run_finished(self, succeeded: bool) -> None:
        # A run may add a results subfolder: show it in the browser too
        if succeeded and isinstance(self.browser_widget, FolderBrowserWidget):
            self.browser_widget.refresh()

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
        if needs_group_operator(selection):
            summary += " (choose AND or OR)"
        text = f"Selection: {summary}"
        if text == getattr(self, "_last_selection_text", None):
            return  # e.g. stepping through epochs (< >) in All Epochs or a set
        self._last_selection_text = text
        self.history_panel.add("selection", text)

    @property
    def tool_names(self) -> list[str]:
        return self.tool_rail.tool_names()


__all__ = [
    "SelectionStrip",
    "ToolRail",
    "NMAppWindow",
]
