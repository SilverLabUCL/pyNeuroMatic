"""Shared pieces of the Plot and Table context panels.

Both panels have a tab bar of channel views (A, B, ...) followed by tool views
(Stats, ...) that appear while the current folder has results from that tool.
"""
from __future__ import annotations

import math
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.gui import table_format
from pyneuromatic.gui.transform_dialog import MARK, MARK_COLOR, steps_tooltip
from pyneuromatic.gui.selection_model import (
    epoch_scope_label,
    scoped_epoch_names,
    source_object,
)

# Tab label used for the data view when a single data array is selected
DATA_VIEW = "Data"


class ChannelToolBar(QtWidgets.QTabBar):
    """Tab bar of channel views followed by tool views.

    Each tab carries ``(kind, name)`` with kind ``"channel"`` or ``"tool"``.
    Tool tabs are drawn in a different colour so the two groups stand apart.
    """

    view_changed = QtCore.pyqtSignal(str, str)  # kind, name

    TOOL_TEXT_COLOR = QtGui.QColor("#1d4ed8")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setShape(QtWidgets.QTabBar.Shape.RoundedNorth)
        self.setUsesScrollButtons(True)
        self.setExpanding(False)
        self._channels: list[str] = []
        self._tools: list[str] = []
        self.currentChanged.connect(self._on_current_changed)
        self.hide()

    @property
    def channel_names(self) -> list[str]:
        return list(self._channels)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def current_view(self) -> tuple[str, str] | None:
        index = self.currentIndex()
        return self.tabData(index) if index >= 0 else None

    def set_views(
        self,
        channels: list[str],
        tools: list[str],
        current_channel: str | None = None,
    ) -> None:
        """Show *channels* then *tools*, without emitting :attr:`view_changed`.

        A selected tool stays selected while it is listed; otherwise
        *current_channel* (or the first view) is selected.
        """
        previous = self.current_view()
        self.blockSignals(True)
        if channels != self._channels or tools != self._tools:
            while self.count():
                self.removeTab(0)
            for name in channels:
                self.setTabData(self.addTab(name), ("channel", name))
            for name in tools:
                index = self.addTab(name)
                self.setTabData(index, ("tool", name))
                self.setTabTextColor(index, self.TOOL_TEXT_COLOR)
                self.setTabToolTip(index, f"{name} results")
            self._channels = list(channels)
            self._tools = list(tools)
        if previous is not None and previous[0] == "tool" and previous[1] in tools:
            target = previous
        elif current_channel in channels:
            target = ("channel", current_channel)
        elif channels:
            target = ("channel", channels[0])
        elif tools:
            target = ("tool", tools[0])
        else:
            target = None
        if target is not None:
            self.setCurrentIndex(self._index_of(target))
        self.blockSignals(False)
        self.setVisible(self.count() > 0)

    def mark_channels(self, tips: dict[str, str]) -> None:
        """Mark channels shown with filters or transforms ("A ƒ"); *tips*
        (by channel name) lists their steps, "" for none."""
        for index in range(self.count()):
            kind, name = self.tabData(index)
            if kind == "channel":
                tip = tips.get(name, "")
                self.setTabText(index, f"{name} {MARK}" if tip else name)
                # Red: one or more steps are on (a tab's text has one colour)
                self.setTabTextColor(index, QtGui.QColor(MARK_COLOR) if tip else QtGui.QColor())
                self.setTabToolTip(index, tip)

    def select_tool(self, name: str) -> bool:
        index = self._index_of(("tool", name))
        if index < 0:
            return False
        self.setCurrentIndex(index)
        return True

    def select_channel(self, name: str) -> bool:
        index = self._index_of(("channel", name))
        if index < 0:
            return False
        self.setCurrentIndex(index)
        return True

    def _index_of(self, view: tuple[str, str]) -> int:
        for index in range(self.count()):
            if self.tabData(index) == view:
                return index
        return -1

    def _on_current_changed(self, index: int) -> None:
        if index >= 0:
            kind, name = self.tabData(index)
            self.view_changed.emit(kind, name)


class ValuesPlot(QtWidgets.QWidget):
    """Plot of values that are not traces, e.g. a Stats results column."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if pg is not None:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setBackground("w")
            self.plot_widget.setTitle("Use Plot or Histogram in Stats2")
            layout.addWidget(self.plot_widget)
        else:  # pragma: no cover - optional GUI dependency
            self.plot_widget = None
            label = QtWidgets.QLabel("Install pyqtgraph to enable plots.", self)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def show_values(
        self,
        title: str,
        y,
        x=None,
        xlabel: str = "",
        ylabel: str = "",
        style: str = "points",
    ) -> None:
        """Plot *y* against *x* (or index), or as a histogram.

        ``style`` is ``"points"`` or ``"histogram"`` (x holds the bin edges,
        one more than y).
        """
        if self.plot_widget is None:
            return
        self.plot_widget.clear()
        if style == "histogram":
            self.plot_widget.plot(
                x,
                y,
                stepMode="center",
                fillLevel=0,
                brush=pg.mkBrush("#93c5fd"),
                pen=pg.mkPen(color="#2563eb", width=1),
            )
        else:
            if x is None:
                x = list(range(len(y)))
            self.plot_widget.plot(
                x,
                y,
                pen=pg.mkPen(color="#78909c", width=1),
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush("#2563eb"),
                symbolPen=None,
                connect="finite",
            )
        self.plot_widget.setLabel("bottom", xlabel)
        self.plot_widget.setLabel("left", ylabel)
        self.plot_widget.setTitle(title)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


class ChannelTableModel(QtCore.QAbstractTableModel):
    """Data arrays side by side: an x column, then one column per array.

    Cells are formatted only when Qt asks for them, so long recordings with
    many epochs stay responsive. Values can be edited, except the x column
    and arrays in a locked data series.

    With a channel set (:meth:`set_transform`), the arrays are shown with its
    filters and transforms, read-only; each array is transformed when first
    shown.
    """

    data_edited = QtCore.pyqtSignal(object)  # the NMData that changed

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._columns: list[NMData] = []
        self._rows = 0
        self._channel = None  # shown with this channel's filters and transforms
        self._transformed: dict[tuple[int, str], Any] = {}  # (array, steps): values

    @property
    def arrays(self) -> list[NMData]:
        return list(self._columns)

    def set_arrays(self, arrays: list[NMData]) -> None:
        self.beginResetModel()
        self._columns = [d for d in arrays if d.nparray is not None]
        self._rows = max((d.nparray.size for d in self._columns), default=0)
        self._transformed.clear()
        self.endResetModel()

    @property
    def transform_channel(self):
        return self._channel

    def set_transform(self, channel) -> None:
        """Show the arrays with *channel*'s filters and transforms (None: as
        they are)."""
        if channel is self._channel:
            self.refresh()  # its steps may have changed
            return
        self._channel = channel
        self._transformed.clear()
        self.refresh()

    def _values(self, column: int):
        """The values shown in *column* (1 or more)."""
        data = self._columns[column - 1]
        channel = self._channel
        if channel is None or not channel.transform_on:
            return data.nparray
        key = (id(data), repr([t.to_dict() for t in channel.transform]))
        if key not in self._transformed:
            try:
                self._transformed[key] = channel.transformed(data).nparray
            except Exception:  # the panel says why; show the values as they are
                self._transformed[key] = data.nparray
        return self._transformed[key]

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid() or not self._columns:
            return 0
        return 1 + len(self._columns)

    def _x_value(self, row: int) -> float | None:
        # The x column follows the first array's x scale
        first = self._columns[0]
        if first.xarray is not None:
            return first.xarray[row] if row < first.xarray.size else None
        return first.xscale.start + row * first.xscale.delta

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        Role = QtCore.Qt.ItemDataRole
        if role not in (Role.DisplayRole, Role.EditRole, Role.ForegroundRole, Role.ToolTipRole):
            return None
        row, column = index.row(), index.column()
        if column == 0:
            value = self._x_value(row)
        else:
            values = self._values(column)
            value = values[row] if row < values.size else None
        if role == Role.EditRole and value is not None:
            return repr(float(value))  # full precision while editing
        text, is_nan = table_format.cell(value)
        if role == Role.ForegroundRole:  # NaN in grey (Settings > Tables)
            color = table_format.nan_foreground() if is_nan else None
            return QtGui.QBrush(color) if color is not None else None
        if role == Role.ToolTipRole:
            return table_format.nan_tooltip() if is_nan else None
        return text

    @staticmethod
    def is_locked(data: NMData) -> bool:
        """True if *data* belongs to a locked data series."""
        container = getattr(data._parent, "data", None)
        if container is None or data.name not in container:
            return False
        # The x array of a locked series is protected as the series' arrays are
        return (any(ds.locked for ds in container.dataseries_of(data.name))
                or container.locked_xarray_series(data.name) is not None)

    def _editable(self, index: QtCore.QModelIndex) -> bool:
        if not index.isValid() or index.column() == 0 or self._channel is not None:
            return False  # transformed values can't be edited
        data = self._columns[index.column() - 1]
        return index.row() < data.nparray.size and not self.is_locked(data)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if self._editable(index):
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role != QtCore.Qt.ItemDataRole.EditRole or not self._editable(index):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        data = self._columns[index.column() - 1]
        if data.nparray.dtype.kind != "f":
            data.nparray = data.nparray.astype(float)  # keep fractional edits
        data.nparray[index.row()] = number
        self.dataChanged.emit(index, index)
        self.data_edited.emit(data)
        return True

    def paste_values(self, row: int, column: int, block: list[list[str]]) -> int:
        """Write *block* (rows of cell texts) from cell (*row*, *column*).
        Empty cells, cells that can't be edited (the x column, a locked
        series, transformed values) and cells past an array's end are left
        as they are. Returns how many values changed."""
        changed: dict[int, NMData] = {}
        count = 0
        for r, cells in enumerate(block):
            for c, text in enumerate(cells):
                index = self.index(row + r, column + c)
                if not text.strip() or not self._editable(index):
                    continue
                try:
                    number = float(text)
                except ValueError:
                    continue
                data = self._columns[index.column() - 1]
                if data.nparray.dtype.kind != "f":
                    data.nparray = data.nparray.astype(float)  # keep fractional values
                data.nparray[index.row()] = number
                changed[id(data)] = data
                count += 1
        if count:
            self.refresh()
            for data in changed.values():
                self.data_edited.emit(data)
        return count

    def raw_value(self, row: int, column: int):
        """A cell's value (x for column 0), or None past an array's end."""
        if column == 0:
            return self._x_value(row)
        values = self._values(column)
        return values[row] if row < values.size else None

    @staticmethod
    def format_value(value) -> str:
        """A cell's text (NaN as the Settings say)."""
        return table_format.cell(value)[0]

    def refresh(self) -> None:
        """Redraw every cell (e.g. after the NaN setting changes)."""
        if self._rows and self._columns:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(self._rows - 1, len(self._columns)))

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Vertical:
            return str(section)
        if section == 0:
            first = self._columns[0] if self._columns else None
            units = first.xscale.units if first is not None else ""
            return f"x ({units})" if units else "x"
        return self._columns[section - 1].name


class ChannelTable(QtWidgets.QWidget):
    """The data arrays of one channel, one column per epoch in scope."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel("", self)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        self.model = ChannelTableModel(self)
        self.view = QtWidgets.QTableView(self)
        self.view.setModel(self.model)
        # One cell is highlighted, not its whole column; the current cell's
        # column is the current array (its header is bold)
        self.view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        # Cells, a header's column, or several (Shift / ⌘); ⌘C copies them
        self.view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        layout.addWidget(self.view, stretch=1)

    def show_arrays(
        self, arrays: list[NMData], title: str, current: NMData | None = None
    ) -> None:
        if [id(d) for d in arrays] != [id(d) for d in self.model.arrays]:
            self.model.set_arrays(arrays)
        self.title_label.setText(title)
        self.select_array(current)

    def select_array(self, data: NMData | None) -> None:
        """Make *data*'s column current, keeping the row (one cell shown)."""
        for column, item in enumerate(self.model.arrays, start=1):
            if item is data:
                row = max(self.view.currentIndex().row(), 0)
                index = self.model.index(min(row, max(self.model.rowCount() - 1, 0)), column)
                self.view.setCurrentIndex(index)
                self.view.scrollTo(index)
                return
        self.view.clearSelection()
        self.view.setCurrentIndex(QtCore.QModelIndex())

    def copy_text(self, names: bool = True, x_values: bool = True) -> str:
        """The selection as tab-separated text: whole columns with their
        names first (*names*) and the x column (*x_values*); full precision."""
        model = self.model

        def value(row, column):
            return table_format.copy_value(model.raw_value(row, column))

        def header(column):
            return str(model.headerData(column, QtCore.Qt.Orientation.Horizontal))

        selection = self.view.selectionModel()
        columns = {i.column() for i in selection.selectedIndexes()}
        whole = bool(columns) and all(
            selection.isColumnSelected(c, QtCore.QModelIndex()) for c in columns
        )
        # The x column with whole columns (a block of cells copies just itself)
        with_x = x_values and whole and 0 not in columns and model.columnCount() > 0
        return table_format.selection_text(
            self.view, value, header,
            row_label=(lambda row: value(row, 0)) if with_x else None,
            row_label_header=header(0) if with_x else "", names=names,
        )

    def paste_text(self, text: str) -> int:
        """Paste tab-separated values (e.g. copied from this table or a
        spreadsheet) from the selection's top-left cell, or the current cell.
        A first line of column names is skipped, and with it a copied x
        column. Returns how many values changed."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").split("\n")
        block = [line.split("\t") for line in lines]
        indexes = self.view.selectionModel().selectedIndexes()
        if indexes:
            row = min(i.row() for i in indexes)
            column = min(i.column() for i in indexes)
        elif self.view.currentIndex().isValid():
            row, column = self.view.currentIndex().row(), self.view.currentIndex().column()
        else:
            return 0
        if block and any(cell.strip() and not _is_number(cell) for cell in block[0]):
            names = block.pop(0)  # column names
            x_header = str(self.model.headerData(0, QtCore.Qt.Orientation.Horizontal))
            if names and names[0] == x_header:  # copied with its x values: not pasted
                block = [cells[1:] for cells in block]
                column = max(column, 1)
        return self.model.paste_values(row, column, block)

    def selected_array(self) -> NMData | None:
        """The array of the current cell's column."""
        index = self.view.currentIndex()
        if not index.isValid() or index.column() == 0:
            return None
        return self.model.arrays[index.column() - 1]


class TablePanel(QtWidgets.QWidget):
    """Table context panel: channel tables and the tables of tools with results."""

    data_edited = QtCore.pyqtSignal(object)  # an NMData value was edited
    # Show in Browser: the current array, or the tool's results folder
    show_in_browser_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.view_tabs = ChannelToolBar(self)
        self.view_tabs.view_changed.connect(self._on_view_changed)
        layout.addWidget(self.view_tabs)

        # Table functions, in a ⋯ menu at the upper-left corner (as the Plot's)
        self.table_menu = QtWidgets.QMenu("Table", self)
        self.copy_action = self.table_menu.addAction("Copy")
        self.copy_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy))
        self.copy_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.copy_action.setToolTip("Copy the selected cells or columns (tab-separated)")
        self.copy_action.triggered.connect(self.copy_selection)
        self.copy_all_action = self.table_menu.addAction("Copy All")
        self.copy_all_action.triggered.connect(self.copy_all)
        self.paste_action = self.table_menu.addAction("Paste")
        self.paste_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Paste))
        self.paste_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.paste_action.setToolTip(
            "Paste tab-separated values into the Data table, from the selected cell")
        self.paste_action.triggered.connect(self.paste)
        # ⌘C and ⌘V are Edit › Copy and Paste, which call copy_selection and
        # paste while the table has focus (the shortcuts are shown here too)
        self.table_menu.addSeparator()
        self.browser_action = self.table_menu.addAction("Show in Browser")
        self.browser_action.setToolTip("Show the current array (or results folder) in the browser")
        self.browser_action.triggered.connect(self.show_in_browser)
        self.table_menu.addSeparator()
        self.transformed_action = self.table_menu.addAction("Show Transformed")
        self.transformed_action.setCheckable(True)
        self.transformed_action.setToolTip(
            "Show the channel's values with its filters and transforms (as plotted "
            "and analysed); they can't be edited"
        )
        self.transformed_action.toggled.connect(self._refresh_channel_table)
        self.names_action = self.table_menu.addAction("Include Column Names")
        self.x_action = self.table_menu.addAction("Include x Values / Row Names")
        for option in (self.names_action, self.x_action):
            option.setCheckable(True)
            option.setChecked(True)
        self.menu_button = QtWidgets.QToolButton(self)
        self.menu_button.setText("⋯")
        self.menu_button.setToolTip("Table functions")
        self.menu_button.setMenu(self.table_menu)
        self.menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
        menu_row = QtWidgets.QHBoxLayout()
        menu_row.addWidget(self.menu_button)
        menu_row.addStretch(1)
        layout.addLayout(menu_row)

        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack, stretch=1)
        self.channel_table = ChannelTable(self.stack)
        self.stack.addWidget(self.channel_table)
        self.channel_table.model.data_edited.connect(self.data_edited)
        self._add_context_menu(self.channel_table)

        self._selection_model: Any = None
        self._last_selection: dict = {}
        self._channel_name: str | None = None
        self._strip_channel_name: str | None = None  # the strip's channel, last seen
        self._tool_pages: dict[str, QtWidgets.QWidget] = {}
        self._tools: list[str] = []
        self._updating = False

    # --- copying -----------------------------------------------------------

    def selection_text(self) -> str:
        """The current table's selection as tab-separated text."""
        page = self.stack.currentWidget()
        names, extra = self.names_action.isChecked(), self.x_action.isChecked()
        if page is self.channel_table:
            return self.channel_table.copy_text(names=names, x_values=extra)
        if hasattr(page, "copy_text"):  # e.g. the Stats tables
            return page.copy_text(names=names, row_names=extra)
        return ""

    def copy_selection(self) -> None:
        text = self.selection_text()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)

    def can_paste(self) -> bool:
        """Whether the table shown takes pasted values (a Data table, not
        showing transformed values)."""
        return (self.stack.currentWidget() is self.channel_table
                and self.channel_table.model.transform_channel is None
                and bool(self.channel_table.model.arrays))

    def paste(self) -> int:
        """Paste the clipboard's tab-separated values into the Data table,
        from the selected cell; returns how many values changed."""
        text = QtWidgets.QApplication.clipboard().text()
        if not text.strip():
            return 0
        count = self.channel_table.paste_text(text) if self.can_paste() else 0
        if count == 0:
            self._show_message(
                "Paste", "Nothing was pasted. Values can be pasted into a Data table's "
                "arrays (not the x column, a locked data series, past an array's end, "
                "or with Show Transformed on).")
        return count

    def _show_message(self, title: str, text: str) -> None:
        """A note (replaced in tests)."""
        QtWidgets.QMessageBox.information(self, title, text)

    def shown_object(self):
        """What the table shows: the current array (Data table), or the
        results folder (a tool's tables)."""
        page = self.stack.currentWidget()
        if page is self.channel_table:
            arrays = self.channel_table.model.arrays
            return self.channel_table.selected_array() or (arrays[0] if arrays else None)
        return getattr(page, "toolfolder", None)

    def show_in_browser(self) -> None:
        shown = self.shown_object()
        if shown is not None:
            self.show_in_browser_requested.emit(shown)

    def copy_all(self) -> None:
        """Copy the whole table (the Stats tables: results and summary)."""
        page = self.stack.currentWidget()
        if page is self.channel_table:
            self.channel_table.view.selectAll()
            self.copy_selection()
        elif hasattr(page, "table_text"):
            QtWidgets.QApplication.clipboard().setText(page.table_text())

    # --- right-click: the ⋯ menu ------------------------------------------

    def _add_context_menu(self, page: QtWidgets.QWidget) -> None:
        """Right-click a cell or column header of *page*'s tables for the
        Table menu."""
        custom = QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        for view in page.findChildren(QtWidgets.QTableView):
            view.setContextMenuPolicy(custom)
            view.customContextMenuRequested.connect(
                lambda pos, v=view: self._show_cell_menu(v, pos)
            )
            header = view.horizontalHeader()
            header.setContextMenuPolicy(custom)
            header.customContextMenuRequested.connect(
                lambda pos, v=view: self._show_header_menu(v, pos)
            )

    def _show_cell_menu(self, view: QtWidgets.QTableView, pos: QtCore.QPoint) -> None:
        # A cell outside the selection becomes the selection (as in a
        # spreadsheet); inside it, the selection is kept for Copy
        index = view.indexAt(pos)
        selection = view.selectionModel()
        if index.isValid() and not selection.isSelected(index):
            selection.setCurrentIndex(
                index, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        self.table_menu.exec(view.viewport().mapToGlobal(pos))

    def _show_header_menu(self, view: QtWidgets.QTableView, pos: QtCore.QPoint) -> None:
        header = view.horizontalHeader()
        column = header.logicalIndexAt(pos)
        selection = view.selectionModel()
        if column >= 0 and not selection.isColumnSelected(column, QtCore.QModelIndex()):
            view.selectColumn(column)
        self.table_menu.exec(header.mapToGlobal(pos))

    # --- tools ------------------------------------------------------------

    def register_tool(self, name: str, page: QtWidgets.QWidget) -> None:
        self._tool_pages[name] = page
        self.stack.addWidget(page)
        self._add_context_menu(page)

    def tool_page(self, name: str) -> QtWidgets.QWidget | None:
        return self._tool_pages.get(name)

    def set_tools(self, names: list[str]) -> None:
        """Show tabs for the registered tools in *names*, in that order."""
        self._tools = [name for name in names if name in self._tool_pages]
        self.update_selection(self._last_selection)

    def show_tool(self, name: str) -> None:
        self.view_tabs.select_tool(name)

    def show_channel_table(self) -> None:
        """Show the channel (or Data) table rather than a tool's tables."""
        channels = self.view_tabs.channel_names
        if channels:
            self.view_tabs.select_channel(self._channel_name or channels[0])

    # --- selection --------------------------------------------------------

    def bind_selection_model(self, model) -> None:
        self._selection_model = model
        model.selection_changed.connect(self.update_selection)
        self.update_selection(model.selection)

    def _channels(self, selection: dict) -> list[str]:
        dataseries = source_object(selection, "dataseries")
        if isinstance(dataseries, NMDataSeries):
            names = list(dataseries.channels.keys())
            channel = selection.get("channel")
            selected = channel.name if channel is not None else None
            # A channel chosen in the strip is shown (a tab shows another until then)
            strip_changed = selected != self._strip_channel_name
            self._strip_channel_name = selected
            if strip_changed and selected in names:
                self._channel_name = selected
            if self._channel_name not in names:
                self._channel_name = (
                    selected if selected in names else names[0] if names else None
                )
            return names
        self._channel_name = None
        if isinstance(source_object(selection, "data"), NMData):
            return [DATA_VIEW]
        return []

    def update_selection(self, selection: dict) -> None:
        self._last_selection = dict(selection)
        channels = self._channels(selection)
        self._updating = True
        self.view_tabs.set_views(
            channels, self._tools, current_channel=self._channel_name or DATA_VIEW
        )
        dataseries = source_object(selection, "dataseries")
        if isinstance(dataseries, NMDataSeries):
            self.view_tabs.mark_channels(
                {name: steps_tooltip(c) for name, c in dataseries.channels.items()}
            )
        self._updating = False
        self._show_current_page()
        self._refresh_channel_table()

    def _on_view_changed(self, kind: str, name: str) -> None:
        if self._updating:
            return
        if kind == "channel" and name != DATA_VIEW:
            self._channel_name = name
        self._show_current_page()
        self._refresh_channel_table()

    def _show_current_page(self) -> None:
        view = self.view_tabs.current_view()
        if view is not None and view[0] == "tool":
            self.stack.setCurrentWidget(self._tool_pages[view[1]])
        else:
            self.stack.setCurrentWidget(self.channel_table)

    def _refresh_channel_table(self) -> None:
        selection = self._last_selection
        dataseries = source_object(selection, "dataseries")
        data = source_object(selection, "data")
        if isinstance(dataseries, NMDataSeries) and self._channel_name is not None:
            try:
                # Excluded epochs (SetX) are still shown, as in the Plot
                names = scoped_epoch_names(dataseries, selection, exclude=False)
            except RuntimeError as error:
                self.channel_table.show_arrays([], str(error))
                return
            scope = epoch_scope_label(selection)
            if names is None:  # no Epoch Set or Group: every epoch
                names = list(dataseries.epochs.keys())
            found = dataseries.epoch_data_map(self._channel_name)
            arrays = [found[name] for name in names if name in found]
            epoch = selection.get("epoch")
            current = found.get(epoch.name) if epoch is not None else None
            noun = "epoch" if len(arrays) == 1 else "epochs"
            title = f"{dataseries.name}  channel {self._channel_name}: {len(arrays)} {noun}"
            if scope:
                title += f" ({scope})"
            title += self._set_transform(dataseries.channels.get(self._channel_name),
                                         current or (arrays[0] if arrays else None))
            self.channel_table.show_arrays(arrays, title, current)
        elif isinstance(data, NMData):
            context = selection.get("toolfolder") or selection.get("folder")
            data_set = selection.get("data_set")
            members = (
                set(context.data.sets.get_items(data_set, get_keys=True) or [])
                if data_set is not None and context is not None else set()
            )
            self._set_transform(None, None)
            if data.name not in members:  # e.g. a new array, in no set yet
                self.channel_table.show_arrays([data], data.name, data)
                return
            arrays = [item for name, item in context.data.items() if name in members]
            noun = "data array" if len(arrays) == 1 else "data arrays"
            title = f"{len(arrays)} {noun} (Data Set: {data_set})"
            self.channel_table.show_arrays(arrays, title, data)
        else:
            self._set_transform(None, None)
            self.channel_table.show_arrays([], "No selection")

    def _set_transform(self, channel, sample: NMData | None) -> str:
        """Show the table with *channel*'s filters and transforms if Show
        Transformed is on; text for the title (e.g. "  ƒ transformed")."""
        on = channel is not None and channel.transform_on
        self.transformed_action.setEnabled(on)
        shown = None
        note = ""
        if on and self.transformed_action.isChecked():
            try:
                if sample is not None:
                    channel.transformed(sample)  # a step may not apply
                shown, note = channel, f"  {MARK} transformed"
            except Exception as e:
                note = f"  ({MARK} not applied: {e})"
        elif on:
            note = f"  ({MARK} not shown)"
        self.channel_table.model.set_transform(shown)
        self.channel_table.title_label.setToolTip(steps_tooltip(channel) if on else "")
        return note


__all__ = [
    "ChannelTable",
    "ChannelTableModel",
    "ChannelToolBar",
    "DATA_VIEW",
    "TablePanel",
    "ValuesPlot",
]
