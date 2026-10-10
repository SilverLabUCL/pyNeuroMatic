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
from pyneuromatic.gui.selection_model import scoped_epoch_names

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


class ChannelTableModel(QtCore.QAbstractTableModel):
    """Data arrays side by side: an x column, then one column per array.

    Cells are formatted only when Qt asks for them, so long recordings with
    many epochs stay responsive.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._columns: list[NMData] = []
        self._rows = 0

    @property
    def arrays(self) -> list[NMData]:
        return list(self._columns)

    def set_arrays(self, arrays: list[NMData]) -> None:
        self.beginResetModel()
        self._columns = [d for d in arrays if d.nparray is not None]
        self._rows = max((d.nparray.size for d in self._columns), default=0)
        self.endResetModel()

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
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        row, column = index.row(), index.column()
        if column == 0:
            value = self._x_value(row)
        else:
            values = self._columns[column - 1].nparray
            value = values[row] if row < values.size else None
        return self.format_value(value)

    @staticmethod
    def format_value(value) -> str:
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return "NaN" if math.isnan(number) else f"{number:.6g}"

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
        self.view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectColumns
        )
        self.view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.view.verticalHeader().setDefaultSectionSize(22)
        layout.addWidget(self.view, stretch=1)

    def show_arrays(
        self, arrays: list[NMData], title: str, current: NMData | None = None
    ) -> None:
        if [id(d) for d in arrays] != [id(d) for d in self.model.arrays]:
            self.model.set_arrays(arrays)
        self.title_label.setText(title)
        self.select_array(current)

    def select_array(self, data: NMData | None) -> None:
        for column, item in enumerate(self.model.arrays, start=1):
            if item is data:
                self.view.selectColumn(column)
                self.view.scrollTo(self.model.index(0, column))
                return
        self.view.clearSelection()

    def selected_array(self) -> NMData | None:
        columns = {index.column() for index in self.view.selectedIndexes()}
        if len(columns) != 1:
            return None
        column = columns.pop()
        return self.model.arrays[column - 1] if column > 0 else None


class TablePanel(QtWidgets.QWidget):
    """Table context panel: channel tables and the tables of tools with results."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.view_tabs = ChannelToolBar(self)
        self.view_tabs.view_changed.connect(self._on_view_changed)
        layout.addWidget(self.view_tabs)
        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack, stretch=1)
        self.channel_table = ChannelTable(self.stack)
        self.stack.addWidget(self.channel_table)

        self._selection_model: Any = None
        self._last_selection: dict = {}
        self._channel_name: str | None = None
        self._tool_pages: dict[str, QtWidgets.QWidget] = {}
        self._tools: list[str] = []
        self._updating = False

    # --- tools ------------------------------------------------------------

    def register_tool(self, name: str, page: QtWidgets.QWidget) -> None:
        self._tool_pages[name] = page
        self.stack.addWidget(page)

    def tool_page(self, name: str) -> QtWidgets.QWidget | None:
        return self._tool_pages.get(name)

    def set_tools(self, names: list[str]) -> None:
        """Show tabs for the registered tools in *names*, in that order."""
        self._tools = [name for name in names if name in self._tool_pages]
        self.update_selection(self._last_selection)

    def show_tool(self, name: str) -> None:
        self.view_tabs.select_tool(name)

    # --- selection --------------------------------------------------------

    def bind_selection_model(self, model) -> None:
        self._selection_model = model
        model.selection_changed.connect(self.update_selection)
        self.update_selection(model.selection)

    def _channels(self, selection: dict) -> list[str]:
        dataseries = selection.get("dataseries")
        if isinstance(dataseries, NMDataSeries):
            names = list(dataseries.channels.keys())
            if self._channel_name not in names:
                channel = selection.get("channel")
                selected = channel.name if channel is not None else None
                self._channel_name = (
                    selected if selected in names else names[0] if names else None
                )
            return names
        self._channel_name = None
        if isinstance(selection.get("data"), NMData):
            return [DATA_VIEW]
        return []

    def update_selection(self, selection: dict) -> None:
        self._last_selection = dict(selection)
        channels = self._channels(selection)
        self._updating = True
        self.view_tabs.set_views(
            channels, self._tools, current_channel=self._channel_name or DATA_VIEW
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
        dataseries = selection.get("dataseries")
        data = selection.get("data")
        if isinstance(dataseries, NMDataSeries) and self._channel_name is not None:
            try:
                names = scoped_epoch_names(dataseries, selection)
            except RuntimeError as error:
                self.channel_table.show_arrays([], str(error))
                return
            scope = selection.get("set") or selection.get("group")
            if names is None:  # no Set or Group: every epoch
                names = list(dataseries.epochs.keys())
            arrays = [
                item
                for name in names
                if isinstance(
                    item := dataseries.get_data(channel=self._channel_name, epoch=name),
                    NMData,
                )
            ]
            epoch = selection.get("epoch")
            current = (
                dataseries.get_data(channel=self._channel_name, epoch=epoch.name)
                if epoch is not None
                else None
            )
            noun = "epoch" if len(arrays) == 1 else "epochs"
            title = f"{dataseries.name}  channel {self._channel_name}: {len(arrays)} {noun}"
            if scope:
                title += f" ({scope})"
            self.channel_table.show_arrays(arrays, title, current)
        elif isinstance(data, NMData):
            self.channel_table.show_arrays([data], data.name, data)
        else:
            self.channel_table.show_arrays([], "No selection")


__all__ = [
    "ChannelTable",
    "ChannelTableModel",
    "ChannelToolBar",
    "DATA_VIEW",
    "TablePanel",
    "ValuesPlot",
]
