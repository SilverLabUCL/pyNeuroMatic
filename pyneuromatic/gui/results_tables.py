"""The two aligned results tables a tool's "2" tab shows (#525).

One row per analysed epoch (or data array) above, a summary of each column
(mean, std, sem...) below. Both share columns, column widths, horizontal
scrolling and the selected column. A tool's tables subclass
:class:`ResultsTables` and say which arrays are columns and what the rows
are (e.g. Stats: a window's ST_ arrays, rows from ST_data_names).
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import pyneuromatic.core.nm_math as nm_math
from pyneuromatic.gui import table_format
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


class ResultsTables(QtWidgets.QWidget):
    """Two aligned tables of a results subfolder's arrays.

    Subclasses override :meth:`_column_names` (the arrays shown), and may
    override :meth:`_rows` (the row names), :meth:`_header` (a column's
    text) and :meth:`_title`.
    """

    # Decimal places of the values shown (Settings); copying keeps them all
    decimals = 3

    # array_stats key -> row label
    SUMMARY_ROWS = (
        ("mean", "mean"),
        ("std", "std"),
        ("sem", "sem"),
        ("N", "N"),
        ("NaNs", "NaNs"),
        ("INFs", "INFs"),
        ("min", "min"),
        ("max", "max"),
    )

    column_selected = QtCore.pyqtSignal()

    toolfolder = None  # the results folder shown (set by show)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns: list[str] = []  # array name per table column
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.title_label = QtWidgets.QLabel("", self)
        self.title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title_label)
        self.message_label = QtWidgets.QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        self.arrays_table = self._make_table()
        self.summary_table = self._make_table()
        # The chosen array's header is bold in both tables (the arrays table's
        # by Qt, as its cell is selected; the summary's here)
        self.arrays_table.currentCellChanged.connect(lambda *_a: self._bold_summary_header())
        # The summary is at most the height of its rows and the arrays get the
        # rest; drag the divider to trade space between them
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.arrays_table)
        self.splitter.addWidget(self.summary_table)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        layout.addWidget(self.splitter, stretch=1)
        self._fit_summary_height()

        for table, other in (
            (self.arrays_table, self.summary_table),
            (self.summary_table, self.arrays_table),
        ):
            table.horizontalScrollBar().valueChanged.connect(
                other.horizontalScrollBar().setValue
            )
            table.horizontalHeader().sectionResized.connect(
                lambda column, _old, size, other=other: self._sync_width(
                    other, column, size
                )
            )
            table.itemSelectionChanged.connect(
                lambda table=table, other=other: self._sync_selection(table, other)
            )

    def _make_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(self)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # One cell is highlighted, not its whole column; the current cell's
        # column is the chosen array (its header is bold)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        # Cells, a header's column, or several (Shift / ⌘); ⌘C copies them
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        return table

    # --- hooks ---------------------------------------------------------------

    def _column_names(self, toolfolder: NMToolFolder, key, **options) -> list[str]:
        """The arrays shown as columns, for *key* (e.g. Stats' window)."""
        return []

    def _rows(self, toolfolder: NMToolFolder, key) -> list[str]:
        """The rows' names (e.g. the data arrays analysed); a row without
        one is named by its index."""
        return []

    def _header(self, name: str, key) -> str:
        """A column's header text for array *name* (its name)."""
        return name

    def _title(self, toolfolder: NMToolFolder, key) -> str:
        return toolfolder.name if key is None else f"{toolfolder.name}  {key}"

    def _summary(self, toolfolder: NMToolFolder, name: str) -> dict:
        """Array *name*'s summary (mean, std, sem, N...), NaNs left out."""
        data = toolfolder.data.get(name)
        values = getattr(data, "nparray", None)
        if not isinstance(values, np.ndarray):
            return {}
        summary: dict = {}
        nm_math.array_stats(values, ignore_nans=True, results=summary)
        return summary

    # --- keeping the two tables aligned -----------------------------------

    def _sync_width(self, other: QtWidgets.QTableWidget, column: int, size: int) -> None:
        if column < other.columnCount() and other.columnWidth(column) != size:
            other.setColumnWidth(column, size)

    def _sync_selection(
        self, table: QtWidgets.QTableWidget, other: QtWidgets.QTableWidget
    ) -> None:
        if self._syncing:
            return
        self._syncing = True
        # The other table follows the column (current, without a highlight)
        columns = {index.column() for index in table.selectedIndexes()}
        column = columns.pop() if len(columns) == 1 else table.currentColumn()
        if table is self.summary_table and column >= 0:  # the arrays table names it
            self._set_current_column(self.arrays_table, column)
        other.clearSelection()
        self._set_current_column(other, column)
        self._syncing = False
        self.column_selected.emit()

    def _bold_summary_header(self) -> None:
        """The summary table's header of the chosen array (the arrays table's
        current column) in bold, as the arrays table's is."""
        column = self.arrays_table.currentColumn()
        for c in range(self.summary_table.columnCount()):
            item = self.summary_table.horizontalHeaderItem(c)
            if item is None:
                continue
            font = item.font()
            if font.bold() != (c == column):
                font.setBold(c == column)
                item.setFont(font)

    @staticmethod
    def _set_current_column(table: QtWidgets.QTableWidget, column: int) -> None:
        """Make *column* current in *table* without highlighting a cell."""
        if 0 <= column < table.columnCount() and table.rowCount():
            row = min(max(table.currentRow(), 0), table.rowCount() - 1)
            table.selectionModel().setCurrentIndex(
                table.model().index(row, column),
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
            )

    # --- content ----------------------------------------------------------

    def show_results(self, toolfolder: NMToolFolder | None, key=None,
                     message: str = "", **options) -> None:
        """Fill both tables from *toolfolder*'s arrays for *key* (e.g. a
        window); with no *toolfolder*, empty (*message* says why)."""
        self.toolfolder = toolfolder  # what the tables show (Show in Browser)
        for table in (self.arrays_table, self.summary_table):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
        self._columns = []
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        if toolfolder is None:
            self.title_label.setText("")
            self.column_selected.emit()
            return
        self.title_label.setText(self._title(toolfolder, key))

        self._columns = self._column_names(toolfolder, key, **options)
        rows = self._rows(toolfolder, key)
        lengths = [len(toolfolder.data.get(name).nparray) for name in self._columns]
        count = max([len(rows)] + lengths)
        rows = rows + [str(i) for i in range(len(rows), count)]
        self.arrays_table.setColumnCount(len(self._columns))
        self.arrays_table.setRowCount(len(rows))
        self.arrays_table.setVerticalHeaderLabels(rows)
        self.summary_table.setColumnCount(len(self._columns))
        self.summary_table.setRowCount(len(self.SUMMARY_ROWS))
        self.summary_table.setVerticalHeaderLabels(
            [label for _key, label in self.SUMMARY_ROWS]
        )

        for column, name in enumerate(self._columns):
            data = toolfolder.data.get(name)
            units = data.yscale.units
            for table in (self.arrays_table, self.summary_table):
                header = QtWidgets.QTableWidgetItem(self._header(name, key))
                header.setToolTip(f"{name} ({units})" if units else name)
                table.setHorizontalHeaderItem(column, header)
            for row, value in enumerate(data.nparray):
                self.arrays_table.setItem(row, column, self._value_item(value))
            summary = self._summary(toolfolder, name)
            for row, (stat, _label) in enumerate(self.SUMMARY_ROWS):
                self.summary_table.setItem(
                    row, column, self._value_item(summary.get(stat, math.nan))
                )
        self._align_columns()
        self._bold_summary_header()
        self.column_selected.emit()

    def _align_columns(self) -> None:
        for table in (self.arrays_table, self.summary_table):
            table.resizeColumnsToContents()
        for column in range(len(self._columns)):
            width = max(
                self.arrays_table.columnWidth(column),
                self.summary_table.columnWidth(column),
            )
            self.arrays_table.setColumnWidth(column, width)
            self.summary_table.setColumnWidth(column, width)
        # Equal row-label widths keep the data columns lined up
        label_width = max(
            self.arrays_table.verticalHeader().sizeHint().width(),
            self.summary_table.verticalHeader().sizeHint().width(),
        )
        for table in (self.arrays_table, self.summary_table):
            table.verticalHeader().setFixedWidth(label_width)
        self._fit_summary_height()

    def _fit_summary_height(self) -> None:
        table = self.summary_table
        height = (
            table.horizontalHeader().sizeHint().height()
            + table.verticalHeader().defaultSectionSize() * len(self.SUMMARY_ROWS)
            + table.horizontalScrollBar().sizeHint().height()
            + 2 * table.frameWidth()
        )
        table.setMaximumHeight(height)

    @classmethod
    def _value_text(cls, value) -> str:
        """Plain text for a value (NaN always as "NaN"), to the decimals shown."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return str(int(value))
        if not math.isfinite(number):
            return "NaN" if math.isnan(number) else ("-inf" if number < 0 else "inf")
        return table_format.fixed(number, cls.decimals)

    @classmethod
    def _value_item(cls, value) -> QtWidgets.QTableWidgetItem:
        """A table cell; NaN as the Settings say (NaN, grey or blank)."""
        item = QtWidgets.QTableWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, value)  # for refresh()
        cls._style_item(item, value)
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        return item

    @classmethod
    def _style_item(cls, item: QtWidgets.QTableWidgetItem, value) -> None:
        text, is_nan = table_format.cell(value, decimals=cls.decimals)
        item.setText(text)
        color = table_format.nan_foreground() if is_nan else None
        item.setForeground(QtGui.QBrush(color) if color is not None else QtGui.QBrush())
        item.setToolTip((table_format.nan_tooltip() or "") if is_nan else "")

    def refresh(self) -> None:
        """Restyle the cells (e.g. after the NaN setting changes)."""
        for table in (self.arrays_table, self.summary_table):
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item is not None:
                        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
                        self._style_item(item, value)

    # --- selection and export ---------------------------------------------

    def selected_array(self) -> str | None:
        """The array name of the current cell's column, if any."""
        column = self.arrays_table.currentColumn()
        if not 0 <= column < len(self._columns):
            return None
        return self._columns[column]

    def select_array(self, name: str) -> None:
        if name in self._columns:
            row = max(self.arrays_table.currentRow(), 0)
            self.arrays_table.setCurrentCell(row, self._columns.index(name))

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @staticmethod
    def _table_lines(table: QtWidgets.QTableWidget) -> list[str]:
        columns = range(table.columnCount())
        lines = [
            "\t".join([""] + [table.horizontalHeaderItem(c).text() for c in columns])
        ]
        for row in range(table.rowCount()):
            cells = [table.item(row, c).text() if table.item(row, c) is not None else ""
                     for c in columns]
            lines.append("\t".join([table.verticalHeaderItem(row).text()] + cells))
        return lines

    def copy_text(self, names: bool = True, row_names: bool = True) -> str:
        """The selected cells (of the results or the summary table) as
        tab-separated text: whole columns with their names first (*names*),
        each row's name first (*row_names*: the data, or the statistic)."""
        table = self.arrays_table
        if not table.selectedIndexes() and self.summary_table.selectedIndexes():
            table = self.summary_table

        def value(row, column):
            item = table.item(row, column)
            if item is None:
                return ""
            raw = item.data(QtCore.Qt.ItemDataRole.UserRole)
            return table_format.copy_value(raw) if raw is not None else item.text()

        def header(column):
            item = table.horizontalHeaderItem(column)
            return item.text() if item is not None else ""

        def row_label(row):
            item = table.verticalHeaderItem(row)
            return item.text() if item is not None else ""

        # Row names with whole columns (a block of cells copies just itself)
        selection = table.selectionModel()
        columns = {i.column() for i in selection.selectedIndexes()}
        whole = bool(columns) and all(
            selection.isColumnSelected(c, QtCore.QModelIndex()) for c in columns
        )
        return table_format.selection_text(
            table, value, header, row_label=row_label if row_names and whole else None,
            names=names,
        )

    def table_text(self) -> str:
        """Both tables, with labels, as tab-separated text (blank line between)."""
        return "\n".join(
            self._table_lines(self.arrays_table)
            + [""]
            + self._table_lines(self.summary_table)
        )


__all__ = ["ResultsTables"]
