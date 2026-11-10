"""Stats2 panel and the Stats tables it shows in the Table context panel."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import table_format
from pyneuromatic.gui.histogram_dialog import HistogramDialog
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.tools.nm_tool_stats import NMToolStats, NMToolStats2


def display_name(name: str) -> str:
    """A measurement as Stats shows it: mean+std, mean+var and mean+sem as
    "mean ± std"... (a mean and its spread, not a sum). The core name, in
    commands and scripts, is unchanged."""
    if name and name.startswith("mean+"):
        return "mean \u00b1 " + name[len("mean+"):]
    return name


_NOTE_PAIR = re.compile(r"(\w+)=([^,()]+)")


def _note_fields(data) -> dict[str, str]:
    """An ST_ array's note as fields, e.g. "NMStats(win=w0, func=max,
    id=main, xbgn=60.0, xend=120.0, n=2)" -> {"func": "max", "id": "main", ...}."""
    for note in data.notes:
        if str(note["note"]).startswith("NMStats("):
            return {k: v.strip() for k, v in _NOTE_PAIR.findall(note["note"])}
    return {}


def _note_limits(note: dict, x_limits: tuple[float, float] | None) -> tuple[str, str]:
    """A note's X range as text, an unbounded edge at the trace's start or
    end (*x_limits*), as Stats1 shows it."""
    bounds = []
    for key, limit in (("xbgn", 0), ("xend", 1)):
        text = note.get(key)
        try:
            if x_limits is not None and math.isinf(float(text)):
                text = x_limits[limit]
        except (TypeError, ValueError):
            pass
        bounds.append(_bound(text))
    return bounds[0], bounds[1]


def _with(text: str, units: str) -> str:
    """*text* and its *units*, e.g. "120 ms"."""
    return f"{text} {units}" if units else text


def _bound(text) -> str:
    """An X range bound from a note, e.g. "60.0" -> "60", "inf" -> "inf"."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return str(text)
    return f"{value:g}" if math.isfinite(value) else ("-inf" if value < 0 else "inf")


def epoch_result_lines(toolfolder: NMToolFolder, window: str, row: int,
                       digits: int = 6, xunits: str = "",
                       x_limits: tuple[float, float] | None = None,
                       known_units: tuple[str, ...] = ()) -> list[str]:
    """A window's saved result for one row (epoch) of its ST_ arrays, as text
    in the format of Stats1's Results (#475), e.g.

        bsln mean = -70 mV  (x 0 to 40)
        peak = -55 mV at 80 ms   Δ = 15 mV
        p0 at 61.774 ms
        p1 at 76.116 ms
        risetime+ = 14.342 ms

    From the saved arrays and their notes, so what was saved; a crossing's
    level (and its %) isn't saved, so it's shown by its step (p0, p1).
    Values in *known_units* (e.g. given once in a header) are shown without
    them; others (e.g. a slope's mV/ms) keep theirs.
    """
    def number(value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NaN"
        return "NaN" if math.isnan(value) else f"{value:.{digits}g}"

    def value(data) -> str:
        values = data.nparray
        text = number(values[row]) if row < len(values) else "NaN"
        units = data.yscale.units
        return f"{text} {units}".strip() if units and units not in known_units else text

    prefix = f"ST_{window}_"
    groups: dict[tuple[str, str], list[tuple[str, object]]] = {}
    fields: dict[tuple[str, str], dict[str, str]] = {}
    for name, data in toolfolder.data.items():
        if not name.startswith(prefix) or name == f"{prefix}data":
            continue
        suffix = name.rsplit("_", 1)[-1]
        if suffix in StatsTables._DETAIL_SUFFIXES or not isinstance(data.nparray, np.ndarray):
            continue
        note = _note_fields(data)
        key = (note.get("id", ""), note.get("step", ""))
        groups.setdefault(key, []).append((name[len(prefix):], data))
        fields.setdefault(key, note)
    lines: list[str] = []
    bsln_units = None
    for key, arrays in groups.items():
        step_id, step = key
        note = fields[key]
        func = note.get("func", "")
        stepfunc = note.get("stepfunc")  # the function that made a step (#476)
        if step_id == "bsln":
            # Its value is the mean (a spread follows): "bsln mean", as the mean's
            label = "bsln " + ("mean" if func.startswith("mean+") else func)
        elif step:  # a crossing: e.g. "level+ 10%", as Stats1 (older: "p0")
            percent = note.get(step)
            label = f"{stepfunc} {_bound(percent)}%" if stepfunc and percent else step
        else:  # a pipeline's peak is the max or min (older: "peak")
            label = func if step_id == "main" else (stepfunc or "peak")
            if label.startswith("mean+"):  # its value is the mean (its std follows)
                label = "mean"
        main = None
        parts: list[str] = []
        delta = dx = None
        for short, data in arrays:
            suffix = short.rsplit("_", 1)[-1]
            base = short[:-len(suffix) - 1]
            if suffix == "y":
                if main is None:
                    main = f"{label} = {value(data)}"
                elif base == "var" or base.endswith("_var"):  # squared units: none
                    parts.append(f", \u00b1{base} "
                                 f"{number(data.nparray[row]) if row < len(data.nparray) else 'NaN'}")
                else:  # mean+std's std: the mean's spread, as ±std (copies as written)
                    parts.append(f", \u00b1{base} {value(data)}")
            elif suffix in ("std", "sem", "var") and base == "bsln":
                # A baseline's spread (e.g. mean ± std), as the mean's
                parts.append(f", \u00b1{suffix} " + (
                    value(data) if suffix != "var" else number(data.nparray[row])))
            elif suffix == "x":
                parts.append(f" at {value(data)}")
            elif suffix == "b":
                parts.append(f", intercept {value(data)}")
            elif suffix == "dy":
                delta = data
            elif suffix == "dx":
                dx = data
        if step_id == "bsln":
            bsln_units = arrays[0][1].yscale.units if arrays else None
        line = (main if main is not None else label) + "".join(parts)
        if step_id == "bsln":
            xbgn, xend = _note_limits(note, x_limits)
            line += f"  (x {xbgn} to {_with(xend, xunits if xunits not in known_units else '')})"
        if delta is not None and bsln_units is not None \
                and delta.yscale.units == bsln_units and number(delta.nparray[row]) != "NaN":
            line += f"   Δ = {value(delta)}"
        lines.append(line)
        if dx is not None:
            lines.append(f"{func} = {value(dx)}")
    return lines


class StatsTables(QtWidgets.QWidget):
    """Two aligned tables of one Stats window's saved ST_ arrays.

    The upper table has one row per analysed data array (from
    ``ST_<win>_data``); the lower table summarises each column (mean, std,
    ...). Both tables share columns, column widths, horizontal scrolling and
    the selected column.
    """

    # Significant digits of the values shown (Settings › Stats, #471);
    # copying keeps them all
    digits = 6

    # Bookkeeping arrays hidden unless all columns are requested
    _DETAIL_SUFFIXES = frozenset(("i", "i0", "i1", "n", "nans", "infs"))
    # NMToolStats2.stats key -> row label
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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns: list[str] = []  # ST_ array name per table column
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

    toolfolder = None  # the results folder shown (set by show_window)

    def _make_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(self)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # One cell is highlighted, not its whole column; the current cell's
        # column is the chosen ST_ array (its header is bold)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        # Cells, a header's column, or several (Shift / ⌘); ⌘C copies them
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        return table

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

    def show_window(
        self,
        toolfolder: NMToolFolder | None,
        window: str | None,
        all_columns: bool = False,
        message: str = "",
    ) -> None:
        """Fill both tables from *window*'s ST_ arrays in *toolfolder*."""
        self.toolfolder = toolfolder  # what the tables show (Show in Browser)
        for table in (self.arrays_table, self.summary_table):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
        self._columns = []
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        if toolfolder is None or window is None:
            self.title_label.setText("")
            self.column_selected.emit()
            return
        self.title_label.setText(f"{toolfolder.name}  {window}")

        prefix = f"ST_{window}_"
        self._columns = self._window_arrays(toolfolder, window, all_columns)
        rows = self._row_labels(toolfolder, window)
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
                # The window's arrays by their own part (max_y); the rows'
                # epoch numbers (ST_epoch_num, no window prefix) as "epoch"
                label = name[len(prefix):] if name.startswith(prefix) else (
                    "epoch" if name == "ST_epoch_num" else name)
                header = QtWidgets.QTableWidgetItem(label)
                header.setToolTip(f"{name} ({units})" if units else name)
                table.setHorizontalHeaderItem(column, header)
            for row, value in enumerate(data.nparray):
                self.arrays_table.setItem(row, column, self._value_item(value))
            summary = NMToolStats2.stats(toolfolder, select=name).get(name, {})
            for row, (key, _label) in enumerate(self.SUMMARY_ROWS):
                self.summary_table.setItem(
                    row, column, self._value_item(summary.get(key, math.nan))
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
    def _window_arrays(
        cls, toolfolder: NMToolFolder, window: str, all_columns: bool
    ) -> list[str]:
        prefix = f"ST_{window}_"
        names = []
        for name, data in toolfolder.data.items():
            if not name.startswith(prefix) or name == f"{prefix}data":
                continue
            values = data.nparray
            if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
                continue  # e.g. warning text arrays
            if name.rsplit("_", 1)[-1] in cls._DETAIL_SUFFIXES and not all_columns:
                continue
            names.append(name)
        if all_columns and "ST_epoch_num" in toolfolder.data:  # the rows' epochs (#476)
            names.insert(0, "ST_epoch_num")
        return names

    @staticmethod
    def _row_labels(toolfolder: NMToolFolder, window: str) -> list[str]:
        """The rows' data array names: ST_data_names (#476), or ST_{window}_data
        in subfolders saved before."""
        data = toolfolder.data.get("ST_data_names") or toolfolder.data.get(f"ST_{window}_data")
        if data is None or data.nparray is None:
            return []
        # Data paths look like "nm.Folder.RecordA0"; show the data name
        return [str(path).rsplit(".", 1)[-1] for path in data.nparray]

    @staticmethod
    def _value_text(value) -> str:
        """Plain text for a value (NaN always as "NaN"), to the digits shown."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return "NaN" if math.isnan(number) else f"{number:.{StatsTables.digits}g}"

    @staticmethod
    def _value_item(value) -> QtWidgets.QTableWidgetItem:
        """A table cell; NaN as the Settings say (NaN, grey or blank)."""
        item = QtWidgets.QTableWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, value)  # for refresh()
        StatsTables._style_item(item, value)
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        return item

    @staticmethod
    def _style_item(item: QtWidgets.QTableWidgetItem, value) -> None:
        text, is_nan = table_format.cell(value, StatsTables.digits)
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
        """ST_ array name of the current cell's column, if any."""
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
            cells = [table.item(row, c).text() for c in columns]
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


class Stats2Panel(QtWidgets.QWidget):
    """Stats2 controls: choose saved Stats results and act on their columns.

    The tables themselves are in :attr:`tables`, shown in the context panel's
    Table tab.
    """

    TOOLFOLDER_PREFIX = NMToolStats.TOOLFOLDER_PREFIX  # "Stats_" (#494)

    # title, y values, x values (or None), x label, y label, style
    plot_requested = QtCore.pyqtSignal(str, object, object, str, str, str)
    # The list of Stats toolfolders in the current folder was reloaded
    results_changed = QtCore.pyqtSignal()
    # The ST_ array chosen in the Array list (or its folder or window) changed
    array_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._selection_model: Any = None
        self._folder: NMFolder | None = None
        # Owned here until a TableTab takes it (addWidget reparents it); hidden
        # so it is never drawn inside this panel
        self.tables = StatsTables(self)
        self.tables.hide()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        # Rows span the panel, so the Array list has its row's width (macOS
        # keeps a form's fields at their size hint, cutting off array names)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        folder_row = QtWidgets.QHBoxLayout()
        self.folder_combo = QtWidgets.QComboBox(self)
        self.folder_combo.setObjectName("stats2Folder")
        # As wide as its subfolder names (refresh)
        folder_row.addWidget(self.folder_combo)
        # The subfolder's actions, as the strip's Subfolder ⋯ (#494): the
        # app's folder actions (set_folder_actions), starting on the one shown
        self.folder_menu = QtWidgets.QMenu("Folder", self)
        self.folder_menu.setToolTipsVisible(True)
        self.rename_folder_action = self.folder_menu.addAction("Rename…")
        self.rename_folder_action.setToolTip("Rename the subfolder (its name keeps Stats_)")
        self.copy_folder_action = self.folder_menu.addAction("Copy…")
        self.copy_folder_action.setToolTip("Copy the subfolder")
        self.delete_folder_action = self.folder_menu.addAction("Delete…")
        self.delete_folder_action.setToolTip("Delete the subfolder and its data")
        self.folder_menu.addSeparator()
        self.browse_folder_action = self.folder_menu.addAction("Show in Browser")
        self.browse_folder_action.setToolTip("Show the subfolder in the browser")
        self.folder_menu_button = QtWidgets.QToolButton(self)
        self.folder_menu_button.setText("⋯")
        self.folder_menu_button.setToolTip("Subfolder actions")
        self.folder_menu_button.setMenu(self.folder_menu)
        self.folder_menu_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.folder_menu_button.setStyleSheet(
            "QToolButton::menu-indicator { image: none; width: 0px; }")
        folder_row.addWidget(self.folder_menu_button)
        folder_row.addStretch(1)
        self._folder_actions: Any = None
        self._folder_names: list[str] = []
        form.addRow("Folder", folder_row)

        window_row = QtWidgets.QHBoxLayout()
        self.window_combo = QtWidgets.QComboBox(self)
        self.window_combo.setObjectName("stats2Window")
        self.window_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        window_row.addWidget(self.window_combo)
        self.all_columns_checkbox = QtWidgets.QCheckBox("All columns", self)
        self.all_columns_checkbox.setToolTip(
            "Also list index and point-count arrays (i, i0, i1, n, nans, infs)"
        )
        window_row.addWidget(self.all_columns_checkbox)
        window_row.addStretch(1)
        form.addRow("Window", window_row)

        array_row = QtWidgets.QHBoxLayout()
        self.array_combo = QtWidgets.QComboBox(self)
        self.array_combo.setObjectName("stats2Array")
        array_row.addWidget(self.array_combo, stretch=1)
        self.plot_button = QtWidgets.QPushButton("Plot", self)
        self.plot_button.setToolTip("Show the chosen array in the Plot panel")
        array_row.addWidget(self.plot_button)
        # Plot's options, in a ⋯ menu beside it (#485)
        self.plot_menu = QtWidgets.QMenu("Plot", self)
        self.plot_menu.setToolTipsVisible(True)
        # Plot against the rows' epoch numbers (ST_epoch_num, #476), not the index
        self.epoch_x_action = self.plot_menu.addAction("x: epoch")
        self.epoch_x_action.setCheckable(True)
        self.epoch_x_action.setChecked(True)
        self.epoch_x_action.setToolTip(
            "Plot against the epoch numbers (ST_epoch_num); off: the row index")
        self.plot_menu.addSeparator()
        # Saved as H_<array>, listed with the ST_ arrays to plot (#492)
        self.histogram_dialog_action = self.plot_menu.addAction("Histogram...")
        self.histogram_dialog_action.setToolTip(
            "Make a histogram of the chosen array, saved in the Stats subfolder "
            "as H_<array> and listed with its arrays")
        self.plot_menu.addSeparator()
        self.browse_array_action = self.plot_menu.addAction("Show in Browser")
        self.browse_array_action.setToolTip("Show the chosen array in the browser")
        self.plot_menu_button = QtWidgets.QToolButton(self)
        self.plot_menu_button.setText("⋯")
        self.plot_menu_button.setToolTip("Plot options")
        self.plot_menu_button.setMenu(self.plot_menu)
        self.plot_menu_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.plot_menu_button.setStyleSheet(
            "QToolButton::menu-indicator { image: none; width: 0px; }")
        array_row.addWidget(self.plot_menu_button)
        form.addRow("Array", array_row)
        layout.addLayout(form)

        self.message_label = QtWidgets.QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        details_group = QtWidgets.QGroupBox(self)  # titled by its Results choice
        details_layout = QtWidgets.QVBoxLayout(details_group)
        # Show: the chosen array's summary, or every epoch's result, a line
        # each (in the format of Stats1's Results, #475)
        show_row = QtWidgets.QHBoxLayout()
        show_row.addWidget(QtWidgets.QLabel("Results", details_group))
        self.results_combo = QtWidgets.QComboBox(details_group)
        self.results_combo.addItem("Summary", "summary")
        self.results_combo.addItem("All epochs", "epochs")
        self.results_combo.setToolTip(
            "Summary: the chosen array (units, notes, mean, std...). All epochs: "
            "the saved result of every epoch the run computed, a line each, to copy")
        show_row.addWidget(self.results_combo)
        show_row.addStretch(1)
        details_layout.addLayout(show_row)
        self.details = QtWidgets.QPlainTextEdit(details_group)
        self.details.setReadOnly(True)
        self.details.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        details_layout.addWidget(self.details)
        layout.addWidget(details_group, stretch=1)

        self.folder_combo.currentIndexChanged.connect(self._on_folder_changed)
        self.window_combo.currentIndexChanged.connect(self._show_tables)
        self.all_columns_checkbox.toggled.connect(self._show_tables)
        self.array_combo.currentIndexChanged.connect(self._on_array_changed)
        self.tables.column_selected.connect(self._on_table_column_selected)
        self.results_combo.currentIndexChanged.connect(self._update_details)
        self.plot_button.clicked.connect(self.plot_selected)
        self.epoch_x_action.toggled.connect(lambda _on: self.array_changed.emit())
        self.histogram_dialog_action.triggered.connect(self.histogram_dialog)
        self.rename_folder_action.triggered.connect(self.rename_folder)
        self.copy_folder_action.triggered.connect(self.copy_folder)
        self.delete_folder_action.triggered.connect(self.delete_folder)
        self.browse_folder_action.triggered.connect(self.show_folder_in_browser)
        self.browse_array_action.triggered.connect(self.show_array_in_browser)
        self._histogram_settings: dict = {}  # the dialog's, shown again next time

        self.refresh()

    # --- selection -------------------------------------------------------

    def bind_selection_model(self, selection_model) -> None:
        self._selection_model = selection_model
        selection_model.selection_changed.connect(self._on_selection_changed)
        self.refresh()

    def _current_folder(self) -> NMFolder | None:
        if self._selection_model is not None:
            folder = self._selection_model.selection.get("folder")
        else:
            folder = self._manager.select_values.get("folder")
        return folder if isinstance(folder, NMFolder) else None

    def _on_selection_changed(self, *_args) -> None:
        # Trace navigation changes the selection often; only a new folder
        # changes what is shown
        if self._current_folder() is not self._folder:
            self.refresh()

    # --- folder and window lists ------------------------------------------

    def toolfolder(self) -> NMToolFolder | None:
        """The Stats toolfolder chosen in the Folder list."""
        if self._folder is None:
            return None
        name = self.folder_combo.currentData()
        if name is None:
            return None
        return self._folder.toolfolders.get(name)

    def refresh(self, select: str | None = None) -> None:
        """Reload the Stats toolfolder list, choosing *select* if given.

        Otherwise the current choice is kept, or the newest folder is chosen.
        """
        self._folder = self._current_folder()
        previous = select or self.folder_combo.currentData()
        names: list[str] = []
        if self._folder is not None:
            names = [
                name
                for name in self._folder.toolfolders.keys()
                if name.startswith(self.TOOLFOLDER_PREFIX)
            ]
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        for name in names:
            self.folder_combo.addItem(name, name)
        self._folder_names = names
        metrics = self.folder_combo.fontMetrics()
        longest = max(names, key=metrics.horizontalAdvance) if names else "Stats_Record_A_All_0"
        self.folder_combo.setFixedWidth(metrics.horizontalAdvance(longest) + 44)

        # The one before, else the newest
        index = names.index(previous) if previous in names else len(names) - 1
        self.folder_combo.setCurrentIndex(index)
        self.folder_combo.blockSignals(False)
        self._on_folder_changed()
        self.results_changed.emit()

    def has_results(self) -> bool:
        """True if the current folder has saved Stats results."""
        return bool(self._folder_names)

    def show_toolfolder(self, toolfolder: NMToolFolder | None) -> None:
        """Show *toolfolder*, e.g. the one a Stats run just wrote, with its
        measurement chosen (not its baseline)."""
        self.refresh(select=toolfolder.name if toolfolder is not None else None)
        measure = self.measure_array()
        if measure is not None:
            self.select_array(measure)

    @staticmethod
    def _windows(toolfolder: NMToolFolder) -> list[str]:
        """The windows with results in *toolfolder*, from the ST_{window}_...
        array names (not the rows' ST_data_names and ST_epoch_num, #476)."""
        windows: list[str] = []
        for name in toolfolder.data.keys():
            if not name.startswith("ST_") or name in ("ST_data_names", "ST_epoch_num"):
                continue
            window = name[len("ST_"):].split("_", 1)[0]
            if window and window not in windows:
                windows.append(window)
        return windows

    def _on_folder_changed(self, *_args) -> None:
        self._update_folder_actions()
        previous = self.window_combo.currentData()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        toolfolder = self.toolfolder()
        if toolfolder is not None:
            for window in self._windows(toolfolder):
                self.window_combo.addItem(window, window)
        index = self.window_combo.findData(previous)
        self.window_combo.setCurrentIndex(max(index, 0))
        self.window_combo.blockSignals(False)
        self._show_tables()

    def _show_tables(self, *_args) -> None:
        toolfolder = self.toolfolder()
        if self._folder is None:
            message = "Select a folder to see its Stats results."
        elif toolfolder is None:
            message = (
                f"No saved Stats results in {self._folder.name}. "
                "Run Stats1 with 'Stats arrays' on."
            )
        else:
            message = ""
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        self.tables.show_window(
            toolfolder,
            self.window_combo.currentData(),
            all_columns=self.all_columns_checkbox.isChecked(),
            message=message,
        )
        self._refresh_array_combo()

    # --- array selection --------------------------------------------------

    def selected_array(self) -> str | None:
        """ST_ array name chosen in the Array list."""
        return self.array_combo.currentData()

    def measure_array(self) -> str | None:
        """The window's measurement among its arrays: a rise time, decay time
        or FWHM's result (``_dx``), else the first that isn't the baseline,
        e.g. ST_w0_max_y rather than ST_w0_bsln_y."""
        names = [n for n in self.tables.columns if n != "ST_epoch_num"]
        for name in names:
            if name.endswith("_dx"):
                return name
        for name in names:
            if "_bsln_" not in name:
                return name
        return names[0] if names else None

    def select_array(self, name: str) -> None:
        index = self.array_combo.findData(name)
        if index >= 0:
            self.array_combo.setCurrentIndex(index)

    def _refresh_array_combo(self) -> None:
        previous = self.array_combo.currentData()
        self.array_combo.blockSignals(True)
        self.array_combo.clear()
        for name in self.tables.columns + self._histograms():
            self.array_combo.addItem(name, name)
        index = self.array_combo.findData(previous)
        if index < 0:  # not in these results: their measurement
            index = self.array_combo.findData(self.measure_array())
        self.array_combo.setCurrentIndex(max(index, 0) if self.array_combo.count() else -1)
        self.array_combo.blockSignals(False)
        self._on_array_changed()

    def _histograms(self) -> list[str]:
        """The window's histograms (H_ST_<window>_..., #492), listed after
        its ST_ arrays; not their bin edges (H_..._edges)."""
        toolfolder = self.toolfolder()
        window = self.window_combo.currentData()
        if toolfolder is None or window is None:
            return []
        prefix = f"H_ST_{window}_"
        return [name for name in toolfolder.data.keys()
                if name.startswith(prefix) and not name.endswith("_edges")]

    @staticmethod
    def is_histogram(name: str | None) -> bool:
        """True if *name* is a saved histogram (H_<array>, #492)."""
        return bool(name) and name.startswith("H_")

    def _on_array_changed(self, *_args) -> None:
        name = self.selected_array()
        if name is not None and self.tables.selected_array() != name:
            self.tables.select_array(name)
        self._update_details()
        self._update_buttons()
        self.array_changed.emit()

    def _on_table_column_selected(self) -> None:
        # Clicking a column in either table picks that array here too
        name = self.tables.selected_array()
        if name is not None and name != self.selected_array():
            self.select_array(name)

    def refresh_details(self) -> None:
        """Show the Results again (e.g. with other digits)."""
        self._update_details()

    def _update_details(self, *_args) -> None:
        toolfolder = self.toolfolder()
        if self.results_combo.currentData() == "epochs":
            self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.details.setPlainText(self.epochs_text() if toolfolder is not None else "")
            return
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        name = self.selected_array()
        if name is None or toolfolder is None:
            self.details.clear()
            return
        self.details.setPlainText(
            self.array_details(toolfolder, self.window_combo.currentData(), name,
                               self._x_units(toolfolder, self.window_combo.currentData()))
        )

    # --- every epoch's saved result, as a list (#475) ----------------------

    def _x_limits(self, toolfolder: NMToolFolder,
                  window: str) -> tuple[float, float] | None:
        """Where the measured traces start and end (the first data array the
        run measured), for an unbounded X range edge (-inf, inf)."""
        folder = self._folder
        for label in StatsTables._row_labels(toolfolder, window):
            source = folder.data.get(label) if folder is not None and label else None
            if source is not None and source.nparray is not None and source.nparray.size:
                first, last = source.get_xvalue(0), source.get_xvalue(source.nparray.size - 1)
                if first is not None and last is not None:
                    return first, last
        return None

    def _x_units(self, toolfolder: NMToolFolder, window: str) -> str:
        """The x units of the run: a saved x array's (e.g. ST_w0_max_x, in
        ms), else those of the data arrays it measured."""
        for name, data in toolfolder.data.items():
            if name.startswith(f"ST_{window}_") and name.endswith("_x") and data.yscale.units:
                return data.yscale.units
        folder = self._folder
        for label in StatsTables._row_labels(toolfolder, window):
            source = folder.data.get(label) if folder is not None and label else None
            if source is not None:
                return source.xscale.units or ""
        return ""

    def epochs_text(self) -> str:
        """Results (All epochs): a line per epoch (row) the run computed, in
        the format of Stats1's Results, to copy."""
        toolfolder, window = self.toolfolder(), self.window_combo.currentData()
        if toolfolder is None or window is None:
            return ""
        main = next((d for n, d in toolfolder.data.items()
                     if n.startswith(f"ST_{window}_") and _note_fields(d).get("id")
                     not in (None, "bsln")), None)
        note = _note_fields(main) if main is not None else {}
        xunits = self._x_units(toolfolder, window)
        x_limits = self._x_limits(toolfolder, window)
        header = f"{toolfolder.name}  {window}"
        if note.get("func"):
            xbgn, xend = _note_limits(note, x_limits)
            header += f" {display_name(note['func'])}  (x {xbgn} to {xend})"  # x units: below
        bsln = next((d for n, d in toolfolder.data.items()
                     if n.startswith(f"ST_{window}_bsln_")), None)
        if bsln is not None:
            bsln_note = _note_fields(bsln)
            xbgn, xend = _note_limits(bsln_note, x_limits)
            header += (f"  bsln {display_name(bsln_note.get('func', ''))} "
                       f"(x {xbgn} to {xend})")
        # Units once, here, not on every value: y (the main array's) and x
        yunits = main.yscale.units if main is not None else ""
        known = tuple(u for u in (yunits, xunits) if u)
        if known:
            header += "  (" + ", ".join(
                f"{axis} in {u}" for axis, u in (("values", yunits), ("x", xunits)) if u) + ")"
        lines = [header]
        for row, label in enumerate(StatsTables._row_labels(toolfolder, window)):
            if not label:  # e.g. an epoch not analysed (one point per epoch)
                continue
            steps = epoch_result_lines(toolfolder, window, row, StatsTables.digits, xunits,
                                       x_limits, known)
            # The baseline's X range is in the header; Δ a step of its own
            steps = [re.sub(r"  \(x [^)]*\)$", "", step).replace("   \u0394 = ", "; \u0394 = ")
                     for step in steps]
            lines.append(f"{label}: " + "; ".join(steps))
        return "\n".join(lines)

    @staticmethod
    def array_details(toolfolder: NMToolFolder, window: str, name: str,
                      xunits: str = "") -> str:
        """Text describing ST_ array *name*: units, notes, summary, NaN rows.
        A histogram (H_<array>): its units and notes (source, bins, output)."""
        data = toolfolder.data.get(name)
        units = data.yscale.units
        if Stats2Panel.is_histogram(name):
            # e.g. "(probability, x in mV)" or "(density in 1/mV, x in mV)"
            output = data.yscale.label or "count"
            given = [f"{output} in {units}" if units else output]
            if data.xscale.units:
                given.append(f"x in {data.xscale.units}")
            lines = [f"{name}  ({', '.join(given)})"]
            return "\n".join(lines + [note["note"] for note in data.notes])
        # Units as in All epochs: "(values in mV, x in ms)"
        given = [f"{axis} in {u}" for axis, u in (("values", units), ("x", xunits)) if u]
        lines = [f"{name}  ({', '.join(given)})" if given else name]
        lines += [note["note"] for note in data.notes]
        lines.append("")
        summary = NMToolStats2.stats(toolfolder, select=name).get(name, {})
        for key, label in StatsTables.SUMMARY_ROWS:
            value = summary.get(key, math.nan)
            lines.append(f"{label:<6}{StatsTables._value_text(value)}")
        rows = StatsTables._row_labels(toolfolder, window)
        missing = [
            rows[index] if index < len(rows) else str(index)
            for index in np.flatnonzero(np.isnan(data.nparray))
        ]
        if missing:
            lines += ["", "NaN for: " + ", ".join(missing)]
        return "\n".join(lines)

    # --- actions ----------------------------------------------------------

    def _update_buttons(self) -> None:
        name = self.selected_array()
        self.plot_button.setEnabled(name is not None)
        self.histogram_dialog_action.setEnabled(name is not None and not self.is_histogram(name))
        self.browse_array_action.setEnabled(name is not None and self._folder_actions is not None)

    def _axis_label(self, name: str) -> str:
        units = self.toolfolder().data.get(name).yscale.units
        return f"{name} ({units})" if units else name

    def selected_plot(self) -> tuple | None:
        """The chosen array as :attr:`plot_requested` values (title, y, x,
        x label, y label, style): an ST_ array's points, or a histogram's bars
        (H_<array>, #492). None if no array is chosen."""
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None:
            return None
        data = toolfolder.data.get(name)
        if self.is_histogram(name):
            self.epoch_x_action.setEnabled(False)  # its x is the bin edges
            counts = data.nparray
            edges = toolfolder.data.get(f"{name}_edges")  # unequal bins only
            if edges is not None:
                x = edges.nparray
            else:
                x = data.xscale.start + data.xscale.delta * np.arange(len(counts) + 1)
            source = name[len("H_"):]
            xlabel = f"{source} ({data.xscale.units})" if data.xscale.units else source
            ylabel = data.yscale.label or "count"
            if data.yscale.units:
                ylabel += f" ({data.yscale.units})"
            return (f"{toolfolder.name}: {name}", counts, x, xlabel, ylabel, "histogram")
        y = data.nparray
        x, xlabel = self.epoch_x(toolfolder, len(y))
        return (f"{toolfolder.name}: {name}", y, x, xlabel, self._axis_label(name), "points")

    def epoch_x(self, toolfolder: NMToolFolder, n: int) -> tuple:
        """The Plot's x: the rows' epoch numbers (ST_epoch_num, #476) if there
        are and "x: epoch" is on, else the row index (None)."""
        epochs = toolfolder.data.get("ST_epoch_num")
        usable = epochs is not None and epochs.nparray is not None and len(epochs.nparray) == n
        self.epoch_x_action.setEnabled(usable)
        if usable and self.epoch_x_action.isChecked():
            return epochs.nparray, "epoch"
        return None, "data index"

    def plot_selected(self) -> None:
        values = self.selected_plot()
        if values is not None:
            self.plot_requested.emit(*values)

    def _ask_histogram(self, name: str, values: np.ndarray, units: str) -> dict | None:
        """The Histogram... dialog's ``nm.stats.histogram`` arguments, or
        None if cancelled (replaced in tests)."""
        dialog = HistogramDialog(name, values, units, self._histogram_settings, self)
        if not dialog.exec():
            return None
        self._histogram_settings = dialog.settings()
        return dialog.kwargs()

    def histogram_dialog(self) -> None:
        """Histogram... (⋯ menu, #492): the chosen array's histogram, saved in
        the Stats subfolder as H_<array> (one command), then chosen, so it's
        plotted."""
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None or self.is_histogram(name):
            return
        data = toolfolder.data.get(name)
        values = np.asarray(data.nparray, dtype=float)
        if not np.isfinite(values).any():
            self.message_label.setText(f"{name} has no finite values to histogram.")
            self.message_label.show()
            return
        kwargs = self._ask_histogram(name, values, data.yscale.units or "")
        if kwargs is None:
            return
        try:
            self._manager.stats.histogram(toolfolder, name, **kwargs)
        except ValueError as error:
            self.message_label.setText(f"No histogram of {name}: {error}")
            self.message_label.show()
            return
        self._refresh_array_combo()
        self.select_array(f"H_{name}")
        self.plot_selected()

    # --- the Folder ⋯ menu (#494) -------------------------------------------

    def set_folder_actions(self, actions) -> None:
        """The app's folder actions (:class:`FolderActions`), whose subfolder
        actions the Folder ⋯ menu uses."""
        self._folder_actions = actions
        self._update_folder_actions()
        self._update_buttons()

    def _update_folder_actions(self) -> None:
        usable = self._folder_actions is not None and self.toolfolder() is not None
        for action in (self.rename_folder_action, self.copy_folder_action,
                       self.delete_folder_action, self.browse_folder_action):
            action.setEnabled(usable)

    def rename_folder(self) -> None:
        """Rename… (the strip's prompt, starting on the subfolder shown); then
        show its new name."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            name = self._folder_actions.rename_subfolder(toolfolder)
            self.refresh(select=name)

    def copy_folder(self) -> None:
        """Copy… (the strip's prompt, starting on the subfolder shown); then
        show the copy."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            name = self._folder_actions.duplicate_subfolder(toolfolder)
            self.refresh(select=name)

    def delete_folder(self) -> None:
        """Delete… (the strip's prompt, starting on the subfolder shown); then
        the one shown, or else the newest."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            self._folder_actions.delete_subfolder(toolfolder)
            self.refresh()

    def show_folder_in_browser(self) -> None:
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            self._folder_actions.subfolder_in_browser(toolfolder)

    def show_array_in_browser(self) -> None:
        """Show in Browser (Plot ⋯): the chosen array (an ST_ array or a
        histogram), in its Stats subfolder."""
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None or self._folder_actions is None:
            return
        data = toolfolder.data.get(name)
        if data is not None:
            self._folder_actions.browse_requested.emit(data)


__all__ = ["Stats2Panel", "StatsTables"]
