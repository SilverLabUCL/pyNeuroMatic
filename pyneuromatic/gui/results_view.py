"""Stats2 panel and the Stats tables it shows in the Table context panel."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.tools.nm_tool_stats import NMToolStats2


class StatsTables(QtWidgets.QWidget):
    """Two aligned tables of one Stats window's saved ST_ arrays.

    The upper table has one row per analysed data array (from
    ``ST_<win>_data``); the lower table summarises each column (mean, std,
    ...). Both tables share columns, column widths, horizontal scrolling and
    the selected column.
    """

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
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectColumns
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
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
        columns = {index.column() for index in table.selectedIndexes()}
        if len(columns) == 1:
            other.selectColumn(columns.pop())
        else:
            other.clearSelection()
        self._syncing = False
        self.column_selected.emit()

    # --- content ----------------------------------------------------------

    def show_window(
        self,
        toolfolder: NMToolFolder | None,
        window: str | None,
        all_columns: bool = False,
        message: str = "",
    ) -> None:
        """Fill both tables from *window*'s ST_ arrays in *toolfolder*."""
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
                header = QtWidgets.QTableWidgetItem(name[len(prefix):])
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
        return names

    @staticmethod
    def _row_labels(toolfolder: NMToolFolder, window: str) -> list[str]:
        data = toolfolder.data.get(f"ST_{window}_data")
        if data is None or data.nparray is None:
            return []
        # Data paths look like "nm.Folder.RecordA0"; show the data name
        return [str(path).rsplit(".", 1)[-1] for path in data.nparray]

    @staticmethod
    def _value_item(value) -> QtWidgets.QTableWidgetItem:
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value)
        else:
            text = "NaN" if math.isnan(number) else f"{number:.6g}"
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        return item

    # --- selection and export ---------------------------------------------

    def selected_array(self) -> str | None:
        """ST_ array name of the selected column, if any."""
        columns = {index.column() for index in self.arrays_table.selectedIndexes()}
        if len(columns) != 1:
            return None
        return self._columns[columns.pop()]

    def select_array(self, name: str) -> None:
        if name in self._columns:
            self.arrays_table.selectColumn(self._columns.index(name))

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

    TOOLFOLDER_PREFIX = "Stats_"

    # title, y values, x values (or None), x label, y label, style
    plot_requested = QtCore.pyqtSignal(str, object, object, str, str, str)
    # The list of Stats toolfolders in the current folder was reloaded
    results_changed = QtCore.pyqtSignal()

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
        folder_row = QtWidgets.QHBoxLayout()
        self.folder_combo = QtWidgets.QComboBox(self)
        self.folder_combo.setObjectName("stats2Folder")
        folder_row.addWidget(self.folder_combo, stretch=1)
        self.delete_button = QtWidgets.QPushButton("Delete folder", self)
        folder_row.addWidget(self.delete_button)
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
        self.copy_button = QtWidgets.QPushButton("Copy tables", self)
        self.copy_button.setToolTip("Copy both tables as tab-separated text")
        window_row.addWidget(self.copy_button)
        form.addRow("Window", window_row)

        array_row = QtWidgets.QHBoxLayout()
        self.array_combo = QtWidgets.QComboBox(self)
        self.array_combo.setObjectName("stats2Array")
        array_row.addWidget(self.array_combo, stretch=1)
        self.plot_button = QtWidgets.QPushButton("Plot", self)
        self.histogram_button = QtWidgets.QPushButton("Histogram", self)
        array_row.addWidget(self.plot_button)
        array_row.addWidget(self.histogram_button)
        form.addRow("Array", array_row)
        layout.addLayout(form)

        self.message_label = QtWidgets.QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        details_group = QtWidgets.QGroupBox("Details", self)
        details_layout = QtWidgets.QVBoxLayout(details_group)
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
        self.plot_button.clicked.connect(self.plot_selected)
        self.histogram_button.clicked.connect(self.histogram_selected)
        self.copy_button.clicked.connect(self.copy_tables)
        self.delete_button.clicked.connect(self.delete_folder)

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
        index = self.folder_combo.findData(previous)
        self.folder_combo.setCurrentIndex(index if index >= 0 else len(names) - 1)
        self.folder_combo.blockSignals(False)
        self._on_folder_changed()
        self.results_changed.emit()

    def has_results(self) -> bool:
        """True if the current folder has saved Stats results."""
        return self.folder_combo.count() > 0

    def show_toolfolder(self, toolfolder: NMToolFolder | None) -> None:
        """Show *toolfolder*, e.g. the one a Stats run just wrote."""
        self.refresh(select=toolfolder.name if toolfolder is not None else None)

    def _on_folder_changed(self, *_args) -> None:
        previous = self.window_combo.currentData()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        toolfolder = self.toolfolder()
        if toolfolder is not None:
            for name in toolfolder.data.keys():
                if name.startswith("ST_") and name.endswith("_data"):
                    window = name[len("ST_"):-len("_data")]
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

    def select_array(self, name: str) -> None:
        index = self.array_combo.findData(name)
        if index >= 0:
            self.array_combo.setCurrentIndex(index)

    def _refresh_array_combo(self) -> None:
        previous = self.array_combo.currentData()
        self.array_combo.blockSignals(True)
        self.array_combo.clear()
        for name in self.tables.columns:
            self.array_combo.addItem(name, name)
        index = self.array_combo.findData(previous)
        self.array_combo.setCurrentIndex(max(index, 0) if self.array_combo.count() else -1)
        self.array_combo.blockSignals(False)
        self._on_array_changed()

    def _on_array_changed(self, *_args) -> None:
        name = self.selected_array()
        if name is not None and self.tables.selected_array() != name:
            self.tables.select_array(name)
        self._update_details()
        self._update_buttons()

    def _on_table_column_selected(self) -> None:
        # Clicking a column in either table picks that array here too
        name = self.tables.selected_array()
        if name is not None and name != self.selected_array():
            self.select_array(name)

    def _update_details(self) -> None:
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None:
            self.details.clear()
            return
        self.details.setPlainText(
            self.array_details(toolfolder, self.window_combo.currentData(), name)
        )

    @staticmethod
    def array_details(toolfolder: NMToolFolder, window: str, name: str) -> str:
        """Text describing ST_ array *name*: units, notes, summary, NaN rows."""
        data = toolfolder.data.get(name)
        units = data.yscale.units
        lines = [f"{name} ({units})" if units else name]
        lines += [note["note"] for note in data.notes]
        lines.append("")
        summary = NMToolStats2.stats(toolfolder, select=name).get(name, {})
        for key, label in StatsTables.SUMMARY_ROWS:
            value = summary.get(key, math.nan)
            lines.append(f"{label:<6}{StatsTables._value_item(value).text()}")
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
        self.histogram_button.setEnabled(name is not None)
        self.copy_button.setEnabled(bool(self.tables.columns))
        self.delete_button.setEnabled(self.toolfolder() is not None)

    def _axis_label(self, name: str) -> str:
        units = self.toolfolder().data.get(name).yscale.units
        return f"{name} ({units})" if units else name

    def plot_selected(self) -> None:
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None:
            return
        self.plot_requested.emit(
            f"{toolfolder.name}: {name}",
            toolfolder.data.get(name).nparray,
            None,
            "data index",
            self._axis_label(name),
            "points",
        )

    def histogram_selected(self) -> None:
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None:
            return
        values = toolfolder.data.get(name).nparray
        finite = int(np.count_nonzero(np.isfinite(values)))
        if finite == 0:
            self.message_label.setText(f"{name} has no finite values to histogram.")
            self.message_label.show()
            return
        # Square-root rule, capped so small result sets stay readable
        bins = max(1, min(20, math.ceil(math.sqrt(finite))))
        result = NMToolStats2.histogram(
            toolfolder, name, bins=bins, save_to_numpy=False
        )
        self.plot_requested.emit(
            f"{toolfolder.name}: {name} histogram",
            result["counts"],
            result["edges"],
            self._axis_label(name),
            "count",
            "histogram",
        )

    def copy_tables(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.tables.table_text())

    def _confirm_delete(self, name: str) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Stats folder",
            f"Delete {name} and all its Stats arrays?",
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def delete_folder(self) -> None:
        toolfolder = self.toolfolder()
        if toolfolder is None or self._folder is None:
            return
        if not self._confirm_delete(toolfolder.name):
            return
        self._folder.toolfolders.pop(toolfolder.name)
        self.refresh()


__all__ = ["Stats2Panel", "StatsTables"]
