"""Spike2: the Spike tool's saved results (#529), on the shared results panel.

Choose a Spike subfolder (Folder) and an array in it (Array): the raster
(SP_RX, every spike's time against its epoch) or a row array (SP_count,
SP_rate, SP_first, SP_isi: one row per epoch). Array ⋯ › Plot plots the
chosen array (Plot x: epoch: a row array against its epochs, else the
row index), Raster the run's raster (the default after a run, in either
results format), Table shows the row arrays in the Table panel, one row per
epoch with a summary below.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6 import QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.results_panel import ResultsPanel
from pyneuromatic.gui.results_tables import ResultsTables
from pyneuromatic.tools.nm_tool_folder import NMToolFolder

# The arrays with a row per epoch, as the Table shows them
SPIKE_ROW_ARRAYS = ("SP_count", "SP_rate", "SP_first", "SP_isi")


class SpikeTables(ResultsTables):
    """A Spike subfolder's row arrays: count, rate, first spike, mean ISI;
    a row per epoch (its data name, or its epoch number if not analysed)."""

    def _column_names(self, toolfolder: NMToolFolder, key, **_options) -> list[str]:
        return [name for name in SPIKE_ROW_ARRAYS if name in toolfolder.data]

    def _rows(self, toolfolder: NMToolFolder, key) -> list[str]:
        names = toolfolder.data.get("SP_data_names")
        numbers = toolfolder.data.get("SP_epoch_num")
        names = list(names.nparray) if names is not None and names.nparray is not None else []
        numbers = (list(numbers.nparray)
                   if numbers is not None and numbers.nparray is not None else [])
        rows = []
        for index, name in enumerate(names):
            if not name and index < len(numbers):  # not analysed (one row per epoch)
                name = f"E{int(numbers[index])}"
            rows.append(str(name))
        return rows

    def _header(self, name: str, key) -> str:
        return name[len("SP_"):]

    def _cell_value(self, name: str, value):
        # A count is a whole number (its array is float, for NaN rows)
        if name == "SP_count" and np.isfinite(value) and float(value).is_integer():
            return int(value)
        return value


class Spike2Panel(ResultsPanel):
    """Spike2: a Spike subfolder's raster and row arrays."""

    TOOLFOLDER_PREFIX = "Spike_"
    TOOL_NAME = "Spike"
    PANEL_NAME = "Spike2"
    OBJECT_PREFIX = "spike2"
    EXAMPLE_ARRAY = "SP_count"
    EPOCH_ARRAY = "SP_epoch_num"  # Plot x: epoch, as Stats2

    def __init__(self, manager: NMManager, tool: Any,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(manager, parent)
        self._tool = tool  # NMToolSpike: reads a subfolder's raster
        self.tables.column_selected.connect(self._on_table_column_selected)
        self.raster_action.triggered.connect(lambda _checked=False: self.plot_raster())
        self.refresh()

    # --- Spike's parts of the panel ------------------------------------------

    def _add_rows(self, form: QtWidgets.QFormLayout) -> None:
        # Owned here until the Table panel takes it; hidden, never drawn here
        self.tables = SpikeTables(self)
        self.tables.hide()

    def _add_array_tools(self, menu: QtWidgets.QMenu) -> None:
        # With Table: the run's results as a whole, whichever array is chosen
        self.raster_action = menu.addAction("Raster")
        self.raster_action.setToolTip(
            "Plot every spike's time against its epoch (a raster), whichever "
            "array is chosen")

    def _add_details(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox(self)  # untitled, as Stats2's
        group_layout = QtWidgets.QVBoxLayout(group)
        self.details = QtWidgets.QPlainTextEdit(group)
        self.details.setReadOnly(True)  # selectable, to copy
        self.details.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.details.setToolTip("The chosen array: its units, the run's note and a summary")
        group_layout.addWidget(self.details)
        layout.addWidget(group, stretch=1)

    def no_results_message(self) -> str:
        return (f"No saved Spike results in {self._folder.name}. "
                "Run Spike1 with Settings › Spike › 'Save SP_ arrays' on.")

    def _on_toolfolder_changed(self) -> None:
        message = self._show_message()
        self.tables.show_results(self.toolfolder(), message=message)
        self._refresh_array_combo()

    # --- the Array list --------------------------------------------------------

    def _array_names(self) -> list[str]:
        toolfolder = self.toolfolder()
        if toolfolder is None:
            return []
        names = ["SP_RX"] if "SP_RX" in toolfolder.data else []
        names += self.tables.columns
        # Histograms (H_SP_PST, H_SP_ISI...), not their bin edges
        names += [name for name in toolfolder.data.keys()
                  if name.startswith("H_SP_") and not name.endswith("_edges")]
        return names

    def default_array(self) -> str | None:
        names = self._array_names()
        for name in ("SP_RX", "SP_count"):
            if name in names:
                return name
        return names[0] if names else None

    def _on_array_chosen(self, name: str | None) -> None:
        if name in self.tables.columns and self.tables.selected_array() != name:
            self.tables.select_array(name)
        self.refresh_details()

    def _on_table_column_selected(self) -> None:
        # A column clicked in either table picks that array here too
        name = self.tables.selected_array()
        if name is not None and name != self.selected_array():
            self.select_array(name)

    def _update_array_tools(self, name: str | None) -> None:
        self.raster_action.setEnabled(self.raster_plot() is not None)

    # --- Plot and Raster ----------------------------------------------------------

    def selected_plot(self) -> tuple | None:
        """The chosen array as :attr:`plot_requested` values: SP_RX as the
        raster, a histogram's bars, or a row array against its epochs
        (SP_epoch_num; the row index with Plot x: epoch off)."""
        name, toolfolder = self.selected_array(), self.toolfolder()
        if name is None or toolfolder is None:
            return None
        if name in ("SP_RX", "SP_RY") or self.is_histogram(name):
            self.epoch_x_action.setEnabled(False)  # x: spike times, or bin edges
            if self.is_histogram(name):
                return self.histogram_plot(toolfolder, name)
            return self.raster_plot()
        y = toolfolder.data.get(name).nparray
        x, xlabel = self.epoch_x(toolfolder, len(y))
        return (f"{toolfolder.name}: {name}", y, x, xlabel, self._axis_label(name), "points")

    def _time_units(self, toolfolder: NMToolFolder) -> str:
        """The spike times' units (e.g. ms): SP_RX's, else SP_first's."""
        for name in ("SP_RX", "SP_first", "SP_isi"):
            data = toolfolder.data.get(name)
            if data is not None and data.yscale.units:
                return data.yscale.units
        return ""

    def raster_plot(self) -> tuple | None:
        """The subfolder's raster as :attr:`plot_requested` values: a tick
        at each spike, x its time, y its epoch; None without spike times."""
        toolfolder = self.toolfolder()
        if toolfolder is None or self._tool is None:
            return None
        try:
            x, y = self._tool.raster_arrays(toolfolder)
        except RuntimeError:
            return None
        units = self._time_units(toolfolder)
        xlabel = f"Time ({units})" if units else "Time"
        return (f"{toolfolder.name}: raster", y, x, xlabel, "Epoch", "raster")

    def plot_raster(self) -> None:
        """Raster (Array ⋯): the subfolder's raster, in the Plot panel; SP_RX
        chosen too, if listed (raster format), so its Results show."""
        if "SP_RX" in self._array_names() and self.selected_array() != "SP_RX":
            self.select_array("SP_RX")  # plotted as it's chosen (array_changed)
            return
        values = self.raster_plot()
        if values is not None:
            self.plot_requested.emit(*values)

    # --- the Results box ----------------------------------------------------------

    def refresh_details(self) -> None:
        """Show the chosen array's details again (e.g. with other decimals)."""
        toolfolder, name = self.toolfolder(), self.selected_array()
        text = self.array_details(toolfolder, name) if toolfolder and name else ""
        if text != self.details.toPlainText():
            self.details.setPlainText(text)

    def array_details(self, toolfolder: NMToolFolder, name: str) -> str:
        """Array *name*'s units, the run's note and a summary (mean, std, N...,
        NaNs left out), e.g. for SP_count "SP_count  (spikes per epoch)"."""
        data = toolfolder.data.get(name)
        if data is None:
            return ""
        label, units = data.yscale.label, data.yscale.units
        given = ", ".join(part for part in (label, f"in {units}" if units else "") if part)
        lines = [f"{name}  ({given})" if given else name]
        if name == "SP_RX":
            lines[0] += "  every spike's time; SP_RY its epoch"
        lines += [str(note["note"]) for note in toolfolder.notes]
        lines += [str(note["note"]) for note in data.notes]
        if self.is_histogram(name):
            return "\n".join(lines)
        lines.append("")
        summary = self.tables._summary(toolfolder, name)
        for key, row_label in SpikeTables.SUMMARY_ROWS:
            lines.append(f"{row_label:<6}{SpikeTables._value_text(summary.get(key, np.nan))}")
        return "\n".join(lines)


__all__ = ["SPIKE_ROW_ARRAYS", "Spike2Panel", "SpikeTables"]
