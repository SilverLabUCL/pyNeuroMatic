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
from pyneuromatic.gui.spike2_dialogs import ISIDialog, PSTDialog, WaveformsDialog
from pyneuromatic.gui.results_tables import ResultsTables
from pyneuromatic.tools.nm_tool_folder import NMToolFolder

# The arrays with a row per epoch, as the Table shows them
SPIKE_ROW_ARRAYS = ("SP_count", "SP_rate", "SP_first", "SP_isi")

# The spike waveforms' data series (Waveforms..., #530), listed as one array
WAVEFORMS = "SPK_"


# The Spike2 tables (*key*): the row arrays, the raster or the waveforms
TABLE_ROWS, TABLE_RASTER, TABLE_WAVEFORMS = "rows", "raster", "waveforms"


def row_arrays(toolfolder: NMToolFolder) -> list[str]:
    """The subfolder's arrays with a row per epoch (count, rate...)."""
    return [name for name in SPIKE_ROW_ARRAYS if name in toolfolder.data]


def waveform_arrays(toolfolder: NMToolFolder) -> list[str]:
    """The subfolder's waveforms (SPK_ arrays), in order."""
    return [name for name in toolfolder.data.keys() if name.startswith(WAVEFORMS)]


class SpikeTables(ResultsTables):
    """A Spike subfolder's arrays, as *key* says (#530):

    - ``"rows"`` (default): the row arrays, count, rate, first spike, mean
      ISI; a row per epoch (its data name, or its epoch number if not
      analysed);
    - ``"raster"``: SP_RX and SP_RY, a row per spike (numbered 0, 1...);
    - ``"waveforms"``: the SPK_ arrays, a column each, a row per point
      (named by its time, when the waveforms share them).
    """

    def _column_names(self, toolfolder: NMToolFolder, key, **_options) -> list[str]:
        if key == TABLE_RASTER:
            return [name for name in ("SP_RX", "SP_RY") if name in toolfolder.data]
        if key == TABLE_WAVEFORMS:
            return waveform_arrays(toolfolder)
        return row_arrays(toolfolder)

    def _title(self, toolfolder: NMToolFolder, key) -> str:
        if key == TABLE_RASTER:
            return f"{toolfolder.name}  raster (a row per spike)"
        if key == TABLE_WAVEFORMS:
            return f"{toolfolder.name}  {WAVEFORMS} waveforms (a column each)"
        return toolfolder.name

    def _rows(self, toolfolder: NMToolFolder, key) -> list[str]:
        if key == TABLE_RASTER:
            return []  # numbered
        if key == TABLE_WAVEFORMS:
            return self._waveform_times(toolfolder)
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

    @classmethod
    def _waveform_times(cls, toolfolder: NMToolFolder) -> list[str]:
        """The waveforms' rows, named by their points' times if they all
        share them (e.g. "-1.000"), else [] (numbered)."""
        names = waveform_arrays(toolfolder)
        if not names:
            return []
        first = toolfolder.data.get(names[0])
        if first.xarray is not None or first.nparray is None:
            return []
        n, start, delta = len(first.nparray), first.xscale.start, first.xscale.delta
        for name in names[1:]:
            data = toolfolder.data.get(name)
            if (data.xarray is not None or data.nparray is None or len(data.nparray) != n
                    or data.xscale.start != start or data.xscale.delta != delta):
                return []
        return [cls._value_text(start + i * delta) for i in range(n)]

    def _header(self, name: str, key) -> str:
        if name.startswith(WAVEFORMS):
            return name[len(WAVEFORMS):]  # SPK_A0: A0
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
        # The analyses' dialogs show their last settings again (#530)
        self._analysis_settings: dict[str, dict] = {}
        for kind, action in self.analysis_actions.items():
            action.triggered.connect(lambda _checked=False, k=kind: self.analysis_dialog(k))
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
        # Analyses of the spike times (#530), saved in the Spike subfolder
        # and listed with its arrays
        self.analysis_actions = {
            "pst": menu.addAction("PST..."),
            "isi": menu.addAction("ISI..."),
            "waveforms": menu.addAction("Waveforms..."),
        }
        self.analysis_actions["pst"].setToolTip(
            "A peri-stimulus time histogram of the spike times, pooled over the "
            "epochs; saved as H_SP_PST")
        self.analysis_actions["isi"].setToolTip(
            "A histogram of the interspike intervals; saved as H_SP_ISI")
        self.analysis_actions["waveforms"].setToolTip(
            "Each spike's waveform, from its recording; saved as SPK_ arrays "
            "(the SPK_ data series) and plotted on top of each other")

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
        self._table_shown = None  # filled again for the array chosen
        message = self._show_message()
        if self.toolfolder() is None:
            self.tables.show_results(None, message=message)
        self._refresh_array_combo()

    @staticmethod
    def table_key(name: str | None) -> str:
        """The table for array *name*: the raster's, the waveforms' or the
        row arrays' (#530)."""
        if name in ("SP_RX", "SP_RY"):
            return TABLE_RASTER
        if name == WAVEFORMS:
            return TABLE_WAVEFORMS
        return TABLE_ROWS

    def _show_table(self, name: str | None) -> None:
        """The tables for array *name*, filled only if they show another
        (a thousand waveforms take a moment)."""
        toolfolder = self.toolfolder()
        if toolfolder is None:
            return
        key = self.table_key(name)
        if getattr(self, "_table_shown", None) != (toolfolder.name, key):
            # Its column as filled isn't a choice (the array chosen stays)
            self._filling_table = True
            try:
                self.tables.show_results(toolfolder, key, message=self._message())
            finally:
                self._filling_table = False
            self._table_shown = (toolfolder.name, key)

    # --- the Array list --------------------------------------------------------

    def _array_names(self) -> list[str]:
        toolfolder = self.toolfolder()
        if toolfolder is None:
            return []
        names = ["SP_RX"] if "SP_RX" in toolfolder.data else []
        names += row_arrays(toolfolder)  # whichever table is shown
        # Histograms (H_SP_PST, H_SP_ISI...), not their bin edges
        names += [name for name in toolfolder.data.keys()
                  if name.startswith("H_SP_") and not name.endswith("_edges")]
        if WAVEFORMS in toolfolder.dataseries:  # the waveforms, as one
            names.append(WAVEFORMS)
        return names

    def default_array(self) -> str | None:
        names = self._array_names()
        for name in ("SP_RX", "SP_count"):
            if name in names:
                return name
        return names[0] if names else None

    def _on_array_chosen(self, name: str | None) -> None:
        self._show_table(name)
        if name in self.tables.columns and self.tables.selected_array() != name:
            self.tables.select_array(name)
        self.refresh_details()

    def _on_table_column_selected(self) -> None:
        # A column clicked in either table picks that array here too
        if getattr(self, "_filling_table", False):
            return
        name = self.tables.selected_array()
        # A waveform's column (SPK_A3) isn't in the Array list: SPK_ stays
        if (name is not None and name != self.selected_array()
                and name in self._array_names()):
            self.select_array(name)

    def _update_array_tools(self, name: str | None) -> None:
        has_spikes = self.raster_plot() is not None
        self.raster_action.setEnabled(has_spikes)
        for action in self.analysis_actions.values():
            action.setEnabled(has_spikes)

    # --- Plot and Raster ----------------------------------------------------------

    def selected_plot(self) -> tuple | None:
        """The chosen array as :attr:`plot_requested` values: SP_RX as the
        raster, a histogram's bars, or a row array against its epochs
        (SP_epoch_num; the row index with Plot x: epoch off)."""
        name, toolfolder = self.selected_array(), self.toolfolder()
        if name is None or toolfolder is None:
            return None
        if name in ("SP_RX", "SP_RY", WAVEFORMS) or self.is_histogram(name):
            # x: spike times, bin edges or the waveforms' times
            self.epoch_x_action.setEnabled(False)
            if self.is_histogram(name):
                return self.histogram_plot(toolfolder, name)
            if name == WAVEFORMS:
                return self.waveforms_plot(toolfolder)
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

    @staticmethod
    def _waveforms(toolfolder: NMToolFolder) -> list:
        """The subfolder's waveforms: its SPK_ arrays, in order."""
        return [toolfolder.data.get(name) for name in toolfolder.data.keys()
                if name.startswith(WAVEFORMS)]

    def waveforms_plot(self, toolfolder: NMToolFolder) -> tuple | None:
        """The waveforms (SPK_, #530) as :attr:`plot_requested` values: y and
        x lists, a waveform each, plotted on top of each other."""
        waveforms = self._waveforms(toolfolder)
        if not waveforms:
            return None
        ys, xs = [], []
        for data in waveforms:
            y = np.asarray(data.nparray, dtype=float)
            if data.xarray is not None:
                x = np.asarray(data.xarray, dtype=float)
            else:
                x = data.xscale.start + data.xscale.delta * np.arange(len(y))
            ys.append(y)
            xs.append(x)
        first = waveforms[0]
        xunits, yunits = first.xscale.units, first.yscale.units
        xlabel = f"Time ({xunits})" if xunits else "Time"
        ylabel = first.yscale.label or ""
        if yunits:
            ylabel = f"{ylabel} ({yunits})".strip()
        return (f"{toolfolder.name}: {WAVEFORMS} ({len(ys)} waveforms)",
                ys, xs, xlabel, ylabel, "waveforms")

    # --- PST..., ISI... and Waveforms... (#530) ---------------------------------

    ANALYSIS_DIALOGS = {"pst": PSTDialog, "isi": ISIDialog, "waveforms": WaveformsDialog}
    ANALYSIS_NAMES = {"pst": "PST", "isi": "ISI", "waveforms": "waveforms"}

    def _ask_analysis(self, kind: str, toolfolder: NMToolFolder, times: np.ndarray,
                      units: str) -> dict | None:
        """The analysis's arguments from its dialog, or None if cancelled
        (replaced in tests)."""
        previous = self._analysis_settings.get(kind)
        if kind == "isi":
            intervals, _ = self._tool.intervals(toolfolder=toolfolder)
            valid = [np.asarray(iv, dtype=float) for iv in intervals if len(iv)]
            isis = np.concatenate(valid) if valid else np.array([])
            dialog = ISIDialog(toolfolder.name, times, isis, units, previous, self)
        else:
            dialog = self.ANALYSIS_DIALOGS[kind](toolfolder.name, times, units, previous, self)
        if not dialog.exec():
            return None
        self._analysis_settings[kind] = dialog.settings()
        return dialog.kwargs()

    def analysis_dialog(self, kind: str) -> None:
        """PST..., ISI... or Waveforms... (Array ⋯, #530): the analysis of
        the subfolder's spike times, saved in it (one command), then chosen,
        so it's plotted."""
        toolfolder = self.toolfolder()
        if toolfolder is None or self._tool is None:
            return
        what = self.ANALYSIS_NAMES[kind]
        try:
            times, _epochs = self._tool.raster_arrays(toolfolder)
        except RuntimeError as error:
            self._say(f"No {what}: {error}")
            return
        times = np.asarray(times, dtype=float)
        times = times[np.isfinite(times)]
        kwargs = self._ask_analysis(kind, toolfolder, times, self._time_units(toolfolder))
        if kwargs is None:
            return
        try:
            if kind == "pst":
                result = self._tool.pst(**kwargs, toolfolder=toolfolder)
            elif kind == "isi":
                result = self._tool.isi(**kwargs, toolfolder=toolfolder)
            else:
                result = self._tool.extract_spike_waveforms(**kwargs, toolfolder=toolfolder)
        except (RuntimeError, ValueError, TypeError, OverflowError) as error:
            self._say(f"No {what}: {error}")
            return
        if not result:  # None, or no waveforms
            reason = {"pst": "no spikes", "isi": "fewer than two spikes in an epoch",
                      "waveforms": "no spike's window fits its recording"}[kind]
            self._say(f"No {what} of {toolfolder.name}: {reason}.")
            return
        self._show_message()  # an earlier "No ..." gone
        self._table_shown = None  # e.g. the waveforms' table: new ones
        name = WAVEFORMS if kind == "waveforms" else result.name
        self._refresh_array_combo()
        self.select_array(name)
        self.plot_selected()

    def _say(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.show()

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
        if name == WAVEFORMS:  # a data series, not an array
            waveforms = self._waveforms(toolfolder)
            names = [data.name for data in waveforms]
            span = f", {names[0]} to {names[-1]}" if len(names) > 1 else ""
            lines = [f"{WAVEFORMS}  {len(names)} spike waveforms{span} "
                     f"(the {WAVEFORMS} data series)"]
            return "\n".join(lines + [str(note["note"]) for note in toolfolder.notes])
        data = toolfolder.data.get(name)
        if data is None:
            return ""
        label, units = data.yscale.label, data.yscale.units
        given = ", ".join(part for part in (label, f"{units}" if units else "") if part)
        lines = [f"{name}  ({given})" if given else name]
        if name == "SP_RX":
            lines[0] += "  spike times; SP_RY corresponding epochs"
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
