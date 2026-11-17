"""Stats2 panel and the Stats tables it shows in the Table context panel."""
from __future__ import annotations

import math
import re

import numpy as np
from PyQt6 import QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.histogram_dialog import HistogramDialog
from pyneuromatic.gui.inequality_dialog import InequalityDialog
from pyneuromatic.gui.results_panel import _NOTE_PAIR, ResultsPanel, note_fields, runs_text
from pyneuromatic.gui.results_tables import ResultsTables
from pyneuromatic.gui import table_format
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.tools.nm_stat_func import FUNC_NAMES_BSLN, FUNC_NAMES_N
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.tools.nm_tool_stats import NMToolStats, NMToolStats2


def display_name(name: str) -> str:
    """A measurement as Stats shows it: mean+std, mean+var and mean+sem as
    "mean ± std"... (a mean and its spread, not a sum). The core name, in
    commands and scripts, is unchanged."""
    if name and name.startswith("mean+"):
        return "mean \u00b1 " + name[len("mean+"):]
    return name


def points_text(n, nans=0) -> str:
    """The number of points a measurement used (#498), e.g. "n = 15", and the
    NaNs left out if any: "n = 15, 2 NaNs"."""
    try:
        text = f"n = {int(n)}"
    except (TypeError, ValueError):
        return "n = NaN"
    try:
        nans = int(nans)
    except (TypeError, ValueError):
        nans = 0
    if nans > 0:
        text += f", {nans} NaN" + ("s" if nans > 1 else "")
    return text


def shows_points(func: str) -> bool:
    """True if Stats shows the number of points of measurement *func* (#498):
    one computed from many points (a mean, std, slope...), not a point found."""
    return func in FUNC_NAMES_N or func in FUNC_NAMES_BSLN


def _note_fields(data) -> dict[str, str]:
    """An ST_ array's note as fields, e.g. "NMStats(win=w0, func=max,
    id=main, xbgn=60.0, xend=120.0, n=2)" -> {"func": "max", "id": "main", ...}."""
    return note_fields(data, "NMStats")


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
                       decimals: int = 3, xunits: str = "",
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
        if not math.isfinite(value):
            return "NaN" if math.isnan(value) else ("-inf" if value < 0 else "inf")
        return table_format.fixed(value, decimals)

    def value(data) -> str:
        values = data.nparray
        text = number(values[row]) if row < len(values) else "NaN"
        units = data.yscale.units
        return f"{text} {units}".strip() if units and units not in known_units else text

    def at_row(data):
        values = data.nparray if data is not None else None
        return values[row] if values is not None and row < len(values) else None

    prefix = f"ST_{window}_"
    groups: dict[tuple[str, str], list[tuple[str, object]]] = {}
    fields: dict[tuple[str, str], dict[str, str]] = {}
    counts: dict[tuple[str, str], dict[str, object]] = {}  # its n and nans arrays
    for name, data in toolfolder.data.items():
        if not name.startswith(prefix) or name == f"{prefix}data":
            continue
        suffix = name.rsplit("_", 1)[-1]
        if not isinstance(data.nparray, np.ndarray):
            continue
        note = _note_fields(data)
        key = (note.get("id", ""), note.get("step", ""))
        if suffix in ("n", "nans"):
            counts.setdefault(key, {})[suffix] = data
        if suffix in StatsTables._DETAIL_SUFFIXES:
            continue
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
        n = counts.get(key, {})
        if "n" in n and shows_points(func if step_id in ("main", "bsln") else stepfunc or ""):
            line += f" ({points_text(at_row(n['n']), at_row(n.get('nans')))})"
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


class StatsTables(ResultsTables):
    """Two aligned tables of one Stats window's saved ST_ arrays.

    The shared results tables (#525): the upper table has one row per
    analysed data array (from ``ST_data_names``); the lower table
    summarises each column (mean, std, ...). Stats' parts: the window's
    ST_ and IQ_ arrays as columns, its detail arrays hidden unless asked.
    """

    # Bookkeeping arrays hidden unless all columns are requested
    _DETAIL_SUFFIXES = frozenset(("i", "i0", "i1", "n", "nans", "infs"))

    def show_window(
        self,
        toolfolder: NMToolFolder | None,
        window: str | None,
        detail_arrays: bool = False,
        message: str = "",
    ) -> None:
        """Fill both tables from *window*'s ST_ arrays in *toolfolder*."""
        self.show_results(toolfolder if window is not None else None, window,
                          message, detail_arrays=detail_arrays)

    def _column_names(self, toolfolder: NMToolFolder, key, detail_arrays: bool = False,
                      **_options) -> list[str]:
        return self._window_arrays(toolfolder, key, detail_arrays)

    def _rows(self, toolfolder: NMToolFolder, key) -> list[str]:
        return self._row_labels(toolfolder, key)

    def _header(self, name: str, key) -> str:
        # The window's arrays by their own part (max_y); the rows' epoch
        # numbers (ST_epoch_num, no window prefix) as "epoch"
        prefix = f"ST_{key}_"
        if name.startswith(prefix):
            return name[len(prefix):]
        if name == "ST_epoch_num":
            return "epoch"
        if name.startswith(f"IQ_{prefix}"):
            return "IQ_" + name[len("IQ_") + len(prefix):]
        return name

    @classmethod
    def _window_arrays(
        cls, toolfolder: NMToolFolder, window: str, detail_arrays: bool
    ) -> list[str]:
        prefix = f"ST_{window}_"
        names = []
        for name, data in toolfolder.data.items():
            if not name.startswith(prefix) or name == f"{prefix}data":
                continue
            values = data.nparray
            if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
                continue  # e.g. warning text arrays
            if name.rsplit("_", 1)[-1] in cls._DETAIL_SUFFIXES and not detail_arrays:
                continue
            names.append(name)
        if detail_arrays and "ST_epoch_num" in toolfolder.data:  # the rows' epochs (#476)
            names.insert(0, "ST_epoch_num")
        # Inequalities of the window's arrays (IQ_ST_<window>_..., #499): a row
        # per epoch, as the ST_ arrays
        names += [name for name in toolfolder.data.keys() if name.startswith(f"IQ_{prefix}")]
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


class Stats2Panel(ResultsPanel):
    """Stats2 controls: choose saved Stats results and act on their columns.

    The shared results panel (#525) plus Stats' own parts: the Window list,
    Plot x: epoch, Histogram... and Inequality..., and the Results box. The
    tables themselves are in :attr:`tables`, shown in the context panel's
    Table tab.
    """

    TOOLFOLDER_PREFIX = NMToolStats.TOOLFOLDER_PREFIX  # "Stats_" (#494)
    TOOL_NAME = "Stats"
    PANEL_NAME = "Stats2"
    OBJECT_PREFIX = "stats2"
    EXAMPLE_ARRAY = "ST_w0_max_y"
    EPOCH_ARRAY = "ST_epoch_num"  # Plot x: epoch (#476)

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(manager, parent)
        self.window_combo.currentIndexChanged.connect(self._show_tables)
        self.tables.column_selected.connect(self._on_table_column_selected)
        self.results_combo.currentIndexChanged.connect(self._update_details)
        self.histogram_dialog_action.triggered.connect(self.histogram_dialog)
        self._histogram_settings: dict = {}  # the dialog's, shown again next time
        self.inequality_dialog_action.triggered.connect(self.inequality_dialog)
        self._inequality_settings: dict = {}
        # The window last chosen in each subfolder, as its array (ResultsPanel)
        self._chosen_windows: dict[tuple[str, str], str] = {}

        self.refresh()

    # --- Stats' parts of the panel ----------------------------------------

    def _add_rows(self, form: QtWidgets.QFormLayout) -> None:
        # Owned here until a TableTab takes it (addWidget reparents it); hidden
        # so it is never drawn inside this panel
        self.tables = StatsTables(self)
        self.tables.hide()
        window_row = QtWidgets.QHBoxLayout()
        self.window_combo = QtWidgets.QComboBox(self)
        self.window_combo.setObjectName("stats2Window")
        self.window_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        window_row.addWidget(self.window_combo)
        window_row.addStretch(1)
        # Settings › Stats: also show the index and count arrays (#496)
        self._detail_arrays = False
        form.addRow("Window", window_row)

    def _add_array_tools(self, menu: QtWidgets.QMenu) -> None:
        # Saved as H_<array>, listed with the ST_ arrays to plot (#492)
        self.histogram_dialog_action = menu.addAction("Histogram...")
        self.histogram_dialog_action.setToolTip(
            "Make a histogram of the chosen array, saved in the Stats subfolder "
            "as H_<array> and listed with its arrays")
        # Saved as IQ_<array>, a column of the Stats tables (#499)
        self.inequality_dialog_action = menu.addAction("Inequality...")
        self.inequality_dialog_action.setToolTip(
            "Test the chosen array's values against a condition (e.g. y > 10): "
            "1 or 0 per epoch, saved as IQ_<array>; optionally Epoch Sets of the "
            "epochs that pass and fail")

    def _add_details(self, layout: QtWidgets.QVBoxLayout) -> None:
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
            "result of every epoch.")
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

    def no_results_message(self) -> str:
        return (
            f"No saved Stats results in {self._folder.name}. "
            "Run Stats1 with Settings › Stats › 'Save ST_ arrays' on."
        )

    # --- the Window list ----------------------------------------------------

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

    def _array_key(self) -> tuple | None:
        # Each window's array: its arrays are the window's
        key = self._toolfolder_key()
        return key + (self.window_combo.currentData(),) if key else None

    def _on_toolfolder_changed(self) -> None:
        previous = self.window_combo.currentData()
        key = self._toolfolder_key()
        remembered = self._chosen_windows.get(key) if key else None
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        toolfolder = self.toolfolder()
        if toolfolder is not None:
            for window in self._windows(toolfolder):
                self.window_combo.addItem(window, window)
        # The subfolder's own window, else the one before, else the first
        index = self.window_combo.findData(remembered) if remembered else -1
        if index < 0:
            index = self.window_combo.findData(previous)
        self.window_combo.setCurrentIndex(max(index, 0))
        self.window_combo.blockSignals(False)
        self._show_tables()

    def _show_tables(self, *_args) -> None:
        key, window = self._toolfolder_key(), self.window_combo.currentData()
        if key is not None and window is not None:
            self._chosen_windows[key] = window
        message = self._show_message()
        self.tables.show_window(
            self.toolfolder(),
            self.window_combo.currentData(),
            detail_arrays=self._detail_arrays,
            message=message,
        )
        self._refresh_array_combo()

    def set_detail_arrays(self, on: bool) -> None:
        """Also show the index and point-count arrays (i, i0, i1, n, nans,
        infs) and the epoch numbers, in the tables and the Array list
        (Settings › Stats, #496); the array chosen stays if still listed."""
        if bool(on) != self._detail_arrays:
            self._detail_arrays = bool(on)
            self._show_tables()

    # --- the Array list -----------------------------------------------------

    def _array_names(self) -> list[str]:
        return self.tables.columns + self._histograms()

    def default_array(self) -> str | None:
        return self.measure_array()

    def measure_array(self) -> str | None:
        """The window's measurement among its arrays: a rise time, decay time
        or FWHM's result (``_dx``), a level's crossing time (``_x``, e.g.
        ST_w0_level_p_x), else the first that isn't the baseline, e.g.
        ST_w0_max_y rather than ST_w0_bsln_y."""
        names = [n for n in self.tables.columns
                 if n.startswith("ST_") and n != "ST_epoch_num"]  # not IQ_ (#499)
        for name in names:
            if name.endswith("_dx"):
                return name
        for name in names:  # level, level+ (level_p), level- (level_m)
            if re.fullmatch(r"ST_\w+?_level(_[pm])?_x", name):
                return name
        for name in names:
            if "_bsln_" not in name:
                return name
        return names[0] if names else None

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

    def _on_array_chosen(self, name: str | None) -> None:
        if name is not None and self.tables.selected_array() != name:
            self.tables.select_array(name)
        self._update_details()

    def _on_table_column_selected(self) -> None:
        # Clicking a column in either table picks that array here too
        name = self.tables.selected_array()
        if name is not None and name != self.selected_array():
            self.select_array(name)

    def _update_array_tools(self, name: str | None) -> None:
        self.histogram_dialog_action.setEnabled(name is not None and not self.is_histogram(name))
        self.inequality_dialog_action.setEnabled(
            name is not None and not self.is_histogram(name) and not name.startswith("IQ_"))

    def refresh_details(self) -> None:
        """Show the Results again (e.g. with other decimals)."""
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
            steps = epoch_result_lines(toolfolder, window, row, StatsTables.decimals, xunits,
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
        if name.startswith("IQ_"):  # its condition and counts (#499)
            lines.append(Stats2Panel.inequality_text(data))
        lines += [note["note"] for note in data.notes]
        lines.append("")
        summary = NMToolStats2.stats(toolfolder, select=name).get(name, {})
        for key, label in StatsTables.SUMMARY_ROWS:
            value = summary.get(key, math.nan)
            lines.append(f"{label:<6}{StatsTables._value_text(value)}")
        if name.startswith("IQ_"):  # its NaNs: in the counts above (#499)
            return "\n".join(lines)
        rows = StatsTables._row_labels(toolfolder, window)
        missing = [
            rows[index] if index < len(rows) else str(index)
            for index in np.flatnonzero(np.isnan(data.nparray))
        ]
        if missing:
            lines += ["", "NaN for: " + runs_text(missing)]  # e.g. RecordA0-3
        return "\n".join(lines)

    # --- Plot, Histogram... and Inequality... ----------------------------------

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
            return self.histogram_plot(toolfolder, name)
        y = data.nparray
        x, xlabel = self.epoch_x(toolfolder, len(y))
        return (f"{toolfolder.name}: {name}", y, x, xlabel, self._axis_label(name), "points")

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

    # --- Inequality... (#499) --------------------------------------------------

    @staticmethod
    def inequality_text(data) -> str:
        """An IQ_ array's condition and counts, from its note, e.g.
        "ST_w0_max_dy > 10: 12 pass, 3 fail (of 15)"."""
        for note in data.notes:
            text = str(note["note"])
            if not text.startswith("NMInequality("):
                continue
            fields = dict(_NOTE_PAIR.findall(text))
            array = fields.get("array", "").strip()
            condition = re.sub(r"\by\b", array, fields.get("condition", "").strip())
            counts = {k: int(fields.get(k, 0)) for k in ("pass", "fail", "nan")}
            line = f"{condition}: {counts['pass']} pass, {counts['fail']} fail"
            if counts["nan"]:
                line += f", {counts['nan']} NaN"
            return line + f" (of {sum(counts.values())})"
        return ""

    def _series_of(self, toolfolder: NMToolFolder):
        """The data series whose epochs *toolfolder*'s rows are (for Epoch
        Sets): from its rows' data names and epoch numbers; None if it has
        none (e.g. a Data Set run)."""
        if self._folder is None or "ST_epoch_num" not in toolfolder.data:
            return None
        names = toolfolder.data.get("ST_data_names")
        for path in (names.nparray if names is not None else []):
            parsed = nmu.parse_data_name(str(path).rsplit(".", 1)[-1]) if path else None
            if parsed is not None:
                return self._folder.dataseries.get(parsed[0])
        return None

    def _ask_inequality(self, name: str, set_names) -> dict | None:
        """The Inequality... dialog's ``nm.stats.inequality`` arguments, or
        None if cancelled (replaced in tests)."""
        dialog = InequalityDialog(name, set_names, self._inequality_settings, self)
        if not dialog.exec():
            return None
        self._inequality_settings = dialog.settings()
        return dialog.kwargs()

    def _confirm_replace_sets(self, names: list[str], series) -> bool:
        """Ask before replacing Epoch Sets *names* (replaced in tests)."""
        answer = QtWidgets.QMessageBox.question(
            self, "Replace Epoch Set",
            f"Replace Epoch Set{'s' * (len(names) > 1)} {', '.join(names)} "
            f"of {series.name}?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def inequality_dialog(self) -> None:
        """Inequality... (⋯ menu, #499): the chosen array tested against a
        condition, saved in the Stats subfolder as IQ_<array> (one command),
        with Epoch Sets of the passes and fails if chosen; then chosen, so
        it's plotted."""
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if (name is None or toolfolder is None or self.is_histogram(name)
                or name.startswith("IQ_")):
            return
        series = self._series_of(toolfolder)
        set_names = None
        if series is not None:  # the next free: Set0, Set1...
            sets = series.epochs.sets
            first = sets.auto_name_next()
            number = int(first[len("Set"):]) + 1
            while f"Set{number}" in sets:
                number += 1
            set_names = (first, f"Set{number}")
        kwargs = self._ask_inequality(name, set_names)
        if kwargs is None:
            return
        chosen = [kwargs[k] for k in ("set_name_success", "set_name_failure") if k in kwargs]
        if chosen:
            existing = [n for n in chosen if n in series.epochs.sets]
            if existing and not self._confirm_replace_sets(existing, series):
                return
            kwargs["dataseries"] = series
        try:
            self._manager.stats.inequality(toolfolder, name, **kwargs)
        except (KeyError, ValueError) as error:
            self.message_label.setText(f"No inequality of {name}: {error}")
            self.message_label.show()
            return
        if chosen and self._folder_actions is not None:
            self._folder_actions.content_changed.emit()  # e.g. the strip's Epoch Sets
        self._show_tables()
        self.select_array(f"IQ_{name}")
        self.plot_selected()


__all__ = ["Stats2Panel", "StatsTables"]
