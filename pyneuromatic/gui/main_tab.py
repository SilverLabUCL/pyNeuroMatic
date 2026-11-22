"""The Main tool's tab (#550, #553): Main1, to plot and table the strip's
selection (Display) and run functions on it (Functions); Main2, to view the
functions' outputs, saved in Main subfolders.

Main2's Plot draws every selected epoch: each channel in a plot of its own,
one above another, the epochs overlaid. Its Table lists their values, an x
column, then a column per array. Both are drawn in the Main2 tab of the Plot
and Table panels, as they were when Plot or Table was clicked: the title says
what they show, and clicking again draws the selection then.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.context_views import ChannelTableModel, ChannelTable
from pyneuromatic.gui.plot_ranges import PlotRange
from pyneuromatic.gui.results_panel import ResultsPanel
from pyneuromatic.gui.selection_model import epoch_scope_label, scoped_epoch_names
from pyneuromatic.gui.tool_run import SelectionRunTab
from pyneuromatic.gui.tool_tab import fit_number_edit
from pyneuromatic.tools.nm_main_op import (
    NMMainOpAverage, NMMainOpConcatenate, NMMainOpMax, NMMainOpMin, NMMainOpSum,
    NMMainOpSumSqr)

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None


@dataclass
class ChannelArrays:
    """One plot's arrays: a channel's selected epochs (or the data arrays)."""

    label: str  # e.g. "A", or "Data"
    arrays: list[NMData]
    channel: NMChannel | None = None  # its filters and transforms, if any
    background: list[NMData] = field(default_factory=list)  # drawn behind, light
    # Each array's error (e.g. an average's Stdv or SEM), drawn as a band
    # from array - error to array + error; None: none
    errors: list = field(default_factory=list)


@dataclass
class Selected:
    """What Main2 shows: the arrays per channel, and a title saying what."""

    title: str
    groups: list[ChannelArrays] = field(default_factory=list)

    @property
    def arrays(self) -> list[NMData]:
        return [d for group in self.groups for d in group.arrays]


def selected_arrays(selection: dict[str, Any]) -> Selected:
    """The arrays of the strip's selection, as Main2 shows them.

    A data series: its channel (or every channel, with All Channels), the
    epochs in the Epoch Set/Group (excluded ones too, as the Plot shows
    them), or the current epoch when neither is chosen. A data array: the
    Data Set's arrays, or the array. Raises RuntimeError if there's nothing.
    """
    source = selection.get("source")
    series = selection.get("dataseries")
    if source == "dataseries" and isinstance(series, NMDataSeries):
        if selection.get("all_channels"):
            channels = list(series.channels.values())
        elif isinstance(selection.get("channel"), NMChannel):
            channels = [selection["channel"]]
        else:
            raise RuntimeError("Choose a channel")
        names = scoped_epoch_names(series, selection, exclude=False)
        scope = epoch_scope_label(selection)
        if names is None:  # no Epoch Set or Group: the current epoch
            epoch = selection.get("epoch")
            if epoch is None:
                raise RuntimeError("Choose an epoch, or an Epoch Set")
            names = [epoch.name]
            scope = epoch.name
        elif not scope:
            scope = "All Epochs"
        groups = []
        for channel in channels:
            found = series.epoch_data_map(channel.name)
            arrays = [found[name] for name in names if isinstance(found.get(name), NMData)]
            groups.append(ChannelArrays(channel.name, arrays, channel))
        count = max((len(g.arrays) for g in groups), default=0)
        noun = "epoch" if count == 1 else "epochs"
        title = (f"{series.name} · {', '.join(g.label for g in groups)} · {scope}"
                 f" · {count} {noun}")
        if not any(g.arrays for g in groups):
            raise RuntimeError(f"{series.name}: no arrays in {scope}")
        return Selected(title, groups)
    data = selection.get("data")
    if source == "data" and isinstance(data, NMData):
        context = selection.get("toolfolder") or selection.get("folder")
        data_set = selection.get("data_set")
        names = (context.data.sets.get_items(data_set, get_keys=True) or []
                 if data_set is not None and context is not None else [])
        if names:
            arrays = [context.data[name] for name in names if name in context.data]
            noun = "array" if len(arrays) == 1 else "arrays"
            return Selected(f"Data Set {data_set} · {len(arrays)} {noun}",
                            [ChannelArrays("Data", arrays)])
        return Selected(data.name, [ChannelArrays("Data", [data])])
    raise RuntimeError("Choose a data array or a data series")


def shown_values(data: NMData, channel: NMChannel | None) -> np.ndarray:
    """*data*'s values as the Plot shows them: with its channel's filters and
    transforms, if on (as they are, if they can't be applied)."""
    if channel is not None and channel.transform_on:
        try:
            return channel.transformed(data).nparray
        except Exception:  # shown as they are
            pass
    return data.nparray


def x_values(data: NMData, n: int) -> np.ndarray:
    """*data*'s x values: its x array if it fits, else from start and delta."""
    xarray = data.xarray
    if xarray is not None and len(xarray) == n and not data.xarray_problem():
        return np.asarray(xarray, dtype=float)
    return data.xscale.start + data.xscale.delta * np.arange(n)


# Rainbow: epochs coloured red to violet in order, in at most this many
# colours, each drawn as one line (fast for thousands of epochs)
RAINBOW_COLORS = 64


def rainbow_color(fraction: float) -> QtGui.QColor:
    """Red (0) to violet (1)."""
    return QtGui.QColor.fromHsvF(0.8 * min(max(fraction, 0.0), 1.0), 0.9, 0.85)


@dataclass
class PlotOptions:
    """How Main2 draws the epochs: a colour, or a rainbow; and each epoch's
    offset from the one before, in x and y (virtual: the data doesn't change)."""

    color: str = "#000000"
    rainbow: bool = True  # the epochs told apart, in order
    x_offset: float = 0.0
    y_offset: float = 0.0
    reverse: bool = False  # the last epoch first: no offset, the first color


class SelectionPlot(QtWidgets.QWidget):
    """Main2's plot: a plot per channel, one above another, sharing x; each
    channel's epochs overlaid (or offset), in one colour or a rainbow, drawn
    as few lines (fast for many epochs)."""

    EMPTY_TITLE = "Use Plot in Main2 to plot the selection"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel(self.EMPTY_TITLE, self)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        self.plots: list = []  # pg.PlotItem, a channel each
        self.options = PlotOptions()
        self.background_color = "#c3ced3"  # Settings › Plot's overlay color, lighter
        if pg is not None:
            self.graphics = pg.GraphicsLayoutWidget(self)
            self.graphics.setBackground("w")
            layout.addWidget(self.graphics, stretch=1)
        else:  # pragma: no cover - optional GUI dependency
            self.graphics = None
            layout.addWidget(QtWidgets.QLabel("Install pyqtgraph to enable plots.", self))
        self.shown: Selected | None = None

    def _lines(self, group: ChannelArrays) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """(x, y, colour) of the lines drawing *group*'s epochs: each epoch
        offset by its number in the group times the offsets; one line per
        colour, the epochs separated by NaN."""
        options = getattr(self, "_drawn_options", None) or self.options
        traces = []
        for data in group.arrays:
            y = shown_values(data, group.channel)
            if y is not None and y.size:
                traces.append((x_values(data, y.size), np.asarray(y, dtype=float)))
        if options.reverse:  # the last epoch first: drawn first, not offset, red
            traces.reverse()
        n = len(traces)
        colors = min(n, RAINBOW_COLORS) if options.rainbow else 1
        lines = []
        for c in range(colors):
            xs, ys = [], []
            for i in range(n):
                if colors > 1 and i * colors // n != c:
                    continue
                x, y = traces[i]
                xs += [x + i * options.x_offset, [np.nan]]
                ys += [y + i * options.y_offset, [np.nan]]
            if not xs:
                continue
            color = (rainbow_color((c + 0.5) / colors if colors > 1 else 0.0).name()
                     if options.rainbow else options.color)
            lines.append((np.concatenate(xs), np.concatenate(ys), color))
        return lines

    def show_selected(self, selected: Selected, options: PlotOptions | None = None) -> None:
        """Draw *selected* with *options* (default: :attr:`options`, Plot…'s
        answer; e.g. one color for Main2's outputs)."""
        self.shown = selected
        self._drawn_options = options or self.options
        self.title_label.setText(selected.title)
        if self.graphics is None:
            return
        self.graphics.clear()
        self.plots = []
        groups = [g for g in selected.groups if g.arrays]
        for row, group in enumerate(groups):
            plot = self.graphics.addPlot(row=row, col=0)
            for axis in ("left", "bottom"):  # units as they are: ms, not "kms"
                plot.getAxis(axis).enableAutoSIPrefix(False)
            if self.plots:
                plot.setXLink(self.plots[0])
            if group.background:  # e.g. an average's epochs, behind it
                xs, ys = [], []
                for data in group.background:
                    y = shown_values(data, None)
                    if y is not None and y.size:
                        xs += [x_values(data, y.size), [np.nan]]
                        ys += [np.asarray(y, dtype=float), [np.nan]]
                if xs:
                    back = plot.plot(np.concatenate(xs), np.concatenate(ys),
                                     pen=pg.mkPen(self.background_color, width=1),
                                     connect="finite")
                    back.setClipToView(True)
            self._draw_bands(plot, group)  # e.g. ± SEM, behind the average
            for x, y, color in self._lines(group):
                curve = plot.plot(x, y, pen=pg.mkPen(color, width=1), connect="finite")
                curve.setClipToView(True)
            first = group.arrays[0]
            units = first.yscale.units or None
            plot.setLabel("left", f"{group.label}  {first.yscale.label or ''}".strip(),
                          units=units)
            if row == len(groups) - 1:
                plot.setLabel("bottom", first.xscale.label or "",
                              units=first.xscale.units or None)
            else:
                plot.getAxis("bottom").setStyle(showValues=False)
            self.plots.append(plot)

    def _draw_bands(self, plot, group: ChannelArrays) -> None:
        """A shaded band from each array - its error to + its error (an
        average's ± Stdv or ± SEM), in the trace color, see-through."""
        options = getattr(self, "_drawn_options", None) or self.options
        for data, error in zip(group.arrays, group.errors):
            y = shown_values(data, group.channel)
            if error is None or y is None or not y.size:
                continue
            n = min(len(y), len(error))
            x = x_values(data, len(y))[:n]
            y = np.asarray(y[:n], dtype=float)
            error = np.asarray(error[:n], dtype=float)
            # The band's edges, in the plot (see-through: with no pen, there's
            # no outline for the band), then the band between
            clear = pg.mkPen((0, 0, 0, 0))
            upper = pg.PlotCurveItem(x, y + error, pen=clear)
            lower = pg.PlotCurveItem(x, y - error, pen=clear)
            plot.addItem(upper)
            plot.addItem(lower)
            color = QtGui.QColor(options.color)
            color.setAlpha(70)
            plot.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(color)))

    def clear(self) -> None:
        self.shown = None
        self.title_label.setText(self.EMPTY_TITLE)
        if self.graphics is not None:
            self.graphics.clear()
        self.plots = []


class _SelectionTableModel(ChannelTableModel):
    """Read-only (a view of the selection); each array with its own
    channel's filters and transforms."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._channel_of: dict[int, NMChannel | None] = {}

    def set_selected(self, selected: Selected) -> None:
        self._channel_of = {id(d): g.channel for g in selected.groups for d in g.arrays}
        self.set_arrays(selected.arrays)

    def _values(self, column: int):
        data = self._columns[column - 1]
        return shown_values(data, self._channel_of.get(id(data)))

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        return super().flags(index) & ~QtCore.Qt.ItemFlag.ItemIsEditable


class SelectionTable(ChannelTable):
    """Main2's table: an x column, then a column per array (channel by
    channel). Read-only: values are edited in the channel's Table."""

    EMPTY_TITLE = "Use Table in Main2 to list the selection's values"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = _SelectionTableModel(self)
        self.view.setModel(self.model)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.title_label.setText(self.EMPTY_TITLE)
        self.shown: Selected | None = None

    def show_selected(self, selected: Selected) -> None:
        self.shown = selected
        self.model.set_selected(selected)
        self.title_label.setText(selected.title)

    def clear(self) -> None:
        self.shown = None
        self.model.set_selected(Selected(""))
        self.title_label.setText(self.EMPTY_TITLE)


class PlotOptionsDialog(QtWidgets.QDialog):
    """Plot…'s prompt: the epochs' color (one, or a rainbow) and each
    epoch's X and Y offset from the one before; Plot or Cancel."""

    def __init__(self, options: PlotOptions, what: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot")
        self.setMinimumWidth(340)  # the heading (what's selected) on a line
        self._color = options.color
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(f"Plot {what}", self)  # what's selected
        heading.setWordWrap(True)
        layout.addWidget(heading)
        form = QtWidgets.QFormLayout()
        color_row = QtWidgets.QHBoxLayout()
        self.color_combo = QtWidgets.QComboBox(self)
        self.color_combo.addItem("One color", False)
        self.color_combo.addItem("Rainbow", True)
        self.color_combo.setToolTip("Rainbow: the epochs red to violet, in order")
        self.color_combo.setCurrentIndex(1 if options.rainbow else 0)
        color_row.addWidget(self.color_combo)
        self.color_button = QtWidgets.QToolButton(self)
        self.color_button.setToolTip("Choose the color")
        self.color_button.setFixedSize(28, 22)
        color_row.addWidget(self.color_button)
        color_row.addStretch(1)
        form.addRow("Color", color_row)
        self.x_offset_edit = QtWidgets.QLineEdit(f"{options.x_offset:g}", self)
        self.y_offset_edit = QtWidgets.QLineEdit(f"{options.y_offset:g}", self)
        for label, edit, axis in (("X offset", self.x_offset_edit, "x"),
                                  ("Y offset", self.y_offset_edit, "y")):
            fit_number_edit(edit)
            edit.setToolTip(
                f"Each epoch's {axis} offset from the one before, in the plot's {axis} "
                "units (0: overlaid). Only the plot moves: the data doesn't change")
            form.addRow(label, edit)
        self.reverse_checkbox = QtWidgets.QCheckBox("Reverse order", self)
        self.reverse_checkbox.setToolTip(
            "The last epoch first: it has no offset and the first color, and the "
            "first epoch is drawn last, on top")
        self.reverse_checkbox.setChecked(options.reverse)
        form.addRow("", self.reverse_checkbox)
        layout.addLayout(form)
        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setStyleSheet("color: #c62828;")
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(self)
        self.plot_button = buttons.addButton(
            "Plot", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.color_combo.currentIndexChanged.connect(self._show_color)
        self.color_button.clicked.connect(self._choose_color)
        self._show_color()

    def _show_color(self) -> None:
        self.color_button.setStyleSheet(
            f"QToolButton {{ background: {self._color}; border: 1px solid #888; "
            "border-radius: 3px; }")
        self.color_button.setEnabled(not self.color_combo.currentData())

    def _choose_color(self) -> None:
        color = self._ask_color(QtGui.QColor(self._color))
        if color is not None:
            self.set_color(color)

    def _ask_color(self, current: QtGui.QColor) -> QtGui.QColor | None:
        """The color chosen (None: cancelled); replaced in tests."""
        color = QtWidgets.QColorDialog.getColor(current, self, "Plot color")
        return color if color.isValid() else None

    def set_color(self, color: QtGui.QColor | str) -> None:
        self._color = QtGui.QColor(color).name()
        self._show_color()

    def options(self) -> PlotOptions:
        """The options entered.

        Raises:
            ValueError: If an offset isn't a number (the message says which).
        """
        offsets = []
        for label, edit in (("X offset", self.x_offset_edit),
                            ("Y offset", self.y_offset_edit)):
            text = edit.text().strip()
            try:
                value = float(text or 0)
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number")
            if not np.isfinite(value):
                raise ValueError(f"{label}: {text} is not a finite number")
            offsets.append(value)
        return PlotOptions(color=self._color, rainbow=bool(self.color_combo.currentData()),
                           x_offset=offsets[0], y_offset=offsets[1],
                           reverse=self.reverse_checkbox.isChecked())

    def accept(self) -> None:
        try:
            self.options()
        except ValueError as error:
            self.error_label.setText(str(error))
            return
        super().accept()


# --- Functions (#553) -----------------------------------------------------

# Main1's functions: (what it makes, its prompt's options as (key, label, tip))
FUNCTIONS = {
    "Average": ("the mean of each channel's arrays (MN_Avg_)", (
        ("compute_stdv", "Standard deviation (MN_Stdv_)", ""),
        ("compute_var", "Variance (MN_Var_)", ""),
        ("compute_sem", "Standard error of the mean (MN_SEM_)", ""),
    )),
    "Sum": ("the sum of each channel's arrays (MN_Sum_)", ()),
    "SumSqr": ("the sum of squares of each channel's arrays (MN_SumSqr_)", ()),
    "Min": ("each point's minimum over each channel's arrays (MN_Min_)", ()),
    "Max": ("each point's maximum over each channel's arrays (MN_Max_)", ()),
    "Concatenate": ("each channel's arrays joined end to end (MN_Cat_)", ()),
}
# Outputs that can be plotted with the arrays they were made from behind them
BACKGROUND_PREFIXES = ("MN_Avg_", "MN_Min_", "MN_Max_")


def light(color: str, amount: float = 0.55) -> str:
    """*color* mixed with white (*amount* of white): the data behind an
    average, so it stands out on top."""
    c = QtGui.QColor(color)
    mix = [round(v + (255 - v) * amount) for v in (c.red(), c.green(), c.blue())]
    return QtGui.QColor(*mix).name()


def make_function(name: str, options: dict[str, bool], ignore_nans: bool = True):
    """*name*'s NMMainOp with *options*, and Ignore NaNs (Settings › Main)."""
    if name == "Average":
        return NMMainOpAverage(ignore_nans=ignore_nans, **options)
    if name == "Concatenate":
        return NMMainOpConcatenate()
    return {"Sum": NMMainOpSum, "SumSqr": NMMainOpSumSqr, "Min": NMMainOpMin,
            "Max": NMMainOpMax}[name](ignore_nans=ignore_nans)


def function_option_keys(name: str) -> list[tuple[str, str, str]]:
    """The options a function's prompt asks (none: it runs without one)."""
    return list(FUNCTIONS[name][1])


class FunctionDialog(QtWidgets.QDialog):
    """A function's prompt on Run: what it makes, its options; Run or Cancel."""

    def __init__(self, name: str, options: dict[str, bool], what: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(name)
        self.setMinimumWidth(340)
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(f"{name} of {what}: {FUNCTIONS[name][0]}", self)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for key, label, tip in function_option_keys(name):
            box = QtWidgets.QCheckBox(label, self)
            box.setToolTip(tip)
            box.setChecked(options.get(key, False))
            layout.addWidget(box)
            self.checkboxes[key] = box
        buttons = QtWidgets.QDialogButtonBox(self)
        self.run_button = buttons.addButton(
            "Run", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict[str, bool]:
        return {key: box.isChecked() for key, box in self.checkboxes.items()}


class Main2Panel(ResultsPanel):
    """Main2: a Main subfolder and an output in it, to Plot or Table (#553)."""

    TOOLFOLDER_PREFIX = "Main_"
    TOOL_NAME = "Main"
    PANEL_NAME = "Main2"
    OBJECT_PREFIX = "main2"
    EXAMPLE_ARRAY = "MN_Avg_RecordA"

    # The output chosen, to plot: its data series' arrays (a Selected)
    outputs_plot_requested = QtCore.pyqtSignal(object)

    def __init__(self, manager: NMManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(manager, parent)
        self.plot_action.setToolTip(
            "Plot the chosen output, with the rest of its data series (each channel's)")
        self.table_action.setToolTip(
            "List the subfolder's outputs in the Table panel, the chosen one's column "
            "selected")
        self.refresh()

    def _add_plot_options(self, menu: QtWidgets.QMenu) -> None:
        # An average (or Min, Max) over the arrays it was made from
        self.background_action = menu.addAction("Plot with its data")
        self.background_action.setCheckable(True)
        self.background_action.setChecked(True)
        self.background_action.setToolTip(
            "Plot an average, minimum or maximum over the arrays it was made from, "
            "in the light overlay color")
        self.background_action.toggled.connect(lambda _on: self.array_changed.emit())
        # An average with a band of its spread: ± Stdv (or from Var) or ± SEM;
        # one or neither
        self.error_actions: dict[str, QtGui.QAction] = {}
        for kind, label in (("stdv", "Plot ± Stdv"), ("sem", "Plot ± SEM")):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setToolTip(
                f"Plot an average with a shaded band from mean - {label[7:]} to mean + "
                f"{label[7:]} (Average… with {label[7:]} saved, or Var for Stdv)")
            action.toggled.connect(lambda on, k=kind: self._on_error_toggled(k, on))
            self.error_actions[kind] = action

    def _on_error_toggled(self, kind: str, on: bool) -> None:
        if on:  # one band at a time
            for other, action in self.error_actions.items():
                if other != kind and action.isChecked():
                    action.blockSignals(True)
                    action.setChecked(False)
                    action.blockSignals(False)
        self.array_changed.emit()

    def _update_array_tools(self, name: str | None) -> None:
        self.background_action.setEnabled(bool(self._background_of(name)))
        for kind, action in self.error_actions.items():
            action.setEnabled(self._error_of(name, kind) is not None)

    def _error_of(self, name: str | None, kind: str) -> np.ndarray | None:
        """Average *name*'s Stdv (or the root of its Var) or SEM, if saved."""
        folder = self.toolfolder()
        if folder is None or not name or not name.startswith("MN_Avg_"):
            return None
        if kind == "sem":
            sem = folder.data.get(name.replace("MN_Avg_", "MN_SEM_", 1))
            return sem.nparray if sem is not None else None
        stdv = folder.data.get(name.replace("MN_Avg_", "MN_Stdv_", 1))
        if stdv is not None:
            return stdv.nparray
        var = folder.data.get(name.replace("MN_Avg_", "MN_Var_", 1))
        return np.sqrt(var.nparray) if var is not None else None

    def _errors(self, outputs) -> list:
        """The band of each of *outputs*, as chosen (± Stdv or ± SEM)."""
        kind = next((k for k, a in self.error_actions.items() if a.isChecked()), None)
        return [self._error_of(d.name, kind) if kind else None for d in outputs]

    def error_label(self) -> str:
        """e.g. " ± SEM", for the plot's title (when one is drawn)."""
        kind = next((k for k, a in self.error_actions.items() if a.isChecked()), None)
        return {"stdv": " ± Stdv", "sem": " ± SEM"}.get(kind, "")

    def no_results_message(self) -> str:
        return (f"No Main outputs in {self._folder.name}: run a function in Main1 "
                "(e.g. Average).")

    def _array_names(self) -> list[str]:
        folder = self.toolfolder()
        return list(folder.data.keys()) if folder is not None else []

    def chosen_data(self) -> NMData | None:
        folder, name = self.toolfolder(), self.selected_array()
        return folder.data.get(name) if folder is not None and name else None

    def _background_of(self, name: str | None) -> list[NMData]:
        """The arrays output *name* was made from, if it's an average (or
        Min, Max) and they're still there."""
        folder = self.toolfolder()
        if folder is None or not name or not name.startswith(BACKGROUND_PREFIXES):
            return []
        return [d for d in folder.sources.get(name, [])
                if getattr(d, "_container", None) is not None
                and d._container.get(d.name) is d]

    def _behind(self, outputs) -> list[NMData]:
        """What's plotted behind *outputs*: the arrays they were made from,
        with Plot with its data on."""
        if not self.background_action.isChecked():
            return []
        return [d for output in outputs for d in self._background_of(output.name)]

    def selected_outputs(self) -> Selected | None:
        """The chosen output with its data series' other channels (e.g.
        MN_Avg_RecordA and MN_Avg_RecordB), or alone; each over its data, if on."""
        folder, data = self.toolfolder(), self.chosen_data()
        if data is None:
            return None
        for series in folder.dataseries.values():
            channels = list(series.channels.values())
            if any(d is data for c in channels for d in c.data):
                groups = [ChannelArrays(c.name, list(c.data), background=self._behind(c.data),
                                        errors=self._errors(c.data))
                          for c in channels]
                banded = any(e is not None for g in groups for e in g.errors)
                return Selected(f"{folder.name}: {series.name}"
                                + (self.error_label() if banded else ""), groups)
        return Selected(f"{folder.name}: {data.name}",
                        [ChannelArrays(data.name, [data], background=self._behind([data]),
                                       errors=self._errors([data]))])

    def all_outputs(self) -> Selected | None:
        """Every output of the subfolder, for its table."""
        folder = self.toolfolder()
        if folder is None or not len(folder.data):
            return None
        return Selected(folder.name, [ChannelArrays("Outputs", list(folder.data.values()))])

    def plot_selected(self) -> None:
        selected = self.selected_outputs()
        if selected is not None:
            self.outputs_plot_requested.emit(selected)


class MainToolTab(SelectionRunTab):
    """Main1: Display (plot or table the strip's selection) and Functions
    (Average, Sum…, run on it). Main2: their outputs, saved in Main
    subfolders."""

    # Its tab in the Plot and Table panels shows Main's plots and tables
    view_label = "Main2"

    _CONFIG_LABELS = {"ignore_nans": "Ignore NaNs", "overwrite": "Overwrite results"}
    _CONFIG_TIPS = {
        "ignore_nans": "Functions leave NaN values out (otherwise a NaN in any array "
                       "gives NaN there)",
        "overwrite": "Functions on the same selection share its first Main "
                     "subfolder (…_0), each run replacing only the outputs it "
                     "makes; off: a new subfolder each run",
    }
    RANGE_COLOR = "#2563eb"  # as Stats' and Spike's

    def __init__(self, manager: NMManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Main", manager, parent)
        self._tool = manager.main  # nm.main
        self._function_options: dict[str, dict[str, bool]] = {}  # kept for next time

        self.mode_tabs = QtWidgets.QTabWidget(self)
        self.mode_tabs.setObjectName("mainModeTabs")
        self.use_tabs_for_parameters(self.mode_tabs)

        # Main1: Display, then Functions
        self.main1_page = QtWidgets.QWidget(self.mode_tabs)
        main1 = QtWidgets.QVBoxLayout(self.main1_page)
        main1.setContentsMargins(6, 8, 6, 8)
        main1.addWidget(self._heading("Display"))
        # What Plot, Table and a function would use (the selection, as it changes)
        self.selection_label = QtWidgets.QLabel("", self.main1_page)
        self.selection_label.setWordWrap(True)
        main1.addWidget(self.selection_label)
        buttons = QtWidgets.QHBoxLayout()
        self.plot_button = QtWidgets.QPushButton("Plot…", self.main1_page)
        self.plot_button.setToolTip(
            "Plot every selected epoch: each channel in a plot of its own, one above "
            "another, the epochs overlaid or offset, in a color or a rainbow")
        self.table_button = QtWidgets.QPushButton("Table", self.main1_page)
        self.table_button.setToolTip(
            "List the selected arrays' values: an x column, then a column per array")
        for button in (self.plot_button, self.table_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        main1.addLayout(buttons)
        main1.addSpacing(8)
        main1.addWidget(self._heading("Functions"))
        function_row = QtWidgets.QHBoxLayout()
        self.function_combo = QtWidgets.QComboBox(self.main1_page)
        self.function_combo.setObjectName("mainFunction")
        for name, (what, _options) in FUNCTIONS.items():
            self.function_combo.addItem(name)
            self.function_combo.setItemData(
                self.function_combo.count() - 1, f"{name}: {what}",
                QtCore.Qt.ItemDataRole.ToolTipRole)
        self.function_combo.setToolTip(
            "Run makes it from the selection, in a Main subfolder (viewed in Main2)")
        function_row.addWidget(self.function_combo)
        function_row.addStretch(1)
        main1.addLayout(function_row)
        # The x range functions use, also as lines to drag on the Plot
        x_row = QtWidgets.QHBoxLayout()
        self.xbgn_edit = QtWidgets.QLineEdit(self.main1_page)
        self.xbgn_edit.setPlaceholderText("-inf")
        self.xend_edit = QtWidgets.QLineEdit(self.main1_page)
        self.xend_edit.setPlaceholderText("inf")
        for label, edit in (("X begin", self.xbgn_edit), ("X end", self.xend_edit)):
            fit_number_edit(edit)
            edit.setToolTip("The x range functions use (-inf, inf: all of each array); "
                            "or drag its lines on the Plot")
            x_row.addWidget(QtWidgets.QLabel(label, self.main1_page))
            x_row.addWidget(edit)
        x_row.addStretch(1)
        main1.addLayout(x_row)
        main1.addStretch(1)
        self.mode_tabs.addTab(self.main1_page, "Main1")

        # Main2: the functions' outputs
        self.main2 = Main2Panel(manager, self.mode_tabs)
        self.main2_page = self.main2
        self.mode_tabs.addTab(self.main2, "Main2")

        # In the app window: the plot and table (Display's, or Main2's)
        self.plot_view = SelectionPlot()
        self.table_view = SelectionTable()
        self.plot_button.clicked.connect(self.plot_selection)
        self.table_button.clicked.connect(self.table_selection)
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.main2.results_changed.connect(self.results_changed)
        self.main2.outputs_plot_requested.connect(self._plot_outputs)
        self.main2.table_requested.connect(self._table_outputs)
        self.main2.array_changed.connect(self._on_output_chosen)
        self.run_finished.connect(self._on_run_finished)
        self.xbgn_edit.editingFinished.connect(self._save_x_range)
        self.xend_edit.editingFinished.connect(self._save_x_range)
        self.results_group.hide()
        self._load_x_range()
        self._show_selection()

    def _heading(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self.main1_page)
        label.setStyleSheet("font-weight: bold;")
        return label

    # --- the app window -------------------------------------------------------

    def preferred_view(self) -> str:
        return "tool" if self.mode_tabs.currentWidget() is self.main2 else "channel"

    def lists_view(self) -> bool:
        return True  # Main2's tab, to switch to

    def has_results(self) -> bool:
        return (self.plot_view.shown is not None or self.table_view.shown is not None
                or self.main2.has_results())

    def bind_selection_model(self, model: Any) -> None:
        super().bind_selection_model(model)
        self.main2.bind_selection_model(model)
        self._show_selection()

    def set_folder_actions(self, actions) -> None:
        self.main2.set_folder_actions(actions)  # its Folder ⋯ menu

    def set_settings(self, settings) -> None:
        super().set_settings(settings)
        # One color starts as the Plot's trace color (Settings › Plot); an
        # average's data behind it, in the overlay color
        self.plot_view.options.color = getattr(settings, "plot_trace_color", "#000000")
        self.plot_view.background_color = light(
            getattr(settings, "plot_overlay_color", "#78909c"))
        self._on_mode_tab_changed(self.mode_tabs.currentIndex())

    def config_options(self) -> list[tuple]:
        """(key, label, value, tip) of each option in Settings › Main."""
        return [(key, label, bool(getattr(self._tool, key)), self._CONFIG_TIPS[key])
                for key, label in self._CONFIG_LABELS.items()]

    def set_config_option(self, key: str, value: Any) -> None:
        if key not in self._CONFIG_LABELS:
            raise KeyError(f"Main has no option {key!r}")
        setattr(self._tool, key, value)
        config = self._tool.config
        if config is not None and key in config._schema:
            setattr(config, key, value)

    def refresh(self) -> None:
        """The data changed (e.g. arrays deleted or edited): drawn again."""
        super().refresh()
        self._load_x_range()  # e.g. set by a command
        self.main2.refresh()
        for view in (self.plot_view, self.table_view):
            if view.shown is not None:
                view.show_selected(self._still_there(view.shown))
        self._show_selection()

    @staticmethod
    def _still_there(selected: Selected) -> Selected:
        """*selected* without arrays since deleted."""
        def kept(data):
            container = getattr(data, "_container", None)
            return container is not None and container.get(data.name) is data
        groups = []
        for g in selected.groups:
            pairs = [(d, e) for d, e in zip(g.arrays, g.errors or [None] * len(g.arrays))
                     if kept(d)]
            groups.append(ChannelArrays(g.label, [d for d, _e in pairs], g.channel,
                                        [d for d in g.background if kept(d)],
                                        [e for _d, e in pairs]))
        return Selected(selected.title, groups)

    def _on_mode_tab_changed(self, _index: int) -> None:
        # Run (and its Progress) belong to Main1; the status stays, as on
        # Stats2's and Spike2's tabs; the view follows
        main1 = self.mode_tabs.currentWidget() is self.main1_page
        self.run_row.setVisible(main1)
        self.display_row.setVisible(main1 and self._settings is not None)
        self.view_requested.emit(self.preferred_view())

    # --- Display: the selection -----------------------------------------------

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        super()._on_selection_changed(selection)
        self._show_selection()

    def _show_selection(self) -> None:
        """Say what Plot and Table would show; they can't without a selection."""
        try:
            selected = selected_arrays(self.selection)
        except RuntimeError as error:
            self.selection_label.setText(str(error))
            self.plot_button.setEnabled(False)
            self.table_button.setEnabled(False)
            return
        self.selection_label.setText(selected.title)
        self.plot_button.setEnabled(True)
        self.table_button.setEnabled(True)

    def _ask_plot_options(self, selected: Selected) -> PlotOptions | None:
        """Plot…'s prompt: how to draw *selected* (None: cancelled); its
        answer is kept for next time. Replaced in tests."""
        dialog = PlotOptionsDialog(self.plot_view.options, selected.title, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.options()

    def plot_selection(self) -> None:
        """Plot the selection in the Main2 tab, and show it."""
        try:
            selected = selected_arrays(self.selection)
        except RuntimeError as error:
            self.selection_label.setText(str(error))
            return
        options = self._ask_plot_options(selected)
        if options is None:  # cancelled
            return
        self.plot_view.options = options
        self.plot_view.show_selected(selected)
        self.results_changed.emit()  # its tab
        self.view_requested.emit("plot")

    def table_selection(self) -> None:
        """List the selection's values in the Main2 tab's table, and show it."""
        try:
            selected = selected_arrays(self.selection)
        except RuntimeError as error:
            self.selection_label.setText(str(error))
            return
        self.table_view.show_selected(selected)
        self.results_changed.emit()
        self.view_requested.emit("table")

    # --- X begin / X end, and its lines on the Plot -----------------------------

    @staticmethod
    def _format_bound(value: float) -> str:
        if value == -math.inf:
            return "-inf"
        if value == math.inf:
            return "inf"
        return f"{value:.10g}"

    def _load_x_range(self) -> None:
        """The X boxes show the tool's range (without saving it)."""
        for edit, value in ((self.xbgn_edit, self._tool.xbgn),
                            (self.xend_edit, self._tool.xend)):
            edit.blockSignals(True)
            edit.setText(self._format_bound(value))
            edit.blockSignals(False)
        self.ranges_changed.emit()

    def _save_x_range(self) -> bool:
        """What's typed, in the tool (a command each changed); False, with a
        warning, if it isn't a range."""
        values = []
        for label, edit in (("X begin", self.xbgn_edit), ("X end", self.xend_edit)):
            text = edit.text().strip() or ("-inf" if edit is self.xbgn_edit else "inf")
            try:
                value = float(text)
            except ValueError:
                self.show_warning(f"{label}: {text!r} is not a number")
                return False
            if math.isnan(value):
                self.show_warning(f"{label}: NaN is not a number")
                return False
            values.append(value)
        xbgn, xend = values
        if not xbgn < xend:
            self.show_warning("X begin must be less than X end")
            return False
        if self._tool.xbgn != xbgn:
            self._tool.xbgn = xbgn
        if self._tool.xend != xend:
            self._tool.xend = xend
        self._load_x_range()
        return True

    def plot_ranges(self) -> list[PlotRange]:
        # Draggable, locked or hidden: the Plot's ⋯ › Tool Ranges & Levels
        return [PlotRange("range", "Main X range", self._tool.xbgn, self._tool.xend,
                          self.RANGE_COLOR)]

    def set_plot_range(self, key: str, xbgn: float, xend: float, final: bool) -> None:
        """A dragged range: X begin / X end follow, saved when released."""
        self.xbgn_edit.setText(self._format_bound(xbgn))
        self.xend_edit.setText(self._format_bound(xend))
        if final:
            self._save_x_range()

    # --- Functions: run on the selection, all channels at once -----------------

    def _ask_function(self, name: str) -> dict[str, bool] | None:
        """A function's prompt: its options (None: cancelled); kept for next
        time. Replaced in tests."""
        try:
            what = selected_arrays(self.selection).title
        except RuntimeError:
            what = "the selection"
        dialog = FunctionDialog(name, self._function_options.get(name, {}), what, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.options()

    def _on_run_clicked(self) -> None:
        if not self._running:  # first, the x range, the function and its options
            if not self._save_x_range():
                return
            name = self.function_combo.currentText()
            options: dict[str, bool] = {}
            if function_option_keys(name):  # e.g. Average's Stdv, Var, SEM
                options = self._ask_function(name)
                if options is None:  # cancelled: no run
                    return
                self._function_options[name] = options
            self._tool.op = make_function(  # nm.main.op = ...
                name, options, ignore_nans=self._tool.ignore_nans)
        super()._on_run_clicked()

    def _scope_runs(self) -> list[tuple[str, dict]] | None:
        """All Sets / All Groups: a run each; All Channels doesn't split a
        run (a function's outputs: a channel each, in one subfolder)."""
        selection = self.selection
        series = selection.get("dataseries")
        if selection.get("source") != "dataseries" or series is None:
            return None
        from pyneuromatic.gui.selection_model import scope_runs
        return scope_runs(series, selection)

    def _selected_targets(self, selection: dict | None = None) -> list[dict]:
        selection = self.selection if selection is None else selection
        series = selection.get("dataseries")
        if (selection.get("source") == "dataseries" and selection.get("all_channels")
                and series is not None):
            targets = []
            for channel in series.channels.values():
                targets += super()._selected_targets(
                    {**selection, "channel": channel, "all_channels": None})
            return targets
        return super()._selected_targets(selection)

    def _run_selection_args(self, targets: list[dict], selection: dict) -> tuple[Any, dict]:
        if selection.get("source") == "dataseries" and selection.get("all_channels"):
            source, args = super()._run_selection_args(
                targets, {**selection, "channel": targets[0]["channel"]})
            args["channel"] = "all"  # one run, every channel
            return source, args
        return super()._run_selection_args(targets, selection)

    def show_run_results(self) -> None:
        """The run's subfolder in Main2, its first output chosen (shown once
        the run has finished): with Overwrite, the subfolder may hold earlier
        runs' outputs too, and the one chosen there before."""
        if self._tool.toolfolder is not None:
            self.main2.show_toolfolder(self._tool.toolfolder)
            outputs = self._tool.outputs
            if outputs:
                self.main2.select_array(outputs[0])

    def _on_run_finished(self, succeeded: bool) -> None:
        # After the Plot is back as it was before the run: Main2, the run's
        # first output plotted, as Stats2 and Spike2 show theirs
        if succeeded and self._tool.toolfolder is not None:
            self.mode_tabs.setCurrentWidget(self.main2)
            self.main2.plot_selected()

    # --- Main2: the outputs ---------------------------------------------------

    def _output_options(self) -> PlotOptions:
        """Outputs are drawn in one color (the Plot's), not offset."""
        return PlotOptions(color=self.plot_view.options.color, rainbow=False)

    def _plot_outputs(self, selected: Selected) -> None:
        self.plot_view.show_selected(selected, self._output_options())
        self.results_changed.emit()
        self.view_requested.emit("plot")

    def _on_output_chosen(self) -> None:
        """The plot follows the output chosen in Main2 (while it's shown)."""
        if self.mode_tabs.currentWidget() is self.main2:
            selected = self.main2.selected_outputs()
            if selected is not None:
                self.plot_view.show_selected(selected, self._output_options())

    def _table_outputs(self) -> None:
        selected = self.main2.all_outputs()
        if selected is None:
            return
        self.table_view.show_selected(selected)
        self.table_view.select_array(self.main2.chosen_data())
        self.results_changed.emit()
        self.view_requested.emit("table")
