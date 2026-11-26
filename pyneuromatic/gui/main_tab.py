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
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.gui.context_views import (
    RAINBOW_COLORS, ChannelTableModel, ChannelTable, rainbow_color)
from pyneuromatic.core.nm_params import BOOLEAN, CHOICE, COLOR, NUMBER, NMParam, keywords
from pyneuromatic.gui.param_form import ParamForm, XScale
from pyneuromatic.gui.plot_levels import PlotLevel
from pyneuromatic.gui.plot_ranges import PlotRange
from pyneuromatic.gui.results_panel import ResultsPanel
from pyneuromatic.gui.selection_model import epoch_scope_label, scoped_epoch_names
from pyneuromatic.gui.tool_run import SelectionRunTab
from pyneuromatic.gui.tool_tab import fit_number_edit
from pyneuromatic.tools.nm_tool_main import NMToolMain
from pyneuromatic.tools.nm_main_op import array_names, find_array
from pyneuromatic.tools.nm_main_op import (
    NMMainOpAlign, NMMainOpInterpolate, NMMainOpResample, NMMainOpRescaleX,
    NMMainOpSetXStart, NMMainOpArithmeticByArray, NMMainOpHistogram,
    NMMainOpInequality,
    NMMainOpArithmetic, NMMainOpAverage, NMMainOpBaseline, NMMainOpConcatenate,
    NMMainOpDeleteNaNs, NMMainOpDeletePoints, NMMainOpDFOF, NMMainOpDifferentiate,
    NMMainOpFilter, NMMainOpInsertPoints, NMMainOpIntegrate, NMMainOpMax, NMMainOpMin,
    NMMainOpNormalize, NMMainOpRedimension, NMMainOpReplaceValues, NMMainOpRescale,
    NMMainOpReverse, NMMainOpRotate, NMMainOpSmooth, NMMainOpSum, NMMainOpSumSqr)

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


@dataclass
class PlotOptions:
    """How Main2 draws the epochs: a colour, or a rainbow; and each epoch's
    offset from the one before, in x and y (virtual: the data doesn't change)."""

    color: str = "#000000"
    rainbow: bool = True  # the epochs told apart, in order
    x_offset: float = 0.0
    y_offset: float = 0.0
    reverse: bool = False  # the last epoch first: no offset, the first color
    reverse_colors: bool = False  # the rainbow from violet to red


def _draw_as_segments(item) -> None:
    """Draw *item*'s line as separate segments (QPainter.drawLines), not one
    path: a path with many pieces, e.g. Inequality's NaN-where-false outputs
    of long arrays (thousands of runs between NaNs), crashes Qt's raster
    stroker (a segfault, not an error)."""
    curve = getattr(item, "curve", item)  # a PlotDataItem's PlotCurveItem
    curve.setSegmentedLineMode("on")


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
            fraction = (c + 0.5) / colors if colors > 1 else 0.0
            if options.reverse_colors:  # violet first, red last
                fraction = 1.0 - fraction if colors > 1 else 1.0
            color = (rainbow_color(fraction).name()
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
                    _draw_as_segments(back)
            self._draw_bands(plot, group)  # e.g. ± SEM, behind the average
            for x, y, color in self._lines(group):
                curve = plot.plot(x, y, pen=pg.mkPen(color, width=1), connect="finite")
                curve.setClipToView(True)
                _draw_as_segments(curve)
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


# --- Functions (#553) -----------------------------------------------------

# Plot's options, as a form (#555): how the selected epochs are drawn
PLOT_PARAMS = (
    NMParam("colors", CHOICE, "Color", default="rainbow",
            choices=(("rainbow", "Rainbow"), ("one", "One color")),
            tip="Plot epochs from red to violet"),
    NMParam("color", COLOR, "One color", default="#000000", when=("colors", "one"),
            tip="Plot epochs in one color"),
    NMParam("reverse_colors", BOOLEAN, "Reverse colors", default=False,
            when=("colors", "rainbow"),
            tip="Rainbow from violet to red (the first epoch violet)"),
    NMParam("x_offset", NUMBER, "X offset", default=0.0, finite=True,
            tip="Apply a virtual offset to each epoch's x-value (0: overlaid)"),
    NMParam("y_offset", NUMBER, "Y offset", default=0.0, finite=True,
            tip="Apply a virtual offset to each epoch's y-value (0: overlaid)"),
    NMParam("reverse", BOOLEAN, "Reverse order", default=False,
            tip="Plot the last epoch first, the first epoch last"),
)


def plot_options(values: dict[str, Any]) -> PlotOptions:
    """Plot's form values as :class:`PlotOptions`."""
    return PlotOptions(color=values["color"], rainbow=values["colors"] == "rainbow",
                       x_offset=values["x_offset"], y_offset=values["y_offset"],
                       reverse=values["reverse"],
                       reverse_colors=values.get("reverse_colors", False))


# Main1's entries, in one list as Stats' measures, groups divided: (what it
# does, its NMMainOp or None, its form's parameters)
DISPLAY = {
    "Plot": ("Plot selected arrays", PLOT_PARAMS),
    "Table": ("Create a table of selected arrays", ()),
}

# Main1's functions: (what it makes, its NMMainOp, whose PARAMS its form shows).
# They work point by point across the selected arrays (a channel's epochs,
# or a Data Set's arrays), not along one array: e.g. Sum's first point is the
# sum of the arrays' first points
POINTWISE = "Compute the {} along each data point of the selected arrays"
FUNCTIONS = {
    "Average": (POINTWISE.format("average"), NMMainOpAverage),
    "Sum": (POINTWISE.format("sum"), NMMainOpSum),
    "SumSqr": (POINTWISE.format("sum of squares"), NMMainOpSumSqr),
    "Min": (POINTWISE.format("minimum"), NMMainOpMin),
    "Max": (POINTWISE.format("maximum"), NMMainOpMax),
    "Concatenate": ("Compute one long array of the selected arrays joined end to end",
                    NMMainOpConcatenate),
    # One output per selected array (or, Histogram, of them all), in a Main
    # subfolder as a data series (#563)
    "Inequality": ("Test each point of the selected arrays (e.g. y > a), making "
                   "an array of the results for each", NMMainOpInequality),
    "Histogram": ("Compute the distribution of the values of each selected "
                  "array, or of them all together", NMMainOpHistogram),
}

# Main1's operations (#557): they change the selected arrays themselves
# (what each does, its NMMainOp)
OPERATIONS = {
    "Arithmetic": ("apply an arithmetic operation to the selected arrays, e.g. + 1 or * 2",
                   NMMainOpArithmetic),
    "Arithmetic by Array": ("apply an array to the selected arrays, point by "
                            "point, e.g. subtract an average", NMMainOpArithmeticByArray),
    "Replace Values": ("replace a value in the selected arrays with another, "
                       "e.g. NaN with 0", NMMainOpReplaceValues),
    "Baseline": ("subtract a baseline window's mean", NMMainOpBaseline),
    "Normalize": ("scale selected arrays so they span from Low to High "
                  "(0 to 1 by default)", NMMainOpNormalize),
    "dF/F": ("ΔF over F0, (F - F0) / F0, where F0 is a baseline mean", NMMainOpDFOF),
    "Rescale Y": ("convert the y units of the selected arrays (e.g. pA to nA)",
                  NMMainOpRescale),
    "Smooth": ("smooth selected arrays (boxcar, binomial, Savitzky-Golay)", NMMainOpSmooth),
    "Filter": ("filter selected arrays (Butterworth, Bessel, notch)", NMMainOpFilter),
    "Differentiate": ("compute each array's derivative", NMMainOpDifferentiate),
    "Integrate": ("compute each array's running integral", NMMainOpIntegrate),
    # The Points operations (#559), listed apart (POINTS)
    "Insert Points": ("insert points into the selected arrays",
                      NMMainOpInsertPoints),
    "Delete Points": ("delete points from the selected arrays",
                      NMMainOpDeletePoints),
    "Delete NaNs": ("delete NaN (or ±inf) points from the selected arrays "
                    "(they get shorter)", NMMainOpDeleteNaNs),
    "Redimension": ("set the number of points of the selected arrays "
                    "(cut or padded at the end)", NMMainOpRedimension),
    "Reverse": ("reverse the order of the points of the selected arrays",
                NMMainOpReverse),
    "Rotate": ("rotate points of the selected arrays, wrapping round: "
               "rotated by 1, each point moves one later and the last becomes "
               "the first; by -1, each moves one earlier and the first becomes "
               "the last", NMMainOpRotate),
    # The X scale operations (#561), listed apart (XSCALE)
    "Rescale X": ("convert the x-scale units of the selected arrays (e.g. s to ms)",
                  NMMainOpRescaleX),
    "Set X Start": ("set the x-scale start of the selected arrays, or shift it by a "
                    "value (e.g. by -10 to move x = 10 to 0)", NMMainOpSetXStart),
    "Resample": ("resample the selected arrays to a new sample interval "
                 "(anti-aliased)", NMMainOpResample),
    "Interpolate": ("interpolate the selected arrays onto a common x axis "
                    "(each channel's), or onto a template array's x values",
                    NMMainOpInterpolate),
    "Align": ("shift the x-start of the selected arrays so their reference x values "
              "line up (e.g. a Stats2 onset time per epoch, at x = 0)", NMMainOpAlign),
}
# Operations a channel's Filter & Transform also has, applied virtually
# (pyneuromatic.core.nm_transform): Main1 points to it
ALSO_VIRTUAL = ("Baseline", "Normalize", "dF/F", "Smooth", "Filter",
                "Differentiate", "Integrate")
POINTS = ("Insert Points", "Delete Points", "Delete NaNs", "Redimension",
          "Reverse", "Rotate")
XSCALE = ("Rescale X", "Set X Start", "Resample", "Interpolate", "Align")
# Operations that change only the x scale (start, delta, units), not the
# values: undone by running them again (e.g. s back to ms), so not
# "permanent", nor asked about
SCALE_ONLY = ("Rescale X", "Set X Start", "Align")
# Points operations drawn on the Plot (#559): Delete Points' points, a range
# (first point, number of points); Insert Points' place, a line (its point)
POINT_RANGES = {
    "Delete Points": ("index", "n_points", "#dc2626"),
    "Insert Points": ("index", None, "#16a34a"),
}
# Outputs that can be plotted with the arrays they were made from behind them
# (after MN_, if outputs are prefixed: Settings › Main)
BACKGROUND_PREFIXES = ("Avg_", "Min_", "Max_")


def output_kind(name: str) -> tuple[str, str]:
    """An output's name as (its MN_ prefix or "", the rest), e.g.
    ("MN_", "Avg_RecordA"), or ("", "Avg_RecordA") unprefixed."""
    prefix = NMToolMain.OUTPUT_PREFIX
    if name.startswith(prefix):
        return prefix, name[len(prefix):]
    return "", name


def light(color: str, amount: float = 0.55) -> str:
    """*color* mixed with white (*amount* of white): the data behind an
    average, so it stands out on top."""
    c = QtGui.QColor(color)
    mix = [round(v + (255 - v) * amount) for v in (c.red(), c.green(), c.blue())]
    return QtGui.QColor(*mix).name()


POINTWISE_FUNCTIONS = ("Average", "Sum", "SumSqr", "Min", "Max")
# Main1's list: its groups, in order, divided: Display; the functions, which
# make MN_ arrays shown in Main2; the Points operations, which change an
# array's points (#559); the other operations
LIST_GROUPS = (
    tuple(DISPLAY),
    POINTWISE_FUNCTIONS,  # Average... point by point across the arrays
    tuple(name for name in FUNCTIONS if name not in POINTWISE_FUNCTIONS),
    POINTS,
    XSCALE,
    tuple(name for name in OPERATIONS if name not in POINTS + XSCALE),
)

def entry_params(name: str) -> tuple:
    """An entry's parameters (Plot's options, an NMMainOp's PARAMS)."""
    if name in DISPLAY:
        return DISPLAY[name][1]
    return (FUNCTIONS.get(name) or OPERATIONS[name])[1].PARAMS


def make_function(name: str, values: dict[str, Any], ignore_nans: bool = True):
    """Function or operation *name*'s NMMainOp from its form's *values*, and
    Ignore NaNs (Settings › Main) if it has a use for it."""
    import inspect
    op_class = (FUNCTIONS.get(name) or OPERATIONS[name])[1]
    kwargs = op_class.form_keywords(keywords(op_class.PARAMS, values))
    if (op_class is not NMMainOpConcatenate
            and "ignore_nans" in inspect.signature(op_class.__init__).parameters):
        kwargs["ignore_nans"] = ignore_nans
    return op_class(**kwargs)


# In x units but not one x position per array, so no use to Align: the
# Spike tool's mean ISI (a duration), all spike times (raster), first spike
NOT_X_POSITIONS = ("SP_isi", "SP_RX", "SP_first")


def is_same_length(data: NMData | None, size: int) -> bool:
    """*data* can be applied point by point to arrays of *size* points:
    numbers, 1-D, as many points (Arithmetic by Array, #563)."""
    values = getattr(data, "nparray", None)
    return (isinstance(values, np.ndarray) and values.ndim == 1
            and values.dtype.kind in "biuf" and values.size == size)


def is_x_values(data: NMData | None, xunits: str) -> bool:
    """*data* looks like x values for Align (#561): numbers in the selected
    arrays' x units (e.g. ms), e.g. Stats2's ST_w0_..._x; not a duration
    (..._dx, e.g. a rise time). Without x units: any numbers but a duration.
    A Stats ..._x array without units (saved before #561) is listed too."""
    values = getattr(data, "nparray", None)
    if (not isinstance(values, np.ndarray) or values.ndim != 1 or values.size == 0
            or values.dtype.kind not in "biuf" or data.name.endswith("_dx")
            or data.name in NOT_X_POSITIONS):
        return False
    units = data.yscale.units
    return not xunits or units == xunits or (not units and data.name.endswith("_x"))


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
            "Plot selected array")
        self.table_action.setToolTip(
            "List subfolder arrays in the Table panel")
        self.refresh()

    def _add_plot_options(self, menu: QtWidgets.QMenu) -> None:
        # An average (or Min, Max) over the arrays it was made from
        self.background_action = menu.addAction("Plot with its data")
        self.background_action.setCheckable(True)
        self.background_action.setChecked(True)
        self.background_action.setToolTip(
            "Plot Average, Min or Max overlaid with their input data")
        self.background_action.toggled.connect(lambda _on: self.array_changed.emit())
        # An average with a band of its spread: ± Stdv (or from Var) or ± SEM;
        # one or neither
        self.error_actions: dict[str, QtGui.QAction] = {}
        for kind, label in (("stdv", "Plot ± Stdv"), ("sem", "Plot ± SEM")):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setToolTip(
                f"Plot Average ± {label[7:]} with a shaded band")
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
        if folder is None or not name:
            return None
        prefix, rest = output_kind(name)
        if not rest.startswith("Avg_"):
            return None
        def companion(kind_prefix):  # e.g. MN_SEM_RecordA, or SEM_RecordA
            return folder.data.get(prefix + rest.replace("Avg_", kind_prefix, 1))
        if kind == "sem":
            sem = companion("SEM_")
            return sem.nparray if sem is not None else None
        stdv = companion("Stdv_")
        if stdv is not None:
            return stdv.nparray
        var = companion("Var_")
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

    def _per_array_series(self) -> dict[str, Any]:
        """The subfolder's data series of an output per array (e.g. IQ_Record:
        IQ_RecordA0, IQ_RecordA1...), listed as one, as Spike2's SPK_ (#563);
        its arrays: in the strip (the subfolder, then the series)."""
        folder = self.toolfolder()
        if folder is None:
            return {}
        per_array = {}
        for name, series in folder.dataseries.items():
            arrays = [d for c in series.channels.values() for d in c.data]
            if arrays and all((nmu.parse_data_name(d.name) or ("",))[0] == name
                              for d in arrays):
                per_array[name] = series
        return per_array

    def _array_names(self) -> list[str]:
        folder = self.toolfolder()
        if folder is None:
            return []
        series = self._per_array_series()
        in_series = {d.name for s in series.values()
                     for c in s.channels.values() for d in c.data}
        names = []
        for name in folder.data.keys():
            if name not in in_series:
                names.append(name)
            else:
                parsed = nmu.parse_data_name(name)
                if parsed is not None and parsed[0] not in names:
                    names.append(parsed[0])  # the series, where its first array was
        return names

    def listed_name(self, name: str) -> str:
        """The entry output *name* is listed under: its own, or its series'."""
        parsed = nmu.parse_data_name(name)
        if parsed is not None and parsed[0] in self._per_array_series():
            return parsed[0]
        return name

    def chosen_data(self) -> NMData | None:
        folder, name = self.toolfolder(), self.selected_array()
        if folder is None or not name:
            return None
        series = self._per_array_series().get(name)
        if series is not None:  # its first array: plotted with all the series'
            return next((d for c in series.channels.values() for d in c.data), None)
        return folder.data.get(name)

    def _background_of(self, name: str | None) -> list[NMData]:
        """The arrays output *name* was made from, if it's an average (or
        Min, Max) and they're still there."""
        folder = self.toolfolder()
        if (folder is None or not name
                or not output_kind(name)[1].startswith(BACKGROUND_PREFIXES)):
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

    _CONFIG_LABELS = {"ignore_nans": "Ignore NaNs", "overwrite": "Overwrite results",
                      "ask_before_changing": "Ask before changing arrays",
                      "prefix_outputs": "Prefix outputs with MN_"}
    _CONFIG_TIPS = {
        "ignore_nans": "Functions ignore NaNs (otherwise a NaN input results "
                       "in a NaN output)",
        "ask_before_changing": "Ask before executing permanent operations on selected data (Baseline, Smooth...) ",
        "prefix_outputs": "Function output arrays start with MN_ (e.g. "
                          "MN_Avg_RecordA); off: Avg_RecordA",
        "overwrite": "For a given selection, functions overwrite their Main "
                     "subfolder (on), or create a new subfolder for each run (off)",
    }
    RANGE_COLOR = "#2563eb"  # as Stats' and Spike's

    def __init__(self, manager: NMManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Main", manager, parent)
        self._tool = manager.main  # nm.main

        self.mode_tabs = QtWidgets.QTabWidget(self)
        self.mode_tabs.setObjectName("mainModeTabs")
        self.use_tabs_for_parameters(self.mode_tabs)

        # Main1: Display, then Functions
        self.main1_page = QtWidgets.QWidget(self.mode_tabs)
        main1 = QtWidgets.QVBoxLayout(self.main1_page)
        main1.setContentsMargins(6, 8, 6, 8)
        # One list, as Stats' measures: Display, then Functions (then
        # Operations...), divided
        function_row = QtWidgets.QHBoxLayout()
        self.function_combo = QtWidgets.QComboBox(self.main1_page)
        self.function_combo.setObjectName("mainFunction")
        self.function_combo.addItem("Choose…", None)  # nothing yet, as Stats1's
        for group in LIST_GROUPS:
            self.function_combo.insertSeparator(self.function_combo.count())
            for name in group:
                what = (DISPLAY.get(name) or FUNCTIONS.get(name) or OPERATIONS[name])[0]
                self.function_combo.addItem(name, name)
                self.function_combo.setItemData(
                    self.function_combo.count() - 1,
                    what[:1].upper() + what[1:],  # beside its name: not repeated
                    QtCore.Qt.ItemDataRole.ToolTipRole)
        self.function_combo.setToolTip(
            "1) Choose function, 2) Set function parameters, 3) Run function on selected arrays")
        function_row.addWidget(self.function_combo)
        self.reset_button = QtWidgets.QPushButton("Reset", self.main1_page)
        self.reset_button.setToolTip("Reset function parameters to default values")
        self.reset_button.clicked.connect(lambda: self.form() and self.form().reset())
        function_row.addWidget(self.reset_button)
        function_row.addStretch(1)
        main1.addLayout(function_row)
        # What it makes, then its parameters: a form each, from their
        # descriptions (#555), kept as set
        self.function_info = QtWidgets.QLabel("", self.main1_page)
        self.function_info.setWordWrap(True)
        main1.addWidget(self.function_info)
        self.forms: dict[str, ParamForm] = {}
        self.form_stack = QtWidgets.QStackedWidget(self.main1_page)
        self.no_form = QtWidgets.QWidget(self.form_stack)  # "Choose…": nothing
        self.form_stack.addWidget(self.no_form)
        for name in list(DISPLAY) + list(FUNCTIONS) + list(OPERATIONS):
            form = ParamForm(entry_params(name), self.form_stack)
            form.ranges_changed.connect(self.ranges_changed)
            if name in POINT_RANGES:  # typed point numbers: the Plot's lines follow
                form.changed.connect(self.ranges_changed)
            if name == "Inequality":  # its a and b on the Plot follow
                form.changed.connect(self.levels_changed)
            if name == "Resample":  # the number of points after, as typed
                form.changed.connect(self._show_resample_note)
            self.forms[name] = form
            self.form_stack.addWidget(form)
        main1.addWidget(self.form_stack)
        main1.addStretch(1)
        self.mode_tabs.addTab(self.main1_page, "Main1")

        # Main2: the functions' outputs
        self.main2 = Main2Panel(manager, self.mode_tabs)
        self.main2_page = self.main2
        self.mode_tabs.addTab(self.main2, "Main2")

        # In the app window: the plot and table (Display's, or Main2's)
        self.plot_view = SelectionPlot()
        self.table_view = SelectionTable()
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.main2.results_changed.connect(self.results_changed)
        self.main2.outputs_plot_requested.connect(self._plot_outputs)
        self.main2.table_requested.connect(self._table_outputs)
        self.main2.array_changed.connect(self._on_output_chosen)
        self.run_finished.connect(self._on_run_finished)
        self.function_combo.currentIndexChanged.connect(self._on_function_changed)
        self.results_group.hide()
        self._on_function_changed()

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

    def set_folder_actions(self, actions) -> None:
        self.main2.set_folder_actions(actions)  # its Folder ⋯ menu

    def set_settings(self, settings) -> None:
        super().set_settings(settings)
        # One color starts as the Plot's trace color (Settings › Plot); an
        # average's data behind it, in the overlay color
        self.plot_view.options.color = getattr(settings, "plot_trace_color", "#000000")
        self.forms["Plot"].set_values({"color": self.plot_view.options.color})
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

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        super()._on_selection_changed(selection)
        self._show_x_scale()

    def _show_x_scale(self) -> None:
        """The forms' points in x (e.g. Rotate by 10 points "(= 5 ms)"), by
        the first selected array's x scale (#559)."""
        try:
            targets = self._selected_targets()
        except RuntimeError:
            targets = []
        arrays = [t["data"] for t in targets if t.get("data") is not None]
        self._selected_arrays = arrays
        data = arrays[0] if arrays else None
        scale = XScale.of(data) if data is not None else None
        # Redimension: the arrays' number of points now
        sizes = sorted({int(d.nparray.size) for d in arrays
                        if hasattr(d.nparray, "size")})
        if "Redimension" in getattr(self, "forms", {}):
            self.forms["Redimension"].set_note("n_points", (
                "" if not sizes else f"currently {sizes[0]} points" if len(sizes) == 1
                else f"currently {sizes[0]} to {sizes[-1]} points"))
        self._x_scale = scale
        for form in getattr(self, "forms", {}).values():
            form.set_x_scale(scale)
        if hasattr(self, "forms"):
            self._show_xscale_forms()
            self.ranges_changed.emit()  # Delete / Insert Points' x follow

    def _show_xscale_forms(self) -> None:
        """The X scale operations' forms (#561): the arrays Align and
        Interpolate can pick (the folder's and its subfolders', e.g. Stats2
        outputs), and what the selected arrays have now."""
        folder = self.selection.get("folder")
        names = array_names(folder)
        arrays = getattr(self, "_selected_arrays", [])
        xunits = arrays[0].xscale.units if arrays else ""
        self.forms["Align"].set_array_names("values_array", [
            n for n in names if is_x_values(find_array(folder, n), xunits)])
        self.forms["Interpolate"].set_array_names("template_name", names)
        # Arithmetic by Array: only arrays as long as the selected ones (#563)
        sizes = sorted({int(d.nparray.size) for d in arrays
                        if hasattr(d.nparray, "size")})
        same = ([n for n in names if is_same_length(find_array(folder, n), sizes[0])]
                if len(sizes) == 1 else [])
        by_array = self.forms["Arithmetic by Array"]
        by_array.set_array_names("ref", same)
        by_array.set_note("ref", (
            "" if not sizes else f"arrays of {sizes[0]} points" if len(sizes) == 1
            else "the selected arrays differ in length: none can be used"))
        xunits = sorted({d.xscale.units or "" for d in arrays})  # Set X Start's
        self.forms["Set X Start"].set_note(
            "value", xunits[0] if len(xunits) == 1 else "(mixed units)" if xunits else "")
        for name, scale in (("Rescale X", "xscale"), ("Rescale Y", "yscale")):
            units = sorted({getattr(d, scale).units or "" for d in arrays})
            self.forms[name].set_note("from_units", (
                "" if not units
                else "currently " + ", ".join(u or "(none)" for u in units)))
        self._show_resample_note()

    def _show_resample_note(self) -> None:
        """Resample: the arrays' interval and number of points now, and
        after (as typed)."""
        arrays = [d for d in getattr(self, "_selected_arrays", [])
                  if hasattr(d.nparray, "size")]
        form, text = self.forms["Resample"], ""
        if arrays:
            first = arrays[0]
            deltas = sorted({d.xscale.delta for d in arrays})
            sizes = sorted({int(d.nparray.size) for d in arrays})
            units = f" {first.xscale.units}" if first.xscale.units else ""
            text = "currently " + (f"{deltas[0]:g}{units}" if len(deltas) == 1
                                   else "mixed intervals")
            text += (f", {sizes[0]} points" if len(sizes) == 1
                     else f", {sizes[0]} to {sizes[-1]} points")
            try:
                new = float(form.control("delta").text())
            except ValueError:
                new = 0.0
            if new > 0 and len(deltas) == 1 and len(sizes) == 1:
                text += f"; then about {round(sizes[0] * deltas[0] / new)}"
        form.set_note("delta", text)

    def refresh(self) -> None:
        """The data changed (e.g. arrays deleted or edited): drawn again."""
        super().refresh()
        self._show_x_scale()
        self.main2.refresh()
        for view in (self.plot_view, self.table_view):
            if view.shown is not None:
                view.show_selected(self._still_there(view.shown))

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
        self.display_row.setVisible(main1 and self._settings is not None
                                    and (self.entry() in FUNCTIONS
                                         or self.entry() in OPERATIONS))
        self.view_requested.emit(self.preferred_view())

    # --- Display: the selection -----------------------------------------------

    def plot_selection(self) -> None:
        """Plot the selection in the Main2 tab, as Plot's form says, and show it."""
        try:
            selected = selected_arrays(self.selection)
        except RuntimeError as error:
            self.show_warning(str(error))
            return
        try:
            options = plot_options(self.forms["Plot"].values())
        except ValueError as error:  # e.g. an offset that isn't a number
            self.show_warning(str(error))
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
            self.show_warning(str(error))
            return
        self.table_view.show_selected(selected)
        self.results_changed.emit()
        self.view_requested.emit("table")

    # --- the function chosen: its form, and its x ranges on the Plot -----------

    def entry(self) -> str | None:
        """The entry chosen in Main1's list (e.g. "Plot", "Average"); None
        for "Choose…"."""
        return self.function_combo.currentData()

    def form(self) -> ParamForm | None:
        """The chosen entry's form (None: nothing chosen)."""
        name = self.entry()
        return self.forms[name] if name is not None else None

    def _on_function_changed(self, *_args) -> None:
        name = self.entry()
        if name in XSCALE + ("Arithmetic by Array",):  # its pickers: the arrays now
            self._show_xscale_forms()
        self.form_stack.setCurrentWidget(self.forms[name] if name else self.no_form)
        self.reset_button.setEnabled(name is not None)
        if name is None:
            self.function_info.setText(
                "Choose function: Plot, Average, Baseline, Filter…")
        elif name in DISPLAY:
            self.function_info.setText(DISPLAY[name][0])
        elif name in FUNCTIONS:
            self.function_info.setText(f"{FUNCTIONS[name][0]}.")
        else:
            info = (f"X scale only: {OPERATIONS[name][0]}. The values aren't changed."
                    if name in SCALE_ONLY
                    else f"Permanent operation: {OPERATIONS[name][0]}.")
            if name in ALSO_VIRTUAL:
                info += " See channel Filter & Transform to apply virtual operations."
            self.function_info.setText(info)
        self._on_mode_tab_changed(self.mode_tabs.currentIndex())  # Progress: functions'
        self.ranges_changed.emit()  # its x ranges on the Plot
        self.levels_changed.emit()  # Inequality's levels

    def plot_ranges(self) -> list[PlotRange]:
        # The chosen function's x ranges (draggable, locked or hidden: the
        # Plot's ⋯ › Tool Ranges & Levels)
        name, form = self.entry(), self.form()
        if form is None:
            return []
        ranges = [PlotRange(param.name,
                            param.label if param.label.startswith(name)
                            else f"{name} {param.label}",  # e.g. "Average X range"
                            begin, end, param.color)
                  for param, (begin, end) in form.x_ranges()]
        points = self._points_range(name, form)
        return ranges + ([points] if points is not None else [])

    def _points_range(self, name: str, form: ParamForm) -> PlotRange | None:
        """Delete Points' points as a range, or Insert Points' place as a
        line, by the first selected array's x scale (#559)."""
        scale = getattr(self, "_x_scale", None)
        if name not in POINT_RANGES or scale is None:
            return None
        at, count, color = POINT_RANGES[name]
        index = form.int_value(at)
        if index is None or index < 0:
            return None
        if count is None:  # a line: where the points go
            x = scale.x(min(index, scale.points or index), "position")
            return None if x is None else PlotRange(
                "points", f"{name} at point {index}", x, x, color, line=True)
        n = form.int_value(count)
        last = index + n - 1 if n is not None and n >= 1 else None
        if last is None or (scale.points is not None and index >= scale.points):
            return None
        if scale.points is not None:
            last = min(last, scale.points - 1)
        bgn, end = scale.x(index, "position"), scale.x(last, "position")
        if bgn is None or end is None:
            return None
        return PlotRange("points", f"{name} {index} to {last}", bgn, end, color)

    INEQUALITY_COLOR = "#7c3aed"

    def plot_levels(self, data) -> list[PlotLevel]:
        """Inequality's a (and b, for a range) as levels on the Plot (#563)."""
        if self.entry() != "Inequality":
            return []
        form = self.form()
        try:
            values = form.values()
        except ValueError:
            return []
        levels = [PlotLevel("a", "Inequality a", values["a"], self.INEQUALITY_COLOR)]
        if form._rows["b"][1].isEnabled():  # a range
            levels.append(PlotLevel("b", "Inequality b", values["b"],
                                    self.INEQUALITY_COLOR))
        return levels

    def set_plot_level(self, key: str, y: float, final: bool) -> None:
        """A dragged level: its box follows."""
        if self.entry() == "Inequality" and key in ("a", "b"):
            self.form().set_values({key: y})

    def set_plot_range(self, key: str, xbgn: float, xend: float, final: bool) -> None:
        """A dragged range: its boxes follow (Delete / Insert Points': its
        point numbers)."""
        form, name = self.form(), self.entry()
        if form is None:
            return
        scale = getattr(self, "_x_scale", None)
        if key == "points" and name in POINT_RANGES and scale is not None:
            at, count, _color = POINT_RANGES[name]
            first = scale.index(xbgn)
            values = {at: first}
            if count is not None:
                values[count] = max(scale.index(xend) - first + 1, 1)
            form.set_values(values)
            return
        form.set_x_range(key, xbgn, xend)

    # --- Functions: run on the selection, all channels at once -----------------

    def _on_run_clicked(self) -> None:
        name = self.entry()
        if not self._running and name is None:
            self.show_warning("Choose function: Plot, Average, Baseline, Filter…")
            return
        if not self._running and name in DISPLAY:  # Plot, Table: drawn, no run
            self.plot_selection() if name == "Plot" else self.table_selection()
            return
        if not self._running:  # first, the function, with its form's values
            try:
                values = self.form().values()
                op = make_function(name, values, ignore_nans=self._tool.ignore_nans)
            except ValueError as error:  # e.g. an x range's begin after its end
                self.show_warning(str(error))
                return
            if op.CHANGES_DATA and not self._may_change(name):
                return
            self._changing = self._run_targets() if op.CHANGES_DATA else []
            self._tool.op = op  # nm.main.op = ...
        super()._on_run_clicked()

    def _all_targets(self) -> list[dict]:
        """Every array the run would use (all of All Sets' runs too)."""
        planned = self._run_batches()
        if planned is not None:
            return [t for _label, targets, _sel in planned[0] for t in targets]
        return self._selected_targets()

    def _run_targets(self) -> list[dict]:
        """The arrays an operation is about to change (none: the run says why)."""
        try:
            return self._all_targets()
        except RuntimeError:
            return []

    def _may_change(self, name: str) -> bool:
        """Before an operation: not a locked series' arrays (refused, as nm.main
        would); then, with Settings › Main's "Ask before changing arrays",
        asked first."""
        try:
            targets = self._all_targets()
        except RuntimeError:
            return True  # the run says why
        series = self._tool.locked_series(targets)
        if series is not None:
            self.show_warning(
                f"{series.name} is locked: unlock it, or copy it and run on the copy")
            return False
        if name in ("Rescale X", "Set X Start"):  # an x array: refused (#561)
            with_x = next((t["data"] for t in targets if t.get("data") is not None
                           and t["data"].xarray is not None), None)
            if with_x is not None:
                self.show_warning(
                    f"{with_x.name} has an x array ({with_x.xarray_name or 'its own'}): "
                    f"{name} changes start and delta, not its values; change "
                    "the x array itself")
                return False
        x_warning = self._xarray_warning(name, targets) + self._channels_warning(name)
        ask = self._tool.ask_before_changing and name not in SCALE_ONLY
        if not targets or (not ask and not x_warning):
            return True
        what = self.selection.get("dataseries")
        source = what.name if what is not None and self.selection.get(
            "source") == "dataseries" else "the selection"
        n = len(targets)
        return self._confirm(
            name, f"{name} changes {n} array{'s' * (n != 1)} of {source}.{x_warning} "
            "Continue?")

    def _channels_warning(self, name: str) -> str:
        """Rescale X or Align on some of a data series' channels: the others
        keep their x, so the channels no longer share a time base (#561);
        asked first, whatever Settings. (Resample and Interpolate keep each
        value at its x: no need.)"""
        selection = self.selection
        series, channel = selection.get("dataseries"), selection.get("channel")
        if (name not in SCALE_ONLY or selection.get("source") != "dataseries"
                or series is None or channel is None or selection.get("all_channels")):
            return ""
        others = [c for c in series.channels.keys() if c != channel.name]
        if not others:
            return ""
        them = (f"Channel {others[0]} isn't" if len(others) == 1
                else f"Channels {', '.join(others)} aren't")
        change = ("its x units will differ from" if name == "Rescale X"
                  else "its epochs will no longer line up in time with")
        if len(others) > 1:
            change = change.replace("its", "their")
        return (f" {them} changed, so {change} {channel.name}'s "
                "(choose All channels to change them together).")

    @staticmethod
    def _xarray_warning(name: str, targets: list[dict]) -> str:
        """A Points operation on arrays with an x array: their x values aren't
        changed with their points (#559); asked first, whatever Settings."""
        if name not in POINTS and name != "Resample":
            return ""
        names = {d.xarray_name or "its own x values"
                 for d in (t.get("data") for t in targets)
                 if d is not None and d.xarray is not None}
        if not names:
            return ""
        what = (f"Their x array ({names.pop()}) isn't" if len(names) == 1
                else "Their x arrays aren't")
        if name in ("Reverse", "Rotate"):
            return f" {what} changed: each point will be at a different x."
        return (f" {what} changed: it won't fit their new number of points, so "
                "x values will come from the x scale's start and delta.")

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
            if outputs:  # (one per array: their series, e.g. IQ_Record)
                self.main2.select_array(self.main2.listed_name(outputs[0]))

    def _on_run_finished(self, succeeded: bool) -> None:
        self._show_x_scale()  # e.g. Delete Points: the arrays' new lengths
        # An operation that changed the arrays' points (e.g. Delete Points,
        # #559): an x array that no longer fits is said (start and delta
        # are used until it's fixed)
        skipped = getattr(self._tool.op, "skipped", None) if succeeded else None
        if skipped:  # Align: no reference value (NaN)
            more = f" (and {len(skipped) - 1} more)" if len(skipped) > 1 else ""
            self.show_warning(f"{skipped[0]}{more} not aligned: its reference "
                              "value is NaN")
        changed, self._changing = getattr(self, "_changing", []), []
        unfit = self._tool.unfit_xarrays(changed) if succeeded else []
        if unfit:
            more = f" (and {len(unfit) - 1} more)" if len(unfit) > 1 else ""
            self.show_warning(
                f"The x array of {unfit[0].name}{more} no longer fits its points; "
                "x values come from the x scale's start and delta")
        # After the Plot is back as it was before the run: Main2, the run's
        # first output plotted, as Stats2 and Spike2 show theirs
        if succeeded and self._tool.toolfolder is not None:
            self.mode_tabs.setCurrentWidget(self.main2)
            self.main2.plot_selected()

    # --- Main2: the outputs ---------------------------------------------------

    def _output_options(self, selected: Selected) -> PlotOptions:
        """Outputs aren't offset; overlaid (e.g. IQ_Record's arrays), in a
        rainbow (as Plot's, reversed if it is), else in the Plot's color."""
        overlaid = any(len(g.arrays) > 1 for g in selected.groups)
        return PlotOptions(color=self.plot_view.options.color, rainbow=overlaid,
                           reverse_colors=self.plot_view.options.reverse_colors)

    def _show_outputs(self) -> bool:
        """The chosen output plotted, and the subfolder's outputs in the
        table (the chosen one's column selected): the Plot | Table toggle
        and Main2's Plot and Table show the same. False: none chosen."""
        selected = self.main2.selected_outputs()
        if selected is None:
            return False
        self.plot_view.show_selected(selected, self._output_options(selected))
        outputs = self.main2.all_outputs()
        if outputs is not None:
            self.table_view.show_selected(outputs)
            self.table_view.select_array(self.main2.chosen_data())
        self.results_changed.emit()
        return True

    def _plot_outputs(self, _selected: Selected | None = None) -> None:
        if self._show_outputs():
            self.view_requested.emit("plot")

    def _on_output_chosen(self) -> None:
        """The plot and table follow the output chosen in Main2 (while it's shown)."""
        if self.mode_tabs.currentWidget() is self.main2:
            self._show_outputs()

    def _table_outputs(self) -> None:
        if self._show_outputs():
            self.view_requested.emit("table")
