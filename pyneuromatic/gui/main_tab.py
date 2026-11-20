"""The Main tool's tab (#550): Main1 (its operations, to come) and Main2,
which plots and tables the strip's selection.

Main2's Plot draws every selected epoch: each channel in a plot of its own,
one above another, the epochs overlaid. Its Table lists their values, an x
column, then a column per array. Both are drawn in the Main2 tab of the Plot
and Table panels, as they were when Plot or Table was clicked: the title says
what they show, and clicking again draws the selection then.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.context_views import ChannelTableModel, ChannelTable
from pyneuromatic.gui.selection_model import epoch_scope_label, scoped_epoch_names
from pyneuromatic.gui.tool_run import SelectionRunTab
from pyneuromatic.gui.tool_tab import fit_number_edit

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
        if pg is not None:
            self.graphics = pg.GraphicsLayoutWidget(self)
            self.graphics.setBackground("w")
            layout.addWidget(self.graphics, stretch=1)
        else:  # pragma: no cover - optional GUI dependency
            self.graphics = None
            layout.addWidget(QtWidgets.QLabel("Install pyqtgraph to enable plots.", self))
        self.shown: Selected | None = None

    def set_options(self, options: PlotOptions) -> None:
        """Draw with *options*: the plot shown again, as it was selected."""
        self.options = options
        if self.shown is not None:
            self.show_selected(self.shown)

    def _lines(self, group: ChannelArrays) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """(x, y, colour) of the lines drawing *group*'s epochs: each epoch
        offset by its number in the group times the offsets; one line per
        colour, the epochs separated by NaN."""
        options = self.options
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

    def show_selected(self, selected: Selected) -> None:
        self.shown = selected
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


class MainToolTab(SelectionRunTab):
    """Main1: the operations (to come, #550). Main2: plot or table the
    strip's selection."""

    # Its tab in the Plot and Table panels shows Main2's plot and table
    view_label = "Main2"

    def __init__(self, manager: NMManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Main", manager, parent)
        self._tool = manager.main  # nm.main: its operations, in Main1 (to come)

        self.mode_tabs = QtWidgets.QTabWidget(self)
        self.mode_tabs.setObjectName("mainModeTabs")
        self.use_tabs_for_parameters(self.mode_tabs)

        self.main1_page = QtWidgets.QWidget(self.mode_tabs)
        main1_layout = QtWidgets.QVBoxLayout(self.main1_page)
        main1_layout.setContentsMargins(6, 8, 6, 8)
        self.main1_note = QtWidgets.QLabel(
            "Functions, Operations, Points and X scale will be here (#550). "
            "To plot or table the selection, use Main2.", self.main1_page)
        self.main1_note.setWordWrap(True)
        main1_layout.addWidget(self.main1_note)
        main1_layout.addStretch(1)
        self.mode_tabs.addTab(self.main1_page, "Main1")

        self.main2_page = QtWidgets.QWidget(self.mode_tabs)
        main2_layout = QtWidgets.QVBoxLayout(self.main2_page)
        main2_layout.setContentsMargins(6, 8, 6, 8)
        heading = QtWidgets.QLabel("Display", self.main2_page)
        heading.setStyleSheet("font-weight: bold;")
        main2_layout.addWidget(heading)
        # What Plot and Table will show (the selection, as it changes)
        self.selection_label = QtWidgets.QLabel("", self.main2_page)
        self.selection_label.setWordWrap(True)
        main2_layout.addWidget(self.selection_label)
        buttons = QtWidgets.QHBoxLayout()
        self.plot_button = QtWidgets.QPushButton("Plot…", self.main2_page)
        self.plot_button.setToolTip(
            "Plot every selected epoch: each channel in a plot of its own, one above "
            "another, the epochs overlaid or offset, in a color or a rainbow")
        self.table_button = QtWidgets.QPushButton("Table", self.main2_page)
        self.table_button.setToolTip(
            "List the selected arrays' values: an x column, then a column per array")
        for button in (self.plot_button, self.table_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        main2_layout.addLayout(buttons)

        main2_layout.addStretch(1)
        self.mode_tabs.addTab(self.main2_page, "Main2")

        # In the app window: Main2's plot and table
        self.plot_view = SelectionPlot()
        self.table_view = SelectionTable()
        self.plot_button.clicked.connect(self.plot_selection)
        self.table_button.clicked.connect(self.table_selection)
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        # Nothing to run yet: the operations come next (#550)
        self.run_row.hide()
        self.display_row.hide()
        self.results_group.hide()
        self.status_label.hide()
        self._show_selection()

    # --- the app window -------------------------------------------------------

    def preferred_view(self) -> str:
        return "tool" if self.mode_tabs.currentWidget() is self.main2_page else "channel"

    def lists_view(self) -> bool:
        return True  # Main2's tab, to switch to

    def has_results(self) -> bool:
        return self.plot_view.shown is not None or self.table_view.shown is not None

    def bind_selection_model(self, model: Any) -> None:
        super().bind_selection_model(model)
        self._show_selection()

    def set_settings(self, settings) -> None:
        super().set_settings(settings)
        self.display_row.hide()  # nothing to run yet
        # One color starts as the Plot's trace color (Settings › Plot)
        self.plot_view.options.color = getattr(settings, "plot_trace_color", "#000000")

    def refresh(self) -> None:
        """The data changed (e.g. arrays deleted or edited): drawn again."""
        super().refresh()
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
        groups = [ChannelArrays(g.label, [d for d in g.arrays if kept(d)], g.channel)
                  for g in selected.groups]
        return Selected(selected.title, groups)

    def _ask_plot_options(self, selected: Selected) -> PlotOptions | None:
        """Plot…'s prompt: how to draw *selected* (None: cancelled); its
        answer is kept for next time. Replaced in tests."""
        dialog = PlotOptionsDialog(self.plot_view.options, selected.title, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.options()

    # --- Main2 ----------------------------------------------------------------

    def _on_mode_tab_changed(self, _index: int) -> None:
        self.view_requested.emit(self.preferred_view())

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        super()._on_selection_changed(selection)
        self.run_row.hide()  # nothing to run yet
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

    def plot_selection(self) -> None:
        """Plot the selection in Main2's tab, and show it."""
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
        """List the selection's values in Main2's table, and show it."""
        try:
            selected = selected_arrays(self.selection)
        except RuntimeError as error:
            self.selection_label.setText(str(error))
            return
        self.table_view.show_selected(selected)
        self.results_changed.emit()
        self.view_requested.emit("table")
