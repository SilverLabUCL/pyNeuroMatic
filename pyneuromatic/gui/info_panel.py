"""Information about one object, for the browser's second column.

A folder, data array, data series, channel, epoch or results folder is shown
as a short form (created date, sizes, scales, which sets it is in...) with
its notes, if it keeps any, and an Add Note... button.
"""
from __future__ import annotations

import numpy

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_math import array_stats
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.gui.transform_dialog import MARK_COLOR, MENU_TEXT, step_text

LOCKED_COLOR = MARK_COLOR  # the app's red (as SetX and the ƒ mark)

# The objects that get an info panel (group rows keep the list instead)
INFO_TYPES = (NMFolder, NMToolFolder, NMData, NMDataSeries, NMChannel, NMEpoch)

_KIND = {
    NMFolder: "Folder",
    NMToolFolder: "Results Folder",
    NMData: "Data Array",
    NMDataSeries: "Data Series",
    NMChannel: "Channel",
    NMEpoch: "Epoch",
}


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _names(names, limit: int = 8) -> str:
    """e.g. "Set1, Set2" or "E0, E1, ... (12)"; "—" for none."""
    names = list(names)
    if not names:
        return "—"
    if len(names) > limit:
        return ", ".join(names[:limit]) + f", ... ({len(names)})"
    return ", ".join(names)


def _text(value) -> str:
    """A label or units, or "—" when empty."""
    return str(value) if value else "—"


def _type_text(dtype: numpy.dtype) -> str:
    """e.g. "float64 (double)", "int16 (16-bit integer)", "text"."""
    if dtype.kind in "OUS":
        return "text"
    if dtype.kind == "b":
        return "bool (true/false)"
    bits = dtype.itemsize * 8
    if dtype.kind == "f":
        precision = {32: "single", 64: "double"}.get(bits, f"{bits}-bit float")
        return f"{dtype.name} ({precision})"
    if dtype.kind == "c":
        precision = {64: "single", 128: "double"}.get(bits, f"{bits}-bit")
        return f"{dtype.name} (complex, {precision})"
    if dtype.kind in "iu":
        signed = "" if dtype.kind == "i" else "unsigned "
        return f"{dtype.name} ({bits}-bit {signed}integer)"
    return dtype.name


def _common_text(values, number: bool) -> str:
    """The value all *values* share, "(mixed)" if they differ, "—" if empty."""
    distinct = set(values)
    if len(distinct) > 1:
        return "(mixed)"
    value = distinct.pop() if distinct else None
    if number and value is not None:
        return f"{value:g}"
    return _text(value)


def _value_text(value: float, units: str) -> str:
    """e.g. "-61.2 mV"; "—" for NaN (e.g. the SD of one point)."""
    if value != value:
        return "—"
    return f"{value:g}{units}"


def _sets_of(container, name: str) -> list[str]:
    sets = getattr(container, "sets", None)
    if sets is None:
        return []
    return [set_name for set_name in sets.keys() if sets.contains(set_name, name)]


class InfoPanel(QtWidgets.QScrollArea):
    """Shows one object: a title, a form of its properties and its notes."""

    # A note was added to the object shown
    notes_changed = QtCore.pyqtSignal()
    # Edit Scales... was clicked for this data array or channel
    edit_scales_requested = QtCore.pyqtSignal(object)
    # Filter & Transform... was clicked for this channel
    edit_transforms_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._object: NMObject | None = None
        self._body = QtWidgets.QWidget(self)
        self.setWidget(self._body)
        layout = QtWidgets.QVBoxLayout(self._body)
        layout.setContentsMargins(8, 6, 8, 6)

        self.title_label = QtWidgets.QLabel("", self._body)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        # Data arrays: their details, or basic statistics (as in Igor's browser)
        self.details_button = QtWidgets.QToolButton(self._body)
        self.details_button.setText("Details")
        self.details_button.setToolTip("Points, type, scales, data series and sets")
        self.stats_button = QtWidgets.QToolButton(self._body)
        self.stats_button.setText("Stats")
        self.stats_button.setToolTip("Mean, median, SD, SEM, RMS, min and max")
        self._view_group = QtWidgets.QButtonGroup(self)
        for button in (self.details_button, self.stats_button):
            button.setCheckable(True)
            self._view_group.addButton(button)
        self.details_button.setChecked(True)  # kept while clicking through arrays
        self._view_group.buttonClicked.connect(lambda _button: self.show_object(self._object))
        layout.addWidget(self.title_label)
        # On its own row, so a narrow column never has to scroll sideways
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(0)
        view_row.addWidget(self.details_button)
        view_row.addWidget(self.stats_button)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        self._form = QtWidgets.QFormLayout()
        self._form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addLayout(self._form)

        # Changes go through a dialog, as in NeuroMatic (data arrays, channels)
        self.edit_scales_button = QtWidgets.QPushButton("Edit Scales...", self._body)
        self.edit_scales_button.setToolTip("Change the X start, delta, label and units, "
                                           "and the Y label and units")
        self.edit_scales_button.clicked.connect(
            lambda: self.edit_scales_requested.emit(self._object)
        )
        # A channel's filters and transforms (they don't change its arrays,
        # so also for a locked data series)
        self.transforms_button = QtWidgets.QPushButton(MENU_TEXT, self._body)
        self.transforms_button.setToolTip(
            "The channel's filters and transforms, as plotted and analysed")
        self.transforms_button.clicked.connect(
            lambda: self.edit_transforms_requested.emit(self._object)
        )
        edit_row = QtWidgets.QHBoxLayout()
        edit_row.addWidget(self.edit_scales_button)
        edit_row.addWidget(self.transforms_button)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)

        # Imported metadata (folders), collapsed to one row per source folder
        self.metadata_tree = QtWidgets.QTreeWidget(self._body)
        self.metadata_tree.setHeaderLabels(["Imported Metadata", "Value"])
        self.metadata_tree.setMinimumHeight(80)
        self.metadata_tree.setMaximumHeight(160)
        layout.addWidget(self.metadata_tree)

        self.notes_view = NotesWidget(self._body)
        self.notes_view.notes_changed.connect(self.notes_changed)
        layout.addWidget(self.notes_view)
        layout.addStretch(1)

    @property
    def shown_object(self) -> NMObject | None:
        return self._object

    @property
    def showing_stats(self) -> bool:
        """True when data arrays show statistics rather than their details."""
        return self.stats_button.isChecked()

    # --- showing an object --------------------------------------------------

    def show_object(self, obj: NMObject | None) -> None:
        self._object = obj
        while self._form.rowCount():
            self._form.removeRow(0)
        is_data = isinstance(obj, NMData)
        self.details_button.setVisible(is_data)
        self.stats_button.setVisible(is_data)
        if obj is None:
            self.title_label.setText("")
            self.edit_scales_button.hide()
            self.transforms_button.hide()
            self._show_metadata({})
            self._show_notes(None)
            return
        kind = next((label for cls, label in _KIND.items() if isinstance(obj, cls)), "")
        self.title_label.setText(f"{obj.name}  ·  {kind}" if kind else obj.name)
        self._add_row("Created", str(obj.parameters.get("created") or "—"))
        copy_of = obj.parameters.get("copy of")
        if copy_of:
            self._add_row("Copy of", str(copy_of))
        if isinstance(obj, NMFolder):
            self._show_folder(obj)
        elif isinstance(obj, NMToolFolder):
            self._show_tool_folder(obj)
        elif isinstance(obj, NMData):
            if self.showing_stats:
                self._show_data_stats(obj)
            else:
                self._show_data(obj)
        elif isinstance(obj, NMDataSeries):
            self._show_dataseries(obj)
        elif isinstance(obj, NMChannel):
            self._show_channel(obj)
        elif isinstance(obj, NMEpoch):
            self._show_epoch(obj)
        can_edit = (
            (is_data and not self.showing_stats and self._locked_series(obj) is None)
            or (isinstance(obj, NMChannel) and not obj._parent.locked)
        )
        self.edit_scales_button.setVisible(can_edit)
        self.transforms_button.setVisible(isinstance(obj, NMChannel))
        self._show_metadata(getattr(obj, "metadata", {}) if isinstance(obj, NMFolder) else {})
        self._show_notes(getattr(obj, "notes", None))

    def row_text(self, label: str) -> str | None:
        """The value shown for *label* (e.g. "Points"), or None."""
        for row in range(self._form.rowCount()):
            label_item = self._form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            field_item = self._form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if label_item is not None and label_item.widget().text() == label:
                return field_item.widget().text()
        return None

    def row_labels(self) -> list[str]:
        labels = []
        for row in range(self._form.rowCount()):
            item = self._form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            if item is not None:
                labels.append(item.widget().text())
        return labels

    def _add_row(self, label: str, value: str, color: str | None = None) -> None:
        field = QtWidgets.QLabel(value, self._body)
        if color is not None:  # e.g. red: a locked data series
            field.setStyleSheet(f"color: {color};")
        # Only long values (e.g. lists of names) wrap: a wrapped label in a
        # form can lose its second line
        field.setWordWrap(len(value) > 40)
        field.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        # A label of its own: addRow(text) would take "&" as a shortcut key
        self._form.addRow(QtWidgets.QLabel(label, self._body), field)

    def _add_scale_rows(self, xscale, yscale) -> None:
        """One row each, in the Edit Scales dialog's order, so a long label
        (e.g. "time after peak") is never squeezed."""
        self._add_row("X Start", f"{xscale.start:g}")
        self._add_row("X Delta", f"{xscale.delta:g}")
        self._add_row("X Label", _text(xscale.label))
        self._add_row("X Units", _text(xscale.units))
        self._add_row("Y Label", _text(yscale.label))
        self._add_row("Y Units", _text(yscale.units))

    # --- per object type ---------------------------------------------------

    def _show_folder(self, folder: NMFolder) -> None:
        self._add_row("Data", _count(len(folder.data), "array"))
        self._add_row("Data Series", _names(folder.dataseries.keys()))
        self._add_row("Subfolders", _names(folder.toolfolders.keys()))
        self._add_row("Data Sets", _names(folder.data.sets.keys()))

    def _show_tool_folder(self, folder: NMToolFolder) -> None:
        self._add_row("Data", _count(len(folder.data), "array"))
        self._add_row("Results", _names(folder.data.keys()))
        if len(folder.dataseries):
            self._add_row("Data Series", _names(folder.dataseries.keys()))

    def _show_data(self, data: NMData) -> None:
        array = data.nparray
        if array is None:
            self._add_row("Points", "—")
        else:
            self._add_row("Points", " × ".join(str(n) for n in array.shape))
            self._add_row("Type", _type_text(array.dtype))
            real = (numpy.issubdtype(array.dtype, numpy.integer)
                    or numpy.issubdtype(array.dtype, numpy.floating))
            # No Range when every point is NaN (nanmin would warn)
            if array.size and real and not numpy.isnan(array.astype(float)).all():
                low, high = numpy.nanmin(array), numpy.nanmax(array)
                if numpy.isfinite(low) or numpy.isfinite(high):
                    self._add_row("Range", f"{low:g} to {high:g}")
        self._add_scale_rows(data.xscale, data.yscale)
        container = data._container
        if container is not None and hasattr(container, "dataseries_of"):
            places = []
            for series in container.dataseries_of(data.name):
                channel = next((c.name for c in series.channels.values()
                                if any(d is data for d in c.data)), None)
                epoch = next((e.name for e in series.epochs.values()
                              if any(d is data for d in e.data)), None)
                places.append(" / ".join(p for p in (series.name, channel, epoch) if p))
            self._add_row("Data Series", _names(places))
        self._add_row("Sets", _names(_sets_of(container, data.name)))
        locked = self._locked_series(data)
        if locked is not None:
            self._add_row("Locked", f"Yes, with data series '{locked.name}'", LOCKED_COLOR)

    def _show_data_stats(self, data: NMData) -> None:
        """Basic statistics, as Igor's WaveStats (NaNs ignored)."""
        array = data.nparray
        if array is None or array.size == 0:
            self._add_row("Stats", "No data")
            return
        if array.dtype.kind not in "iuf":
            kind = "text" if array.dtype.kind in "OUS" else _type_text(array.dtype).split()[0]
            self._add_row("Stats", f"No statistics for {kind} data")
            return
        values = array.astype(float).ravel()
        stats = array_stats(values, ignore_nans=True)
        yunits = f" {data.yscale.units}" if data.yscale.units else ""
        self._add_row("Points", f"{array.size}  (NaNs {stats['NaNs']}, INFs {stats['INFs']})")
        if stats["N"] == 0:
            self._add_row("Stats", "Every point is NaN")
            return
        with numpy.errstate(all="ignore"):
            median = float(numpy.nanmedian(values))
        for label, value in (("Mean", stats["mean"]), ("Median", median),
                             ("SD", stats["std"]), ("SEM", stats["sem"]),
                             ("RMS", stats["rms"])):
            self._add_row(label, _value_text(value, yunits))
        for label, value, find in (("Min", stats["min"], numpy.nanargmin),
                                   ("Max", stats["max"], numpy.nanargmax)):
            text = _value_text(value, yunits)
            if array.ndim == 1:  # where it is, like Igor's V_minloc / V_maxloc
                text += f" at {self._x_at(data, int(find(values)))}"
            self._add_row(label, text)

    @staticmethod
    def _x_at(data: NMData, index: int) -> str:
        """The x value of point *index* (from the x array, or the x scale)."""
        xarray = data.xarray
        if xarray is not None and len(xarray) > index:
            x = float(xarray[index])
        else:
            x = data.xscale.start + index * data.xscale.delta
        units = f" {data.xscale.units}" if data.xscale.units else ""
        return f"{x:g}{units}"

    @staticmethod
    def _locked_series(data: NMData):
        """The locked data series holding *data* (its scales can't change)."""
        container = data._container
        if container is None or not hasattr(container, "dataseries_of"):
            return None
        return next((ds for ds in container.dataseries_of(data.name) if ds.locked), None)

    def _show_dataseries(self, series: NMDataSeries) -> None:
        self._add_row("Channels", _names(series.channels.keys()))
        self._add_row("Epochs", str(len(series.epochs)))
        lengths = {len(d.nparray) for c in series.channels.values() for d in c.data
                   if d.nparray is not None}
        if lengths:
            low, high = min(lengths), max(lengths)
            self._add_row("Points", str(low) if low == high else f"{low} to {high}")
        if series.locked:
            self._add_row("Locked", "Yes", LOCKED_COLOR)
        else:
            self._add_row("Locked", "No")
        self._add_row("Filter & Transform", _names(
            name for name, channel in series.channels.items() if channel.transform_on))
        self._add_row("Epoch Sets", _names(series.epochs.sets.keys()))
        self._add_row("Groups", _names(str(n) for n in series.epochs.groups.group_numbers))

    def _show_channel(self, channel: NMChannel) -> None:
        self._add_row("Data", _count(len(channel.data), "array"))
        if not channel.data:
            self._add_scale_rows(channel.xscale, channel.yscale)
        else:  # what its arrays have (the channel's own scale is a template)
            arrays = channel.data
            for label, values, number in (
                ("X Start", [d.xscale.start for d in arrays], True),
                ("X Delta", [d.xscale.delta for d in arrays], True),
                ("X Label", [d.xscale.label for d in arrays], False),
                ("X Units", [d.xscale.units for d in arrays], False),
                ("Y Label", [d.yscale.label for d in arrays], False),
                ("Y Units", [d.yscale.units for d in arrays], False),
            ):
                self._add_row(label, _common_text(values, number))
        steps = [step_text(t) for t in channel.transform if t.on]
        self._add_row("Filter & Transform", "\n".join(
            f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else "—")
        series = channel._parent
        if isinstance(series, NMDataSeries) and series.locked:
            self._add_row("Locked", f"Yes, with data series '{series.name}'", LOCKED_COLOR)

    def _show_epoch(self, epoch: NMEpoch) -> None:
        series = epoch._parent
        arrays = []
        for data in epoch.data:
            channel = None
            if isinstance(series, NMDataSeries):
                channel = next((c.name for c in series.channels.values()
                                if any(d is data for d in c.data)), None)
            arrays.append(f"{channel}: {data.name}" if channel else data.name)
        self._add_row("Data", _names(arrays))
        container = epoch._container
        self._add_row("Sets", _names(_sets_of(container, epoch.name)))
        groups = getattr(container, "groups", None)
        group = groups.get_group(epoch.name) if groups is not None else None
        self._add_row("Group", "—" if group is None else str(group))

    # --- metadata and notes ------------------------------------------------

    def _show_metadata(self, metadata: dict) -> None:
        self.metadata_tree.clear()
        self.metadata_tree.setVisible(bool(metadata))

        def add(parent, key, value):
            if isinstance(value, dict):
                item = QtWidgets.QTreeWidgetItem(parent, [str(key), f"({len(value)})"])
                for child_key, child_value in value.items():
                    add(item, child_key, child_value)
            else:
                QtWidgets.QTreeWidgetItem(parent, [str(key), str(value)])

        for key, value in metadata.items():
            add(self.metadata_tree, key, value)
        self.metadata_tree.collapseAll()
        self.metadata_tree.resizeColumnToContents(0)

    def _show_notes(self, notes) -> None:
        self.notes_view.setVisible(notes is not None)
        self.notes_view.set_notes(notes, self._object.name if self._object else "")

    # The notes widgets, by their old names
    @property
    def notes_table(self) -> QtWidgets.QTableWidget:
        return self.notes_view.table

    @property
    def add_note_button(self) -> QtWidgets.QPushButton:
        return self.notes_view.add_button

    def add_note(self) -> None:
        self.notes_view.add_note()


class NotesWidget(QtWidgets.QWidget):
    """Timestamped notes (date over time, then the note) with Add Note...

    Used by the info panel and for the notes of a set or group.
    """

    # A note was added
    notes_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._notes = None
        self._owner = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.header = QtWidgets.QLabel("Notes", self)
        self.header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.header)
        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Date", "Note"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        # The Note column takes whatever width the Date column leaves
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setMinimumHeight(100)
        layout.addWidget(self.table)
        self.add_button = QtWidgets.QPushButton("Add Note...", self)
        self.add_button.clicked.connect(self.add_note)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.add_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

    def set_notes(self, notes, owner: str) -> None:
        """Show *notes* (an NMNotes, or None) of *owner* (e.g. "Set1")."""
        self._notes, self._owner = notes, owner
        self.table.setRowCount(0)
        for entry in notes or []:
            row = self.table.rowCount()
            self.table.insertRow(row)
            date = str(entry.get("date", ""))
            # Date over time, so the column stays narrow
            date_item = QtWidgets.QTableWidgetItem(date.replace(" ", "\n", 1))
            date_item.setToolTip(date)
            self.table.setItem(row, 0, date_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(entry.get("note", ""))))
        self.table.resizeColumnToContents(0)
        self.table.resizeRowsToContents()

    def _ask_note(self) -> str | None:
        """The Add Note dialog (replaced in tests)."""
        text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self, "Add Note", f"Note for {self._owner}:"
        )
        return text if accepted else None

    def add_note(self) -> None:
        if self._notes is None:
            return
        text = self._ask_note()
        if not text or not text.strip():
            return
        self._notes.add(text.strip())
        self.set_notes(self._notes, self._owner)
        self.notes_changed.emit()


__all__ = ["INFO_TYPES", "InfoPanel", "NotesWidget"]
