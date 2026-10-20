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
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder

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


def _scale_text(scale, with_start: bool) -> str:
    """e.g. "start 0, delta 0.1 ms (Time)"."""
    units = f" {scale.units}" if scale.units else ""
    text = f"start {scale.start:g}, delta {scale.delta:g}{units}" if with_start else (
        scale.units or "—"
    )
    return text + (f" ({scale.label})" if scale.label else "")


def _sets_of(container, name: str) -> list[str]:
    sets = getattr(container, "sets", None)
    if sets is None:
        return []
    return [set_name for set_name in sets.keys() if sets.contains(set_name, name)]


class InfoPanel(QtWidgets.QScrollArea):
    """Shows one object: a title, a form of its properties and its notes."""

    # A note was added to the object shown
    notes_changed = QtCore.pyqtSignal()

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
        layout.addWidget(self.title_label)

        self._form = QtWidgets.QFormLayout()
        self._form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addLayout(self._form)

        # Imported metadata (folders), collapsed to one row per source folder
        self.metadata_tree = QtWidgets.QTreeWidget(self._body)
        self.metadata_tree.setHeaderLabels(["Imported Metadata", "Value"])
        self.metadata_tree.setMinimumHeight(80)
        self.metadata_tree.setMaximumHeight(160)
        layout.addWidget(self.metadata_tree)

        self.notes_header = QtWidgets.QLabel("Notes", self._body)
        self.notes_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.notes_header)
        self.notes_table = QtWidgets.QTableWidget(0, 2, self._body)
        self.notes_table.setHorizontalHeaderLabels(["Date", "Note"])
        self.notes_table.horizontalHeader().setStretchLastSection(True)
        self.notes_table.verticalHeader().hide()
        self.notes_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.notes_table.setWordWrap(True)
        # The Note column takes whatever width the Date column leaves
        self.notes_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.notes_table.setMinimumHeight(100)
        layout.addWidget(self.notes_table)
        self.add_note_button = QtWidgets.QPushButton("Add Note...", self._body)
        self.add_note_button.clicked.connect(self.add_note)
        note_row = QtWidgets.QHBoxLayout()
        note_row.addWidget(self.add_note_button)
        note_row.addStretch(1)
        layout.addLayout(note_row)
        layout.addStretch(1)

    @property
    def shown_object(self) -> NMObject | None:
        return self._object

    # --- showing an object --------------------------------------------------

    def show_object(self, obj: NMObject | None) -> None:
        self._object = obj
        while self._form.rowCount():
            self._form.removeRow(0)
        if obj is None:
            self.title_label.setText("")
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
            self._show_data(obj)
        elif isinstance(obj, NMDataSeries):
            self._show_dataseries(obj)
        elif isinstance(obj, NMChannel):
            self._show_channel(obj)
        elif isinstance(obj, NMEpoch):
            self._show_epoch(obj)
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

    def _add_row(self, label: str, value: str) -> None:
        field = QtWidgets.QLabel(value, self._body)
        # Only long values (e.g. lists of names) wrap: a wrapped label in a
        # form can lose its second line
        field.setWordWrap(len(value) > 40)
        field.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._form.addRow(label, field)

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
            self._add_row("Points", str(len(array)))
            if len(array) and numpy.issubdtype(array.dtype, numpy.number):
                with numpy.errstate(all="ignore"):
                    low, high = numpy.nanmin(array), numpy.nanmax(array)
                if numpy.isfinite(low) or numpy.isfinite(high):
                    self._add_row("Range", f"{low:g} to {high:g}")
        self._add_row("X Scale", _scale_text(data.xscale, with_start=True))
        self._add_row("Y Scale", _scale_text(data.yscale, with_start=False))
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

    def _show_dataseries(self, series: NMDataSeries) -> None:
        self._add_row("Channels", _names(series.channels.keys()))
        self._add_row("Epochs", str(len(series.epochs)))
        lengths = {len(d.nparray) for c in series.channels.values() for d in c.data
                   if d.nparray is not None}
        if lengths:
            low, high = min(lengths), max(lengths)
            self._add_row("Points", str(low) if low == high else f"{low} to {high}")
        self._add_row("Locked", "Yes (its data arrays can't be changed)"
                      if series.locked else "No")
        self._add_row("Epoch Sets", _names(series.epochs.sets.keys()))
        self._add_row("Groups", _names(str(n) for n in series.epochs.groups.group_numbers))

    def _show_channel(self, channel: NMChannel) -> None:
        self._add_row("Data", _count(len(channel.data), "array"))
        self._add_row("X Scale", _scale_text(channel.xscale, with_start=True))
        self._add_row("Y Scale", _scale_text(channel.yscale, with_start=False))

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
        has_notes = notes is not None
        for widget in (self.notes_header, self.notes_table, self.add_note_button):
            widget.setVisible(has_notes)
        self.notes_table.setRowCount(0)
        if not has_notes:
            return
        for entry in notes:
            row = self.notes_table.rowCount()
            self.notes_table.insertRow(row)
            date = str(entry.get("date", ""))
            # Date over time, so the column stays narrow
            date_item = QtWidgets.QTableWidgetItem(date.replace(" ", "\n", 1))
            date_item.setToolTip(date)
            self.notes_table.setItem(row, 0, date_item)
            self.notes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(entry.get("note", ""))))
        self.notes_table.resizeColumnToContents(0)
        self.notes_table.resizeRowsToContents()

    def _ask_note(self) -> str | None:
        """The Add Note dialog (replaced in tests)."""
        text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self, "Add Note", f"Note for {self._object.name}:"
        )
        return text if accepted else None

    def add_note(self) -> None:
        notes = getattr(self._object, "notes", None)
        if notes is None:
            return
        text = self._ask_note()
        if not text or not text.strip():
            return
        notes.add(text.strip())
        self._show_notes(notes)
        self.notes_changed.emit()


__all__ = ["INFO_TYPES", "InfoPanel"]
