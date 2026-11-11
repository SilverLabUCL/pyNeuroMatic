"""Notes of an object (a folder, data array, set...), one editor everywhere
(#505): Date, then Note, with Add…, Edit… and Delete….

Used by the Notes window (the strip's Notes…, the epoch ⋯ menu, Stats2's
Folder ⋯ › Notes…) and the browser's info panel (and a set's or group's notes). An edited note keeps its date;
Delete… asks first. Notes pyNeuroMatic wrote itself (read-only, e.g. how a
run's results were made) can't be edited or deleted. Adding, editing and
deleting a note is a history message, not a command (#507).
"""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

import pyneuromatic.core.nm_history as nmh

MESSAGE_LENGTH = 60  # a note's text in a history message, at most (then "…")


def short_text(text: str, length: int = MESSAGE_LENGTH) -> str:
    """*text* on one line, cut to *length* characters with "…" (#507)."""
    line = " ".join(str(text).split())
    return line if len(line) <= length else line[:length - 1].rstrip() + "…"


# --- whose notes: "Name (Parent)", the Notes window's title -------------------------
# A folder: its name. A subfolder, data array, Data Set or data series: its
# folder ("Demo / Stats_0" inside a subfolder). An epoch, channel, Epoch Set
# or group: its folder and data series ("Demo / Record"), so it's clear which
# folder's (e.g. in the history, #507).

def context_text(owner) -> str:
    """A folder's name, or "Folder / Subfolder" for a subfolder."""
    folder = getattr(owner, "_folder", None)
    if folder is None or folder is owner:
        return owner.name
    return f"{folder.name} / {owner.name}"


def series_text(series) -> str:
    """A data series with its folder, e.g. "Demo / Record" ("Demo / Stats_0 /
    Record" inside a subfolder)."""
    parent = getattr(series, "_parent", None)
    return f"{context_text(parent)} / {series.name}" if parent is not None else series.name


def notes_owner(obj) -> str:
    """Whose notes, for the Notes window's title: e.g. "Demo", "Record (Demo)",
    "E3 (Demo / Record)", "A (Demo / Record)", "ST_w0_max_y (Demo / Stats_0)"."""
    from pyneuromatic.core.nm_dataseries import NMDataSeries
    from pyneuromatic.tools.nm_tool_folder import NMToolFolder
    parent = getattr(obj, "_parent", None)
    folder = getattr(obj, "_folder", None)
    if folder is None or folder is obj:  # a folder
        return obj.name
    if isinstance(parent, NMDataSeries):  # an epoch or channel
        return f"{obj.name} ({series_text(parent)})"
    if isinstance(obj, NMToolFolder):
        return f"{obj.name} ({folder.name})"
    return f"{obj.name} ({context_text(parent if parent is not None else folder)})"


def set_owner(name: str, container) -> str:
    """A set's: its folder and data series (an Epoch Set), or folder (a Data Set)."""
    from pyneuromatic.core.nm_dataseries import NMDataSeries
    parent = getattr(container, "_parent", None)
    if isinstance(parent, NMDataSeries):
        return f"{name} ({series_text(parent)})"
    return f"{name} ({context_text(parent)})" if parent is not None else name


def group_owner(number: int, series) -> str:
    return f"Group {number} ({series_text(series)})"


class NotesTable(QtWidgets.QTableWidget):
    """Notes, a row each: Date (over time, so it stays narrow), then Note
    (wrapped). Edit › Copy (⌘C) copies the chosen rows, tab-separated."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Date", "Note"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setWordWrap(True)
        # The Note column takes whatever width the Date column leaves
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(100)

    def show_notes(self, notes) -> None:
        self.setRowCount(0)
        for row, note in enumerate(notes or []):
            self.insertRow(row)
            date = str(note.get("date", ""))
            date_item = QtWidgets.QTableWidgetItem(date.replace(" ", "\n", 1))
            date_item.setData(QtCore.Qt.ItemDataRole.UserRole, date)  # copied on one line
            note_item = QtWidgets.QTableWidgetItem(str(note.get("note", "")))
            tip = "Written by pyNeuroMatic (read-only)" if note.get("readonly") else date
            for item in (date_item, note_item):
                item.setToolTip(tip)
            self.setItem(row, 0, date_item)
            self.setItem(row, 1, note_item)
        self.resizeColumnToContents(0)
        self.resizeRowsToContents()

    def chosen_rows(self) -> list[int]:
        return sorted({index.row() for index in self.selectionModel().selectedRows()})

    def copy_selection(self) -> None:
        """The chosen notes, "date<TAB>note" a line each, to the clipboard."""
        lines = [f"{self.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)}\t"
                 f"{self.item(row, 1).text()}" for row in self.chosen_rows()]
        if lines:
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))


class NotesEditor(QtWidgets.QWidget):
    """An object's notes (an NMNotes) with Add…, Edit… and Delete….

    Emits :attr:`notes_changed` after a note is added, edited or deleted.
    """

    notes_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None,
                 header: bool = True) -> None:
        super().__init__(parent)
        self._notes = None
        self._owner = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.header = QtWidgets.QLabel("Notes", self)
        self.header.setStyleSheet("font-weight: bold;")
        self.header.setVisible(header)
        layout.addWidget(self.header)
        self.table = NotesTable(self)
        layout.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add…", self)
        self.add_button.setToolTip("Add a note")
        self.edit_button = QtWidgets.QPushButton("Edit…", self)
        self.edit_button.setToolTip("Change the chosen note's text (its date stays)")
        self.delete_button = QtWidgets.QPushButton("Delete…", self)
        self.delete_button.setToolTip("Delete the chosen notes")
        for button in (self.add_button, self.edit_button, self.delete_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.add_button.clicked.connect(self.add_note)
        self.edit_button.clicked.connect(self.edit_note)
        self.delete_button.clicked.connect(self.delete_notes)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self._update_buttons()

    @property
    def notes(self):
        return self._notes

    def set_notes(self, notes, owner: str = "") -> None:
        """Show *notes* (an NMNotes, or None) of *owner* (e.g. "Set1")."""
        self._notes, self._owner = notes, owner
        self.refresh()

    def refresh(self) -> None:
        self.table.show_notes(self._notes)
        self._update_buttons()

    def _editable(self, rows: list[int]) -> list[int]:
        notes = self._notes if self._notes is not None else []
        return [row for row in rows if row < len(notes) and not notes.is_readonly(row)]

    def _update_buttons(self) -> None:
        rows = self.table.chosen_rows()
        editable = self._editable(rows)
        self.add_button.setEnabled(self._notes is not None)
        # Notes pyNeuroMatic wrote (read-only) stay as written
        self.edit_button.setEnabled(len(rows) == 1 and len(editable) == 1)
        self.delete_button.setEnabled(bool(rows) and len(editable) == len(rows))

    # --- prompts (replaced in tests) -------------------------------------------

    def _ask_note(self, title: str, text: str = "") -> str | None:
        """A note's text, or None if cancelled."""
        label = f"Note for {self._owner}:" if self._owner else "Note:"
        answer, ok = QtWidgets.QInputDialog.getMultiLineText(self, title, label, text)
        return answer if ok else None

    def _confirm_delete(self, count: int) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self, "Delete Notes", f"Delete {'this note' if count == 1 else f'{count} notes'}?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    # --- actions ------------------------------------------------------------------

    def _message(self, what: str, number: int | None, text: str) -> None:
        """A history message (not a command, #507), e.g. "Stats_Control
        (Demo): edited note 2: 'E8 excluded: noisy'"; numbered from 1."""
        note = f"note {number}:" if number is not None else "note"
        nmh.history(f"{what} {note} '{short_text(text)}'",
                    path=self._owner or "Notes", quiet=False)

    def _changed(self, select: int | None = None) -> None:
        # Announced first: what's told (e.g. Stats2, the browser) may show the
        # notes again, so the note is chosen after
        self.notes_changed.emit()
        self.refresh()
        if select is not None:
            self.table.selectRow(select)

    def add_note(self) -> None:
        """Add… a note (dated now)."""
        if self._notes is None:
            return
        text = self._ask_note("Add Note")
        if text is None or not text.strip():
            return
        self._notes.add(text.strip())
        self._message("added", None, text.strip())
        self._changed(select=len(self._notes) - 1)

    def edit_note(self) -> None:
        """Edit… the chosen note's text; its date stays."""
        rows = self._editable(self.table.chosen_rows())
        if self._notes is None or len(rows) != 1:
            return
        text = self._ask_note("Edit Note", self._notes[rows[0]]["note"])
        if text is None or not text.strip():
            return
        self._notes.edit(rows[0], text.strip())
        self._message("edited", rows[0] + 1, text.strip())
        self._changed(select=rows[0])

    def delete_notes(self) -> None:
        """Delete… the chosen notes, after asking."""
        rows = self.table.chosen_rows()
        if self._notes is None or not rows or self._editable(rows) != rows:
            return
        if not self._confirm_delete(len(rows)):
            return
        texts = [self._notes[row]["note"] for row in rows]
        for row in reversed(rows):
            self._notes.remove(row)
        for row, text in zip(rows, texts):  # numbered as they were
            self._message("deleted", row + 1, text)
        self._changed()


class NotesDialog(QtWidgets.QDialog):
    """An object's notes on their own (the strip's Notes…), closed with Close."""

    def __init__(self, notes, owner: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Notes: {owner}")
        layout = QtWidgets.QVBoxLayout(self)
        self.editor = NotesEditor(self, header=False)
        self.editor.set_notes(notes, owner)
        layout.addWidget(self.editor)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 320)


__all__ = ["NotesDialog", "NotesEditor", "NotesTable", "context_text", "group_owner",
           "notes_owner", "series_text", "set_owner"]
