"""The History panel: what happened, by kind, with the kinds shown chosen.

Entries come from the core history log (``nm_history``, through a logging
handler) and from the GUI (selection changes, tool runs). Each has a kind:

- command: a Python command for what was done (e.g. ``nm.folders.new('Demo')``)
- message: the history log's notes (e.g. "added channel(s) C, D")
- warning / error: alerts and errors, in amber and red
- selection: the selection, after each change
- tool: a tool run, e.g. "Stats: Complete → Stats_Record_A_S1_0"

The panel only shows the history; tools and the core write it.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from collections import deque

from PyQt6 import QtCore, QtGui, QtWidgets

KINDS = ("command", "message", "warning", "error", "selection", "tool")
LABELS = {
    "command": "Commands",
    "message": "Messages",
    "warning": "Warnings",
    "error": "Errors",
    "selection": "Selection",
    "tool": "Tool Runs",
}
TIPS = {
    "command": "Python commands for what was done",
    "message": "Notes from the history log, e.g. what was added or changed",
    "warning": "Alerts",
    "error": "Errors, e.g. a tool run that failed",
    "selection": "The selection, after each change",
    "tool": "Tool runs and where their results went",
}
COLORS = {"warning": "#b26a00", "error": "#c62828"}
# The most recent entries kept (about 30 MB for 100,000); older ones go
MAX_ENTRIES = 100000

# The toggles look like tabs (as Plot / Table): joined, those on like the
# selected tab, those off greyed. Hover changes only the border, so a click
# shows at once. "segment" is first, middle, last or only.
SEGMENT_STYLE = """
QToolButton[segment] {
    border: 1px solid palette(mid); border-left: none;
    padding: 2px 10px; background: palette(button); color: palette(mid);
}
QToolButton[segment="first"], QToolButton[segment="only"] { border-left: 1px solid palette(mid); }
QToolButton[segment="first"] { border-top-left-radius: 4px; border-bottom-left-radius: 4px; }
QToolButton[segment="last"] { border-top-right-radius: 4px; border-bottom-right-radius: 4px; }
QToolButton[segment="only"] { border-radius: 4px; }
QToolButton[segment]:checked { background: palette(base); color: palette(text); }
QToolButton[segment]:hover { border-color: palette(dark); }
"""


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
    time: datetime.datetime
    kind: str
    text: str


def entry_from_record(record: logging.LogRecord) -> HistoryEntry:
    """A history log record as an entry: "path: message", by kind."""
    if getattr(record, "kind", "") == "command":
        kind = "command"
    elif record.levelno >= logging.ERROR:
        kind = "error"
    elif record.levelno >= logging.WARNING:
        kind = "warning"
    else:
        kind = "message"
    text = record.getMessage()
    path = getattr(record, "path", "")
    if path and path.upper() != "NONE":
        text = f"{path}: {text}"
    return HistoryEntry(datetime.datetime.fromtimestamp(record.created), kind, text)


class _Emitter(QtCore.QObject):
    entry = QtCore.pyqtSignal(object)


class QtHistoryHandler(logging.Handler):
    """Sends the history log's records to the panel, as entries."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.emitter = _Emitter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.entry.emit(entry_from_record(record))
        except RuntimeError:  # the panel has gone (e.g. the window closed)
            pass


class HistoryModel(QtCore.QAbstractListModel):
    """The entries, newest last (the most recent MAX_ENTRIES kept)."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._entries: deque[HistoryEntry] = deque()
        self.show_time = False

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def add(self, entries: list[HistoryEntry]) -> list[HistoryEntry]:
        """Add *entries* in one go; returns those dropped (the oldest) to
        keep MAX_ENTRIES."""
        entries = entries[-MAX_ENTRIES:]
        dropped = []
        excess = len(self._entries) + len(entries) - MAX_ENTRIES
        if excess > 0:
            self.beginRemoveRows(QtCore.QModelIndex(), 0, excess - 1)
            dropped = [self._entries.popleft() for _ in range(excess)]
            self.endRemoveRows()
        row = len(self._entries)
        self.beginInsertRows(QtCore.QModelIndex(), row, row + len(entries) - 1)
        self._entries.extend(entries)
        self.endInsertRows()
        return dropped

    def clear(self) -> None:
        self.beginResetModel()
        self._entries.clear()
        self.endResetModel()

    def set_show_time(self, show: bool) -> None:
        self.show_time = show
        if self._entries:
            self.dataChanged.emit(self.index(0), self.index(len(self._entries) - 1))

    def kind(self, row: int) -> str:
        return self._entries[row].kind

    def text(self, row: int) -> str:
        entry = self._entries[row]
        if self.show_time:
            return f"{entry.time:%H:%M:%S}  {entry.text}"
        return entry.text

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        Role = QtCore.Qt.ItemDataRole
        kind = self._entries[index.row()].kind
        if role == Role.DisplayRole:
            return self.text(index.row())
        if role == Role.ForegroundRole:
            if kind in COLORS:
                return QtGui.QBrush(QtGui.QColor(COLORS[kind]))
            if kind == "selection":  # quieter
                return QtWidgets.QApplication.palette().brush(QtGui.QPalette.ColorRole.PlaceholderText)
            return None
        if role == Role.FontRole and kind == "command":  # code
            return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        if role == Role.ToolTipRole:
            return f"{self._entries[index.row()].time:%Y-%m-%d %H:%M:%S} · {LABELS[kind]}"
        return None


class _KindFilter(QtCore.QSortFilterProxyModel):
    """Only the kinds chosen."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.kinds = set(KINDS)

    def set_kinds(self, kinds: set[str]) -> None:
        self.kinds = set(kinds)
        self.invalidateFilter()

    def filterAcceptsRow(self, row: int, parent: QtCore.QModelIndex) -> bool:  # noqa: N802
        return self.sourceModel().kind(row) in self.kinds


class HistoryPanel(QtWidgets.QWidget):
    """The entries in a list, a toggle for each kind, Time, and a ⋯ menu
    (Copy, Copy All, Clear)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = HistoryModel(self)
        self.filter = _KindFilter(self)
        self.filter.setSourceModel(self.model)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)  # the kinds, joined as tabs
        self.kind_buttons: dict[str, QtWidgets.QToolButton] = {}
        for kind in KINDS:
            button = QtWidgets.QToolButton(self)
            button.setText(LABELS[kind])
            button.setToolTip(f"Show {LABELS[kind].lower()}: {TIPS[kind]}")
            button.setCheckable(True)
            button.setChecked(True)
            button.toggled.connect(self._update_filter)
            row.addWidget(button)
            self.kind_buttons[kind] = button
        row.addSpacing(12)
        self.time_button = QtWidgets.QToolButton(self)
        self.time_button.setText("Time")
        self.time_button.setToolTip("Show when each entry was written")
        self.time_button.setCheckable(True)
        self.time_button.toggled.connect(self.model.set_show_time)
        row.addWidget(self.time_button)
        buttons = list(self.kind_buttons.values())
        for index, button in enumerate(buttons):
            button.setProperty("segment", "first" if index == 0
                               else "last" if index == len(buttons) - 1 else "middle")
        self.time_button.setProperty("segment", "only")
        self.setStyleSheet(SEGMENT_STYLE)
        row.addStretch(1)

        self.menu = QtWidgets.QMenu("History", self)
        self.copy_action = self.menu.addAction("Copy")
        self.copy_action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy))
        self.copy_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.copy_action.setToolTip("Copy the selected entries")
        self.copy_action.triggered.connect(self.copy_selection)
        self.copy_all_action = self.menu.addAction("Copy All")
        self.copy_all_action.setToolTip("Copy the entries shown")
        self.copy_all_action.triggered.connect(self.copy_all)
        self.menu.addSeparator()
        self.clear_action = self.menu.addAction("Clear")
        self.clear_action.setToolTip("Remove all entries from the panel")
        self.clear_action.triggered.connect(self.ask_clear)
        self.menu_button = QtWidgets.QToolButton(self)
        self.menu_button.setText("⋯")
        self.menu_button.setToolTip("History functions")
        self.menu_button.setMenu(self.menu)
        self.menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
        row.addWidget(self.menu_button)

        self.view = QtWidgets.QListView(self)
        self.view.setModel(self.filter)
        self.view.setUniformItemSizes(True)  # quick with many entries
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setMinimumHeight(40)
        self.view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            lambda pos: self.menu.exec(self.view.viewport().mapToGlobal(pos)))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.view, stretch=1)

        self.handler = QtHistoryHandler()
        self.handler.emitter.entry.connect(self.add_entry)
        # Entries written together (e.g. a 1000-epoch series) are shown in
        # one go, once control returns to the event loop
        self._pending: list[HistoryEntry] = []
        self._counts = {"warning": 0, "error": 0}
        self._update_counts()

    # --- entries -----------------------------------------------------------

    def add(self, kind: str, text: str) -> None:
        """Add an entry from the GUI (e.g. kind "selection" or "tool")."""
        self.add_entry(HistoryEntry(datetime.datetime.now(), kind, text))

    def add_entry(self, entry: HistoryEntry) -> None:
        if not self._pending:
            QtCore.QTimer.singleShot(0, self.flush)
        self._pending.append(entry)

    def flush(self) -> None:
        """Show the entries waiting to be shown."""
        if not self._pending:
            return
        entries, self._pending = self._pending, []
        bar = self.view.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 2
        dropped = self.model.add(entries)
        for entry, change in [(e, 1) for e in entries] + [(e, -1) for e in dropped]:
            if entry.kind in self._counts:
                self._counts[entry.kind] += change
        self._update_counts()
        if at_end:  # follow new entries, unless scrolled back to read
            self.view.scrollToBottom()

    def ask_clear(self) -> None:
        """Clear, once confirmed (it can't be undone)."""
        self.flush()
        count = self.model.rowCount()
        if count and self._confirm(
                "Clear History",
                f"Remove all {count:,} entries from the History panel? This can't be undone."):
            self.clear()

    def _confirm(self, title: str, text: str) -> bool:
        """Yes or No (replaced in tests)."""
        answer = QtWidgets.QMessageBox.question(
            self, title, text, defaultButton=QtWidgets.QMessageBox.StandardButton.No)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def clear(self) -> None:
        self._pending.clear()
        self.model.clear()
        self._counts = dict.fromkeys(self._counts, 0)
        self._update_counts()

    def attach(self, logger_name: str = "pyneuromatic") -> None:
        """Show the core history log's records as they are written."""
        logger = logging.getLogger(logger_name)
        if self.handler not in logger.handlers:
            logger.addHandler(self.handler)

    def detach(self, logger_name: str = "pyneuromatic") -> None:
        logging.getLogger(logger_name).removeHandler(self.handler)

    # --- which kinds -------------------------------------------------------

    def shown_kinds(self) -> set[str]:
        return {kind for kind, button in self.kind_buttons.items() if button.isChecked()}

    def set_shown_kinds(self, kinds) -> None:
        for kind, button in self.kind_buttons.items():
            button.setChecked(kind in kinds)

    def _update_filter(self, *_args) -> None:
        self.filter.set_kinds(self.shown_kinds())
        self.view.scrollToBottom()

    def _update_counts(self) -> None:
        """Warnings and Errors say how many there are."""
        for kind, count in self._counts.items():
            button = self.kind_buttons[kind]
            button.setText(f"{LABELS[kind]} ({count})" if count else LABELS[kind])
            button.updateGeometry()  # wider for the count, not cut short

    # --- text and copying ----------------------------------------------------

    def _row_text(self, proxy_row: int) -> str:
        return self.model.text(self.filter.mapToSource(self.filter.index(proxy_row, 0)).row())

    def text(self) -> str:
        """The entries shown, one per line (as copied)."""
        self.flush()
        return "\n".join(self._row_text(row) for row in range(self.filter.rowCount()))

    def selection_text(self) -> str:
        self.flush()
        rows = sorted(index.row() for index in self.view.selectionModel().selectedRows())
        return "\n".join(self._row_text(row) for row in rows)

    def copy_selection(self) -> None:
        text = self.selection_text()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)

    def copy_all(self) -> None:
        text = self.text()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)


__all__ = ["HistoryEntry", "HistoryPanel", "KINDS", "LABELS", "QtHistoryHandler",
           "entry_from_record"]
