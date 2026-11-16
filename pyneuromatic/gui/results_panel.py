"""The results panel a tool's "2" tab shows (Stats2, Spike2...), #525.

Choose a results subfolder (``Stats_...``, ``Spike_...``) of the current
folder and an array in it, and act on them: the Folder ⋯ menu (Rename,
Copy, Notes, Delete, Show in Browser) and the Array ⋯ menu (Plot, Table,
Show in Browser). A tool's panel subclasses :class:`ResultsPanel` and adds
its own parts through the hooks below (e.g. Stats2: a Window list,
Histogram..., Inequality..., its tables).
"""
from __future__ import annotations

import re
from typing import Any

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.gui.notes_view import NotesChoices, NotesDialog, notes_owner
from pyneuromatic.tools.nm_tool_folder import NMToolFolder

MENU_BUTTON_STYLE = "QToolButton::menu-indicator { image: none; width: 0px; }"

_NOTE_PAIR = re.compile(r"(\w+)=([^,()]+)")


def note_fields(data, kind: str) -> dict[str, str]:
    """The fields of *data*'s note written by tool *kind*, e.g. "NMStats(win=w0,
    func=max, id=main, xbgn=60.0, n=2)" -> {"win": "w0", "func": "max", ...}
    for kind "NMStats" (or "NMSpike"...); {} if it has none."""
    for note in data.notes:
        if str(note["note"]).startswith(kind + "("):
            return {k: v.strip() for k, v in _NOTE_PAIR.findall(note["note"])}
    return {}


def runs_text(names: list[str]) -> str:
    """Data names with consecutive numbers shortened, e.g. "RecordA0,
    RecordA1, RecordA2, RecordA3, RecordA5" -> "RecordA0-3, RecordA5"; a name
    that isn't prefix + channel + number is shown as it is."""
    runs: list[list] = []  # [prefix + channel, first, last] or [name]
    for name in names:
        parsed = nmu.parse_data_name(name)
        if parsed is None:
            runs.append([name])
            continue
        stem, number = parsed[0] + parsed[1], parsed[2]
        last = runs[-1] if runs else None
        if last is not None and len(last) == 3 and last[0] == stem and last[2] == number - 1:
            last[2] = number
        else:
            runs.append([stem, number, number])
    return ", ".join(
        run[0] if len(run) == 1
        else f"{run[0]}{run[1]}" if run[1] == run[2]
        else f"{run[0]}{run[1]}-{run[2]}" for run in runs)




def menu_button(menu: QtWidgets.QMenu, tip: str,
                parent: QtWidgets.QWidget) -> QtWidgets.QToolButton:
    """A ⋯ button showing *menu*, as the strip's (no menu arrow)."""
    button = QtWidgets.QToolButton(parent)
    button.setText("⋯")
    button.setToolTip(tip)
    button.setMenu(menu)
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setStyleSheet(MENU_BUTTON_STYLE)
    return button


class ResultsPanel(QtWidgets.QWidget):
    """A tool's results: a subfolder (Folder) and an array in it (Array).

    Subclasses set the names below and may override the hooks:

    - :meth:`_add_rows`: rows between Folder and Array (e.g. Stats2's Window)
    - :meth:`_add_plot_options`: Array ⋯ items after Plot (Plot x: epoch)
    - :meth:`_add_array_tools`: Array ⋯ items after Table (Histogram...)
    - :meth:`_add_details`: below the message (e.g. Stats2's Results box)
    - :meth:`_on_toolfolder_changed`: the subfolder chosen changed
    - :meth:`_array_names`, :meth:`default_array`: the arrays listed, and the
      one chosen first
    - :meth:`_on_array_chosen`, :meth:`_update_array_tools`: an array chosen
    - :meth:`selected_plot`: the chosen array as :attr:`plot_requested` values

    A subclass calls :meth:`refresh` at the end of its ``__init__``.
    """

    TOOLFOLDER_PREFIX = ""  # e.g. "Stats_": the subfolders listed
    TOOL_NAME = ""  # e.g. "Stats", in messages
    PANEL_NAME = ""  # e.g. "Stats2", in tooltips
    OBJECT_PREFIX = "results"  # the lists' object names, e.g. "stats2Folder"
    EXAMPLE_ARRAY = ""  # the Array list's width while it's empty

    # title, y values, x values (or None), x label, y label, style
    plot_requested = QtCore.pyqtSignal(str, object, object, str, str, str)
    # The list of the tool's subfolders in the current folder was reloaded
    results_changed = QtCore.pyqtSignal()
    # The array chosen in the Array list (or its folder) changed
    array_changed = QtCore.pyqtSignal()
    # ⋯ › Table: show the tables, the chosen array's column selected (#523)
    table_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._selection_model: Any = None
        self._folder: NMFolder | None = None
        self._folder_actions: Any = None
        self._folder_names: list[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        # Rows span the panel, so the Array list has its row's width (macOS
        # keeps a form's fields at their size hint, cutting off array names)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Folder", self._folder_row())
        self._add_rows(form)
        form.addRow("Array", self._array_row())
        layout.addLayout(form)

        self.message_label = QtWidgets.QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self._add_details(layout)

        self.folder_combo.currentIndexChanged.connect(self._on_folder_changed)
        self.array_combo.currentIndexChanged.connect(self._on_array_changed)
        self.rename_folder_action.triggered.connect(self.rename_folder)
        self.copy_folder_action.triggered.connect(self.copy_folder)
        self.delete_folder_action.triggered.connect(self.delete_folder)
        self.browse_folder_action.triggered.connect(self.show_folder_in_browser)
        self.notes_folder_action.triggered.connect(self.show_folder_notes)
        self.browse_array_action.triggered.connect(self.show_array_in_browser)

    # --- the rows ----------------------------------------------------------

    def _folder_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.folder_combo = QtWidgets.QComboBox(self)
        self.folder_combo.setObjectName(f"{self.OBJECT_PREFIX}Folder")
        # As wide as its subfolder names (refresh)
        row.addWidget(self.folder_combo)
        # The subfolder's actions, as the strip's Subfolder ⋯ (#494): the
        # app's folder actions (set_folder_actions), starting on the one shown
        self.folder_menu = QtWidgets.QMenu("Folder", self)
        self.folder_menu.setToolTipsVisible(True)
        self.rename_folder_action = self.folder_menu.addAction("Rename…")
        self.rename_folder_action.setToolTip(
            f"Rename the subfolder (its name keeps {self.TOOLFOLDER_PREFIX})")
        self.copy_folder_action = self.folder_menu.addAction("Copy…")
        self.copy_folder_action.setToolTip("Copy the subfolder")
        # The subfolder's notes, in the Notes window every object's use (#501, #505)
        self.notes_folder_action = self.folder_menu.addAction("Notes…")
        self.notes_folder_action.setToolTip(
            "Show, add, edit and delete the subfolder's notes")
        self.delete_folder_action = self.folder_menu.addAction("Delete…")
        self.delete_folder_action.setToolTip("Delete the subfolder and its data")
        self.folder_menu.addSeparator()
        self.browse_folder_action = self.folder_menu.addAction("Show in Browser")
        self.browse_folder_action.setToolTip("Show the subfolder in the browser")
        self.folder_menu_button = menu_button(self.folder_menu, "Subfolder actions", self)
        row.addWidget(self.folder_menu_button)
        row.addStretch(1)
        return row

    def _array_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.array_combo = QtWidgets.QComboBox(self)
        self.array_combo.setObjectName(f"{self.OBJECT_PREFIX}Array")
        # As wide as its array names (_refresh_array_combo), as the Folder list
        row.addWidget(self.array_combo)
        # The chosen array's actions, in a ⋯ menu beside it (#485): Plot and
        # its options; Table and the tool's tools; the Browser (#523)
        self.plot_menu = QtWidgets.QMenu("Array", self)
        self.plot_menu.setToolTipsVisible(True)
        self.plot_action = self.plot_menu.addAction("Plot")
        self.plot_action.setToolTip("Show the chosen array in the Plot panel")
        self.plot_action.triggered.connect(lambda _checked=False: self.plot_selected())
        self._add_plot_options(self.plot_menu)
        self.plot_menu.addSeparator()
        self.table_action = self.plot_menu.addAction("Table")
        self.table_action.setToolTip(
            f"Show the {self.PANEL_NAME} tables in the Table panel, the chosen "
            "array's column selected")
        self.table_action.triggered.connect(lambda _checked=False: self.table_requested.emit())
        self._add_array_tools(self.plot_menu)
        self.plot_menu.addSeparator()
        self.browse_array_action = self.plot_menu.addAction("Show in Browser")
        self.browse_array_action.setToolTip("Show the chosen array in the browser")
        self.plot_menu_button = menu_button(
            self.plot_menu, "Plot, Table and more, for the chosen array", self)
        row.addWidget(self.plot_menu_button)
        row.addStretch(1)
        return row

    # --- hooks ---------------------------------------------------------------

    def _add_rows(self, form: QtWidgets.QFormLayout) -> None:
        """Rows between Folder and Array (none)."""

    def _add_plot_options(self, menu: QtWidgets.QMenu) -> None:
        """Array ⋯ items after Plot (none)."""

    def _add_array_tools(self, menu: QtWidgets.QMenu) -> None:
        """Array ⋯ items after Table (none)."""

    def _add_details(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Below the message (nothing)."""

    def _array_names(self) -> list[str]:
        """The arrays listed for the subfolder chosen (none)."""
        return []

    def default_array(self) -> str | None:
        """The array chosen first, e.g. after a run (the first listed)."""
        names = self._array_names()
        return names[0] if names else None

    def _on_toolfolder_changed(self) -> None:
        """The subfolder chosen changed: its message and arrays."""
        self._show_message()
        self._refresh_array_combo()

    def _on_array_chosen(self, name: str | None) -> None:
        """An array was chosen (e.g. select it in the tables)."""

    def _update_array_tools(self, name: str | None) -> None:
        """Enable the tool's Array ⋯ items for array *name*."""

    def selected_plot(self) -> tuple | None:
        """The chosen array as :attr:`plot_requested` values (title, y, x,
        x label, y label, style); None if there's nothing to plot."""
        return None

    def no_results_message(self) -> str:
        """Said when the folder has none of the tool's subfolders."""
        return f"No saved {self.TOOL_NAME} results in {self._folder.name}."

    # --- selection -------------------------------------------------------

    def bind_selection_model(self, selection_model) -> None:
        self._selection_model = selection_model
        selection_model.selection_changed.connect(self._on_selection_changed)
        self.refresh()

    def _current_folder(self) -> NMFolder | None:
        if self._selection_model is not None:
            folder = self._selection_model.selection.get("folder")
        else:
            folder = self._manager.select_values.get("folder")
        return folder if isinstance(folder, NMFolder) else None

    def _on_selection_changed(self, *_args) -> None:
        # Trace navigation changes the selection often; only a new folder
        # changes what is shown
        if self._current_folder() is not self._folder:
            self.refresh()

    # --- the Folder list -----------------------------------------------------

    def toolfolder(self) -> NMToolFolder | None:
        """The subfolder chosen in the Folder list."""
        if self._folder is None:
            return None
        name = self.folder_combo.currentData()
        if name is None:
            return None
        return self._folder.toolfolders.get(name)

    def refresh(self, select: str | None = None) -> None:
        """Reload the tool's subfolder list, choosing *select* if given.

        Otherwise the current choice is kept, or the newest folder is chosen.
        """
        self._folder = self._current_folder()
        previous = select or self.folder_combo.currentData()
        names: list[str] = []
        if self._folder is not None:
            names = [
                name
                for name in self._folder.toolfolders.keys()
                if name.startswith(self.TOOLFOLDER_PREFIX)
            ]
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        for name in names:
            self.folder_combo.addItem(name, name)
        self._folder_names = names
        metrics = self.folder_combo.fontMetrics()
        example = f"{self.TOOLFOLDER_PREFIX}Record_A_All_0"
        longest = max(names, key=metrics.horizontalAdvance) if names else example
        self.folder_combo.setFixedWidth(metrics.horizontalAdvance(longest) + 44)

        # The one before, else the newest
        index = names.index(previous) if previous in names else len(names) - 1
        self.folder_combo.setCurrentIndex(index)
        self.folder_combo.blockSignals(False)
        self._on_folder_changed()
        self.results_changed.emit()

    def has_results(self) -> bool:
        """True if the current folder has saved results of the tool."""
        return bool(self._folder_names)

    def show_toolfolder(self, toolfolder: NMToolFolder | None) -> None:
        """Show *toolfolder*, e.g. the one a run just wrote, with its default
        array chosen (e.g. Stats: the measurement, not its baseline)."""
        self.refresh(select=toolfolder.name if toolfolder is not None else None)
        name = self.default_array()
        if name is not None:
            self.select_array(name)

    def _on_folder_changed(self, *_args) -> None:
        self._update_folder_actions()
        self._on_toolfolder_changed()

    def _message(self) -> str:
        """What the panel says instead of results ("" if there are some)."""
        if self._folder is None:
            return f"Select a folder to see its {self.TOOL_NAME} results."
        if self.toolfolder() is None:
            return self.no_results_message()
        return ""

    def _show_message(self) -> str:
        message = self._message()
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        return message

    # --- the Array list --------------------------------------------------

    def selected_array(self) -> str | None:
        """The array name chosen in the Array list."""
        return self.array_combo.currentData()

    def select_array(self, name: str) -> None:
        index = self.array_combo.findData(name)
        if index >= 0:
            self.array_combo.setCurrentIndex(index)

    def _refresh_array_combo(self) -> None:
        previous = self.array_combo.currentData()
        self.array_combo.blockSignals(True)
        self.array_combo.clear()
        names = self._array_names()
        for name in names:
            self.array_combo.addItem(name, name)
        metrics = self.array_combo.fontMetrics()
        longest = max(names, key=metrics.horizontalAdvance) if names else self.EXAMPLE_ARRAY
        self.array_combo.setFixedWidth(metrics.horizontalAdvance(longest) + 44)
        index = self.array_combo.findData(previous)
        if index < 0:  # not in these results: their default
            index = self.array_combo.findData(self.default_array())
        self.array_combo.setCurrentIndex(max(index, 0) if self.array_combo.count() else -1)
        self.array_combo.blockSignals(False)
        self._on_array_changed()

    @staticmethod
    def is_histogram(name: str | None) -> bool:
        """True if *name* is a saved histogram (H_<array>, #492)."""
        return bool(name) and name.startswith("H_")

    def _on_array_changed(self, *_args) -> None:
        self._on_array_chosen(self.selected_array())
        self._update_buttons()
        self.array_changed.emit()

    def _update_buttons(self) -> None:
        name = self.selected_array()
        self.plot_action.setEnabled(name is not None)
        self.table_action.setEnabled(name is not None)
        self._update_array_tools(name)
        self.browse_array_action.setEnabled(name is not None and self._folder_actions is not None)

    def _axis_label(self, name: str) -> str:
        units = self.toolfolder().data.get(name).yscale.units
        return f"{name} ({units})" if units else name

    def plot_selected(self) -> None:
        values = self.selected_plot()
        if values is not None:
            self.plot_requested.emit(*values)

    def show_array_in_browser(self) -> None:
        """Show in Browser (Array ⋯): the chosen array, in its subfolder."""
        name = self.selected_array()
        toolfolder = self.toolfolder()
        if name is None or toolfolder is None or self._folder_actions is None:
            return
        data = toolfolder.data.get(name)
        if data is not None:
            self._folder_actions.browse_requested.emit(data)

    # --- Notes (#501) -------------------------------------------------------------

    def _notes_changed(self) -> None:
        if self._folder_actions is not None:  # e.g. the browser's info panel
            self._folder_actions.content_changed.emit()

    def show_folder_notes(self) -> None:
        """Notes… (Folder ⋯): the subfolder's notes, in the Notes window
        (#501, #505); the run's own note is first, read-only."""
        toolfolder = self.toolfolder()
        if toolfolder is not None:
            folders = self._folder.toolfolders
            # The tool's other subfolders listed here, to switch to (#509)
            choices = NotesChoices(
                "Folder", list(self._folder_names), toolfolder.name,
                lambda name: (folders[name].notes, notes_owner(folders[name])))
            self._open_notes(toolfolder.notes, notes_owner(toolfolder), choices=choices)

    def _open_notes(self, notes, owner: str, choices=None) -> None:
        """The Notes window (replaced in tests)."""
        dialog = NotesDialog(notes, owner, self, choices)
        dialog.editor.notes_changed.connect(self._notes_changed)
        dialog.exec()

    # --- the Folder ⋯ menu (#494) -------------------------------------------

    def set_folder_actions(self, actions) -> None:
        """The app's folder actions (:class:`FolderActions`), whose subfolder
        actions the Folder ⋯ menu uses."""
        self._folder_actions = actions
        self._update_folder_actions()
        self._update_buttons()

    def _update_folder_actions(self) -> None:
        self.notes_folder_action.setEnabled(self.toolfolder() is not None)
        usable = self._folder_actions is not None and self.toolfolder() is not None
        for action in (self.rename_folder_action, self.copy_folder_action,
                       self.delete_folder_action, self.browse_folder_action):
            action.setEnabled(usable)

    def rename_folder(self) -> None:
        """Rename… (the strip's prompt, starting on the subfolder shown); then
        show its new name."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            name = self._folder_actions.rename_subfolder(toolfolder)
            self.refresh(select=name)

    def copy_folder(self) -> None:
        """Copy… (the strip's prompt, starting on the subfolder shown); then
        show the copy."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            name = self._folder_actions.duplicate_subfolder(toolfolder)
            self.refresh(select=name)

    def delete_folder(self) -> None:
        """Delete… (the strip's prompt, starting on the subfolder shown); then
        the one shown, or else the newest."""
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            self._folder_actions.delete_subfolder(toolfolder)
            self.refresh()

    def show_folder_in_browser(self) -> None:
        toolfolder = self.toolfolder()
        if toolfolder is not None and self._folder_actions is not None:
            self._folder_actions.subfolder_in_browser(toolfolder)


__all__ = ["ResultsPanel", "menu_button", "note_fields", "runs_text"]
