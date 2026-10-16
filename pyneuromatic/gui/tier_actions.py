"""Actions for selection tiers, shared by the strip's ⋯ menus and the menu bar.

Each tier's actions are QActions created once; a strip ⋯ button and a menu
bar menu both show the same objects, so enabled state and shortcuts agree.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

import numpy as np

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.gui.selection_model import SelectionModel


def _read_abf(path, folder):
    from pyneuromatic.io.abf import read_abf
    return read_abf(path, folder=folder)


def _read_axograph(path, folder):
    from pyneuromatic.io.axograph import read_axograph
    return read_axograph(path, folder=folder)


def _read_pxp(path, folder):
    from pyneuromatic.io.pxp import read_pxp
    return read_pxp(path, folder=folder)


# File suffix -> reader(path, folder) that adds the file's data to folder
READERS: dict[str, Callable] = {
    ".abf": _read_abf,
    ".axgd": _read_axograph,
    ".axgx": _read_axograph,
    ".pxp": _read_pxp,
}
OPEN_FILTER = (
    "Data files (*.abf *.axgd *.axgx *.pxp);;"
    "Axon Binary (*.abf);;AxoGraph (*.axgd *.axgx);;"
    "Igor packed experiment (*.pxp);;All files (*)"
)


def unique_name(container: NMObjectContainer, base: str) -> str:
    """A valid name not yet in *container*, based on *base* (e.g. a file name)."""
    name = re.sub(r"[^0-9A-Za-z_]", "_", base).strip("_")
    if not name or not nmu.name_ok(name):
        return container.auto_name_next()
    existing = {key.lower() for key in container.keys()}
    candidate, number = name, 1
    while candidate.lower() in existing:
        candidate = f"{name}_{number}"
        number += 1
    return candidate


# Prefix for duplicated data, keeping copies in NeuroMatic's naming pattern
# (C_RecordA0 belongs to series "C_Record", not "Record")
COPY_PREFIX = "C"


def copy_name(container: NMObjectContainer, name: str, prefix: str = COPY_PREFIX) -> str:
    """Name for a copy of *name*: ``C_name``, then ``C1_name``, ``C2_name``, ..."""
    existing = {key.lower() for key in container.keys()}
    candidate, number = f"{prefix}_{name}", 1
    while candidate.lower() in existing:
        candidate = f"{prefix}{number}_{name}"
        number += 1
    return candidate


def copy_target_name(container: NMObjectContainer, name: str) -> str:
    """Name for a copy of *name* in another *container*: the same name if it
    is free there, otherwise a prefixed copy name (``C_name``, ``C1_name``...)."""
    return name if container._getkey(name) is None else copy_name(container, name)


class ChooseItemDialog(QtWidgets.QDialog):
    """Choose an item (e.g. a folder) and/or enter a name, in one dialog.

    *items* adds an item dropdown, starting at *current*. *name_label* adds a
    name field; *default_name(item, *extras)* fills it whenever the item or
    an extra choice changes.
    *extras* adds further dropdowns, as ``(label, items, current)``.
    *message* is a note below the fields, or a function of the item.
    *context* adds a read-only first row, as ``(label, text)`` (e.g. the
    folder an action applies to).
    *validate(item, name, *extras)* returns an error message, shown inline
    while the dialog stays open, or None to accept.
    """

    def __init__(
        self,
        title: str,
        item_label: str | None = None,
        items: list[str] | None = None,
        current: str | None = None,
        name_label: str | None = None,
        default_name: Callable[[str | None], str] | None = None,
        validate: Callable[..., str | None] | None = None,
        message: str | Callable[[str | None], str] = "",
        extras: list[tuple[str, list[str], str | None]] | None = None,
        context: tuple[str, str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._default_name = default_name
        self._validate = validate
        self._message = message
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.context_label: QtWidgets.QLabel | None = None
        if context is not None:
            self.context_label = QtWidgets.QLabel(context[1], self)
            self.context_label.setStyleSheet("font-weight: bold;")
            form.addRow(context[0], self.context_label)

        self.item_combo: QtWidgets.QComboBox | None = None
        if items is not None:
            self.item_combo = QtWidgets.QComboBox(self)
            self.item_combo.addItems(items)
            if current in items:
                self.item_combo.setCurrentText(current)
            form.addRow(item_label or "Item", self.item_combo)
        self.extra_combos: list[QtWidgets.QComboBox] = []
        for label, extra_items, extra_current in extras or []:
            combo = QtWidgets.QComboBox(self)
            combo.addItems(extra_items)
            if extra_current in extra_items:
                combo.setCurrentText(extra_current)
            form.addRow(label, combo)
            self.extra_combos.append(combo)
        self.name_edit: QtWidgets.QLineEdit | None = None
        if name_label is not None:
            self.name_edit = QtWidgets.QLineEdit(self)
            form.addRow(name_label, self.name_edit)
        self.message_label = QtWidgets.QLabel("", self)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setStyleSheet("color: #b91c1c;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self.item_combo is not None:
            self.item_combo.currentTextChanged.connect(self._fill_default_name)
        for combo in self.extra_combos:
            combo.currentTextChanged.connect(self._fill_default_name)
        self._fill_default_name()
        if self.name_edit is not None:
            self.name_edit.selectAll()
            self.name_edit.setFocus()

    def item(self) -> str | None:
        return self.item_combo.currentText() if self.item_combo is not None else None

    def name(self) -> str | None:
        return self.name_edit.text().strip() if self.name_edit is not None else None

    def extras(self) -> list[str]:
        return [combo.currentText() for combo in self.extra_combos]

    def _fill_default_name(self, *_args) -> None:
        if self.name_edit is not None and self._default_name is not None:
            self.name_edit.setText(self._default_name(self.item(), *self.extras()))
        message = self._message(self.item()) if callable(self._message) else self._message
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        self.error_label.hide()

    def accept(self) -> None:
        error = (
            self._validate(self.item(), self.name(), *self.extras())
            if self._validate else None
        )
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


class NewDataDialog(QtWidgets.QDialog):
    """Name, size, fill value and scales for a new data array.

    *defaults* fills the fields (keys as in :meth:`values`); *validate(name)*
    returns a name error, shown inline while the dialog stays open.
    """

    def __init__(
        self,
        defaults: dict,
        validate: Callable[[str], str | None] | None = None,
        context: tuple[str, str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Data")
        self._validate = validate
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        if context is not None:
            context_label = QtWidgets.QLabel(context[1], self)
            context_label.setStyleSheet("font-weight: bold;")
            form.addRow(context[0], context_label)

        def line(key: str) -> QtWidgets.QLineEdit:
            edit = QtWidgets.QLineEdit(str(defaults.get(key, "")), self)
            edit.setObjectName(key)
            return edit

        self.edits = {key: line(key) for key in (
            "name", "value", "xstart", "xdelta", "xlabel", "xunits", "ylabel", "yunits",
        )}
        self.points_spin = QtWidgets.QSpinBox(self)
        self.points_spin.setRange(1, 100_000_000)
        self.points_spin.setValue(int(defaults.get("points", 100)))
        self.edits["value"].setToolTip("Initial value of every point, e.g. 0 or NaN")

        form.addRow("Name", self.edits["name"])
        form.addRow("Points", self.points_spin)
        form.addRow("Value", self.edits["value"])
        for axis, keys in (("X", ("xstart", "xdelta")), ("X", ("xlabel", "xunits")),
                           ("Y", ("ylabel", "yunits"))):
            row = QtWidgets.QHBoxLayout()
            for key in keys:
                row.addWidget(QtWidgets.QLabel(key[1:].capitalize(), self))
                row.addWidget(self.edits[key])
            form.addRow(axis, row)

        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setStyleSheet("color: #b91c1c;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.edits["name"].selectAll()
        self.edits["name"].setFocus()

    def values(self) -> dict:
        """The fields, with numbers converted (raises ValueError if invalid)."""
        text = {key: edit.text().strip() for key, edit in self.edits.items()}
        return {
            "name": text["name"],
            "points": self.points_spin.value(),
            "value": float(text["value"] or 0),
            "xstart": float(text["xstart"] or 0),
            "xdelta": float(text["xdelta"] or 1),
            "xlabel": text["xlabel"],
            "xunits": text["xunits"],
            "ylabel": text["ylabel"],
            "yunits": text["yunits"],
        }

    def accept(self) -> None:
        try:
            values = self.values()
        except ValueError:
            error = "Value, Start and Delta must be numbers"
        else:
            error = None
            if values["xdelta"] == 0:
                error = "Delta must not be 0"
            elif self._validate is not None:
                error = self._validate(values["name"])
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


def new_data_defaults(container, points: int = 100) -> dict:
    """Fields for a new data array: the next auto name, *points* points,
    value 0, x from 0 in steps of 1, and no labels or units."""
    return {
        "name": container.auto_name_next(), "points": points, "value": 0,
        "xstart": 0, "xdelta": 1,
        "xlabel": "", "xunits": "", "ylabel": "", "yunits": "",
    }


def ask_new_data(
    defaults: dict,
    validate: Callable[[str], str | None],
    context: tuple[str, str] | None = None,
    parent: QtWidgets.QWidget | None = None,
) -> dict | None:
    """Show a :class:`NewDataDialog`; returns its values, or None if cancelled."""
    dialog = NewDataDialog(defaults, validate, context=context, parent=parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.values()


def create_data(container, values: dict):
    """Add a data array to *container* from New Data dialog *values*."""
    return container.new(
        values["name"],
        nparray=np.full(values["points"], values["value"], dtype=float),
        xscale={"start": values["xstart"], "delta": values["xdelta"],
                "label": values["xlabel"], "units": values["xunits"]},
        yscale={"label": values["ylabel"], "units": values["yunits"]},
    )


def name_error(container: NMObjectContainer, name: str | None) -> str | None:
    """Why *name* cannot be added to *container*, or None if it can."""
    if not name:
        return "Enter a name"
    try:
        container._newkey(name)  # validates without adding
    except KeyError:
        return f"'{name}' already exists"
    except ValueError:
        return f"'{name}' is not a valid name (use letters, digits and _)"
    return None


class TierActions(QtCore.QObject):
    """Base for a tier's actions: QAction creation and shared dialogs.

    Emits :attr:`content_changed` after items are added, renamed or removed,
    so views that list them can reload.
    """

    content_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._selection = selection_model
        self._parent_widget = parent

    def _action(self, text, slot, shortcut=None, tip="") -> QtGui.QAction:
        item = QtGui.QAction(text, self)
        if shortcut is not None:
            item.setShortcut(QtGui.QKeySequence(shortcut))
        item.setStatusTip(tip)
        item.setToolTip(tip or text)
        item.triggered.connect(lambda _checked=False: slot())
        return item

    def update_enabled(self, *_args) -> None:
        """Enable the actions that apply to the current selection."""

    # --- dialogs (replaced in tests) ----------------------------------------

    def _choose(self, title: str, **options) -> tuple | None:
        """Show a :class:`ChooseItemDialog`; returns (item, name, *extras) or None."""
        dialog = ChooseItemDialog(title, parent=self._parent_widget, **options)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return (dialog.item(), dialog.name(), *dialog.extras())

    def _show_message(self, title: str, text: str, warning: bool = False) -> None:
        show = QtWidgets.QMessageBox.warning if warning else QtWidgets.QMessageBox.information
        show(self._parent_widget, title, text)

    def _confirm(self, title: str, text: str) -> bool:
        answer = QtWidgets.QMessageBox.question(self._parent_widget, title, text)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_series_name(self, title: str, hint: str, verb: str) -> bool:
        """Ask before using a name that fits or looks like a data series."""
        return not hint or self._confirm(title, f"{hint}\n\n{verb} anyway?")

    # --- helpers ------------------------------------------------------------

    def _folder(self):
        return self._selection.selection.get("folder")

    def _current_name(self, key: str) -> str | None:
        value = self._selection.selection.get(key)
        if isinstance(value, str) or value is None:
            return value
        return value.name

    def _changed(self) -> None:
        self._selection.refresh()
        self.update_enabled()
        self.content_changed.emit()


class FolderActions(TierActions):
    """Folder and subfolder (toolfolder) actions."""

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        Key = QtGui.QKeySequence.StandardKey
        self.new_action = action("New Folder…", self.new_folder, Key.New,
                                 "Create an empty folder")
        self.open_action = action("Open / Import…", self.open_files, Key.Open,
                                  "Import ABF, AxoGraph or Igor files, one folder per file")
        self.save_action = action("Save…", self.save_folder, Key.Save,
                                  "Save the folder")
        self.duplicate_action = action("Duplicate…", self.duplicate_folder,
                                       tip="Copy the folder with its data")
        self.rename_action = action("Rename…", self.rename_folder,
                                    tip="Rename the folder")
        self.close_action = action("Close…", self.close_folder,
                                   tip="Remove the folder from the session")
        self.subfolder_rename_action = action("Rename…", self.rename_subfolder,
                                              tip="Rename the subfolder")
        self.subfolder_duplicate_action = action("Duplicate…", self.duplicate_subfolder,
                                                 tip="Copy the subfolder")
        self.subfolder_delete_action = action("Delete…", self.delete_subfolder,
                                              tip="Delete the subfolder and its data")

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    # --- menus --------------------------------------------------------------

    def folder_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Folder", parent)
        menu.addAction(self.new_action)
        menu.addAction(self.open_action)
        menu.addAction(self.save_action)
        menu.addSeparator()
        menu.addAction(self.duplicate_action)
        menu.addAction(self.rename_action)
        menu.addSeparator()
        menu.addAction(self.close_action)
        return menu

    def subfolder_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Subfolder", parent)
        menu.addAction(self.subfolder_duplicate_action)
        menu.addAction(self.subfolder_rename_action)
        menu.addSeparator()
        menu.addAction(self.subfolder_delete_action)
        return menu

    def update_enabled(self, *_args) -> None:
        # Actions choose their folder in their dialog, so they only need one
        folder = self._folder()
        has_folders = len(self._folders) > 0
        has_subfolders = folder is not None and len(folder.toolfolders) > 0
        for item in (self.duplicate_action, self.rename_action, self.close_action):
            item.setEnabled(has_folders)
        self.save_action.setEnabled(folder is not None)
        for item in (self.subfolder_rename_action, self.subfolder_duplicate_action,
                     self.subfolder_delete_action):
            item.setEnabled(has_subfolders)

    # --- dialogs (replaced in tests) ----------------------------------------

    def _ask_open_paths(self) -> list[str]:
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self._parent_widget, "Open / Import", "", OPEN_FILTER
        )
        return paths

    # --- helpers ------------------------------------------------------------

    @property
    def _folders(self):
        return self._selection.manager.folders

    def _subfolder(self):
        return self._selection.selection.get("toolfolder")

    def _choose_and_name(self, container, title, item_label, current, default_name):
        """Choose an item of *container* and a new, unused name for it."""
        if len(container) == 0:
            return None
        return self._choose(
            title,
            item_label=item_label,
            items=list(container.keys()),
            current=current,
            name_label="New name",
            default_name=default_name,
            validate=lambda item, name: (
                None if name == item and title.startswith("Rename")
                else name_error(container, name)
            ),
        )

    # --- folder actions -----------------------------------------------------

    def new_folder(self) -> None:
        folders = self._folders
        answer = self._choose(
            "New Folder",
            name_label="Folder name",
            default_name=lambda _item: folders.auto_name_next(),
            validate=lambda _item, name: name_error(folders, name),
        )
        if answer is None:
            return
        folder = folders.new(answer[1])
        self._selection.update(folder=folder)
        self._changed()

    def open_files(self, paths: list[str] | None = None) -> None:
        """Import each file into a new folder named after it."""
        if paths is None:
            paths = self._ask_open_paths()
        imported = None
        problems: list[str] = []
        for path in paths:
            path = Path(path)
            reader = READERS.get(path.suffix.lower())
            if reader is None:
                problems.append(f"{path.name}: unsupported file type")
                continue
            folder = self._folders.new(unique_name(self._folders, path.stem))
            try:
                reader(path, folder)
            except ImportError as error:
                self._folders.pop(folder.name)
                problems.append(f"{path.name}: {error}")
            except Exception as error:  # reader errors vary by format
                self._folders.pop(folder.name)
                problems.append(f"{path.name}: could not read ({error})")
            else:
                imported = folder
        if imported is not None:
            self._selection.update(folder=imported)
            self._changed()
        if problems:
            self._show_message("Open / Import", "\n".join(problems), warning=True)

    def save_folder(self) -> None:
        self._show_message("Save", "Save is under construction.")

    def duplicate_folder(self) -> None:
        folders = self._folders
        answer = self._choose_and_name(
            folders, "Duplicate Folder", "Folder", self._current_name("folder"),
            lambda item: unique_name(folders, f"{item}_copy"),
        )
        if answer is None:
            return
        copy = folders.duplicate(answer[0], answer[1])
        self._selection.update(folder=copy)
        self._changed()

    def rename_folder(self) -> None:
        folders = self._folders
        answer = self._choose_and_name(
            folders, "Rename Folder", "Folder", self._current_name("folder"),
            lambda item: item,
        )
        if answer is None or answer[0] == answer[1]:
            return
        folders.rename(answer[0], answer[1])
        self._changed()

    def close_folder(self) -> None:
        folders = self._folders
        if len(folders) == 0:
            return
        answer = self._choose(
            "Close Folder",
            item_label="Folder",
            items=list(folders.keys()),
            current=self._current_name("folder"),
            message="The folder and its data will be removed from this session.",
        )
        if answer is None:
            return
        name = answer[0]
        was_selected = name == self._current_name("folder")
        names = list(folders.keys())
        index = names.index(name)
        neighbour = (names[index + 1:] + names[:index][::-1])[:1]
        folders.pop(name)
        if was_selected:
            if neighbour:
                self._selection.update(folder=folders[neighbour[0]])
            else:
                self._selection.clear()
        self._changed()

    # --- subfolder actions --------------------------------------------------

    def duplicate_subfolder(self) -> None:
        folder = self._folder()
        if folder is None:
            return
        container = folder.toolfolders
        answer = self._choose_and_name(
            container, "Duplicate Subfolder", "Subfolder",
            self._current_name("toolfolder"),
            lambda item: unique_name(container, f"{item}_copy"),
        )
        if answer is None:
            return
        copy = container.duplicate(answer[0], answer[1])
        self._selection.update(toolfolder=copy)
        self._changed()

    def rename_subfolder(self) -> None:
        folder = self._folder()
        if folder is None:
            return
        container = folder.toolfolders
        answer = self._choose_and_name(
            container, "Rename Subfolder", "Subfolder",
            self._current_name("toolfolder"), lambda item: item,
        )
        if answer is None or answer[0] == answer[1]:
            return
        container.rename(answer[0], answer[1])
        self._changed()

    def delete_subfolder(self) -> None:
        folder = self._folder()
        if folder is None or len(folder.toolfolders) == 0:
            return
        answer = self._choose(
            "Delete Subfolder",
            item_label="Subfolder",
            items=list(folder.toolfolders.keys()),
            current=self._current_name("toolfolder"),
            message="The subfolder and all its data will be deleted.",
        )
        if answer is None:
            return
        name = answer[0]
        if name == self._current_name("toolfolder"):
            self._selection.update(toolfolder=None)
        folder.toolfolders.pop(name)
        self._changed()


def series_name_hint(owner, name: str | None, joins_now: bool = False) -> str:
    """Warning if *name* fits, or looks like, one of *owner*'s data series.

    *owner* is a folder or subfolder. A data series picks up data named
    ``<prefix><channel><epoch>`` when it is updated; renaming data updates
    the matching series straight away (*joins_now*). A name that only starts
    with a series prefix (e.g. ``StimAA`` for series ``Stim``) is flagged too,
    since it looks like part of the series but will not join it.
    """
    series = getattr(owner, "dataseries", None)
    if not name or series is None:
        return ""
    parsed = nmu.parse_data_name(name)
    key = series._getkey(parsed[0]) if parsed is not None else None
    if key is not None:
        if series[key].locked:
            return ""  # a locked series gives series_name_error instead
        _prefix, channel, epoch = parsed
        when = (
            "will join it now" if joins_now
            else "will be added to it the next time the series is updated"
        )
        return (
            f"'{name}' fits data series '{key}' (channel {channel}, epoch {epoch}) "
            f"and {when}."
        )
    # Longest matching prefix, so "Stimulus" wins over "Stim" if both exist
    for prefix in sorted(series.keys(), key=len, reverse=True):
        if name.lower().startswith(prefix.lower()):
            return (
                f"'{name}' starts with the prefix of data series '{prefix}' but "
                f"does not fit its naming pattern (e.g. {prefix}A0), so it will "
                "not be part of the series."
            )
    return ""


def series_name_error(owner, name: str | None) -> str | None:
    """Error if *name* fits the pattern of a locked data series of *owner*.

    Data with that name would join the series, which the lock prevents.
    """
    data = getattr(owner, "data", None)
    if not name or data is None:
        return None
    locked = data.locked_dataseries_for_name(name)
    if locked is None:
        return None
    return (
        f"'{name}' fits locked data series '{locked.name}'. Unlock it, or copy "
        "the series under a new prefix, to add data to it."
    )


def set_name_error(sets, name: str | None) -> str | None:
    """Why *name* cannot be a new set in *sets*, or None if it can."""
    if not name:
        return "Enter a name"
    if not nmu.name_ok(name):
        return f"'{name}' is not a valid name (use letters, digits and _)"
    if name.lower() in (key.lower() for key in sets.keys()):
        return f"'{name}' already exists"
    return None


class DataActions(TierActions):
    """Data and Data Set actions, for the folder's (or subfolder's) data.

    Renaming or deleting data in a data series removes it from the series;
    data in a locked series cannot be renamed or deleted.
    """

    # Ask the app to show a folder's (or subfolder's) Data list in the browser,
    # or its data Sets list (True)
    browse_requested = QtCore.pyqtSignal(object, bool)
    # Ask the app to show the current data array in the Table panel
    table_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        self.new_action = action("New…", self.new_data,
                                 tip="Create a data array and open it in the Table")
        self.duplicate_action = action("Duplicate…", self.duplicate_data,
                                       tip="Copy a data array in this folder")
        self.rename_action = action("Rename…", self.rename_data, tip="Rename a data array")
        self.copy_action = action("Copy to Folder…", self.copy_data_to_folder,
                                  tip="Copy a data array into another folder")
        self.delete_action = action("Delete…", self.delete_data, tip="Delete a data array")
        self.browse_action = action(
            "Choose in Browser…", self.choose_in_browser,
            tip="Select several data arrays in the browser (Shift/Cmd-click)",
        )
        self.set_new_action = action("New Set…", self.new_set, tip="Create an empty data set")
        self.set_add_action = action("Add Data to Set…", self.add_to_set,
                                     tip="Add a data array to a data set")
        self.set_remove_action = action("Remove Data from Set…", self.remove_from_set,
                                        tip="Remove a data array from a data set")
        self.set_combine_action = action("Combine Sets…", self.combine_sets,
                                         tip="Make a set from two sets with AND or OR")
        self.set_clear_action = action("Clear Set…", self.clear_set,
                                       tip="Remove all data from a set, keeping the set")
        self.set_rename_action = action("Rename Set…", self.rename_set, tip="Rename a data set")
        self.set_delete_action = action("Delete Set…", self.delete_set, tip="Delete a data set")
        self.set_browse_action = action(
            "Choose in Browser…", self.choose_sets_in_browser,
            tip="Show the data sets in the browser",
        )

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    # --- menus --------------------------------------------------------------

    def data_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Data", parent)
        menu.addAction(self.new_action)
        menu.addSeparator()
        menu.addAction(self.duplicate_action)
        menu.addAction(self.rename_action)
        menu.addAction(self.copy_action)
        menu.addSeparator()
        menu.addAction(self.delete_action)
        menu.addSeparator()
        menu.addAction(self.browse_action)
        return menu

    def data_set_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Data Set", parent)
        menu.addAction(self.set_new_action)
        menu.addAction(self.set_add_action)
        menu.addAction(self.set_remove_action)
        menu.addAction(self.set_combine_action)
        menu.addSeparator()
        menu.addAction(self.set_clear_action)
        menu.addAction(self.set_rename_action)
        menu.addAction(self.set_delete_action)
        menu.addSeparator()
        menu.addAction(self.set_browse_action)
        return menu

    def update_enabled(self, *_args) -> None:
        data = self._data_container()
        has_data = data is not None and len(data) > 0
        has_sets = data is not None and len(list(data.sets.keys())) > 0
        for item in (self.duplicate_action, self.rename_action, self.delete_action):
            item.setEnabled(has_data)
        self.copy_action.setEnabled(has_data and len(self._selection.manager.folders) > 1)
        self.browse_action.setEnabled(data is not None)
        self.new_action.setEnabled(data is not None)
        self.set_browse_action.setEnabled(data is not None)
        self.set_new_action.setEnabled(data is not None)
        for item in (self.set_add_action, self.set_remove_action, self.set_combine_action,
                     self.set_clear_action, self.set_rename_action, self.set_delete_action):
            item.setEnabled(has_sets)
        self.set_add_action.setEnabled(has_sets and has_data)
        self.set_remove_action.setEnabled(has_sets and has_data)
        # Combining needs two different sets
        self.set_combine_action.setEnabled(
            data is not None and len(list(data.sets.keys())) > 1
        )

    # --- helpers ------------------------------------------------------------

    def _choose(self, title: str, **options) -> tuple | None:
        # Every data dialog shows the folder (and subfolder) it applies to
        options.setdefault("context", ("Folder", self._context_text()))
        return super()._choose(title, **options)

    def _context_text(self) -> str:
        folder = self._folder()
        subfolder = self._selection.selection.get("toolfolder")
        names = [obj.name for obj in (folder, subfolder) if obj is not None]
        return " / ".join(names)

    def _context(self):
        """The subfolder if one is selected, else the folder."""
        return self._selection.selection.get("toolfolder") or self._folder()

    def _data_container(self):
        context = self._context()
        return context.data if context is not None else None

    @staticmethod
    def _series_note(container, name: str | None) -> str:
        if name is None or name not in container:
            return ""
        series = container.dataseries_of(name)
        if not series:
            return ""
        names = ", ".join(f"'{ds.name}'" for ds in series)
        if any(ds.locked for ds in series):
            return f"Part of locked data series {names}."
        return f"Part of data series {names}; it will be removed from the series."

    @staticmethod
    def _lock_error(container, name: str | None) -> str | None:
        if name is None or name not in container:
            return None
        for ds in container.dataseries_of(name):
            if ds.locked:
                return (f"'{name}' is part of locked data series '{ds.name}'. "
                        "Unlock the series to change its data.")
        return None

    # --- data actions -------------------------------------------------------

    def _ask_new_data(self, defaults: dict, validate) -> dict | None:
        """The New Data dialog (replaced in tests)."""
        return ask_new_data(
            defaults, validate, context=("Folder", self._context_text()),
            parent=self._parent_widget,
        )

    def new_data(self) -> None:
        container = self._data_container()
        if container is None:
            return
        # Match the current array's length; scales start from the defaults
        current = self._selection.selection.get("data")
        points = current.nparray.size if current is not None and current.nparray is not None else 100
        values = self._ask_new_data(
            new_data_defaults(container, points),
            lambda name: name_error(container, name)
            or series_name_error(self._context(), name),
        )
        if values is None:
            return
        if not self._confirm_series_name(
            "New Data", series_name_hint(self._context(), values["name"]), "Create"
        ):
            return
        data = create_data(container, values)
        self._selection.update(data=data)
        self._changed()
        self.table_requested.emit()

    def duplicate_data(self) -> None:
        container = self._data_container()
        if container is None or len(container) == 0:
            return
        answer = self._choose(
            "Duplicate Data",
            item_label="Data",
            items=list(container.keys()),
            current=self._current_name("data"),
            name_label="New name",
            default_name=lambda item: copy_name(container, item),
            validate=lambda item, name: (
                name_error(container, name) or series_name_error(self._context(), name)
            ),
        )
        if answer is None:
            return
        if not self._confirm_series_name(
            "Duplicate Data", series_name_hint(self._context(), answer[1]), "Duplicate"
        ):
            return
        copy = container.duplicate(answer[0], answer[1])
        self._selection.update(data=copy)
        self._changed()

    def rename_data(self) -> None:
        container = self._data_container()
        if container is None or len(container) == 0:
            return
        answer = self._choose(
            "Rename Data",
            item_label="Data",
            items=list(container.keys()),
            current=self._current_name("data"),
            name_label="New name",
            default_name=lambda item: item,
            message=lambda item: self._series_note(container, item),
            validate=lambda item, name: (
                self._lock_error(container, item)
                or (None if name == item else (
                    name_error(container, name)
                    or series_name_error(self._context(), name)
                ))
            ),
        )
        if answer is None or answer[0] == answer[1]:
            return
        if not self._confirm_series_name(
            "Rename Data",
            series_name_hint(self._context(), answer[1], joins_now=True),
            "Rename",
        ):
            return
        container.rename(answer[0], answer[1])
        self._changed()

    def copy_data_to_folder(self) -> None:
        container = self._data_container()
        folders = self._selection.manager.folders
        here = self._folder()
        targets = [name for name in folders.keys() if here is None or name != here.name]
        if container is None or len(container) == 0 or not targets:
            return
        answer = self._choose(
            "Copy Data to Folder",
            context=("From folder", self._context_text()),
            item_label="Data",
            items=list(container.keys()),
            current=self._current_name("data"),
            extras=[("To folder", targets, targets[0])],
            name_label="Name",
            default_name=lambda item, target: copy_target_name(folders[target].data, item),
            validate=lambda item, name, target: (
                name_error(folders[target].data, name)
                or series_name_error(folders[target], name)
            ),
        )
        if answer is None:
            return
        item, name, target = answer
        if not self._confirm_series_name(
            "Copy Data to Folder", series_name_hint(folders[target], name), "Copy"
        ):
            return
        container.copy_to(item, folders[target].data, name)
        self._changed()

    def delete_data(self) -> None:
        container = self._data_container()
        if container is None or len(container) == 0:
            return
        answer = self._choose(
            "Delete Data",
            item_label="Data",
            items=list(container.keys()),
            current=self._current_name("data"),
            message=lambda item: self._series_note(container, item)
            or "The data array will be deleted.",
            validate=lambda item, name: self._lock_error(container, item),
        )
        if answer is None:
            return
        container.pop(answer[0])
        self._changed()

    def choose_in_browser(self) -> None:
        context = self._context()
        if context is not None:
            self.browse_requested.emit(context, False)

    def choose_sets_in_browser(self) -> None:
        context = self._context()
        if context is not None:
            self.browse_requested.emit(context, True)

    # --- data set actions ---------------------------------------------------

    def _choose_set(self, title: str, **options):
        container = self._data_container()
        if container is None or not list(container.sets.keys()):
            return None
        return self._choose(
            title,
            item_label="Data Set",
            items=list(container.sets.keys()),
            current=self._current_name("data_set"),
            **options,
        )

    def new_set(self) -> None:
        container = self._data_container()
        if container is None:
            return
        sets = container.sets
        answer = self._choose(
            "New Data Set",
            name_label="Set name",
            default_name=lambda _item: sets.auto_name_next(),
            validate=lambda _item, name: set_name_error(sets, name),
        )
        if answer is None:
            return
        sets.add(answer[1])
        self._selection.update(data_set=answer[1])
        self._changed()

    def add_to_set(self) -> None:
        container = self._data_container()
        answer = self._choose_set(
            "Add Data to Set",
            extras=[("Data", list(container.keys()) if container else [],
                     self._current_name("data"))],
        )
        if answer is None:
            return
        set_name, _name, data_name = answer
        container.sets.add(set_name, [container[data_name]])
        self._changed()

    def remove_from_set(self) -> None:
        container = self._data_container()

        def not_member(set_name, _name, data_name):
            if container.sets.contains(set_name, [container[data_name]]):
                return None
            return f"'{data_name}' is not in '{set_name}'"

        answer = self._choose_set(
            "Remove Data from Set",
            extras=[("Data", list(container.keys()) if container else [],
                     self._current_name("data"))],
            validate=not_member,
        )
        if answer is None:
            return
        set_name, _name, data_name = answer
        container.sets.remove(set_name, [container[data_name]])
        self._changed()

    def combine_sets(self) -> None:
        container = self._data_container()
        if container is None:
            return
        sets = container.sets
        names = list(sets.keys())
        if not names:
            return
        current = self._current_name("data_set")
        first = current if current in names else names[0]
        # Start with a different set, as combining a set with itself does nothing
        second = next((n for n in names if n != first), first)
        answer = self._choose_set(
            "Combine Sets",
            extras=[("Operator", ["AND", "OR"], "OR"),
                    ("With Set", names, second)],
            name_label="New Set",
            default_name=lambda _item, *_extras: sets.auto_name_next(),
            validate=lambda item, name, _operator, other: (
                "Choose two different sets" if item == other
                else set_name_error(sets, name)
            ),
        )
        if answer is None:
            return
        first, name, operator, second = answer
        define = sets.define_and if operator == "AND" else sets.define_or
        define(name, first, second)
        self._selection.update(data_set=name)
        self._changed()

    def clear_set(self) -> None:
        answer = self._choose_set(
            "Clear Set", message="All data will be removed from the set; the set is kept."
        )
        if answer is None:
            return
        self._data_container().sets.empty(answer[0])
        self._changed()

    def rename_set(self) -> None:
        container = self._data_container()
        sets = container.sets if container is not None else None
        answer = self._choose_set(
            "Rename Set",
            name_label="New name",
            default_name=lambda item: item,
            validate=lambda item, name: None if name == item else set_name_error(sets, name),
        )
        if answer is None or answer[0] == answer[1]:
            return
        was_current = answer[0] == self._current_name("data_set")
        sets.rename(answer[0], answer[1])
        if was_current:
            self._selection.update(data_set=answer[1])
        self._changed()

    def delete_set(self) -> None:
        answer = self._choose_set(
            "Delete Set", message="The set will be deleted; its data is kept."
        )
        if answer is None:
            return
        if answer[0] == self._current_name("data_set"):
            self._selection.update(data_set=None)
        self._data_container().sets.pop(answer[0])
        self._changed()


__all__ = [
    "ChooseItemDialog",
    "DataActions",
    "NewDataDialog",
    "FolderActions",
    "TierActions",
    "COPY_PREFIX",
    "ask_new_data",
    "create_data",
    "new_data_defaults",
    "OPEN_FILTER",
    "READERS",
    "copy_name",
    "copy_target_name",
    "name_error",
    "series_name_error",
    "series_name_hint",
    "set_name_error",
    "unique_name",
]
