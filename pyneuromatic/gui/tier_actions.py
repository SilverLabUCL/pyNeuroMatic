"""Actions for selection tiers, shared by the strip's ⋯ menus and the menu bar.

Each tier's actions are QActions created once; a strip ⋯ button and a menu
bar menu both show the same objects, so enabled state and shortcuts agree.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

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


class ChooseItemDialog(QtWidgets.QDialog):
    """Choose an item (e.g. a folder) and/or enter a name, in one dialog.

    *items* adds an item dropdown, starting at *current*. *name_label* adds a
    name field; *default_name(item)* fills it whenever the item changes.
    *validate(item, name)* returns an error message, shown inline while the
    dialog stays open, or None to accept.
    """

    def __init__(
        self,
        title: str,
        item_label: str | None = None,
        items: list[str] | None = None,
        current: str | None = None,
        name_label: str | None = None,
        default_name: Callable[[str | None], str] | None = None,
        validate: Callable[[str | None, str | None], str | None] | None = None,
        message: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._default_name = default_name
        self._validate = validate
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.item_combo: QtWidgets.QComboBox | None = None
        if items is not None:
            self.item_combo = QtWidgets.QComboBox(self)
            self.item_combo.addItems(items)
            if current in items:
                self.item_combo.setCurrentText(current)
            form.addRow(item_label or "Item", self.item_combo)
        self.name_edit: QtWidgets.QLineEdit | None = None
        if name_label is not None:
            self.name_edit = QtWidgets.QLineEdit(self)
            form.addRow(name_label, self.name_edit)
        if message:
            note = QtWidgets.QLabel(message, self)
            note.setWordWrap(True)
            layout.addWidget(note)
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
        self._fill_default_name()
        if self.name_edit is not None:
            self.name_edit.selectAll()
            self.name_edit.setFocus()

    def item(self) -> str | None:
        return self.item_combo.currentText() if self.item_combo is not None else None

    def name(self) -> str | None:
        return self.name_edit.text().strip() if self.name_edit is not None else None

    def _fill_default_name(self, *_args) -> None:
        if self.name_edit is not None and self._default_name is not None:
            self.name_edit.setText(self._default_name(self.item()))
        self.error_label.hide()

    def accept(self) -> None:
        error = self._validate(self.item(), self.name()) if self._validate else None
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


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


class FolderActions(QtCore.QObject):
    """Folder and subfolder (toolfolder) actions.

    Emits :attr:`content_changed` after folders or subfolders are added,
    renamed or removed, so views that list them can reload.
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

        def action(text, slot, shortcut=None, tip=""):
            item = QtGui.QAction(text, self)
            if shortcut is not None:
                item.setShortcut(QtGui.QKeySequence(shortcut))
            item.setStatusTip(tip)
            item.setToolTip(tip or text)
            item.triggered.connect(lambda _checked=False: slot())
            return item

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

    def _choose(self, title: str, **options) -> tuple[str | None, str | None] | None:
        """Show a :class:`ChooseItemDialog`; returns (item, name) or None."""
        dialog = ChooseItemDialog(title, parent=self._parent_widget, **options)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.item(), dialog.name()

    def _ask_open_paths(self) -> list[str]:
        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self._parent_widget, "Open / Import", "", OPEN_FILTER
        )
        return paths

    def _show_message(self, title: str, text: str, warning: bool = False) -> None:
        show = QtWidgets.QMessageBox.warning if warning else QtWidgets.QMessageBox.information
        show(self._parent_widget, title, text)

    # --- helpers ------------------------------------------------------------

    @property
    def _folders(self):
        return self._selection.manager.folders

    def _folder(self):
        return self._selection.selection.get("folder")

    def _subfolder(self):
        return self._selection.selection.get("toolfolder")

    def _current_name(self, key: str) -> str | None:
        value = self._selection.selection.get(key)
        return value.name if value is not None else None

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

    def _changed(self) -> None:
        self._selection.refresh()
        self.update_enabled()
        self.content_changed.emit()

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
        copy = folders.duplicate(*answer)
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
        folders.rename(*answer)
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
        copy = container.duplicate(*answer)
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
        container.rename(*answer)
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


__all__ = [
    "ChooseItemDialog",
    "FolderActions",
    "OPEN_FILTER",
    "READERS",
    "name_error",
    "unique_name",
]
