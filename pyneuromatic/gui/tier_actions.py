"""Actions for selection tiers, shared by the strip's ⋯ menus and the menu bar.

Each tier's actions are QActions created once; a strip ⋯ button and a menu
bar menu both show the same objects, so enabled state and shortcuts agree.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

import numpy as np

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_epoch import EXCLUDE_SET
from pyneuromatic.gui.selection_model import BUILTIN_EPOCH_SETS, SelectionModel


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


@dataclasses.dataclass(frozen=True)
class Spin:
    """An integer field with up/down arrows, for :class:`ChooseItemDialog`
    extras: ``(label, Spin(minimum, maximum, value, prefix))``. Its value is
    returned as text with the prefix, e.g. ``"E250"``."""

    minimum: int
    maximum: int
    value: int
    prefix: str = ""


class ChooseItemDialog(QtWidgets.QDialog):
    """Choose an item (e.g. a folder) and/or enter a name, in one dialog.

    *items* adds an item dropdown, starting at *current*. *name_label* adds a
    name field; *default_name(item, *extras)* fills it whenever the item or
    an extra choice changes.
    *extras* adds further dropdowns, as ``(label, items, current)``, or
    number boxes, as ``(label, Spin(...))``.
    *message* is a note below the fields, or a function of the item and
    extra choices, updated when they change.
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
        # Dropdowns (QComboBox) or number boxes (QSpinBox), in order
        self.extra_widgets: list[QtWidgets.QWidget] = []
        for extra in extras or []:
            label = extra[0]
            if isinstance(extra[1], Spin):
                spec = extra[1]
                widget = QtWidgets.QSpinBox(self)
                widget.setRange(spec.minimum, spec.maximum)
                widget.setValue(spec.value)
                widget.setPrefix(spec.prefix)
                widget.setKeyboardTracking(False)  # update after typing
            else:
                widget = QtWidgets.QComboBox(self)
                widget.addItems(extra[1])
                if extra[2] in extra[1]:
                    widget.setCurrentText(extra[2])
            form.addRow(label, widget)
            self.extra_widgets.append(widget)
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
        for widget in self.extra_widgets:
            if isinstance(widget, QtWidgets.QSpinBox):
                widget.valueChanged.connect(self._fill_default_name)
            else:
                widget.currentTextChanged.connect(self._fill_default_name)
        self._fill_default_name()
        if self.name_edit is not None:
            self.name_edit.selectAll()
            self.name_edit.setFocus()

    def item(self) -> str | None:
        return self.item_combo.currentText() if self.item_combo is not None else None

    def name(self) -> str | None:
        return self.name_edit.text().strip() if self.name_edit is not None else None

    @property
    def extra_combos(self) -> list[QtWidgets.QComboBox]:
        """The extra dropdowns (not number boxes)."""
        return [w for w in self.extra_widgets if isinstance(w, QtWidgets.QComboBox)]

    def extras(self) -> list[str]:
        return [
            f"{w.prefix()}{w.value()}" if isinstance(w, QtWidgets.QSpinBox) else w.currentText()
            for w in self.extra_widgets
        ]

    def _fill_default_name(self, *_args) -> None:
        if self.name_edit is not None and self._default_name is not None:
            self.name_edit.setText(self._default_name(self.item(), *self.extras()))
        message = (
            self._message(self.item(), *self.extras())
            if callable(self._message) else self._message
        )
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
    """Name, size, fill value and scales for a new data array, or with
    *series*, a prefix, channel and epoch counts and noise for a data series.

    *defaults* fills the fields (keys as in :meth:`values`); *validate(values)*
    returns an error, shown inline while the dialog stays open.
    """

    def __init__(
        self,
        defaults: dict,
        validate: Callable[[dict], str | None] | None = None,
        context: tuple[str, str] | None = None,
        series: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Data Series" if series else "New Data")
        self._validate = validate
        self._series = series
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

        def spin(key: str, low: int, high: int, default: int) -> QtWidgets.QSpinBox:
            box = QtWidgets.QSpinBox(self)
            box.setObjectName(key)
            box.setRange(low, high)
            box.setValue(int(defaults.get(key, default)))
            return box

        keys = ["name", "value", "xstart", "xdelta", "xlabel", "xunits", "ylabel", "yunits"]
        if series:
            keys.append("noise")
        self.edits = {key: line(key) for key in keys}
        self.points_spin = spin("points", 1, 100_000_000, 100)
        self.edits["value"].setToolTip("Initial value of every point, e.g. 0 or NaN")

        if series:
            self.channels_spin = spin("n_channels", 1, 26, 1)
            self.epochs_spin = spin("n_epochs", 1, 1_000_000, 10)
            self.edits["noise"].setToolTip(
                "Standard deviation of Gaussian noise added to every point (0: none)"
            )
            form.addRow("Prefix", self.edits["name"])
            counts = QtWidgets.QHBoxLayout()
            counts.addWidget(self.channels_spin)
            counts.addWidget(QtWidgets.QLabel("Epochs", self))
            counts.addWidget(self.epochs_spin)
            form.addRow("Channels", counts)
            form.addRow("Points", self.points_spin)
            fill = QtWidgets.QHBoxLayout()
            fill.addWidget(self.edits["value"])
            fill.addWidget(QtWidgets.QLabel("Noise SD", self))
            fill.addWidget(self.edits["noise"])
            form.addRow("Value", fill)
        else:
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
        values = {} if not self._series else {
            "n_channels": self.channels_spin.value(),
            "n_epochs": self.epochs_spin.value(),
            "noise": float(text["noise"] or 0),
        }
        return values | {
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
            error = (
                "Value, Noise SD, Start and Delta must be numbers" if self._series
                else "Value, Start and Delta must be numbers"
            )
        else:
            error = None
            if values["xdelta"] == 0:
                error = "Delta must not be 0"
            elif values.get("noise", 0) < 0:
                error = "Noise SD must not be negative"
            elif self._validate is not None:
                error = self._validate(values)
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


def common_value(values) -> str | None:
    """The shared value of *values* as text, or None if they differ ("mixed")."""
    values = set(values)
    if len(values) > 1:
        return None
    if not values:
        return ""
    value = values.pop()
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def scale_fields(arrays) -> dict[str, str | None]:
    """Shared x start/delta/label/units of *arrays* (None where they differ)."""
    return {
        "xstart": common_value(d.xscale.start for d in arrays),
        "xdelta": common_value(d.xscale.delta for d in arrays),
        "xlabel": common_value(d.xscale.label for d in arrays),
        "xunits": common_value(d.xscale.units for d in arrays),
    }


class EditScalesDialog(QtWidgets.QDialog):
    """Edit x start/delta/label/units and the y label/units of each row
    (each channel of a data series, or a single array).

    Fields start at the current values; a field whose arrays differ is
    blank, marked "(mixed)". Only fields you change are applied.
    *error* (e.g. a locked series) is shown when OK is pressed.
    """

    def __init__(
        self,
        x: dict[str, str | None],
        y_rows: list[tuple[str, str | None, str | None]],
        context: tuple[str, str] | None = None,
        error: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Scales")
        self._error = error
        self._initial: dict = {}
        layout = QtWidgets.QVBoxLayout(self)
        if context is not None:
            form = QtWidgets.QFormLayout()
            context_label = QtWidgets.QLabel(context[1], self)
            context_label.setStyleSheet("font-weight: bold;")
            form.addRow(context[0], context_label)
            layout.addLayout(form)

        def edit(key, value) -> QtWidgets.QLineEdit:
            line = QtWidgets.QLineEdit(value or "", self)
            line.setObjectName(str(key))
            if value is None:
                line.setPlaceholderText("(mixed)")
            self._initial[key] = line.text()
            return line

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("<b>X</b>", self), 0, 0)
        self.x_edits = {key: edit(key, x.get(key)) for key in
                        ("xstart", "xdelta", "xlabel", "xunits")}
        for row, (first, second) in enumerate((("xstart", "xdelta"), ("xlabel", "xunits")), 1):
            for col, key in ((0, first), (2, second)):
                grid.addWidget(QtWidgets.QLabel(key[1:].capitalize(), self), row, col)
                grid.addWidget(self.x_edits[key], row, col + 1)
        grid.addWidget(QtWidgets.QLabel("<b>Y</b>", self), 3, 0)
        self.y_edits: dict[str, dict[str, QtWidgets.QLineEdit]] = {}
        multi = len(y_rows) > 1
        if multi:
            grid.addWidget(QtWidgets.QLabel("Channel", self), 4, 0)
            grid.addWidget(QtWidgets.QLabel("Label", self), 4, 1)
        for offset, (name, label, units) in enumerate(y_rows, 5 if multi else 4):
            self.y_edits[name] = {
                "label": edit((name, "label"), label),
                "units": edit((name, "units"), units),
            }
            # One row per channel, labelled by the channel; a single array's
            # row is labelled "Label"
            grid.addWidget(QtWidgets.QLabel(name if multi else "Label", self), offset, 0)
            grid.addWidget(self.y_edits[name]["label"], offset, 1)
            grid.addWidget(QtWidgets.QLabel("Units", self), offset, 2)
            grid.addWidget(self.y_edits[name]["units"], offset, 3)
        layout.addLayout(grid)

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

    def changes(self) -> dict:
        """Changed fields: ``xstart``... (numbers converted) and ``y`` as
        ``{row: {"label": ..., "units": ...}}``. Raises ValueError if a
        changed Start or Delta is not a number."""
        changed: dict = {}
        for key, line in self.x_edits.items():
            if line.text() != self._initial[key]:
                text = line.text().strip()
                changed[key] = float(text) if key in ("xstart", "xdelta") else text
        y: dict = {}
        for name, edits in self.y_edits.items():
            for field, line in edits.items():
                if line.text() != self._initial[(name, field)]:
                    y.setdefault(name, {})[field] = line.text().strip()
        if y:
            changed["y"] = y
        return changed

    def accept(self) -> None:
        try:
            changed = self.changes()
        except ValueError:
            error = "Start and Delta must be numbers"
        else:
            error = self._error
            if error is None and changed.get("xdelta", 1) == 0:
                error = "Delta must not be 0"
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
    validate: Callable[[dict], str | None],
    context: tuple[str, str] | None = None,
    parent: QtWidgets.QWidget | None = None,
    series: bool = False,
) -> dict | None:
    """Show a :class:`NewDataDialog`; returns its values, or None if cancelled."""
    dialog = NewDataDialog(defaults, validate, context=context, series=series, parent=parent)
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

    def _ask_scales(self, x, y_rows, context, error) -> dict | None:
        """The Edit Scales dialog (replaced in tests); returns the changes."""
        dialog = EditScalesDialog(x, y_rows, context, error, parent=self._parent_widget)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.changes()

    def _confirm(self, title: str, text: str) -> bool:
        answer = QtWidgets.QMessageBox.question(self._parent_widget, title, text)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_series_name(self, title: str, hint: str, verb: str) -> bool:
        """Ask before using a name that fits or looks like a data series."""
        return not hint or self._confirm(title, f"{hint}\n\n{verb} anyway?")

    # --- helpers ------------------------------------------------------------

    def _folder(self):
        return self._selection.selection.get("folder")

    def _context(self):
        """The subfolder if one is selected, else the folder."""
        return self._selection.selection.get("toolfolder") or self._folder()

    def _context_text(self) -> str:
        folder = self._folder()
        subfolder = self._selection.selection.get("toolfolder")
        names = [obj.name for obj in (folder, subfolder) if obj is not None]
        return " / ".join(names)

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
        self.scales_action = action("Edit Scales…", self.edit_scales,
                                    tip="Edit the x and y labels, units, start and delta")
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
        menu.addAction(self.scales_action)
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
        self.scales_action.setEnabled(self._selection.selection.get("data") is not None)
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
            lambda values: name_error(container, values["name"])
            or series_name_error(self._context(), values["name"]),
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
            message=lambda item, *_: self._series_note(container, item),
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

    def edit_scales(self) -> None:
        data = self._selection.selection.get("data")
        container = self._data_container()
        if data is None or container is None:
            return
        lock_error = self._lock_error(container, data.name)
        changes = self._ask_scales(
            scale_fields([data]),
            [("Y", data.yscale.label, data.yscale.units)],
            ("Data", f"{self._context_text()} › {data.name}"),
            lock_error,
        )
        if not changes or lock_error:  # the dialog only shows a locked array
            return
        for key, attribute in (("xstart", "start"), ("xdelta", "delta"),
                               ("xlabel", "label"), ("xunits", "units")):
            if key in changes:
                setattr(data.xscale, attribute, changes[key])
        for field, value in changes.get("y", {}).get("Y", {}).items():
            setattr(data.yscale, field, value)
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
            message=lambda item, *_: self._series_note(container, item)
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


def prefix_error(owner, prefix: str | None, suffixes: list[str] | None = None,
                 renaming=None) -> str | None:
    """Why *prefix* cannot name a new data series in *owner*, or None.

    *suffixes* are the channel+epoch endings of its data (e.g. ``A0``); a
    data name they would make that already exists is an error, except for
    data of the series being renamed (*renaming*).
    """
    if not prefix:
        return "Enter a prefix"
    if not nmu.name_ok(prefix):
        return f"'{prefix}' is not a valid prefix (use letters, digits and _)"
    if owner.dataseries._getkey(prefix) is not None:
        return f"Data series '{prefix}' already exists"
    own = set()
    if renaming is not None:
        own = {d.name.lower() for ch in renaming.channels.values() for d in ch.data}
    for suffix in suffixes or []:
        key = owner.data._getkey(prefix + suffix)
        if key is not None and key.lower() not in own:
            return f"Data '{prefix + suffix}' already exists"
    return None


def series_suffixes(dataseries) -> list[str]:
    """Channel+epoch endings of a series' data names (``A0``, ``B12``, ...)."""
    names = {d.name for ch in dataseries.channels.values() for d in ch.data}
    return sorted(n[len(dataseries.name):] for n in names
                  if n.lower().startswith(dataseries.name.lower()))


def epoch_ranges(numbers: list[int]) -> str:
    """Compact epoch list: [0, 1, 2, 5] -> "E0–E2, E5"."""
    parts, numbers = [], sorted(numbers)
    start = previous = None
    for number in numbers + [None]:
        if number is not None and previous is not None and number == previous + 1:
            previous = number
            continue
        if start is not None:
            parts.append(f"E{start}" if start == previous else f"E{start}–E{previous}")
        start = previous = number
    return ", ".join(parts)


def series_update_preview(owner, series) -> dict:
    """What updating *series* from the folder's data would add.

    Returns ``new_channels`` (channel letters), ``new_epochs`` (epoch numbers)
    and ``new_arrays`` (data names) not yet in the series.
    """
    result = owner._scan_dataseries(series.name)
    matches = result[1] if result is not None else {}
    linked = {id(d) for channel in series.channels.values() for d in channel.data}
    existing_epochs = set()
    for name in series.epochs.keys():
        try:
            existing_epochs.add(int(name[1:]))
        except ValueError:
            pass
    new = {key: data for key, data in matches.items() if id(data) not in linked}
    return {
        "new_channels": sorted({ch for ch, _ in new} - set(series.channels.keys())),
        "new_epochs": sorted({ep for _, ep in new} - existing_epochs),
        "new_arrays": sorted(data.name for data in new.values()),
    }


class SeriesActions(TierActions):
    """Data series (Prefix) actions, for the folder's data series."""

    # Ask the app to select a data series in the browser
    browse_requested = QtCore.pyqtSignal(object)

    KEEP, DELETE = "Keep", "Delete"

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        self.new_action = action("New Series…", self.new_series,
                                 tip="Create a data series of new arrays, with optional noise")
        self.from_data_action = action(
            "Series from Data…", self.series_from_data,
            tip="Make a data series from arrays named <prefix><channel><epoch>",
        )
        self.update_action = action("Update from Data", self.update_series,
                                    tip="Add newly named or imported arrays to the series")
        self.copy_action = action("Copy Series…", self.copy_series,
                                  tip="Copy the series under a new prefix (the copy is unlocked)")
        self.rename_action = action("Rename Series…", self.rename_series,
                                    tip="Rename the series prefix and its arrays")
        self.scales_action = action("Edit Scales…", self.edit_scales,
                                    tip="Edit the series' x and y labels, units, start and delta")
        self.lock_action = action("Lock Series", self.toggle_lock,
                                  tip="A locked series' arrays cannot be changed")
        self.lock_action.setCheckable(True)
        self.remove_channel_action = action("Remove Channel…", self.remove_channel,
                                            tip="Remove a channel from the series")
        self.remove_epoch_action = action("Remove Epochs…", self.remove_epoch,
                                          tip="Remove an epoch, or a range of epochs")
        self.delete_action = action("Delete Series…", self.delete_series,
                                    tip="Delete the series, optionally with its arrays")
        self.browse_action = action("Choose in Browser…", self.choose_in_browser,
                                    tip="Select the series in the browser")

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def series_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Series", parent)
        for item in (self.new_action, self.from_data_action, self.update_action, None,
                     self.copy_action, self.rename_action, self.scales_action, None,
                     self.lock_action, None,
                     self.remove_channel_action, self.remove_epoch_action,
                     self.delete_action, None, self.browse_action):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        return menu

    # --- helpers ------------------------------------------------------------

    def _owner(self):
        """Folder (or subfolder) whose data series these actions edit."""
        context = self._context()
        return context if hasattr(context, "new_dataseries") else None

    def _series(self):
        series = self._selection.selection.get("dataseries")
        owner = self._owner()
        if series is None or owner is None or series.name not in owner.dataseries:
            return None
        return series

    def _candidate_prefixes(self) -> list[str]:
        owner = self._owner()
        if owner is None:
            return []
        return [p for p in owner.detect_data_prefixes() if p not in owner.dataseries]

    def _locked_actions(self) -> tuple:
        """Actions that change the *current* series, greyed out while it is
        locked. Actions whose dialog chooses the series (Update, Rename,
        Delete) check the lock of the chosen series in the dialog instead."""
        return (self.scales_action, self.remove_channel_action, self.remove_epoch_action)

    def update_enabled(self, *_args) -> None:
        owner = self._owner()
        series = self._series()
        has_series = owner is not None and len(owner.dataseries) > 0
        locked = series is not None and series.locked
        self.new_action.setEnabled(owner is not None)
        self.from_data_action.setEnabled(bool(self._candidate_prefixes()))
        for item in (self.copy_action, self.rename_action, self.delete_action,
                     self.browse_action):
            item.setEnabled(has_series)
        for item in (self.update_action, self.lock_action, self.remove_channel_action,
                     self.remove_epoch_action, self.scales_action):
            item.setEnabled(series is not None)
        self.lock_action.setChecked(locked)
        if not hasattr(self, "_tips"):
            self._tips = {item: item.toolTip() for item in self._locked_actions()}
        for item in self._locked_actions():
            if locked:
                item.setEnabled(False)
                item.setToolTip(f"Data series '{series.name}' is locked.")
            else:
                item.setToolTip(self._tips[item])

    def _series_context(self, series) -> tuple[str, str]:
        return ("Series", f"{self._context_text()} › {series.name}")

    @staticmethod
    def _lock_error(owner, name: str | None) -> str | None:
        series = owner.dataseries.get(name) if name else None
        if series is not None and series.locked:
            return (f"Data series '{name}' is locked. Untick Lock Series "
                    "to change it.")
        return None

    def _choose_series(self, title: str, **options):
        owner = self._owner()
        if owner is None or len(owner.dataseries) == 0:
            return None
        current = self._series()
        options.setdefault("context", ("Folder", self._context_text()))
        return self._choose(
            title, item_label="Series", items=list(owner.dataseries.keys()),
            current=current.name if current is not None else None, **options,
        )

    # --- actions ------------------------------------------------------------

    def _ask_new_series(self, defaults: dict, validate) -> dict | None:
        """The New Data Series dialog (replaced in tests)."""
        return ask_new_data(
            defaults, validate, context=("Folder", self._context_text()),
            parent=self._parent_widget, series=True,
        )

    def new_series(self) -> None:
        owner = self._owner()
        if owner is None:
            return

        def error(values):
            suffixes = [f"{nmu.channel_char(c)}{e}" for c in range(values["n_channels"])
                        for e in range(values["n_epochs"])]
            return prefix_error(owner, values["name"], suffixes)

        prefix = next(p for p in ("Record", *(f"Record{n}" for n in range(1, 1000)))
                      if error({"name": p, "n_channels": 1, "n_epochs": 1}) is None)
        defaults = new_data_defaults(owner.data) | {
            "name": prefix, "n_channels": 1, "n_epochs": 10, "noise": 0,
        }
        values = self._ask_new_series(defaults, error)
        if values is None:
            return
        value, noise = values["value"], values["noise"]
        if noise > 0:
            rng = np.random.default_rng()
            fill = lambda n: value + rng.normal(0.0, noise, n)  # noqa: E731
        else:
            fill = value
        series = owner.new_dataseries(
            values["name"],
            n_channels=values["n_channels"], n_epochs=values["n_epochs"],
            n_points=values["points"], dx=values["xdelta"], x_start=values["xstart"],
            x_label=values["xlabel"], x_units=values["xunits"],
            y_label=values["ylabel"], y_units=values["yunits"], fill=fill,
        )
        self._selection.update(dataseries=series)
        self._changed()

    def series_from_data(self) -> None:
        owner = self._owner()
        prefixes = self._candidate_prefixes()
        if owner is None or not prefixes:
            return
        answer = self._choose(
            "Series from Data", context=("Folder", self._context_text()),
            item_label="Prefix", items=prefixes, current=prefixes[0],
            message="Arrays named <prefix><channel><epoch> become the series.",
        )
        if answer is None:
            return
        series = owner.sync_dataseries(answer[0])
        self._selection.update(dataseries=series)
        self._changed()

    def update_summary(self, name: str, *_extras) -> str:
        """The series' channels and epochs, and what an update would add."""
        owner = self._owner()
        series = owner.dataseries.get(name)
        channels = list(series.channels.keys())
        epochs = [int(e[1:]) for e in series.epochs.keys() if e[1:].isdigit()]
        lines = [
            f"Channels: {', '.join(channels) or 'none'} ({len(channels)})",
            f"Epochs: {epoch_ranges(epochs) or 'none'} ({len(series.epochs)})",
            "",
        ]
        preview = series_update_preview(owner, series)
        arrays = preview["new_arrays"]
        if not arrays:
            lines.append("Up to date: no new arrays.")
        else:
            adds = []
            if preview["new_channels"]:
                adds.append(f"channel{'s' if len(preview['new_channels']) > 1 else ''} "
                            + ", ".join(preview["new_channels"]))
            if preview["new_epochs"]:
                adds.append("epochs " + epoch_ranges(preview["new_epochs"]))
            noun = "array" if len(arrays) == 1 else "arrays"
            lines.append(f"Update adds {len(arrays)} {noun}"
                         + (": " + "; ".join(adds) if adds else "") + ".")
            shown = ", ".join(arrays[:8]) + (", …" if len(arrays) > 8 else "")
            lines.append(shown)
            if series.locked:
                lines.append("")
                lines.append("The series is locked, so it will not be updated.")
        return "\n".join(lines)

    def update_series(self) -> None:
        owner = self._owner()

        def error(item):
            series = owner.dataseries.get(item)
            if not series_update_preview(owner, series)["new_arrays"]:
                return f"Data series '{item}' is already up to date"
            return self._lock_error(owner, item)

        answer = self._choose_series(
            "Update from Data",
            message=self.update_summary,
            validate=lambda item, name: error(item),
        )
        if answer is None:
            return
        owner.sync_dataseries(answer[0])
        self._changed()

    def copy_series(self) -> None:
        owner = self._owner()
        folders = self._selection.manager.folders
        here = self._folder()
        targets = ([here.name] if here is not None else []) + [
            name for name in folders.keys() if here is None or name != here.name
        ]

        def target_owner(target):
            return owner if target == here.name and owner is not None else folders[target]

        def default_prefix(item, target):
            target_folder = target_owner(target)
            suffixes = series_suffixes(owner.dataseries[item])
            candidate, number = f"{COPY_PREFIX}_{item}", 1
            while prefix_error(target_folder, candidate, suffixes) is not None:
                candidate, number = f"{COPY_PREFIX}{number}_{item}", number + 1
            return candidate

        answer = self._choose_series(
            "Copy Series",
            extras=[("To folder", targets, targets[0] if targets else None)],
            name_label="New prefix",
            default_name=default_prefix,
            validate=lambda item, name, target: prefix_error(
                target_owner(target), name, series_suffixes(owner.dataseries[item])
            ),
        )
        if answer is None:
            return
        item, name, target = answer
        destination = target_owner(target)
        copy = owner.copy_dataseries(
            item, name, folder=None if destination is owner else destination
        )
        if destination is owner:
            self._selection.update(dataseries=copy)
        self._changed()

    def rename_series(self) -> None:
        owner = self._owner()
        answer = self._choose_series(
            "Rename Series",
            name_label="New prefix",
            default_name=lambda item: item,
            validate=lambda item, name: self._lock_error(owner, item) or (
                None if name == item else prefix_error(
                    owner, name, series_suffixes(owner.dataseries[item]),
                    renaming=owner.dataseries[item],
                )
            ),
        )
        if answer is None or answer[0] == answer[1]:
            return
        owner.rename_dataseries(answer[0], answer[1])
        self._changed()

    def edit_scales(self) -> None:
        series, owner = self._series(), self._owner()
        if series is None:
            return
        arrays = [d for ch in series.channels.values() for d in ch.data]
        y_rows = [
            (name, common_value(d.yscale.label for d in channel.data),
             common_value(d.yscale.units for d in channel.data))
            for name, channel in series.channels.items()
        ]
        lock_error = self._lock_error(owner, series.name)
        changes = self._ask_scales(
            scale_fields(arrays), y_rows, self._series_context(series), lock_error,
        )
        if not changes or lock_error:  # the dialog only shows a locked series
            return
        for key, setter in (("xstart", series.set_xstart), ("xdelta", series.set_xdelta),
                            ("xlabel", series.set_xlabel), ("xunits", series.set_xunits)):
            if key in changes:
                setter(changes[key])
        for channel, fields in changes.get("y", {}).items():
            if "label" in fields:
                series.set_ylabel(fields["label"], channel=channel)
            if "units" in fields:
                series.set_yunits(fields["units"], channel=channel)
        self._changed()

    def toggle_lock(self) -> None:
        series = self._series()
        if series is not None:
            series.locked = not series.locked
        self._changed()

    def _remove_part(self, title: str, label: str, names: list[str], remove) -> None:
        series, owner = self._series(), self._owner()
        if series is None or not names:
            return
        answer = self._choose(
            title, context=self._series_context(series), item_label=label,
            items=names, extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message=f"Keep the {label.lower()}'s arrays in the folder, or delete them.",
            validate=lambda item, name, arrays: self._lock_error(owner, series.name),
        )
        if answer is None:
            return
        item, _name, arrays = answer
        remove(series.name, item, delete_data=arrays == self.DELETE)
        self._changed()

    def remove_channel(self) -> None:
        series, owner = self._series(), self._owner()
        if series is not None:
            self._remove_part("Remove Channel", "Channel", list(series.channels.keys()),
                              owner.remove_dataseries_channel)

    def remove_epoch(self) -> None:
        """Remove a range of epochs, From..To inclusive (one if they match)."""
        series, owner = self._series(), self._owner()
        if series is None or len(series.epochs) == 0:
            return
        names = list(series.epochs.keys())
        epoch = self._selection.selection.get("epoch")
        current = epoch.name if epoch is not None and epoch.name in names else names[0]

        def chosen(first: str, last: str) -> list[str]:
            return epoch_range(names, first, last)

        def summary(_item, first, last, _arrays):
            return range_summary("Removes", chosen(first, last), 1)

        answer = self._choose(
            "Remove Epochs", context=self._series_context(series),
            extras=epoch_spins(names, current)
            + [("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message=summary,
            validate=lambda _item, _name, first, last, _arrays: (
                self._lock_error(owner, series.name) or range_error(names, first, last)
            ),
        )
        if answer is None:
            return
        _item, _name, first, last, arrays = answer
        owner.remove_dataseries_epoch(
            series.name, chosen(first, last), delete_data=arrays == self.DELETE
        )
        self._changed()

    def delete_series(self) -> None:
        owner = self._owner()
        answer = self._choose_series(
            "Delete Series",
            extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message="Keep the series' arrays in the folder, or delete them too.",
            validate=lambda item, name, arrays: self._lock_error(owner, item),
        )
        if answer is None:
            return
        item, _name, arrays = answer
        owner.remove_dataseries(item, delete_data=arrays == self.DELETE)
        self._changed()

    def choose_in_browser(self) -> None:
        series = self._series()
        if series is not None:
            self.browse_requested.emit(series)


def epoch_number(name: str) -> int | None:
    """The number of epoch *name*: "E12" or "12" -> 12; otherwise None."""
    text = name[1:] if name[:1] in ("E", "e") else name
    return int(text) if text.isdigit() else None


def epoch_range(names: list[str], first: str, last: str, every: int = 1) -> list[str]:
    """Epochs numbered *first* to *last* (inclusive), every *every*-th.

    Works by number, so gaps (e.g. removed epochs) are skipped.
    """
    low, high = epoch_number(first), epoch_number(last)
    if low is None or high is None:
        return []
    within = [n for n in names if (k := epoch_number(n)) is not None and low <= k <= high]
    return within[::max(every, 1)]


def range_error(names: list[str], first: str, last: str, every: int = 1) -> str | None:
    """Why the range *first*..*last* selects no epochs, or None."""
    low, high = epoch_number(first), epoch_number(last)
    if low is not None and high is not None and low > high:
        return "To must not be before From"
    if not epoch_range(names, first, last, every):
        return f"No epochs numbered {low} to {high}"
    return None


def epoch_spins(names: list[str], first: str | None = None,
                last: str | None = None) -> list:
    """From/To Epoch number boxes over the epoch numbers of *names*."""
    numbers = [k for n in names if (k := epoch_number(n)) is not None] or [0]
    low, high = min(numbers), max(numbers)

    def value(name):
        number = epoch_number(name) if name else None
        return number if number is not None else low

    return [("From Epoch", Spin(low, high, value(first))),
            ("To Epoch", Spin(low, high, value(last if last else first)))]


def range_summary(verb: str, epochs: list[str], every: int) -> str:
    """e.g. "Adds 10 epochs (E2–E20, every 2nd)." """
    if not epochs:
        return ""
    span = epochs[0] if len(epochs) == 1 else f"{epochs[0]}–{epochs[-1]}"
    if every > 1 and len(epochs) > 1:
        suffix = {2: "nd", 3: "rd"}.get(every, "th")
        span += f", every {every}{suffix}"
    return f"{verb} {len(epochs)} epoch{'s' * (len(epochs) != 1)} ({span})."


def epoch_set_name_error(sets, name: str | None) -> str | None:
    """Like set_name_error, also refusing the built-in All and Current Epoch."""
    if name is not None and name.lower() in (b.lower() for b in BUILTIN_EPOCH_SETS):
        return f"'{name}' is a built-in Epoch Set choice"
    return set_name_error(sets, name)


class EpochSetActions(TierActions):
    """Epoch Set actions, for the current data series' epochs.

    Sets select epochs for analysis and do not change the recording, so
    they can be edited on a locked series too.
    """

    # Ask the app to show a series' epoch sets in the browser (set to select)
    browse_requested = QtCore.pyqtSignal(object, object)

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        self.new_action = action("New Set…", self.new_set, tip="Create an empty epoch set")
        self.add_action = action("Add Epochs…", self.add_epochs,
                                 tip="Add an epoch, or a range of epochs, to a set")
        self.remove_action = action("Remove Epochs…", self.remove_epochs,
                                    tip="Remove an epoch, or a range of epochs, from a set")
        self.exclude_action = action(
            "Exclude Epochs…", self.exclude_epochs,
            tip=f"Add epochs to {EXCLUDE_SET}, leaving them out of every Run",
        )
        self.include_action = action(
            "Include Epochs…", self.include_epochs,
            tip=f"Remove epochs from {EXCLUDE_SET}, so Run uses them again",
        )
        self.invert_action = action("Invert…", self.invert_set,
                                    tip="Make a set of the epochs not in a set")
        self.combine_action = action("Combine Sets…", self.combine_sets,
                                     tip="Make a set from two sets with AND or OR")
        self.clear_action = action("Clear Set…", self.clear_set,
                                   tip="Remove all epochs from a set, keeping the set")
        self.rename_action = action("Rename Set…", self.rename_set, tip="Rename an epoch set")
        self.delete_action = action("Delete Set…", self.delete_set, tip="Delete an epoch set")
        self.browse_action = action("Choose in Browser…", self.choose_in_browser,
                                    tip="Show the epoch sets in the browser")
        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def epoch_set_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Epoch Set", parent)
        for item in (self.new_action, self.add_action, self.remove_action, None,
                     self.exclude_action, self.include_action, None,
                     self.invert_action, self.combine_action, None, self.clear_action,
                     self.rename_action, self.delete_action, None, self.browse_action):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        return menu

    # --- helpers ------------------------------------------------------------

    def _series(self):
        series = self._selection.selection.get("dataseries")
        context = self._context()
        if series is None or context is None or series.name not in context.dataseries:
            return None
        return series

    def _set_names(self) -> list[str]:
        series = self._series()
        return list(series.epochs.sets.keys()) if series is not None else []

    def update_enabled(self, *_args) -> None:
        series = self._series()
        sets = self._set_names()
        has_epochs = series is not None and len(series.epochs) > 0
        self.new_action.setEnabled(series is not None)
        self.browse_action.setEnabled(series is not None)
        self.exclude_action.setEnabled(has_epochs)
        self.include_action.setEnabled(
            series is not None and bool(series.epochs.excluded_names())
        )
        for item in (self.clear_action, self.rename_action, self.delete_action):
            item.setEnabled(bool(sets))
        for item in (self.add_action, self.remove_action, self.invert_action):
            item.setEnabled(bool(sets) and has_epochs)
        self.combine_action.setEnabled(len(sets) > 1)

    def _series_context(self, series) -> tuple[str, str]:
        return ("Series", f"{self._context_text()} › {series.name}")

    @staticmethod
    def _exclude_set_error(name: str | None, done: str) -> str | None:
        if name is not None and name.lower() == EXCLUDE_SET.lower():
            return (f"{EXCLUDE_SET} is the exclusion set and cannot be {done}; "
                    "clear it instead")
        return None

    def _choose_set(self, title: str, **options):
        series = self._series()
        sets = self._set_names()
        if series is None or not sets:
            return None
        current = self._selection.selection.get("epoch_set")
        return self._choose(
            title, context=self._series_context(series), item_label="Epoch Set",
            items=sets, current=current if current in sets else sets[0], **options,
        )

    def _range_extras(self, series) -> list:
        names = list(series.epochs.keys())
        epoch = self._selection.selection.get("epoch")
        current = epoch.name if epoch is not None and epoch.name in names else names[0]
        return epoch_spins(names, current) + [("Every", Spin(1, max(len(names), 1), 1))]

    # --- actions ------------------------------------------------------------

    def new_set(self) -> None:
        series = self._series()
        if series is None:
            return
        sets = series.epochs.sets
        answer = self._choose(
            "New Epoch Set", context=self._series_context(series),
            name_label="Set name", default_name=lambda _item: sets.auto_name_next(),
            validate=lambda _item, name: epoch_set_name_error(sets, name),
        )
        if answer is None:
            return
        sets.add(answer[1])
        self._selection.update(epoch_set=answer[1])
        self._changed()

    def _edit_range(self, title: str, verb: str, adding: bool, fixed: str | None = None) -> None:
        """Add or remove a range of epochs to a chosen set, or to *fixed*."""
        series = self._series()
        if series is None or len(series.epochs) == 0:
            return
        names = list(series.epochs.keys())
        sets = series.epochs.sets
        if fixed is not None and sets._getkey(fixed) is None:
            sets.add(fixed, quiet=True)  # e.g. SetX of an older series

        def chosen(first, last, every):
            return epoch_range(names, first, last, int(every))

        def targets(set_name, first, last, every):
            set_name = fixed or set_name
            epochs = chosen(first, last, every)
            if adding:
                return epochs
            members = set(sets.get_items(set_name, get_keys=True) or [])
            return [e for e in epochs if e in members]

        def error(set_name, _name, first, last, every):
            set_name = fixed or set_name
            if range_error(names, first, last, int(every)):
                return range_error(names, first, last, int(every))
            if not adding and not targets(set_name, first, last, every):
                return f"None of these epochs are in '{set_name}'"
            return None

        options = dict(
            extras=self._range_extras(series),
            message=lambda set_name, first, last, every: range_summary(
                verb, targets(set_name, first, last, every), int(every)),
            validate=error,
        )
        if fixed is None:
            answer = self._choose_set(title, **options)
        else:
            answer = self._choose(title, context=self._series_context(series), **options)
        if answer is None:
            return
        set_name, _name, first, last, every = answer
        set_name = fixed or set_name
        epochs = [series.epochs[e] for e in targets(set_name, first, last, every)]
        if adding:
            sets.add(set_name, epochs)
        else:
            sets.remove(set_name, epochs)
        self._changed()

    def add_epochs(self) -> None:
        self._edit_range("Add Epochs to Set", "Adds", adding=True)

    def remove_epochs(self) -> None:
        self._edit_range("Remove Epochs from Set", "Removes", adding=False)

    def exclude_epochs(self) -> None:
        """Add a range of epochs to SetX, leaving them out of every run."""
        self._edit_range("Exclude Epochs", "Excludes", adding=True, fixed=EXCLUDE_SET)

    def include_epochs(self) -> None:
        """Remove a range of epochs from SetX, so runs use them again."""
        self._edit_range("Include Epochs", "Includes", adding=False, fixed=EXCLUDE_SET)

    def invert_set(self) -> None:
        series = self._series()
        if series is None:
            return
        sets = series.epochs.sets
        names = list(series.epochs.keys())

        def complement(set_name):
            members = set(sets.get_items(set_name, get_keys=True) or [])
            return [e for e in names if e not in members]

        answer = self._choose_set(
            "Invert Epoch Set", name_label="New set",
            default_name=lambda _item: sets.auto_name_next(),
            message=lambda set_name, *_: (
                f"The new set has the {len(complement(set_name))} epochs not in "
                f"'{set_name}'."
            ),
            validate=lambda _item, name: epoch_set_name_error(sets, name),
        )
        if answer is None:
            return
        set_name, name = answer[0], answer[1]
        sets.add(name, [series.epochs[e] for e in complement(set_name)])
        self._selection.update(epoch_set=name)
        self._changed()

    def combine_sets(self) -> None:
        series = self._series()
        names = self._set_names()
        if series is None or len(names) < 2:
            return
        sets = series.epochs.sets
        current = self._selection.selection.get("epoch_set")
        first = current if current in names else names[0]
        second = next((n for n in names if n != first), first)
        answer = self._choose_set(
            "Combine Sets",
            extras=[("Operator", ["AND", "OR"], "OR"), ("With Set", names, second)],
            name_label="New Set",
            default_name=lambda _item, *_extras: sets.auto_name_next(),
            validate=lambda item, name, _operator, other: (
                "Choose two different sets" if item == other
                else epoch_set_name_error(sets, name)
            ),
        )
        if answer is None:
            return
        first, name, operator, second = answer
        (sets.define_and if operator == "AND" else sets.define_or)(name, first, second)
        self._selection.update(epoch_set=name)
        self._changed()

    def clear_set(self) -> None:
        answer = self._choose_set(
            "Clear Epoch Set",
            message="All epochs will be removed from the set; the set is kept.",
        )
        if answer is None:
            return
        self._series().epochs.sets.empty(answer[0])
        self._changed()

    def rename_set(self) -> None:
        series = self._series()
        sets = series.epochs.sets if series is not None else None
        answer = self._choose_set(
            "Rename Epoch Set", name_label="New name", default_name=lambda item: item,
            validate=lambda item, name: (
                self._exclude_set_error(item, "renamed")
                or (None if name == item else epoch_set_name_error(sets, name))
            ),
        )
        if answer is None or answer[0] == answer[1]:
            return
        was_current = answer[0] == self._selection.selection.get("epoch_set")
        sets.rename(answer[0], answer[1])
        if was_current:
            self._selection.update(epoch_set=answer[1])
        self._changed()

    def delete_set(self) -> None:
        answer = self._choose_set(
            "Delete Epoch Set", message="The set will be deleted; its epochs are kept.",
            validate=lambda item, _name: self._exclude_set_error(item, "deleted"),
        )
        if answer is None:
            return
        if answer[0] == self._selection.selection.get("epoch_set"):
            self._selection.update(epoch_set=None)
        self._series().epochs.sets.pop(answer[0])
        self._changed()

    def choose_in_browser(self) -> None:
        series = self._series()
        if series is not None:
            self.browse_requested.emit(series, self._selection.selection.get("epoch_set"))


def cyclic_groups(n_epochs: int, n_groups: int, block: int = 1) -> list[int]:
    """Group numbers of a cyclic assignment: (i // block) % n_groups."""
    return [(i // max(block, 1)) % max(n_groups, 1) for i in range(n_epochs)]


class GroupActions(TierActions):
    """Group actions, assigning the current data series' epochs to groups.

    Groups select epochs for analysis and do not change the recording, so
    they can be edited on a locked series too.
    """

    # Ask the app to show a series' epochs in the browser
    browse_requested = QtCore.pyqtSignal(object)

    PREVIEW = 16  # group numbers shown in the Assign Cyclic preview

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        self.new_action = action(
            "New Groups…", self.new_groups,
            tip="Create empty groups 0, 1, 2…, to fill later (e.g. in the browser's Edit mode)",
        )
        self.cyclic_action = action(
            "Assign Cyclic…", self.assign_cyclic,
            tip="Assign epochs to groups in turn, e.g. 0 1 0 1 or 000 111 (block 3)",
        )
        self.assign_action = action("Assign Epochs…", self.assign_epochs,
                                    tip="Assign an epoch, or a range of epochs, to a group")
        self.unassign_action = action("Unassign Epochs…", self.unassign_epochs,
                                      tip="Take an epoch, or a range of epochs, out of groups")
        self.clear_action = action("Clear Groups…", self.clear_groups,
                                   tip="Remove every group assignment")
        self.browse_action = action("Choose in Browser…", self.choose_in_browser,
                                    tip="Show the series' epochs in the browser")
        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def group_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Group", parent)
        for item in (self.new_action, None, self.cyclic_action, self.assign_action,
                     self.unassign_action, None, self.clear_action, None, self.browse_action):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        return menu

    # --- helpers ------------------------------------------------------------

    def _series(self):
        series = self._selection.selection.get("dataseries")
        context = self._context()
        if series is None or context is None or series.name not in context.dataseries:
            return None
        return series

    def _assigned(self, series) -> dict[str, int]:
        groups = series.epochs.groups
        return {name: groups.get_group(name) for name in series.epochs.keys()
                if groups.get_group(name) is not None}

    def update_enabled(self, *_args) -> None:
        series = self._series()
        has_epochs = series is not None and len(series.epochs) > 0
        has_groups = series is not None and len(series.epochs.groups.group_numbers) > 0
        self.new_action.setEnabled(series is not None)
        self.cyclic_action.setEnabled(has_epochs)
        self.assign_action.setEnabled(has_epochs)
        self.unassign_action.setEnabled(series is not None and bool(self._assigned(series)))
        self.clear_action.setEnabled(has_groups)
        self.browse_action.setEnabled(series is not None)

    def _series_context(self, series) -> tuple[str, str]:
        return ("Series", f"{self._context_text()} › {series.name}")

    def _changed(self) -> None:
        series = self._series()
        group = self._selection.selection.get("group")
        if series is not None and group is not None and group not in series.epochs.groups.group_numbers:
            self._selection.update(group=None)  # the chosen group no longer exists
        super()._changed()

    # --- actions ------------------------------------------------------------

    def new_groups(self) -> None:
        """Create empty groups 0 … N-1 (those that do not exist yet)."""
        series = self._series()
        if series is None:
            return
        groups = series.epochs.groups
        existing = set(groups.group_numbers)

        def missing(count):
            return [n for n in range(int(count)) if n not in existing]

        def summary(_item, count):
            new = missing(count)
            if not new:
                return ""
            span = f"Group {new[0]}" if len(new) == 1 else f"Groups {new[0]}–{new[-1]}"
            return f"Creates {span}, with no epochs yet."

        answer = self._choose(
            "New Groups", context=self._series_context(series),
            extras=[("Groups", Spin(1, 1000, max(len(existing) + 1, 2)))],
            message=summary,
            validate=lambda _item, _name, count: (
                None if missing(count) else f"Groups 0–{int(count) - 1} already exist"
            ),
        )
        if answer is None:
            return
        for number in missing(answer[2]):
            groups.add_group(number, quiet=True)
        self._changed()

    def assign_cyclic(self) -> None:
        series = self._series()
        if series is None or len(series.epochs) == 0:
            return
        names = list(series.epochs.keys())

        def chosen(first, last):
            return epoch_range(names, first, last)

        def preview(_item, groups, block, first, last):
            epochs = chosen(first, last)
            if not epochs:
                return ""
            numbers = cyclic_groups(len(epochs), int(groups), int(block))
            shown = " ".join(str(n) for n in numbers[:self.PREVIEW])
            more = " …" if len(numbers) > self.PREVIEW else ""
            noun = "epoch" if len(epochs) == 1 else "epochs"
            return (f"{shown}{more} ({groups} group{'s' * (groups != '1')}, "
                    f"block {block}, {len(epochs)} {noun})")

        answer = self._choose(
            "Assign Cyclic", context=self._series_context(series),
            extras=[("Groups", Spin(1, 1000, 2)), ("Block", Spin(1, 100_000, 1))]
            + epoch_spins(names, names[0], names[-1]),
            message=preview,
            validate=lambda _item, _name, _groups, _block, first, last: (
                range_error(names, first, last)
            ),
        )
        if answer is None:
            return
        _item, _name, groups, block, first, last = answer
        series.epochs.groups.assign_cyclic(chosen(first, last), int(groups), block=int(block))
        self._changed()

    def _range_extras(self, series, group_choices=None) -> list:
        names = list(series.epochs.keys())
        epoch = self._selection.selection.get("epoch")
        current = epoch.name if epoch is not None and epoch.name in names else names[0]
        extras = epoch_spins(names, current) + [("Every", Spin(1, max(len(names), 1), 1))]
        return (group_choices or []) + extras

    def assign_epochs(self) -> None:
        series = self._series()
        if series is None or len(series.epochs) == 0:
            return
        names = list(series.epochs.keys())
        numbers = series.epochs.groups.group_numbers
        current = self._selection.selection.get("group")
        default = current if current is not None else (numbers[0] if numbers else 0)

        def chosen(first, last, every):
            return epoch_range(names, first, last, int(every))

        answer = self._choose(
            "Assign Epochs to Group", context=self._series_context(series),
            extras=self._range_extras(series, [("Group", Spin(0, 9999, default))]),
            message=lambda _item, group, first, last, every: (
                range_summary("Assigns", chosen(first, last, every), int(every))[:-1]
                + f" to group {group}." if chosen(first, last, every) else ""
            ),
            validate=lambda _item, _name, _group, first, last, every: (
                range_error(names, first, last, int(every))
            ),
        )
        if answer is None:
            return
        _item, _name, group, first, last, every = answer
        groups = series.epochs.groups
        for name in chosen(first, last, every):
            groups.assign(name, int(group), quiet=True)
        self._changed()

    def unassign_epochs(self) -> None:
        series = self._series()
        if series is None or not self._assigned(series):
            return
        names = list(series.epochs.keys())

        def targets(first, last, every):
            assigned = self._assigned(series)
            return [e for e in epoch_range(names, first, last, int(every)) if e in assigned]

        def error(_item, _name, first, last, every):
            if range_error(names, first, last, int(every)):
                return range_error(names, first, last, int(every))
            if not targets(first, last, every):
                return "None of these epochs are in a group"
            return None

        answer = self._choose(
            "Unassign Epochs", context=self._series_context(series),
            extras=self._range_extras(series),
            message=lambda _item, first, last, every: range_summary(
                "Unassigns", targets(first, last, every), int(every)),
            validate=error,
        )
        if answer is None:
            return
        _item, _name, first, last, every = answer
        groups = series.epochs.groups
        for name in targets(first, last, every):
            groups.unassign(name, quiet=True)
        self._changed()

    def clear_groups(self) -> None:
        series = self._series()
        if series is None:
            return
        n_groups = len(series.epochs.groups.group_numbers)
        answer = self._choose(
            "Clear Groups", context=self._series_context(series),
            message=f"Every group assignment ({n_groups} group"
                    f"{'s' * (n_groups != 1)}) will be removed; the epochs are kept.",
        )
        if answer is None:
            return
        series.epochs.groups.clear()
        self._changed()

    def choose_in_browser(self) -> None:
        series = self._series()
        if series is not None:
            self.browse_requested.emit(series)


__all__ = [
    "ChooseItemDialog",
    "DataActions",
    "EditScalesDialog",
    "EpochSetActions",
    "GroupActions",
    "NewDataDialog",
    "SeriesActions",
    "FolderActions",
    "TierActions",
    "COPY_PREFIX",
    "Spin",
    "ask_new_data",
    "create_data",
    "new_data_defaults",
    "OPEN_FILTER",
    "READERS",
    "copy_name",
    "copy_target_name",
    "name_error",
    "prefix_error",
    "series_name_error",
    "series_suffixes",
    "series_name_hint",
    "series_update_preview",
    "cyclic_groups",
    "epoch_range",
    "epoch_set_name_error",
    "epoch_ranges",
    "range_summary",
    "set_name_error",
    "unique_name",
]
