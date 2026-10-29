"""Actions for selection tiers, shared by the strip's ⋯ menus and the menu bar.

Each tier's actions are QActions created once; a strip ⋯ button and a menu
bar menu both show the same objects, so enabled state and shortcuts agree.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

import numpy as np

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_epoch import EXCLUDE_SET
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.gui.selection_model import BUILTIN_EPOCH_SETS, SelectionModel
from pyneuromatic.gui.membership import unassign_from_group
from pyneuromatic.gui.transform_dialog import MARK_COLOR, MENU_TEXT


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


def warning_html(text: str) -> str:
    """*text* in the app's red, for a dialog's message (rich text)."""
    from html import escape
    return f'<span style="color: {MARK_COLOR};">{escape(text)}</span>'


def copy_target_name(container: NMObjectContainer, name: str) -> str:
    """Name for a copy of *name* in another *container*: the same name if it
    is free there, otherwise a prefixed copy name (``C_name``, ``C1_name``...)."""
    return name if container._getkey(name) is None else copy_name(container, name)


# The strip's dialogs share one minimum width, so they look alike and titles
# aren't cut short (wider when a title needs it)
DIALOG_WIDTH = 380


def set_dialog_width(dialog: QtWidgets.QDialog, minimum: int = DIALOG_WIDTH) -> None:
    """Give *dialog* the shared minimum width, or more for a long title (the
    window's title bar uses a slightly larger font than the dialog)."""
    title = dialog.fontMetrics().horizontalAdvance(dialog.windowTitle())
    dialog.setMinimumWidth(max(minimum, int(title * 1.3) + 140))


@dataclasses.dataclass(frozen=True)
class Spin:
    """An integer field with up/down arrows, for :class:`ChooseItemDialog`
    extras: ``(label, Spin(minimum, maximum, value, prefix))``. Its value is
    returned as text with the prefix, e.g. ``"E250"``. *suffix* is shown
    after the number only (e.g. ``"  (max 14)"``), not returned."""

    minimum: int
    maximum: int
    value: int
    prefix: str = ""
    suffix: str = ""


@dataclasses.dataclass(frozen=True)
class Check:
    """A checkbox, for :class:`ChooseItemDialog` extras: ``(label,
    Check(text))``. Its value is returned as True or False. With
    *enabled(item, *extras)*, it can be ticked only while that is true
    (otherwise it is cleared)."""

    text: str
    checked: bool = False
    enabled: Callable[..., bool] | None = None


class ChooseItemDialog(QtWidgets.QDialog):
    """Choose an item (e.g. a folder) and/or enter a name, in one dialog.

    *items* adds an item dropdown, starting at *current*. *name_label* adds a
    name field; *default_name(item, *extras)* fills it whenever the item or
    an extra choice changes.
    *extras* adds further dropdowns, as ``(label, items, current)``,
    number boxes, as ``(label, Spin(...))``, or checkboxes, as
    ``(label, Check(...))``.
    *message* is a note below the fields, or a function of the item and
    extra choices, updated when they change.
    *context* adds a read-only first row, as ``(label, text)`` (e.g. the
    folder an action applies to).
    *validate(item, name, *extras)* returns an error message, shown inline
    while the dialog stays open, or None to accept. With *validate_live*,
    it is also shown as the fields change (e.g. a name taken in the folder
    chosen), not only when OK is pressed.
    *extra_items* maps an extra dropdown's position to *f(item)*, giving its
    choices for the chosen item (e.g. the data series of the folder chosen);
    they are refilled when the item changes.
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
        validate_live: bool = False,
        extra_items: dict[int, Callable[[str | None], list[str]]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._default_name = default_name
        self._validate = validate
        self._validate_live = validate_live
        self._extra_items = extra_items or {}
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
            form.addRow("Item" if item_label is None else item_label, self.item_combo)
        # Dropdowns (QComboBox), number boxes (QSpinBox) or checkboxes, in order
        self.extra_widgets: list[QtWidgets.QWidget] = []
        self._checks: list[tuple[QtWidgets.QCheckBox, Check]] = []
        for extra in extras or []:
            label = extra[0]
            if isinstance(extra[1], Spin):
                spec = extra[1]
                widget = QtWidgets.QSpinBox(self)
                widget.setRange(spec.minimum, spec.maximum)
                widget.setValue(spec.value)
                widget.setPrefix(spec.prefix)
                widget.setSuffix(spec.suffix)
                # The message follows typing too, not only the arrows
                widget.setKeyboardTracking(True)
            elif isinstance(extra[1], Check):
                widget = QtWidgets.QCheckBox(extra[1].text, self)
                widget.setChecked(extra[1].checked)
                self._checks.append((widget, extra[1]))
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
            self.item_combo.currentTextChanged.connect(self._refill_extras)
            self.item_combo.currentTextChanged.connect(self._fill_default_name)
        self._refill_extras()
        for widget in self.extra_widgets:
            if isinstance(widget, QtWidgets.QSpinBox):
                widget.valueChanged.connect(self._fill_default_name)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._fill_default_name)
            else:
                widget.currentTextChanged.connect(self._fill_default_name)
        if self.name_edit is not None and validate_live:
            self.name_edit.textChanged.connect(self._show_live_error)
        self._fill_default_name()
        if self.name_edit is not None:
            self.name_edit.selectAll()
            self.name_edit.setFocus()
        set_dialog_width(self)

    def item(self) -> str | None:
        return self.item_combo.currentText() if self.item_combo is not None else None

    def name(self) -> str | None:
        return self.name_edit.text().strip() if self.name_edit is not None else None

    @property
    def extra_combos(self) -> list[QtWidgets.QComboBox]:
        """The extra dropdowns (not number boxes)."""
        return [w for w in self.extra_widgets if isinstance(w, QtWidgets.QComboBox)]

    def extras(self) -> list:
        return [
            f"{w.prefix()}{w.value()}" if isinstance(w, QtWidgets.QSpinBox)
            else w.isChecked() if isinstance(w, QtWidgets.QCheckBox)
            else w.currentText()
            for w in self.extra_widgets
        ]

    def _enable_checks(self) -> None:
        """Checkboxes that can be ticked only sometimes (e.g. Overwrite, while
        the set chosen exists)."""
        for widget, spec in self._checks:
            if spec.enabled is None:
                continue
            on = bool(spec.enabled(self.item(), *self.extras()))
            widget.blockSignals(True)
            if not on:
                widget.setChecked(False)
            widget.setEnabled(on)
            widget.blockSignals(False)

    def _refill_extras(self, *_args) -> None:
        """Choices of dependent extra dropdowns, for the item chosen (keeping
        the current choice where it is still offered)."""
        for index, choices_for in self._extra_items.items():
            combo = self.extra_widgets[index]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            choices = choices_for(self.item())
            combo.addItems(choices)
            if current in choices:
                combo.setCurrentText(current)
            combo.setEnabled(bool(choices))
            combo.blockSignals(False)

    def _fill_default_name(self, *_args) -> None:
        self._enable_checks()
        if self.name_edit is not None and self._default_name is not None:
            self.name_edit.setText(self._default_name(self.item(), *self.extras()))
        message = (
            self._message(self.item(), *self.extras())
            if callable(self._message) else self._message
        )
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        self.error_label.hide()
        self._show_live_error()

    def _show_live_error(self, *_args) -> None:
        if not self._validate_live or self._validate is None:
            return
        error = self._validate(self.item(), self.name(), *self.extras())
        self.error_label.setText(error or "")
        self.error_label.setVisible(bool(error))

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
        set_dialog_width(self)

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


class AddToSeriesDialog(QtWidgets.QDialog):
    """How many channels or epochs to add to a data series, and the new
    arrays' values (a number, e.g. 0 or NaN, plus optional noise) and, for
    channels, y label and units.

    *preview(count)* describes the new arrays; *validate(values)* returns an
    error, shown inline while the dialog stays open.
    """

    def __init__(
        self,
        kind: str,  # "channels" or "epochs"
        context: tuple[str, str],
        preview: Callable[[int], str],
        validate: Callable[[dict], str | None] | None = None,
        maximum: int = 1_000_000,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._kind, self._preview, self._validate = kind, preview, validate
        self.setWindowTitle(f"Add {kind.capitalize()}")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        context_label = QtWidgets.QLabel(context[1], self)
        context_label.setStyleSheet("font-weight: bold;")
        form.addRow(context[0], context_label)
        self.count_spin = QtWidgets.QSpinBox(self)
        self.count_spin.setRange(1, max(1, maximum))
        self.count_spin.valueChanged.connect(self._update_preview)
        form.addRow(kind.capitalize(), self.count_spin)
        self.value_edit = QtWidgets.QLineEdit("0", self)
        self.value_edit.setToolTip("Initial value of every point, e.g. 0 or NaN")
        self.noise_edit = QtWidgets.QLineEdit("0", self)
        self.noise_edit.setToolTip(
            "Standard deviation of Gaussian noise added to every point (0: none)")
        fill = QtWidgets.QHBoxLayout()
        fill.addWidget(self.value_edit)
        fill.addWidget(QtWidgets.QLabel("Noise SD", self))
        fill.addWidget(self.noise_edit)
        form.addRow("Value", fill)
        self.ylabel_edit = self.yunits_edit = None
        if kind == "channels":  # a new channel records something else
            self.ylabel_edit = QtWidgets.QLineEdit("", self)
            self.yunits_edit = QtWidgets.QLineEdit("", self)
            y_row = QtWidgets.QHBoxLayout()
            y_row.addWidget(QtWidgets.QLabel("Label", self))
            y_row.addWidget(self.ylabel_edit)
            y_row.addWidget(QtWidgets.QLabel("Units", self))
            y_row.addWidget(self.yunits_edit)
            form.addRow("Y", y_row)
        self.preview_label = QtWidgets.QLabel("", self)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setStyleSheet("color: #b91c1c;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_preview()
        set_dialog_width(self)

    def _update_preview(self, *_args) -> None:
        self.preview_label.setText(self._preview(self.count_spin.value()))
        self.error_label.hide()

    def values(self) -> dict:
        """The fields, with numbers converted (raises ValueError if invalid)."""
        return {
            "count": self.count_spin.value(),
            "value": float(self.value_edit.text().strip() or 0),  # also "NaN"
            "noise": float(self.noise_edit.text().strip() or 0),
            "ylabel": self.ylabel_edit.text().strip() if self.ylabel_edit else "",
            "yunits": self.yunits_edit.text().strip() if self.yunits_edit else "",
        }

    def accept(self) -> None:
        try:
            values = self.values()
        except ValueError:
            error = "Value and Noise SD must be numbers (Value may be NaN)"
        else:
            error = ("Noise SD must not be negative" if values["noise"] < 0
                     else self._validate(values) if self._validate else None)
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


def fill_values(value: float, noise: float):
    """The fill for new arrays: *value*, plus Gaussian noise of SD *noise*."""
    if noise > 0:
        rng = np.random.default_rng()
        return lambda n: value + rng.normal(0.0, noise, n)
    return value


def array_names_text(names: list[str], limit: int = 2) -> str:
    """e.g. "RecordC0, RecordC1, ... RecordD9" for a preview."""
    if len(names) <= limit + 1:
        return ", ".join(names)
    return f"{', '.join(names[:limit])}, ... {names[-1]}"


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

    With *x_arrays* (names of arrays that fit), "X Array" chooses an array
    whose values are the x values, or None (start and delta); *x_array* is
    the current choice ("" for none, None for mixed) and *validate_x(name)* returns why an
    array can't be used, or None. *x_array_scales* gives each array's
    (label, units): choosing an array fills X Label and Units from them
    (where set), and a note says when X Units differ from the array's.
    *x_array_fixed* (e.g. "Set by data series 'Record'") shows the choice
    without allowing a change, with that note.
    """

    START_DELTA = "None"  # no x array: x from start and delta
    MIXED = "(mixed)"

    def __init__(
        self,
        x: dict[str, str | None],
        y_rows: list[tuple[str, str | None, str | None]],
        context: tuple[str, str] | None = None,
        error: str | None = None,
        parent: QtWidgets.QWidget | None = None,
        x_arrays: list[str] | None = None,
        x_array: str | None = "",
        validate_x: Callable[[str], str | None] | None = None,
        x_array_scales: dict[str, tuple[str, str]] | None = None,
        x_array_fixed: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Scales")
        self._error = error
        self._validate_x = validate_x
        self._x_array_scales = x_array_scales or {}
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
        # X Array: None (start and delta), or an array whose values are x
        self.x_from_combo: QtWidgets.QComboBox | None = None
        x_row = 1
        if x_arrays is not None:
            combo = QtWidgets.QComboBox(self)
            combo.setObjectName("xfrom")
            if x_array is None:
                combo.addItem(self.MIXED)
            combo.addItem(self.START_DELTA)
            names = list(x_arrays)
            if x_array and x_array not in names:
                names.insert(0, x_array)  # e.g. an x array in another folder
            combo.addItems(names)
            combo.setCurrentText(self.MIXED if x_array is None else (x_array or self.START_DELTA))
            combo.setToolTip("An array whose values are the x values (same number of "
                             "points, increasing): changing it changes the x values. "
                             "None: x from Start and Delta")
            combo.currentTextChanged.connect(self._update_x_from)
            combo.currentTextChanged.connect(self._fill_x_from_array)
            self._initial["xfrom"] = combo.currentText()
            self.x_from_combo = combo
            grid.addWidget(QtWidgets.QLabel("X Array", self), 1, 0)
            grid.addWidget(combo, 1, 1, 1, 3)
            x_row = 2
        # e.g. an array of a data series: the series sets its x array
        self.x_array_fixed_note: QtWidgets.QLabel | None = None
        if x_array_fixed and self.x_from_combo is not None:
            self.x_from_combo.setEnabled(False)
            self.x_from_combo.setToolTip(x_array_fixed)
            note = QtWidgets.QLabel(x_array_fixed, self)
            note.setStyleSheet("color: gray;")
            note.setWordWrap(True)
            grid.addWidget(note, 2, 1, 1, 3)
            self.x_array_fixed_note = note
            x_row = 3
        for row, (first, second) in enumerate((("xstart", "xdelta"), ("xlabel", "xunits")), x_row):
            for col, key in ((0, first), (2, second)):
                grid.addWidget(QtWidgets.QLabel(key[1:].capitalize(), self), row, col)
                grid.addWidget(self.x_edits[key], row, col + 1)
        # e.g. "Times is in s": the x array's units differ from X Units
        self.x_units_note = QtWidgets.QLabel("", self)
        self.x_units_note.setStyleSheet("color: #b45309;")
        self.x_units_note.setWordWrap(True)
        self.x_units_note.hide()
        grid.addWidget(self.x_units_note, x_row + 2, 0, 1, 4)
        self.x_edits["xunits"].textChanged.connect(self._update_units_note)
        y_header = x_row + 3
        grid.addWidget(QtWidgets.QLabel("<b>Y</b>", self), y_header, 0)
        self.y_edits: dict[str, dict[str, QtWidgets.QLineEdit]] = {}
        multi = len(y_rows) > 1
        if multi:
            grid.addWidget(QtWidgets.QLabel("Channel", self), y_header + 1, 0)
            grid.addWidget(QtWidgets.QLabel("Label", self), y_header + 1, 1)
        for offset, (name, label, units) in enumerate(y_rows, y_header + (2 if multi else 1)):
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
        self._update_x_from()
        set_dialog_width(self)

    def x_from(self) -> str | None:
        """The array chosen for x values, "" for start and delta, or None
        (mixed, unchanged)."""
        if self.x_from_combo is None:
            return ""
        text = self.x_from_combo.currentText()
        return None if text == self.MIXED else ("" if text == self.START_DELTA else text)

    def _update_x_from(self, *_args) -> None:
        # Start and delta aren't used while x comes from an array
        from_array = bool(self.x_from())
        for key in ("xstart", "xdelta"):
            self.x_edits[key].setEnabled(not from_array)
        self._update_units_note()

    def _fill_x_from_array(self, *_args) -> None:
        """X Label and Units from the chosen array's (where it has them);
        they stay the series' own, and can still be edited."""
        label, units = self._x_array_scales.get(self.x_from() or "", ("", ""))
        if label:
            self.x_edits["xlabel"].setText(label)
        if units:
            self.x_edits["xunits"].setText(units)

    def _update_units_note(self, *_args) -> None:
        name = self.x_from()
        units = self._x_array_scales.get(name or "", ("", ""))[1]
        shown = self.x_edits["xunits"].text().strip()
        differ = bool(name) and bool(units) and units != shown
        self.x_units_note.setText(
            f"{name} is in {units}; X Units say {shown or '(none)'}." if differ else "")
        self.x_units_note.setVisible(differ)

    def changes(self) -> dict:
        """Changed fields: ``xstart``... (numbers converted), ``xarray`` (an
        array name, or "" for start and delta) and ``y`` as
        ``{row: {"label": ..., "units": ...}}``. Raises ValueError if a
        changed Start or Delta is not a number."""
        changed: dict = {}
        if self.x_from_combo is not None \
                and self.x_from_combo.currentText() != self._initial["xfrom"] \
                and self.x_from() is not None:
            changed["xarray"] = self.x_from()
        for key, line in self.x_edits.items():
            if key in ("xstart", "xdelta") and not line.isEnabled():
                continue  # x comes from an array
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
            if error is None and changed.get("xarray") and self._validate_x is not None:
                error = self._validate_x(changed["xarray"])
        if error:
            self.error_label.setText(error)
            self.error_label.show()
            return
        super().accept()


def x_array_choices(folder, arrays) -> list[str]:
    """Arrays of *folder* that can give x values to *arrays*: numeric, one
    dimension, as many points as each of *arrays*, and not one of them."""
    sizes = {d.nparray.size for d in arrays if d.nparray is not None}
    if len(sizes) != 1:
        return []
    size = sizes.pop()
    own = {id(d) for d in arrays}
    return [
        name for name, data in folder.data.items()
        if id(data) not in own and isinstance(data.nparray, np.ndarray)
        and data.nparray.ndim == 1 and data.nparray.size == size
        and data.nparray.dtype.kind in "biuf"
    ]


def x_array_scales(folder, names: list[str]) -> dict[str, tuple[str, str]]:
    """Each array's (y label, y units): what its values are, e.g. ("Time", "ms")."""
    return {name: (folder.data[name].yscale.label, folder.data[name].yscale.units)
            for name in names if name in folder.data}


def x_array_current(arrays) -> str | None:
    """The shared x choice of *arrays*: an array name, "" for start and
    delta, or None if they differ."""
    names = {d.xarray_name or ("(x values)" if d.xarray is not None else "")
             for d in arrays}
    return names.pop() if len(names) == 1 else None


def x_array_error(folder, arrays, name: str) -> str | None:
    """Why array *name* can't give x values to *arrays*, or None."""
    from pyneuromatic.core.nm_data import check_xarray

    source = folder.data.get(name)
    if source is None:
        return None  # e.g. an x array in another folder: unchanged
    try:
        for data in arrays:
            if data is source:
                return f"'{name}' can't be its own x array"
            check_xarray(source, data)
    except (TypeError, ValueError) as error:
        return str(error)[0].upper() + str(error)[1:]
    return None


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
        self._start: dict = {}  # see starting_on

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

    def _ask_scales(self, x, y_rows, context, error, **x_options) -> dict | None:
        """The Edit Scales dialog (replaced in tests); returns the changes.
        *x_options*: ``x_arrays``, ``x_array`` and ``validate_x`` (see
        :class:`EditScalesDialog`)."""
        dialog = EditScalesDialog(x, y_rows, context, error, parent=self._parent_widget,
                                  **x_options)
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

    @contextlib.contextmanager
    def starting_on(self, context=None, **items):
        """Run actions on *context* (a folder or subfolder) with *items* as
        their starting choices (e.g. ``dataseries=series``, ``data_set="Set1"``),
        instead of the strip's selection: e.g. for an item right-clicked in
        the browser, so it gets the same prompt as the strip's ⋯ menu. The
        strip's selection is changed only if *context* is the strip's."""
        saved = self._start
        self._start = {"context": context, **items}
        try:
            yield self
        finally:
            self._start = saved

    def _folder(self):
        context = self._start.get("context")
        if context is not None:
            return context._folder
        return self._selection.selection.get("folder")

    def _context(self):
        """The subfolder if one is selected, else the folder."""
        context = self._start.get("context")
        if context is not None:
            return context
        return self._selection.selection.get("toolfolder") or self._folder()

    def _context_text(self) -> str:
        context = self._start.get("context")
        if context is not None:
            return _owner_text(context)
        folder = self._folder()
        subfolder = self._selection.selection.get("toolfolder")
        names = [obj.name for obj in (folder, subfolder) if obj is not None]
        return " / ".join(names)

    def _current_name(self, key: str) -> str | None:
        value = self._start[key] if key in self._start else self._selection.selection.get(key)
        if isinstance(value, str) or value is None:
            return value
        return value.name

    def _elsewhere(self) -> bool:
        """True while acting (starting_on) on a folder that isn't the strip's."""
        context = self._start.get("context")
        if context is None:
            return False
        selection = self._selection.selection
        return context is not (selection.get("toolfolder") or selection.get("folder"))

    def _select(self, **changes) -> None:
        """Change the strip's selection, unless acting on another folder."""
        if not self._elsewhere():
            self._selection.update(**changes)

    def _changed(self) -> None:
        self._selection.refresh()
        self.update_enabled()
        self.content_changed.emit()


class FolderActions(TierActions):
    """Folder and subfolder (toolfolder) actions."""

    # Ask the app to show a folder or subfolder in the browser
    browse_requested = QtCore.pyqtSignal(object)

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
        self.duplicate_action = action("Copy…", self.duplicate_folder,
                                       tip="Copy the folder with its data")
        self.rename_action = action("Rename…", self.rename_folder,
                                    tip="Rename the folder")
        self.delete_action = action("Delete…", self.delete_folder,
                                    tip="Delete the folder and its data")
        self.subfolder_rename_action = action("Rename…", self.rename_subfolder,
                                              tip="Rename the subfolder")
        self.subfolder_duplicate_action = action("Copy…", self.duplicate_subfolder,
                                                 tip="Copy the subfolder")
        self.subfolder_delete_action = action("Delete…", self.delete_subfolder,
                                              tip="Delete the subfolder and its data")
        self.browse_action = action("Show in Browser", self.folder_in_browser,
                                    tip="Show the folder in the browser")
        self.subfolder_browse_action = action("Show in Browser", self.subfolder_in_browser,
                                              tip="Show the subfolder in the browser")

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    # --- menus --------------------------------------------------------------

    def folder_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Folder", parent)
        menu.addAction(self.new_action)
        menu.addAction(self.open_action)
        menu.addSeparator()
        menu.addAction(self.rename_action)
        menu.addAction(self.duplicate_action)
        menu.addAction(self.save_action)
        menu.addAction(self.delete_action)
        menu.addSeparator()
        menu.addAction(self.browse_action)
        return menu

    def subfolder_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Subfolder", parent)
        menu.addAction(self.subfolder_rename_action)
        menu.addAction(self.subfolder_duplicate_action)
        menu.addAction(self.subfolder_delete_action)
        menu.addSeparator()
        menu.addAction(self.subfolder_browse_action)
        return menu

    def update_enabled(self, *_args) -> None:
        # Actions choose their folder in their dialog, so they only need one
        folder = self._folder()
        has_folders = len(self._folders) > 0
        has_subfolders = folder is not None and len(folder.toolfolders) > 0
        for item in (self.duplicate_action, self.rename_action, self.delete_action):
            item.setEnabled(has_folders)
        self.save_action.setEnabled(folder is not None)
        for item in (self.subfolder_rename_action, self.subfolder_duplicate_action,
                     self.subfolder_delete_action):
            item.setEnabled(has_subfolders)
        self.browse_action.setEnabled(folder is not None)
        self.subfolder_browse_action.setEnabled(
            self._selection.selection.get("toolfolder") is not None)

    def folder_in_browser(self) -> None:
        folder = self._folder()
        if folder is not None:
            self.browse_requested.emit(folder)

    def subfolder_in_browser(self) -> None:
        subfolder = self._selection.selection.get("toolfolder")
        if subfolder is not None:
            self.browse_requested.emit(subfolder)

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
        self._select(folder=folder)
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
            self._select(folder=imported)
            self._changed()
            # Shown in the Browser, opened: its Info lists the imported metadata
            self.browse_requested.emit(imported)
        if problems:
            self._show_message("Open / Import", "\n".join(problems), warning=True)

    def save_folder(self) -> None:
        self._show_message("Save", "Save is under construction.")

    # Each folder action works on the selected folder, or on *name* (e.g. a
    # folder right-clicked in the browser), which its dialog starts on

    def duplicate_folder(self, name: str | None = None) -> None:
        folders = self._folders
        answer = self._choose_and_name(
            folders, "Copy Folder", "Folder", name or self._current_name("folder"),
            lambda item: unique_name(folders, f"{item}_copy"),
        )
        if answer is None:
            return
        copy = folders.duplicate(answer[0], answer[1])
        self._select(folder=copy)
        self._changed()

    def rename_folder(self, name: str | None = None) -> None:
        folders = self._folders
        answer = self._choose_and_name(
            folders, "Rename Folder", "Folder", name or self._current_name("folder"),
            lambda item: item,
        )
        if answer is None or answer[0] == answer[1]:
            return
        folders.rename(answer[0], answer[1])
        self._changed()

    def delete_folder(self, name: str | None = None) -> None:
        folders = self._folders
        if len(folders) == 0:
            return
        answer = self._choose(
            "Delete Folder",
            item_label="Folder",
            items=list(folders.keys()),
            current=name or self._current_name("folder"),
            message="The folder and all its data will be deleted.",
        )
        if answer is None:
            return
        self.delete_folders([answer[0]])

    def delete_folders(self, names: list[str]) -> None:
        """Delete folders *names* (no questions asked); if the selected one
        goes, select its neighbour."""
        folders = self._folders
        current = self._current_name("folder")
        order = list(folders.keys())
        neighbour = None
        if current in names:
            index = order.index(current)
            neighbour = next((n for n in order[index + 1:] + order[:index][::-1]
                              if n not in names), None)
        for name in names:
            folders.pop(name)
        if current in names:
            if neighbour is not None:
                self._select(folder=folders[neighbour])
            else:
                self._selection.clear()
        self._changed()

    # --- subfolder actions --------------------------------------------------

    # Each subfolder action works on the selected folder's subfolders, or on
    # *subfolder* (e.g. one right-clicked in the browser), which its dialog
    # starts on

    def _subfolder_target(self, subfolder) -> tuple:
        """(folder, current subfolder name) for a subfolder action."""
        if subfolder is not None:
            return subfolder._folder, subfolder.name
        return self._folder(), self._current_name("toolfolder")

    def duplicate_subfolder(self, subfolder=None) -> None:
        folder, current = self._subfolder_target(subfolder)
        if folder is None:
            return
        container = folder.toolfolders
        answer = self._choose_and_name(
            container, "Copy Subfolder", "Subfolder", current,
            lambda item: unique_name(container, f"{item}_copy"),
        )
        if answer is None:
            return
        copy = container.duplicate(answer[0], answer[1])
        if folder is self._folder():
            self._select(toolfolder=copy)
        self._changed()

    def rename_subfolder(self, subfolder=None) -> None:
        folder, current = self._subfolder_target(subfolder)
        if folder is None:
            return
        container = folder.toolfolders
        answer = self._choose_and_name(
            container, "Rename Subfolder", "Subfolder", current, lambda item: item,
        )
        if answer is None or answer[0] == answer[1]:
            return
        container.rename(answer[0], answer[1])
        self._changed()

    def delete_subfolder(self, subfolder=None) -> None:
        folder, current = self._subfolder_target(subfolder)
        if folder is None or len(folder.toolfolders) == 0:
            return
        answer = self._choose(
            "Delete Subfolder",
            item_label="Subfolder",
            items=list(folder.toolfolders.keys()),
            current=current,
            message="The subfolder and all its data will be deleted.",
        )
        if answer is None:
            return
        self.delete_subfolders(folder, [answer[0]])

    def delete_subfolders(self, folder, names: list[str]) -> None:
        """Delete *folder*'s subfolders *names* (no questions asked)."""
        if folder is self._folder() and self._current_name("toolfolder") in names:
            self._select(toolfolder=None)
        for name in names:
            folder.toolfolders.pop(name)
        self._changed()


def _owner_text(owner) -> str:
    """e.g. "Demo" or "Demo / Stats0" for a folder or subfolder."""
    folder = owner._folder
    return " / ".join(o.name for o in (folder, owner if owner is not folder else None)
                      if o is not None)


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


# Sets are numbered, like groups: Set0, Set1, Set2... (SetX excludes epochs).
# Their notes say what they are; short names keep tool subfolders short.

def set_name(number) -> str:
    return f"Set{int(number)}"


def next_set_number(sets) -> int:
    """The first unused set number."""
    number = 0
    while set_name(number) in sets:
        number += 1
    return number


def set_number_spin(sets, value: int | None = None) -> tuple:
    """The "New Set number" box for a dialog, starting at the next free number."""
    return ("New Set number", Spin(0, 9999, next_set_number(sets) if value is None else value))


def set_number_error(sets, number, keep: str | None = None) -> str | None:
    """Why Set<number> can't be made (it exists), or None; *keep* is a set
    being renumbered, which may keep its number."""
    name = set_name(number)
    if keep is not None and name.lower() == keep.lower():
        return None
    if name in sets:
        return f"'{name}' already exists."
    return None


def new_set_summary(sets, number) -> str:
    """e.g. "Create 'Set2', an empty set." ("" if Set<number> exists)."""
    if set_number_error(sets, number):
        return ""
    return f"Create '{set_name(number)}', an empty set."


def renumber_set_summary(sets, old: str | None, number) -> str:
    """e.g. "Renumber 'Set1' to 'Set3'." ("" if Set<number> exists)."""
    if not old or set_number_error(sets, number, keep=old):
        return ""
    new = set_name(number)
    if new.lower() == old.lower():
        return f"Renumber '{old}' to '{new}' (no change)."
    return f"Renumber '{old}' to '{new}'."


def set_number_of(name: str) -> int | None:
    """3 for "Set3"; None for other names (e.g. SetX or imported names)."""
    suffix = name[3:]
    return int(suffix) if name[:3].lower() == "set" and suffix.isdigit() else None


def set_name_error(sets, name: str | None) -> str | None:
    """Why *name* cannot be a new set in *sets*, or None if it can."""
    if not name:
        return "Enter a name"
    if not nmu.name_ok(name):
        return f"'{name}' is not a valid name (use letters, digits and _)"
    if name.lower() in (key.lower() for key in sets.keys()):
        return f"'{name}' already exists."
    return None


def set_invert_error(sets, name: str | None, kind: str = "data") -> str | None:
    """Why set *name* can't be inverted, or None: a combined set follows its
    two sets; SetX would exclude every other epoch from runs."""
    if name is None:
        return None
    if sets.is_equation(name):
        return (f"'{name}' is a combined set (AND / OR of two sets); invert one of "
                "those sets instead")
    if kind == "epoch" and name.lower() == EXCLUDE_SET.lower():
        return f"{EXCLUDE_SET} can't be inverted: every other epoch would be left out of runs"
    return None


def invert_in_place(sets, name: str, items: list[str]) -> None:
    """Make set *name* hold the items of *items* it didn't hold (in order)."""
    members = set(sets.get_items(name, get_keys=True) or [])
    others = [item for item in items if item not in members]
    sets.empty(name)
    if others:
        sets.add(name, others)


def invert_set_summary(sets, name: str, total: int, kind: str) -> str:
    """e.g. "Invert epoch assignments of 'Set0'; 6 assigned : 24 unassigned
    → 6 unassigned : 24 assigned" """
    assigned = len(sets.get_items(name, get_keys=True) or [])
    unassigned = total - assigned
    return (f"Invert {kind} assignments of '{name}'; "
            f"{assigned} assigned : {unassigned} unassigned → "
            f"{assigned} unassigned : {unassigned} assigned")


def overwrite_check(sets) -> tuple:
    """An "Overwrite" checkbox to follow a "New Set number" box (the last two
    extras); it can be ticked only while that number's set exists."""
    return ("", Check("Overwrite existing set", enabled=lambda _item, *extras: (
        set_number_error(sets, extras[-2]) is not None)))


def set_uses(sets, name: str, target: str) -> bool:
    """Whether set *name* is *target*, or is combined from it."""
    if name.lower() == target.lower():
        return True
    value = sets.get_items(name, get_equation=True)
    return sets.tuple_is_equation(value) and any(
        set_uses(sets, other, target) for other in value[1:])


def replace_set(sets, name: str, fill: Callable[[], object]) -> None:
    """Make set *name* with *fill()*, replacing it (in its place in the list)
    if it exists."""
    order = sets.keys()
    if name in sets:
        sets.pop(name, quiet=True)
    fill()
    if len(order) == len(sets):
        sets.reorder(order)


def copy_set_options(sets) -> dict:
    """The Copy Set prompt's fields after the set: New Set number, Overwrite."""

    def error(item, _name, number, overwrite):
        if overwrite and set_name(number).lower() == (item or "").lower():
            return f"'{item}' can't be copied onto itself."
        return None if overwrite else set_number_error(sets, number)

    def summary(item, number, overwrite):
        """e.g. "Copy 'Set1' to 'Set4'." ("" while there's an error)."""
        if not item or error(item, None, number, overwrite):
            return ""
        kind = (" (its current members; the copy isn't combined)"
                if sets.is_equation(item) else "")
        if overwrite:
            return f"Overwrite '{set_name(number)}' with a copy of '{item}'{kind}."
        return f"Copy '{item}' to '{set_name(number)}'{kind}."

    return dict(extras=[set_number_spin(sets), overwrite_check(sets)],
                message=summary, validate=error, validate_live=True)


def copy_set_to(sets, answer) -> str:
    """Copy the set chosen in a Copy Set prompt; returns the new set's name."""
    source, _name, number, _overwrite = answer
    new = set_name(number)
    items = list(sets.get_items(source, get_keys=True) or [])
    replace_set(sets, new, lambda: sets.add(new, items))
    return new


def combine_set_options(sets, names: list[str], first: str) -> dict:
    """The Combine Sets prompt's fields: Combine (the set), Operator, With,
    New Set number and Overwrite, starting with two different sets."""
    second = next((n for n in names if n != first), first)

    def error(item, _name, _operator, other, number, overwrite):
        if item == other:
            return "Choose two different sets."
        target = set_name(number)
        if overwrite and (set_uses(sets, item, target) or set_uses(sets, other, target)):
            return f"'{target}' can't be overwritten: the combination uses it."
        return None if overwrite else set_number_error(sets, number)

    def summary(item, operator, other, number, overwrite):
        """e.g. "Set2 = Set0 OR Set1" ("" while there's an error)."""
        if not item or error(item, None, operator, other, number, overwrite):
            return ""
        return (f"{set_name(number)} = {item} {operator} {other}"
                + (" (overwrite)" if overwrite else ""))

    return dict(item_label="Combine",
                extras=[("Operator", ["AND", "OR"], "OR"), ("With", names, second),
                        set_number_spin(sets), overwrite_check(sets)],
                message=summary, validate=error, validate_live=True)


def combine_sets_to(sets, answer) -> str:
    """Make the set a Combine Sets prompt defines; returns its name."""
    first, _name, operator, second, number, _overwrite = answer
    name = set_name(number)
    define = sets.define_and if operator == "AND" else sets.define_or
    replace_set(sets, name, lambda: define(name, first, second))
    return name


def clear_set_summary(sets, name: str, noun: str) -> str:
    """e.g. "Remove 3 epochs from 'Set1'." """
    if not name:
        return ""
    count = len(sets.get_items(name, get_keys=True) or [])
    if count == 1:
        noun = noun.replace("epochs", "epoch").replace("arrays", "array")
    return f"Remove {count} {noun} from '{name}'."


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
        self.rename_action = action("Rename…", self.rename_data, tip="Rename a data array")
        self.scales_action = action("Edit Scales…", self.edit_scales,
                                    tip="Edit the x and y labels, units, start and delta")
        self.copy_action = action(
            "Copy…", self.copy_data,
            tip="Copy a data array, in this folder or to another")
        self.move_action = action(
            "Move…", self.move_data_to_folder,
            tip="Move a data array (not part of a data series) to another folder")
        self.delete_action = action("Delete…", self.delete_data, tip="Delete a data array")
        self.browse_action = action(
            "Show in Browser", self.choose_in_browser,
            tip="Select several data arrays in the browser (Shift/Cmd-click)",
        )
        self.set_new_action = action("New Set…", self.new_set, tip="Create an empty data set")
        self.set_add_action = action("Add to Set…", self.add_to_set,
                                     tip="Add a data array to a data set")
        self.set_remove_action = action("Remove from Set…", self.remove_from_set,
                                        tip="Remove a data array from a data set")
        self.set_copy_action = action("Copy…", self.copy_set,
                                      tip="Copy a data set to a new set")
        self.set_invert_action = action(
            "Invert…", self.invert_set,
            tip="Invert a data set in place: it holds the data arrays it didn't")
        self.set_combine_action = action("Combine…", self.combine_sets,
                                         tip="Make a set from two sets with AND or OR")
        self.set_clear_action = action("Clear…", self.clear_set,
                                       tip="Remove all data from a set, keeping the set")
        self.set_rename_action = action("Renumber…", self.rename_set,
                                        tip="Give a data set another number")
        self.set_delete_action = action("Delete…", self.delete_set, tip="Delete a data set")
        self.set_browse_action = action(
            "Show in Browser", self.choose_sets_in_browser,
            tip="Show the data sets in the browser",
        )

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    # --- menus --------------------------------------------------------------

    def data_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Data", parent)
        menu.addAction(self.new_action)
        menu.addSeparator()
        menu.addAction(self.scales_action)
        menu.addSeparator()
        for item in (self.rename_action, self.copy_action, self.move_action,
                     self.delete_action):
            menu.addAction(item)
        menu.addSeparator()
        menu.addAction(self.browse_action)
        return menu

    def data_set_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Data Set", parent)
        menu.addAction(self.set_new_action)
        menu.addSeparator()
        menu.addAction(self.set_add_action)
        menu.addAction(self.set_remove_action)
        menu.addSeparator()
        menu.addAction(self.set_rename_action)
        menu.addAction(self.set_copy_action)
        menu.addAction(self.set_invert_action)
        menu.addAction(self.set_combine_action)
        menu.addAction(self.set_clear_action)
        menu.addAction(self.set_delete_action)
        menu.addSeparator()
        menu.addAction(self.set_browse_action)
        return menu

    def update_enabled(self, *_args) -> None:
        data = self._data_container()
        has_data = data is not None and len(data) > 0
        has_sets = data is not None and len(list(data.sets.keys())) > 0
        for item in (self.copy_action, self.rename_action, self.delete_action):
            item.setEnabled(has_data)
        container = self._data_container()
        in_folder = container is not None and isinstance(container._parent, NMFolder)
        self.move_action.setEnabled(
            has_data and in_folder and len(self._selection.manager.folders) > 1)
        self.browse_action.setEnabled(data is not None)
        self.new_action.setEnabled(data is not None)
        self.scales_action.setEnabled(self._selection.selection.get("data") is not None)
        self.set_browse_action.setEnabled(data is not None)
        self.set_new_action.setEnabled(data is not None)
        for item in (self.set_add_action, self.set_remove_action, self.set_combine_action,
                     self.set_clear_action, self.set_rename_action, self.set_delete_action,
                     self.set_copy_action, self.set_invert_action):
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

    def _data_target(self, data=None) -> tuple:
        """(container, owner, folder text, current name) for a data prompt:
        *data*'s (e.g. an array right-clicked in the browser), or the strip's
        selected folder and array."""
        if data is None:
            return (self._data_container(), self._context(), self._context_text(),
                    self._current_name("data"))
        container = data._container
        owner = getattr(container, "_parent", None)
        folder = data._folder
        text = " / ".join(obj.name for obj in (folder, owner if owner is not folder else None)
                          if obj is not None)
        return container, owner, text, data.name

    def _selects_here(self, container) -> bool:
        """True if *container* is the strip's (so a new array can be selected)."""
        return container is self._data_container()

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
    def _xarray_error(container, name: str | None) -> str | None:
        """Why *name* can't be deleted: other arrays take their x values from it."""
        if name is None or name not in container:
            return None
        text = container.xarray_in_use_text(name)
        return text[0].upper() + text[1:] + "." if text else None

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
        self._select(data=data)
        self._changed()
        self.table_requested.emit()

    def copy_data(self, data=None) -> None:
        """Copy a data array in this folder (named C_name by default) or to
        another folder (keeping its name; a clash there is shown at once).
        Its x array comes too. Starts on *data* (e.g. from the browser), or
        the strip's array."""
        container, context_owner, here, current = self._data_target(data)
        if container is None or len(container) == 0:
            return
        folders = self._selection.manager.folders
        here_folder = data._folder if data is not None else self._folder()
        targets = {here: (container, context_owner)}
        for name, folder in folders.items():
            if folder is not here_folder:
                targets[name] = (folder.data, folder)

        def default_name(item, target):
            destination = targets[target][0]
            return copy_name(destination, item) if destination is container else item

        def error(item, name, target):
            destination, owner = targets[target]
            problem = name_error(destination, name) or series_name_error(owner, name)
            if problem and name and destination is not container:
                return f"{problem}. Enter a new name."
            return problem

        answer = self._choose(
            "Copy Data",
            context=("From folder", here),
            item_label="Data",
            items=list(container.keys()),
            current=current,
            extras=[("To folder", list(targets), here)],
            name_label="Name",
            default_name=default_name,
            validate=error, validate_live=True,
        )
        if answer is None:
            return
        item, name, target = answer
        destination, owner = targets[target]
        if not self._confirm_series_name("Copy Data", series_name_hint(owner, name), "Copy"):
            return
        if destination is container:
            copy = container.duplicate(item, name)
            if self._selects_here(container):
                self._select(data=copy)
        else:
            container.copy_to(item, destination, name)
        self._changed()

    def rename_data(self, data=None) -> None:
        """Rename a data array: *data* (e.g. from the browser), or the strip's."""
        container, owner, _text, current = self._data_target(data)
        if container is None or len(container) == 0:
            return
        answer = self._choose(
            "Rename Data",
            item_label="Data",
            items=list(container.keys()),
            current=current,
            name_label="New name",
            default_name=lambda item: item,
            # A series member is a warning: red
            message=lambda item, *_: (
                warning_html(self._series_note(container, item))
                if self._series_note(container, item) else ""),
            validate=lambda item, name: (
                self._lock_error(container, item)
                or (None if name == item else (
                    name_error(container, name)
                    or series_name_error(owner, name)
                ))
            ),
        )
        if answer is None or answer[0] == answer[1]:
            return
        if not self._confirm_series_name(
            "Rename Data",
            series_name_hint(owner, answer[1], joins_now=True),
            "Rename",
        ):
            return
        container.rename(answer[0], answer[1])
        self._changed()

    def edit_scales(self, data=None) -> None:
        """Edit the scales of *data* (e.g. the array the browser shows), or
        of the selected data array."""
        if data is None:
            data = self._selection.selection.get("data")
            container = self._data_container()
            context = self._context_text()
        else:
            container = data._container
            owner = getattr(container, "_parent", None)
            folder = data._folder
            context = " / ".join(
                obj.name for obj in (folder, owner if owner is not folder else None)
                if obj is not None
            )
        if data is None or container is None:
            return
        lock_error = self._lock_error(container, data.name)
        # X Array: chosen here for a plain array; an array of a data series
        # shows its series' (set in the series' Edit Scales)
        owner = container._parent
        series = container.dataseries_of(data.name) if data.name in container else []
        choices = [] if series else x_array_choices(owner, [data])
        fixed = (f"Set by data series '{series[0].name}' (its Edit Scales)"
                 if series else None)
        changes = self._ask_scales(
            scale_fields([data]),
            [("Y", data.yscale.label, data.yscale.units)],
            ("Data", f"{context} › {data.name}"),
            lock_error,
            x_arrays=choices, x_array=x_array_current([data]),
            validate_x=lambda name: x_array_error(owner, [data], name),
            x_array_scales=x_array_scales(owner, choices),
            x_array_fixed=fixed,
        )
        if not changes or lock_error:  # the dialog only shows a locked array
            return
        if "xarray" in changes and not series:
            name = changes["xarray"]
            source = container.get(name) if name else None
            if not name or source is not None:  # e.g. one in another folder: kept
                data.set_xarray_from(source)
        for key, attribute in (("xstart", "start"), ("xdelta", "delta"),
                               ("xlabel", "label"), ("xunits", "units")):
            if key in changes:
                setattr(data.xscale, attribute, changes[key])
        for field, value in changes.get("y", {}).get("Y", {}).items():
            setattr(data.yscale, field, value)
        self._changed()

    def move_data_to_folder(self, data=None) -> None:
        """Move a data array (the same array) to another folder; arrays of a
        data series are refused (move the whole series). Starts on *data*
        (e.g. from the browser), or the strip's array."""
        container, owner, here, current = self._data_target(data)
        folders = self._selection.manager.folders
        if not isinstance(owner, NMFolder) or len(container) == 0:
            return
        targets = [name for name in folders.keys() if name != owner.name]
        if not targets:
            return

        def error(item, name, target):
            try:
                owner.move_data_plan(item, folders[target])
            except (KeyError, ValueError, RuntimeError) as problem:
                text = str(problem.args[0]) if problem.args else str(problem)
                return text[0].upper() + text[1:] + "."
            problem = name_error(folders[target].data, name) \
                or series_name_error(folders[target], name)
            return f"{problem}. Enter a new name." if problem and name else problem

        def message(item, target):
            note = owner.move_data_note(item, folders[target])
            return "The array moves (not a copy)." + (f" {note}" if note else "")

        answer = self._choose(
            "Move Data to Folder",
            context=("From folder", here),
            item_label="Data",
            items=list(container.keys()),
            current=current,
            extras=[("To folder", targets, targets[0])],
            name_label="Name",
            default_name=lambda item, _target: item,
            message=message, validate=error, validate_live=True,
        )
        if answer is None:
            return
        item, name, target = answer
        owner.move_data(item, folders[target], new_name=name)
        self._changed()

    def delete_data(self, data=None) -> None:
        """Delete a data array: *data* (e.g. from the browser), or the strip's."""
        container, _owner, _text, current = self._data_target(data)
        if container is None or len(container) == 0:
            return
        answer = self._choose(
            "Delete Data",
            item_label="Data",
            items=list(container.keys()),
            current=current,
            # A series member is a warning: red
            message=lambda item, *_: (
                warning_html(self._series_note(container, item))
                if self._series_note(container, item) else "The data array will be deleted."),
            validate=lambda item, name: self._lock_error(container, item)
            or self._xarray_error(container, item),
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
        options.setdefault("item_label", "")  # the names say it: Set0, Set1...
        return self._choose(
            title,
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
            extras=[set_number_spin(sets)],
            message=lambda _item, number: new_set_summary(sets, number),
            validate=lambda _item, _name, number: set_number_error(sets, number),
            validate_live=True,
        )
        if answer is None:
            return
        name = set_name(answer[-1])
        sets.add(name)
        self._select(data_set=name)
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
        answer = self._choose_set(
            "Combine Data Sets", **combine_set_options(sets, names, first))
        if answer is None:
            return
        self._select(data_set=combine_sets_to(sets, answer))
        self._changed()

    def copy_set(self) -> None:
        """Copy a data set's members to a new set (a list of its own)."""
        container = self._data_container()
        if container is None:
            return
        sets = container.sets
        answer = self._choose_set("Copy Data Set", **copy_set_options(sets))
        if answer is None:
            return
        self._select(data_set=copy_set_to(sets, answer))
        self._changed()

    def invert_set(self) -> None:
        """Invert a data set in place: it holds the folder's data arrays it
        didn't (inverting again restores it)."""
        container = self._data_container()
        if container is None:
            return
        sets = container.sets
        items = list(container.keys())

        def summary(name, *_):
            if set_invert_error(sets, name):
                return ""
            return invert_set_summary(sets, name, len(items), "data")

        answer = self._choose_set(
            "Invert Data Set", message=summary,
            validate=lambda name, _name: set_invert_error(sets, name),
        )
        if answer is None:
            return
        invert_in_place(sets, answer[0], items)
        self._changed()

    def clear_set(self) -> None:
        answer = self._choose_set(
            "Clear Data Set", message=lambda name: clear_set_summary(
                self._data_container().sets, name, "data arrays"),
        )
        if answer is None:
            return
        self._data_container().sets.empty(answer[0])
        self._changed()

    def rename_set(self) -> None:
        container = self._data_container()
        sets = container.sets if container is not None else None
        answer = self._choose_set(
            "Renumber Data Set",
            extras=[set_number_spin(sets) if sets is not None else ("New Set number", Spin(0, 9999, 0))],
            message=lambda item, number: renumber_set_summary(sets, item, number),
            validate=lambda item, _name, number: set_number_error(sets, number, keep=item),
            validate_live=True,
        )
        if answer is None or set_name(answer[-1]).lower() == answer[0].lower():
            return
        old, new = answer[0], set_name(answer[-1])
        was_current = old == self._current_name("data_set")
        sets.rename(old, new)
        if was_current:
            self._select(data_set=new)
        self._changed()

    def delete_set(self) -> None:
        answer = self._choose_set(
            "Delete Data Set", message=lambda name: (
                f"Delete '{name}'; data assigned to this set will be unaltered.")
        )
        if answer is None:
            return
        if answer[0] == self._current_name("data_set"):
            self._select(data_set=None)
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
        self.copy_action = action(
            "Copy…", self.copy_series,
            tip="Copy the series, in this folder or to another (the copy is unlocked)")
        self.rename_action = action("Rename…", self.rename_series,
                                    tip="Rename the series prefix and its arrays")
        self.scales_action = action("Edit Scales…", self.edit_scales,
                                    tip="Edit the series' x and y labels, units, start and delta")
        self.lock_action = action("Lock", self.toggle_lock,
                                  tip="A locked series' arrays cannot be changed")
        self.lock_action.setCheckable(True)
        self.move_action = action(
            "Move…", self.move_series,
            tip="Move the series, with its arrays, sets and groups, to another folder")
        self.append_action = action(
            "Append…", self.append_series,
            tip="Add another series' epochs (from this or another folder) after this "
                "series' last epoch")
        self.add_epochs_action = action(
            "Add Epochs…", self.add_epochs,
            tip="Add epochs after the last one: a new array for each channel")
        self.remove_channel_action = action("Remove Channel…", self.remove_channel,
                                            tip="Remove a channel from the series")
        self.renumber_action = action(
            "Renumber Epochs…", self.renumber_epochs,
            tip="Number the epochs E0, E1, … in order, closing gaps, and rename "
                "their arrays to match")
        self.remove_epoch_action = action("Remove Epochs…", self.remove_epoch,
                                          tip="Remove an epoch, or a range of epochs")
        self.delete_action = action("Delete…", self.delete_series,
                                    tip="Delete the series, optionally with its arrays")
        self.browse_action = action("Show in Browser", self.choose_in_browser,
                                    tip="Select the series in the browser")

        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def series_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Series", parent)
        for item in (self.new_action, self.from_data_action, self.update_action, None,
                     self.scales_action, None,
                     self.rename_action, self.copy_action,
                     self.move_action, self.append_action, self.delete_action, None,
                     self.add_epochs_action, self.remove_epoch_action,
                     self.renumber_action, None,
                     self.lock_action, None, self.browse_action):
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
        if "dataseries" in self._start:  # see starting_on
            return self._start["dataseries"]
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
        return (self.scales_action, self.add_epochs_action, self.remove_channel_action,
                self.remove_epoch_action, self.move_action, self.append_action,
                self.renumber_action)

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
                     self.remove_epoch_action, self.scales_action, self.add_epochs_action,
                     self.renumber_action):
            item.setEnabled(series is not None)
        # Folder-level series only (not those of a results subfolder)
        at_folder = series is not None and isinstance(owner, NMFolder)
        others = [n for n in self._selection.manager.folders.keys()
                  if not at_folder or n != owner.name]
        self.move_action.setEnabled(at_folder and bool(others))
        self.append_action.setEnabled(at_folder and any(
            self._append_sources(folder) for folder in self._selection.manager.folders.values()))
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
            return (f"Data series '{name}' is locked. Untick Lock (Prefix ⋯ menu) "
                    "to change it.")
        return None

    def _choose_series(self, title: str, owner=None, current: str | None = None, **options):
        """Choose a series of the selected folder, or of *owner* (starting at
        *current*, e.g. a series right-clicked in the browser)."""
        if owner is None:
            owner = self._owner()
            series = self._series()
            current = series.name if series is not None else None
            context = self._context_text()
        else:
            context = _owner_text(owner)
        if owner is None or len(owner.dataseries) == 0:
            return None
        options.setdefault("context", ("Folder", context))
        return self._choose(
            title, item_label="Series", items=list(owner.dataseries.keys()),
            current=current, **options,
        )

    # --- actions ------------------------------------------------------------

    def _ask_new_series(self, defaults: dict, validate) -> dict | None:
        """The New Data Series dialog (replaced in tests)."""
        return ask_new_data(
            defaults, validate, context=("Folder", self._context_text()),
            parent=self._parent_widget, series=True,
        )

    def new_series(self, owner=None) -> None:
        """New data series in the selected folder, or in *owner* (e.g. the
        folder whose Data Series row was right-clicked in the browser)."""
        selected_owner = self._owner()
        owner = owner if owner is not None else selected_owner
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
        fill = fill_values(values["value"], values["noise"])
        series = owner.new_dataseries(
            values["name"],
            n_channels=values["n_channels"], n_epochs=values["n_epochs"],
            n_points=values["points"], dx=values["xdelta"], x_start=values["xstart"],
            x_label=values["xlabel"], x_units=values["xunits"],
            y_label=values["ylabel"], y_units=values["yunits"], fill=fill,
        )
        if owner is selected_owner:  # a series elsewhere isn't selectable here
            self._select(dataseries=series)
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
        self._select(dataseries=series)
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
        """Copy a series in this folder (under a new prefix, C_Record by
        default) or to another folder (keeping its prefix; a clash there is
        shown at once). The copy is unlocked, with its sets, groups and x
        array."""
        owner = self._owner()
        folders = self._selection.manager.folders
        here = self._folder()
        if owner is None:
            return
        targets = ([here.name] if here is not None else []) + [
            name for name in folders.keys() if here is None or name != here.name]

        def target_owner(target):
            return owner if here is not None and target == here.name else folders[target]

        def default_prefix(item, target):
            destination = target_owner(target)
            if destination is not owner:
                return item  # another folder: the same prefix
            suffixes = series_suffixes(owner.dataseries[item])
            candidate, number = f"{COPY_PREFIX}_{item}", 1
            while prefix_error(destination, candidate, suffixes) is not None:
                candidate, number = f"{COPY_PREFIX}{number}_{item}", number + 1
            return candidate

        def error(item, name, target):
            problem = prefix_error(target_owner(target), name,
                                   series_suffixes(owner.dataseries[item]))
            return f"{problem}. Enter a new prefix." if problem and name else problem

        answer = self._choose_series(
            "Copy Series",
            extras=[("To folder", targets, targets[0] if targets else None)],
            name_label="Prefix",
            default_name=default_prefix,
            validate=error, validate_live=True,
        )
        if answer is None:
            return
        item, name, target = answer
        destination = target_owner(target)
        copy = owner.copy_dataseries(
            item, name, folder=None if destination is owner else destination)
        if destination is owner:
            self._select(dataseries=copy)
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

    def edit_scales(self, series=None) -> None:
        """Edit the selected series' scales, or *series*' (e.g. selected in
        the browser)."""
        if series is None:
            series, owner = self._series(), self._owner()
            if series is None:
                return
            context = self._series_context(series)
        else:
            owner = series._container._parent
            context = ("Series", f"{_owner_text(owner)} › {series.name}")
        arrays = [d for ch in series.channels.values() for d in ch.data]
        y_rows = [
            (name, common_value(d.yscale.label for d in channel.data),
             common_value(d.yscale.units for d in channel.data))
            for name, channel in series.channels.items()
        ]
        lock_error = self._lock_error(owner, series.name)
        choices = x_array_choices(owner, arrays)
        changes = self._ask_scales(
            scale_fields(arrays), y_rows, context, lock_error,
            x_arrays=choices, x_array=x_array_current(arrays),
            validate_x=lambda name: x_array_error(owner, arrays, name),
            x_array_scales=x_array_scales(owner, choices),
        )
        if not changes or lock_error:  # the dialog only shows a locked series
            return
        self._apply_x_array(owner, series, changes)
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

    @staticmethod
    def _apply_x_array(owner, series, changes: dict, channel: str | None = None) -> None:
        """Set or clear the x array chosen in Edit Scales."""
        if "xarray" not in changes:
            return
        name = changes["xarray"]
        source = owner.data.get(name) if name else None
        if name and source is None:
            return  # e.g. an x array in another folder: unchanged
        series.set_xarray(source, channel=channel)

    def edit_channel_scales(self, channel) -> None:
        """Edit the scales of one channel of a series (e.g. right from the
        browser): its arrays take their scales from the channel."""
        series = channel._parent
        owner = series._container._parent
        y_row = ("Y", common_value(d.yscale.label for d in channel.data),
                 common_value(d.yscale.units for d in channel.data))
        lock_error = self._lock_error(owner, series.name)
        changes = self._ask_scales(
            scale_fields(channel.data) if channel.data else {
                "xstart": f"{channel.xscale.start:g}", "xdelta": f"{channel.xscale.delta:g}",
                "xlabel": channel.xscale.label, "xunits": channel.xscale.units,
            },
            [y_row],
            ("Channel", f"{_owner_text(owner)} › {series.name} › {channel.name}"),
            lock_error,
            x_arrays=x_array_choices(owner, channel.data), x_array=x_array_current(channel.data),
            validate_x=lambda name: x_array_error(owner, channel.data, name),
            x_array_scales=x_array_scales(owner, x_array_choices(owner, channel.data)),
        )
        if not changes or lock_error:  # the dialog only shows a locked series
            return
        self._apply_x_array(owner, series, changes, channel=channel.name)
        for key, setter in (("xstart", series.set_xstart), ("xdelta", series.set_xdelta),
                            ("xlabel", series.set_xlabel), ("xunits", series.set_xunits)):
            if key in changes:
                setter(changes[key], channel=channel.name)
        fields = changes.get("y", {}).get("Y", {})
        if "label" in fields:
            series.set_ylabel(fields["label"], channel=channel.name)
        if "units" in fields:
            series.set_yunits(fields["units"], channel=channel.name)
        # The channel's own scale too: the template for its new arrays
        for key, attribute in (("xstart", "start"), ("xdelta", "delta"),
                               ("xlabel", "label"), ("xunits", "units")):
            if key in changes:
                setattr(channel.xscale, attribute, changes[key])
        for field, value in fields.items():
            setattr(channel.yscale, field, value)
        self._changed()

    def toggle_lock(self) -> None:
        series = self._series()
        if series is not None:
            series.locked = not series.locked
        self._changed()

    def _remove_part(self, title: str, label: str, names: list[str], remove,
                     current: str | None = None) -> None:
        series, owner = self._series(), self._owner()
        if series is None or not names:
            return
        answer = self._choose(
            title, context=self._series_context(series), item_label=label,
            items=names, current=current, extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message=f"Keep the {label.lower()}'s arrays in the folder, or delete them.",
            validate=lambda item, name, arrays: self._lock_error(owner, series.name),
        )
        if answer is None:
            return
        item, _name, arrays = answer
        remove(series.name, item, delete_data=arrays == self.DELETE)
        self._changed()

    def remove_parts(self, series, label: str, names: list[str]) -> None:
        """Remove several epochs or channels of *series* (e.g. ticked in the
        browser): one dialog, keeping or deleting their arrays, as Remove
        Epoch / Remove Channel do. *label* is "Epoch" or "Channel"."""
        if not names:
            return
        owner = series._container._parent
        noun = label.lower() + ("s" if len(names) != 1 else "")
        shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        answer = self._choose(
            f"Remove {label}" + ("s" if len(names) != 1 else ""),
            context=("Series", f"{_owner_text(owner)} › {series.name}"),
            extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message=f"Remove {len(names)} {noun}: {shown}?\n"
                    "Keep their arrays in the folder, or delete them.",
            validate=lambda _item, _name, _arrays: self._lock_error(owner, series.name),
        )
        if answer is None:
            return
        remove = (owner.remove_dataseries_epoch if label == "Epoch"
                  else owner.remove_dataseries_channel)
        remove(series.name, names if len(names) > 1 else names[0],
               delete_data=answer[-1] == self.DELETE)
        self._changed()

    def remove_channel(self) -> None:
        series, owner = self._series(), self._owner()
        if series is not None:
            channel = self._selection.selection.get("channel")
            self._remove_part("Remove Channel", "Channel", list(series.channels.keys()),
                              owner.remove_dataseries_channel,
                              current=channel.name if channel is not None else None)

    REMOVE = "Remove"

    def renumber_epochs(self) -> None:
        """Number the selected series' epochs E0, E1, … in order (closing
        gaps; empty epochs can be removed first), renaming their arrays to
        match; a preview shows the changes."""
        series, owner = self._series(), self._owner()
        if series is None or not isinstance(owner, NMFolder):
            return
        has_empty = any(not e.data for e in series.epochs.values())

        def plan(choice):
            try:
                return owner.renumber_epochs_plan(series.name, choice == self.REMOVE), None
            except (ValueError, RuntimeError) as error:
                text = str(error.args[0]) if error.args else str(error)
                return ({}, {}, []), text[0].upper() + text[1:] + "."

        def message(_item=None, choice=None):
            (epochs, arrays, empty), problem = plan(choice)
            if problem:
                return problem
            parts = []
            if empty:
                parts.append(f"Removes empty epoch{'s' * (len(empty) != 1)} "
                             f"{', '.join(empty[:6])}{', …' if len(empty) > 6 else ''}.")
            if epochs:
                shown = list(epochs.items())
                changes = ", ".join(f"{old} → {new}" for old, new in shown[:6])
                if len(shown) > 6:
                    changes += f", … ({len(shown)} epochs)"
                first = next(iter(arrays.items()), None)
                example = f" (e.g. {first[0]} → {first[1]})" if first else ""
                parts.append(f"Renames {changes}, and {len(arrays)} array"
                             f"{'s' * (len(arrays) != 1)}{example}. Epoch sets and groups "
                             "follow; notes record the old names.")
            if not parts:
                kept = len(series.epochs) - len(empty)
                parts.append(f"The epochs are already numbered E0 to E{kept - 1}; "
                             "nothing to renumber." if kept else "There are no epochs.")
            return " ".join(parts)

        def error(_item, _name, choice=None):
            (epochs, _arrays, empty), problem = plan(choice)
            return problem or (None if epochs or empty else "Nothing to renumber.")

        # Empty epochs (e.g. their arrays deleted in the Data list): removed
        # first, by default, so the numbering closes over them
        extras = [("Empty epochs", [self.REMOVE, self.KEEP], self.REMOVE)] if has_empty else []
        answer = self._choose(
            "Renumber Epochs", context=self._series_context(series),
            extras=extras, message=message if has_empty else message(),
            validate=error,
        )
        if answer is None:
            return
        choice = answer[2] if has_empty else None
        owner.renumber_dataseries_epochs(series.name, remove_empty=choice == self.REMOVE)
        self._changed()

    def move_series(self) -> None:
        """Move the selected series, with its arrays, to another folder."""
        series, owner = self._series(), self._owner()
        if series is None or not isinstance(owner, NMFolder):
            return
        folders = self._selection.manager.folders
        targets = [name for name in folders.keys() if name != owner.name]
        if not targets:
            return
        suffixes = series_suffixes(series)

        arrays = [d for c in series.channels.values() for d in c.data]
        x_names = [source.name for source, _a, _o in owner._move_xarray_plan(series, arrays)]

        def error(_item, name, target):
            problem = self._lock_error(owner, series.name)
            if problem:
                return problem
            problem = prefix_error(folders[target], name, suffixes)
            if problem:
                return f"{problem}. Enter a new prefix." if name else problem
            taken = next((n for n in x_names if n in folders[target].data), None)
            if taken:  # the x array goes too, and needs its name there
                return (f"Folder '{target}' already has an array '{taken}' (the series' "
                        "x array). Rename one of them first.")
            return None

        note = owner.move_dataseries_note(series.name)
        answer = self._choose(
            "Move Series", context=self._series_context(series),
            extras=[("To folder", targets, targets[0])],
            name_label="Prefix", default_name=lambda _item, _target: series.name,
            message="The series moves with its arrays, epoch sets, groups and notes."
                    + (f" {note}" if note else ""),
            validate=error, validate_live=True,
        )
        if answer is None:
            return
        _item, name, target = answer
        try:
            moved = owner.move_dataseries(series.name, folders[target],
                                          new_prefix=name if name != series.name else None)
        except (ValueError, RuntimeError) as error:
            self._show_message("Move Series", str(error), warning=True)
            return
        # The strip follows the series to its new folder
        self._select(folder=folders[target])
        self._select(dataseries=moved)
        self._changed()

    def _append_sources(self, folder) -> list[str]:
        """Series of *folder* that can be appended to the selected series."""
        series = self._series()
        return [name for name, ds in folder.dataseries.items() if ds is not series]

    def append_series(self) -> None:
        """Add another series' epochs, from this or another folder, after this
        series' last epoch."""
        series, owner = self._series(), self._owner()
        if series is None or not isinstance(owner, NMFolder):
            return
        folders = self._selection.manager.folders
        names = [name for name, folder in folders.items() if self._append_sources(folder)]
        if not names:
            return

        def sources(folder_name):
            return self._append_sources(folders[folder_name]) if folder_name in folders else []

        def plan(folder_name, source, keep):
            return owner._append_plan(series.name, source, folders[folder_name],
                                      keep == self.DELETE)

        def summary(folder_name, source, keep):
            try:
                _ds, _src, first, new_epoch, arrays, new_names = plan(folder_name, source, keep)
            except (ValueError, RuntimeError, KeyError):
                return ""
            last = first + len(new_epoch) - 1
            epochs = f"E{first}" if last == first else f"E{first} to E{last}"
            what = "moved" if keep == self.DELETE else "copied"
            where = "" if folders[folder_name] is owner else f" in '{folder_name}'"
            note = owner.append_dataseries_note(series.name, source, folders[folder_name])
            return (f"Adds {len(new_epoch)} epoch{'s' * (len(new_epoch) != 1)} ({epochs}) "
                    f"from '{source}'{where}: {len(arrays)} arrays {what} and renamed "
                    f"(e.g. {arrays[0][2].name} → {new_names[id(arrays[0][2])]})."
                    + (f" {note}" if note else "") if arrays else "")

        def error(folder_name, _name, source, keep):
            if not source:
                return f"Folder '{folder_name}' has no data series to append."
            try:
                plan(folder_name, source, keep)
            except (ValueError, RuntimeError) as problem:
                text = str(problem.args[0]) if problem.args else str(problem)
                return text[0].upper() + text[1:]
            return None

        here = owner.name if owner.name in names else names[0]
        answer = self._choose(
            "Append Series", context=self._series_context(series),
            item_label="From folder", items=names, current=here,
            extras=[("Append", sources(here), None),
                    ("Source series", [self.KEEP, self.DELETE], self.KEEP)],
            extra_items={0: sources},
            message=summary, validate=error,
        )
        if answer is None:
            return
        folder_name, _name, source, keep = answer
        owner.append_dataseries(series.name, source, source_folder=folders[folder_name],
                                delete_source=keep == self.DELETE)
        self._changed()

    def _ask_add(self, kind, context, preview, validate, maximum=1_000_000) -> dict | None:
        """The Add Channels / Add Epochs dialog (replaced in tests)."""
        dialog = AddToSeriesDialog(kind, context, preview, validate, maximum,
                                   parent=self._parent_widget)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.values()

    def add_epochs(self) -> None:
        """Add epochs after the last one, with a new array for each channel."""
        series, owner = self._series(), self._owner()
        if series is None or owner is None or len(series.channels) == 0:
            return
        numbers = [int(e[1:]) for e in series.epochs.keys() if e[1:].isdigit()]
        first = max(numbers) + 1 if numbers else 0
        channels = list(series.channels.keys())

        def names(count: int) -> list[str]:
            return [f"{series.name}{c}{n}" for n in range(first, first + count)
                    for c in channels]

        def preview(count: int) -> str:
            last = f" to E{first + count - 1}" if count > 1 else ""
            return (f"Adds E{first}{last}: {len(names(count))} new arrays "
                    f"({array_names_text(names(count))}), sized as each channel's last.")

        def validate(values: dict) -> str | None:
            clash = next((n for n in names(values["count"]) if n in owner.data), None)
            return self._lock_error(owner, series.name) or (
                f"Data '{clash}' already exists." if clash else None)

        values = self._ask_add("epochs", self._series_context(series), preview, validate)
        if values is None:
            return
        owner.add_dataseries_epochs(series.name, values["count"],
                                    fill=fill_values(values["value"], values["noise"]))
        self._changed()

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

    def delete_series(self, series=None) -> None:
        """Delete the selected series, or *series* (e.g. one right-clicked in
        the browser); its arrays are kept in the folder, or deleted too."""
        owner = series._container._parent if series is not None else self._owner()
        answer = self._choose_series(
            "Delete Series",
            owner=owner if series is not None else None,
            current=series.name if series is not None else None,
            extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message="Keep the series' arrays in the folder, or delete them too.",
            validate=lambda item, name, arrays: self._lock_error(owner, item),
        )
        if answer is None:
            return
        item, _name, arrays = answer
        owner.remove_dataseries(item, delete_data=arrays == self.DELETE)
        self._changed()

    def delete_series_list(self, owner, names: list[str]) -> None:
        """Delete several series of *owner* (e.g. ticked in the browser),
        with the same Keep / Delete arrays choice as Delete Series."""
        if not names:
            return
        shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        answer = self._choose(
            "Delete Series",
            context=("Folder", _owner_text(owner)),
            extras=[("Arrays", [self.KEEP, self.DELETE], self.KEEP)],
            message=f"Delete {len(names)} data series: {shown}?\n"
                    "Keep their arrays in the folder, or delete them too.",
            validate=lambda _item, _name, _arrays: next(
                (error for name in names if (error := self._lock_error(owner, name))), None
            ),
        )
        if answer is None:
            return
        delete_data = answer[-1] == self.DELETE
        for name in names:
            owner.remove_dataseries(name, delete_data=delete_data)
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

    # The last epoch number is shown in To (and the first in From, if not 0)
    return [("From Epoch", Spin(low, high, value(first),
                                suffix=f"  (min {low})" if low else "")),
            ("To Epoch", Spin(low, high, value(last if last else first),
                              suffix=f"  (max {high})"))]


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
        # SetX is one of the sets: adding epochs to it leaves them out of
        # every Run, and removing them uses them again
        self.add_action = action(
            "Add to Set…", self.add_epochs,
            tip=f"Add an epoch, or a range of epochs, to a set (or to {EXCLUDE_SET}, "
                "leaving them out of every Run)")
        self.remove_action = action(
            "Remove from Set…", self.remove_epochs,
            tip=f"Remove an epoch, or a range of epochs, from a set (or from {EXCLUDE_SET})")
        self.copy_action = action("Copy…", self.copy_set,
                                  tip="Copy an epoch set to a new set")
        self.invert_action = action(
            "Invert…", self.invert_set,
            tip="Invert an epoch set in place: it holds the epochs it didn't")
        self.combine_action = action("Combine…", self.combine_sets,
                                     tip="Make a set from two sets with AND or OR")
        self.clear_action = action("Clear…", self.clear_set,
                                   tip="Remove all epochs from a set, keeping the set")
        self.rename_action = action("Renumber…", self.rename_set,
                                    tip="Give an epoch set another number")
        self.delete_action = action("Delete…", self.delete_set, tip="Delete an epoch set")
        self.browse_action = action("Show in Browser", self.choose_in_browser,
                                    tip="Show the epoch sets in the browser")
        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def epoch_set_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Epoch Set", parent)
        for item in (self.new_action, None, self.add_action, self.remove_action, None,
                     self.rename_action, self.copy_action, self.invert_action,
                     self.combine_action, self.clear_action, self.delete_action, None,
                     self.browse_action):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        return menu

    # --- helpers ------------------------------------------------------------

    def _series(self):
        if "dataseries" in self._start:  # see starting_on
            return self._start["dataseries"]
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
        for item in (self.clear_action, self.rename_action, self.delete_action,
                     self.copy_action):
            item.setEnabled(bool(sets))
        self.add_action.setEnabled(has_epochs)  # SetX is made if missing
        for item in (self.remove_action, self.invert_action):
            item.setEnabled(bool(sets) and has_epochs)
        self.combine_action.setEnabled(len(sets) > 1)

    def _series_context(self, series) -> tuple[str, str]:
        return ("Data Series", f"{self._context_text()} › {series.name}")

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
        current = self._current_name("epoch_set")
        options.setdefault("item_label", "")  # the names say it: Set0, Set1...
        return self._choose(
            title, context=self._series_context(series),
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
            extras=[set_number_spin(sets)],
            message=lambda _item, number: new_set_summary(sets, number),
            validate=lambda _item, _name, number: set_number_error(sets, number),
            validate_live=True,
        )
        if answer is None:
            return
        name = set_name(answer[-1])
        sets.add(name)
        self._select(epoch_set=name)
        self._changed()

    def _edit_range(self, title: str, verb: str, adding: bool) -> None:
        """Add or remove a range of epochs to a chosen set (SetX included)."""
        series = self._series()
        if series is None or len(series.epochs) == 0:
            return
        names = list(series.epochs.keys())
        sets = series.epochs.sets
        if adding and sets._getkey(EXCLUDE_SET) is None:
            sets.add(EXCLUDE_SET, quiet=True)  # e.g. an older series: SetX to add to

        def chosen(first, last, every):
            return epoch_range(names, first, last, int(every))

        def targets(set_name, first, last, every):
            epochs = chosen(first, last, every)
            if adding:
                return epochs
            members = set(sets.get_items(set_name, get_keys=True) or [])
            return [e for e in epochs if e in members]

        def error(set_name, _name, first, last, every):
            if range_error(names, first, last, int(every)):
                return range_error(names, first, last, int(every))
            if not adding and not targets(set_name, first, last, every):
                return f"None of these epochs are in '{set_name}'"
            return None

        options = dict(
            extras=self._range_extras(series),
            message=lambda set_name, first, last, every: (
                range_summary(verb, targets(set_name, first, last, every), int(every))[:-1]
                + f" {'to' if adding else 'from'} '{set_name}'."
                if targets(set_name, first, last, every) else ""),
            validate=error,
        )
        answer = self._choose_set(title, **options)
        if answer is None:
            return
        set_name, _name, first, last, every = answer
        epochs = [series.epochs[e] for e in targets(set_name, first, last, every)]
        if adding:
            sets.add(set_name, epochs)
        else:
            sets.remove(set_name, epochs)
        self._changed()

    def add_epochs(self) -> None:
        self._edit_range("Add Epochs to Set", "Add", adding=True)

    def remove_epochs(self) -> None:
        self._edit_range("Remove Epochs from Set", "Remove", adding=False)


    def copy_set(self) -> None:
        """Copy an epoch set's members to a new set (a list of its own)."""
        series = self._series()
        if series is None:
            return
        sets = series.epochs.sets
        answer = self._choose_set("Copy Epoch Set", **copy_set_options(sets))
        if answer is None:
            return
        self._select(epoch_set=copy_set_to(sets, answer))
        self._changed()

    def invert_set(self) -> None:
        """Invert an epoch set in place: it holds the series' epochs it didn't
        (inverting again restores it). Combined sets and SetX are refused."""
        series = self._series()
        if series is None:
            return
        sets = series.epochs.sets
        items = list(series.epochs.keys())

        def summary(name, *_):
            if set_invert_error(sets, name, "epoch"):
                return ""
            return invert_set_summary(sets, name, len(items), "epoch")

        answer = self._choose_set(
            "Invert Epoch Set", message=summary,
            validate=lambda name, _name: set_invert_error(sets, name, "epoch"),
        )
        if answer is None:
            return
        invert_in_place(sets, answer[0], items)
        self._changed()

    def combine_sets(self) -> None:
        series = self._series()
        names = self._set_names()
        if series is None or len(names) < 2:
            return
        sets = series.epochs.sets
        current = self._current_name("epoch_set")
        first = current if current in names else names[0]
        answer = self._choose_set(
            "Combine Epoch Sets", **combine_set_options(sets, names, first))
        if answer is None:
            return
        self._select(epoch_set=combine_sets_to(sets, answer))
        self._changed()

    def clear_set(self) -> None:
        answer = self._choose_set(
            "Clear Epoch Set",
            message=lambda name: clear_set_summary(self._series().epochs.sets, name, "epochs"),
        )
        if answer is None:
            return
        self._series().epochs.sets.empty(answer[0])
        self._changed()

    def rename_set(self) -> None:
        series = self._series()
        sets = series.epochs.sets if series is not None else None
        answer = self._choose_set(
            "Renumber Epoch Set",
            extras=[set_number_spin(sets) if sets is not None else ("New Set number", Spin(0, 9999, 0))],
            message=lambda item, number: (
                "" if self._exclude_set_error(item, "renumbered")
                else renumber_set_summary(sets, item, number)),
            validate=lambda item, _name, number: (
                self._exclude_set_error(item, "renumbered")
                or set_number_error(sets, number, keep=item)
            ),
            validate_live=True,
        )
        if answer is None or set_name(answer[-1]).lower() == answer[0].lower():
            return
        old, new = answer[0], set_name(answer[-1])
        was_current = old == self._selection.selection.get("epoch_set")
        sets.rename(old, new)
        if was_current:
            self._select(epoch_set=new)
        self._changed()

    def delete_set(self) -> None:
        answer = self._choose_set(
            "Delete Epoch Set",
            message=lambda name: (
                f"Delete '{name}'; epochs assigned to this set will be unaltered."),
            validate=lambda item, _name: self._exclude_set_error(item, "deleted"),
        )
        if answer is None:
            return
        if answer[0] == self._selection.selection.get("epoch_set"):
            self._select(epoch_set=None)
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

    # The Assign Cyclic preview shows the pattern: at least two full cycles,
    # between these many group numbers
    PREVIEW_MIN, PREVIEW_MAX = 16, 32

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
        self.assign_action = action("Add to Group…", self.assign_epochs,
                                    tip="Add an epoch, or a range of epochs, to a group")
        self.unassign_action = action("Remove from Group…", self.unassign_epochs,
                                      tip="Remove an epoch, or a range of epochs, from their group")
        self.clear_action = action(
            "Clear Group…", self.clear_group,
            tip="Take the epochs out of a group, or all groups (the groups are kept)")
        self.delete_action = action(
            "Delete Group…", self.delete_group,
            tip="Delete a group, or all groups; their epochs are kept, with no group")
        self.browse_action = action("Show in Browser", self.choose_in_browser,
                                    tip="Show the series' epochs in the browser")
        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def group_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Group", parent)
        for item in (self.new_action, self.cyclic_action, None, self.assign_action,
                     self.unassign_action, None, self.clear_action, self.delete_action, None,
                     self.browse_action):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        return menu

    # --- helpers ------------------------------------------------------------

    def _series(self):
        if "dataseries" in self._start:  # see starting_on
            return self._start["dataseries"]
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
        self.delete_action.setEnabled(has_groups)
        self.browse_action.setEnabled(series is not None)

    def _series_context(self, series) -> tuple[str, str]:
        return ("Data Series", f"{self._context_text()} › {series.name}")

    def _changed(self) -> None:
        series = self._series()
        group = self._selection.selection.get("group")
        numbers = series.epochs.groups.group_numbers if series is not None else []
        if series is not None and group is not None and (
            group not in numbers if group != "All Groups" else not numbers
        ):
            self._select(group=None)  # the chosen group(s) no longer exist
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
            return f"Creates {span}, with no epoch assignments."

        # The smallest number that creates a group: one past the first unused
        # number (e.g. 4 when groups 0-2 exist; 2 when only 0 and 2 do)
        first_free = next(n for n in range(len(existing) + 1) if n not in existing)
        minimum = first_free + 1
        answer = self._choose(
            "New Groups", context=self._series_context(series),
            extras=[("Number of Groups",
                     Spin(minimum, 1000, max(minimum, len(existing) + 1, 2)))],
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
            # The pattern itself (whatever the range), then where it goes
            cycle = int(groups) * int(block)
            length = min(max(self.PREVIEW_MIN, 2 * cycle), self.PREVIEW_MAX)
            pattern = " ".join(str(n) for n in cyclic_groups(length, int(groups), int(block)))
            low, high = epoch_number(epochs[0]), epoch_number(epochs[-1])
            where = f"{low}" if len(epochs) == 1 else f"{low}–{high}"
            return f"Pattern: {pattern} …\nFor epoch {where} (n={len(epochs)})."

        # Start at the groups already made (e.g. 3 after New Groups...)
        n_groups = max(len(series.epochs.groups.group_numbers), 2)
        answer = self._choose(
            "Assign Epochs to Cyclic Groups", context=self._series_context(series),
            extras=[("Number of Groups", Spin(1, 1000, n_groups)),
                    ("Block", Spin(1, 100_000, 1))]
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
        existing = series.epochs.groups.group_numbers
        numbers = existing or [0]  # none yet: group 0
        current = self._selection.selection.get("group")
        default = current if current in numbers else numbers[0]

        def chosen(first, last, every):
            return epoch_range(names, first, last, int(every))

        # A number box (its arrows step through the groups), not a popup list;
        # From / To start at the current epoch, Every at 1
        extras = self._range_extras(series, [("Group", Spin(numbers[0], numbers[-1], default))])
        answer = self._choose(
            "Add Epochs to Group", context=self._series_context(series),
            extras=extras,
            message=lambda _item, group, first, last, every: (
                range_summary("Adds", chosen(first, last, every), int(every))[:-1]
                + f" to group {group}." if chosen(first, last, every) else ""
            ),
            validate=lambda _item, _name, group, first, last, every: (
                (f"Group {group} doesn't exist; New Groups… makes it"
                 if existing and int(group) not in existing else None)
                or range_error(names, first, last, int(every))
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
            "Remove Epochs from Group", context=self._series_context(series),
            extras=self._range_extras(series),
            message=lambda _item, first, last, every: (
                range_summary("Removes", targets(first, last, every), int(every))[:-1]
                + " from their group." if targets(first, last, every) else ""),
            validate=error,
        )
        if answer is None:
            return
        _item, _name, first, last, every = answer
        groups = series.epochs.groups
        for name in targets(first, last, every):
            groups.unassign(name, quiet=True)
        self._changed()

    ALL = "All"

    def _choose_groups(self, title: str, describe) -> list[int] | None:
        """The shared Clear Group / Delete Group prompt: a group number or All,
        starting on the strip's group (or the one given with starting_on).
        *describe(numbers)* says what will happen. Returns the numbers chosen."""
        series = self._series()
        if series is None:
            return None
        groups = series.epochs.groups
        numbers = groups.group_numbers
        if not numbers:
            return None
        items = [str(n) for n in numbers] + [self.ALL]
        start = self._start.get("group", self._selection.selection.get("group"))
        current = str(start) if str(start) in items else self.ALL

        def chosen(item):
            if item == self.ALL:
                return list(groups.group_numbers)
            return [int(item)] if item.isdigit() and int(item) in groups.group_numbers else []

        answer = self._choose(
            title, context=self._series_context(series), item_label="Group",
            items=items, current=current, message=lambda item: describe(chosen(item)),
        )
        return None if answer is None else chosen(answer[0])

    def _groups_text(self, numbers: list[int]) -> tuple[str, str]:
        groups = self._series().epochs.groups
        count = sum(len(groups.get_items(n)) for n in numbers)
        which = (f"group {numbers[0]}" if len(numbers) == 1
                 else f"all {len(numbers)} groups")
        return which, f"{count} epoch{'s' * (count != 1)}"

    def clear_group(self) -> None:
        """Take the epochs out of a group, or all groups; the (empty) groups
        are kept."""
        def describe(numbers):
            if not numbers:
                return ""
            which, epochs = self._groups_text(numbers)
            return (f"The {epochs} of {which} will be removed from it; the groups and "
                    "epochs are kept.") if len(numbers) == 1 else (
                f"The {epochs} of {which} will be removed from them; the groups and "
                "epochs are kept.")

        numbers = self._choose_groups("Clear Epoch Group", describe)
        if not numbers:
            return
        groups = self._series().epochs.groups
        for number in numbers:
            unassign_from_group(groups, groups.get_items(number), number)
        self._changed()

    def delete_group(self) -> None:
        """Delete a group, or all groups; their epochs are kept, with no group."""
        def describe(numbers):
            if not numbers:
                return ""
            which, epochs = self._groups_text(numbers)
            return (f"{which[0].upper()}{which[1:]} will be deleted; its {epochs} are kept, "
                    "with no group." if len(numbers) == 1 else
                    f"{which[0].upper()}{which[1:]} will be deleted; their {epochs} are "
                    "kept, with no group.")

        numbers = self._choose_groups("Delete Epoch Group", describe)
        if not numbers:
            return
        groups = self._series().epochs.groups
        strip_group = self._selection.selection.get("group")
        if strip_group in numbers or len(numbers) == len(groups.group_numbers):
            self._select(group=None)  # the strip's group goes
        for number in numbers:
            groups.delete_group(number)
        self._changed()

    def choose_in_browser(self) -> None:
        series = self._series()
        if series is not None:
            self.browse_requested.emit(series)


class ChannelActions(TierActions):
    """Actions for the current data series' channel (Add and Delete may
    follow).

    Filters and transforms don't change the arrays, so they can be edited on
    a locked series too.
    """

    # Ask the app to show the channel in the browser
    browse_requested = QtCore.pyqtSignal(object)
    # Ask the app to open Filter & Transform for the channel
    transforms_requested = QtCore.pyqtSignal(object)

    def __init__(
        self,
        selection_model: SelectionModel,
        parent: QtWidgets.QWidget | None = None,
        remove_action: QtGui.QAction | None = None,
    ) -> None:
        super().__init__(selection_model, parent)
        action = self._action
        # Remove Channel… is the data series' action (shared with Remove
        # Epochs and the browser), shown in this menu
        self.remove_action = remove_action
        self.add_action = action(
            "Add Channels…", self.add_channels,
            tip="Add channels after the last one: a new array for each epoch")
        self.transform_action = action(
            MENU_TEXT, self.edit_transforms,
            tip="The channel's filters and transforms, as plotted and analysed",
        )
        self.browse_action = action("Show in Browser", self.choose_in_browser,
                                    tip="Show the channel in the browser")
        selection_model.selection_changed.connect(self.update_enabled)
        self.update_enabled()

    def channel_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Channel", parent)
        menu.addAction(self.add_action)
        if self.remove_action is not None:
            menu.addAction(self.remove_action)
        menu.addSeparator()
        menu.addAction(self.transform_action)
        menu.addSeparator()
        menu.addAction(self.browse_action)
        return menu

    def _channel(self):
        """The selected channel of the selected data series, or None."""
        series = self._selection.selection.get("dataseries")
        channel = self._selection.selection.get("channel")
        if series is None or channel is None:
            return None
        return channel if any(c is channel for c in series.channels.values()) else None

    def update_enabled(self, *_args) -> None:
        has_channel = self._channel() is not None
        self.transform_action.setEnabled(has_channel)
        self.browse_action.setEnabled(has_channel)
        # Adding channels changes the series: not while it is locked
        series = self._selection.selection.get("dataseries")
        locked = series is not None and series.locked
        self.add_action.setEnabled(series is not None and not locked)
        self.add_action.setToolTip(
            f"Data series '{series.name}' is locked." if locked
            else "Add channels after the last one: a new array for each epoch")

    def _ask_add(self, kind, context, preview, validate, maximum=1_000_000) -> dict | None:
        """The Add Channels dialog (replaced in tests)."""
        dialog = AddToSeriesDialog(kind, context, preview, validate, maximum,
                                   parent=self._parent_widget)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.values()

    def add_channels(self) -> None:
        """Add channels after the last one, with a new array for each epoch."""
        series = self._selection.selection.get("dataseries")
        owner = self._context()
        if series is None or owner is None or len(series.epochs) == 0:
            return
        letters = [c for c in series.channels.keys() if c in nmu.CHANNEL_CHARS]
        first = max(nmu.CHANNEL_CHARS.index(c) for c in letters) + 1 if letters else 0
        room = len(nmu.CHANNEL_CHARS) - first
        epochs = [e[1:] for e in series.epochs.keys()]

        def new_letters(count: int) -> list[str]:
            return [nmu.channel_char(i) for i in range(first, first + count)]

        def names(count: int) -> list[str]:
            return [f"{series.name}{c}{e}" for c in new_letters(count) for e in epochs]

        def preview(count: int) -> str:
            if count > room:
                return ""
            letters_text = ", ".join(new_letters(count))
            return (f"Adds channel{'s' * (count > 1)} {letters_text}: {len(names(count))} "
                    f"new arrays ({array_names_text(names(count))}), sized as each epoch's.")

        def validate(values: dict) -> str | None:
            if series.locked:
                return f"Data series '{series.name}' is locked. Untick Lock (Prefix ⋯ menu) to change it."
            clash = next((n for n in names(values["count"]) if n in owner.data), None)
            return f"Data '{clash}' already exists." if clash else None

        if room < 1:
            self._show_message("Add Channels", "The series already has channels A to Z.")
            return
        context = ("Series", f"{self._context_text()} › {series.name}")
        values = self._ask_add("channels", context, preview, validate, maximum=room)
        if values is None:
            return
        added = owner.add_dataseries_channels(
            series.name, values["count"], fill=fill_values(values["value"], values["noise"]),
            y_label=values["ylabel"], y_units=values["yunits"])
        self._select(channel=added[0])  # shown in the Plot and Table
        self._changed()

    def edit_transforms(self) -> None:
        channel = self._channel()
        if channel is not None:
            self.transforms_requested.emit(channel)

    def choose_in_browser(self) -> None:
        channel = self._channel()
        if channel is not None:
            self.browse_requested.emit(channel)


__all__ = [
    "DIALOG_WIDTH",
    "set_dialog_width",
    "AddToSeriesDialog",
    "ChannelActions",
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
    "next_set_number",
    "set_name",
    "set_name_error",
    "set_number_error",
    "set_number_of",
    "unique_name",
]
