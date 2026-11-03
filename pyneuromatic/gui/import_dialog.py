"""The import options prompt: what to import from a data file (#456).

Shown by Open / Import before each file is read (Settings › General › "Ask
for import options"). Its answers are ``NMFolderContainer.import_file``
options; only those changed from the defaults are returned, so the
recorded command stays short.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object_container import unique_name
from pyneuromatic.io.base import FileInfo


def parse_epochs(text: str, available: list[int]) -> list[int] | range | None:
    """Epoch numbers from *text*, e.g. "0-9, 12, 15-18"; None for all ("" or
    "all", or every epoch in *available*). A run of numbers is a range.

    Raises:
        ValueError: If *text* isn't numbers and ranges, or names epochs
            not in *available*.
    """
    text = text.strip()
    if text.lower() in ("", "all"):
        return None
    numbers: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        first, _dash, last = item.partition("-")
        try:
            begin = int(first)
            end = int(last) if last.strip() else begin
        except ValueError:
            raise ValueError(f"epochs: {item!r} is not a number or range (e.g. 0-9)")
        if end < begin:
            raise ValueError(f"epochs: {item!r} ends before it begins")
        numbers.update(range(begin, end + 1))
    missing = sorted(numbers - set(available))
    if missing:
        raise ValueError("epochs not in the file: %s" % ", ".join(map(str, missing[:5])))
    if not numbers:
        raise ValueError("epochs: choose at least one")
    chosen = sorted(numbers)
    if chosen == sorted(available):
        return None
    if chosen == list(range(chosen[0], chosen[-1] + 1)):
        return range(chosen[0], chosen[-1] + 1)
    return chosen


def _epochs_text(epochs: list[int]) -> str:
    """e.g. "0-18", or "0, 2, 5"."""
    if epochs and epochs == list(range(epochs[0], epochs[-1] + 1)):
        return f"{epochs[0]}-{epochs[-1]}" if len(epochs) > 1 else str(epochs[0])
    return ", ".join(map(str, epochs))


def info_summary(info: FileInfo) -> str:
    """e.g. "2 channels · 19 epochs (0-18) · 60000 points × 0.02 ms = 1200 ms"."""
    n_channels, n_epochs = len(info.channels), len(info.epochs)
    duration = info.n_points * info.x_delta
    return (f"{n_channels} channel{'s' * (n_channels != 1)} · "
            f"{n_epochs} epoch{'s' * (n_epochs != 1)} ({_epochs_text(info.epochs)}) · "
            f"{info.n_points} points × {info.x_delta:g} {info.x_units} = "
            f"{duration:g} {info.x_units}")


class ImportOptionsDialog(QtWidgets.QDialog):
    """Folder name, prefix, channels, epochs, X range and data series for
    one file; Import or Cancel."""

    def __init__(self, path: str | Path, info: FileInfo, folders,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        path = Path(path)
        self.setWindowTitle(f"Import {path.name}")
        self._info = info
        self._folders = folders
        self._default_name = unique_name(folders, path.stem)

        layout = QtWidgets.QVBoxLayout(self)
        suffix = path.suffix.lower()
        self.format_label = QtWidgets.QLabel(
            f"{info.file_format} ({suffix})" if info.file_format else suffix, self)
        layout.addWidget(self.format_label)
        self.summary_label = QtWidgets.QLabel(info_summary(info), self)
        layout.addWidget(self.summary_label)

        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self._default_name, self)
        form.addRow("Folder name", self.name_edit)
        self.prefix_edit = QtWidgets.QLineEdit(info.prefix, self)
        self.prefix_edit.setToolTip("Names of the imported arrays, e.g. RecordA0")
        form.addRow("Prefix", self.prefix_edit)

        channel_box = QtWidgets.QWidget(self)
        channel_layout = QtWidgets.QVBoxLayout(channel_box)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for letter, label, units in info.channels:
            text = letter + (f"  {label}" if label else "") + (f" ({units})" if units else "")
            box = QtWidgets.QCheckBox(text, channel_box)
            box.setChecked(True)
            channel_layout.addWidget(box)
            self.channel_checkboxes[letter] = box
        form.addRow("Channels", channel_box)

        self.epochs_edit = QtWidgets.QLineEdit(self)
        self.epochs_edit.setPlaceholderText(f"all ({_epochs_text(info.epochs)})")
        self.epochs_edit.setToolTip("Numbers and ranges, e.g. 0-9, 12, 15-18; "
                                    "blank for all. Imported epochs start again at E0.")
        form.addRow("Epochs", self.epochs_edit)

        x_row = QtWidgets.QHBoxLayout()
        last_x = info.x_start + (info.n_points - 1) * info.x_delta
        self.xbgn_edit = QtWidgets.QLineEdit(self)
        self.xbgn_edit.setPlaceholderText(f"{info.x_start:g}")
        self.xend_edit = QtWidgets.QLineEdit(self)
        self.xend_edit.setPlaceholderText(f"{last_x:g}")
        x_row.addWidget(self.xbgn_edit)
        x_row.addWidget(QtWidgets.QLabel("to", self))
        x_row.addWidget(self.xend_edit)
        x_row.addWidget(QtWidgets.QLabel(info.x_units, self))
        form.addRow("X range", x_row)

        self.dataseries_checkbox = QtWidgets.QCheckBox("Make data series", self)
        self.dataseries_checkbox.setChecked(True)
        form.addRow("", self.dataseries_checkbox)
        layout.addLayout(form)

        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setStyleSheet("color: #c62828;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QtWidgets.QDialogButtonBox(self)
        self.import_button = buttons.addButton(
            "Import", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict:
        """The ``import_file`` options changed from the defaults.

        Raises:
            ValueError: If an entry isn't valid (the message says which).
        """
        options: dict = {}
        name = self.name_edit.text().strip()
        if name != self._default_name:
            if not name or not nmu.name_ok(name):
                raise ValueError(f"Folder name: {name!r} is not a valid name")
            if name in self._folders:
                raise ValueError(f"Folder name: {name} already exists")
            options["name"] = name
        prefix = self.prefix_edit.text().strip()
        if prefix != self._info.prefix:
            if not prefix or not nmu.name_ok(prefix):
                raise ValueError(f"Prefix: {prefix!r} is not a valid name")
            options["prefix"] = prefix
        chosen = [letter for letter, box in self.channel_checkboxes.items() if box.isChecked()]
        if not chosen:
            raise ValueError("Channels: choose at least one")
        if len(chosen) < len(self.channel_checkboxes):
            options["channels"] = chosen
        epochs = parse_epochs(self.epochs_edit.text(), self._info.epochs)
        if epochs is not None:
            options["epochs"] = epochs
        xrange = self._xrange()
        if xrange is not None:
            options["xrange"] = xrange
        if not self.dataseries_checkbox.isChecked():
            options["make_dataseries"] = False
        return options

    def _xrange(self) -> tuple[float | None, float | None] | None:
        values = []
        for label, edit in (("X begin", self.xbgn_edit), ("X end", self.xend_edit)):
            text = edit.text().strip()
            try:
                values.append(float(text) if text else None)
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number")
        xbgn, xend = values
        if xbgn is None and xend is None:
            return None
        if xbgn is not None and xend is not None and not xbgn < xend:
            raise ValueError("X range: begin must be less than end")
        return xbgn, xend

    def accept(self) -> None:
        try:
            self.options()
        except ValueError as error:
            self.error_label.setText(str(error))
            return  # stays open, so the entry can be fixed
        super().accept()


__all__ = ["ImportOptionsDialog", "info_summary", "parse_epochs"]
