"""A form of a function's parameters, built from their descriptions (#555).

From :class:`~pyneuromatic.core.nm_params.NMParam` descriptions, a control
each: a number box, a checkbox, a dropdown (a choice, or an array to pick),
or an x range's two boxes. :meth:`ParamForm.values` reads them back, checked;
:meth:`ParamForm.x_ranges` gives the x ranges to draw on the Plot, and
:meth:`ParamForm.set_x_range` follows a dragged one.

Used by the Main tool's functions; Event and Fit next.
"""
from __future__ import annotations

import math
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_params import (
    ARRAY, BOOLEAN, CHOICE, COLOR, INTEGER, NUMBER, TEXT, XRANGE, NMParam)
from pyneuromatic.gui.tool_tab import fit_number_edit


def bound_text(value: float) -> str:
    """A number as its box shows it: "-inf", "inf", or up to 10 digits."""
    if value == -math.inf:
        return "-inf"
    if value == math.inf:
        return "inf"
    return f"{value:.10g}"


class ParamForm(QtWidgets.QWidget):
    """The controls of *params*, a row each (a checkbox spans its row)."""

    # A value was edited (typed and finished, ticked, or chosen)
    changed = QtCore.pyqtSignal()
    # An x range was edited: the Plot's lines follow
    ranges_changed = QtCore.pyqtSignal()

    def __init__(self, params: tuple[NMParam, ...],
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.params = tuple(params)
        self._controls: dict[str, Any] = {}  # name: widget (an x range: two edits)
        self._colors: dict[str, str] = {}  # a color parameter's value
        self._rows: dict[str, list[QtWidgets.QWidget]] = {}  # enabled with "when"
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        # At the left, whatever its width (macOS centres a form's rows)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft
                              | QtCore.Qt.AlignmentFlag.AlignTop)
        for param in self.params:
            control = self._control(param)
            if param.kind == BOOLEAN:
                form.addRow(control)
                self._rows[param.name] = [control]
            else:
                label = QtWidgets.QLabel(param.label, self)
                label.setToolTip(param.tip)
                form.addRow(label, control)
                self._rows[param.name] = [label, control]
        self.changed.connect(self._enable)
        if not self.params:
            form.addRow(QtWidgets.QLabel("No settings", self))
        self.set_values(self.defaults())

    # --- the controls ---------------------------------------------------------

    def _enable(self) -> None:
        """A parameter shown with *when*: enabled only while the other has
        its value (e.g. a color only with "One color")."""
        current = {}
        for param in self.params:
            control = self._controls[param.name]
            if param.kind == CHOICE:
                current[param.name] = control.currentData()
            elif param.kind == BOOLEAN:
                current[param.name] = control.isChecked()
        for param in self.params:
            if param.when:
                for widget in self._rows[param.name]:
                    widget.setEnabled(param.enabled(current))

    def _control(self, param: NMParam) -> QtWidgets.QWidget:
        if param.kind == BOOLEAN:
            box = QtWidgets.QCheckBox(param.label, self)
            box.setToolTip(param.tip)
            box.toggled.connect(lambda _on: self.changed.emit())
            self._controls[param.name] = box
            return box
        if param.kind == COLOR:  # a swatch: click to choose
            button = QtWidgets.QToolButton(self)
            button.setToolTip(param.tip or "Choose the color")
            button.setFixedSize(28, 22)
            button.clicked.connect(lambda _c=False, p=param: self._choose_color(p))
            self._controls[param.name] = button
            self._show_color(param.name, param.default or "#000000")
            return button
        if param.kind == TEXT:
            edit = QtWidgets.QLineEdit(self)
            edit.setToolTip(param.tip)
            edit.setMaximumWidth(120)
            edit.editingFinished.connect(self.changed)
            self._controls[param.name] = edit
            return edit
        if param.kind in (CHOICE, ARRAY):
            combo = QtWidgets.QComboBox(self)
            combo.setToolTip(param.tip)
            for value, shown in param.choices:
                combo.addItem(str(shown), value)
            combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
            self._controls[param.name] = combo
            return combo
        row = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edits = []
        for i in range(2 if param.kind == XRANGE else 1):
            if i:
                layout.addWidget(QtWidgets.QLabel("to", row))
            edit = QtWidgets.QLineEdit(row)
            fit_number_edit(edit)
            edit.setToolTip(param.tip)
            edit.editingFinished.connect(self.changed)
            if param.kind == XRANGE:
                edit.editingFinished.connect(self.ranges_changed)
            layout.addWidget(edit)
            edits.append(edit)
        if param.units:
            layout.addWidget(QtWidgets.QLabel(param.units, row))
        layout.addStretch(1)
        self._controls[param.name] = edits if param.kind == XRANGE else edits[0]
        return row

    def _show_color(self, name: str, color: str) -> None:
        self._colors[name] = QtGui.QColor(color).name()
        self._controls[name].setStyleSheet(
            f"QToolButton {{ background: {self._colors[name]}; border: 1px solid #888; "
            "border-radius: 3px; }")

    def _choose_color(self, param: NMParam) -> None:
        color = self._ask_color(QtGui.QColor(self._colors[param.name]))
        if color is not None:
            self._show_color(param.name, color.name())
            self.changed.emit()

    def _ask_color(self, current: QtGui.QColor) -> QtGui.QColor | None:
        """The color chosen (None: cancelled); replaced in tests."""
        color = QtWidgets.QColorDialog.getColor(current, self, "Color")
        return color if color.isValid() else None

    def control(self, name: str) -> Any:
        """Parameter *name*'s control (an x range: its two boxes)."""
        return self._controls[name]

    def set_array_names(self, name: str, names: list[str]) -> None:
        """The arrays an array parameter can pick, keeping its choice."""
        combo = self._controls[name]
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for array in names:
            combo.addItem(array, array)
        index = combo.findData(current)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    # --- values -----------------------------------------------------------------

    def defaults(self) -> dict[str, Any]:
        """Each parameter's default (an array to pick has none: kept)."""
        return {p.name: p.default for p in self.params if p.default is not None}

    def reset(self) -> None:
        """Every control back to its default; the Plot's lines follow."""
        self.set_values(self.defaults())
        self.changed.emit()
        if any(p.kind == XRANGE for p in self.params):
            self.ranges_changed.emit()

    def values(self) -> dict[str, Any]:
        """Each parameter's value, checked (an x range: a (begin, end)
        pair); raises ValueError, naming the parameter, if one isn't valid."""
        values = {}
        for param in self.params:
            control = self._controls[param.name]
            if param.kind == BOOLEAN:
                raw = control.isChecked()
            elif param.kind in (CHOICE, ARRAY):
                raw = control.currentData()
            elif param.kind == XRANGE:
                raw = tuple(edit.text().strip() or default for edit, default
                            in zip(control, ("-inf", "inf")))
            elif param.kind == COLOR:
                raw = self._colors[param.name]
            else:
                raw = control.text()
            values[param.name] = param.check(raw)
        return values

    def set_values(self, values: dict[str, Any]) -> None:
        """Show *values* (by parameter name), without :attr:`changed`."""
        for param in self.params:
            if param.name not in values:
                continue
            value, control = values[param.name], self._controls[param.name]
            if param.kind == BOOLEAN:
                control.blockSignals(True)
                control.setChecked(bool(value))
                control.blockSignals(False)
            elif param.kind in (CHOICE, ARRAY):
                control.blockSignals(True)
                control.setCurrentIndex(max(control.findData(value), 0))
                control.blockSignals(False)
            elif param.kind == XRANGE:
                for edit, bound in zip(control, value):
                    edit.setText(bound_text(float(bound)))
            elif param.kind == COLOR:
                self._show_color(param.name, value)
            elif param.kind in (NUMBER, INTEGER):
                control.setText(bound_text(float(value)) if param.kind == NUMBER
                                else str(int(value)))
            elif param.kind == TEXT:
                control.setText(str(value or ""))
        self._enable()

    # --- x ranges, on the Plot -------------------------------------------------

    def x_ranges(self) -> list[tuple[NMParam, tuple[float, float]]]:
        """The x ranges as typed, to draw (one that isn't valid is left out)."""
        ranges = []
        for param in self.params:
            if param.kind != XRANGE:
                continue
            begin, end = self._controls[param.name]
            try:
                ranges.append((param, param.check((begin.text().strip() or "-inf",
                                                   end.text().strip() or "inf"))))
            except ValueError:
                pass
        return ranges

    def set_x_range(self, name: str, begin: float, end: float) -> None:
        """An x range dragged on the Plot: its boxes follow."""
        edits = self._controls.get(name)
        if not isinstance(edits, list):
            return
        edits[0].setText(bound_text(begin))
        edits[1].setText(bound_text(end))
