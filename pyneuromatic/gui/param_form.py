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

import numpy as np

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


class XScale:
    """The arrays' x scale, to show a number of points in x (as_x): from
    start and delta, or an x array's values."""

    def __init__(self, start: float = 0.0, delta: float = 1.0, units: str = "",
                 xarray: Any = None, points: int | None = None) -> None:
        self.start, self.delta, self.units = float(start), float(delta), units or ""
        self.xarray = xarray
        self.points = points  # the array's number of points, if known

    @classmethod
    def of(cls, data: Any) -> "XScale":
        """*data*'s (an NMData): its x array if it fits, else start and delta."""
        values, xarray = data.nparray, data.xarray
        fits = (xarray is not None and not data.xarray_problem()
                and hasattr(values, "size") and xarray.size == values.size)
        scale = data.xscale
        points = int(values.size) if hasattr(values, "size") else None
        return cls(scale.start, scale.delta, scale.units, xarray if fits else None,
                   points)

    def shown(self) -> bool:
        """Points in x say more than the points: not plain point numbers."""
        return (self.xarray is not None or bool(self.units)
                or (self.start, self.delta) != (0.0, 1.0))

    def x(self, points: int, as_x: str) -> float | None:
        """A number of *points* ("length") or a point number ("position") in
        x; None where it has none (e.g. a length, with an uneven x array)."""
        if self.xarray is not None:
            if as_x == "position" and 0 <= points < self.xarray.size:
                return float(self.xarray[points])
            return None
        if as_x == "length":
            return points * self.delta
        return self.start + points * self.delta

    def index(self, x: float) -> int:
        """The point number nearest *x* (at least 0)."""
        if self.xarray is not None:
            return int(np.argmin(np.abs(np.asarray(self.xarray, dtype=float) - x)))
        if self.delta == 0:
            return 0
        return max(int(round((x - self.start) / self.delta)), 0)


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
        # A number of points in x too (as_x), by the arrays' x scale
        self._x_labels: dict[str, QtWidgets.QLabel] = {}
        self._x_scale: XScale | None = None
        self._notes: dict[str, QtWidgets.QLabel] = {}  # set_note
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
        if param.as_x:  # e.g. "(= 5 ms)", as it's typed
            x_label = QtWidgets.QLabel("", row)
            self._x_labels[param.name] = x_label
            layout.addWidget(x_label)
            edits[0].textChanged.connect(lambda _t, p=param: self._show_x(p))
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

    # --- points in x (as_x) ------------------------------------------------

    def set_x_scale(self, scale: XScale | None) -> None:
        """The arrays' x scale, to show points in x (None: not shown)."""
        self._x_scale = scale
        for param in self.params:
            if param.name in self._x_labels:
                self._show_x(param)

    def int_value(self, name: str) -> int | None:
        """Whole-number parameter *name* as typed, or None if it isn't one."""
        try:
            return int(float(self._controls[name].text().strip()))
        except ValueError:
            return None

    def set_note(self, name: str, text: str) -> None:
        """A note at the end of parameter *name*'s row (e.g. Redimension's
        "currently 200 points"); "" removes it."""
        label = self._notes.get(name)
        if label is None:
            if not text:
                return
            row = self._rows[name][-1]
            label = QtWidgets.QLabel("", row)
            layout = row.layout()
            layout.insertWidget(layout.count() - 1, label)  # before its stretch
            self._notes[name] = label
        label.setText(text)
        label.setVisible(bool(text))

    def note(self, name: str) -> str:
        label = self._notes.get(name)
        return label.text() if label is not None else ""

    def x_text(self, name: str) -> str:
        """Parameter *name* in x, as shown (e.g. "(= 5 ms)"), or ""."""
        return self._x_labels[name].text()

    def _show_x(self, param: NMParam) -> None:
        scale, text = self._x_scale, ""
        points = self.int_value(param.name)
        if scale is not None and scale.shown() and points is not None:
            x = scale.x(points, param.as_x)
            if x is not None:
                units = f" {scale.units}" if scale.units else ""
                text = (f"(= {x:.6g}{units})" if param.as_x == "length"
                        else f"(x = {x:.6g}{units})")
        self._x_labels[param.name].setText(text)

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
