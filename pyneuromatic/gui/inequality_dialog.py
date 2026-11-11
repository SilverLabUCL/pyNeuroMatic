"""The Inequality... dialog (Stats2's ⋯ menu, #499): a condition on an ST_
array's values (e.g. y > 10, 5 ≤ y ≤ 20), for ``nm.stats.inequality``.

Each value gives 1 (true) or 0 (false), or the value or NaN; optionally the
epochs that pass and fail become Epoch Sets of the data series.
"""
from __future__ import annotations

from PyQt6 import QtWidgets

import pyneuromatic.core.nm_utilities as nmu

# Core's operator -> how it's shown
OPERATORS = {
    ">": "y > a",
    ">=": "y ≥ a",
    "<": "y < a",
    "<=": "y ≤ a",
    "==": "y = a",
    "!=": "y ≠ a",
    "<<": "a < y < b",
    "<=<=": "a ≤ y ≤ b",
    "<=<": "a ≤ y < b",
    "<<=": "a < y ≤ b",
}
RANGE_OPERATORS = ("<<", "<=<=", "<=<", "<<=")


class InequalityDialog(QtWidgets.QDialog):
    """Choose a condition, its output and Epoch Sets for ST_ array *name*.

    *set_names*: the next free Epoch Set names (passes, fails), or None if
    the subfolder's epochs can't be found (no Epoch Sets: greyed out).
    *previous* (the dialog's last :meth:`settings`) is shown again.
    """

    def __init__(
        self,
        name: str,
        set_names: tuple[str, str] | None = None,
        previous: dict | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inequality")
        previous = previous or {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Inequality of {name}", self))

        form = QtWidgets.QFormLayout()
        self.op_combo = QtWidgets.QComboBox(self)
        for op, label in OPERATORS.items():
            self.op_combo.addItem(label, op)
        self.op_combo.setCurrentIndex(max(0, self.op_combo.findData(previous.get("op", ">"))))
        form.addRow("Condition", self.op_combo)
        self.a_edit = QtWidgets.QLineEdit(previous.get("a", ""), self)
        form.addRow("a", self.a_edit)
        self.b_label = QtWidgets.QLabel("b", self)
        self.b_edit = QtWidgets.QLineEdit(previous.get("b", ""), self)
        form.addRow(self.b_label, self.b_edit)
        self.output_combo = QtWidgets.QComboBox(self)
        self.output_combo.addItem("1 or 0", True)
        self.output_combo.addItem("Value or NaN", False)
        self.output_combo.setCurrentIndex(0 if previous.get("binary_output", True) else 1)
        self.output_combo.setToolTip(
            "1 where the condition is true, 0 where it's false; or the value where "
            "it's true, NaN where it's false. A row without a value is NaN either way")
        form.addRow("Output", self.output_combo)
        layout.addLayout(form)

        sets_group = QtWidgets.QGroupBox(self)  # untitled: its checkboxes say "Epoch Set"
        sets_form = QtWidgets.QGridLayout(sets_group)
        passes, fails = set_names or ("", "")
        self.pass_checkbox = QtWidgets.QCheckBox("Epoch Set of passes:", sets_group)
        self.pass_edit = QtWidgets.QLineEdit(passes, sets_group)
        self.fail_checkbox = QtWidgets.QCheckBox("Epoch Set of fails:", sets_group)
        self.fail_edit = QtWidgets.QLineEdit(fails, sets_group)
        for row, (box, edit) in enumerate(((self.pass_checkbox, self.pass_edit),
                                           (self.fail_checkbox, self.fail_edit))):
            sets_form.addWidget(box, row, 0)
            sets_form.addWidget(edit, row, 1)
        self.pass_checkbox.setChecked(bool(previous.get("pass_set")))
        self.fail_checkbox.setChecked(bool(previous.get("fail_set")))
        sets_group.setEnabled(set_names is not None)
        sets_group.setToolTip(
            "The epochs that pass or fail, as Epoch Sets of the data series (rows "
            "without a value are in neither)" if set_names is not None else
            "Only for results of a data series run (with epoch numbers)")
        layout.addWidget(sets_group)
        self._sets_usable = set_names is not None

        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.op_combo.currentIndexChanged.connect(self._update_enabled)
        self.pass_checkbox.toggled.connect(self._update_enabled)
        self.fail_checkbox.toggled.connect(self._update_enabled)
        self._update_enabled()

    def is_range(self) -> bool:
        return self.op_combo.currentData() in RANGE_OPERATORS

    def _update_enabled(self, *_args) -> None:
        self.b_label.setVisible(self.is_range())  # b: a range's upper bound only
        self.b_edit.setVisible(self.is_range())
        self.pass_edit.setEnabled(self.pass_checkbox.isChecked())
        self.fail_edit.setEnabled(self.fail_checkbox.isChecked())

    def settings(self) -> dict:
        """What's chosen, to show again next time (see *previous*); not the
        set names (the next free ones are offered)."""
        return {"op": self.op_combo.currentData(), "a": self.a_edit.text(),
                "b": self.b_edit.text(), "binary_output": self.output_combo.currentData(),
                "pass_set": self.pass_checkbox.isChecked(),
                "fail_set": self.fail_checkbox.isChecked()}

    def kwargs(self) -> dict:
        """``nm.stats.inequality`` arguments for what's chosen (op, a, b,
        binary_output, set_name_success, set_name_failure; the data series
        is the caller's). Raises ValueError if they're bad."""
        try:
            a = float(self.a_edit.text())
        except ValueError:
            raise ValueError("a isn't a number.") from None
        kwargs: dict = {"op": self.op_combo.currentData(), "a": a}
        if self.is_range():
            try:
                b = float(self.b_edit.text())
            except ValueError:
                raise ValueError("b isn't a number.") from None
            if not b > a:
                raise ValueError("b must be more than a.")
            kwargs["b"] = b
        kwargs["binary_output"] = bool(self.output_combo.currentData())
        names = []
        for key, box, edit in (("set_name_success", self.pass_checkbox, self.pass_edit),
                               ("set_name_failure", self.fail_checkbox, self.fail_edit)):
            if not (self._sets_usable and box.isChecked()):
                continue
            text = edit.text().strip()
            if not nmu.name_ok(text):
                raise ValueError(f"'{text}' isn't a valid Epoch Set name "
                                 "(use letters, digits and _).")
            names.append(text.lower())
            kwargs[key] = text
        if len(names) == 2 and names[0] == names[1]:
            raise ValueError("The passes and fails need different Epoch Sets.")
        return kwargs

    def accept(self) -> None:
        try:
            self.kwargs()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        super().accept()


__all__ = ["InequalityDialog", "OPERATORS", "RANGE_OPERATORS"]
