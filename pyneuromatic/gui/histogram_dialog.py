"""The Histogram... dialog (Stats2's ⋯ menu, #492): an ST_ array's bins and
output, for ``nm.stats.histogram``.

Bins are Auto (numpy's "auto" width), a bin width, or a number of bins;
without an x range their edges are multiples of the width (round numbers),
with empty bins at each end. The output is the counts, the probability
(heights add up to 1) or the probability density (areas add up to 1).
"""
from __future__ import annotations

import numpy as np
from PyQt6 import QtWidgets

import pyneuromatic.core.nm_math as nm_math

OUTPUT_LABELS = {
    "count": "Count",
    "probability": "Probability",
    "density": "Probability density",
}


def _text(value: float) -> str:
    return f"{value:.6g}"


class HistogramDialog(QtWidgets.QDialog):
    """Choose a histogram's bins and output for ST_ array *name*.

    *previous* (the dialog's last :meth:`settings`) is shown again, so e.g.
    two arrays can be made with the same bin width and compared bin by bin.
    """

    MIN_BINS = 2

    def __init__(
        self,
        name: str,
        values: np.ndarray,
        units: str = "",
        previous: dict | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        finite = values[np.isfinite(values)].astype(float)
        auto = nm_math.histogram_round_edges(finite, pad=0)
        self._auto_width = auto["width"] or 1.0
        unit_text = f" {units}" if units else ""
        previous = previous or {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Histogram of {name}", self))

        bins_group = QtWidgets.QGroupBox("Bins", self)
        bins_form = QtWidgets.QGridLayout(bins_group)
        self.auto_radio = QtWidgets.QRadioButton("Auto", bins_group)
        self.auto_radio.setToolTip(
            "numpy's \"auto\" width: the larger of Freedman–Diaconis (from the "
            "values' spread) and Sturges (for small sets)")
        bins_form.addWidget(self.auto_radio, 0, 0)
        bins_form.addWidget(QtWidgets.QLabel(
            f"width {_text(self._auto_width)}{unit_text}", bins_group), 0, 1)
        self.width_radio = QtWidgets.QRadioButton("Bin width", bins_group)
        bins_form.addWidget(self.width_radio, 1, 0)
        self.width_edit = QtWidgets.QLineEdit(
            _text(previous.get("binwidth", self._auto_width)), bins_group)
        bins_form.addWidget(self.width_edit, 1, 1)
        bins_form.addWidget(QtWidgets.QLabel(units, bins_group), 1, 2)
        self.number_radio = QtWidgets.QRadioButton("Number of bins", bins_group)
        bins_form.addWidget(self.number_radio, 2, 0)
        self.number_box = QtWidgets.QSpinBox(bins_group)
        self.number_box.setRange(self.MIN_BINS, 100000)  # one bin isn't a histogram
        self.number_box.setValue(int(previous.get("nbins", len(auto["counts"]))))
        bins_form.addWidget(self.number_box, 2, 1)
        mode = previous.get("mode", "auto")
        {"auto": self.auto_radio, "width": self.width_radio,
         "number": self.number_radio}.get(mode, self.auto_radio).setChecked(True)
        layout.addWidget(bins_group)

        form = QtWidgets.QFormLayout()
        range_row = QtWidgets.QHBoxLayout()
        self.range_checkbox = QtWidgets.QCheckBox("from", self)
        self.range_checkbox.setToolTip(
            "Bins over this range, as given (edges not rounded; no empty bins); "
            "values outside it are left out")
        self.range_checkbox.setChecked(bool(previous.get("xrange")))
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        lo, hi = previous.get("xrange") or (lo, hi)
        self.min_edit = QtWidgets.QLineEdit(_text(lo), self)
        self.max_edit = QtWidgets.QLineEdit(_text(hi), self)
        range_row.addWidget(self.range_checkbox)
        range_row.addWidget(self.min_edit)
        range_row.addWidget(QtWidgets.QLabel("to", self))
        range_row.addWidget(self.max_edit)
        range_row.addWidget(QtWidgets.QLabel(units, self))
        form.addRow("X range", range_row)
        self.pad_box = QtWidgets.QSpinBox(self)
        self.pad_box.setRange(0, 100)
        self.pad_box.setValue(int(previous.get("pad", 1)))
        self.pad_box.setToolTip("So the histogram starts and ends at zero (0: none)")
        form.addRow("Empty bins at each end", self.pad_box)
        self.output_combo = QtWidgets.QComboBox(self)
        for key, label in OUTPUT_LABELS.items():
            self.output_combo.addItem(label, key)
        self.output_combo.setCurrentIndex(
            max(0, self.output_combo.findData(previous.get("output", "count"))))
        self.output_combo.setToolTip(
            "Probability: count ÷ number of values (heights add up to 1). "
            "Probability density: count ÷ (number of values × bin width) "
            "(areas add up to 1)")
        form.addRow("Output", self.output_combo)
        layout.addLayout(form)

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

        for radio in (self.auto_radio, self.width_radio, self.number_radio):
            radio.toggled.connect(self._update_enabled)
        self.range_checkbox.toggled.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self, *_args) -> None:
        self.width_edit.setEnabled(self.width_radio.isChecked())
        self.number_box.setEnabled(self.number_radio.isChecked())
        ranged = self.range_checkbox.isChecked()
        self.min_edit.setEnabled(ranged)
        self.max_edit.setEnabled(ranged)
        self.pad_box.setEnabled(not ranged)  # a range is used as given

    def settings(self) -> dict:
        """What's chosen, to show again next time (see *previous*)."""
        settings = {"mode": ("width" if self.width_radio.isChecked()
                             else "number" if self.number_radio.isChecked() else "auto"),
                    "nbins": self.number_box.value(),
                    "pad": self.pad_box.value(),
                    "output": self.output_combo.currentData()}
        try:
            settings["binwidth"] = float(self.width_edit.text())
        except ValueError:
            pass
        if self.range_checkbox.isChecked():
            try:
                settings["xrange"] = (float(self.min_edit.text()), float(self.max_edit.text()))
            except ValueError:
                pass
        return settings

    def kwargs(self) -> dict:
        """``nm.stats.histogram`` arguments for what's chosen (bins or
        binwidth, xrange, pad, output). Raises ValueError if they're bad."""
        kwargs: dict = {}
        if self.width_radio.isChecked():
            try:
                width = float(self.width_edit.text())
            except ValueError:
                raise ValueError("The bin width isn't a number.") from None
            if not width > 0:
                raise ValueError("The bin width must be more than 0.")
            kwargs["binwidth"] = width
        elif self.number_radio.isChecked():
            # The text typed, not the box's last good value (e.g. "0" isn't 2)
            if not self.number_box.hasAcceptableInput():
                raise ValueError(f"The number of bins must be at least {self.MIN_BINS}.")
            kwargs["bins"] = self.number_box.value()
        if self.range_checkbox.isChecked():
            try:
                lo, hi = float(self.min_edit.text()), float(self.max_edit.text())
            except ValueError:
                raise ValueError("The x range isn't two numbers.") from None
            if not hi > lo:
                raise ValueError("The x range's max must be more than its min.")
            kwargs["xrange"] = (lo, hi)
        else:
            if not self.pad_box.hasAcceptableInput():
                raise ValueError("Empty bins at each end must be 0 or more.")
            kwargs["pad"] = self.pad_box.value()
        kwargs["output"] = self.output_combo.currentData()
        return kwargs

    def accept(self) -> None:
        try:
            self.kwargs()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        super().accept()


__all__ = ["HistogramDialog", "OUTPUT_LABELS"]
