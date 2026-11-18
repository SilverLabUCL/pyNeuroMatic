"""Spike2's analysis dialogs (#530), from its Array ⋯ menu: PST...,
ISI... and Waveforms..., for ``nm.spike.pst``, ``nm.spike.isi`` and
``nm.spike.extract_spike_waveforms``.

Each shows its last settings again (see *previous*), so e.g. two runs'
PSTs can be made with the same bins and compared.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6 import QtWidgets

import pyneuromatic.core.nm_math as nm_math


def _text(value: float) -> str:
    return f"{value:.6g}"


def _span(values: np.ndarray) -> tuple[float, float]:
    """The finite values' (min, max); (0, 1) if none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)


def nice_width(values: np.ndarray) -> float:
    """A bin width for *values*: numpy's "auto" width (as Stats2's
    Histogram), rounded to the nearest 1, 2 or 5 × 10ⁿ (e.g. 6 → 5, 8 →
    10); 1 if it can't be found."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    try:
        width = nm_math.histogram_round_edges(values, pad=0)["width"]
    except ValueError:
        width = None
    if not width or not math.isfinite(width) or width <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(width))
    candidates = [step * power for step in (1, 2, 5, 10)]
    nearest = min(candidates, key=lambda c: abs(math.log(c / width)))
    return float(f"{nearest:.12g}")


def _number(edit: QtWidgets.QLineEdit, what: str) -> float:
    try:
        return float(edit.text())
    except ValueError:
        raise ValueError(f"{what} isn't a number.") from None


def parse_spikes(text: str) -> tuple[int, int | None] | None:
    """A Spikes range typed, as ``extract_spike_waveforms``' *spikes*:
    "" or "all" -> None (every spike); "0-99" -> (0, 99); "100-" -> (100,
    None) (to the end); "5" -> (5, 5). ValueError if it isn't one."""
    text = text.strip().lower()
    if text in ("", "all"):
        return None
    first, dash, last = (part.strip() for part in text.partition("-"))
    try:
        start = int(first)
        end = (None if not last else int(last)) if dash else start
    except ValueError:
        raise ValueError(f"Spikes: {text!r} isn't a range, e.g. 0-99 or 100-.") from None
    if start < 0 or (end is not None and end < start):
        raise ValueError(f"Spikes: {text!r} (the first can't be after the last).")
    return start, end


def spikes_text(spikes: tuple[int, int | None] | None) -> str:
    """*spikes* as typed: None -> "", (0, 99) -> "0-99", (100, None) ->
    "100-", (5, 5) -> "5"."""
    if spikes is None:
        return ""
    first, last = spikes
    if last is None:
        return f"{first}-"
    return str(first) if first == last else f"{first}-{last}"


class _AnalysisDialog(QtWidgets.QDialog):
    """A form, an error line and OK / Cancel; OK checks :meth:`kwargs`."""

    def __init__(self, title: str, heading: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(QtWidgets.QLabel(heading, self))
        self.form = QtWidgets.QFormLayout()
        self._layout.addLayout(self.form)

    def _finish(self) -> None:
        """The error line and buttons, after the subclass's rows."""
        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._layout.addWidget(buttons)

    def _range_row(self, label: str, tip: str, values: tuple[float, float],
                   on: bool, units: str) -> tuple:
        """A "from [lo] to [hi] units" row with a checkbox: (checkbox, lo, hi)."""
        row = QtWidgets.QHBoxLayout()
        checkbox = QtWidgets.QCheckBox("from", self)
        checkbox.setToolTip(tip)
        checkbox.setChecked(on)
        lo = QtWidgets.QLineEdit(_text(values[0]), self)
        hi = QtWidgets.QLineEdit(_text(values[1]), self)
        for widget in (checkbox, lo, QtWidgets.QLabel("to", self), hi,
                       QtWidgets.QLabel(units, self)):
            row.addWidget(widget)
        self.form.addRow(label, row)

        def enable(checked: bool) -> None:
            lo.setEnabled(checked)
            hi.setEnabled(checked)

        checkbox.toggled.connect(enable)
        enable(on)
        return checkbox, lo, hi

    @staticmethod
    def _range(checkbox, lo, hi, what: str) -> tuple[float, float] | None:
        """The range if its checkbox is on (else None); ValueError if bad."""
        if not checkbox.isChecked():
            return None
        low, high = _number(lo, f"{what}'s min"), _number(hi, f"{what}'s max")
        if not high > low:
            raise ValueError(f"{what}'s max must be more than its min.")
        return low, high

    def _width_edit(self, values: np.ndarray, units: str, previous: dict,
                    tip: str) -> QtWidgets.QLineEdit:
        """The bin width (its last, else a round one for *values*)."""
        row = QtWidgets.QHBoxLayout()
        edit = QtWidgets.QLineEdit(
            _text(previous.get("binwidth", nice_width(values))), self)
        edit.setToolTip(tip)
        row.addWidget(edit)
        row.addWidget(QtWidgets.QLabel(units, self))
        self.form.addRow("Bin width", row)
        return edit

    def _output_combo(self, labels: dict[str, str], tip: str,
                      previous: dict) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(self)
        for key, label in labels.items():
            combo.addItem(label, key)
        combo.setCurrentIndex(max(0, combo.findData(previous.get("output_mode", "count"))))
        combo.setToolTip(tip)
        self.form.addRow("Output", combo)
        return combo

    def _binwidth(self) -> float:
        width = _number(self.width_edit, "The bin width")
        if not (width > 0 and math.isfinite(width)):
            raise ValueError("The bin width must be more than 0.")
        return width

    def kwargs(self) -> dict:
        """The analysis's arguments; ValueError if they're bad."""
        raise NotImplementedError

    def settings(self) -> dict:
        """What's chosen, to show again next time: :meth:`kwargs` and which
        ranges are on, as far as they're good."""
        try:
            return self.kwargs()
        except ValueError:
            return {}

    def accept(self) -> None:
        try:
            self.kwargs()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        super().accept()


class PSTDialog(_AnalysisDialog):
    """PST...: a peri-stimulus time histogram of every spike time, pooled
    over the epochs (``nm.spike.pst``), saved as H_SP_PST."""

    OUTPUT_LABELS = {"count": "Count", "rate": "Rate",
                     "probability": "Probability"}

    def __init__(self, name: str, times: np.ndarray, units: str = "",
                 previous: dict | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("PST", f"Peri-stimulus time histogram of {name}", parent)
        previous = previous or {}
        self.width_edit = self._width_edit(
            times, units, previous,
            "Bins of this width, from a multiple of it (or the x range's start), "
            "so the same width gives the same bins for any run")
        times = _span(times)
        on = "xbgn" in previous
        self.range_checkbox, self.xbgn_edit, self.xend_edit = self._range_row(
            "X range", "Bins over this range of times (otherwise the first to "
            "last spike); spikes outside it are left out",
            (previous.get("xbgn", times[0]), previous.get("xend", times[1])), on, units)
        self.output_combo = self._output_combo(
            self.OUTPUT_LABELS,
            "Count: spikes per bin. Rate: count ÷ (epochs × bin width). "
            "Probability: count ÷ epochs.", previous)
        self._finish()

    def kwargs(self) -> dict:
        kwargs: dict = {"binwidth": self._binwidth()}
        xrange = self._range(self.range_checkbox, self.xbgn_edit, self.xend_edit,
                             "The x range")
        if xrange is not None:
            kwargs["xbgn"], kwargs["xend"] = xrange
        kwargs["output_mode"] = self.output_combo.currentData()
        return kwargs


class ISIDialog(_AnalysisDialog):
    """ISI...: a histogram of the interspike intervals within each epoch,
    pooled (``nm.spike.isi``), saved as H_SP_ISI."""

    OUTPUT_LABELS = {"count": "Count", "probability": "Probability"}

    def __init__(self, name: str, times: np.ndarray, intervals: np.ndarray,
                 units: str = "", previous: dict | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("ISI", f"Interspike interval histogram of {name}", parent)
        previous = previous or {}
        self.width_edit = self._width_edit(
            intervals, units, previous,
            "Bins of this width, from a multiple of it (or the ISI range's start)")
        times, intervals = _span(times), _span(intervals)
        self.times_checkbox, self.xbgn_edit, self.xend_edit = self._range_row(
            "Spike times", "Only the spikes in this range of times (their "
            "intervals between them)",
            (previous.get("xbgn", times[0]), previous.get("xend", times[1])),
            "xbgn" in previous, units)
        self.isi_checkbox, self.min_edit, self.max_edit = self._range_row(
            "ISI range", "Bins over this range of intervals (otherwise the "
            "shortest to longest); e.g. leave out refractory artefacts",
            (previous.get("min_isi", intervals[0]), previous.get("max_isi", intervals[1])),
            "min_isi" in previous, units)
        self.output_combo = self._output_combo(
            self.OUTPUT_LABELS,
            "Count: intervals per bin. Probability: count ÷ intervals "
            "(heights add up to 1).", previous)
        self._finish()

    def kwargs(self) -> dict:
        kwargs: dict = {"binwidth": self._binwidth()}
        times = self._range(self.times_checkbox, self.xbgn_edit, self.xend_edit,
                            "The spike times")
        if times is not None:
            kwargs["xbgn"], kwargs["xend"] = times
        isis = self._range(self.isi_checkbox, self.min_edit, self.max_edit, "The ISI range")
        if isis is not None:
            kwargs["min_isi"], kwargs["max_isi"] = isis
        kwargs["output_mode"] = self.output_combo.currentData()
        return kwargs


class WaveformsDialog(_AnalysisDialog):
    """Waveforms...: each spike's waveform, a window of its recording
    around it (``nm.spike.extract_spike_waveforms``), saved in the Spike
    subfolder as SPK_ arrays (the SPK_ data series)."""

    EDGE_LABELS = {"skip": "Skip the spike", "pad": "Pad with NaN"}
    ALIGN_LABELS = {"spike": "Spike at 0", "zero": "Start at 0",
                    "source": "As recorded"}

    # A window's default, in the spike times' units (e.g. 1 ms before, 2 after)
    DEFAULT_WINDOW = {"s": (0.001, 0.002)}

    def __init__(self, name: str, times: np.ndarray, units: str = "",
                 previous: dict | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Waveforms", f"Copy spike waveforms from {name}", parent)
        previous = previous or {}
        values = np.asarray(times, dtype=float)
        self._times = values[np.isfinite(values)]  # every spike's, to count them
        times = _span(times)
        pre, post = self.DEFAULT_WINDOW.get(units, (1.0, 2.0))
        window_row = QtWidgets.QHBoxLayout()
        self.pre_edit = QtWidgets.QLineEdit(_text(previous.get("pre", pre)), self)
        self.post_edit = QtWidgets.QLineEdit(_text(previous.get("post", post)), self)
        for widget in (self.pre_edit, QtWidgets.QLabel("before,", self),
                       self.post_edit, QtWidgets.QLabel(f"{units} after".strip(), self)):
            window_row.addWidget(widget)
        self.pre_edit.setToolTip("Time before each spike's threshold crossing")
        self.post_edit.setToolTip("Time after each spike's threshold crossing")
        self.form.addRow("Window", window_row)
        self.times_checkbox, self.xbgn_edit, self.xend_edit = self._range_row(
            "Spike times", "Only the spikes in this range of times",
            (previous.get("xbgn", times[0]), previous.get("xend", times[1])),
            "xbgn" in previous, units)
        self.spikes_edit = QtWidgets.QLineEdit(spikes_text(previous.get("spikes")), self)
        self.spikes_edit.setPlaceholderText("all")
        self.spikes_edit.setToolTip(
            "Which spikes to include, "
            "e.g. 0-99 (first 100), 100- (from "
            "101st on), or 5; blank: all")
        spikes_row = QtWidgets.QHBoxLayout()
        spikes_row.addWidget(self.spikes_edit)
        # The numbers there are, e.g. "0-1920 (1921 spikes)": within Spike times
        self.spikes_label = QtWidgets.QLabel("", self)
        spikes_row.addWidget(self.spikes_label)
        self.form.addRow("Spikes", spikes_row)
        self.times_checkbox.toggled.connect(self._count_spikes)
        self.xbgn_edit.textChanged.connect(self._count_spikes)
        self.xend_edit.textChanged.connect(self._count_spikes)
        self._count_spikes()
        self.align_combo = QtWidgets.QComboBox(self)
        for key, label in self.ALIGN_LABELS.items():
            self.align_combo.addItem(label, key)
        self.align_combo.setCurrentIndex(
            max(0, self.align_combo.findData(previous.get("align", "spike"))))
        self.align_combo.setToolTip(
            "Each waveform's x: 0 at its spike (−before to +after), starting at "
            "0, or its times in the recording")
        self.form.addRow("X", self.align_combo)
        self.edge_combo = QtWidgets.QComboBox(self)
        for key, label in self.EDGE_LABELS.items():
            self.edge_combo.addItem(label, key)
        self.edge_combo.setCurrentIndex(
            max(0, self.edge_combo.findData(previous.get("edge", "skip"))))
        self.edge_combo.setToolTip(
            "A spike whose window runs past the recording's start or end: left "
            "out, or its waveform padded with NaN")
        self.form.addRow("At an edge", self.edge_combo)
        self.clip_checkbox = QtWidgets.QCheckBox("Clip at the next spike", self)
        self.clip_checkbox.setChecked(bool(previous.get("clip_to_next_spike", False)))
        self.clip_checkbox.setToolTip(
            "NaN after the next spike's crossing, so a waveform doesn't show the next")
        self.form.addRow("", self.clip_checkbox)
        self.group_checkbox = QtWidgets.QCheckBox("Assign to Groups by Epochs", self)
        self.group_checkbox.setChecked(bool(previous.get("group_by_epoch", False)))
        self.group_checkbox.setToolTip(
            "Assign waveforms to Groups based on Epoch numbers "
            "(E0 spikes → G0, E5 spikes → G5...), to view/analyse an "
            "epoch's spikes together")
        self.form.addRow("", self.group_checkbox)
        self._finish()

    def spike_count(self) -> int:
        """How many spikes there are to number: those within Spike times if
        it's on (and good), else all."""
        try:
            times = self._range(self.times_checkbox, self.xbgn_edit, self.xend_edit,
                                "The spike times")
        except ValueError:
            times = None  # still being typed: all of them
        if times is None:
            return int(self._times.size)
        lo, hi = times
        return int(np.count_nonzero((self._times >= lo) & (self._times <= hi)))

    def _count_spikes(self, *_args) -> None:
        count = self.spike_count()
        noun = "spike" if count == 1 else "spikes"
        span = f"0-{count - 1} " if count > 1 else ("0 " if count == 1 else "")
        self.spikes_label.setText(f"{span}({count} {noun})")

    def kwargs(self) -> dict:
        pre = _number(self.pre_edit, "The time before")
        post = _number(self.post_edit, "The time after")
        if not (pre > 0 and post > 0):
            raise ValueError("The times before and after must be more than 0.")
        kwargs: dict = {"pre": pre, "post": post}
        times = self._range(self.times_checkbox, self.xbgn_edit, self.xend_edit,
                            "The spike times")
        if times is not None:
            kwargs["xbgn"], kwargs["xend"] = times
        spikes = parse_spikes(self.spikes_edit.text())
        if spikes is not None:
            kwargs["spikes"] = spikes
        kwargs["clip_to_next_spike"] = self.clip_checkbox.isChecked()
        if self.group_checkbox.isChecked():
            kwargs["group_by_epoch"] = True
        kwargs["edge"] = self.edge_combo.currentData()
        kwargs["align"] = self.align_combo.currentData()
        return kwargs


__all__ = ["ISIDialog", "PSTDialog", "WaveformsDialog", "parse_spikes", "spikes_text"]
