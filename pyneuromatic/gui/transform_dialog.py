"""The Filter & Transform window: a channel's filters and transforms.

Steps are applied in order to the channel as plotted and analysed (e.g. by
Stats); the data arrays are never changed. Edits take effect as they are
made (:attr:`TransformDialog.changed`); Clear removes every step, and Undo
puts back the steps the window opened with.
"""
from __future__ import annotations

import html
import math

from PyQt6 import QtCore, QtGui, QtWidgets

import pyneuromatic.core.nm_math as nm_math
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_transform import (
    NMTransform,
    NMTransformBaseline,
    NMTransformDFOF,
    NMTransformDifferentiate,
    NMTransformDoubleDifferentiate,
    NMTransformFilter,
    NMTransformIntegrate,
    NMTransformInvert,
    NMTransformLn,
    NMTransformLog,
    NMTransformNormalize,
    NMTransformSmooth,
)

# The menu item and button that open this window ("&&" shows "&": a single
# "&" would mark a keyboard shortcut)
MENU_TEXT = "Filter && Transform..."

# Marks a channel shown with filters or transforms (tabs, strip, trace name),
# in red: one or more steps are on
MARK = "ƒ"
MARK_COLOR = "#c62828"
MARK_STYLE = f"color: {MARK_COLOR}; font-weight: bold;"  # the same everywhere


def mark_html(text: str) -> str:
    """*text* followed by a red ƒ, for a rich-text label."""
    return f'{html.escape(text)}&nbsp;&nbsp;<span style="{MARK_STYLE}">{MARK}</span>'

# The Add menu, in order (None: a separator)
STEP_TYPES: list[tuple[str, type[NMTransform]] | None] = [
    ("Filter", NMTransformFilter),
    ("Smooth", NMTransformSmooth),
    ("Baseline", NMTransformBaseline),
    ("Normalize", NMTransformNormalize),
    ("dF/F", NMTransformDFOF),
    None,
    ("Invert", NMTransformInvert),
    ("Differentiate", NMTransformDifferentiate),
    ("Double Differentiate", NMTransformDoubleDifferentiate),
    ("Integrate", NMTransformIntegrate),
    ("Log10", NMTransformLog),
    ("Ln", NMTransformLn),
]
_NAMES = {cls: name for name, cls in (t for t in STEP_TYPES if t is not None)}

# Settings with a fixed list of values
CHOICES = {
    "filter_type": ["butterworth", "bessel", "notch"],
    "btype": ["low", "high", "bandpass"],
    "method": ["boxcar", "binomial", "savgol"],
    "min_fxn": ["mean", "min", "mean@min"],
    "max_fxn": ["mean", "max", "mean@max"],
}

LABELS = {
    "filter_type": "Type",
    "cutoff": "Cutoff (Hz)",
    "sample_rate": "Sample rate (Hz)",
    "order": "Order",
    "btype": "Band",
    "q": "Q",
    "method": "Method",
    "window": "Window (points)",
    "passes": "Passes",
    "polyorder": "Polynomial order",
    "xbgn": "x begin",
    "xend": "x end",
    "min_xbgn": "Low: x begin",
    "min_xend": "Low: x end",
    "min_fxn": "Low: function",
    "min_mean_n": "Low: points",
    "max_xbgn": "High: x begin",
    "max_xend": "High: x end",
    "max_fxn": "High: function",
    "max_mean_n": "High: points",
    "norm_min": "Normalized low",
    "norm_max": "Normalized high",
}

# Settings not in the form: means over x windows (Baseline, dF/F) always
# ignore NaNs here, so one NaN doesn't turn the whole trace NaN
HIDDEN = {"ignore_nans"}

# Spin box (minimum, step) of whole-number settings; a smoothing window is
# odd, so its arrows step by 2
SPIN = {
    "window": (3, 2),
    "passes": (1, 1),
    "polyorder": (1, 1),
    "order": (1, 1),
    "min_mean_n": (1, 1),
    "max_mean_n": (1, 1),
}

TOOLTIPS = {  # (Cutoff's follows the filter type: cutoff_tip)
    "min_mean_n": "Points averaged around the minimum (Low: function mean@min)",
    "max_mean_n": "Points averaged around the maximum (High: function mean@max)",
    "window": "Odd number of points (3, 5, 7, ...)",
    "sample_rate": "Blank: from the data's x scale",
}


def step_name(t: NMTransform) -> str:
    """e.g. "Filter"."""
    return _NAMES.get(type(t), t.type_str)


def step_text(t: NMTransform) -> str:
    """A step's name and main settings, e.g. "Filter: butterworth low, 1000 Hz"."""
    name = step_name(t)
    if isinstance(t, NMTransformFilter):
        band = "" if t.filter_type == "notch" else f" {t.btype}"
        return f"{name}: {t.filter_type}{band}, {_text(t.cutoff)} Hz"
    if isinstance(t, NMTransformSmooth):
        size = f"{t.passes} passes" if t.method == "binomial" else f"{t.window} points"
        return f"{name}: {t.method}, {size}"
    if isinstance(t, (NMTransformBaseline, NMTransformDFOF)):
        return f"{name}: x {_text(t.xbgn)} to {_text(t.xend)}"
    if isinstance(t, NMTransformNormalize):
        return f"{name}: {_text(t.norm_min)} to {_text(t.norm_max)}"
    return name


def steps_tooltip(channel: NMChannel | None) -> str:
    """The steps that are on, one per line ("" if none)."""
    if channel is None or not channel.transform_on:
        return ""
    steps = [step_text(t) for t in channel.transform if t.on]
    return "Filter & Transform:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def applies(t: NMTransform, key: str) -> bool:
    """False for settings the step's current type or method doesn't use."""
    if isinstance(t, NMTransformFilter):
        notch = t.filter_type == "notch"
        if key == "q":
            return notch
        if key in ("order", "btype"):
            return not notch
    if isinstance(t, NMTransformSmooth):
        uses = {"boxcar": ("window", "passes"), "binomial": ("passes",),
                "savgol": ("window", "polyorder")}[t.method]
        if key in ("window", "passes", "polyorder"):
            return key in uses
    if isinstance(t, NMTransformNormalize):  # points around the min / max
        if key == "min_mean_n":
            return t.min_fxn == "mean@min"
        if key == "max_mean_n":
            return t.max_fxn == "mean@max"
    return True


DEFAULT_CUTOFF = 1000.0  # Hz, if below the Nyquist frequency
NOTCH_CUTOFF = 50.0  # Hz, mains


def start_cutoff(rate: float | None, preferred: float = DEFAULT_CUTOFF) -> float:
    """A filter's first cutoff (Hz): *preferred* if it is below the Nyquist
    frequency (half the sample *rate*), else a round number well below it
    (a fifth of Nyquist, rounded down to 1, 2 or 5 x 10^n)."""
    if rate is None or rate <= 0:
        return float(preferred)
    nyquist = rate / 2
    if preferred < nyquist:
        return float(preferred)
    return nm_math.round_down_125(nyquist / 5)


def cutoff_tip(t: NMTransformFilter, rate: float | None) -> str:
    """The Cutoff box's tooltip, for the filter's type and band (examples
    below the Nyquist frequency of *rate*)."""
    if t.filter_type == "notch":
        return "The frequency to remove, in Hz (e.g. 50 for mains)"
    if t.btype == "bandpass":
        low, high = nm_math.bandpass_example(rate)
        return f"Low, high, in Hz (e.g. {low:g}, {high:g})"
    edge = "below" if t.btype == "low" else "above"
    return f"One frequency, in Hz: the filter passes frequencies {edge} it"


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    if isinstance(value, float):
        return "inf" if value == math.inf else "-inf" if value == -math.inf else f"{value:g}"
    return str(value)


def _parse(key: str, text: str):
    """A line edit's text as a setting's value."""
    text = text.strip()
    if key == "sample_rate" and not text:
        return None
    if key == "cutoff" and "," in text:
        return [float(part) for part in text.split(",")]
    return float(text)  # also "inf", "-inf"


class TransformDialog(QtWidgets.QDialog):
    """Edit one channel's filters and transforms (stays open while you work)."""

    changed = QtCore.pyqtSignal()  # the channel's steps changed

    def __init__(self, channel: NMChannel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filter & Transform")
        self.setModal(False)
        self._channel: NMChannel | None = None
        self._opened_with: list[dict] = []
        self._editors: dict[str, QtWidgets.QWidget] = {}
        # Cutoffs the window chose (by step), changed with the filter type
        # until a value is typed
        self._auto_cutoffs: dict[int, float] = {}
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel("", self)
        self.title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title_label)
        note = QtWidgets.QLabel(
            "Steps are applied in order to the channel as plotted and analysed "
            "(e.g. by Stats). The data arrays are not changed.", self)
        note.setWordWrap(True)
        layout.addWidget(note)

        body = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        self.step_list = QtWidgets.QListWidget(self)
        self.step_list.setToolTip("Tick a step to turn it on or off")
        self.step_list.currentRowChanged.connect(self._show_settings)
        self.step_list.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.step_list, stretch=1)
        buttons = QtWidgets.QHBoxLayout()
        self.add_menu = QtWidgets.QMenu("Add", self)
        self.add_actions: dict[str, QtGui.QAction] = {}
        for entry in STEP_TYPES:
            if entry is None:
                self.add_menu.addSeparator()
                continue
            name, cls = entry
            action = self.add_menu.addAction(name)
            action.triggered.connect(lambda _=False, c=cls: self.add_step(self._new_step(c)))
            self.add_actions[name] = action
        self.add_button = QtWidgets.QToolButton(self)
        self.add_button.setText("Add")
        self.add_button.setMenu(self.add_menu)
        self.add_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        buttons.addWidget(self.add_button)
        self.remove_button = QtWidgets.QPushButton("Remove", self)
        self.remove_button.clicked.connect(self.remove_step)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        left.addLayout(buttons)
        body.addLayout(left, stretch=1)

        self.settings_box = QtWidgets.QGroupBox("Settings", self)
        self.settings_form = QtWidgets.QFormLayout(self.settings_box)
        body.addWidget(self.settings_box, stretch=1)
        layout.addLayout(body, stretch=1)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #c62828;")
        layout.addWidget(self.status_label)

        bottom = QtWidgets.QDialogButtonBox(self)
        self.clear_button = bottom.addButton(
            "Clear", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        self.clear_button.setToolTip("Remove every step")
        self.clear_button.clicked.connect(self.clear)
        self.undo_button = bottom.addButton(
            "Undo", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        self.undo_button.setToolTip("Put back the steps the window opened with")
        self.undo_button.clicked.connect(self.undo_changes)
        self.close_button = bottom.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(bottom)

        # Return finishes typing a value; it must not press a button (e.g.
        # Remove, the first push button, is otherwise the dialog's default)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        self.resize(560, 340)
        self.set_channel(channel)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt)
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            return  # the editor has taken the value (editingFinished)
        super().keyPressEvent(event)

    @property
    def channel(self) -> NMChannel | None:
        return self._channel

    def set_channel(self, channel: NMChannel | None, reopened: bool = False) -> None:
        """Edit *channel*'s steps (e.g. when the plot shows another channel).

        Undo goes back to its steps now; also when *reopened* (the
        window opens again on the same channel).
        """
        if channel is self._channel and not reopened:
            return
        self._channel = channel
        self._opened_with = [t.to_dict() for t in channel.transform] if channel else []
        self._refresh_list(0)

    def set_status(self, text: str) -> None:
        """A problem applying the steps (e.g. to the plotted trace), or ""."""
        self.status_label.setText(text)

    # --- the list of steps ------------------------------------------------

    def _steps(self) -> list[NMTransform]:
        return self._channel.transform if self._channel is not None else []

    def _refresh_list(self, row: int | None = None) -> None:
        channel = self._channel
        if channel is None:
            self.title_label.setText("No channel")
        else:
            series = channel._parent
            prefix = f"{series.name}  " if series is not None else ""
            self.title_label.setText(f"{prefix}channel {channel.name}")
        self._updating = True
        current = self.step_list.currentRow() if row is None else row
        self.step_list.clear()
        for t in self._steps():
            item = QtWidgets.QListWidgetItem(step_text(t))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if t.on
                               else QtCore.Qt.CheckState.Unchecked)
            self.step_list.addItem(item)
        self._updating = False
        count = self.step_list.count()
        self.step_list.setCurrentRow(min(max(current, 0), count - 1) if count else -1)
        self._show_settings(self.step_list.currentRow())
        self._update_buttons()

    def _update_buttons(self) -> None:
        row, count = self.step_list.currentRow(), self.step_list.count()
        has_channel = self._channel is not None
        self.add_button.setEnabled(has_channel)
        self.remove_button.setEnabled(row >= 0)
        self.clear_button.setEnabled(count > 0)
        self.undo_button.setEnabled(
            has_channel and [t.to_dict() for t in self._steps()] != self._opened_with
        )

    def _set_steps(self, steps: list[NMTransform], row: int) -> None:
        self._channel.transform = steps  # recorded in the history
        self._refresh_list(row)
        self.changed.emit()

    def add_step(self, step: NMTransform) -> None:
        """Add *step* after the current one."""
        if self._channel is None:
            return
        steps = list(self._steps())
        row = self.step_list.currentRow() + 1 if steps else 0
        steps.insert(row, step)
        self._set_steps(steps, row)

    def remove_step(self) -> None:
        row = self.step_list.currentRow()
        if row < 0:
            return
        steps = list(self._steps())
        del steps[row]
        self._set_steps(steps, row)

    def clear(self) -> None:
        """Remove every step (Undo puts them back)."""
        if self._channel is not None and self._steps():
            self._set_steps([], 0)

    def undo_changes(self) -> None:
        if self._channel is not None:
            self._channel.transform = [dict(d) for d in self._opened_with]
            self._refresh_list()
            self.changed.emit()

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._updating:
            return
        t = self._steps()[self.step_list.row(item)]
        t.on = item.checkState() == QtCore.Qt.CheckState.Checked
        self._update_buttons()
        self.changed.emit()

    # --- the current step's settings --------------------------------------

    def _show_settings(self, row: int) -> None:
        while self.settings_form.rowCount():
            self.settings_form.removeRow(0)
        self._editors = {}
        steps = self._steps()
        self._update_buttons()
        if not 0 <= row < len(steps):
            self.settings_box.setTitle("Settings")
            return
        t = steps[row]
        self.settings_box.setTitle(f"{step_name(t)} Settings")
        values = {k: v for k, v in t.to_dict().items()
                  if k not in ("type", "on") and k not in HIDDEN}
        if not values:
            self.settings_form.addRow(QtWidgets.QLabel("No settings", self.settings_box))
        for key, value in values.items():
            editor = self._make_editor(t, key, value)
            editor.setEnabled(applies(t, key))
            if key in TOOLTIPS:
                editor.setToolTip(TOOLTIPS[key])
            self.settings_form.addRow(LABELS.get(key, key), editor)
            self._editors[key] = editor
        self._update_cutoff_tip(t)

    def _make_editor(self, t: NMTransform, key: str, value) -> QtWidgets.QWidget:
        box = self.settings_box
        if key in CHOICES:
            editor = QtWidgets.QComboBox(box)
            editor.addItems(CHOICES[key])
            editor.setCurrentText(str(value))
            editor.currentTextChanged.connect(
                lambda text, k=key: self._set_value(t, k, text))
        elif isinstance(value, bool):
            editor = QtWidgets.QCheckBox(box)
            editor.setChecked(value)
            editor.toggled.connect(lambda checked, k=key: self._set_value(t, k, checked))
        elif isinstance(value, int):
            editor = QtWidgets.QSpinBox(box)
            minimum, step = SPIN.get(key, (0, 1))
            editor.setRange(minimum, 999_999)
            editor.setSingleStep(step)
            editor.setValue(value)
            editor.valueChanged.connect(lambda number, k=key: self._set_value(t, k, number))
        else:  # floats, a cutoff list, an optional sample rate
            editor = QtWidgets.QLineEdit(_text(value), box)
            if key == "sample_rate":  # blank: the rate from the x scale
                editor.setPlaceholderText(self._xscale_rate_text(t))
            editor.editingFinished.connect(
                lambda k=key, e=editor: self._set_text(t, k, e))
        return editor

    def _update_cutoff_tip(self, t: NMTransform) -> None:
        editor = self._editors.get("cutoff")
        if editor is not None and isinstance(t, NMTransformFilter):
            editor.setToolTip(cutoff_tip(t, t.sample_rate or self._xscale_rate()))

    def _xscale_rate(self) -> float | None:
        """The sample rate (Hz) from the x scale of the channel's first array."""
        arrays = self._channel.data if self._channel is not None else []
        try:
            return NMTransformFilter()._resolve_sample_rate(arrays[0].xscale)
        except (IndexError, ValueError, TypeError, KeyError):
            return None

    def _xscale_rate_text(self, t: NMTransformFilter) -> str:
        """e.g. "20000 (x scale)", in Hz."""
        rate = self._xscale_rate()
        return "from x scale" if rate is None else f"{rate:g} (x scale)"

    def _new_step(self, cls: type[NMTransform]) -> NMTransform:
        """A new step; a filter's cutoff is below the Nyquist frequency."""
        t = cls()
        if isinstance(t, NMTransformFilter):
            t.cutoff = start_cutoff(self._xscale_rate())
            self._auto_cutoffs[id(t)] = t.cutoff
        return t

    def _follow_filter_type(self, t: NMTransformFilter) -> None:
        """A notch starts at 50 Hz (mains), other types at the usual cutoff,
        unless a cutoff was typed."""
        if self._auto_cutoffs.get(id(t)) != t.cutoff:
            return
        rate = t.sample_rate or self._xscale_rate()
        preferred = NOTCH_CUTOFF if t.filter_type == "notch" else DEFAULT_CUTOFF
        t.cutoff = start_cutoff(rate, preferred)
        self._auto_cutoffs[id(t)] = t.cutoff
        self._reset_editor(t, "cutoff")

    def _set_text(self, t: NMTransform, key: str, editor: QtWidgets.QLineEdit) -> None:
        current = getattr(t, key)
        if editor.text().strip() == _text(current):
            return
        try:
            value = _parse(key, editor.text())
        except ValueError:
            self.set_status(f"{LABELS.get(key, key)}: not a number: {editor.text()!r}")
            editor.setText(_text(current))
            return
        self._set_value(t, key, value)
        editor.setText(_text(getattr(t, key)))

    def _reset_editor(self, t: NMTransform, key: str) -> None:
        editor, value = self._editors.get(key), getattr(t, key)
        if editor is None:
            return
        editor.blockSignals(True)
        if isinstance(editor, QtWidgets.QComboBox):
            editor.setCurrentText(str(value))
        elif isinstance(editor, QtWidgets.QCheckBox):
            editor.setChecked(value)
        elif isinstance(editor, QtWidgets.QSpinBox):
            editor.setValue(value)
        else:
            editor.setText(_text(value))
        editor.blockSignals(False)

    def _set_value(self, t: NMTransform, key: str, value) -> None:
        try:
            setattr(t, key, value)
        except (TypeError, ValueError) as error:
            self.set_status(f"{LABELS.get(key, key)}: {error}")
            self._reset_editor(t, key)  # back to the value kept
            return
        if key == "filter_type":
            self._follow_filter_type(t)
        self.set_status("")
        row = self.step_list.currentRow()
        item = self.step_list.item(row)
        if item is not None:
            self._updating = True
            item.setText(step_text(t))
            self._updating = False
        for k, editor in self._editors.items():  # e.g. a notch filter uses Q
            editor.setEnabled(applies(t, k))
        self._update_cutoff_tip(t)
        self._update_buttons()
        self.changed.emit()


__all__ = ["MENU_TEXT", "MARK", "MARK_COLOR", "MARK_STYLE", "mark_html", "STEP_TYPES", "TransformDialog", "applies", "step_name",
           "step_text", "steps_tooltip"]
