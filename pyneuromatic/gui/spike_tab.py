"""Spike1: the Spike tool's detection tab (#528), backed by NMToolSpike.

Spikes are threshold crossings: the Threshold (a Y level), the Crossing
(rising, falling or any) and the X range searched. While Spike is the tool
chosen, the Plot shows the X range (draggable edges) and the threshold (a
draggable line), and marks each crossing of the trace shown, as Run would
find it. The Results box lists them. Run analyses the strip's selection as
Stats1 does (epochs, Epoch Sets, Groups, All Channels; Display, Omit),
saving SP_ arrays in a Spike subfolder (#526).
"""
from __future__ import annotations

import math
from typing import Any

from PyQt6 import QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import table_format
from pyneuromatic.gui.plot_levels import PlotLevel
from pyneuromatic.gui.plot_marks import PlotMark
from pyneuromatic.gui.plot_ranges import PlotRange
from pyneuromatic.gui.tool_run import SelectionRunTab
from pyneuromatic.tools.nm_tool_spike import NMToolSpike
from pyneuromatic.tools.nm_tool_utilities import find_level_crossings_nmdata

# The crossing direction: (shown, the tool's func_name)
SPIKE_CROSSINGS = (("rising", "level+"), ("falling", "level-"), ("any", "level"))


class SpikeToolTab(SelectionRunTab):
    """Spike1: set up spike detection and run it."""

    # As Stats1's: its X range and level in blue, what it finds in red
    RANGE_COLOR = "#2563eb"
    LEVEL_COLOR = "#2563eb"
    MARK_COLOR = "#dc2626"

    def __init__(self, manager: NMManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Spike", manager, parent)
        tool = manager.spike
        if tool is None:
            manager.tool_add("spike")
            tool = manager.spike
        if not isinstance(tool, NMToolSpike):
            raise TypeError("manager Spike tool must be an NMToolSpike instance")
        self._tool = tool
        self._trace = None  # the trace the Plot shows (set_trace)

        # Spike1 sets up and runs detection (Spike2, its results, to come)
        self.parameters_group.setTitle("")
        self.mode_tabs = QtWidgets.QTabWidget(self)
        self.mode_tabs.setObjectName("spikeModeTabs")
        self.parameter_layout.addWidget(self.mode_tabs)
        self.spike1_page = QtWidgets.QWidget(self.mode_tabs)
        self.mode_tabs.addTab(self.spike1_page, "Spike1")
        layout = QtWidgets.QVBoxLayout(self.spike1_page)

        level_row = QtWidgets.QHBoxLayout()
        level_row.addWidget(QtWidgets.QLabel("Threshold", self))
        self.ylevel_edit = QtWidgets.QLineEdit(self)
        self.ylevel_edit.setObjectName("spikeThreshold")
        self.ylevel_edit.setToolTip("The Y level a spike crosses (or drag its line on the Plot)")
        level_row.addWidget(self.ylevel_edit, stretch=1)
        level_row.addWidget(QtWidgets.QLabel("Crossing", self))
        self.crossing_combo = QtWidgets.QComboBox(self)
        for shown, func_name in SPIKE_CROSSINGS:
            self.crossing_combo.addItem(shown, func_name)
        self.crossing_combo.setToolTip(
            "The direction of the crossing: rising, falling or either")
        level_row.addWidget(self.crossing_combo)
        layout.addLayout(level_row)

        x_row = QtWidgets.QHBoxLayout()
        self.xbgn_edit = QtWidgets.QLineEdit(self)
        self.xbgn_edit.setPlaceholderText("-inf")
        x_row.addWidget(QtWidgets.QLabel("X begin", self))
        x_row.addWidget(self.xbgn_edit, stretch=1)
        self.xend_edit = QtWidgets.QLineEdit(self)
        self.xend_edit.setPlaceholderText("inf")
        x_row.addWidget(QtWidgets.QLabel("X end", self))
        x_row.addWidget(self.xend_edit, stretch=1)
        layout.addLayout(x_row)

        # Results: detected spikes of the current epoch, as Run would find them
        details_group = QtWidgets.QGroupBox(self.spike1_page)  # its text says what it is
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.details = QtWidgets.QPlainTextEdit(details_group)
        self.details.setReadOnly(True)  # selectable, to copy
        self.details.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.details.setToolTip(
            "Detected spikes of the current epoch")
        self.details.setMaximumHeight(110)
        details_layout.addWidget(self.details)
        layout.addWidget(details_group)
        layout.addStretch(1)

        # Results are listed in the Results box (and saved by Run)
        self.results_group.hide()

        self.ylevel_edit.editingFinished.connect(self._save_controls)
        self.crossing_combo.currentIndexChanged.connect(self._save_controls)
        self.xbgn_edit.editingFinished.connect(self._save_controls)
        self.xend_edit.editingFinished.connect(self._save_controls)
        self.marks_changed.connect(self._show_details)
        self.run_finished.connect(lambda _ok: self._show_details())
        self._load_controls()

    @property
    def spike_tool(self) -> NMToolSpike:
        return self._tool

    # --- the controls and the tool ------------------------------------------

    def _controls(self) -> list[QtWidgets.QWidget]:
        return [self.ylevel_edit, self.crossing_combo, self.xbgn_edit, self.xend_edit]

    @staticmethod
    def _format_bound(value: float) -> str:
        if value == -math.inf:
            return "-inf"
        if value == math.inf:
            return "inf"
        return f"{value:.10g}"

    def _load_controls(self) -> None:
        """The controls show the tool's settings (without saving them)."""
        for control in self._controls():
            control.blockSignals(True)
        self.ylevel_edit.setText(f"{self._tool.ylevel:.10g}")
        self.crossing_combo.setCurrentIndex(
            self.crossing_combo.findData(self._tool.func_name))
        self.xbgn_edit.setText(self._format_bound(self._tool.xbgn))
        self.xend_edit.setText(self._format_bound(self._tool.xend))
        for control in self._controls():
            control.blockSignals(False)
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.ranges_changed.emit()
        self.levels_changed.emit()
        self.marks_changed.emit()

    def _read_controls(self) -> tuple[float, str, float, float]:
        """(ylevel, func_name, xbgn, xend) as typed; raises ValueError."""
        try:
            ylevel = float(self.ylevel_edit.text().strip())
        except ValueError:
            raise ValueError("Threshold must be a number") from None
        if not math.isfinite(ylevel):
            raise ValueError("Threshold must be a number")
        try:
            xbgn = float(self.xbgn_edit.text().strip() or "-inf")
            xend = float(self.xend_edit.text().strip() or "inf")
        except ValueError as error:
            raise ValueError(f"X bounds must be numbers: {error}") from None
        if xbgn >= xend:
            raise ValueError("X begin must be less than X end")
        return ylevel, self.crossing_combo.currentData(), xbgn, xend

    def _save_controls(self, *_args) -> bool:
        """Save what's typed in the tool: one command for each value changed.
        False (with a warning) if a value isn't valid."""
        try:
            values = self._read_controls()
        except ValueError as error:
            self.show_warning(str(error))
            return False
        tool = self._tool
        for key, value in zip(("ylevel", "func_name", "xbgn", "xend"), values):
            if getattr(tool, key) != value:
                setattr(tool, key, value)
        self._emit_changed()
        return True

    def refresh(self) -> None:
        self._load_controls()  # e.g. set by a command

    # --- the Plot: X range, threshold and crossings ----------------------------

    def plot_ranges(self) -> list[PlotRange]:
        return [PlotRange("range", "Spike X range", self._tool.xbgn, self._tool.xend,
                          self.RANGE_COLOR)]

    def set_plot_range(self, key: str, xbgn: float, xend: float, final: bool) -> None:
        """A dragged range: X begin / X end follow, saved when released."""
        self.xbgn_edit.setText(self._format_bound(xbgn))
        self.xend_edit.setText(self._format_bound(xend))
        if final:
            self._save_controls()

    def plot_levels(self, data) -> list[PlotLevel]:
        return [PlotLevel("threshold", "Spike threshold", self._tool.ylevel, self.LEVEL_COLOR)]

    def set_plot_level(self, key: str, y: float, final: bool) -> None:
        """A dragged threshold: its field follows, saved when released."""
        self.ylevel_edit.setText(f"{y:.10g}")
        if final:
            self._save_controls()

    def crossings(self, data) -> list[float]:
        """The x of each spike in *data* (the trace shown, as plotted), as
        Run would find them; during a run, as it's found them."""
        if data is None or data.nparray is None or data.nparray.size == 0:
            return []
        tool = self._tool
        _indexes, times = find_level_crossings_nmdata(
            data, tool.ylevel, func_name=tool.func_name, xbgn=tool.xbgn,
            xend=tool.xend, ignore_nans=tool.ignore_nans)
        return [float(t) for t in times]

    def plot_marks(self, data) -> list[PlotMark]:
        """A mark at each crossing of the threshold."""
        times = self.crossings(data)
        if not times:
            return []
        return [PlotMark("points", times, [self._tool.ylevel] * len(times),
                         self.MARK_COLOR, label="spikes")]

    # --- the Results box ---------------------------------------------------

    def set_trace(self, data) -> None:
        self._trace = data
        self._show_details()

    def details_text(self) -> str:
        """The spikes of the trace shown, e.g. "RecordA0  rising 0 (x in ms)",
        then "3 spikes: 12.400, 55.100, 98.700" (to the decimals in Settings)."""
        data = self._trace
        if data is None or data.nparray is None or data.nparray.size == 0:
            return ""
        crossing = self.crossing_combo.currentText()
        header = f"{data.name}  {crossing} {self._tool.ylevel:g}"
        if data.xscale.units:
            header += f"  (x in {data.xscale.units})"
        try:
            times = self.crossings(data)
        except Exception as error:  # e.g. the range outside the trace
            return f"{header}\nno spikes: {error}"
        if not times:
            return f"{header}\nNo spikes"
        noun = "spike" if len(times) == 1 else "spikes"
        decimals = self.decimals()
        return f"{header}\n{len(times)} {noun}: " + ", ".join(
            table_format.fixed(t, decimals) for t in times)

    def decimals(self) -> int:
        """Decimal places of the spike times shown (Settings: spike_decimals)."""
        return getattr(self._settings, "spike_decimals", 3)

    def apply_settings(self) -> None:
        super().apply_settings()  # the Display control
        self._show_details()  # e.g. other decimals

    def _show_details(self) -> None:
        text = self.details_text()
        if text != self.details.toPlainText():
            self.details.setPlainText(text)

    # --- Run -------------------------------------------------------------------

    def check_run(self) -> None:
        if not self._save_controls():
            raise RuntimeError(self.warnings[-1] if self.warnings else "Check the settings")

    def check_results(self) -> None:
        if not any(len(times) for times in self._tool._spike_times):
            self.show_warning("No spikes found")

    # --- Settings › Spike ----------------------------------------------------

    _CONFIG_LABELS = {
        "ignore_nans": "Ignore NaNs",
        "overwrite": "Overwrite results",
        "results_to_numpy": "Save SP_ arrays",
    }
    _CONFIG_TIPS = {
        "ignore_nans": "Find crossings across NaN gaps (between the values either side)",
        "overwrite": "Clear and reuse the run's first Spike subfolder (…_0), "
                     "instead of making a new one each run",
        "results_to_numpy": "Save each run's results as SP_ arrays in a Spike subfolder",
    }
    _FORMATS = {
        "raster": "Raster (SP_RX, SP_RY)",
        "epochs": "An array per epoch (SP_<data>)",
    }

    def config_options(self) -> list[tuple]:
        """(key, label, value, tip[, choices]) of each option in Settings › Spike."""
        options: list[tuple] = [
            (key, label, bool(getattr(self._tool, key)), self._CONFIG_TIPS[key])
            for key, label in self._CONFIG_LABELS.items()]
        options.append((
            "results_format", "Spike times saved as", self._tool.results_format,
            "Raster: all the spike times in SP_RX and their epochs in SP_RY (for a "
            "raster plot); or an SP_ array of times per epoch. Both save a row per "
            "epoch of counts, rates, first spikes and intervals",
            dict(self._FORMATS)))
        if self._settings is not None:  # how times show: the app's settings
            spec = self._settings._schema["spike_decimals"]
            options.append((
                "decimals", "Decimals displayed", self.decimals(),
                "Decimal places of the spike times in Spike1's Results",
                (spec["min"], spec["max"])))
        return options

    def set_config_option(self, key: str, value: Any) -> None:
        """Set an option from Settings › Spike (used from the next run)."""
        if key == "decimals":  # how times show (settings, not the tool's)
            self._settings.spike_decimals = int(value)
            self._settings.save_user()  # only if loaded from the user's file
            self.apply_settings()
            return
        if key != "results_format" and key not in self._CONFIG_LABELS:
            raise KeyError(f"Spike has no option {key!r}")
        setattr(self._tool, key, value)
        config = self._tool.config
        if config is not None and key in config._schema:
            setattr(config, key, value)
        if key == "ignore_nans":
            self.marks_changed.emit()


__all__ = ["SPIKE_CROSSINGS", "SpikeToolTab"]
