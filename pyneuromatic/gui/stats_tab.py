"""Stats1 tool tab backed by the NMToolStats analysis engine."""
from __future__ import annotations

import math

from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_epoch import EXCLUDE_SET, NMEpoch
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS, NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_stat_func import (
    DECAY_TIME_DEFAULT_PCT,
    FUNC_NAMES_BASIC,
    FUNC_NAMES_BSLN,
    FUNC_NAMES_DECAYTIME,
    FUNC_NAMES_FALLTIME,
    FUNC_NAMES_FWHM,
    FUNC_NAMES_LEVEL,
    FUNC_NAMES_MAXMIN,
    FUNC_NAMES_RISETIME,
    NMStatFunc,
    _stat_func_from_dict,
)
from pyneuromatic.tools.nm_tool_stats import NMToolStats
from pyneuromatic.gui.context_views import ValuesPlot
from pyneuromatic.gui.plot_marks import PlotMark
from pyneuromatic.gui.plot_ranges import PlotRange
from pyneuromatic.tools.nm_stat_win import NMStatWin
from pyneuromatic.gui.results_view import Stats2Panel
from pyneuromatic.gui.selection_model import (
    ALL_EPOCHS,
    SelectionModel,
    scope_runs,
    combine_scope_names,
    scoped_epoch_names,
)
from pyneuromatic.gui.tool_tab import ToolTabWidget


class _ChooseComboBox(QtWidgets.QComboBox):
    """A dropdown whose box is in *color* while its "Choose ..." item (no
    data) shows. Its list keeps the usual colours, so only that item is red
    there: a style sheet would colour the whole list, so it is cleared while
    the list is open."""

    def __init__(self, color: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self._popup_open = False

    def restyle(self) -> None:
        choosing = self.currentData() is None and not self._popup_open
        self.setStyleSheet(f"QComboBox {{ color: {self._color}; }}" if choosing else "")

    def showPopup(self) -> None:  # noqa: N802 (Qt)
        self._popup_open = True
        self.restyle()
        super().showPopup()

    def hidePopup(self) -> None:  # noqa: N802 (Qt)
        super().hidePopup()
        self._popup_open = False
        self.restyle()


# Results that are a level of the trace, drawn as a line across their range
LEVEL_FUNCS = ("median", "mean", "mean+var", "mean+std", "mean+sem")


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def result_marks(results: list[dict], x_first: float, x_last: float,
                 color: str, bsln_color: str) -> list[PlotMark]:
    """What to draw for a window's ``compute()`` *results* (#463).

    - a point found (``x``), e.g. the max or a level crossing: a marker at
      its value, or at the level it crossed (with a dashed line at the level)
    - a level of the trace (mean, median...): a line across its range
    - slope: the line fitted; value@xbgn / value@xend: a marker
    - the baseline: a line across its range, in *bsln_color*

    Results not in the trace's units (std, area, count...) aren't drawn.
    Unbounded ranges are drawn to the trace's ends (*x_first*, *x_last*).
    """
    marks: list[PlotMark] = []
    for r in results:
        func = r.get("func") or {}
        name = func.get("name", "")
        xbgn = r.get("xbgn", -math.inf)
        xend = r.get("xend", math.inf)
        xbgn = x_first if not _finite(xbgn) else float(xbgn)
        xend = x_last if not _finite(xend) else float(xend)
        s = r.get("s")
        if r.get("id") == "bsln":
            if name in LEVEL_FUNCS and _finite(s):
                marks.append(PlotMark("line", [xbgn, xend], [s, s], bsln_color,
                                      label="baseline " + name))
            continue
        if "x" in r:  # a point found
            level = func.get("ylevel")
            y = s if _finite(s) else level
            if _finite(r["x"]) and _finite(y):
                marks.append(PlotMark("points", [float(r["x"])], [float(y)], color, label=name))
            if _finite(level):
                marks.append(PlotMark("line", [xbgn, xend], [level, level], color,
                                      dashed=True, label=f"{name} level"))
        elif name == "slope" and _finite(s) and _finite(r.get("b")):
            b = float(r["b"])
            marks.append(PlotMark("line", [xbgn, xend], [b + s * xbgn, b + s * xend], color,
                                  label=name))
        elif name in ("value@xbgn", "value@xend") and _finite(s):
            x = xbgn if name == "value@xbgn" else xend
            marks.append(PlotMark("points", [x], [float(s)], color, label=name))
        elif name in LEVEL_FUNCS and _finite(s):
            marks.append(PlotMark("line", [xbgn, xend], [s, s], color, label=name))
    return marks


class StatsToolTab(ToolTabWidget):
    """Single-window Stats1 editor and runner using the project's Stats tool."""

    # "Choose measurement" (no measurement yet), as SetX's red
    CHOOSE_COLOR = "#c62828"
    # The current window's X range and its baseline range, on the Plot (#436)
    WINDOW_COLOR = "#2563eb"
    BSLN_COLOR = "#16a34a"
    # Its measurements of the trace shown (#463)
    MARK_COLOR = "#dc2626"
    BSLN_MARK_COLOR = "#15803d"

    NEW_WINDOW_ACTION = "__new_stats_window__"
    DELETE_WINDOW_ACTION = "__delete_stats_window__"
    # Dropdown groups, separated in the measurement menu
    _MEASUREMENT_GROUPS = (
        FUNC_NAMES_BASIC,
        FUNC_NAMES_MAXMIN,
        FUNC_NAMES_LEVEL,
        FUNC_NAMES_RISETIME,
        FUNC_NAMES_FALLTIME,
        FUNC_NAMES_DECAYTIME,
        FUNC_NAMES_FWHM,
    )
    # func parameter key -> (label, type, default)
    _PARAM_SPECS: dict[str, tuple[str, type, int | float]] = {
        "n_mean": ("n_mean", int, 3),
        "ylevel": ("Y level", float, 0.0),
        "n_std": ("n std", float, 2.0),
        "p0": ("p0 %", float, 10.0),
        "p1": ("p1 %", float, 90.0),
    }
    _MEASUREMENT_PARAMS: dict[str, tuple[str, ...]] = {
        "mean@max": ("n_mean",),
        "mean@min": ("n_mean",),
        **{name: ("p0", "p1") for name in FUNC_NAMES_RISETIME},
        **{name: ("p0", "p1") for name in FUNC_NAMES_FALLTIME},
        **{name: ("p0",) for name in FUNC_NAMES_DECAYTIME},
        **{name: ("p0", "p1") for name in FUNC_NAMES_FWHM},
    }
    # Percent parameters mean different things per family, so each family
    # keeps its own values and defaults
    _PARAM_FAMILIES: dict[str, str] = {
        **{name: "risetime" for name in FUNC_NAMES_RISETIME},
        **{name: "falltime" for name in FUNC_NAMES_FALLTIME},
        **{name: "decaytime" for name in FUNC_NAMES_DECAYTIME},
        **{name: "fwhm" for name in FUNC_NAMES_FWHM},
    }
    _FAMILY_DEFAULTS: dict[str, dict[str, float]] = {
        "falltime": {"p0": 90.0, "p1": 10.0},
        "decaytime": {"p0": DECAY_TIME_DEFAULT_PCT},
        "fwhm": {"p0": 50.0, "p1": 50.0},
    }
    # Level thresholds are either an absolute Y level or baseline mean + n std
    _LEVEL_MODES = ("ylevel", "n_std")
    # Measurements whose result is a Y value, so an optional baseline gives Δs
    _DELTA_MEASUREMENTS = frozenset(
        ("median", "mean", "mean+var", "mean+std", "mean+sem",
         "value@xbgn", "value@xend")
        + FUNC_NAMES_MAXMIN
        + FUNC_NAMES_LEVEL
    )

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Stats", parent)
        self._manager = manager

        tool = manager.stats
        if tool is None:
            manager.tool_add("stats")
            tool = manager.stats
        if not isinstance(tool, NMToolStats):
            raise TypeError("manager Stats tool must be an NMToolStats instance")
        self._tool = tool

        # Stats1 configures and runs windows; Stats2 works on saved results
        self.parameters_group.setTitle("")
        self.mode_tabs = QtWidgets.QTabWidget(self)
        self.mode_tabs.setObjectName("statsModeTabs")
        self.parameter_layout.addWidget(self.mode_tabs)
        stats1_tab = QtWidgets.QWidget(self.mode_tabs)
        stats1_tab_layout = QtWidgets.QVBoxLayout(stats1_tab)
        self.mode_tabs.addTab(stats1_tab, "Stats1")
        self.stats2 = Stats2Panel(manager, self.mode_tabs)
        self.mode_tabs.addTab(self.stats2, "Stats2")

        # Its options (Ignore NaNs...) are edited in Settings (the ⋯ by the title)
        self.stats1_page = QtWidgets.QWidget(stats1_tab)
        stats1_tab_layout.addWidget(self.stats1_page)
        stats1_layout = QtWidgets.QVBoxLayout(self.stats1_page)
        stats1_layout.setContentsMargins(0, 0, 0, 0)

        window_row = QtWidgets.QHBoxLayout()
        self.window_combo = QtWidgets.QComboBox(self)
        self.window_combo.setObjectName("statsWindowSelector")
        # Wide enough for a window name (w0, w12), not "New Window...": its
        # menu opens wider to show that and "Delete Window w12"
        self.window_combo.setFixedWidth(70)
        self.window_combo.view().setMinimumWidth(
            self.window_combo.fontMetrics().horizontalAdvance("Delete Window w00") + 40)
        window_row.addWidget(QtWidgets.QLabel("Window", self))
        window_row.addWidget(self.window_combo)
        self.measurement_combo = _ChooseComboBox(self.CHOOSE_COLOR, self)
        self.measurement_combo.addItem("Choose measurement", None)
        # In red until a measurement is chosen (the list item and the combo)
        self.measurement_combo.setItemData(
            0, QtGui.QBrush(QtGui.QColor(self.CHOOSE_COLOR)),
            QtCore.Qt.ItemDataRole.ForegroundRole,
        )
        for group in self._MEASUREMENT_GROUPS:
            if self.measurement_combo.count() > 1:
                self.measurement_combo.insertSeparator(self.measurement_combo.count())
            for name in group:
                self.measurement_combo.addItem(name, name)
        window_row.addWidget(self.measurement_combo, stretch=1)
        self.measurement_combo.currentIndexChanged.connect(self._style_measurement_combo)
        self._style_measurement_combo()
        self.window_enabled = QtWidgets.QCheckBox("On", self)
        window_row.addWidget(self.window_enabled)
        self.reset_button = QtWidgets.QToolButton(self)
        self.reset_button.setText("Reset")
        self.reset_button.setToolTip(
            "Return this window to its defaults: no measurement, X begin / X end "
            "and the baseline cleared")
        self.reset_button.clicked.connect(self._reset_window)
        window_row.addWidget(self.reset_button)
        stats1_layout.addLayout(window_row)

        # Parameter row: only the inputs the current measurement needs are shown
        self.param_row = QtWidgets.QWidget(self.stats1_page)
        param_layout = QtWidgets.QHBoxLayout(self.param_row)
        param_layout.setContentsMargins(0, 0, 0, 0)
        self.level_mode_combo = QtWidgets.QComboBox(self.param_row)
        for key in self._LEVEL_MODES:
            self.level_mode_combo.addItem(self._PARAM_SPECS[key][0], key)
        param_layout.addWidget(self.level_mode_combo)
        self.param_labels: dict[str, QtWidgets.QLabel] = {}
        self.param_edits: dict[str, QtWidgets.QLineEdit] = {}
        # (family, key) -> last value entered; family "" is shared
        self._param_values: dict[tuple[str, str], int | float] = {}
        self._shown_family = ""
        for key, (label, _type, default) in self._PARAM_SPECS.items():
            self.param_labels[key] = QtWidgets.QLabel(label, self.param_row)
            edit = QtWidgets.QLineEdit(f"{default:g}", self.param_row)
            edit.setObjectName(f"statsParam_{key}")
            self.param_edits[key] = edit
            param_layout.addWidget(self.param_labels[key])
            param_layout.addWidget(edit, stretch=1)
        stats1_layout.addWidget(self.param_row)

        x_row = QtWidgets.QHBoxLayout()
        self.xbgn_edit = QtWidgets.QLineEdit(self)
        self.xbgn_edit.setPlaceholderText("-inf")
        x_row.addWidget(QtWidgets.QLabel("X begin", self))
        x_row.addWidget(self.xbgn_edit, stretch=1)
        self.xend_edit = QtWidgets.QLineEdit(self)
        self.xend_edit.setPlaceholderText("inf")
        x_row.addWidget(QtWidgets.QLabel("X end", self))
        x_row.addWidget(self.xend_edit, stretch=1)
        stats1_layout.addLayout(x_row)

        bsln_row = QtWidgets.QHBoxLayout()
        self.bsln_checkbox = QtWidgets.QCheckBox("Bsln", self)
        bsln_row.addWidget(self.bsln_checkbox)
        self.bsln_combo = QtWidgets.QComboBox(self)
        self.bsln_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        for name in FUNC_NAMES_BSLN:
            self.bsln_combo.addItem(name, name)
        self.bsln_combo.setCurrentIndex(self.bsln_combo.findData("mean"))
        bsln_row.addWidget(self.bsln_combo)
        self.bsln_xbgn_edit = QtWidgets.QLineEdit(self)
        self.bsln_xbgn_edit.setPlaceholderText("-inf")
        bsln_row.addWidget(self.bsln_xbgn_edit, stretch=1)
        bsln_row.addWidget(QtWidgets.QLabel("to", self))
        self.bsln_xend_edit = QtWidgets.QLineEdit(self)
        self.bsln_xend_edit.setPlaceholderText("inf")
        bsln_row.addWidget(self.bsln_xend_edit, stretch=1)
        stats1_layout.addLayout(bsln_row)
        # The user's own choice for an optional baseline, kept while a
        # measurement forces the checkbox on or off
        self._bsln_user_on = False

        output_group = QtWidgets.QGroupBox("Output", self.stats1_page)
        output_layout = QtWidgets.QHBoxLayout(output_group)
        self.output_history_checkbox = QtWidgets.QCheckBox("History", output_group)
        self.output_cache_checkbox = QtWidgets.QCheckBox("Results cache", output_group)
        self.output_arrays_checkbox = QtWidgets.QCheckBox("Stats arrays", output_group)
        for checkbox in (
            self.output_history_checkbox,
            self.output_cache_checkbox,
            self.output_arrays_checkbox,
        ):
            output_layout.addWidget(checkbox)
        stats1_layout.addWidget(output_group)
        # Keep the settings at the top when Stats2 makes the tabs taller
        stats1_layout.addStretch(1)

        # The options listed in Settings › Stats
        self._config_keys = ("ignore_nans", "overwrite", "results_all_epochs")
        self._config_labels = {
            "ignore_nans": "Ignore NaNs",
            "overwrite": "Overwrite Stats tool folder",
            "results_all_epochs": "ST_ arrays: one point per epoch",
        }
        self._config_tips = {
            "ignore_nans": "Leave NaN values out of the measurements (otherwise a NaN "
                           "in the window gives NaN)",
            "overwrite": "Clear and reuse the run's first Stats subfolder (…_0), "
                         "instead of making a new one each run",
            "results_all_epochs": "For a data series, one point per epoch of the series "
                                  "(NaN where an epoch wasn't analysed), so ST_ arrays "
                                  "line up with the series",
        }

        self.window_combo.currentIndexChanged.connect(self._on_window_changed)
        self.window_enabled.toggled.connect(self._save_window_controls)
        self.measurement_combo.currentIndexChanged.connect(self._save_window_controls)
        self.xbgn_edit.editingFinished.connect(self._save_window_controls)
        self.xend_edit.editingFinished.connect(self._save_window_controls)
        for edit in self.param_edits.values():
            edit.editingFinished.connect(self._save_window_controls)
        self.level_mode_combo.currentIndexChanged.connect(self._save_window_controls)
        self.bsln_checkbox.toggled.connect(self._on_bsln_toggled)
        self.bsln_combo.currentIndexChanged.connect(self._save_window_controls)
        self.bsln_xbgn_edit.editingFinished.connect(self._save_window_controls)
        self.bsln_xend_edit.editingFinished.connect(self._save_window_controls)
        self.output_history_checkbox.toggled.connect(
            lambda checked: self._set_tool_option("results_to_history", checked)
        )
        self.output_cache_checkbox.toggled.connect(
            lambda checked: self._set_tool_option("results_to_cache", checked)
        )
        self.output_arrays_checkbox.toggled.connect(
            lambda checked: self._set_tool_option("results_to_numpy", checked)
        )
        self._sync_output_options()
        self._refresh_window_combo()
        self._load_selected_window()

        # Results are shown in Stats2 (Details) and the Table pane instead
        self.results_group.hide()
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.run_finished.connect(self._on_run_finished)

        # In the app window: Stats2's plots and its tables
        self.plot_view = ValuesPlot()
        self.table_view = self.stats2.tables
        self.stats2.results_changed.connect(self.results_changed)
        self.stats2.plot_requested.connect(self._show_plot_values)
        # The plot follows the array chosen in Stats2, e.g. after a run
        self.stats2.array_changed.connect(self._plot_selected_array)
        self._plot_selected_array()

    def _on_mode_tab_changed(self, _index: int) -> None:
        # Run belongs to Stats1
        self.run_row.setVisible(self.mode_tabs.currentWidget() is not self.stats2)

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        super()._on_selection_changed(selection)
        self._update_run_info()

    def _update_run_info(self) -> None:
        """Show beside Run how many epochs or data arrays it will analyse, e.g. "n=9"."""
        try:
            planned = self._run_batches()
            targets = self._selected_targets() if planned is None else []
        except RuntimeError as error:
            self.run_info_label.setText(str(error))
            return
        if planned is not None:  # All Sets / All Groups
            batches, skipped = planned
            runs = len(batches)
            self.run_info_label.setText(f"n={runs} run{'s' * (runs != 1)}")
            tip = "Run goes once per set or group: " + ", ".join(
                f"{label}: {len(targets)}" for label, targets, _selection in batches
            ) if batches else "No set or group has epochs to analyze"
            if skipped:
                tip += f"\nSkipped (no epochs): {', '.join(skipped)}"
            self.run_info_label.setToolTip(tip)
            return
        count = len(targets)
        if self.selection.get("source") == "dataseries":
            noun = "epoch" if count == 1 else "epochs"
            tip = f"Run analyses {count} {noun}"
            excluded = self._excluded_in_scope()
            if excluded:
                tip += f" ({excluded} excluded by SetX)"
        else:
            noun = "data array" if count == 1 else "data arrays"
            tip = f"Run analyses {count} {noun}"
        self.run_info_label.setText(f"n={count}")
        self.run_info_label.setToolTip(tip)

    @staticmethod
    def _excluded_names(dataseries, selection) -> set[str]:
        """Epochs left out of runs (SetX), unless SetX is the Epoch Set."""
        if (selection.get("epoch_set") or "").lower() == EXCLUDE_SET.lower():
            return set()
        return dataseries.epochs.excluded_names()

    def _excluded_in_scope(self) -> int:
        """How many epochs of the Epoch Set/Group scope SetX leaves out."""
        selection = self.selection
        dataseries = selection.get("dataseries")
        if dataseries is None:
            return 0
        excluded = self._excluded_names(dataseries, selection)
        try:
            names = scoped_epoch_names(dataseries, selection, exclude=False)
        except RuntimeError:
            return 0
        if names is None:
            epoch = selection.get("epoch")
            names = [epoch.name] if epoch is not None else []
        return sum(name in excluded for name in names)

    def _on_run_finished(self, succeeded: bool) -> None:
        # Saved arrays are viewed in Stats2; without them there is nothing new
        if succeeded and self._tool.toolfolder is not None:
            self.mode_tabs.setCurrentWidget(self.stats2)
            self.view_requested.emit("table")

    def _plot_selected_array(self) -> None:
        """Draw the array chosen in Stats2 (the panel shown doesn't change)."""
        values = self.stats2.selected_plot()
        if values is None:
            self.plot_view.clear()
        else:
            self.plot_view.show_values(*values)

    def _show_plot_values(self, title, y, x, xlabel, ylabel, style) -> None:
        """A Stats2 plot or histogram, in the Plot panel."""
        self.plot_view.show_values(title, y, x, xlabel, ylabel, style)
        self.view_requested.emit("plot")

    def has_results(self) -> bool:
        return self.stats2.has_results()

    def refresh(self) -> None:
        self.stats2.refresh()

    def apply_settings(self) -> None:
        self.stats2.tables.refresh()

    def bind_selection_model(self, model: Any) -> None:
        super().bind_selection_model(model)
        self.stats2.bind_selection_model(model)

    @property
    def stats_tool(self) -> NMToolStats:
        return self._tool

    def analysis_tool(self) -> NMToolStats:
        return self._tool  # reports a run's progress (#450)

    def _refresh_window_combo(self) -> None:
        selected_name = self._tool.windows.selected_name
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for window in self._tool.windows:
            self.window_combo.addItem(window.name, window.name)
        if self.window_combo.count():
            self.window_combo.insertSeparator(self.window_combo.count())
        self.window_combo.addItem("New Window...", self.NEW_WINDOW_ACTION)
        windows = list(self._tool.windows)
        if len(windows) > 1:  # only the last, so names stay in sequence (w0, w1...)
            self.window_combo.addItem(f"Delete Window {windows[-1].name}",
                                      self.DELETE_WINDOW_ACTION)
        index = self.window_combo.findData(selected_name)
        if index < 0 and self.window_combo.count() > 1:
            index = 0
        self.window_combo.setCurrentIndex(index)
        self.window_combo.blockSignals(False)
        if index >= 0 and self.window_combo.currentData() not in (
                self.NEW_WINDOW_ACTION, self.DELETE_WINDOW_ACTION):
            self._tool.windows.selected_name = self.window_combo.currentData()

    def _selected_window(self):
        name = self.window_combo.currentData()
        if name is None:
            return None
        return self._tool.windows[name]

    def _on_window_changed(self, _index: int) -> None:
        name = self.window_combo.currentData()
        if name == self.NEW_WINDOW_ACTION:
            self._add_window()
            return
        if name == self.DELETE_WINDOW_ACTION:
            self._delete_last_window()
            return
        if name is not None:
            self._tool.windows.selected_name = name
        self._load_selected_window()

    def _load_selected_window(self) -> None:
        window = self._selected_window()
        if window is None:
            return
        controls = self._window_controls()
        for control in controls:
            control.blockSignals(True)

        self.window_enabled.setChecked(window.on)
        func = window.func
        func_name = func.get("name")
        self.measurement_combo.setCurrentIndex(self.measurement_combo.findData(func_name))
        if func_name in FUNC_NAMES_LEVEL:
            mode = "n_std" if "n_std" in func else "ylevel"
            self.level_mode_combo.setCurrentIndex(self.level_mode_combo.findData(mode))
        family = self._PARAM_FAMILIES.get(func_name, "")
        for key in self._param_keys(func_name):
            if func.get(key) is not None:
                self._param_values[(family, key)] = func[key]
        self._show_param_values(family)
        self.xbgn_edit.setText(self._format_bound(window.xbgn))
        self.xend_edit.setText(self._format_bound(window.xend))

        self._bsln_user_on = window.bsln_on
        bsln_name = window.bsln_func.get("name")
        if bsln_name is not None:
            self.bsln_combo.setCurrentIndex(self.bsln_combo.findData(bsln_name))
        self.bsln_xbgn_edit.setText(self._format_bound(window.bsln_xbgn))
        self.bsln_xend_edit.setText(self._format_bound(window.bsln_xend))
        self._update_param_row()
        self._update_baseline_row()

        for control in controls:
            control.blockSignals(False)
        self.ranges_changed.emit()
        self.marks_changed.emit()

    def _window_controls(self) -> list[QtWidgets.QWidget]:
        return [
            self.window_enabled,
            self.measurement_combo,
            self.level_mode_combo,
            self.xbgn_edit,
            self.xend_edit,
            self.bsln_checkbox,
            self.bsln_combo,
            self.bsln_xbgn_edit,
            self.bsln_xend_edit,
            *self.param_edits.values(),
        ]

    def _style_measurement_combo(self, *_args) -> None:
        """Red while "Choose measurement" shows, so a missing choice stands out."""
        self.measurement_combo.restyle()

    # --- the options edited in Settings › Stats -----------------------------

    def config_options(self) -> list[tuple[str, str, bool, str]]:
        """(key, label, value, tip) of each option in Settings › Stats."""
        return [(key, self._config_labels[key], bool(getattr(self._tool, key)),
                 self._config_tips[key]) for key in self._config_keys]

    def set_config_option(self, key: str, value: bool) -> None:
        """Set an option from Settings › Stats (used from the next run)."""
        self._set_tool_option(key, value)
        self.marks_changed.emit()  # e.g. Ignore NaNs

    def _set_tool_option(self, key: str, value: bool) -> None:
        setattr(self._tool, key, value)
        config = self._tool.config
        if config is not None and key in config._schema:
            setattr(config, key, value)

    def _sync_output_options(self) -> None:
        for checkbox, key in (
            (self.output_history_checkbox, "results_to_history"),
            (self.output_cache_checkbox, "results_to_cache"),
            (self.output_arrays_checkbox, "results_to_numpy"),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(getattr(self._tool, key)))
            checkbox.blockSignals(False)

    @staticmethod
    def _format_bound(value: float) -> str:
        if value == float("-inf"):
            return "-inf"
        if value == float("inf"):
            return "inf"
        return f"{value:.10g}"  # e.g. a dragged edge, snapped to a sample

    def _param_keys(self, measurement: str | None) -> tuple[str, ...]:
        if measurement in FUNC_NAMES_LEVEL:
            return (self.level_mode_combo.currentData(),)
        return self._MEASUREMENT_PARAMS.get(measurement, ())

    def _param_value(self, family: str, key: str) -> int | float:
        if (family, key) in self._param_values:
            return self._param_values[(family, key)]
        default = self._PARAM_SPECS[key][2]
        return self._FAMILY_DEFAULTS.get(family, {}).get(key, default)

    def _show_param_values(self, family: str) -> None:
        for key, edit in self.param_edits.items():
            edit.setText(f"{self._param_value(family, key):g}")
        self._shown_family = family

    def _update_param_row(self) -> None:
        measurement = self.measurement_combo.currentData()
        keys = self._param_keys(measurement)
        is_level = measurement in FUNC_NAMES_LEVEL
        self.level_mode_combo.setVisible(is_level)
        for key in self._PARAM_SPECS:
            # The level mode combo labels the level input
            self.param_labels[key].setVisible(key in keys and not is_level)
            self.param_edits[key].setVisible(key in keys)
        self.param_row.setVisible(bool(keys))

    def _current_func(self) -> NMStatFunc | None:
        """Engine func for the measurement shown, used for baseline rules."""
        measurement = self.measurement_combo.currentData()
        if not measurement:
            return None
        func: dict[str, Any] = {"name": measurement}
        family = self._PARAM_FAMILIES.get(measurement, "")
        for key in self._param_keys(measurement):
            func[key] = self._param_value(family, key)
        try:
            return _stat_func_from_dict(func)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _allowed_baselines(func: NMStatFunc | None) -> list[str]:
        if func is None:
            return list(FUNC_NAMES_BSLN)
        allowed = []
        for name in FUNC_NAMES_BSLN:
            try:
                func.validate_baseline(name)
            except RuntimeError:
                continue
            allowed.append(name)
        return allowed

    def _update_baseline_row(self) -> None:
        """Lock or free the baseline controls to suit the measurement."""
        measurement = self.measurement_combo.currentData()
        func = self._current_func()
        required = func is not None and func.needs_baseline
        optional = measurement in self._DELTA_MEASUREMENTS
        allowed = self._allowed_baselines(func)

        self.bsln_checkbox.blockSignals(True)
        self.bsln_combo.blockSignals(True)
        self.bsln_checkbox.setChecked(required or (optional and self._bsln_user_on))
        self.bsln_checkbox.setEnabled(optional and not required)
        if required:
            tip = f"{measurement} requires a baseline"
        elif not optional:
            tip = "Baseline is not used by this measurement"
        else:
            tip = "Subtract a baseline to add Δs to the results"
        self.bsln_checkbox.setToolTip(tip)

        model = self.bsln_combo.model()
        for index in range(self.bsln_combo.count()):
            item = model.item(index)
            item.setEnabled(self.bsln_combo.itemData(index) in allowed)
        if self.bsln_combo.currentData() not in allowed and allowed:
            fallback = "mean" if "mean" in allowed else allowed[0]
            self.bsln_combo.setCurrentIndex(self.bsln_combo.findData(fallback))

        enabled = self.bsln_checkbox.isChecked()
        for widget in (self.bsln_combo, self.bsln_xbgn_edit, self.bsln_xend_edit):
            widget.setEnabled(enabled)
        self.bsln_checkbox.blockSignals(False)
        self.bsln_combo.blockSignals(False)

    def _on_bsln_toggled(self, checked: bool) -> None:
        self._bsln_user_on = checked
        self._save_window_controls()

    def _prefill_baseline_range(self) -> None:
        """Default an unset baseline range to trace start .. main X begin."""
        if self.bsln_xbgn_edit.text().strip() not in ("", "-inf"):
            return
        if self.bsln_xend_edit.text().strip() not in ("", "inf"):
            return
        try:
            xbgn = float(self.xbgn_edit.text().strip() or "-inf")
        except ValueError:
            return
        if xbgn == float("-inf"):
            return
        start = 0.0
        try:
            targets = self._selected_targets()
        except RuntimeError:
            targets = []
        if targets:
            start = float(targets[0]["data"].xscale.start)
        if start < xbgn:
            self.bsln_xbgn_edit.setText(f"{start:g}")
            self.bsln_xend_edit.setText(f"{xbgn:g}")

    def _read_params(self, measurement: str | None) -> dict[str, int | float] | None:
        params: dict[str, int | float] = {}
        for key in self._param_keys(measurement):
            label, value_type, _default = self._PARAM_SPECS[key]
            text = self.param_edits[key].text().strip()
            try:
                params[key] = value_type(text)
            except ValueError:
                kind = "a whole number" if value_type is int else "a number"
                self.show_warning(f"{label} must be {kind}")
                return None
        return params

    def _save_window_controls(self, *_args) -> None:
        self._update_param_row()
        window = self._selected_window()
        if window is None:
            return
        measurement = self.measurement_combo.currentData()
        family = self._PARAM_FAMILIES.get(measurement, "")
        if family != self._shown_family:
            self._show_param_values(family)
        params = self._read_params(measurement)
        if params is None:
            return
        self._param_values.update({(family, key): v for key, v in params.items()})
        self._update_baseline_row()
        if self.bsln_checkbox.isChecked():
            self._prefill_baseline_range()
        try:
            xbgn = float(self.xbgn_edit.text().strip() or "-inf")
            xend = float(self.xend_edit.text().strip() or "inf")
        except ValueError as error:
            self.show_warning(f"X bounds must be numbers: {error}")
            return
        if xbgn >= xend:
            self.show_warning("X begin must be less than X end")
            return
        try:
            bsln_xbgn = float(self.bsln_xbgn_edit.text().strip() or "-inf")
            bsln_xend = float(self.bsln_xend_edit.text().strip() or "inf")
        except ValueError as error:
            self.show_warning(f"Baseline X bounds must be numbers: {error}")
            return
        if bsln_xbgn >= bsln_xend:
            self.show_warning("Baseline X begin must be less than X end")
            return

        try:
            func = _stat_func_from_dict({"name": measurement, **params}) if measurement else None
        except (KeyError, TypeError, ValueError) as error:
            self.show_warning(f"Invalid {measurement} parameter: {error}")
            return
        values: dict[str, Any] = {
            "on": self.window_enabled.isChecked(),
            "func": func.to_dict() if func is not None else {},
            "xbgn": xbgn,
            "xend": xend,
            "bsln_on": self.bsln_checkbox.isChecked(),
        }
        if values["bsln_on"]:  # its fields are left as they are while it is off
            values.update(bsln_func={"name": self.bsln_combo.currentData()},
                          bsln_xbgn=bsln_xbgn, bsln_xend=bsln_xend)
        # Only what changed, each one command (#448)
        for key, value in values.items():
            if getattr(window, key) != value:
                setattr(window, key, value if value != {} else None)
        self.ranges_changed.emit()
        self.marks_changed.emit()

    # --- the window's X ranges on the Plot (#436) ----------------------------

    def plot_marks(self, data) -> list[PlotMark]:
        """The current window's measurement of *data*, as Run would make it.

        A copy of the window measures, so nothing is recorded and the
        window's results are kept."""
        window = self._selected_window()
        if (window is None or not window.func or data is None or data.nparray is None
                or data.nparray.size == 0):
            return []
        settings = {k: v for k, v in window.to_dict().items() if k != "name"}
        preview = NMStatWin(window.name, win=settings)
        results = preview.compute(data, ignore_nans=bool(self._tool.ignore_nans), note=False)
        last = data.nparray.size - 1
        x_first, x_last = data.get_xvalue(0), data.get_xvalue(last)
        if x_first is None or x_last is None:  # no x scale: the index, as plotted
            x_first, x_last = 0, last
        return result_marks(results, float(x_first), float(x_last),
                            self.MARK_COLOR, self.BSLN_MARK_COLOR)

    def plot_ranges(self) -> list[PlotRange]:
        """The current window's X range, and its baseline range while Bsln is on."""
        window = self._selected_window()
        if window is None:
            return []
        ranges = [PlotRange("window", window.name, window.xbgn, window.xend, self.WINDOW_COLOR)]
        if window.bsln_on:
            ranges.append(PlotRange("bsln", f"{window.name} baseline", window.bsln_xbgn,
                                    window.bsln_xend, self.BSLN_COLOR))
        return ranges

    def set_plot_range(self, key: str, xbgn: float, xend: float, final: bool) -> None:
        """A dragged range: its X begin / X end fields follow, and it is saved
        (one command for each value changed) when released."""
        if key == "window":
            edits = (self.xbgn_edit, self.xend_edit)
        else:
            edits = (self.bsln_xbgn_edit, self.bsln_xend_edit)
        for edit, value in zip(edits, (xbgn, xend)):
            edit.setText(self._format_bound(value))
        if final:
            self._save_window_controls()

    def _reset_window(self) -> None:
        """Reset: the current window back to its defaults (one command), after
        asking; nothing to do if it has them."""
        window = self._selected_window()
        if window is None:
            return
        defaults = NMStatWin(window.name).to_dict()
        if window.to_dict() == defaults:
            return
        if not self._confirm(
            "Reset window",
            f"Reset {window.name} to its defaults? Its measurement, X begin / X end "
            "and baseline will be cleared.",
        ):
            return
        window.reset()
        self._param_values.clear()  # parameters (e.g. rise time %) back to defaults
        self._load_selected_window()

    def _delete_last_window(self) -> None:
        """Delete Window: the last one (one command), after asking."""
        last = list(self._tool.windows)[-1].name
        if self._confirm("Delete window", f"Delete {last}? Its settings will be lost."):
            self._tool.windows.delete_last()
        self._refresh_window_combo()  # also back from the Delete item if cancelled
        self._load_selected_window()

    def _add_window(self) -> None:
        self._tool.windows.new(select=True)
        self._refresh_window_combo()
        self._load_selected_window()

    def _run_selection_args(self, targets: list[dict], selection: dict) -> tuple[Any, dict]:
        """The run as the Stats tool's ``run_selection`` arguments (its command,
        #448): the data series, its channel, Epoch Set ("all", "Set1"...) or
        the current epoch, with any Group and AND / OR; or a data array
        run's folder and Data Set (or array)."""
        if selection.get("source") != "dataseries":
            data = selection.get("data_set") or targets[0]["data"].name
            return selection.get("toolfolder") or selection.get("folder"), {"data": data}
        series = selection["dataseries"]
        epoch_set = selection.get("epoch_set")
        group = selection.get("group")
        if epoch_set == ALL_EPOCHS:
            epochs = "all"
        elif epoch_set is not None:
            epochs = epoch_set
        elif group is None:  # the current epoch
            epochs = targets[0]["epoch"].name
        else:
            epochs = None  # the Group alone
        args: dict[str, Any] = {"channel": selection["channel"].name, "epochs": epochs}
        if group is not None:
            args["group"] = group
            if epochs not in (None, "all"):
                args["group_operator"] = selection.get("group_operator")
        return series, args

    def _run_selection(self, targets: list[dict], selection: dict) -> bool:
        source, args = self._run_selection_args(targets, selection)
        return self._tool.run_selection(source, **args)

    def _scope_runs(self) -> list[tuple[str, dict]] | None:
        """For All Channels / All Sets / All Groups: (label, selection) per
        run, each naming one channel, set or group; None for a single run.
        Channels combine with sets or groups (channel x set). Raises
        RuntimeError for All Sets with All Groups."""
        selection = self.selection
        series = selection.get("dataseries")
        if selection.get("source") != "dataseries" or series is None:
            return None
        runs = scope_runs(series, selection)
        if not selection.get("all_channels"):
            return runs
        per_channel = runs if runs is not None else [("", selection)]
        return [
            (f"{channel.name} {label}".strip(),
             {**run_selection, "channel": channel, "all_channels": None})
            for channel in series.channels.values()
            for label, run_selection in per_channel
        ]

    def _run_batches(self) -> tuple[list[tuple[str, list, dict]], list[str]] | None:
        """For All Sets / All Groups: the runs with epochs to analyse, as
        (label, targets, selection), and the labels of empty ones (skipped);
        None for a single run."""
        runs = self._scope_runs()
        if runs is None:
            return None
        batches, skipped = [], []
        for label, run_selection in runs:
            targets = self._selected_targets(run_selection)
            if targets:
                batches.append((label, targets, run_selection))
            else:
                skipped.append(label)
        return batches, skipped

    def _selected_targets(self, selection: dict | None = None) -> list[dict[str, NMObject]]:
        """Data to analyse from the selection's source (data or data series).

        A data series uses the epochs in the Epoch Set/Group, or the current
        epoch when neither is chosen. A data array uses the Data Set, or the
        current data array when none is chosen. *selection* defaults to the
        strip's.
        """
        selection = self.selection if selection is None else selection
        source = selection.get("source")

        if source == "dataseries":
            dataseries = selection.get("dataseries")
            channel = selection.get("channel")
            if not isinstance(channel, NMChannel):
                raise RuntimeError("Select a channel before running Stats")
            target_names = scoped_epoch_names(dataseries, selection)
            if target_names is None:  # Current Epoch, unless it is excluded
                epoch = selection.get("epoch")
                target_names = (
                    [epoch.name]
                    if isinstance(epoch, NMEpoch)
                    and epoch.name not in self._excluded_names(dataseries, selection)
                    else []
                )
            targets = []
            wanted = set(target_names)
            found = dataseries.epoch_data_map(channel.name)
            for epoch in dataseries.epochs.values():
                if epoch.name not in wanted:
                    continue
                item = found.get(epoch.name)
                if isinstance(item, NMData):
                    target = self._manager_target(selection)
                    target.update(
                        {
                            "dataseries": dataseries,
                            "channel": channel,
                            "epoch": epoch,
                            "data": item,
                        }
                    )
                    targets.append(target)
            return targets

        if source == "data":
            data = selection.get("data")
            context = selection.get("toolfolder") or selection.get("folder")
            if context is None:
                return []
            data_set = selection.get("data_set")
            if data_set is None:
                data_names = [data.name]
            else:
                members = context.data.sets.get_items(data_set, get_keys=True) or []
                data_names = combine_scope_names(
                    list(context.data.keys()), members, None, None
                )
            targets = []
            for name in data_names:
                item = context.data.get(name)
                if isinstance(item, NMData):
                    target = self._manager_target(selection)
                    target["data"] = item
                    # Analyse the data array, not the selected data series:
                    # None (not left out), so an earlier series run's
                    # selection doesn't stay in the tool (or its folder name)
                    for tier in ("dataseries", "channel", "epoch"):
                        target[tier] = None
                    targets.append(target)
            return targets

        raise RuntimeError("Select a data array or a Data Series channel and epoch first")

    def _manager_target(self, selection: dict[str, Any]) -> dict[str, NMObject]:
        return {
            tier: value
            for tier in HIERARCHY_SELECT_KEYS
            if isinstance((value := selection.get(tier)), NMObject)
        }

    @staticmethod
    def _check_baseline(window) -> None:
        """Raise RuntimeError if *window*'s baseline settings cannot run."""
        func = _stat_func_from_dict(window.func)
        if func is None:
            return
        if func.needs_baseline and not window.bsln_on:
            raise RuntimeError(f"{window.name}: {func.name} requires a baseline")
        if not window.bsln_on:
            return
        func.validate_baseline(window.bsln_func.get("name"))
        if window.bsln_xbgn == float("-inf") and window.bsln_xend == float("inf"):
            raise RuntimeError(f"{window.name}: set a baseline X range")

    def _confirm(self, title: str, text: str) -> bool:
        """Yes/No question (replaced in tests)."""
        answer = QtWidgets.QMessageBox.question(self, title, text)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def run_tool(self) -> Any:
        self._save_window_controls()
        windows = list(self._tool.windows)
        if not any(window.on and window.func for window in windows):
            raise RuntimeError("Choose a measurement in an enabled Stats window first")
        for window in windows:
            if window.on and window.func:
                self._check_baseline(window)
        selection = self.selection
        if (
            selection.get("source") == "dataseries"
            and (selection.get("epoch_set") or "").lower() == EXCLUDE_SET.lower()
            and not self._confirm(
                "Run on SetX",
                f"{EXCLUDE_SET} lists epochs excluded from analysis. Run on them anyway?",
            )
        ):
            raise RuntimeError(f"Cancelled: {EXCLUDE_SET} lists excluded epochs")

        planned = self._run_batches()
        if planned is not None:  # All Sets / All Groups: once per set or group
            return self._run_each(*planned)
        targets = self._selected_targets()
        if not targets:
            raise RuntimeError("The current selection has no data to analyze")
        # One command, naming what it runs on (the subfolder too, e.g. _S1_)
        if not self._run_selection(targets, self.selection):
            raise RuntimeError("Stats analysis did not complete")

        records = self._tool.results
        if not records:
            self.show_warning("No results were produced")
        self.stats2.show_toolfolder(self._tool.toolfolder)
        return None

    def run_summary(self) -> str:
        """e.g. "Complete → Stats_Record_A_S1_0" (the last run's subfolder)."""
        folder = self._tool.toolfolder
        status = self.status_label.text()
        return f"{status} → {folder.name}" if folder is not None else status

    def _run_each(self, batches: list[tuple[str, list, dict]], skipped: list[str]) -> None:
        """Run once per set or group, each saved in its own subfolder."""
        if not batches:
            raise RuntimeError("None of the sets or groups has epochs to analyze")
        self.set_run_total(sum(len(targets) for _label, targets, _sel in batches))
        for _label, targets, selection in batches:  # a command each
            if not self._run_selection(targets, selection):
                raise RuntimeError("Stats analysis did not complete")
            if self.stop_requested:  # Stop: not the sets or groups after it
                break
        if skipped:
            self.show_warning(f"Skipped (no epochs to analyze): {', '.join(skipped)}")
        self.stats2.show_toolfolder(self._tool.toolfolder)  # the last run's
        return None


__all__ = ["StatsToolTab"]