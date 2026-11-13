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
from pyneuromatic.gui.results_view import (
    Stats2Panel, StatsTables, display_name, points_text, shows_points,
)
from pyneuromatic.gui.selection_model import (
    ALL_EPOCHS,
    SelectionModel,
    scope_runs,
    combine_scope_names,
    scoped_epoch_names,
)
from pyneuromatic.gui.tool_tab import ToolTabWidget


# The measurement menu lists each measurement once (#478): a peak
# measurement's sign (+ a positive peak, - a negative one) and a level's
# crossing direction are choices beside it. The core names are unchanged
# (risetime+, level-...), so commands, saved windows and scripts are too.
SIGNED_MEASUREMENTS = ("risetime", "risetimeslope", "falltime", "falltimeslope",
                       "decaytime", "fwhm")
PEAK_SIGNS = (("+", "+"), ("\u2212", "-"))  # (shown, in the core name)
CROSSINGS = (("any", ""), ("rising", "+"), ("falling", "-"))
# Not in the menu: every measurement records its points, NaNs and INFs
# (n, nans, infs; ST_ arrays _n, _nans, _infs)
MENU_HIDDEN = ("count", "count_nans", "count_infs")


def split_measurement(name: str | None) -> tuple[str | None, str]:
    """A core name as (menu name, sign), e.g. "risetime-" -> ("risetime",
    "-"), "level+" -> ("level", "+"), "mean+std" -> ("mean+std", "")."""
    if name and name[-1] in "+-" and (name[:-1] in SIGNED_MEASUREMENTS or name[:-1] == "level"):
        return name[:-1], name[-1]
    return name, ""


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


def _number(value, digits: int = 6) -> str:
    """*value* to *digits* significant digits; ±inf as "-inf" / "inf"."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if math.isinf(number):
        return "-inf" if number < 0 else "inf"
    return "NaN" if math.isnan(number) else f"{number:.{digits}g}"


def _limits(xbgn, xend, x_limits: tuple[float, float] | None) -> tuple:
    """An X range as measured: an unbounded edge (-inf, inf) at the trace's
    start or end (*x_limits*), as the Plot draws it."""
    if x_limits is None:
        return xbgn, xend
    try:
        xbgn = x_limits[0] if math.isinf(float(xbgn)) else xbgn
        xend = x_limits[1] if math.isinf(float(xend)) else xend
    except (TypeError, ValueError):
        pass
    return xbgn, xend


def result_lines(results: list[dict], digits: int = 6, xunits: str = "",
                 x_limits: tuple[float, float] | None = None,
                 known_units: tuple[str, ...] = (),
                 window_x: tuple | None = None) -> list[str]:
    """A window's ``compute()`` *results* as text, a line per step (#471), e.g.

        bsln mean = -70 mV  (x 0 to 40, n = 41)
        max = -55 mV at 80 ms  (Δ = 15 mV)
        level+ 10% = -68.5 mV at 61.77 ms
        level+ 90% = -56.5 mV at 76.12 ms
        risetime+ = 14.34 ms  (x 60 to 120)

    *window_x*, the window's X range (xbgn, xend), is shown on the
    measurement's line, as the baseline's, with its number of points:
    "mean = -64.8  (x 60 to 120, n = 61)"; for a measurement of several
    steps (e.g. a rise time), on its result's line.

    Δ (the difference from the baseline) is shown when it's in the
    baseline's units (not, e.g., for a slope). A measurement from many
    points (a mean, std, slope...; not a point found) ends with how many,
    e.g. "(n = 15)", and the NaNs left out, if any (#498). Values in *known_units*
    (given once, on the top line) are shown without them; others (e.g. a
    slope's intercept) keep theirs.
    """
    def number(value) -> str:
        return _number(value, digits)

    def _value(value, units: str = "") -> str:
        if not units or units in known_units:
            return number(value)
        return f"{number(value)} {units}"

    def window_text(points: str = "") -> str:
        xbgn, xend = window_x
        return (f"  (x {number(xbgn)} to {_value(xend, xunits)}"
                + (f", {points}" if points else "") + ")")

    lines: list[str] = []
    bsln_units = None
    window_line = None  # the line the window's X range is on
    for r in results:
        func = r.get("func") or {}
        name = func.get("name", "")
        if r.get("id") == "bsln":
            bsln_units = r.get("sunits", "")
            xbgn, xend = _limits(r.get("xbgn"), r.get("xend"), x_limits)
            spread = "".join(  # a baseline's spread (e.g. mean ± std), as the mean's
                f", \u00b1{key} {_value(r[key], bsln_units if key != 'var' else '')}"
                for key in ("std", "sem", "var") if key in r)
            # Its value is the mean (a spread follows): "bsln mean", as the mean's
            label = "mean" if name.startswith("mean+") else name
            points = f", {points_text(r['n'], r.get('nans', 0))}" if "n" in r else ""
            lines.append(f"bsln {label} = {_value(r.get('s'), bsln_units)}"
                         f"{spread}  (x {number(xbgn)} to {_value(xend, xunits)}{points})")
            continue
        # mean+std's value is the mean (its std follows): not "mean+std = ..."
        label = "mean" if name.startswith("mean+") else name
        percent = r.get("p0", r.get("p1"))
        if percent is not None:
            label += f" {float(percent):g}%"
        units = r.get("sunits") or func.get("yunits", "")
        if "x" in r:  # a point found: its value (or the level crossed) and x
            y = r.get("s") if _finite(r.get("s")) else func.get("ylevel")
            line = f"{label} = {_value(y, units)} at {_value(r['x'], r.get('xunits', ''))}"
        elif "s" in r:
            line = f"{label} = {_value(r['s'], units)}"
            for key in ("std", "sem", "var"):
                if key in r:  # the mean's spread, as ±std (copies as written)
                    # var's units would be squared: none shown
                    line += f", \u00b1{key} {_value(r[key], units if key != 'var' else '')}"
            if "b" in r:
                line += f", intercept {_value(r['b'], r.get('bunits', ''))}"
        else:
            line = f"{label}: no value"
        if (_finite(r.get("Δs")) and bsln_units is not None and units == bsln_units):
            line += f"  (Δ = {_value(r['Δs'], units)})"
        points = (points_text(r["n"], r.get("nans", 0))
                  if "n" in r and "s" in r and shows_points(name) else "")
        if window_x is not None and r.get("id") == "main":
            line += window_text(points)  # e.g. "(x 60 to 120, n = 61)" (#498)
            window_line = len(lines)
        elif points:
            line += f"  ({points})"
        lines.append(line)
        if "dx" in r:  # e.g. the rise time: the crossings' difference
            lines.append(f"{r.get('id')} = {_value(r['dx'], r.get('xunits', ''))}")
            if window_x is not None and window_line is None:
                lines[-1] += window_text()
                window_line = len(lines) - 1
    if window_x is not None and window_line is None:  # e.g. a rise time not found
        last = next((i for i in range(len(lines) - 1, -1, -1)
                     if not lines[i].startswith("bsln ")), None)
        if last is not None:
            lines[last] += window_text()
    return lines


class StatsToolTab(ToolTabWidget):
    """Single-window Stats1 editor and runner using the project's Stats tool."""

    # Its tab shows Stats2's plots and tables (#523)
    view_label = "Stats2"
    NO_RESULTS_TITLE = "No Stats2 results yet: Run in Stats1"

    # The current window's X range and its baseline range, on the Plot (#436)
    WINDOW_COLOR = "#2563eb"
    BSLN_COLOR = "#16a34a"
    # Its measurements of the trace shown (#463)
    MARK_COLOR = "#dc2626"
    BSLN_MARK_COLOR = "#15803d"

    # Dropdown groups, separated in the measurement menu
    _MEASUREMENT_GROUPS = (
        FUNC_NAMES_BASIC,
        FUNC_NAMES_MAXMIN,
        FUNC_NAMES_LEVEL,
        FUNC_NAMES_RISETIME + FUNC_NAMES_FALLTIME,  # rise and fall together
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
        self.window_combo.setFixedWidth(70)  # a window name (w0, w12)
        window_row.addWidget(QtWidgets.QLabel("Window", self))
        window_row.addWidget(self.window_combo)
        # Its actions under ⋯, as the strip's selectors (#514): the list
        # holds only the windows, so moving through it can't start one
        self.window_menu = QtWidgets.QMenu(self)
        self.new_window_action = self.window_menu.addAction("New Window…")
        self.new_window_action.setToolTip(
            "Add a window and show it: the next number, or one to fill a gap")
        self.new_window_action.triggered.connect(lambda _checked=False: self._new_window())
        self.copy_window_action = self.window_menu.addAction("Copy…")
        self.copy_window_action.setToolTip(
            "A new window with a window's settings (e.g. to measure another range)")
        self.copy_window_action.triggered.connect(lambda _checked=False: self._copy_window())
        self.reset_window_action = self.window_menu.addAction("Reset…")
        self.reset_window_action.setToolTip("Return a window to its defaults")
        self.reset_window_action.triggered.connect(lambda _checked=False: self._reset_window())
        self.delete_window_action = self.window_menu.addAction("Delete…")
        self.delete_window_action.setToolTip("Delete a window; the others keep their names")
        self.delete_window_action.triggered.connect(
            lambda _checked=False: self._delete_window())
        self.window_actions_button = QtWidgets.QToolButton(self)
        self.window_actions_button.setText("⋯")
        self.window_actions_button.setObjectName("statsWindowActions")
        self.window_actions_button.setToolTip("Window actions")
        self.window_actions_button.setStyleSheet(
            "QToolButton::menu-indicator { image: none; width: 0px; }")
        self.window_actions_button.setMenu(self.window_menu)
        self.window_actions_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        window_row.addWidget(self.window_actions_button)
        self.measurement_combo = QtWidgets.QComboBox(self)
        self.measurement_combo.addItem("Choose measure", None)  # none yet
        for group in self._MEASUREMENT_GROUPS:
            names = []  # each measurement once, without its sign (#478)
            for name in group:
                base = split_measurement(name)[0]
                if base not in names and base not in MENU_HIDDEN:
                    names.append(base)
            if self.measurement_combo.count() > 1:
                self.measurement_combo.insertSeparator(self.measurement_combo.count())
            for name in names:
                self.measurement_combo.addItem(display_name(name), name)
        # As wide as its longest item (e.g. "Choose measure"), not the row
        self.measurement_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        window_row.addWidget(self.measurement_combo)
        self.window_enabled = QtWidgets.QCheckBox("On", self)
        window_row.addWidget(self.window_enabled)
        window_row.addStretch(1)
        stats1_layout.addLayout(window_row)

        # Parameter row: only the inputs the current measurement needs are shown
        self.param_row = QtWidgets.QWidget(self.stats1_page)
        param_layout = QtWidgets.QHBoxLayout(self.param_row)
        param_layout.setContentsMargins(0, 0, 0, 0)
        # The sign: a peak's (+ / -), a level crossing's direction (#478)
        self.peak_label = QtWidgets.QLabel("Peak", self.param_row)
        self.peak_combo = QtWidgets.QComboBox(self.param_row)
        for shown, sign in PEAK_SIGNS:
            self.peak_combo.addItem(shown, sign)
        self.peak_combo.setToolTip("+ a positive peak (max), \u2212 a negative peak (min)")
        self.crossing_label = QtWidgets.QLabel("Crossing", self.param_row)
        self.crossing_combo = QtWidgets.QComboBox(self.param_row)
        for shown, sign in CROSSINGS:
            self.crossing_combo.addItem(shown, sign)
        self.crossing_combo.setToolTip("The direction of the crossing: either, rising or falling")
        for widget in (self.peak_label, self.peak_combo, self.crossing_label,
                       self.crossing_combo):
            param_layout.addWidget(widget)
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
            self.bsln_combo.addItem(display_name(name), name)
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

        # Results: the current window's measurement of the trace shown (#471)
        details_group = QtWidgets.QGroupBox(self.stats1_page)  # untitled: its text says what it is
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.details = QtWidgets.QPlainTextEdit(details_group)
        self.details.setReadOnly(True)  # selectable, to copy
        self.details.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        # Six lines (e.g. rise time with Bsln), measured as the box draws
        # them; shorter than Stats2's, which has more to show
        self.details.document().setDocumentMargin(3)
        sample = QtGui.QTextDocument()
        sample.setDefaultFont(self.details.font())
        sample.setDocumentMargin(3)
        sample.setPlainText("\n".join(["x"] * 6))
        self.details.setFixedHeight(
            int(sample.size().height()) + 2 * self.details.frameWidth() + 2)
        self.details.setToolTip(
            "The current window's measurement of the trace in the Plot "
            "(as Run would make it; during a run, the run's own)")
        details_layout.addWidget(self.details)
        stats1_layout.addWidget(details_group)
        self._trace = None  # the trace the Plot shows (set_trace)
        self.marks_changed.connect(self._show_details)  # e.g. its function or range
        # Keep the settings at the top when Stats2 makes the tabs taller
        stats1_layout.addStretch(1)

        # The options listed in Settings › Stats
        # Settings › Stats (the Stats arrays option was the panel's Output, #480)
        self._config_keys = ("ignore_nans", "overwrite", "results_to_numpy",
                             "results_all_epochs")
        self._config_labels = {
            "ignore_nans": "Ignore NaNs",
            "overwrite": "Overwrite Stats tool folder",
            "results_to_numpy": "Save ST_ arrays",
            "results_all_epochs": "ST_ arrays: one point per epoch",
        }
        self._config_tips = {
            "ignore_nans": "Leave NaN values out of the measurements (otherwise a NaN "
                           "in the window gives NaN)",
            "overwrite": "Clear and reuse the run's first Stats subfolder (…_0), "
                         "instead of making a new one each run",
            "results_to_numpy": "Save each run's results as ST_ arrays in a Stats "
                                "subfolder (what Stats2, its tables and plots show)",
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
        self.peak_combo.currentIndexChanged.connect(self._save_window_controls)
        self.crossing_combo.currentIndexChanged.connect(self._save_window_controls)
        self.bsln_checkbox.toggled.connect(self._on_bsln_toggled)
        self.bsln_combo.currentIndexChanged.connect(self._save_window_controls)
        self.bsln_xbgn_edit.editingFinished.connect(self._save_window_controls)
        self.bsln_xend_edit.editingFinished.connect(self._save_window_controls)
        self._refresh_window_combo()
        self._load_selected_window()

        # Results are shown in Stats2 (Results) and the Table pane instead
        self.results_group.hide()
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.run_finished.connect(self._on_run_finished)

        # In the app window: Stats2's plots and its tables
        self.plot_view = ValuesPlot()
        self.table_view = self.stats2.tables
        self.stats2.results_changed.connect(self.results_changed)
        self.stats2.plot_requested.connect(self._show_plot_values)
        # Stats2 ⋯ › Table: its tables, the chosen array's column (#523)
        self.stats2.table_requested.connect(lambda: self.view_requested.emit("table"))
        # The plot follows the array chosen in Stats2, e.g. after a run
        self.stats2.array_changed.connect(self._plot_selected_array)
        self._plot_selected_array()

    def _on_mode_tab_changed(self, _index: int) -> None:
        # Run (and its Display) belong to Stats1
        stats1 = self.mode_tabs.currentWidget() is not self.stats2
        self.run_row.setVisible(stats1)
        self.display_row.setVisible(stats1 and self._settings is not None)
        # The view follows (#523): Stats1, the channel (its measure on the
        # trace); Stats2, its tab; Plot or Table stays as it is
        self.view_requested.emit(self.preferred_view())

    def preferred_view(self) -> str:
        return "tool" if self.mode_tabs.currentWidget() is self.stats2 else "channel"

    def lists_view(self) -> bool:
        return True  # Stats2's tab, to switch to (saying so, without results)

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
        # Results: a preview again (during the run, the run's own)
        self._show_details()
        # Saved arrays are viewed in Stats2; without them there is nothing new
        if succeeded and self._tool.toolfolder is not None:
            self.mode_tabs.setCurrentWidget(self.stats2)
            self.view_requested.emit("plot")  # its plot of the Stats2 array; Channel a click away

    def _plot_selected_array(self) -> None:
        """Draw the array chosen in Stats2 (the panel shown doesn't change)."""
        values = self.stats2.selected_plot()
        if values is None:
            self.plot_view.clear(None if self.has_results() else self.NO_RESULTS_TITLE)
        else:
            self.plot_view.show_values(*values)

    def _show_plot_values(self, title, y, x, xlabel, ylabel, style) -> None:
        """A Stats2 plot or histogram, in the Plot panel."""
        self.plot_view.show_values(title, y, x, xlabel, ylabel, style)
        self.view_requested.emit("plot")

    def has_results(self) -> bool:
        return self.stats2.has_results()

    def set_folder_actions(self, actions) -> None:
        self.stats2.set_folder_actions(actions)  # its Folder ⋯ menu (#494)

    def refresh(self) -> None:
        self.stats2.refresh()

    def apply_settings(self) -> None:
        super().apply_settings()  # the Display control
        StatsTables.digits = self.digits()
        self.stats2.set_detail_arrays(self.detail_arrays())
        self.stats2.tables.refresh()
        self.stats2.refresh_details()
        self._show_details()

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
        index = self.window_combo.findData(selected_name)
        if index < 0 and self.window_combo.count():
            index = 0
        self.window_combo.setCurrentIndex(index)
        self.window_combo.blockSignals(False)
        if index >= 0:
            self._tool.windows.selected_name = self.window_combo.currentData()
        count = len(self._tool.windows)
        self.reset_window_action.setEnabled(count > 0)
        self.delete_window_action.setEnabled(count > 1)  # one is always kept

    def _selected_window(self):
        name = self.window_combo.currentData()
        if name is None:
            return None
        return self._tool.windows[name]

    def _on_window_changed(self, _index: int) -> None:
        name = self.window_combo.currentData()
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
        self._show_measurement(func_name)
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
            self.peak_combo,
            self.crossing_combo,
            self.level_mode_combo,
            self.xbgn_edit,
            self.xend_edit,
            self.bsln_checkbox,
            self.bsln_combo,
            self.bsln_xbgn_edit,
            self.bsln_xend_edit,
            *self.param_edits.values(),
        ]

    # --- the options edited in Settings › Stats -----------------------------

    def config_options(self) -> list[tuple]:
        """(key, label, value, tip) of each option in Settings › Stats; a
        number also has its (minimum, maximum)."""
        options: list[tuple] = [
            (key, self._config_labels[key], bool(getattr(self._tool, key)),
             self._config_tips[key]) for key in self._config_keys]
        if self._settings is not None:  # how values show: the app's settings
            options.append((
                "detail_arrays", "Show index and count ST_ arrays", self.detail_arrays(),
                "Also list the index and point-count arrays (i, i0, i1, n, nans, "
                "infs) and the epoch numbers, in the Stats tables and Stats2's "
                "Array list"))
            spec = self._settings._schema["stats_digits"]
            options.append((
                "digits", "Digits displayed", self.digits(),
                "Significant digits of the values in Stats1's Results and "
                "Stats2's tables and Results (copying keeps them all)",
                (spec["min"], spec["max"])))
        return options

    def set_config_option(self, key: str, value) -> None:
        """Set an option from Settings › Stats (used from the next run)."""
        if key in ("digits", "detail_arrays"):  # how values show (settings, not the tool's)
            if key == "digits":
                self._settings.stats_digits = int(value)
            else:
                self._settings.stats_detail_arrays = bool(value)
            self._settings.save_user()  # only if loaded from the user's file
            self.apply_settings()
            return
        self._set_tool_option(key, value)
        self.marks_changed.emit()  # e.g. Ignore NaNs

    def digits(self) -> int:
        """Significant digits of the values shown (Settings: stats_digits)."""
        return getattr(self._settings, "stats_digits", 6)

    def detail_arrays(self) -> bool:
        """Show the index and count arrays (Settings: stats_detail_arrays)."""
        return bool(getattr(self._settings, "stats_detail_arrays", False))

    def set_settings(self, settings) -> None:
        super().set_settings(settings)
        self.apply_settings()  # the digits shown

    def _set_tool_option(self, key: str, value: bool) -> None:
        setattr(self._tool, key, value)
        config = self._tool.config
        if config is not None and key in config._schema:
            setattr(config, key, value)

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
        measurement = self.measurement()
        keys = self._param_keys(measurement)
        is_level = measurement in FUNC_NAMES_LEVEL
        base = self.measurement_combo.currentData()
        signed = base in SIGNED_MEASUREMENTS
        self.peak_label.setVisible(signed)
        self.peak_combo.setVisible(signed)
        self.crossing_label.setVisible(is_level)
        self.crossing_combo.setVisible(is_level)
        self.level_mode_combo.setVisible(is_level)
        for key in self._PARAM_SPECS:
            # The level mode combo labels the level input
            self.param_labels[key].setVisible(key in keys and not is_level)
            self.param_edits[key].setVisible(key in keys)
        self.param_row.setVisible(bool(keys) or signed)

    # --- the measurement: the menu and its sign (#478) ----------------------

    def measurement(self) -> str | None:
        """The core name of the measurement shown, e.g. "risetime-" (the menu's
        risetime with Peak -), "level+" (level, Crossing rising)."""
        base = self.measurement_combo.currentData()
        if base in SIGNED_MEASUREMENTS:
            return base + self.peak_combo.currentData()
        if base == "level":
            return base + self.crossing_combo.currentData()
        return base

    def _show_measurement(self, name: str | None) -> None:
        """Show core name *name* in the menu and its sign (without saving).
        One not in the menu (e.g. count, set from a script) is added."""
        base, sign = split_measurement(name)
        if base is not None and self.measurement_combo.findData(base) < 0:
            self.measurement_combo.addItem(base, base)
        self.measurement_combo.setCurrentIndex(self.measurement_combo.findData(base))
        if base in SIGNED_MEASUREMENTS:
            self.peak_combo.setCurrentIndex(self.peak_combo.findData(sign or "+"))
        elif base == "level":
            self.crossing_combo.setCurrentIndex(self.crossing_combo.findData(sign))

    def set_measurement(self, name: str | None) -> None:
        """Choose core name *name* (e.g. "fwhm-"), as choosing it in the menu
        and its sign would: saved, one command."""
        controls = (self.measurement_combo, self.peak_combo, self.crossing_combo)
        for control in controls:
            control.blockSignals(True)
        self._show_measurement(name)
        for control in controls:
            control.blockSignals(False)
        self._save_window_controls()

    def _current_func(self) -> NMStatFunc | None:
        """Engine func for the measurement shown, used for baseline rules."""
        measurement = self.measurement()
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
        measurement = self.measurement()
        func = self._current_func()
        required = func is not None and func.needs_baseline
        optional = measurement in self._DELTA_MEASUREMENTS
        allowed = self._allowed_baselines(func)

        self.bsln_checkbox.blockSignals(True)
        self.bsln_combo.blockSignals(True)
        self.bsln_checkbox.setChecked(required or (optional and self._bsln_user_on))
        # Required: shown on, not faded, but it can't be turned off (clicks and
        # keys don't reach it). Not used: faded (off and unavailable)
        self.bsln_checkbox.setEnabled(optional or required)
        self.bsln_checkbox.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, required)
        self.bsln_checkbox.setFocusPolicy(
            QtCore.Qt.FocusPolicy.NoFocus if required else QtCore.Qt.FocusPolicy.StrongFocus)
        if required:
            tip = f"{measurement} requires a baseline (always on)"
        elif not optional:
            tip = "Baseline is not used by this measurement"
        else:
            tip = "Measure baseline and report Δ (value − baseline)"
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
        measurement = self.measurement()
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
        window's results are kept. During a run, the run's own results for
        the epoch shown (#468), so the Plot shows what's saved."""
        window = self._selected_window()
        if (window is None or not window.func or data is None or data.nparray is None
                or data.nparray.size == 0):
            return []
        results = self._measure(window, data)
        last = data.nparray.size - 1
        x_first, x_last = data.get_xvalue(0), data.get_xvalue(last)
        if x_first is None or x_last is None:  # no x scale: the index, as plotted
            x_first, x_last = 0, last
        return result_marks(results, float(x_first), float(x_last),
                            self.MARK_COLOR, self.BSLN_MARK_COLOR)

    def _measure(self, window, data) -> list[dict]:
        """*window*'s measurement of *data*, as Run would make it: a copy of
        the window measures (nothing recorded, its results kept). During a
        run, the run's own results for the epoch shown (#468)."""
        if self.running:
            return window.results if window.on else []
        settings = {k: v for k, v in window.to_dict().items() if k != "name"}
        preview = NMStatWin(window.name, win=settings)
        return preview.compute(data, ignore_nans=bool(self._tool.ignore_nans), note=False)

    def set_trace(self, data) -> None:
        self._trace = data
        self._show_details()

    def details_text(self) -> str:
        """The Results box's text: the current window's measurement of the
        trace shown (#471)."""
        window, data = self._selected_window(), self._trace
        if window is None or data is None or data.nparray is None or data.nparray.size == 0:
            return ""
        name = window.func.get("name") if window.func else None
        header = f"{data.name}  {window.name}" + (f" {display_name(name)}" if name else "")
        if not name:
            return f"{header}\nChoose a measurement"
        digits = self.digits()
        xunits = data.xscale.units or ""  # the window's X range, in x units
        last = data.nparray.size - 1  # an unbounded edge: the trace's start or end
        first_x, last_x = data.get_xvalue(0), data.get_xvalue(last)
        x_limits = (first_x, last_x) if first_x is not None and last_x is not None else None
        # The window's X range: on the measurement's line, as the baseline's (#498)
        window_x = _limits(window.xbgn, window.xend, x_limits)
        # Units once, here, as Stats2's: "(values in mV, x in ms)"
        yunits = data.yscale.units or ""
        known = tuple(u for u in (yunits, xunits) if u)
        if known:
            header += "  (" + ", ".join(
                f"{axis} in {u}" for axis, u in (("values", yunits), ("x", xunits)) if u) + ")"
        if not window.on:
            return f"{header}\nWindow off"
        try:
            lines = result_lines(self._measure(window, data), digits, xunits, x_limits, known,
                                 window_x)
        except Exception as error:  # e.g. the window outside the trace
            lines = [f"no value: {error}"]
        return "\n".join([header, *lines])

    def _show_details(self) -> None:
        text = self.details_text()
        if text != self.details.toPlainText():
            self.details.setPlainText(text)

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

    def _choose(self, title: str, **options) -> tuple | None:
        """A prompt (as the strip's ⋯ menus'); returns (item, name, *extras)
        or None if cancelled (replaced in tests)."""
        from pyneuromatic.gui.tier_actions import ChooseItemDialog
        dialog = ChooseItemDialog(title, parent=self, **options)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return (dialog.item(), dialog.name(), *dialog.extras())

    def _choose_window(self, title: str, message, validate=None) -> str | None:
        """A window, chosen in a prompt starting on the window shown."""
        self._save_window_controls()  # e.g. a range being typed
        windows = self._tool.windows
        answer = self._choose(
            title, item_label="Window", items=[w.name for w in windows],
            current=windows.selected_name, message=message,
            validate=(lambda item, _name: validate(item)) if validate else None,
        )
        return None if answer is None else answer[0]

    @staticmethod
    def _has_defaults(window: NMStatWin) -> bool:
        return window.to_dict() == NMStatWin(window.name).to_dict()

    def _reset_window(self) -> None:
        """Reset… (Window ⋯): a window back to its defaults (one command)."""
        windows = self._tool.windows
        name = self._choose_window(
            "Reset Stats Window",
            message=lambda item: (
                f"{item} already has its defaults." if self._has_defaults(windows[item])
                else f"{item}'s measurement, X begin / X end and baseline will be cleared."),
        )
        if name is None or self._has_defaults(windows[name]):
            return  # nothing to reset: no command
        windows[name].reset()
        if name == windows.selected_name:
            self._param_values.clear()  # parameters (e.g. rise time %) back to defaults
        self._refresh_window_combo()
        self._load_selected_window()

    def _delete_window(self) -> None:
        """Delete… (Window ⋯): any window but the only one (one command); the
        others keep their names (w0, w2), as results and run notes name them."""
        windows = self._tool.windows
        if len(windows) <= 1:
            return
        name = self._choose_window(
            "Delete Stats Window",
            message=lambda item: (f"{item} will be deleted; its settings will be lost. "
                                  "The other windows keep their names."),
        )
        if name is None:
            return
        windows.delete(name)
        self._refresh_window_combo()
        self._load_selected_window()

    def _new_window(self) -> None:
        """New Window… (Window ⋯): a window with the number chosen, starting
        on the one after the highest (w0, w1, w3 → w4); a gap can be filled
        (w2)."""
        error = self._window_number_error
        answer = self._choose(
            "New Stats Window",
            extras=[self._window_number_spin("Window")],
            message=lambda _item, number: "" if error(number) else f"Creates {number}.",
            validate=lambda _item, _name, number: error(number),
            validate_live=True,
        )
        if answer is None:
            return
        self._add_window(int(answer[-1][1:]))

    def _window_number_spin(self, label: str) -> tuple:
        """A number box for a new window (w4), starting after the highest."""
        from pyneuromatic.gui.tier_actions import Spin
        return (label, Spin(0, 9999, self._tool.windows.next_number, prefix="w"))

    def _window_number_error(self, name: str) -> str | None:
        """Why a new window can't be *name* (e.g. "w3"), or None."""
        return f"{name} already exists." if name in self._tool.windows else None

    def _copy_window(self) -> None:
        """Copy… (Window ⋯, #515): a window with a window's settings (starting
        on the one shown), the number chosen as New Window…'s, or an
        existing one with Overwrite (as Copy Set); the copy is shown, ready
        to change (e.g. another range)."""
        from pyneuromatic.gui.tier_actions import Check, warning_html
        self._save_window_controls()  # e.g. a range being typed
        windows = self._tool.windows

        def error(item, _name, number, overwrite):
            if number == item:
                return f"{item} can't be copied onto itself."
            return None if overwrite else self._window_number_error(number)

        def message(item, number, overwrite):
            if error(item, None, number, overwrite):
                return ""
            if overwrite:
                return warning_html(f"{number}'s settings will be replaced with {item}'s.")
            return f"Copies {item}'s settings to {number}."

        answer = self._choose(
            "Copy Stats Window", item_label="Window", items=[w.name for w in windows],
            current=windows.selected_name,
            extras=[self._window_number_spin("To"),
                    ("", Check("Overwrite existing window", enabled=lambda _item, number, *_:
                               number in self._tool.windows))],
            message=message, validate=error, validate_live=True,
        )
        if answer is None:
            return
        item, _name, number, overwrite = answer
        windows.copy(item, number=int(number[1:]), select=True, overwrite=bool(overwrite))
        self._refresh_window_combo()
        self._load_selected_window()

    def _add_window(self, number: int | None = None) -> None:
        """A new window (*number*, or the one after the highest), shown."""
        self._tool.windows.new(select=True, number=number)
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
            if dataseries is None:  # the source, with nothing chosen (#521)
                raise RuntimeError("Choose a data series")
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
                if data is None:  # the source, with nothing chosen (#521)
                    raise RuntimeError("Choose a data array or Data Set")
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