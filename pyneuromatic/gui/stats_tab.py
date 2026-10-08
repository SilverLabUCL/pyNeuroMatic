"""Stats1 tool tab backed by the NMToolStats analysis engine."""
from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
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
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.gui.selection_model import SelectionModel
from pyneuromatic.gui.tool_tab import ToolTabWidget


class StatsToolTab(ToolTabWidget):
    """Single-window Stats1 editor and runner using the project's Stats tool."""

    NEW_WINDOW_ACTION = "__new_stats_window__"
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

        self.parameters_group.setTitle("Stats1")
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addStretch(1)
        self.config_checkbox = QtWidgets.QCheckBox("Config", self)
        mode_row.addWidget(self.config_checkbox)
        self.parameter_layout.addLayout(mode_row)

        self.settings_stack = QtWidgets.QStackedWidget(self)
        self.parameter_layout.addWidget(self.settings_stack)
        self.stats1_page = QtWidgets.QWidget(self.settings_stack)
        stats1_layout = QtWidgets.QVBoxLayout(self.stats1_page)
        stats1_layout.setContentsMargins(0, 0, 0, 0)

        window_row = QtWidgets.QHBoxLayout()
        self.window_combo = QtWidgets.QComboBox(self)
        self.window_combo.setObjectName("statsWindowSelector")
        self.window_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        window_row.addWidget(QtWidgets.QLabel("Window", self))
        window_row.addWidget(self.window_combo)
        self.measurement_combo = QtWidgets.QComboBox(self)
        self.measurement_combo.addItem("Choose measurement", None)
        for group in self._MEASUREMENT_GROUPS:
            if self.measurement_combo.count() > 1:
                self.measurement_combo.insertSeparator(self.measurement_combo.count())
            for name in group:
                self.measurement_combo.addItem(name, name)
        window_row.addWidget(self.measurement_combo, stretch=1)
        self.window_enabled = QtWidgets.QCheckBox("On", self)
        window_row.addWidget(self.window_enabled)
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

        self.tool_config_list = QtWidgets.QListWidget(self)
        self.tool_config_list.setObjectName("statsToolConfigurationList")
        self._config_keys = ("ignore_nans", "overwrite")
        self._config_labels = {
            "ignore_nans": "Ignore NaNs",
            "overwrite": "Overwrite Stats tool folder",
        }
        self._populate_config_list()
        self.settings_stack.addWidget(self.stats1_page)
        self.settings_stack.addWidget(self.tool_config_list)
        self.settings_stack.setCurrentWidget(self.stats1_page)

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
        self.config_checkbox.toggled.connect(self._show_configuration)
        self.tool_config_list.itemChanged.connect(self._on_config_item_changed)
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

    @property
    def stats_tool(self) -> NMToolStats:
        return self._tool

    def _refresh_window_combo(self) -> None:
        selected_name = self._tool.windows.selected_name
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for window in self._tool.windows:
            self.window_combo.addItem(window.name, window.name)
        if self.window_combo.count():
            self.window_combo.insertSeparator(self.window_combo.count())
        self.window_combo.addItem("New Window...", self.NEW_WINDOW_ACTION)
        index = self.window_combo.findData(selected_name)
        if index < 0 and self.window_combo.count() > 1:
            index = 0
        self.window_combo.setCurrentIndex(index)
        self.window_combo.blockSignals(False)
        if index >= 0 and self.window_combo.currentData() != self.NEW_WINDOW_ACTION:
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

    def _show_configuration(self, enabled: bool) -> None:
        self.settings_stack.setCurrentWidget(
            self.tool_config_list if enabled else self.stats1_page
        )

    def _populate_config_list(self) -> None:
        for key in self._config_keys:
            item = QtWidgets.QListWidgetItem(self._config_labels[key])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(
                item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            self.tool_config_list.addItem(item)
        self._sync_config_list()

    def _sync_config_list(self) -> None:
        self.tool_config_list.blockSignals(True)
        for index, key in enumerate(self._config_keys):
            checked = bool(getattr(self._tool, key))
            self.tool_config_list.item(index).setCheckState(
                QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
            )
        self.tool_config_list.blockSignals(False)

    def _on_config_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        self._set_tool_option(key, checked)

    def _set_tool_option(self, key: str, value: bool) -> None:
        setattr(self._tool, key, value)
        config = self._tool.config
        if config is not None and key in config._schema:
            setattr(config, key, value)
        if key in self._config_keys:
            self._sync_config_list()

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
        return f"{value:g}"

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
            window._win_set(
                {
                    "on": self.window_enabled.isChecked(),
                    "func": {"name": measurement, **params} if measurement else {},
                    "xbgn": xbgn,
                    "xend": xend,
                    "bsln_on": self.bsln_checkbox.isChecked(),
                    "bsln_func": self.bsln_combo.currentData(),
                    "bsln_xbgn": bsln_xbgn,
                    "bsln_xend": bsln_xend,
                },
                quiet=True,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.show_warning(f"Invalid {measurement} parameter: {error}")
            return

    def _add_window(self) -> None:
        window = self._tool.windows.new()
        self._tool.windows.selected_name = window.name
        self._refresh_window_combo()
        self._load_selected_window()

    def _related_dataseries(self, data: NMData, context):
        dataseries = data._dataseries
        if dataseries is not None:
            return dataseries
        parsed_name = nmu.parse_data_name(data.name)
        if parsed_name is None:
            return None
        prefix, _, _ = parsed_name
        return context.dataseries.get(prefix)

    def _selected_targets(self) -> list[dict[str, NMObject]]:
        selection = self.selection
        data = selection.get("data")
        dataseries = selection.get("dataseries")
        channel = selection.get("channel")
        selected_set = selection.get("set")
        selected_group = selection.get("group")
        operator = selection.get("group_operator")

        if isinstance(dataseries, NMDataSeries):
            if not isinstance(channel, NMChannel):
                raise RuntimeError("Select a channel before running Stats")
            all_epochs = list(dataseries.epochs.values())
            epoch_names = [epoch.name for epoch in all_epochs]
            set_names = None
            group_names = None
            if selected_set is not None:
                members = dataseries.epochs.sets.get_items(selected_set, get_keys=True)
                set_names = members or []
            if selected_group is not None:
                group_names = dataseries.epochs.groups.get_items(selected_group)
            target_names = self._combine_scope_names(
                epoch_names, set_names, group_names, operator
            )
            if set_names is None and group_names is None:
                epoch = selection.get("epoch")
                target_names = [epoch.name] if isinstance(epoch, NMEpoch) else []

            targets = []
            for epoch in all_epochs:
                if epoch.name not in target_names:
                    continue
                item = dataseries.get_data(channel=channel.name, epoch=epoch.name)
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

        if isinstance(data, NMData):
            folder = selection.get("folder")
            context = selection.get("toolfolder") or folder
            if context is None:
                return []
            set_names = None
            group_names = None
            if selected_set is not None:
                members = context.data.sets.get_items(selected_set, get_keys=True)
                set_names = members or []
            if selected_group is not None:
                series = self._related_dataseries(data, context)
                if series is None:
                    raise RuntimeError("The selected data has no epoch groups")
                epoch_names = series.epochs.groups.get_items(selected_group)
                parsed_by_data = {
                    item.name: nmu.parse_data_name(item.name)
                    for item in context.data.values()
                }
                group_names = [
                    name
                    for name, parsed in parsed_by_data.items()
                    if parsed is not None
                    and parsed[0] == series.name
                    and f"E{parsed[2]}" in epoch_names
                ]
            data_names = self._combine_scope_names(
                list(context.data.keys()), set_names, group_names, operator
            )
            if set_names is None and group_names is None:
                data_names = [data.name]

            targets = []
            for name in data_names:
                item = context.data.get(name)
                if isinstance(item, NMData):
                    target = self._manager_target(selection)
                    target["data"] = item
                    targets.append(target)
            return targets

        raise RuntimeError("Select a data array or a Data Series channel and epoch first")

    @staticmethod
    def _combine_scope_names(
        all_names: list[str],
        set_names: list[str] | None,
        group_names: list[str] | None,
        operator: str | None,
    ) -> list[str]:
        if set_names is not None and group_names is not None:
            if operator not in ("AND", "OR"):
                raise RuntimeError("Choose AND or OR to combine the selected Set and Group")
            allowed = (
                set(set_names).intersection(group_names)
                if operator == "AND"
                else set(set_names).union(group_names)
            )
        elif set_names is not None:
            allowed = set(set_names)
        elif group_names is not None:
            allowed = set(group_names)
        else:
            return all_names
        return [name for name in all_names if name in allowed]

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

    def run_tool(self) -> Any:
        self._save_window_controls()
        windows = list(self._tool.windows)
        if not any(window.on and window.func for window in windows):
            raise RuntimeError("Choose a measurement in an enabled Stats window first")
        for window in windows:
            if window.on and window.func:
                self._check_baseline(window)

        targets = self._selected_targets()
        if not targets:
            raise RuntimeError("The current selection has no data to analyze")
        selected_window = self._tool.windows.selected_name
        succeeded = self._tool.run_all(targets)
        self._tool.windows.selected_name = selected_window
        if not succeeded:
            raise RuntimeError("Stats analysis did not complete")

        records = self._tool.results
        if not records:
            self.show_warning("No results were produced")
        return records


__all__ = ["StatsToolTab"]