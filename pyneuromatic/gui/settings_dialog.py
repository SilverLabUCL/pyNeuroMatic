"""The Settings window: general pyNeuroMatic settings, the Plot's trace style,
then one page per tool.

General settings (``manager.settings``) are saved to the user's settings file
as they change. A tool's page lists its options; this is the one place they
are edited (each tool's ⋯ button, by its title, opens its page).
"""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_settings import (
    NMSettings,
    TABLE_NAN_BLANK,
    TABLE_NAN_GREY,
    TABLE_NAN_TEXT,
    TRACE_LINES,
    TRACE_SYMBOLS,
    RUN_DISPLAYS,
)

LINE_LABELS = {"solid": "Solid", "dash": "Dashed", "dot": "Dotted",
                "dash-dot": "Dash-dot", "none": "None (symbols only)"}
_PLOT_KEYS = ("plot_trace_color", "plot_trace_width", "plot_trace_line", "plot_trace_symbol",
              "plot_trace_symbol_size", "plot_overlay_color", "plot_overlay_width")

# While a tool runs (#468); the tool panel's Display control says it briefly
_RUN_DISPLAY_LABELS = {"off": "Off (fastest)", "each": "Show each epoch, then wait",
                       "ask": "Show each epoch, then ask to go on"}

_NAN_LABELS = {
    TABLE_NAN_TEXT: "NaN",
    TABLE_NAN_GREY: "NaN, in grey",
    TABLE_NAN_BLANK: "blank (hover to see NaN)",
}


class SettingsDialog(QtWidgets.QDialog):
    """Pages down the left: General, then each tool (e.g. Stats).

    *tool_pages* is a list of (title, tool tab) where the tab has
    ``config_options()`` (key, label, value, tip) and
    ``set_config_option(key, value)``.
    """

    # A general setting changed (e.g. how tables show NaN)
    settings_changed = QtCore.pyqtSignal()

    def __init__(self, settings: NMSettings, tool_pages: list[tuple[str, object]] | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = settings
        self.page_list = QtWidgets.QListWidget(self)
        self.page_list.setFixedWidth(120)
        self.pages = QtWidgets.QStackedWidget(self)
        self.page_list.currentRowChanged.connect(self.pages.setCurrentIndex)

        # General
        general = QtWidgets.QWidget(self)
        general_layout = QtWidgets.QVBoxLayout(general)

        # Table: how tables show values
        table_page = QtWidgets.QWidget(self)
        table_layout = QtWidgets.QVBoxLayout(table_page)
        tables = QtWidgets.QGroupBox(table_page)  # its label says what it is
        tables_layout = QtWidgets.QVBoxLayout(tables)
        tables_layout.addWidget(QtWidgets.QLabel("Show NaN values as:", tables))
        self.nan_buttons: dict[str, QtWidgets.QRadioButton] = {}
        group = QtWidgets.QButtonGroup(self)
        for mode, label in _NAN_LABELS.items():
            button = QtWidgets.QRadioButton(label, tables)
            button.setChecked(settings.table_nan == mode)
            button.toggled.connect(
                lambda checked, m=mode: checked and self._set("table_nan", m)
            )
            group.addButton(button)
            tables_layout.addWidget(button)
            self.nan_buttons[mode] = button
        table_layout.addWidget(tables)
        table_layout.addStretch(1)

        self.confirm_quit_checkbox = QtWidgets.QCheckBox("Ask before quitting", general)
        self.confirm_quit_checkbox.setToolTip(
            "When quitting with data or history, ask first (and offer to export "
            "the history as a script)")
        self.confirm_quit_checkbox.setChecked(settings.confirm_quit)
        self.confirm_quit_checkbox.toggled.connect(
            lambda checked: self._set("confirm_quit", checked))
        general_layout.addWidget(self.confirm_quit_checkbox)
        self.import_options_checkbox = QtWidgets.QCheckBox("Ask for import options", general)
        self.import_options_checkbox.setToolTip(
            "Before importing a file, choose its folder name, prefix, channels, "
            "epochs and X range")
        self.import_options_checkbox.setChecked(settings.ask_import_options)
        self.import_options_checkbox.toggled.connect(
            lambda checked: self._set("ask_import_options", checked))
        general_layout.addWidget(self.import_options_checkbox)

        general_layout.addSpacing(8)
        runs = QtWidgets.QGroupBox(general)  # titled by the heading above it
        runs_form = QtWidgets.QFormLayout(runs)
        self.run_display_combo = QtWidgets.QComboBox(runs)
        for key in RUN_DISPLAYS:
            self.run_display_combo.addItem(_RUN_DISPLAY_LABELS[key], key)
        self.run_display_combo.setCurrentIndex(
            self.run_display_combo.findData(settings.run_display))
        self.run_display_combo.setToolTip(
            "While a tool runs, show each epoch in the Plot with the tool's "
            "measurements (also in each tool's panel, by Run)")
        self.run_display_combo.currentIndexChanged.connect(self._on_run_display_changed)
        runs_form.addRow("Progress", self.run_display_combo)
        self.run_delay_box = QtWidgets.QDoubleSpinBox(runs)
        self.run_delay_box.setRange(0.0, 10.0)
        self.run_delay_box.setSingleStep(0.1)
        self.run_delay_box.setDecimals(1)
        self.run_delay_box.setSuffix(" s")
        self.run_delay_box.setValue(settings.run_delay)
        self.run_delay_box.setToolTip(
            "Wait after showing each epoch")
        self.run_delay_box.valueChanged.connect(
            lambda value: self._set("run_delay", float(value)))
        runs_form.addRow("Delay", self.run_delay_box)
        self._runs_form = runs_form
        self._show_run_delay()
        general_layout.addLayout(self._section("Tool runs", runs, general))
        general_layout.addStretch(1)
        self._add_page("General", general)
        self._add_page("Plot", self._plot_page())
        self._add_page("Table", table_page)

        # One page per tool: its options
        self.tool_checkboxes: dict[str, dict[str, QtWidgets.QCheckBox]] = {}
        self.tool_spinboxes: dict[str, dict[str, QtWidgets.QSpinBox]] = {}
        for title, tab in tool_pages or []:
            page = QtWidgets.QWidget(self)
            page_layout = QtWidgets.QVBoxLayout(page)
            boxes = {}
            spins = {}
            for option in tab.config_options():
                key, label, value, tip = option[:4]
                if isinstance(value, bool):
                    box = QtWidgets.QCheckBox(label, page)
                    box.setChecked(value)
                    box.setToolTip(tip)
                    box.toggled.connect(
                        lambda checked, t=tab, k=key: t.set_config_option(k, checked)
                    )
                    page_layout.addWidget(box)
                    boxes[key] = box
                    continue
                # A number: (key, label, value, tip, (minimum, maximum))
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(label, page))
                spin = QtWidgets.QSpinBox(page)
                spin.setRange(*option[4])
                spin.setValue(value)
                spin.setToolTip(tip)
                spin.valueChanged.connect(
                    lambda number, t=tab, k=key: t.set_config_option(k, number))
                row.addWidget(spin)
                row.addStretch(1)
                page_layout.addLayout(row)
                spins[key] = spin
            page_layout.addStretch(1)
            self.tool_checkboxes[title] = boxes
            self.tool_spinboxes[title] = spins
            self._add_page(title, page)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, self
        )
        buttons.rejected.connect(self.reject)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.page_list)
        body.addWidget(self.pages, stretch=1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)
        self.page_list.setCurrentRow(0)
        self.resize(460, 280)

    def page_titles(self) -> list[str]:
        return [self.page_list.item(i).text() for i in range(self.page_list.count())]

    def show_page(self, title: str) -> None:
        """Show the page called *title* (e.g. "Stats"), if there is one."""
        if title in self.page_titles():
            self.page_list.setCurrentRow(self.page_titles().index(title))

    def current_page(self) -> str:
        return self.page_titles()[self.page_list.currentRow()]

    @staticmethod
    def _section(text: str, box: QtWidgets.QWidget,
                 parent: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        """A bold heading in the dialog's font size, just above *box* (a group
        box's own title is drawn small on macOS; the style's spacing between
        a label and a group box is large, so it's set here)."""
        label = QtWidgets.QLabel(text, parent)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        section = QtWidgets.QVBoxLayout()
        section.setSpacing(4)
        section.addWidget(label)
        section.addWidget(box)
        return section

    def _on_run_display_changed(self, _index: int) -> None:
        self._set("run_display", self.run_display_combo.currentData())
        self._show_run_delay()

    def _show_run_delay(self) -> None:
        """The delay is for "each", and "ask" after Continue All."""
        self._runs_form.setRowVisible(
            self.run_delay_box, self.run_display_combo.currentData() != "off")

    # --- Plot: how the traces are drawn (#465) --------------------------------

    def _plot_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        self.plot_controls: dict[str, QtWidgets.QWidget] = {}

        current = QtWidgets.QGroupBox(page)  # titled by _section
        form = QtWidgets.QFormLayout(current)
        form.addRow("Colour", self._color_button("plot_trace_color", current))
        form.addRow("Line width", self._width_box("plot_trace_width", current))
        form.addRow("Line", self._choice_box("plot_trace_line", TRACE_LINES, LINE_LABELS,
                                             current))
        form.addRow("Symbol", self._choice_box(
            "plot_trace_symbol", TRACE_SYMBOLS, {name: name.capitalize() for name in TRACE_SYMBOLS},
            current))
        form.addRow("Symbol size", self._width_box("plot_trace_symbol_size", current))
        layout.addLayout(self._section("Current trace", current, page))
        layout.addSpacing(8)

        overlay = QtWidgets.QGroupBox(page)  # titled by _section
        form = QtWidgets.QFormLayout(overlay)
        form.addRow("Colour", self._color_button("plot_overlay_color", overlay))
        form.addRow("Line width", self._width_box("plot_overlay_width", overlay))
        layout.addLayout(self._section("Overlay traces", overlay, page))

        self.plot_defaults_button = QtWidgets.QPushButton("Default Style", page)
        self.plot_defaults_button.setAutoDefault(False)
        self.plot_defaults_button.setToolTip("Black current traces, grey overlays")
        self.plot_defaults_button.clicked.connect(self.default_plot_style)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.plot_defaults_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        self._show_plot_values()
        return page

    def _color_button(self, key: str, parent: QtWidgets.QWidget) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(parent)
        button.setAutoDefault(False)  # Return closes Settings, not open a picker
        button.setToolTip("Choose a colour")
        button.clicked.connect(lambda: self._pick_color(key))
        self.plot_controls[key] = button
        return button

    def _width_box(self, key: str, parent: QtWidgets.QWidget) -> QtWidgets.QDoubleSpinBox:
        spec = self._settings._schema[key]
        box = QtWidgets.QDoubleSpinBox(parent)
        box.setRange(spec["min"], spec["max"])
        box.setSingleStep(0.5)
        box.setDecimals(1)
        box.valueChanged.connect(lambda value: self._set(key, float(value)))
        self.plot_controls[key] = box
        return box

    def _choice_box(self, key: str, names, labels: dict, parent: QtWidgets.QWidget):
        box = QtWidgets.QComboBox(parent)
        for name in names:
            box.addItem(labels[name], name)
        box.currentIndexChanged.connect(lambda _i: self._set(key, box.currentData()))
        self.plot_controls[key] = box
        return box

    def _show_plot_values(self) -> None:
        """The controls show the settings (without setting them again)."""
        for key in _PLOT_KEYS:
            control, value = self.plot_controls[key], getattr(self._settings, key)
            control.blockSignals(True)
            if isinstance(control, QtWidgets.QPushButton):
                swatch = QtGui.QPixmap(28, 14)
                swatch.fill(QtGui.QColor(value))
                control.setIcon(QtGui.QIcon(swatch))
                control.setIconSize(swatch.size())
                control.setText(value)
            elif isinstance(control, QtWidgets.QDoubleSpinBox):
                control.setValue(value)
            else:
                control.setCurrentIndex(control.findData(value))
            control.blockSignals(False)

    def _choose_color(self, initial: str) -> str | None:
        """A colour from the colour picker, or None if cancelled (replaced in tests)."""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(initial), self, "Trace colour")
        return color.name() if color.isValid() else None

    def _pick_color(self, key: str) -> None:
        color = self._choose_color(getattr(self._settings, key))
        if color is not None:
            self._set(key, color)
            self._show_plot_values()

    def default_plot_style(self) -> None:
        """Default Style: the trace settings back to their defaults."""
        for key in _PLOT_KEYS:
            setattr(self._settings, key, self._settings._schema[key]["default"])
        self._settings.save_user()
        self._show_plot_values()
        self.settings_changed.emit()

    def _add_page(self, title: str, page: QtWidgets.QWidget) -> None:
        self.page_list.addItem(title)
        self.pages.addWidget(page)

    def _set(self, key: str, value) -> None:
        setattr(self._settings, key, value)
        self._settings.save_user()  # only if loaded from the user's file
        self.settings_changed.emit()


__all__ = ["LINE_LABELS", "SettingsDialog"]
