"""The Settings window: general pyNeuroMatic settings, then one page per tool.

General settings (``manager.settings``) are saved to the user's settings file
as they change. A tool's page lists its options; this is the one place they
are edited (each tool's ⋯ button, by its title, opens its page).
"""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_settings import (
    NMSettings,
    TABLE_NAN_BLANK,
    TABLE_NAN_GREY,
    TABLE_NAN_TEXT,
)

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
        tables = QtWidgets.QGroupBox("Tables", general)
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
        general_layout.addWidget(tables)
        general_layout.addStretch(1)
        self._add_page("General", general)

        # One page per tool: its options
        self.tool_checkboxes: dict[str, dict[str, QtWidgets.QCheckBox]] = {}
        for title, tab in tool_pages or []:
            page = QtWidgets.QWidget(self)
            page_layout = QtWidgets.QVBoxLayout(page)
            boxes = {}
            for key, label, value, tip in tab.config_options():
                box = QtWidgets.QCheckBox(label, page)
                box.setChecked(value)
                box.setToolTip(tip)
                box.toggled.connect(
                    lambda checked, t=tab, k=key: t.set_config_option(k, checked)
                )
                page_layout.addWidget(box)
                boxes[key] = box
            page_layout.addStretch(1)
            self.tool_checkboxes[title] = boxes
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

    def _add_page(self, title: str, page: QtWidgets.QWidget) -> None:
        self.page_list.addItem(title)
        self.pages.addWidget(page)

    def _set(self, key: str, value) -> None:
        setattr(self._settings, key, value)
        self._settings.save_user()  # only if loaded from the user's file
        self.settings_changed.emit()


__all__ = ["SettingsDialog"]
