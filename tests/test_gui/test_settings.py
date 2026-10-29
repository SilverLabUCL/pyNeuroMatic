"""Tests for the Settings window and how tables show NaN."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.results_view import StatsTables

pytestmark = pytest.mark.gui

Role = QtCore.Qt.ItemDataRole


@pytest.fixture
def app(qtbot):
    """Folder Demo with data W = [1, NaN, 3], shown in the Table."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    folder.data.new("W", nparray=np.array([1.0, np.nan, 3.0]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(data=folder.data["W"], source="data")
    window.manager = manager
    return window


def _cell(app, row):
    model = app.table_panel.channel_table.model
    index = model.index(row, 1)
    brush = model.data(index, Role.ForegroundRole)
    return model.data(index, Role.DisplayRole), brush, model.data(index, Role.ToolTipRole)


def test_settings_menu_item(app):
    action = app.settings_action
    assert action.text() == "Settings…"
    assert action.menuRole() == QtGui.QAction.MenuRole.PreferencesRole
    assert action.shortcut() == QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Preferences)


def test_pages(app):
    dialog = app.settings_dialog()
    pages = [dialog.page_list.item(i).text() for i in range(dialog.page_list.count())]
    assert pages == ["General", "Stats"]
    assert dialog.nan_buttons["NaN"].isChecked()


def test_table_shows_nan_as_chosen(app):
    assert _cell(app, 1) == ("NaN", None, None)  # default
    assert _cell(app, 0)[0] == "1"

    dialog = app.settings_dialog()
    dialog.nan_buttons["NaN (grey)"].setChecked(True)
    assert app.manager.settings.table_nan == "NaN (grey)"
    text, brush, tooltip = _cell(app, 1)
    assert text == "NaN" and brush.color().name() == "#b0b0b0" and tooltip is None
    assert _cell(app, 0)[1] is None  # numbers stay black

    dialog.nan_buttons["blank"].setChecked(True)
    text, brush, tooltip = _cell(app, 1)
    assert text == "" and brush is None and tooltip == "NaN"


def test_stats_page_edits_the_stats_config(app):
    dialog = app.settings_dialog()
    boxes = dialog.tool_checkboxes["Stats"]
    assert list(boxes) == ["ignore_nans", "overwrite", "results_all_epochs"]
    assert boxes["results_all_epochs"].text() == "ST_ arrays: one point per epoch"
    boxes["results_all_epochs"].setChecked(True)
    stats = app.manager.stats
    assert stats.results_all_epochs and stats.config.results_all_epochs
    options = {key: value for key, _label, value, _tip in app.stats_tab.config_options()}
    assert options["results_all_epochs"] is True
    # Each option says what it does
    assert all(box.toolTip() for box in boxes.values())


def test_stats_dots_open_settings_at_its_page(app, monkeypatch):
    from pyneuromatic.gui.settings_dialog import SettingsDialog

    button = app.stats_tab.settings_button
    assert button.text() == "⋯" and button.menu() is None  # straight to Settings
    opened = []
    monkeypatch.setattr(SettingsDialog, "exec", lambda dialog: opened.append(dialog.current_page()))
    button.click()
    app.settings_action.trigger()  # the menu bar's: General
    assert opened == ["Stats", "General"]


def test_one_name_settings(app):
    """'Settings' in what users see; no Config, Configuration or Preferences."""
    dialog = app.settings_dialog()
    texts = [dialog.windowTitle()] + dialog.page_titles()
    texts += [w.text() for w in dialog.findChildren(QtWidgets.QAbstractButton)]
    texts += [w.toolTip() for w in dialog.findChildren(QtWidgets.QAbstractButton)]
    texts += [w.text() for w in dialog.findChildren(QtWidgets.QLabel)]

    def menu_texts(menu):
        for action in menu.actions():
            yield action.text()
            yield action.toolTip()
            if action.menu() is not None:
                yield from menu_texts(action.menu())

    for action in app.menuBar().actions():
        texts += list(menu_texts(action.menu()))
    texts.append(app.stats_tab.settings_button.toolTip())
    texts += [w.text() for w in app.stats_tab.findChildren(QtWidgets.QAbstractButton)]
    for word in ("config", "preferences"):
        assert not [t for t in texts if word in t.lower()], word


def test_changes_are_saved_once_loaded(app, tmp_path):
    path = tmp_path / "settings.toml"
    app.manager.settings.load_user(path)
    app.settings_dialog().nan_buttons["blank"].setChecked(True)
    assert path.exists()
    from pyneuromatic.core.nm_settings import NMSettings
    assert NMSettings.load(path).table_nan == "blank"


def test_not_saved_when_never_loaded(app, tmp_path, monkeypatch):
    from pyneuromatic.core.nm_settings import NMSettings
    monkeypatch.setattr(NMSettings, "user_path", staticmethod(lambda: tmp_path / "s.toml"))
    app.settings_dialog().nan_buttons["blank"].setChecked(True)
    assert not (tmp_path / "s.toml").exists()


def test_app_starts_with_the_saved_setting(qtbot):
    manager = NMManager(quiet=True)
    manager.settings.table_nan = "blank"
    folder = manager.folders.new("Demo", select=True)
    folder.data.new("W", nparray=np.array([np.nan]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(data=folder.data["W"], source="data")
    model = window.table_panel.channel_table.model
    assert model.data(model.index(0, 1), Role.DisplayRole) == ""


def test_stats_results_table_follows(qtbot):
    tables = StatsTables()
    qtbot.addWidget(tables)
    item = tables._value_item(float("nan"))
    tables.arrays_table.setRowCount(1)
    tables.arrays_table.setColumnCount(1)
    tables.arrays_table.setItem(0, 0, item)
    assert item.text() == "NaN"
    from pyneuromatic.gui import table_format
    table_format.set_nan_mode("blank")
    tables.refresh()
    assert item.text() == "" and item.toolTip() == "NaN"
    table_format.set_nan_mode("NaN (grey)")
    tables.refresh()
    assert item.text() == "NaN" and item.foreground().color().name() == "#b0b0b0"
    assert tables._value_text(float("nan")) == "NaN"  # summary text: always NaN
