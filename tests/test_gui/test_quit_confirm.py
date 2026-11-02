"""Tests for asking before quitting (#445)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.settings_dialog import SettingsDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def asked(monkeypatch):
    """The quit question's answer (set by each test) and how often it was asked."""
    state = {"answer": ("cancel", False), "count": 0}

    def ask(self):
        state["count"] += 1
        return state["answer"]

    monkeypatch.setattr(NMAppWindow, "_ask_quit", ask)
    return state


def _window(qtbot, with_folder=True):
    manager = NMManager(quiet=True)
    if with_folder:
        folder = manager.folders.new("Demo")
        folder.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.show()
    return window


def test_cancel_keeps_the_window_open(qtbot, asked):
    window = _window(qtbot)
    assert not window.close()
    assert window.isVisible() and asked["count"] == 1


def test_quit_closes(qtbot, asked):
    window = _window(qtbot)
    asked["answer"] = ("quit", False)
    assert window.close()
    assert not window.isVisible()


def test_an_empty_session_quits_without_asking(qtbot, asked):
    window = _window(qtbot, with_folder=False)
    assert window.close() and asked["count"] == 0


def test_commands_since_the_start_are_something_to_lose(qtbot, asked):
    window = _window(qtbot, with_folder=False)
    assert not window.has_session_to_lose()  # only "nm = NMManager()"
    window.stats_tab.set_config_option("overwrite", True)  # a command
    assert window.has_session_to_lose()
    assert not window.close()


def test_quit_menu_item_asks_too(qtbot, asked):
    window = _window(qtbot)
    quit_action = next(a for a in window.menuBar().actions()[0].menu().actions()
                       if a.text() == "Quit")
    quit_action.trigger()
    assert asked["count"] == 1 and window.isVisible()


def test_dont_ask_again_turns_the_setting_off(qtbot, asked):
    window = _window(qtbot)
    asked["answer"] = ("quit", True)
    assert window.close()
    assert window._manager.settings.confirm_quit is False

    other = _window(qtbot)
    other._manager.settings.confirm_quit = False
    asked["count"] = 0
    assert other.close() and asked["count"] == 0


def test_dont_ask_again_is_ignored_on_cancel(qtbot, asked):
    window = _window(qtbot)
    asked["answer"] = ("cancel", True)
    assert not window.close()
    assert window._manager.settings.confirm_quit is True


def test_export_history_then_quit(qtbot, asked, monkeypatch, tmp_path):
    window = _window(qtbot)
    path = tmp_path / "history.py"
    monkeypatch.setattr(NMAppWindow, "_export_history_path", lambda self: str(path))
    asked["answer"] = ("export", False)
    assert window.close()
    script = path.read_text()
    assert "nm = NMManager()" in script


def test_cancelling_the_export_keeps_the_window_open(qtbot, asked, monkeypatch):
    window = _window(qtbot)
    monkeypatch.setattr(NMAppWindow, "_export_history_path", lambda self: "")
    asked["answer"] = ("export", False)
    assert not window.close()
    assert window.isVisible()


def test_export_history_menu_item(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot)
    path = tmp_path / "session.py"
    monkeypatch.setattr(NMAppWindow, "_export_history_path", lambda self: str(path))
    window.export_history_action.trigger()
    assert path.exists()


def test_setting_in_settings_general(qtbot):
    manager = NMManager(quiet=True)
    dialog = SettingsDialog(manager.settings)
    qtbot.addWidget(dialog)
    assert dialog.confirm_quit_checkbox.isChecked()  # on by default
    dialog.confirm_quit_checkbox.setChecked(False)
    assert manager.settings.confirm_quit is False
