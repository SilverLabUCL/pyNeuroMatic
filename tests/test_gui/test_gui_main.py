"""Starting the window: python -m pyneuromatic.gui."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.gui import __main__ as gui_main
from pyneuromatic.gui.app_window import NMAppWindow


def test_starts_with_one_folder_chosen(qtbot, monkeypatch):
    """One empty folder, folder0, chosen: ready to import into."""
    shown = []
    monkeypatch.setattr(NMAppWindow, "show", lambda self: shown.append(self))
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)
    from pyneuromatic.core.nm_settings import NMSettings
    monkeypatch.setattr(NMSettings, "load_user", lambda self, *a, **k: None)
    assert gui_main.main(argv=["pyneuromatic"]) == 0
    (window,) = shown
    qtbot.addWidget(window)
    window._may_quit = lambda: True
    folders = window.selection_model.manager.folders
    assert list(folders.keys()) == ["folder0"]
    assert window.selection_model.selection.get("folder") is folders["folder0"]
    # Shown: the strip's Folder lists and shows it (not just chosen)
    combo = window.selection_strip.combo("folder")
    assert [combo.itemText(i) for i in range(combo.count())
            if combo.itemText(i)] == ["folder0"]
    assert combo.currentText() == "folder0"
