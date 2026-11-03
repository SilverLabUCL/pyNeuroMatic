"""Tests for the import options prompt (#456)."""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_folder import file_info
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.import_dialog import ImportOptionsDialog, info_summary, parse_epochs
from pyneuromatic.gui.settings_dialog import SettingsDialog
from pyneuromatic.gui.tier_actions import FolderActions

pytestmark = pytest.mark.gui

AXGD = Path(__file__).parents[1] / "test_io" / "fixtures" / "Vers6_060523 004.axgd"
EPOCHS = list(range(19))


def test_parse_epochs():
    assert parse_epochs("", EPOCHS) is None and parse_epochs("all", EPOCHS) is None
    assert parse_epochs("0-18", EPOCHS) is None  # every epoch
    assert parse_epochs("0-9", EPOCHS) == range(0, 10)
    assert parse_epochs("5", EPOCHS) == range(5, 6)
    assert parse_epochs("0-2, 12, 15-16", EPOCHS) == [0, 1, 2, 12, 15, 16]
    for bad in ("x", "3-1", "0-30", ","):
        with pytest.raises(ValueError):
            parse_epochs(bad, EPOCHS)


@pytest.fixture
def dialog(qtbot):
    folders = NMManager(quiet=True).folders
    dialog = ImportOptionsDialog(AXGD, file_info(str(AXGD)), folders)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_shows_the_file(dialog):
    assert dialog.windowTitle() == "Import Vers6_060523 004.axgd"
    assert dialog.format_label.text() == "AxoGraph X (.axgd)"  # from its header
    assert dialog.summary_label.text().startswith("3 channels · 2 epochs (0-1) · 79840 points")
    assert dialog.name_edit.text() == "Vers6_060523_004"
    assert dialog.prefix_edit.text() == "Record"
    assert list(dialog.channel_checkboxes) == ["A", "B", "C"]
    assert dialog.channel_checkboxes["A"].text() == "A  Membrane Voltage-1 (mV)"


def test_defaults_give_no_options(dialog):
    assert dialog.options() == {}


def test_changed_entries_give_options(dialog):
    dialog.name_edit.setText("Cell1")
    dialog.prefix_edit.setText("Rec")
    dialog.channel_checkboxes["B"].setChecked(False)
    dialog.epochs_edit.setText("1")
    dialog.xbgn_edit.setText("10")
    dialog.dataseries_checkbox.setChecked(False)
    assert dialog.options() == {
        "name": "Cell1", "prefix": "Rec", "channels": ["A", "C"],
        "epochs": range(1, 2), "xrange": (10.0, None), "make_dataseries": False,
    }


@pytest.mark.parametrize("entry, text, message", [
    ("name_edit", "bad name", "Folder name"),
    ("prefix_edit", "", "Prefix"),
    ("epochs_edit", "7", "epochs not in the file"),
    ("xend_edit", "abc", "X end"),
])
def test_bad_entries_keep_the_prompt_open(dialog, entry, text, message):
    getattr(dialog, entry).setText(text)
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert message in dialog.error_label.text()


def test_no_channel_chosen(dialog):
    for box in dialog.channel_checkboxes.values():
        box.setChecked(False)
    dialog.accept()
    assert "choose at least one" in dialog.error_label.text()


# --- Open / Import ---------------------------------------------------------------

@pytest.fixture
def app(qtbot):
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    return window


def test_open_asks_then_imports_with_the_options(app, monkeypatch):
    asked = []

    def ask(self, path, info):
        asked.append((path.name, info_summary(info)))
        return {"name": "Cell1", "channels": ["C"]}

    monkeypatch.setattr(FolderActions, "_ask_import_options", ask)
    app.folder_actions.open_files([str(AXGD)])
    manager = app.selection_model.manager
    assert asked[0][0] == "Vers6_060523 004.axgd"
    assert list(manager.folders.keys()) == ["Cell1"]
    assert list(manager.folders["Cell1"].data.keys()) == ["RecordA0", "RecordA1"]
    commands = [e["command"] for e in manager.command_history.buffer
                if e["command"].startswith("nm.folders")]
    assert commands == ["nm.folders.import_file(%r, name='Cell1', channels=['C'])" % str(AXGD)]


def test_cancel_skips_the_file(app, monkeypatch):
    monkeypatch.setattr(FolderActions, "_ask_import_options", lambda self, path, info: None)
    app.folder_actions.open_files([str(AXGD)])
    assert len(app.selection_model.manager.folders) == 0
    assert not getattr(app, "messages", [])  # cancelling isn't a problem


def test_setting_off_imports_without_asking(app, monkeypatch):
    def ask(self, path, info):
        raise AssertionError("asked")

    monkeypatch.setattr(FolderActions, "_ask_import_options", ask)
    app.selection_model.manager.settings.ask_import_options = False
    app.folder_actions.open_files([str(AXGD)])
    assert list(app.selection_model.manager.folders.keys()) == ["Vers6_060523_004"]


def test_setting_in_settings_general(qtbot):
    manager = NMManager(quiet=True)
    dialog = SettingsDialog(manager.settings)
    qtbot.addWidget(dialog)
    assert dialog.import_options_checkbox.isChecked()  # on by default
    dialog.import_options_checkbox.setChecked(False)
    assert manager.settings.ask_import_options is False
