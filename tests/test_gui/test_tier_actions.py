"""Tests for the Folder and Subfolder actions."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import tier_actions
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tier_actions import unique_name

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """App window with folders Demo (RecordA0, subfolder Stats_0) and Other.

    The choose dialog is replaced: each call records its options in
    ``app.dialogs`` and returns the next ``(item, name)`` from ``app.answers``
    (None, or no answers left, means Cancel). Messages are collected.
    """
    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=False)
    demo.data.new("RecordA0", nparray=np.array([1.0, 2.0]))
    demo.toolfolders.new("Stats_0").data.new("ST_w0_mean_y", nparray=np.array([1.5]))
    manager.folders.new("Other", select=False)
    manager.select_value_set(demo)

    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    actions = window.folder_actions
    window.answers = []
    window.dialogs = []
    window.messages = []

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    monkeypatch.setattr(actions, "_choose", choose)
    monkeypatch.setattr(actions, "_show_message",
                        lambda title, text, warning=False: window.messages.append(text))
    return window


def _combo_values(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_unique_name_sanitises_and_avoids_existing():
    folders = NMManager(quiet=True).folders
    folders.new("Rec_1")
    assert unique_name(folders, "my rec-1") == "my_rec_1"
    assert unique_name(folders, "Rec 1") == "Rec_1_1"
    assert unique_name(folders, "***") == folders.auto_name_next()


def test_actions_appear_in_strip_and_menu_bar(app):
    strip = app.selection_strip
    folder_button = strip.action_button("folder")
    assert folder_button.isEnabled()
    assert [a.text() for a in folder_button.menu().actions() if a.text()] == [
        "New Folder…", "Open / Import…", "Save…", "Duplicate…", "Rename…", "Delete…",
        "Show in Browser",
    ]
    assert strip.action_button("toolfolder").isEnabled()

    menus = {a.text(): a.menu() for a in app.menuBar().actions()}
    assert "Folder" in menus
    file_actions = menus["File"].actions()
    assert app.folder_actions.open_action in file_actions
    assert app.folder_actions.save_action in file_actions


def test_enabled_state_needs_items_not_a_selection(app):
    actions = app.folder_actions
    assert actions.rename_action.isEnabled()
    # Demo has a subfolder, so its actions work without one being selected
    assert app.selection_model.selection["toolfolder"] is None
    assert actions.subfolder_rename_action.isEnabled()

    app.selection_model.update(folder="Other")  # Other has no subfolders
    assert not actions.subfolder_rename_action.isEnabled()

    app.selection_model.clear()
    assert actions.rename_action.isEnabled()  # folders still exist
    assert not actions.save_action.isEnabled()
    assert not actions.subfolder_rename_action.isEnabled()


def test_new_folder_is_created_and_selected(app):
    app.answers = [(None, "Fresh")]
    app.folder_actions.new_action.trigger()
    manager = app.selection_model.manager
    assert "Fresh" in manager.folders
    assert manager.select_values["folder"] is manager.folders["Fresh"]
    folder_menu = app.selection_strip.combo("folder")
    assert folder_menu.currentText() == "Fresh"
    assert "Fresh" in _combo_values(folder_menu)


def test_dialog_checks_names_inline(qtbot):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, name_error

    folders = NMManager(quiet=True).folders
    folders.new("Demo")
    folders.new("Other")
    dialog = ChooseItemDialog(
        "Rename Folder",
        item_label="Folder",
        items=list(folders.keys()),
        current="Other",
        name_label="New name",
        default_name=lambda item: item,
        validate=lambda item, name: name_error(folders, name),
    )
    qtbot.addWidget(dialog)
    assert dialog.item() == "Other" and dialog.name() == "Other"

    dialog.item_combo.setCurrentText("Demo")
    assert dialog.name() == "Demo"  # refilled for the chosen folder

    dialog.name_edit.setText("other")
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert not dialog.error_label.isHidden()
    assert dialog.error_label.text() == "'other' already exists"

    dialog.name_edit.setText("bad name!")
    dialog.accept()
    assert "not a valid name" in dialog.error_label.text()

    dialog.name_edit.setText("Fine")
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted


def test_rename_dialog_lists_folders_starting_at_selection(app):
    app.folder_actions.rename_folder()  # cancelled
    title, options = app.dialogs[-1]
    assert title == "Rename Folder"
    assert options["items"] == ["Demo", "Other"]
    assert options["current"] == "Demo"
    assert options["default_name"]("Other") == "Other"
    # Keeping the same name is allowed; another folder's name is not
    assert options["validate"]("Other", "Other") is None
    assert options["validate"]("Other", "Demo") == "'Demo' already exists"


def test_rename_a_folder_that_is_not_selected(app):
    manager = app.selection_model.manager
    app.answers = [("Other", "Second")]
    app.folder_actions.rename_folder()
    assert list(manager.folders.keys()) == ["Demo", "Second"]
    assert manager.select_values["folder"] is manager.folders["Demo"]
    assert "Second" in _combo_values(app.selection_strip.combo("folder"))


def test_rename_folder_keeps_it_selected(app):
    app.answers = [("Demo", "Renamed")]
    app.folder_actions.rename_action.trigger()
    manager = app.selection_model.manager
    assert list(manager.folders.keys()) == ["Renamed", "Other"]
    assert manager.select_values["folder"] is manager.folders["Renamed"]
    assert app.selection_strip.combo("folder").currentText() == "Renamed"
    assert app.selection_strip.combo("data").currentText() == "RecordA0"


def test_duplicate_folder_copies_data(app):
    app.answers = [("Demo", "Demo_copy")]
    app.folder_actions.duplicate_folder()
    assert app.dialogs[-1][1]["default_name"]("Other") == "Other_copy"
    manager = app.selection_model.manager
    copy = manager.folders["Demo_copy"]
    assert manager.select_values["folder"] is copy
    assert list(copy.data.keys()) == ["RecordA0"]
    assert copy.data["RecordA0"] is not manager.folders["Demo"].data["RecordA0"]


def test_delete_folder_asks_then_selects_neighbour(app):
    manager = app.selection_model.manager
    app.folder_actions.delete_folder()  # cancelled
    assert "Demo" in manager.folders
    assert app.dialogs[-1][1]["current"] == "Demo"

    app.answers = [("Demo", None)]
    app.folder_actions.delete_folder()
    assert list(manager.folders.keys()) == ["Other"]
    assert manager.select_values["folder"] is manager.folders["Other"]

    app.answers = [("Other", None)]
    app.folder_actions.delete_folder()
    assert len(manager.folders) == 0
    assert manager.select_values["folder"] is None
    assert not app.folder_actions.delete_action.isEnabled()


def test_close_a_folder_that_is_not_selected(app):
    manager = app.selection_model.manager
    app.answers = [("Other", None)]
    app.folder_actions.delete_folder()
    assert list(manager.folders.keys()) == ["Demo"]
    assert manager.select_values["folder"] is manager.folders["Demo"]


def test_save_is_under_construction(app):
    app.folder_actions.save_action.trigger()
    assert app.messages == ["Save is under construction."]


def test_open_imports_each_file_into_a_new_folder(app, monkeypatch, tmp_path):
    def fake_reader(path, folder):
        folder.data.new("RecordA0", nparray=np.array([7.0, 8.0]))
        return folder

    def missing_reader(path, folder):
        raise ImportError("pyabf is required to read ABF files")

    monkeypatch.setattr(tier_actions, "READERS",
                        {".fake": fake_reader, ".abf": missing_reader})
    app.folder_actions.open_files([
        str(tmp_path / "my rec-1.fake"),
        str(tmp_path / "cell.abf"),
        str(tmp_path / "notes.txt"),
    ])

    manager = app.selection_model.manager
    assert list(manager.folders.keys()) == ["Demo", "Other", "my_rec_1"]
    assert manager.select_values["folder"] is manager.folders["my_rec_1"]
    assert app.selection_strip.combo("data").currentText() == "RecordA0"
    message = app.messages[0].splitlines()
    assert message == [
        "cell.abf: pyabf is required to read ABF files",
        "notes.txt: unsupported file type",
    ]


def test_subfolder_duplicate_rename_and_delete(app):
    actions = app.folder_actions
    strip = app.selection_strip
    demo = app.selection_model.manager.folders["Demo"]
    strip.combo("toolfolder").setCurrentText("Stats_0")

    app.answers = [("Stats_0", "Stats_copy")]
    actions.duplicate_subfolder()
    assert list(demo.toolfolders.keys()) == ["Stats_0", "Stats_copy"]
    assert strip.combo("toolfolder").currentText() == "Stats_copy"

    app.answers = [("Stats_copy", "Stats_1")]
    actions.rename_subfolder()
    assert list(demo.toolfolders.keys()) == ["Stats_0", "Stats_1"]
    assert strip.combo("toolfolder").currentText() == "Stats_1"
    # Stats2 lists the renamed results folder
    stats2 = app.stats_tab.stats2
    assert [stats2.folder_combo.itemText(i) for i in range(stats2.folder_combo.count())] == [
        "Stats_0", "Stats_1",
    ]

    app.answers = [("Stats_1", None)]
    actions.delete_subfolder()
    assert list(demo.toolfolders.keys()) == ["Stats_0"]
    assert strip.combo("toolfolder").currentText() == "—"

    # Delete a subfolder without selecting it first
    app.answers = [("Stats_0", None)]
    actions.delete_subfolder()
    assert len(demo.toolfolders) == 0
    assert not actions.subfolder_delete_action.isEnabled()


def test_copy_name_uses_prefix_and_numbers_it():
    from pyneuromatic.gui.tier_actions import copy_name

    folder = NMManager(quiet=True).folders.new("Demo")
    data = folder.data
    data.new("RecordA0", nparray=np.array([1.0]))
    assert copy_name(data, "RecordA0") == "C_RecordA0"
    data.new("c_recorda0", nparray=np.array([1.0]))  # names are case insensitive
    assert copy_name(data, "RecordA0") == "C1_RecordA0"
    data.new("C1_RecordA0", nparray=np.array([1.0]))
    assert copy_name(data, "RecordA0") == "C2_RecordA0"
    assert copy_name(data, "RecordA0", prefix="Avg") == "Avg_RecordA0"


def test_folder_and_subfolder_show_in_browser(app):
    actions = app.folder_actions
    folder = app.selection_model.selection["folder"]
    actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    tree = app.browser_widget.tree
    assert tree.currentIndex().internalPointer() is folder
    assert tree.isExpanded(tree.currentIndex())  # open, showing its contents
    assert not actions.subfolder_browse_action.isEnabled()  # no subfolder selected
    subfolder = next(iter(folder.toolfolders.values()))
    app.selection_model.update(toolfolder=subfolder)
    assert actions.subfolder_browse_action.isEnabled()
    actions.subfolder_browse_action.trigger()
    assert tree.currentIndex().internalPointer() is subfolder
    assert tree.isExpanded(tree.currentIndex())
    menu = app.selection_strip.action_button("toolfolder").menu()
    assert [a.text() for a in menu.actions() if a.text()][-1] == "Show in Browser"
