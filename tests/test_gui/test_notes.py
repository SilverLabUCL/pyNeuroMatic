"""Notes everywhere (#505): read-only notes, the notes editor, its dialog and
Notes… in the strip's ⋯ menus."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.notes_view import NotesDialog, NotesEditor


# --- core: read-only notes ------------------------------------------------------

def test_readonly_notes():
    notes = NMNotes()
    notes.add("how the results were made", readonly=True)
    notes.add("E7 excluded")
    assert notes[0]["readonly"] is True and "readonly" not in notes[1]
    assert notes.is_readonly(0) and not notes.is_readonly(1)
    with pytest.raises(ValueError):
        notes.edit(0, "changed")
    with pytest.raises(ValueError):
        notes.remove(0)
    notes.edit(1, "E8 excluded")
    assert notes.remove(1)["note"] == "E8 excluded"
    assert NMNotes.ok([{"date": "d", "note": "n", "readonly": True}, {"date": "d", "note": "n"}])
    assert not NMNotes.ok([{"date": "d", "note": "n", "readonly": "yes"}])
    assert not NMNotes.ok([{"date": "d", "note": "n", "other": True}])


# --- the notes editor and its dialog -------------------------------------------------

def _texts(editor):
    table = editor.table
    return [table.item(row, 1).text() for row in range(table.rowCount())]


def test_notes_editor(qtbot):
    notes = NMNotes()
    notes.add("NMStats: Record...", readonly=True)
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(notes, "Stats_0")
    changed = []
    editor.notes_changed.connect(lambda: changed.append(1))
    editor.table.selectRow(0)  # read-only: no Edit or Delete
    assert not editor.edit_button.isEnabled() and not editor.delete_button.isEnabled()
    editor._ask_note = lambda title, text="": "first"
    editor.add_button.click()
    editor._ask_note = lambda title, text="": "second"
    editor.add_button.click()
    assert _texts(editor) == ["NMStats: Record...", "first", "second"] and len(changed) == 2
    assert editor.edit_button.isEnabled()  # the new note is chosen
    date = notes[2]["date"]
    editor._ask_note = lambda title, text="": text + ", corrected"
    editor.edit_button.click()
    assert notes[2] == {"date": date, "note": "second, corrected"}
    # Several chosen: Delete (not Edit); with the read-only one: neither
    editor.table.selectAll()
    assert not editor.delete_button.isEnabled()
    editor.table.clearSelection()
    editor.table.selectRow(1)
    asked = []
    editor._confirm_delete = lambda count: asked.append(count) or True
    editor.delete_button.click()
    assert asked == [1] and _texts(editor) == ["NMStats: Record...", "second, corrected"]


def test_notes_dialog(qtbot):
    notes = NMNotes()
    dialog = NotesDialog(notes, "Set2 (Record)")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Notes: Set2 (Record)"
    assert not dialog.editor.header.isVisibleTo(dialog)  # the title says whose
    dialog.editor._ask_note = lambda title, text="": "Responses > 10 mV"
    dialog.editor.add_button.click()
    assert notes[0]["note"] == "Responses > 10 mV"


# --- Notes… in the strip's ⋯ menus -----------------------------------------------------

@pytest.fixture
def app(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=10)
    series.epochs.groups.assign("E0", 0)
    folder.data.new("Single", nparray=np.array([1.0, 2.0]))
    folder.data.sets.add("Mine", [folder.data["Single"]])
    folder.toolfolders.new("Stats_0")
    manager.select_keys = {"folder": "Demo", "dataseries": "Record", "channel": "A",
                           "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.demo, window.series = folder, series
    return window


def _notes_action(app, key):
    menu = app.selection_strip.action_button(key).menu()
    return next(a for a in menu.actions() if a.text() == "Notes…")


TIERS = {"folder": "folder_actions", "toolfolder": "folder_actions",
         "data": "data_actions", "data_set": "data_actions",
         "dataseries": "series_actions", "channel": "channel_actions",
         "epoch_set": "epoch_set_actions", "group": "group_actions"}


def test_notes_in_every_strip_menu(app):
    demo, series = app.demo, app.series
    model = app.selection_model
    model.update(data=demo.data["Single"], data_set="Mine", source="data")
    model.update(dataseries=series, source="dataseries")
    model.update(epoch_set="Set1", group=0)
    expected = {
        # "Name (Parent)": a folder's name; its folder; an epoch's... data series
        "folder": (demo.notes, "Demo"),
        "data": (demo.data["Single"].notes, "Single (Demo)"),
        "data_set": (demo.data.sets.notes("Mine"), "Mine (Demo)"),
        "dataseries": (series.notes, "Record (Demo)"),
        "channel": (series.channels["A"].notes, "A (Record)"),
        "epoch_set": (series.epochs.sets.notes("Set1"), "Set1 (Record)"),
        "group": (series.epochs.groups.notes(0), "Group 0 (Record)"),
    }
    for key, (notes, owner) in expected.items():
        actions = getattr(app, TIERS[key])
        opened = []
        actions._open_notes = lambda n, o, opened=opened: opened.append((n, o))
        item = _notes_action(app, key)
        assert item.isEnabled(), key
        item.trigger()
        assert opened == [(notes, owner)], key
        # Just before Delete (Channel: after Filter & Transform)
        texts = [a.text() for a in app.selection_strip.action_button(key).menu().actions()]
        after = texts[texts.index("Notes…") + 1]
        assert after.startswith("Delete") or key == "channel", (key, after)
    # A subfolder
    model.update(toolfolder=demo.toolfolders["Stats_0"])
    opened = []
    app.folder_actions._open_notes = lambda n, o: opened.append((n, o))
    _notes_action(app, "toolfolder").trigger()
    assert opened == [(demo.toolfolders["Stats_0"].notes, "Stats_0 (Demo)")]


def test_notes_greyed_out_without_a_choice(app):
    model = app.selection_model
    model.update(epoch_set="All", group=None)  # built-in choices: no notes
    assert not _notes_action(app, "epoch_set").isEnabled()
    assert not _notes_action(app, "group").isEnabled()
    assert not _notes_action(app, "toolfolder").isEnabled()  # no subfolder chosen
    model.update(epoch_set="Set0")
    assert _notes_action(app, "epoch_set").isEnabled()


def test_a_note_added_from_the_strip_is_in_the_browser(app, qtbot):
    """The strip's dialog and the browser's info panel show the same notes,
    in the same editor."""
    def open_notes(notes, owner):
        dialog = NotesDialog(notes, owner)
        qtbot.addWidget(dialog)
        dialog.editor.notes_changed.connect(app.series_actions.content_changed)
        dialog.editor._ask_note = lambda title, text="": "20 Hz train"
        dialog.editor.add_button.click()
    app.series_actions._open_notes = open_notes
    _notes_action(app, "dataseries").trigger()
    assert app.series.notes[-1]["note"] == "20 Hz train"
    from pyneuromatic.gui.info_panel import InfoPanel
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show_object(app.series)
    assert isinstance(panel.notes_view, NotesEditor)  # add, edit and delete there too
    table = panel.notes_table
    assert table.item(table.rowCount() - 1, 1).text() == "20 Hz train"



def test_notes_titles_are_name_then_parent(qtbot):
    """"Notes: Name (Parent)": the folder for a subfolder, data array, Data Set
    or data series ("Folder / Subfolder" inside one); the data series for an
    epoch, channel, Epoch Set or group."""
    from pyneuromatic.gui.notes_view import group_owner, notes_owner, set_owner
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=5)
    subfolder = folder.toolfolders.new("Stats_0")
    array = subfolder.data.new("ST_w0_max_y", nparray=np.array([1.0]))
    assert notes_owner(folder) == "Demo"
    assert notes_owner(folder.data["RecordA0"]) == "RecordA0 (Demo)"
    assert notes_owner(series) == "Record (Demo)"
    assert notes_owner(subfolder) == "Stats_0 (Demo)"
    assert notes_owner(array) == "ST_w0_max_y (Demo / Stats_0)"
    assert notes_owner(series.epochs["E0"]) == "E0 (Record)"
    assert notes_owner(series.channels["A"]) == "A (Record)"
    assert set_owner("Set1", series.epochs) == "Set1 (Record)"
    assert set_owner("Mine", folder.data) == "Mine (Demo)"
    assert group_owner(0, series) == "Group 0 (Record)"
    dialog = NotesDialog(series.epochs["E0"].notes, notes_owner(series.epochs["E0"]))
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Notes: E0 (Record)"
