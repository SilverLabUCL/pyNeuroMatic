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
    dialog = NotesDialog(notes, "Set2 (Demo / Record)")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Notes: Set2 (Demo / Record)"
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
        # "Name (Parent)": a folder's name; its folder; an epoch's... folder and
        # data series
        "folder": (demo.notes, "Demo"),
        "data": (demo.data["Single"].notes, "Single (Demo)"),
        "data_set": (demo.data.sets.notes("Mine"), "Mine (Demo)"),
        "dataseries": (series.notes, "Record (Demo)"),
        "channel": (series.channels["A"].notes, "A (Demo / Record)"),
        "epoch_set": (series.epochs.sets.notes("Set1"), "Set1 (Demo / Record)"),
        "group": (series.epochs.groups.notes(0), "Group0 (Demo / Record)"),
    }
    for key, (notes, owner) in expected.items():
        actions = getattr(app, TIERS[key])
        opened = []
        actions._open_notes = lambda n, o, choices=None, opened=opened: opened.append((n, o))
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
    app.folder_actions._open_notes = lambda n, o, choices=None: opened.append((n, o))
    _notes_action(app, "toolfolder").trigger()
    assert opened == [(demo.toolfolders["Stats_0"].notes, "Stats_0 (Demo)")]


def test_notes_without_a_choice_start_blank(app, qtbot):
    """No set or group chosen (e.g. All Epochs, All Groups): Notes… is still
    there, its window starting blank, to choose in it, as Clear… and
    Delete… (#509); greyed out only when there's none."""
    model, series = app.selection_model, app.series
    model.update(epoch_set="All", group=None)  # built-in choices
    opened = []
    for key in ("epoch_set", "group"):
        actions = getattr(app, TIERS[key])
        actions._open_notes = lambda n, o, choices=None: opened.append((n, o, choices))
        item = _notes_action(app, key)
        assert item.isEnabled(), key
        item.trigger()
    assert [(n, o, c.current) for n, o, c in opened] == [(None, None, None)] * 2
    # The window: blank, nothing to add to, until one's chosen
    dialog = NotesDialog(None, None, choices=opened[1][2])
    qtbot.addWidget(dialog)
    combo = dialog.choice_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "0"]
    assert dialog.windowTitle() == "Notes"
    assert not dialog.editor.add_button.isEnabled()
    combo.setCurrentText("0")
    assert dialog.windowTitle() == "Notes: Group0 (Demo / Record)"
    assert dialog.editor.add_button.isEnabled()
    combo.setCurrentIndex(0)  # back to blank
    assert dialog.windowTitle() == "Notes" and not dialog.editor.add_button.isEnabled()
    # No groups: nothing to choose, greyed out
    series.epochs.groups.delete_group(series.epochs.groups.group_numbers)
    app.group_actions._update_notes_actions()
    assert not _notes_action(app, "group").isEnabled()

def test_a_note_added_from_the_strip_is_in_the_browser(app, qtbot):
    """The strip's dialog and the browser's info panel show the same notes,
    in the same editor."""
    def open_notes(notes, owner, choices=None):
        dialog = NotesDialog(notes, owner, choices=choices)
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
    or data series ("Folder / Subfolder" inside one); the folder and data
    series for an epoch, channel, Epoch Set or group ("Folder / Series")."""
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
    assert notes_owner(series.epochs["E0"]) == "E0 (Demo / Record)"
    assert notes_owner(series.channels["A"]) == "A (Demo / Record)"
    assert set_owner("Set1", series.epochs) == "Set1 (Demo / Record)"
    assert set_owner("Mine", folder.data) == "Mine (Demo)"
    assert group_owner(0, series) == "Group0 (Demo / Record)"
    # The same series name in another folder: which folder's, e.g. in the history
    other = manager.folders.new("Demo2", select=False)
    other_series = other.new_dataseries("Record", n_channels=1, n_epochs=1, n_points=5)
    assert notes_owner(other_series.epochs["E0"]) == "E0 (Demo2 / Record)"
    dialog = NotesDialog(series.epochs["E0"].notes, notes_owner(series.epochs["E0"]))
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Notes: E0 (Demo / Record)"


# --- notes in the history: messages, not commands (#507) -------------------------------

def _history(app, start=0):
    QtWidgets.QApplication.processEvents()
    app.history_panel.flush()
    return [(e.kind, e.text) for e in app.history_panel.model.entries[start:]]


def test_notes_are_history_messages(app, qtbot):
    series = app.series
    dialog = NotesDialog(series.epochs.sets.notes("Set1"), "Set1 (Demo / Record)")
    qtbot.addWidget(dialog)
    editor = dialog.editor
    start = len(_history(app))
    editor._ask_note = lambda title, text="": None  # cancelled: no message
    editor.add_button.click()
    assert _history(app, start) == []
    editor._ask_note = lambda title, text="": "Responses > 10 mV"
    editor.add_button.click()
    editor._ask_note = lambda title, text="": "Responses > 12 mV"
    editor.edit_button.click()  # the new note is chosen
    editor._confirm_delete = lambda count: True
    editor.delete_button.click()
    assert _history(app, start) == [
        ("message", "Set1 (Demo / Record): added note 'Responses > 10 mV'"),
        ("message", "Set1 (Demo / Record): edited note 1: 'Responses > 12 mV'"),
        ("message", "Set1 (Demo / Record): deleted note 1: 'Responses > 12 mV'"),
    ]  # no commands


def test_a_long_note_is_cut_in_its_message(app, qtbot):
    from pyneuromatic.gui.notes_view import short_text
    text = "Responses were measured after the drug washed in; " * 3
    assert len(short_text(text)) == 60 and short_text(text).endswith("…")
    assert short_text("two\nlines") == "two lines"
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(app.series.notes, "Record (Demo)")
    start = len(_history(app))
    editor._ask_note = lambda title, txt="": text
    editor.add_button.click()
    assert _history(app, start) == [
        ("message", f"Record (Demo): added note '{short_text(text)}'")]
    assert app.series.notes[-1]["note"] == text.strip()  # in full in the notes


def test_notes_written_by_pynm_are_not_messages(app):
    """A Stats run's note (read-only) is written without a message."""
    tab = app.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    start = len(_history(app))
    tab.run_button.click()
    assert tab.stats_tool.toolfolder.notes[0]["note"].startswith("NMStats:")
    assert not [text for _kind, text in _history(app, start) if "added note" in text]


# --- notes as commands: a setting, off by default (#510) ------------------------------

@pytest.fixture
def cmd_app(qtbot):
    """As app, with commands recorded."""
    manager = NMManager(quiet=True, command_history=True)
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
    window.demo, window.series, window.manager = folder, series, manager
    return window


def _commands(app, start=0):
    return [text for kind, text in _history(app, start) if kind == "command"]


def _edit(editor):
    """Add two notes, edit the first, delete the second."""
    editor._ask_note = lambda title, text="": "first"
    editor.add_button.click()
    editor._ask_note = lambda title, text="": "second"
    editor.add_button.click()
    editor.table.selectRow(len(editor.notes) - 2)
    editor._ask_note = lambda title, text="": text + ", corrected"
    editor.edit_button.click()
    editor.table.selectRow(len(editor.notes) - 1)
    editor._confirm_delete = lambda count: True
    editor.delete_button.click()


def test_notes_as_commands_off_by_default(cmd_app, qtbot):
    settings = cmd_app.manager.settings
    assert settings.notes_as_commands is False
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(cmd_app.series.notes, "Record (Demo)")
    assert not editor.commands_checkbox.isHidden()  # shown: the app has settings
    assert not editor.commands_checkbox.isChecked()
    start = len(_history(cmd_app))
    _edit(editor)
    assert _commands(cmd_app, start) == []  # messages (#507)
    assert len([k for k, _t in _history(cmd_app, start) if k == "message"]) == 4


def test_notes_as_commands_replay(cmd_app, qtbot):
    """On: each note added, edited or deleted is a command, no message; the
    commands replayed recreate the notes, for every kind of owner."""
    demo, series, manager = cmd_app.demo, cmd_app.series, cmd_app.manager
    manager.settings.notes_as_commands = True
    owners = [demo.notes, demo.toolfolders["Stats_0"].notes, demo.data["Single"].notes,
              series.notes, series.channels["A"].notes, series.epochs["E0"].notes,
              demo.data.sets.notes("Mine"), series.epochs.sets.notes("Set1"),
              series.epochs.groups.notes(0)]
    for notes in owners:
        editor = NotesEditor()
        qtbot.addWidget(editor)
        editor.set_notes(notes, "owner")
        assert editor.commands_checkbox.isChecked()
        start = len(_history(cmd_app))
        _edit(editor)
        commands = _commands(cmd_app, start)
        path = notes.command_path
        assert commands == [f"{path}.add('first')", f"{path}.add('second')",
                            f"{path}.edit(0, 'first, corrected')", f"{path}.remove(1)"]
        assert not [k for k, _t in _history(cmd_app, start) if k == "message"]
        made = [n["note"] for n in notes]
        notes.clear()
        for command in commands:
            exec(command, {"nm": manager})
        assert [n["note"] for n in notes] == made == ["first, corrected"]


def test_the_editors_checkbox_is_the_setting(cmd_app, qtbot):
    settings = cmd_app.manager.settings
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(cmd_app.series.notes, "Record (Demo)")
    editor.commands_checkbox.setChecked(True)
    assert settings.notes_as_commands is True
    dialog = cmd_app.settings_dialog()
    qtbot.addWidget(dialog)
    assert dialog.notes_commands_checkbox.isChecked()  # Settings › General
    dialog.notes_commands_checkbox.setChecked(False)
    assert settings.notes_as_commands is False
    other = NotesDialog(cmd_app.series.notes, "Record (Demo)")
    qtbot.addWidget(other)
    assert not other.editor.commands_checkbox.isChecked()  # one setting


def test_readonly_notes_give_no_command(cmd_app, qtbot):
    cmd_app.manager.settings.notes_as_commands = True
    notes = cmd_app.series.notes
    notes.add("renumbered epochs", readonly=True)
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(notes, "Record (Demo)")
    editor.table.selectRow(0)
    assert not editor.edit_button.isEnabled() and not editor.delete_button.isEnabled()



def test_notes_as_messages_when_commands_arent_recorded(app, qtbot):
    """The setting on, but commands not recorded (e.g. history off): a
    message, so it's in the history either way."""
    import pyneuromatic.core.nm_command_history as nmch
    history = nmch.get_command_history()
    NotesEditor.settings.notes_as_commands = True
    editor = NotesEditor()
    qtbot.addWidget(editor)
    editor.set_notes(app.series.notes, "Record (Demo)")
    start = len(_history(app))
    enabled, history.enabled = history.enabled, False
    try:
        editor._ask_note = lambda title, text="": "first"
        editor.add_button.click()
    finally:
        history.enabled = enabled
    assert _history(app, start) == [("message", "Record (Demo): added note 'first'")]


def test_set_and_group_notes_commands_read_like_epochs(cmd_app):
    """sets['Set1'].notes and groups['Group0'].notes, as epochs['E0'].notes (#509)."""
    demo, series = cmd_app.demo, cmd_app.series
    base = "nm.folders['Demo'].dataseries['Record']"
    assert series.epochs["E0"].notes.command_path == f"{base}.epochs['E0'].notes"
    assert series.epochs.sets.notes("Set1").command_path == f"{base}.epochs.sets['Set1'].notes"
    assert series.epochs.groups.notes(0).command_path == f"{base}.epochs.groups['Group0'].notes"
    assert demo.data.sets.notes("Mine").command_path == "nm.folders['Demo'].data.sets['Mine'].notes"
    # The same notes, by name
    assert series.epochs.sets["Set1"].notes is series.epochs.sets.notes("Set1")
    assert series.epochs.groups["Group0"].notes is series.epochs.groups.notes(0)


# --- the Notes window's chooser: another's notes (#509) -------------------------------

def test_choose_another_in_the_notes_window(app, qtbot):
    """Each strip menu's Notes… window can switch to the chosen one's
    siblings (as a prompt's chooser), starting on the strip's choice."""
    demo, series = app.demo, app.series
    model = app.selection_model
    series.epochs.groups.assign("E1", 1)
    demo.data.sets.add("Other", [])
    model.update(data=demo.data["Single"], data_set="Mine", source="data")
    model.update(dataseries=series, source="dataseries")
    model.update(epoch_set="Set1", group=0)
    expected = {  # label, names, current
        "folder": ("Folder", ["Demo"], "Demo"),
        "data": ("Data", list(demo.data.keys()), "Single"),
        "data_set": ("Data Set", ["Mine", "Other"], "Mine"),
        "dataseries": ("Prefix", ["Record"], "Record"),
        "channel": ("Channel", ["A"], "A"),
        "epoch_set": ("Epoch Set", list(series.epochs.sets.keys()), "Set1"),
        "group": ("Group", ["0", "1"], "0"),
    }
    for key, (label, names, current) in expected.items():
        actions = getattr(app, TIERS[key])
        kind = {"folder": "folder", "toolfolder": "subfolder", "data": "data",
                "data_set": "data_set"}.get(key, "")
        choices = actions.notes_choices(kind)
        assert (choices.label, list(choices.names), choices.current) == (label, names, current), key
    # A subfolder chosen (the strip then works inside it): the subfolders
    model.update(toolfolder=demo.toolfolders["Stats_0"])
    choices = app.folder_actions.notes_choices("subfolder")
    assert (choices.label, list(choices.names), choices.current) == (
        "Subfolder", ["Stats_0"], "Stats_0")
    model.update(toolfolder=None)

    # Switching: the other's notes and title; a note goes to the one chosen
    choices = app.epoch_set_actions.notes_choices()
    dialog = NotesDialog(*app.epoch_set_actions.notes_target(), choices=choices)
    qtbot.addWidget(dialog)
    assert dialog.choice_combo.currentText() == "Set1"
    dialog.choice_combo.setCurrentText("Set0")
    assert dialog.windowTitle() == "Notes: Set0 (Demo / Record)"
    dialog.editor._ask_note = lambda title, text="": "for Set0"
    dialog.editor.add_button.click()
    assert series.epochs.sets.notes("Set0")[-1]["note"] == "for Set0"
    assert not len(series.epochs.sets.notes("Set1"))

    # Groups: their numbers, as the other Group lists
    choices = app.group_actions.notes_choices()
    assert choices.target("1") == (series.epochs.groups.notes(1), "Group1 (Demo / Record)")


def test_choose_another_epoch(app, qtbot):
    panel = app.plot_widget
    choices = panel.notes_choices()
    assert choices.label == "Epoch" and choices.names == ["E0", "E1"]
    assert choices.current == panel._current_item.name
    assert choices.target("E1")[1] == "E1 (Demo / Record)"
    dialog = NotesDialog(*panel.notes_target(), choices=choices)
    qtbot.addWidget(dialog)
    dialog.choice_combo.setCurrentText("E1")
    assert dialog.editor.notes is app.series.epochs["E1"].notes


def test_no_chooser_without_choices(qtbot):
    dialog = NotesDialog(NMNotes(), "Demo")
    qtbot.addWidget(dialog)
    assert dialog.choice_combo is None


def test_notes_available_once_there_is_one_to_choose(app):
    """Not only after the selection changes: New Groups makes groups to choose
    from (no group chosen), and Epoch Sets made elsewhere (e.g. Stats2's
    Inequality) are noticed with the content."""
    series, groups = app.series, app.series.epochs.groups
    groups.delete_group(groups.group_numbers)
    app._on_content_changed()  # as an action does after a change
    assert not _notes_action(app, "group").isEnabled()
    app.group_actions._choose = lambda title, **options: ("", None, "2")  # New Groups: 2
    app.group_actions.new_action.trigger()
    assert groups.group_numbers == [0, 1]
    assert _notes_action(app, "group").isEnabled()  # no group chosen yet
    for name in list(series.epochs.sets.keys()):
        series.epochs.sets.pop(name)
    app._on_content_changed()
    assert not _notes_action(app, "epoch_set").isEnabled()
    series.epochs.sets.add("Big", [])  # e.g. by Inequality, in Stats2
    app._on_content_changed()  # as Stats2 says (its folder actions' content_changed)
    assert _notes_action(app, "epoch_set").isEnabled()
