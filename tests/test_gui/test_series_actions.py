"""Tests for the data series (Prefix) actions."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Folder Demo: series Record (RecordA0, RecordA1, RecordB0) and arrays
    StimA0, StimA1 not yet a series; folder Other. Record is selected.

    Dialogs are replaced: ``app.answers`` feed the choose dialog (each call's
    options are kept in ``app.dialogs``); messages are collected.
    """
    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=False)
    for name in ("RecordA0", "RecordA1", "RecordB0", "StimA0", "StimA1"):
        demo.data.new(name, nparray=np.array([1.0, 2.0]))
    demo.sync_dataseries("Record")
    manager.folders.new("Other", select=False)
    manager.select_value_set(demo)

    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=demo.dataseries["Record"])
    window.answers, window.dialogs, window.messages = [], [], []
    actions = window.series_actions

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    monkeypatch.setattr(actions, "_choose", choose)
    monkeypatch.setattr(actions, "_show_message",
                        lambda title, text, warning=False: window.messages.append(text))
    window.demo = demo
    window.other = manager.folders["Other"]
    window.series = demo.dataseries["Record"]
    return window


def _menu_texts(menu):
    return [action.text() for action in menu.actions() if action.text()]


def _series_values(**changes):
    values = {
        "name": "Test", "n_channels": 2, "n_epochs": 3, "points": 2000,
        "value": 5.0, "noise": 2.0, "xstart": 0.0, "xdelta": 0.1,
        "xlabel": "Time", "xunits": "ms", "ylabel": "", "yunits": "",
    }
    values.update(changes)
    return values


def test_prefix_menu_in_strip_and_menu_bar(app):
    expected = [
        "New Series…", "Series from Data…", "Update from Data", "Copy…",
        "Rename…", "Delete…", "Edit Scales…", "Lock", "Add Epochs…",
        "Remove Epochs…", "Show in Browser…",
    ]
    button = app.selection_strip.action_button("dataseries")
    assert button.isEnabled()
    assert _menu_texts(button.menu()) == expected
    menus = {action.text(): action.menu() for action in app.menuBar().actions()}
    assert _menu_texts(menus["Series"]) == expected + ["Channel", "Epoch Set", "Group"]


def test_new_series_with_noise(app, monkeypatch):
    seen = {}

    def ask(defaults, validate):
        seen["defaults"], seen["validate"] = defaults, validate
        return _series_values()

    monkeypatch.setattr(app.series_actions, "_ask_new_series", ask)
    app.series_actions.new_action.trigger()

    series = app.demo.dataseries["Test"]
    assert list(series.channels.keys()) == ["A", "B"]
    assert list(series.epochs.keys()) == ["E0", "E1", "E2"]
    data = app.demo.data["TestB2"]
    assert data.nparray.size == 2000
    assert data.nparray.mean() == pytest.approx(5.0, abs=0.3)
    assert data.nparray.std() == pytest.approx(2.0, abs=0.3)
    assert (data.xscale.delta, data.xscale.units) == (0.1, "ms")
    assert not np.array_equal(data.nparray, app.demo.data["TestA0"].nparray)
    assert not series.locked
    assert app.selection_model.selection["dataseries"] is series
    assert app.selection_model.selection["source"] == "dataseries"

    # Defaults: a free prefix, 10 epochs, no noise; names are checked
    assert seen["defaults"]["name"] == "Record1"
    assert (seen["defaults"]["n_epochs"], seen["defaults"]["noise"]) == (10, 0)
    validate = seen["validate"]
    assert validate({"name": "Record", "n_channels": 1, "n_epochs": 1}) == (
        "Data series 'Record' already exists"
    )
    assert validate({"name": "Stim", "n_channels": 1, "n_epochs": 1}) == (
        "Data 'StimA0' already exists"
    )
    assert validate({"name": "Fresh", "n_channels": 2, "n_epochs": 5}) is None


def test_new_series_without_noise_is_constant(app, monkeypatch):
    monkeypatch.setattr(app.series_actions, "_ask_new_series",
                        lambda defaults, validate: _series_values(noise=0.0, points=4))
    app.series_actions.new_series()
    assert list(app.demo.data["TestA1"].nparray) == [5.0, 5.0, 5.0, 5.0]


def test_series_from_data(app):
    app.answers = [("Stim", None)]
    app.series_actions.from_data_action.trigger()
    assert app.dialogs[-1][1]["items"] == ["Stim"]
    stim = app.demo.dataseries["Stim"]
    assert stim.get_data("A", "E1") is app.demo.data["StimA1"]
    assert app.selection_model.selection["dataseries"] is stim
    assert not app.series_actions.from_data_action.isEnabled()  # nothing left


def test_epoch_ranges():
    from pyneuromatic.gui.tier_actions import epoch_ranges

    assert epoch_ranges([0, 1, 2, 5, 7, 8]) == "E0–E2, E5, E7–E8"
    assert epoch_ranges([3]) == "E3"
    assert epoch_ranges([]) == ""


def test_update_from_data_previews_changes(app):
    actions = app.series_actions
    actions.update_series()  # cancelled; check the preview
    title, options = app.dialogs[-1]
    assert title == "Update from Data" and options["current"] == "Record"
    assert options["message"]("Record").splitlines() == [
        "Channels: A, B (2)", "Epochs: E0–E1 (2)", "", "Up to date: no new arrays.",
    ]
    assert options["validate"]("Record", None) == "Data series 'Record' is already up to date"

    for name in ("RecordA2", "RecordB2", "RecordC0"):
        app.demo.data.new(name, nparray=np.array([1.0, 2.0]))
    summary = options["message"]("Record").splitlines()
    assert summary[3] == "Update adds 3 arrays: channel C; epochs E2."
    assert summary[4] == "RecordA2, RecordB2, RecordC0"
    assert options["validate"]("Record", None) is None

    app.series.locked = True
    assert options["message"]("Record").endswith("so it will not be updated.")
    assert "is locked" in options["validate"]("Record", None)

    app.series.locked = False
    app.answers = [("Record", None)]
    actions.update_action.trigger()
    assert list(app.series.channels.keys()) == ["A", "B", "C"]
    assert app.series.get_data("B", "E2") is app.demo.data["RecordB2"]
    assert options["message"]("Record").endswith("Up to date: no new arrays.")


def test_copy_series_here_and_to_another_folder(app):
    app.series.locked = True
    app.series_actions.copy_series()  # cancelled; check the options
    options = app.dialogs[-1][1]
    assert options["extras"] == [("To folder", ["Demo", "Other"], "Demo")]
    assert options["default_name"]("Record", "Demo") == "C_Record"
    assert options["validate"]("Record", "Stim", "Demo") == "Data 'StimA0' already exists"

    app.answers = [("Record", "C_Record", "Demo")]
    app.series_actions.copy_series()
    copy = app.demo.dataseries["C_Record"]
    assert not copy.locked  # the copy can be edited
    assert copy.get_data("B", "E0").name == "C_RecordB0"
    assert app.selection_model.selection["dataseries"] is copy

    app.answers = [("Record", "Record", "Other")]
    app.series_actions.copy_series()
    assert "RecordA1" in app.other.data and "Record" in app.other.dataseries


def test_rename_series(app):
    app.answers = [("Record", "Cell")]
    app.series_actions.rename_action.trigger()
    assert list(app.demo.dataseries.keys()) == ["Cell"]
    assert {"CellA0", "CellA1", "CellB0"} <= set(app.demo.data.keys())
    combo = app.selection_strip.combo("dataseries")
    assert combo.currentText() == "Cell"

    app.series.locked = True
    app.series_actions.rename_series()
    validate = app.dialogs[-1][1]["validate"]
    assert "is locked" in validate("Cell", "Other")
    assert validate("Cell", "Cell") == validate("Cell", "Other")  # locked either way


def test_lock_toggle_marks_series(app):
    action = app.series_actions.lock_action
    assert action.isCheckable() and not action.isChecked()
    action.trigger()
    assert app.series.locked and action.isChecked()
    combo = app.selection_strip.combo("dataseries")
    assert combo.currentText() == "🔒 Record"
    assert combo.currentData() == "Record"
    action.trigger()
    assert not app.series.locked and combo.currentText() == "Record"


def test_remove_channel_and_epoch(app):
    app.answers = [("B", None, "Keep")]
    app.series_actions.remove_channel()
    assert app.dialogs[-1][1]["context"] == ("Series", "Demo › Record")
    assert list(app.series.channels.keys()) == ["A"]
    assert "RecordB0" in app.demo.data  # kept

    app.answers = [(None, None, "E1", "E1", "Delete")]
    app.series_actions.remove_epoch()
    assert list(app.series.epochs.keys()) == ["E0"]
    assert "RecordA1" not in app.demo.data

    app.series.locked = True
    app.series_actions.remove_epoch()
    assert "is locked" in app.dialogs[-1][1]["validate"](None, None, "E0", "E0", "Keep")


def test_remove_a_range_of_epochs(app, monkeypatch):
    actions = app.series_actions
    monkeypatch.setattr(actions, "_ask_new_series",
                        lambda defaults, validate: _series_values(
                            name="Long", n_channels=1, n_epochs=1000, points=3, noise=0.0))
    actions.new_series()
    series = app.demo.dataseries["Long"]
    app.selection_model.update(epoch=series.epochs["E10"])

    actions.remove_epoch()  # cancelled; check the options
    options = app.dialogs[-1][1]
    from_spin, to_spin = options["extras"][0][1], options["extras"][1][1]
    assert (from_spin.minimum, from_spin.maximum, from_spin.value) == (0, 999, 10)
    assert to_spin.value == 10
    assert options["message"](None, "E10", "E999", "Keep") == "Removes 990 epochs (E10–E999)."
    assert options["message"](None, "E5", "E5", "Keep") == "Removes 1 epoch (E5)."
    assert options["validate"](None, None, "E20", "E10", "Keep") == "To must not be before From"

    # Keep only the first 10 recordings
    app.answers = [(None, None, "E10", "E999", "Delete")]
    actions.remove_epoch()
    assert list(series.epochs.keys()) == [f"E{i}" for i in range(10)]
    assert "LongA9" in app.demo.data and "LongA10" not in app.demo.data


def test_delete_series(app):
    app.series.locked = True
    app.series_actions.delete_series()
    assert "is locked" in app.dialogs[-1][1]["validate"]("Record", None, "Keep")

    app.series.locked = False
    app.answers = [("Record", None, "Keep")]
    app.series_actions.delete_action.trigger()
    assert "Record" not in app.demo.dataseries
    assert "RecordA0" in app.demo.data

    app.demo.sync_dataseries("Stim")
    app.answers = [("Stim", None, "Delete")]
    app.series_actions.delete_series()
    assert "StimA0" not in app.demo.data


def test_choose_in_browser_selects_series(app):
    app.series_actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    assert app.browser_widget.tree.currentIndex().internalPointer() is app.series


def test_edit_scales_dialog(qtbot):
    from pyneuromatic.gui.tier_actions import EditScalesDialog

    dialog = EditScalesDialog(
        {"xstart": "0", "xdelta": None, "xlabel": "Time", "xunits": "ms"},
        [("A", "Vm", "mV"), ("B", None, "pA")],
        context=("Series", "Demo › Record"),
    )
    qtbot.addWidget(dialog)
    assert dialog.x_edits["xdelta"].text() == ""
    assert dialog.x_edits["xdelta"].placeholderText() == "(mixed)"
    assert dialog.changes() == {}  # nothing touched, mixed fields kept

    dialog.x_edits["xunits"].setText("s")
    dialog.y_edits["B"]["label"].setText("Current")
    assert dialog.changes() == {"xunits": "s", "y": {"B": {"label": "Current"}}}

    dialog.x_edits["xdelta"].setText("abc")
    dialog.accept()
    assert dialog.error_label.text() == "Start and Delta must be numbers"
    dialog.x_edits["xdelta"].setText("0")
    dialog.accept()
    assert dialog.error_label.text() == "Delta must not be 0"

    locked = EditScalesDialog({}, [("Y", "", "")], error="Data series 'Record' is locked.")
    qtbot.addWidget(locked)
    locked.accept()
    assert locked.result() != locked.DialogCode.Accepted
    assert "locked" in locked.error_label.text()


def test_series_edit_scales(app, monkeypatch):
    for name, units in (("RecordA0", "mV"), ("RecordA1", "mV"), ("RecordB0", "pA")):
        app.demo.data[name].yscale.units = units
    app.demo.data["RecordA1"].xscale.delta = 2.0
    seen = {}

    def ask(x, y_rows, context, error, **x_options):
        seen.update(x=x, y_rows=y_rows, context=context, error=error, x_options=x_options)
        return {"xdelta": 0.5, "xunits": "ms", "y": {"A": {"label": "Vm"}, "B": {"units": "nA"}}}

    monkeypatch.setattr(app.series_actions, "_ask_scales", ask)
    app.series_actions.scales_action.trigger()

    assert seen["x"]["xdelta"] is None  # 1 and 2: mixed
    assert seen["x"]["xstart"] == "0"
    assert seen["y_rows"] == [("A", "", "mV"), ("B", "", "pA")]
    assert seen["context"] == ("Series", "Demo › Record") and seen["error"] is None
    data = app.demo.data
    assert {data[n].xscale.delta for n in ("RecordA0", "RecordA1", "RecordB0")} == {0.5}
    assert data["RecordB0"].xscale.units == "ms"
    assert (data["RecordA0"].yscale.label, data["RecordA0"].yscale.units) == ("Vm", "mV")
    assert (data["RecordB0"].yscale.label, data["RecordB0"].yscale.units) == ("", "nA")

    # A locked series: the dialog reports it, and the engine refuses too
    app.series.locked = True
    monkeypatch.setattr(app.series_actions, "_ask_scales",
                        lambda x, y_rows, context, error, **_: seen.update(error=error))
    app.series_actions.edit_scales()
    assert "is locked" in seen["error"]
    with pytest.raises(RuntimeError, match="locked"):
        app.series.set_xunits("s")


def test_data_edit_scales(app, monkeypatch):
    data = app.demo.data["StimA0"]
    app.selection_model.update(data=data)
    seen = {}

    def ask(x, y_rows, context, error):
        seen.update(y_rows=y_rows, context=context, error=error)
        return {"xstart": 5.0, "y": {"Y": {"label": "Stim", "units": "pA"}}}

    monkeypatch.setattr(app.data_actions, "_ask_scales", ask)
    app.data_actions.scales_action.trigger()
    assert seen["context"] == ("Data", "Demo › StimA0")
    assert seen["y_rows"] == [("Y", "", "")]
    assert data.xscale.start == 5.0
    assert (data.yscale.label, data.yscale.units) == ("Stim", "pA")

    # An array in a locked series
    app.series.locked = True
    app.selection_model.update(data=app.demo.data["RecordA0"])
    app.data_actions.edit_scales()
    assert "locked data series 'Record'" in seen["error"]
    assert app.demo.data["RecordA0"].xscale.start == 0  # not applied


def test_locked_current_series_greys_out_only_its_own_actions(app):
    actions = app.series_actions
    # These act on the current series only
    current_only = [actions.scales_action, actions.remove_channel_action,
                    actions.remove_epoch_action]
    # These choose their series in the dialog, which checks its lock
    chosen = [actions.update_action, actions.rename_action, actions.delete_action,
              actions.copy_action]
    tips = [item.toolTip() for item in current_only]

    actions.lock_action.trigger()
    assert not any(item.isEnabled() for item in current_only)
    assert all(item.toolTip() == "Data series 'Record' is locked." for item in current_only)
    assert all(item.isEnabled() for item in chosen)

    # Another, unlocked series can still be renamed from the dialog
    app.demo.sync_dataseries("Stim")
    app.answers = [("Stim", "Cue")]
    actions.rename_series()
    assert "Cue" in app.demo.dataseries
    assert "is locked" in app.dialogs[-1][1]["validate"]("Record", "Other")

    actions.lock_action.trigger()  # unlock
    assert all(item.isEnabled() for item in current_only)
    assert [item.toolTip() for item in current_only] == tips
