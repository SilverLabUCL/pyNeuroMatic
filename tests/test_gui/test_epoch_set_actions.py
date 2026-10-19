"""Tests for the Epoch Set actions."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tier_actions import epoch_range, range_summary

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Folder Demo with series Record of 1 channel and 10 epochs (E0-E9),
    epoch set Set0 = [E0, E1]; Record and epoch E2 are selected."""
    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=True)
    series = demo.new_dataseries("Record", n_epochs=10, n_points=3)
    series.epochs.sets.add("Set0", [series.epochs["E0"], series.epochs["E1"]])
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series, epoch=series.epochs["E2"])
    window.answers, window.dialogs = [], []

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    monkeypatch.setattr(window.epoch_set_actions, "_choose", choose)
    window.series = series
    window.sets = series.epochs.sets
    return window


def _members(app, name):
    return app.sets.get_items(name, get_keys=True)


def test_range_helpers():
    names = [f"E{i}" for i in range(10)]
    assert epoch_range(names, "E2", "E8", 2) == ["E2", "E4", "E6", "E8"]
    assert epoch_range(names, "E5", "E5") == ["E5"]
    assert epoch_range(names, "E5", "E2") == []
    assert range_summary("Adds", ["E2", "E4", "E6"], 2) == "Adds 3 epochs (E2–E6, every 2nd)."
    assert range_summary("Removes", ["E3"], 1) == "Removes 1 epoch (E3)."


def test_epoch_set_menu_in_strip_and_series_menu(app):
    texts = [
        "New Set…", "Add Epochs…", "Remove Epochs…", "Exclude Epochs…",
        "Include Epochs…", "Invert…", "Combine Sets…",
        "Clear Set…", "Rename Set…", "Delete Set…", "Choose in Browser…",
    ]
    button = app.selection_strip.action_button("epoch_set")
    assert button.isEnabled()
    assert [a.text() for a in button.menu().actions() if a.text()] == texts
    menus = {a.text(): a.menu() for a in app.menuBar().actions()}
    submenu = next(a.menu() for a in menus["Series"].actions() if a.text() == "Epoch Set")
    assert [a.text() for a in submenu.actions() if a.text()] == texts


def test_new_set_is_selected(app):
    app.answers = [(None, "Odd")]
    app.epoch_set_actions.new_set()
    assert app.dialogs[-1][1]["context"] == ("Series", "Demo › Record")
    assert "Odd" in app.sets.keys()
    assert app.selection_model.selection["epoch_set"] == "Odd"
    assert app.selection_strip.combo("epoch_set").currentText() == "Odd"


def test_add_and_remove_a_range(app):
    actions = app.epoch_set_actions
    app.selection_model.update(epoch_set="Set0")
    actions.add_epochs()  # cancelled; check the options
    options = app.dialogs[-1][1]
    assert options["current"] == "Set0"
    assert [label for label, *_ in options["extras"]] == ["From Epoch", "To Epoch", "Every"]
    assert options["extras"][0][1].value == 2 and options["extras"][1][1].value == 2
    assert options["extras"][0][1].prefix == ""
    # The boxes give plain numbers
    assert options["message"]("Set0", "1", "9", "2") == "Adds 5 epochs (E1–E9, every 2nd)."
    assert options["message"]("Set0", "E1", "E9", "2") == "Adds 5 epochs (E1–E9, every 2nd)."
    assert options["validate"]("Set0", None, "E5", "E2", "1") == "To must not be before From"

    app.answers = [("Set0", None, "E1", "E9", "2")]
    actions.add_epochs()
    assert _members(app, "Set0") == ["E0", "E1", "E3", "E5", "E7", "E9"]

    actions.remove_epochs()  # cancelled; removal counts only members
    options = app.dialogs[-1][1]
    assert options["message"]("Set0", "E0", "E4", "1") == "Removes 3 epochs (E0–E3)."
    assert options["validate"]("Set0", None, "E2", "E2", "1") == (
        "None of these epochs are in 'Set0'"
    )
    app.answers = [("Set0", None, "E0", "E4", "1")]
    actions.remove_epochs()
    assert _members(app, "Set0") == ["E5", "E7", "E9"]


def test_invert_and_combine(app):
    actions = app.epoch_set_actions
    app.answers = [("Set0", "Rest")]
    actions.invert_set()
    assert _members(app, "Rest") == [f"E{i}" for i in range(2, 10)]
    assert app.selection_model.selection["epoch_set"] == "Rest"

    app.answers = [("Set0", "All", "OR", "Rest")]
    actions.combine_sets()
    options = app.dialogs[-1][1]
    assert options["extras"][0] == ("Operator", ["AND", "OR"], "OR")
    assert len(_members(app, "All")) == 10
    assert options["validate"]("Rest", "X", "OR", "Rest") == "Choose two different sets"


def test_clear_rename_delete(app):
    actions = app.epoch_set_actions
    app.selection_model.update(epoch_set="Set0")
    app.answers = [("Set0", "First")]
    actions.rename_set()
    assert app.selection_model.selection["epoch_set"] == "First"
    app.answers = [("First", None)]
    actions.clear_set()
    assert _members(app, "First") == []
    app.answers = [("First", None)]
    actions.delete_set()
    assert "First" not in app.sets.keys()
    assert app.selection_model.selection["epoch_set"] is None
    assert list(app.sets.keys()) == ["Set1", "Set2", "SetX"]  # the defaults stay


def test_epoch_sets_work_on_a_locked_series(app):
    app.series.locked = True
    app.epoch_set_actions.update_enabled()
    assert app.epoch_set_actions.add_action.isEnabled()
    app.answers = [("Set0", None, "E8", "E9", "1")]
    app.epoch_set_actions.add_epochs()
    assert _members(app, "Set0") == ["E0", "E1", "E8", "E9"]


def test_choose_in_browser_selects_epoch_set(app):
    from pyneuromatic.gui.folder_model import _SetNode

    app.selection_model.update(epoch_set="Set0")
    app.epoch_set_actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    node = app.browser_widget.tree.currentIndex().internalPointer()
    assert isinstance(node, _SetNode) and node.name == "Set0"


def test_epoch_set_list_starts_with_all_and_current(app):
    combo = app.selection_strip.combo("epoch_set")
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts[:2] == ["All", "Current Epoch"]
    assert texts[2:] == ["Set1", "Set2", "SetX", "Set0"]  # creation order
    assert combo.currentText() == "All"  # the default for a series
    # AND/OR only combines a stored set with a group
    app.series.epochs.groups.assign_cyclic(list(app.series.epochs.keys()), n_groups=2)
    app.selection_model.update(group=0)
    assert not app.selection_strip.combo("group_operator").isEnabled()
    app.selection_model.update(epoch_set="Set0")
    assert app.selection_strip.combo("group_operator").isEnabled()


def test_exclude_and_include_epochs(app):
    actions = app.epoch_set_actions
    assert not actions.include_action.isEnabled()  # nothing excluded yet
    app.answers = [(None, None, "E7", "E9", "1")]
    actions.exclude_epochs()
    title, options = app.dialogs[-1]
    assert title == "Exclude Epochs" and "items" not in options  # always SetX
    assert _members(app, "SetX") == ["E7", "E8", "E9"]
    assert actions.include_action.isEnabled()

    actions.include_epochs()  # cancelled; only excluded epochs count
    options = app.dialogs[-1][1]
    assert options["message"](None, "E0", "E8", "1") == "Includes 2 epochs (E7–E8)."
    app.answers = [(None, None, "E8", "E8", "1")]
    actions.include_epochs()
    assert _members(app, "SetX") == ["E7", "E9"]


def test_setx_is_protected_and_builtin_names_reserved(app):
    actions = app.epoch_set_actions
    actions.rename_set()
    assert "cannot be renamed" in app.dialogs[-1][1]["validate"]("SetX", "Other")
    actions.delete_set()
    assert "cannot be deleted" in app.dialogs[-1][1]["validate"]("SetX", None)
    actions.new_set()
    validate = app.dialogs[-1][1]["validate"]
    assert validate(None, "All") == "'All' is a built-in Epoch Set choice"
    assert validate(None, "current epoch") == (
        "'current epoch' is a built-in Epoch Set choice"
    )


def test_run_leaves_out_setx_and_reports_it(app, monkeypatch):
    tab = app.stats_tab
    assert tab.run_info_label.text() == "10 epochs"
    app.sets.add("SetX", [app.series.epochs[e] for e in ("E7", "E8", "E9")])
    app.selection_model.refresh()
    tab._update_run_info()
    assert tab.run_info_label.text() == "7 epochs (3 excluded)"

    app.selection_model.update(epoch_set="Current Epoch", epoch=app.series.epochs["E8"])
    assert tab.run_info_label.text() == "0 epochs (1 excluded)"

    # The Table still shows every epoch, excluded ones too
    app.selection_model.update(epoch_set="All")
    assert len(app.table_panel.channel_table.model.arrays) == 10

    # Running on SetX itself asks first
    asked = []
    monkeypatch.setattr(tab, "_confirm", lambda title, text: asked.append(text) or False)
    app.selection_model.update(epoch_set="SetX")
    tab.measurement_combo.setCurrentText("mean")
    tab.run_button.click()
    assert "excluded from analysis" in asked[0]
    assert tab.status_label.text().startswith("Failed: Cancelled")


def test_number_boxes_and_numeric_ranges(qtbot, app):
    from pyneuromatic.gui.tier_actions import (
        ChooseItemDialog, Spin, epoch_range, range_error,
    )

    # Ranges work by number, so gaps from removed epochs are skipped
    names = ["E0", "E1", "E5", "E6", "E9"]
    assert epoch_range(names, "E1", "E6") == ["E1", "E5", "E6"]
    assert epoch_range(names, "E0", "E9", 2) == ["E0", "E5", "E9"]
    assert range_error(names, "E2", "E4") == "No epochs numbered 2 to 4"
    assert epoch_range(names, "1", "6") == ["E1", "E5", "E6"]  # plain numbers too
    assert range_error(names, "E6", "E1") == "To must not be before From"

    dialog = ChooseItemDialog(
        "Add Epochs", extras=[("From", Spin(0, 999, 10, "E")), ("To", Spin(0, 999, 10, "E")),
                              ("Every", Spin(1, 1000, 1))],
        message=lambda _item, first, last, every: f"{first}-{last}/{every}",
    )
    qtbot.addWidget(dialog)
    spin = dialog.extra_widgets[1]
    assert spin.text() == "E10" and spin.maximum() == 999
    spin.setValue(250)
    assert dialog.extras() == ["E10", "E250", "1"]
    assert dialog.message_label.text() == "E10-E250/1"  # updates with the box
