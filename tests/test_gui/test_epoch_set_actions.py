"""Tests for the Epoch Set actions."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.selection_model import scoped_epoch_names
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
        "New Set…", "Add to Set…", "Remove from Set…", "Renumber…", "Copy…", "Invert…",
        "Combine…", "Clear…", "Delete…", "Show in Browser",
    ]
    button = app.selection_strip.action_button("epoch_set")
    assert button.isEnabled()
    assert [a.text() for a in button.menu().actions() if a.text()] == texts
    menus = {a.text(): a.menu() for a in app.menuBar().actions()}
    submenu = next(a.menu() for a in menus["Series"].actions() if a.text() == "Epoch Set")
    assert [a.text() for a in submenu.actions() if a.text()] == texts


def test_new_set_is_numbered_and_selected(app):
    app.epoch_set_actions.new_set()  # cancelled; check the options
    options = app.dialogs[-1][1]
    assert options["context"] == ("Data Series", "Demo › Record")
    label, spin = options["extras"][0]
    assert label == "New Set number" and spin.value == 2  # Set0, Set1 (and SetX) exist
    assert options["validate"](None, None, "1") == "'Set1' already exists."
    assert options["validate"](None, None, "2") is None
    assert options["message"](None, "2") == "Create 'Set2', an empty set."
    assert options["message"](None, "1") == ""  # the error says why
    assert options["validate_live"] is True
    app.answers = [(None, None, "2")]
    app.epoch_set_actions.new_set()
    assert "Set2" in app.sets.keys()
    assert app.selection_model.selection["epoch_set"] == "Set2"
    assert app.selection_strip.combo("epoch_set").currentText() == "Set2"


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
    assert options["message"]("Set0", "1", "9", "2") == "Add 5 epochs (E1–E9, every 2nd) to 'Set0'."
    assert options["message"]("Set0", "E1", "E9", "2") == "Add 5 epochs (E1–E9, every 2nd) to 'Set0'."
    assert options["validate"]("Set0", None, "E5", "E2", "1") == "To must not be before From"

    app.answers = [("Set0", None, "E1", "E9", "2")]
    actions.add_epochs()
    assert _members(app, "Set0") == ["E0", "E1", "E3", "E5", "E7", "E9"]

    actions.remove_epochs()  # cancelled; removal counts only members
    options = app.dialogs[-1][1]
    assert options["message"]("Set0", "E0", "E4", "1") == "Remove 3 epochs (E0–E3) from 'Set0'."
    assert options["validate"]("Set0", None, "E2", "E2", "1") == (
        "None of these epochs are in 'Set0'"
    )
    app.answers = [("Set0", None, "E0", "E4", "1")]
    actions.remove_epochs()
    assert _members(app, "Set0") == ["E5", "E7", "E9"]


def test_copy_invert_and_combine(app):
    actions = app.epoch_set_actions
    # Copy: a new set with the same members (a list of its own)
    app.answers = [("Set0", None, "3", False)]
    actions.copy_set()
    title, options = app.dialogs[-1]
    assert title == "Copy Epoch Set"
    assert options["message"]("Set0", "5", False) == "Copy 'Set0' to 'Set5'."
    assert options["validate_live"] is True  # "'Set3' already exists." at once
    assert options["message"]("Set0", "3", False) == ""  # Set3 exists now: the error says why
    assert _members(app, "Set3") == ["E0", "E1"] == _members(app, "Set0")
    assert app.selection_model.selection["epoch_set"] == "Set3"
    # Invert, in place: the epochs it didn't hold (Set0 unchanged)
    app.answers = [("Set3", None)]
    actions.invert_set()
    title, options = app.dialogs[-1]
    assert title == "Invert Epoch Set"
    assert options["message"]("Set0") == (
        "Invert epoch assignments of 'Set0'; 2 assigned : 8 unassigned → "
        "2 unassigned : 8 assigned")
    assert _members(app, "Set3") == [f"E{i}" for i in range(2, 10)]
    assert _members(app, "Set0") == ["E0", "E1"]
    app.answers = [("Set3", None)]
    actions.invert_set()  # again: back as it was
    assert _members(app, "Set3") == ["E0", "E1"]
    app.answers = [("Set3", None)]
    actions.invert_set()

    app.answers = [("Set0", None, "OR", "Set3", "4", False)]
    actions.combine_sets()
    title, options = app.dialogs[-1]
    assert title == "Combine Epoch Sets" and options["item_label"] == "Combine"
    assert options["extras"][0] == ("Operator", ["AND", "OR"], "OR")
    assert options["extras"][1][0] == "With" and options["validate_live"] is True
    assert options["message"]("Set0", "AND", "Set3", "5", False) == "Set5 = Set0 AND Set3"
    assert options["message"]("Set0", "OR", "Set3", "4", False) == ""  # Set4 exists now
    assert options["extras"][2][0] == "New Set number"
    assert len(_members(app, "Set4")) == 10
    assert options["validate"]("Set3", None, "OR", "Set3", "5", False) == (
        "Choose two different sets.")
    assert options["validate"]("Set0", None, "OR", "Set3", "4", False) == "'Set4' already exists."


def test_clear_rename_delete(app):
    actions = app.epoch_set_actions
    app.selection_model.update(epoch_set="Set0")
    actions.rename_set()  # cancelled; check the message (old to new)
    title, options = app.dialogs[-1]
    assert title == "Renumber Epoch Set" and options["item_label"] == ""
    assert options["message"]("Set0", "7") == "Renumber 'Set0' to 'Set7'."
    assert options["message"]("Set0", "0") == "Renumber 'Set0' to 'Set0' (no change)."
    assert options["message"]("Set0", "1") == ""  # Set1 exists: the error says why
    assert options["message"]("SetX", "7") == ""  # SetX can't be renumbered
    app.answers = [("Set0", None, "7")]  # Renumber Set0 -> Set7
    actions.rename_set()
    assert app.selection_model.selection["epoch_set"] == "Set7"
    actions.clear_set()  # cancelled; check the message
    assert app.dialogs[-1][1]["message"]("Set7") == "Remove 2 epochs from 'Set7'."
    app.answers = [("Set7", None)]
    actions.clear_set()
    assert _members(app, "Set7") == []
    actions.delete_set()  # cancelled; check the message
    assert app.dialogs[-1][1]["message"]("Set7") == (
        "Delete 'Set7'; epochs assigned to this set will be unaltered.")
    app.answers = [("Set7", None)]
    actions.delete_set()
    assert "Set7" not in app.sets.keys()
    assert app.selection_model.selection["epoch_set"] is None
    assert list(app.sets.keys()) == ["Set1", "SetX"]  # Set0 was renumbered, then deleted


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
    assert texts[:3] == ["", "All", "Current Epoch"]  # blank, like Data Set
    assert texts[3:] == ["All Sets", "Set0", "Set1", "SetX"]  # in order, from 0
    assert combo.currentText() == "All"  # the default for a series
    # AND/OR only combines a stored set with a group
    app.series.epochs.groups.assign_cyclic(list(app.series.epochs.keys()), n_groups=2)
    app.selection_model.update(group=0)
    assert not app.selection_strip.combo("group_operator").isEnabled()
    app.selection_model.update(epoch_set="Set0")
    assert app.selection_strip.combo("group_operator").isEnabled()


def test_exclude_and_include_with_add_to_set(app):
    # SetX is one of the sets: Add to Set / Remove from Set exclude and include
    actions = app.epoch_set_actions
    app.answers = [("SetX", None, "E7", "E9", "1")]
    actions.add_action.trigger()
    title, options = app.dialogs[-1]
    assert title == "Add Epochs to Set" and "SetX" in options["items"]
    assert _members(app, "SetX") == ["E7", "E8", "E9"]
    app.answers = [("SetX", None, "E8", "E8", "1")]
    actions.remove_action.trigger()
    assert _members(app, "SetX") == ["E7", "E9"]


def test_add_to_set_makes_setx_for_an_older_series(app):
    app.sets.pop("SetX", quiet=True)  # e.g. a series from before SetX
    actions = app.epoch_set_actions
    app.answers = [None]
    actions.add_action.trigger()
    assert "SetX" in app.dialogs[-1][1]["items"]


def test_setx_is_protected(app):
    actions = app.epoch_set_actions
    actions.rename_set()
    assert "cannot be renumbered" in app.dialogs[-1][1]["validate"]("SetX", None, "5")
    # Keeping a set's own number is fine; taking another set's isn't
    assert app.dialogs[-1][1]["validate"]("Set1", None, "1") is None
    assert app.dialogs[-1][1]["validate"]("Set1", None, "0") == "'Set0' already exists."
    actions.delete_set()
    assert "cannot be deleted" in app.dialogs[-1][1]["validate"]("SetX", None)


def test_run_leaves_out_setx_and_reports_it(app, monkeypatch):
    tab = app.stats_tab
    assert tab.run_info_label.text() == "n=10"
    app.sets.add("SetX", [app.series.epochs[e] for e in ("E7", "E8", "E9")])
    app.selection_model.refresh()
    tab._update_run_info()
    assert tab.run_info_label.text() == "n=7"
    assert tab.run_info_label.toolTip() == "Run analyses 7 epochs (3 excluded by SetX)"

    app.selection_model.update(epoch_set="Current Epoch", epoch=app.series.epochs["E8"])
    assert tab.run_info_label.text() == "n=0"

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


def test_blank_epoch_set_scopes_to_the_group_alone(app):
    app.series.epochs.groups.assign_cyclic(list(app.series.epochs.keys()), n_groups=2)
    combo = app.selection_strip.combo("epoch_set")
    combo.setCurrentIndex(0)  # the blank
    assert app.selection_model.selection["epoch_set"] is None
    app.selection_model.update(group=1)
    names = scoped_epoch_names(app.series, app.selection_model.selection)
    assert names == app.series.epochs.groups.get_items(1)
    assert app.selection_strip.combo("group_operator").count() == 1  # no AND/OR


def test_invert_refuses_combined_sets_and_setx(app):
    sets = app.series.epochs.sets
    sets.add("Set2", ["E5"])
    sets.define_or("Set4", "Set0", "Set2")
    app.epoch_set_actions.invert_set()  # cancelled; check the reasons
    validate = app.dialogs[-1][1]["validate"]
    assert validate("Set4", None).startswith("'Set4' is a combined set")
    assert validate("SetX", None).startswith("SetX can't be inverted")
    assert validate("Set0", None) is None


def test_copy_of_a_combined_set_is_a_plain_set(app):
    sets = app.series.epochs.sets
    sets.add("Set2", ["E5"])
    sets.define_or("Set4", "Set0", "Set2")
    app.answers = [("Set4", None, "6", False)]
    app.epoch_set_actions.copy_set()
    assert "isn't combined" in app.dialogs[-1][1]["message"]("Set4", "7", False)
    assert _members(app, "Set6") == ["E0", "E1", "E5"]
    assert not sets.is_equation("Set6")  # its members, not the definition
    sets.add("Set2", ["E7"])
    assert _members(app, "Set6") == ["E0", "E1", "E5"]  # doesn't follow Set2


def test_copy_and_combine_can_overwrite_a_set(app):
    sets = app.series.epochs.sets
    sets.add("Set2", ["E5"])
    order = sets.keys()
    actions = app.epoch_set_actions
    actions.copy_set()  # cancelled; check the options
    options = app.dialogs[-1][1]
    label, check = options["extras"][1]
    assert check.text == "Overwrite existing set"
    assert check.enabled(None, "2", False) and not check.enabled(None, "3", False)
    assert options["validate"]("Set0", None, "2", True) is None
    assert options["message"]("Set0", "2", True) == "Overwrite 'Set2' with a copy of 'Set0'."
    assert options["validate"]("Set0", None, "0", True) == "'Set0' can't be copied onto itself."
    app.answers = [("Set0", None, "2", True)]
    actions.copy_set()
    assert _members(app, "Set2") == ["E0", "E1"]
    assert sets.keys() == order  # in its place in the list

    # Combine over a plain set, and over a combined one
    app.answers = [("Set0", None, "OR", "Set2", "1", True)]
    actions.combine_sets()
    options = app.dialogs[-1][1]
    assert options["message"]("Set0", "OR", "Set2", "1", True) == "Set1 = Set0 OR Set2 (overwrite)"
    assert sets.is_equation("Set1") and sets.keys() == order
    assert options["validate"]("Set0", None, "AND", "Set1", "1", True) == (
        "'Set1' can't be overwritten: the combination uses it.")
    # ... nor a set that a combined set chosen is made from (Set1 uses Set2)
    assert options["validate"]("Set0", None, "AND", "Set1", "2", True) == (
        "'Set2' can't be overwritten: the combination uses it.")
    app.answers = [("Set0", None, "AND", "Set2", "1", True)]
    actions.combine_sets()
    assert sets.get_items("Set1", get_equation=True) == ("and", "Set0", "Set2")


def test_overwrite_checkbox_follows_the_set_number(qtbot):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, copy_set_options
    from pyneuromatic.core.nm_sets import NMSets

    sets = NMSets(name="sets", nmobjects={})
    sets.add("Set0", [])
    sets.add("Set1", [])
    dialog = ChooseItemDialog("Copy Epoch Set", item_label="", items=["Set0", "Set1"],
                              current="Set0", **copy_set_options(sets))
    qtbot.addWidget(dialog)
    spin, check = dialog.extra_widgets
    assert spin.value() == 2 and not check.isEnabled()  # Set2 is free
    spin.setValue(1)
    assert check.isEnabled() and not check.isChecked()
    assert dialog.error_label.text() == "'Set1' already exists."
    check.setChecked(True)
    assert dialog.error_label.isHidden()
    assert dialog.message_label.text() == "Overwrite 'Set1' with a copy of 'Set0'."
    assert dialog.extras() == ["1", True]
    spin.setValue(3)  # free again: the box is cleared
    assert not check.isEnabled() and not check.isChecked()
    assert dialog.message_label.text() == "Copy 'Set0' to 'Set3'."
