"""Tests for the Group actions."""

import re

import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tier_actions import cyclic_groups

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Folder Demo with series Record of 15 epochs (E0-E14); E3 selected."""
    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=True)
    series = demo.new_dataseries("Record", n_epochs=15, n_points=3)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series, epoch=series.epochs["E3"])
    window.answers, window.dialogs = [], []

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    monkeypatch.setattr(window.group_actions, "_choose", choose)
    window.series = series
    window.groups = series.epochs.groups
    return window


def _assignment(app):
    return [app.groups.get_group(f"E{i}") for i in range(15)]


def test_cyclic_groups():
    assert cyclic_groups(8, 2) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert cyclic_groups(13, 4, block=3) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]


def test_group_menu_in_strip_and_series_menu(app):
    texts = ["New Groups…", "Assign Cyclic…", "Add to Group…", "Remove from Group…",
             "Clear Group…", "Delete Group…", "Show in Browser"]
    button = app.selection_strip.action_button("group")
    assert button.isEnabled()
    assert [a.text() for a in button.menu().actions() if a.text()] == texts
    menus = {a.text(): a.menu() for a in app.menuBar().actions()}
    submenu = next(a.menu() for a in menus["Series"].actions() if a.text() == "Group")
    assert [a.text() for a in submenu.actions() if a.text()] == texts
    actions = app.group_actions
    assert not actions.unassign_action.isEnabled()  # nothing assigned yet
    assert not actions.clear_action.isEnabled()


def test_assign_cyclic_with_block(app):
    actions = app.group_actions
    actions.assign_cyclic()  # cancelled; check the options
    title, options = app.dialogs[-1]
    assert title == "Assign Epochs to Cyclic Groups"
    assert options["context"] == ("Data Series", "Demo › Record")
    assert [(label, spin.value) for label, spin in options["extras"]] == [
        ("Number of Groups", 2), ("Block", 1), ("From Epoch", 0), ("To Epoch", 14),
    ]
    # The pattern (two full cycles, 16 to 32 numbers), then where it goes
    assert options["message"](None, "4", "3", "E0", "E14") == (
        "Pattern: 0 0 0 1 1 1 2 2 2 3 3 3 0 0 0 1 1 1 2 2 2 3 3 3 …\n"
        "For epoch 0–14 (n=15).")
    assert options["message"](None, "2", "1", "E0", "E2") == (  # a short range: still 16
        "Pattern: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 …\nFor epoch 0–2 (n=3).")
    assert options["message"](None, "3", "20", "E4", "E4").endswith(
        "For epoch 4 (n=1).")
    long_cycle = options["message"](None, "3", "20", "E0", "E14").splitlines()[0]
    assert len(long_cycle.split()[1:-1]) == 32  # at most 32 numbers, then …
    assert options["validate"](None, None, "2", "1", "E5", "E1") == "To must not be before From"
    assert options["validate"](None, None, "2", "1", "E5", "E20") is None  # by number

    app.answers = [(None, None, "4", "3", "E0", "E14")]
    actions.cyclic_action.trigger()
    assert _assignment(app) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]
    combo = app.selection_strip.combo("group")
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "All", "0", "1", "2", "3"]
    assert actions.clear_action.isEnabled() and actions.unassign_action.isEnabled()


def test_assign_and_unassign_a_range(app):
    actions = app.group_actions
    actions.assign_epochs()  # cancelled; check the options
    options = app.dialogs[-1][1]
    group = options["extras"][0][1]
    assert options["extras"][0][0] == "Group"
    assert (group.minimum, group.maximum, group.value) == (0, 0, 0)  # none yet: group 0
    # From / To at the current epoch (E3), Every 1
    assert [spin.value for _label, spin in options["extras"][1:]] == [3, 3, 1]
    assert options["message"](None, "0", "E2", "E8", "2") == (
        "Adds 4 epochs (E2–E8, every 2nd) to group 0."
    )

    app.answers = [(None, None, "0", "E0", "E4", "1")]
    actions.assign_epochs()
    app.answers = [(None, None, "1", "E5", "E9", "1")]
    actions.assign_epochs()
    assert _assignment(app)[:10] == [0] * 5 + [1] * 5

    actions.unassign_epochs()  # cancelled; only assigned epochs count
    options = app.dialogs[-1][1]
    assert options["message"](None, "E8", "E14", "1") == (
        "Removes 2 epochs (E8–E9) from their group.")
    assert options["validate"](None, None, "E12", "E14", "1") == (
        "None of these epochs are in a group"
    )
    app.answers = [(None, None, "E8", "E14", "1")]
    actions.unassign_epochs()
    assert _assignment(app)[7:10] == [1, None, None]


def test_clear_group_one_or_all(app):
    app.groups.assign_cyclic([f"E{i}" for i in range(15)], n_groups=2)
    app.selection_model.update(group=1)
    app.answers = [("1", None)]
    app.group_actions.clear_group()
    options = app.dialogs[-1][1]
    assert options["items"] == ["0", "1", "All"] and options["current"] == "1"
    assert app.groups.group_numbers == [0, 1] and app.groups.get_items(1) == []
    assert len(app.groups.get_items(0)) == 8  # the other group is untouched
    app.answers = [("All", None)]
    app.group_actions.clear_group()
    assert app.groups.group_numbers == [0, 1]  # the groups are kept, empty
    assert app.groups.get_items(0) == []
    assert app.selection_model.selection["group"] == 1  # clearing keeps the choice


def test_groups_work_on_a_locked_series(app):
    app.series.locked = True
    app.group_actions.update_enabled()
    assert app.group_actions.cyclic_action.isEnabled()
    app.answers = [(None, None, "2", "1", "E0", "E3")]
    app.group_actions.assign_cyclic()
    assert _assignment(app)[:4] == [0, 1, 0, 1]


def test_run_uses_group_minus_setx(app):
    app.groups.assign_cyclic([f"E{i}" for i in range(15)], n_groups=3)
    app.series.epochs.sets.add("SetX", [app.series.epochs["E0"]])
    app.selection_model.refresh()
    app.selection_model.update(group=0)
    assert app.stats_tab.run_info_label.text() == "n=4"


def test_choose_in_browser_selects_epochs(app):
    app.group_actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    from pyneuromatic.gui.folder_model import _GroupsNode

    node = app.browser_widget.tree.currentIndex().internalPointer()
    assert isinstance(node, _GroupsNode) and node.group.owner is app.series


def _child(model, parent, label):
    for row in range(model.rowCount(parent)):
        index = model.index(row, 0, parent)
        if model.data(index) == label:
            return index
    raise AssertionError(f"row not found: {label}")


def _labels(model, parent):
    return [model.data(model.index(r, 0, parent)) for r in range(model.rowCount(parent))]


def test_browser_lists_groups_under_epochs(app):
    model = app.browser_widget.tree.model()
    folder = _child(model, _child(model, model.index(0, 0), "Demo"), "Data Series")
    epochs = _child(model, _child(model, folder, "Record"), "Epochs")
    assert _labels(model, epochs)[:2] == ["Sets", "Groups"]  # always listed
    assert _labels(model, _child(model, epochs, "Groups")) == []

    app.groups.assign_cyclic([f"E{i}" for i in range(6)], n_groups=2)
    app.browser_widget.refresh()
    folder = _child(model, _child(model, model.index(0, 0), "Demo"), "Data Series")
    epochs = _child(model, _child(model, folder, "Record"), "Epochs")
    assert _labels(model, epochs)[:2] == ["Sets", "Groups"]
    groups = _child(model, epochs, "Groups")
    assert _labels(model, groups) == ["Group 0", "Group 1"]
    group1 = _child(model, groups, "Group 1")
    assert _labels(model, group1) == ["E1", "E3", "E5"]
    assert model.parent(model.index(0, 0, group1)) == group1  # unambiguous parents


def test_choosing_a_group_row_selects_the_group(app):
    from pyneuromatic.gui.folder_model import _GroupNumberNode

    app.groups.assign_cyclic([f"E{i}" for i in range(15)], n_groups=3)
    app.selection_model.update(group=2)
    app.group_actions.choose_in_browser()
    browser = app.browser_widget
    current = browser.tree.currentIndex()
    node = current.internalPointer()
    assert isinstance(node, _GroupNumberNode) and node.number == 2

    # Requesting a different group's row selects that group in the strip
    group0 = browser.tree.model().index(0, 0, current.parent())
    browser._request_selection(group0)
    assert app.selection_model.selection["group"] == 0
    assert app.selection_model.selection["dataseries"] is app.series
    assert app.selection_strip.combo("group").currentText() == "0"


def _shown(label):
    """A rich-text label's text as it reads on screen."""
    return re.sub(r"<[^>]+>", "", label.text())


def test_strip_shows_epoch_count_beside_group(app):
    label = app.selection_strip.epoch_count_label
    assert _shown(label) == "n=15"  # All
    app.groups.assign_cyclic([f"E{i}" for i in range(15)], n_groups=3)
    app.series.epochs.sets.add("SetX", [app.series.epochs["E0"]])
    app.selection_strip.refresh_options()
    assert _shown(label) == "n=15-1"
    assert 'color: #c62828;">-1</span>' in label.text()  # the excluded in red
    assert label.toolTip() == "15 epochs selected, 1 excluded (SetX): 14 analysed"

    app.selection_model.update(group=0)
    assert _shown(label) == "n=5-1"
    app.selection_model.update(epoch_set="Set1", group_operator=None)
    assert _shown(label) == "Choose AND or OR"
    app.selection_model.update(group_operator="OR")
    assert _shown(label) == "n=5-1"  # Set1 is empty
    app.selection_model.update(epoch_set="Current Epoch")
    assert _shown(label) == "n=1"  # E3
    assert label.toolTip() == "1 epoch selected"
    app.selection_model.update(epoch=app.series.epochs["E0"])
    assert _shown(label) == "n=1-1"


def test_new_groups_creates_empty_groups(app):
    actions = app.group_actions
    actions.new_groups()  # cancelled; check the options
    options = app.dialogs[-1][1]
    assert options["context"][0] == "Data Series"
    assert options["extras"][0][0] == "Number of Groups"
    assert options["extras"][0][1].value == 2
    assert options["message"](None, "3") == "Creates Groups 0–2, with no epoch assignments."

    app.answers = [(None, None, "3")]
    actions.new_action.trigger()
    assert app.groups.group_numbers == [0, 1, 2]
    assert all(app.groups.get_items(n) == [] for n in range(3))
    combo = app.selection_strip.combo("group")
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "All", "0", "1", "2"]

    actions.new_groups()
    options = app.dialogs[-1][1]
    assert options["message"](None, "4") == "Creates Group 3, with no epoch assignments."
    assert options["validate"](None, None, "3") == "Groups 0–2 already exist"


def test_new_groups_then_fill_in_browser_edit_mode(app):
    from PyQt6 import QtCore

    app.answers = [(None, None, "2")]
    app.group_actions.new_groups()
    browser = app.browser_widget
    browser.show_groups_of(app.series, 1)
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    model.setData(model.index(4, 0), QtCore.Qt.CheckState.Checked.value,
                  QtCore.Qt.ItemDataRole.CheckStateRole)
    assert app.groups.get_items(1) == ["E4"]


def test_assign_cyclic_starts_at_the_groups_made(app):
    for number in range(3):  # New Groups... with 3
        app.groups.add_group(number)
    app.group_actions.assign_cyclic()  # cancelled; check the options
    extras = dict((label, spin.value) for label, spin in app.dialogs[-1][1]["extras"])
    assert extras["Number of Groups"] == 3
    app.answers = [(None, None, "3", "1", "E0", "E5")]
    app.group_actions.assign_cyclic()
    assert app.groups.group_numbers == [0, 1, 2]
    assert [app.groups.get_group(f"E{i}") for i in range(6)] == [0, 1, 2, 0, 1, 2]


def test_delete_group_one_or_all(app, monkeypatch):
    groups = app.series.epochs.groups
    groups.add_group(0)
    groups.add_group(1)
    groups.add_group(2)
    app.selection_model.update(group=1)
    actions = app.group_actions
    actions.update_enabled()
    assert actions.delete_action.isEnabled()
    asked = []
    answers = ["1", "All"]

    def choose(title, **options):
        asked.append((title, options, options["message"](answers[0])))  # while it's open
        return (answers.pop(0), None)

    monkeypatch.setattr(actions, "_choose", choose)
    actions.delete_action.trigger()
    title, options, message = asked[0]
    assert title == "Delete Epoch Group" and options["items"] == ["0", "1", "2", "All"]
    assert options["current"] == "1"  # the strip's group
    assert message.startswith("Group 1 will be deleted; its ")
    assert groups.group_numbers == [0, 2]
    assert app.selection_model.selection["group"] is None  # the strip's group went
    actions.delete_action.trigger()
    assert asked[1][2].startswith("All 2 groups will be deleted")
    assert groups.group_numbers == []
    assert len(app.series.epochs) > 0  # the epochs are kept


def test_epoch_boxes_show_the_last_epoch(qtbot):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, epoch_spins

    names = [f"E{i}" for i in range(15)]
    first, last = epoch_spins(names)
    assert first[1].suffix == "" and last[1].suffix == "  (max 14)"  # From is from 0
    later = epoch_spins([f"E{i}" for i in range(3, 9)])
    assert later[0][1].suffix == "  (min 3)" and later[1][1].suffix == "  (max 8)"
    dialog = ChooseItemDialog("Remove Epochs", extras=epoch_spins(names, "E2", "E5"))
    qtbot.addWidget(dialog)
    assert dialog.extra_widgets[1].text() == "5  (max 14)"  # shown in the box
    assert dialog.extras() == ["2", "5"]  # but only the number is read


def test_number_box_messages_follow_typing(qtbot):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, Spin

    dialog = ChooseItemDialog("New Groups", extras=[("Number of Groups", Spin(1, 1000, 2))],
                              message=lambda _item, count: f"{count} groups")
    qtbot.addWidget(dialog)
    box = dialog.extra_widgets[0]
    box.lineEdit().selectAll()
    qtbot.keyClicks(box, "5")  # typed, not the arrows
    assert dialog.message_label.text() == "5 groups"


def test_new_groups_minimum_follows_the_groups_there_are(app):
    actions = app.group_actions
    actions.new_groups()
    spin = app.dialogs[-1][1]["extras"][0][1]
    assert (spin.minimum, spin.value) == (1, 2)  # no groups yet
    for number in (0, 1, 2):
        app.groups.add_group(number)
    actions.new_groups()
    spin = app.dialogs[-1][1]["extras"][0][1]
    assert (spin.minimum, spin.value) == (4, 4)  # groups 0-2 exist: 4 makes group 3
    app.groups.delete_group(1)
    actions.new_groups()
    spin = app.dialogs[-1][1]["extras"][0][1]
    assert spin.minimum == 2  # 2 refills group 1


def test_assign_epochs_starts_at_the_current_group_and_epoch(app):
    for number in (0, 1, 2):
        app.groups.add_group(number)
    app.selection_model.update(group=1, epoch=app.series.epochs["E5"])
    app.group_actions.assign_epochs()  # cancelled; check the defaults
    options = app.dialogs[-1][1]
    group = options["extras"][0][1]  # a number box over the existing groups
    assert (group.minimum, group.maximum, group.value) == (0, 2, 1)  # the strip's group
    assert [spin.value for _label, spin in options["extras"][1:]] == [5, 5, 1]
def test_assign_epochs_refuses_a_missing_group(app):
    for number in (0, 2):  # group 1 was deleted
        app.groups.add_group(number)
    app.group_actions.assign_epochs()
    validate = app.dialogs[-1][1]["validate"]
    assert validate(None, None, "1", "0", "5", "1") == "Group 1 doesn't exist; New Groups… makes it"
    assert validate(None, None, "2", "0", "5", "1") is None
