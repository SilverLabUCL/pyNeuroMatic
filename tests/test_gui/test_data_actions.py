"""Tests for the Data and Data Set actions, and browser data editing."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Folder Demo: series Record (RecordA0, RecordA1, RecordB0), Loose0 and
    data set DataSet = [Loose0]; folder Other. RecordA0 is selected.

    The choose dialog is replaced as in test_tier_actions: answers come from
    ``app.answers`` and each call's options are kept in ``app.dialogs``.
    """
    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=False)
    for name in ("RecordA0", "RecordA1", "RecordB0", "Loose0"):
        demo.data.new(name, nparray=np.array([1.0, 2.0]))
    demo.sync_dataseries("Record")
    demo.data.sets.add("DataSet", [demo.data["Loose0"]])
    manager.folders.new("Other", select=False)
    manager.select_value_set(demo)
    manager.select_value_set(demo.data["RecordA0"])

    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.answers, window.dialogs, window.messages = [], [], []
    window.confirms, window.confirm_answer = [], True

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    for actions in (window.data_actions, window.folder_actions):
        monkeypatch.setattr(actions, "_choose", choose)
        monkeypatch.setattr(actions, "_show_message",
                            lambda title, text, warning=False: window.messages.append(text))
        monkeypatch.setattr(
            actions, "_confirm",
            lambda title, text: window.confirms.append((title, text)) or window.confirm_answer,
        )
    window.demo = demo
    window.series = demo.dataseries["Record"]
    return window


def _menu_texts(menu):
    return [action.text() for action in menu.actions() if action.text()]


def test_data_menus_in_strip_and_menu_bar(app):
    strip = app.selection_strip
    assert _menu_texts(strip.action_button("data").menu()) == [
        "New…", "Rename…", "Edit Scales…", "Duplicate…", "Copy to Folder…", "Move to Folder…",
        "Delete…", "Show in Browser",
    ]
    assert _menu_texts(strip.action_button("data_set").menu()) == [
        "New Set…", "Add Data to Set…", "Remove Data from Set…", "Combine…",
        "Clear…", "Renumber…", "Delete…", "Show in Browser",
    ]
    menus = {action.text(): action.menu() for action in app.menuBar().actions()}
    assert "Data" in menus
    assert "Data Set" in _menu_texts(menus["Data"])


def test_set_actions_need_a_set(app):
    actions = app.data_actions
    assert actions.set_rename_action.isEnabled()
    app.demo.data.sets.pop("DataSet")
    actions.update_enabled()
    assert actions.set_new_action.isEnabled()
    assert not actions.set_rename_action.isEnabled()
    assert not actions.set_combine_action.isEnabled()


def test_duplicate_data_selects_copy(app):
    app.answers = [("Loose0", "C_Loose0")]
    app.data_actions.duplicate_data()
    default_name = app.dialogs[-1][1]["default_name"]
    assert default_name("RecordA1") == "C_RecordA1"
    assert default_name("Loose0") == "C1_Loose0"  # C_Loose0 now exists
    assert app.selection_model.selection["data"] is app.demo.data["C_Loose0"]
    assert app.selection_strip.combo("data").currentText() == "C_Loose0"


def test_rename_member_leaves_series_with_note(app):
    app.data_actions.rename_data()  # cancelled; check the note
    options = app.dialogs[-1][1]
    assert options["message"]("RecordA1") == (
        "Part of data series 'Record'; it will be removed from the series."
    )
    assert options["message"]("Loose0") == ""

    app.answers = [("RecordA1", "Renamed")]
    app.data_actions.rename_data()
    assert "Renamed" in app.demo.data
    assert app.series.get_data("A", "E1") is None
    assert "Renamed" in [app.selection_strip.combo("data").itemText(i)
                         for i in range(app.selection_strip.combo("data").count())]


def test_locked_series_data_cannot_be_renamed_or_deleted(app):
    app.series.locked = True
    app.data_actions.rename_data()
    options = app.dialogs[-1][1]
    assert options["message"]("RecordA0") == "Part of locked data series 'Record'."
    assert "locked data series 'Record'" in options["validate"]("RecordA0", "X")
    assert options["validate"]("Loose0", "Free0") is None

    app.data_actions.delete_data()
    assert "locked" in app.dialogs[-1][1]["validate"]("RecordA0", None)


def test_delete_member_unlinks_it_from_series(app):
    app.answers = [("RecordA1", None)]
    app.data_actions.delete_data()
    assert "RecordA1" not in app.demo.data
    assert app.series.get_data("A", "E1") is None
    assert app.series.get_data("A", "E0") is app.demo.data["RecordA0"]


def test_copy_to_folder(app):
    other = app.selection_model.manager.folders["Other"]
    app.answers = [("RecordA0", "RecordA0", "Other")]
    app.data_actions.copy_data_to_folder()
    options = app.dialogs[-1][1]
    assert options["extras"] == [("To folder", ["Other"], "Other")]
    assert "RecordA0" in other.data
    assert other.data.dataseries_of("RecordA0") == []
    # The target folder's names are checked
    assert options["validate"]("RecordA0", "RecordA0", "Other") == "'RecordA0' already exists"


def test_new_add_remove_and_combine_sets(app):
    actions, data = app.data_actions, app.demo.data
    actions.new_set()  # cancelled; sets are numbered
    label, spin = app.dialogs[-1][1]["extras"][0]
    assert label == "Set number" and spin.value == 0
    app.answers = [(None, None, "0")]
    actions.new_set()
    assert "Set0" in data.sets.keys()
    assert app.selection_model.selection["data_set"] == "Set0"

    app.answers = [("Set0", None, "RecordA0")]
    actions.add_to_set()
    assert data.sets.get_items("Set0", get_keys=True) == ["RecordA0"]

    actions.remove_from_set()  # cancelled; check the validation
    validate = app.dialogs[-1][1]["validate"]
    assert validate("Set0", None, "Loose0") == "'Loose0' is not in 'Set0'"
    assert validate("Set0", None, "RecordA0") is None

    app.answers = [("Set0", None, "OR", "DataSet", "1")]
    actions.combine_sets()
    assert app.dialogs[-1][1]["extras"][0] == ("Operator", ["AND", "OR"], "OR")
    assert sorted(data.sets.get_items("Set1", get_keys=True)) == ["Loose0", "RecordA0"]
    assert app.selection_model.selection["data_set"] == "Set1"

    app.answers = [("Set0", None, "RecordA0")]
    actions.remove_from_set()
    assert data.sets.get_items("Set0", get_keys=True) == []


def test_clear_rename_and_delete_sets(app):
    actions, data = app.data_actions, app.demo.data
    app.selection_model.update(data_set="DataSet")

    app.answers = [("DataSet", None, "4")]  # a named (e.g. imported) set renumbered
    actions.rename_set()
    assert app.dialogs[-1][0] == "Renumber Set"
    assert "Set4" in data.sets.keys() and "DataSet" not in data.sets.keys()
    assert app.selection_model.selection["data_set"] == "Set4"
    assert app.selection_strip.combo("data_set").currentText() == "Set4"

    app.answers = [("Set4", None)]
    actions.clear_set()
    assert data.sets.get_items("Set4", get_keys=True) == []
    assert "Set4" in data.sets.keys()

    app.answers = [("Set4", None)]
    actions.delete_set()
    assert "Set4" not in data.sets.keys()
    assert app.selection_model.selection["data_set"] is None
    assert "Loose0" in data  # the data is kept


def test_combine_sets_starts_with_two_different_sets(app):
    data = app.demo.data
    assert not app.data_actions.set_combine_action.isEnabled()  # one set only
    data.sets.add("Set1", [])
    app.selection_model.update(data_set="Set1")
    assert app.data_actions.set_combine_action.isEnabled()
    app.data_actions.combine_sets()  # cancelled; check the options
    options = app.dialogs[-1][1]
    assert options["current"] == "Set1"
    assert options["extras"][:2] == [
        ("Operator", ["AND", "OR"], "OR"),
        ("With Set", ["DataSet", "Set1"], "DataSet"),
    ]
    label, spin = options["extras"][2]
    assert label == "Set number" and spin.value == 0
    assert options["validate"]("Set1", None, "OR", "Set1", "0") == "Choose two different sets"
    assert options["validate"]("Set1", None, "OR", "DataSet", "0") is None
    assert options["validate"]("Set1", None, "OR", "DataSet", "1") == "'Set1' already exists"


def test_choose_sets_in_browser_shows_sets_list(app):
    from pyneuromatic.gui.folder_model import _SetNode, _SetsNode

    app.demo.data.sets.add("Other", [])
    app.selection_model.update(data_set="DataSet")
    app.data_actions.set_browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    browser = app.browser_widget
    # The set is chosen in the tree, so the detail pane lists its arrays
    current = browser.tree.currentIndex()
    node = current.internalPointer()
    assert isinstance(node, _SetNode) and node.name == "DataSet"
    assert browser.tree.isExpanded(current.parent())  # Data > Sets open
    assert browser.tree.isExpanded(current.parent().parent())
    detail = browser._detail
    members = [
        detail.model().index(row, 0, detail.rootIndex()).internalPointer().obj.name
        for row in range(detail.model().rowCount(detail.rootIndex()))
    ]
    assert members == ["Loose0"]

    # No current set: the Sets row is chosen, listing the sets
    app.selection_model.update(data_set=None)
    app.data_actions.choose_sets_in_browser()
    node = browser.tree.currentIndex().internalPointer()
    assert isinstance(node, _SetsNode) and node.container is app.demo.data

    # Without sets, the (empty) Sets row is chosen: new sets are made there
    app.demo.data.sets.pop("DataSet")
    app.demo.data.sets.pop("Other")
    app.browser_widget.refresh()
    app.data_actions.choose_sets_in_browser()
    node = app.browser_widget.tree.currentIndex().internalPointer()
    assert isinstance(node, _SetsNode) and node.container is app.demo.data


def test_choose_in_browser_shows_data_list(app):
    app.selection_model.update(data=app.demo.data["RecordA1"])
    app.data_actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    browser = app.browser_widget
    # The current data array is chosen in the tree, under an expanded Data row
    current = browser.tree.currentIndex()
    assert current.internalPointer() is app.demo.data["RecordA1"]
    data_row = current.parent().internalPointer()
    assert data_row.kind == "data" and data_row.owner is app.demo
    assert browser.tree.isExpanded(current.parent())

    # No current data array: the Data row is chosen
    app.selection_model.update(data=None)
    app.data_actions.choose_in_browser()
    node = browser.tree.currentIndex().internalPointer()
    assert node.kind == "data" and node.owner is app.demo


def test_browser_delete_keeps_locked_data_and_reports(app, monkeypatch):
    browser = app.browser_widget
    data = app.demo.data
    app.series.locked = True
    warnings = []
    monkeypatch.setattr(browser, "_current_container", lambda: data)
    monkeypatch.setattr(browser, "_selected_objects",
                        lambda: [data["RecordA0"], data["Loose0"]])
    monkeypatch.setattr(browser, "_confirm", lambda title, text: True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *args: warnings.append(args[2])))

    browser._delete_selected()

    assert "RecordA0" in data and "Loose0" not in data
    assert "locked data series 'Record'" in warnings[0]
    # The strip reloads after a browser change
    combo = app.selection_strip.combo("data")
    assert "Loose0" not in [combo.itemText(i) for i in range(combo.count())]


def test_browser_duplicate_and_copy_to_folder(app):
    browser = app.browser_widget
    data = app.demo.data
    other = app.selection_model.manager.folders["Other"]
    browser._duplicate_selected(data, [data["RecordA0"], data["Loose0"]])
    browser._duplicate_selected(data, [data["RecordA0"]])
    assert {"C_RecordA0", "C_Loose0", "C1_RecordA0"} <= set(data.keys())
    # Copies stay in the naming pattern: a separate C_Record series, not Record
    assert data.dataseries_of("C_RecordA0") == []
    assert app.demo.sync_dataseries("C_Record") is not None
    assert app.series.get_data("A", "E0") is data["RecordA0"]
    browser._copy_selected_to(data, [data["Loose0"]], other)
    browser._copy_selected_to(data, [data["Loose0"]], other)
    browser._copy_selected_to(data, [data["Loose0"]], other)
    # Same name where free, then the copy prefix, never a _1 suffix
    assert {"Loose0", "C_Loose0", "C1_Loose0"} <= set(other.data.keys())
    assert "Loose0_1" not in other.data


def test_series_name_hint(app):
    from pyneuromatic.gui.tier_actions import series_name_hint

    assert series_name_hint(app.demo, "RecordA7") == (
        "'RecordA7' fits data series 'Record' (channel A, epoch 7) and will be "
        "added to it the next time the series is updated."
    )
    assert series_name_hint(app.demo, "recordb2", joins_now=True).endswith(
        "and will join it now."
    )
    for name in ("C_RecordA0", "Loose9", "", None):
        assert series_name_hint(app.demo, name) == ""


def test_series_name_hint_for_names_that_only_start_with_a_prefix(app):
    from pyneuromatic.gui.tier_actions import series_name_hint

    expected = (
        "'{}' starts with the prefix of data series 'Record' but does not fit "
        "its naming pattern (e.g. RecordA0), so it will not be part of the series."
    )
    for name in ("RecordAA", "Record", "record_new", "RecordingA0"):
        assert series_name_hint(app.demo, name) == expected.format(name)

    # Also for a locked series, as nothing is added to it
    app.series.locked = True
    assert series_name_hint(app.demo, "RecordAA") == expected.format("RecordAA")

    # The longest matching prefix is named
    app.demo.data.new("RecordingA0", nparray=np.array([1.0]))
    app.demo.sync_dataseries("Recording")
    assert "data series 'Recording'" in series_name_hint(app.demo, "RecordingXX")


def test_actions_ask_when_name_fits_a_series(app):
    actions, data = app.data_actions, app.demo.data

    # A plain name: no question
    app.answers = [("Loose0", "C_Loose0")]
    actions.duplicate_data()
    assert app.confirms == [] and "C_Loose0" in data

    # Declined: nothing happens
    app.confirm_answer = False
    app.answers = [("Loose0", "RecordA7")]
    actions.duplicate_data()
    title, text = app.confirms[-1]
    assert title == "Duplicate Data"
    assert text.startswith("'RecordA7' fits data series 'Record'")
    assert text.endswith("Duplicate anyway?")
    assert "RecordA7" not in data

    # Accepted: the action goes ahead
    app.confirm_answer = True
    app.answers = [("Loose0", "RecordX")]
    actions.rename_data()
    title, text = app.confirms[-1]
    assert title == "Rename Data" and text.endswith("Rename anyway?")
    assert "starts with the prefix of data series 'Record'" in text
    assert "RecordX" in data

    other = app.selection_model.manager.folders["Other"]
    other.data.new("CellA0", nparray=np.array([1.0]))
    other.sync_dataseries("Cell")
    app.answers = [("C_Loose0", "CellA4", "Other")]
    actions.copy_data_to_folder()
    title, text = app.confirms[-1]
    assert title == "Copy Data to Folder" and "data series 'Cell'" in text
    assert "CellA4" in other.data


def test_browser_rename_confirms_when_name_fits_a_series(app, monkeypatch):
    browser = app.browser_widget
    data = app.demo.data
    asked = []
    monkeypatch.setattr(browser, "_current_container", lambda: data)
    monkeypatch.setattr(browser, "_selected_objects", lambda: [data["Loose0"]])
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("RecordA5", True)))
    monkeypatch.setattr(browser, "_confirm", lambda title, text: asked.append(text) or False)

    browser._rename_selected()
    assert "fits data series 'Record'" in asked[0]
    assert "Loose0" in data  # declined

    monkeypatch.setattr(browser, "_confirm", lambda title, text: True)
    browser._rename_selected()
    assert "RecordA5" in data
    assert [s.name for s in data.dataseries_of("RecordA5")] == ["Record"]


def test_names_fitting_a_locked_series_are_blocked(app):
    from pyneuromatic.gui.tier_actions import series_name_error, series_name_hint

    app.series.locked = True
    error = series_name_error(app.demo, "RecordA7")
    assert error.startswith("'RecordA7' fits locked data series 'Record'.")
    assert "copy the series under a new prefix" in error
    assert series_name_hint(app.demo, "RecordA7") == ""  # error, not warning
    assert series_name_error(app.demo, "C_RecordA7") is None

    actions = app.data_actions
    actions.duplicate_data()
    assert "locked data series" in app.dialogs[-1][1]["validate"]("Loose0", "RecordA7")
    actions.rename_data()
    assert "locked data series" in app.dialogs[-1][1]["validate"]("Loose0", "RecordA7")
    assert app.dialogs[-1][1]["validate"]("Loose0", "Free0") is None

    # Unlocked again: a warning only
    app.series.locked = False
    assert series_name_error(app.demo, "RecordA7") is None
    assert series_name_hint(app.demo, "RecordA7") != ""


def test_browser_rename_into_locked_series_reports(app, monkeypatch):
    browser = app.browser_widget
    data = app.demo.data
    app.series.locked = True
    warnings = []
    monkeypatch.setattr(browser, "_current_container", lambda: data)
    monkeypatch.setattr(browser, "_selected_objects", lambda: [data["Loose0"]])
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("RecordA7", True)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *args: warnings.append(args[2])))
    browser._rename_selected()
    assert "Loose0" in data and "RecordA7" not in data
    assert "fits locked data series 'Record'" in warnings[0]


def test_copy_to_folder_suggests_prefix_on_name_clash(qtbot, app):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, copy_target_name

    manager = app.selection_model.manager
    other = manager.folders["Other"]
    third = manager.folders.new("Third")
    other.data.new("Loose0", nparray=np.array([1.0]))
    assert copy_target_name(third.data, "Loose0") == "Loose0"
    assert copy_target_name(other.data, "Loose0") == "C_Loose0"

    app.data_actions.copy_data_to_folder()  # cancelled; check the default name
    options = app.dialogs[-1][1]
    assert options["default_name"]("Loose0", "Other") == "C_Loose0"
    assert options["default_name"]("Loose0", "Third") == "Loose0"

    # In the dialog, the name follows the chosen target folder
    dialog = ChooseItemDialog("Copy Data to Folder", **options)
    qtbot.addWidget(dialog)
    dialog.item_combo.setCurrentText("Loose0")
    dialog.extra_combos[0].setCurrentText("Other")
    assert dialog.name() == "C_Loose0"
    dialog.extra_combos[0].setCurrentText("Third")
    assert dialog.name() == "Loose0"


def test_data_dialogs_show_the_folder(qtbot, monkeypatch):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog, TierActions

    manager = NMManager(quiet=True)
    demo = manager.folders.new("Demo", select=False)
    demo.data.new("RecordA0", nparray=np.array([1.0]))
    stats = demo.toolfolders.new("Stats_0")
    stats.data.new("ST_w0_mean_y", nparray=np.array([1.0]))
    stats.data.sets.add("Picked", [])
    manager.folders.new("Other", select=False)
    manager.select_value_set(demo)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)

    seen = []
    monkeypatch.setattr(TierActions, "_choose",
                        lambda self, title, **options: seen.append(options) or None)
    actions = window.data_actions
    actions.rename_data()
    assert seen[-1]["context"] == ("Folder", "Demo")
    actions.copy_data_to_folder()
    assert seen[-1]["context"] == ("From folder", "Demo")

    window.selection_model.update(toolfolder=stats)
    actions.delete_data()
    assert seen[-1]["context"] == ("Folder", "Demo / Stats_0")
    actions.rename_set()
    assert seen[-1]["context"] == ("Folder", "Demo / Stats_0")

    dialog = ChooseItemDialog("Rename Data", context=("Folder", "Demo / Stats_0"),
                              item_label="Data", items=["ST_w0_mean_y"])
    qtbot.addWidget(dialog)
    assert dialog.context_label.text() == "Demo / Stats_0"


def _tree_children(browser, index):
    model = browser.tree.model()
    return [model.data(model.index(r, 0, index)) for r in range(model.rowCount(index))]


def test_browser_add_to_set_updates_browser_and_strip(app, monkeypatch):
    browser = app.browser_widget
    data = app.demo.data
    app.selection_model.update(data_set="DataSet")
    app.data_actions.choose_sets_in_browser()  # DataSet chosen in the tree
    set_index = browser.tree.currentIndex()
    assert _tree_children(browser, set_index) == ["Loose0"]

    monkeypatch.setattr(browser, "_current_container", lambda: data)
    monkeypatch.setattr(browser, "_selected_objects",
                        lambda: [data["RecordA0"], data["RecordA1"]])
    browser._add_selected_to_set("DataSet")

    # The tree shows the new members without a manual refresh
    set_index = browser.tree.currentIndex()
    assert sorted(_tree_children(browser, set_index)) == ["Loose0", "RecordA0", "RecordA1"]

    # A new set from the browser appears in the strip's Data Set list
    monkeypatch.setattr(browser, "_ask_set_number", lambda *a: (5, True))
    browser._add_selected_to_new_set()
    combo = app.selection_strip.combo("data_set")
    assert "Set5" in [combo.itemText(i) for i in range(combo.count())]



def _new_data_values(**changes):
    values = {
        "name": "Fresh0", "points": 5, "value": float("nan"),
        "xstart": 10.0, "xdelta": 0.5, "xlabel": "Time", "xunits": "ms",
        "ylabel": "Vm", "yunits": "mV",
    }
    values.update(changes)
    return values


def test_new_data_is_created_selected_and_shown_in_table(app, monkeypatch):
    seen = {}

    def ask(defaults, validate):
        seen["defaults"], seen["validate"] = defaults, validate
        return _new_data_values()

    monkeypatch.setattr(app.data_actions, "_ask_new_data", ask)
    app.selection_model.update(data_set="DataSet")  # Fresh0 is not in it
    app.data_actions.new_action.trigger()

    data = app.demo.data["Fresh0"]
    assert data.nparray.size == 5 and np.isnan(data.nparray).all()
    assert (data.xscale.start, data.xscale.delta) == (10.0, 0.5)
    assert (data.xscale.units, data.yscale.label, data.yscale.units) == ("ms", "Vm", "mV")
    assert app.selection_model.selection["data"] is data

    # Opened in the Table panel, on its own column
    table = app.table_panel
    assert app.context_panel.currentWidget() is table
    assert table.stack.currentWidget() is table.channel_table
    model = table.channel_table.model
    assert model.arrays == [data]
    assert model.headerData(0, QtCore.Qt.Orientation.Horizontal) == "x (ms)"
    assert model.data(model.index(1, 0)) == "10.5"

    # Defaults: the current array's length; x from 0 by 1; no labels or units
    assert seen["defaults"]["points"] == 2
    assert (seen["defaults"]["xstart"], seen["defaults"]["xdelta"]) == (0, 1)
    assert seen["defaults"]["xunits"] == seen["defaults"]["ylabel"] == ""
    assert seen["defaults"]["name"].startswith("data")  # the next auto name
    assert seen["validate"]({"name": "Loose0"}) == "'Loose0' already exists"


def test_new_data_asks_when_name_fits_a_series(app, monkeypatch):
    monkeypatch.setattr(app.data_actions, "_ask_new_data",
                        lambda defaults, validate: _new_data_values(name="RecordA9"))
    app.confirm_answer = False
    app.data_actions.new_data()
    assert app.confirms[-1][1].endswith("Create anyway?")
    assert "RecordA9" not in app.demo.data


def test_new_data_dialog_checks_fields_inline(qtbot):
    from pyneuromatic.gui.tier_actions import NewDataDialog

    dialog = NewDataDialog(
        {"name": "data0", "points": 10, "value": 0, "xstart": 0, "xdelta": 1},
        validate=lambda values: (
            "'taken' already exists" if values["name"] == "taken" else None
        ),
        context=("Folder", "Demo"),
    )
    qtbot.addWidget(dialog)
    dialog.edits["xdelta"].setText("abc")
    dialog.accept()
    assert dialog.error_label.text() == "Value, Start and Delta must be numbers"
    dialog.edits["xdelta"].setText("0")
    dialog.accept()
    assert dialog.error_label.text() == "Delta must not be 0"
    dialog.edits["xdelta"].setText("0.1")
    dialog.edits["name"].setText("taken")
    dialog.accept()
    assert dialog.error_label.text() == "'taken' already exists"
    dialog.edits["name"].setText("fresh")
    dialog.edits["value"].setText("nan")
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert np.isnan(dialog.values()["value"])


def test_table_cells_can_be_edited(app):
    data = app.demo.data["RecordA0"]
    app.selection_model.update(data=data)
    model = app.table_panel.channel_table.model
    value_index, x_index = model.index(1, 1), model.index(1, 0)
    edits = []
    app.table_panel.data_edited.connect(edits.append)

    assert model.flags(value_index) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(x_index) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert model.setData(value_index, "7.25")
    assert data.nparray[1] == 7.25 and edits == [data]
    assert not model.setData(value_index, "abc")
    assert not model.setData(x_index, "3")

    # Arrays in a locked series are read-only
    app.series.locked = True
    assert not model.flags(value_index) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.setData(value_index, "1")
    assert data.nparray[1] == 7.25



def test_browser_new_data_opens_in_table(app, monkeypatch):
    browser = app.browser_widget
    data = app.demo.data
    monkeypatch.setattr(browser, "_current_container", lambda: data)
    monkeypatch.setattr(browser, "_ask_new_data",
                        lambda defaults, validate, context: _new_data_values(name="Made0"))
    browser._add_item()
    made = data["Made0"]
    assert app.selection_model.selection["data"] is made
    assert app.context_panel.currentWidget() is app.table_panel
    assert app.table_panel.channel_table.model.arrays == [made]
