"""Tests for Prefix ⋯ › Move to Folder… and Append Series…"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Folder A: Record (channels A, B x E0..E2) and Stim (A, B x E0, E1);
    folder B empty. Record is selected. The choose dialog is replaced:
    ``app.answers`` feed it and each call's options are kept in ``app.dialogs``."""
    manager = NMManager(quiet=True)
    a = manager.folders.new("A", select=False)
    b = manager.folders.new("B", select=False)
    a.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    a.new_dataseries("Stim", n_channels=2, n_epochs=2, n_points=4)
    manager.select_value_set(a)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=a.dataseries["Record"])
    window.answers, window.dialogs, window.messages = [], [], []

    def choose(title, **options):
        window.dialogs.append((title, options))
        return window.answers.pop(0) if window.answers else None

    actions = window.series_actions
    monkeypatch.setattr(actions, "_choose", choose)
    monkeypatch.setattr(actions, "_show_message",
                        lambda title, text, warning=False: window.messages.append(text))
    window.a, window.b = a, b
    return window


def test_menu_items_enabled(app):
    actions = app.series_actions
    assert actions.move_action.isEnabled() and actions.append_action.isEnabled()
    app.selection_model.update(dataseries=app.a.dataseries["Stim"])
    app.a.remove_dataseries("Record")
    app._on_content_changed()
    assert not actions.append_action.isEnabled()  # nothing else to append


def test_move_to_folder(app):
    actions = app.series_actions
    actions.move_action.trigger()  # cancelled: look at the dialog
    title, options = app.dialogs[0]
    assert title == "Move Series" and options["extras"][0] == ("To folder", ["B"], "B")
    assert options["default_name"](None, "B") == "Record"  # free there
    app.answers = [(None, "Record", "B")]
    actions.move_action.trigger()
    assert "Record" in app.b.dataseries and "Record" not in app.a.dataseries
    # The strip follows the series to its new folder
    selection = app.selection_model.selection
    assert selection["folder"] is app.b and selection["dataseries"] is app.b.dataseries["Record"]


def test_move_keeps_the_prefix_and_says_when_it_is_taken(app):
    app.b.new_dataseries("Record", n_channels=1, n_epochs=1, n_points=4)
    actions = app.series_actions
    actions.move_action.trigger()  # cancelled
    options = app.dialogs[0][1]
    assert options["default_name"](None, "B") == "Record"  # kept, not changed
    assert options["validate_live"] is True  # the problem shows as it opens
    assert options["validate"](None, "Record", "B") == (
        "Data series 'Record' already exists. Enter a new prefix.")
    assert options["validate"](None, "Record2", "B") is None


def test_move_dialog_shows_the_problem_at_once(qtbot, app):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog

    app.b.new_dataseries("Record", n_channels=1, n_epochs=1, n_points=4)
    app.series_actions.move_action.trigger()
    title, options = app.dialogs[0]
    dialog = ChooseItemDialog(title, **options)
    qtbot.addWidget(dialog)
    assert dialog.name() == "Record"
    assert not dialog.error_label.isHidden()
    assert dialog.error_label.text() == "Data series 'Record' already exists. Enter a new prefix."
    assert "renamed" not in dialog.message_label.text()
    dialog.name_edit.setText("Record2")
    assert dialog.error_label.isHidden()


def test_append_series(app):
    actions = app.series_actions
    actions.append_action.trigger()  # cancelled: look at the dialog
    title, options = app.dialogs[0]
    assert title == "Append Series" and options["items"] == ["A"]  # B has no series
    assert options["current"] == "A" and options["extras"][0] == ("Append", ["Stim"], None)
    assert options["message"]("A", "Stim", "Keep") == (
        "Adds 2 epochs (E3 to E4) from 'Stim': 4 arrays copied and renamed "
        "(e.g. StimA0 → RecordA3).")
    assert "moved" in options["message"]("A", "Stim", "Delete")
    app.answers = [("A", None, "Stim", "Delete")]
    actions.append_action.trigger()
    record = app.a.dataseries["Record"]
    assert list(record.epochs.keys()) == ["E0", "E1", "E2", "E3", "E4"]
    assert "Stim" not in app.a.dataseries
    assert app.plot_widget.trace_count_label.text() == "of 5"


def test_append_explains_a_problem(app):
    app.a.new_dataseries("Odd", n_channels=3, n_epochs=1, n_points=4)
    actions = app.series_actions
    actions.append_action.trigger()
    validate = app.dialogs[0][1]["validate"]
    assert validate("A", None, "Odd", "Keep") == (
        "The channels differ: 'Record' has A, B, 'Odd' has A, B, C")
    assert validate("A", None, "Stim", "Keep") is None


def test_move_dialog_says_the_x_array_goes_too(app):
    times = app.a.data.new("Times", nparray=np.array([0.0, 1.0, 2.0, 5.0]))
    app.a.dataseries["Record"].set_xarray(times)
    actions = app.series_actions
    actions.move_action.trigger()  # cancelled
    options = app.dialogs[0][1]
    assert options["message"].endswith("X array 'Times' moves too.")
    app.b.data.new("Times", nparray=np.zeros(4))
    assert options["validate"](None, "Record", "B") == (
        "Folder 'B' already has an array 'Times' (the series' x array). "
        "Rename one of them first.")


def test_append_summary_notes_the_x_array(app):
    times = app.a.data.new("Times", nparray=np.array([0.0, 1.0, 2.0, 5.0]))
    app.a.dataseries["Record"].set_xarray(times)
    app.series_actions.append_action.trigger()
    message = app.dialogs[0][1]["message"]("A", "Stim", "Keep")
    assert message.endswith("The appended arrays use x array 'Times', as 'Record' does "
                            "('Stim' used Start and Delta).")


# --- Data ⋯ › Move to Folder… -------------------------------------------------

def test_data_move_to_folder(app, monkeypatch):
    single = app.a.data.new("Single", nparray=np.zeros(4))
    actions = app.data_actions
    dialogs = []

    def choose(title, **options):
        dialogs.append((title, options))
        return ("Single", "Single", "B")

    monkeypatch.setattr(actions, "_choose", choose)
    assert actions.move_action.isEnabled()
    actions.move_action.trigger()
    title, options = dialogs[0]
    assert title == "Move Data to Folder" and options["validate_live"] is True
    assert options["default_name"]("Single", "B") == "Single"
    assert app.b.data["Single"] is single and "Single" not in app.a.data
    # A series' array is refused, with where to move the series instead
    assert options["validate"]("RecordA0", "RecordA0", "B") == (
        "'RecordA0' is part of data series 'Record'; move the whole series with "
        "Prefix ⋯ › Move to Folder.")


def test_browser_move_to_folder(app, monkeypatch):
    single = app.a.data.new("Single", nparray=np.zeros(4))
    warned = []
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.warning",
                        lambda *a: warned.append(a[2]))
    browser = app.browser_widget
    browser._move_selected_to(app.a.data, [single, app.a.data["RecordA0"]], app.b)
    assert app.b.data["Single"] is single
    assert "RecordA0" in app.a.data and "part of data series 'Record'" in warned[0]


# --- Append Series from another folder (#420) ---------------------------------

def test_append_from_another_folder(app):
    other = app.b.new_dataseries("Record", n_channels=2, n_epochs=2, n_points=4)
    for d in other.channels["A"].data:
        d.nparray = np.full(4, 7.0)
    actions = app.series_actions
    actions.append_action.trigger()  # cancelled: look at the dialog
    options = app.dialogs[0][1]
    assert options["items"] == ["A", "B"]
    assert options["extra_items"][0]("B") == ["Record"]  # the series of the folder chosen
    assert options["extra_items"][0]("A") == ["Stim"]  # not the target itself
    assert options["message"]("B", "Record", "Delete") == (
        "Adds 2 epochs (E3 to E4) from 'Record' in 'B': 4 arrays moved and renamed "
        "(e.g. RecordA0 → RecordA3).")
    app.answers = [("B", None, "Record", "Delete")]
    actions.append_action.trigger()
    record = app.a.dataseries["Record"]
    assert list(record.epochs.keys()) == ["E0", "E1", "E2", "E3", "E4"]
    assert list(app.a.data["RecordA3"].nparray) == [7.0] * 4
    assert app.a.data["RecordA3"]._parent is app.a
    assert "Record" not in app.b.dataseries and len(app.b.data) == 0  # moved over


def test_append_from_another_folder_keeps_the_source(app):
    app.b.new_dataseries("Late", n_channels=2, n_epochs=1, n_points=4)
    app.answers = [("B", None, "Late", "Keep")]
    app.series_actions.append_action.trigger()
    assert "RecordA3" in app.a.data
    assert "Late" in app.b.dataseries and "LateA0" in app.b.data  # copied


def test_append_enabled_by_a_series_in_another_folder(app):
    actions = app.series_actions
    app.a.remove_dataseries("Stim")
    app._on_content_changed()
    assert not actions.append_action.isEnabled()  # nothing anywhere
    app.b.new_dataseries("Late", n_channels=2, n_epochs=1, n_points=4)
    app._on_content_changed()
    assert actions.append_action.isEnabled()


def test_dialog_refills_the_series_for_the_folder(qtbot):
    from pyneuromatic.gui.tier_actions import ChooseItemDialog

    series = {"A": ["Stim"], "B": ["Record", "Late"]}
    dialog = ChooseItemDialog(
        "Append Series", item_label="From folder", items=["A", "B"], current="A",
        extras=[("Append", series["A"], None), ("Source series", ["Keep", "Delete"], "Keep")],
        extra_items={0: lambda folder: series.get(folder, [])})
    qtbot.addWidget(dialog)
    combo = dialog.extra_widgets[0]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Stim"]
    dialog.item_combo.setCurrentText("B")
    assert [combo.itemText(i) for i in range(combo.count())] == ["Record", "Late"]
    assert dialog.extras() == ["Record", "Keep"]
