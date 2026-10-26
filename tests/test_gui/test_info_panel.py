"""Tests for the browser's info panel (second column)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Folder Demo: series Record (2 channels x 3 epochs), Set1 = [E1], E0 in group 0."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    series.epochs.sets.add("Set1", [series.epochs["E1"]])
    series.epochs.groups.assign("E0", 0)
    folder.data.new("Single", nparray=np.array([1.0, -2.5, 4.0]),
                    xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
                    yscale={"label": "Vm", "units": "mV"})
    folder.data.sets.add("Mine", [folder.data["Single"]])
    folder.notes.add("Slice 3, cell 2")
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.folder, window.series = folder, series
    return window


def test_folder_shows_info_not_its_rows(app):
    browser = app.browser_widget
    browser.show_object(app.folder)
    panel = browser.info_panel
    assert browser.is_showing_info
    assert panel.title_label.text() == "Demo  ·  Folder"
    assert panel.row_labels()[0] == "Created"
    assert panel.row_text("Data") == "7 arrays"
    assert panel.row_text("Data Series") == "Record"
    assert panel.row_text("Data Sets") == "Mine"
    assert panel.notes_table.rowCount() == 1
    assert panel.notes_table.item(0, 1).text() == "Slice 3, cell 2"
    date = panel.notes_table.item(0, 0)
    assert "\n" in date.text() and date.toolTip() == app.folder.notes[0]["date"]
    assert not panel.metadata_tree.isVisibleTo(panel)  # nothing imported


def test_group_rows_keep_their_list(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder)  # the Data row
    assert not browser.is_showing_info
    browser.show_data_of(app.series, sets=True, select="Set1", kind="epochs")
    assert not browser.is_showing_info  # a set lists its members (and edit mode)
    assert browser.edit_checkbox.isVisibleTo(browser)


def test_data_array(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="Single")
    panel = browser.info_panel
    assert panel.title_label.text() == "Single  ·  Data Array"
    assert panel.row_text("Points") == "3"
    assert panel.row_text("Type") == "float64 (double)"
    assert panel.row_text("Range") == "-2.5 to 4"
    assert [panel.row_text(label) for label in (
        "X Start", "X Delta", "X Label", "X Units", "Y Label", "Y Units",
    )] == ["0", "0.1", "Time", "ms", "Vm", "mV"]
    assert panel.row_text("Data Series") == "—"
    assert panel.row_text("Sets") == "Mine"
    assert panel.notes_table.isVisibleTo(panel) and panel.notes_table.rowCount() == 0


def test_data_array_in_a_series(app):
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="RecordB1")
    assert browser.info_panel.row_text("Data Series") == "Record / B / E1"


def test_data_series_channel_and_epoch(app):
    browser = app.browser_widget
    panel = browser.info_panel
    browser.show_object(app.series)
    assert panel.row_text("Channels") == "A, B"
    assert panel.row_text("Epochs") == "3"
    assert panel.row_text("Points") == "4"
    assert panel.row_text("Locked") == "No"
    assert panel.row_text("Epoch Sets") == "Set0, Set1, SetX"
    assert panel.row_text("Groups") == "0"
    assert panel.notes_table.isVisibleTo(panel)  # notes for a series, e.g. "Stim"

    browser.show_object(app.series.channels["B"])
    assert panel.title_label.text() == "B  ·  Channel"
    assert panel.row_text("Data") == "3 arrays"

    browser.show_object(app.series.epochs["E1"])
    assert panel.title_label.text() == "E1  ·  Epoch"
    assert panel.row_text("Data") == "A: RecordA1, B: RecordB1"
    assert panel.row_text("Sets") == "Set1"
    assert panel.row_text("Group") == "—"


def test_set_member_shows_the_epoch(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set1", kind="epochs")
    tree = browser.tree.model()
    set1 = browser.tree.currentIndex()
    browser.tree.setCurrentIndex(tree.index(0, 0, set1))  # E1 under Set1
    assert browser.is_showing_info
    assert browser.info_panel.title_label.text() == "E1  ·  Epoch"


def test_add_note(app, monkeypatch):
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="Single")
    panel = browser.info_panel
    monkeypatch.setattr(panel.notes_view, "_ask_note", lambda: "  Baseline drift  ")
    panel.add_note_button.click()
    data = app.folder.data["Single"]
    assert [entry["note"] for entry in data.notes] == ["Baseline drift"]
    assert panel.notes_table.rowCount() == 1
    assert panel.notes_table.item(0, 1).text() == "Baseline drift"

    monkeypatch.setattr(panel.notes_view, "_ask_note", lambda: None)  # cancelled
    panel.add_note_button.click()
    assert len(data.notes) == 1


def test_imported_metadata(app):
    app.folder.metadata["root"] = {"AcqMode": "episodic", "Notes": {"H_Name": "JR"}}
    browser = app.browser_widget
    browser.show_object(app.folder)
    tree = browser.info_panel.metadata_tree
    assert tree.isVisibleTo(browser.info_panel)
    root = tree.topLevelItem(0)
    assert (root.text(0), root.childCount()) == ("root", 2)
    assert not root.isExpanded()


def test_results_folder(app):
    results = app.folder.toolfolders.new("Stats0")
    results.data.new("ST_w0_mean", nparray=np.array([1.0, 2.0]))
    app.browser_widget.refresh()
    browser = app.browser_widget
    browser.show_object(results)
    panel = browser.info_panel
    assert panel.title_label.text() == "Stats0  ·  Results Folder"
    assert panel.row_text("Results") == "ST_w0_mean"


def test_panel_follows_changes(app):
    browser = app.browser_widget
    browser.show_object(app.series.epochs["E2"])
    app.series.epochs.sets.add("Set2", [app.series.epochs["E2"]])
    app._on_content_changed()
    assert browser.info_panel.row_text("Sets") == "Set2"


def test_folders_row_and_current_folder_start_open(app):
    tree = app.browser_widget.tree
    folders = tree.model().index(0, 0)
    assert tree.model().data(folders) == "Folders"
    assert tree.isExpanded(folders)
    assert tree.isExpanded(tree.model().index(0, 0, folders))  # the current folder


def test_only_the_current_folder_starts_open(qtbot):
    manager = NMManager(quiet=True)
    manager.folders.new("Demo1")
    manager.folders.new("Demo2", select=True)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tree = window.browser_widget.tree
    folders = tree.model().index(0, 0)
    opened = [tree.model().data(tree.model().index(row, 0, folders))
              for row in range(tree.model().rowCount(folders))
              if tree.isExpanded(tree.model().index(row, 0, folders))]
    assert opened == ["Demo2"]


def test_add_note_to_a_data_series(app, monkeypatch):
    browser = app.browser_widget
    browser.show_object(app.series)
    panel = browser.info_panel
    monkeypatch.setattr(panel.notes_view, "_ask_note", lambda: "20 Hz train, 10 pulses")
    panel.add_note_button.click()
    assert app.series.notes.note == "20 Hz train, 10 pulses"
    assert panel.notes_table.item(0, 1).text() == "20 Hz train, 10 pulses"


def test_edit_scales_button_only_for_data_arrays(app):
    browser = app.browser_widget
    panel = browser.info_panel
    browser.show_data_of(app.folder, select="Single")
    assert panel.edit_scales_button.isVisibleTo(panel)
    for obj in (app.folder, app.series, app.series.epochs["E0"]):
        browser.show_object(obj)
        assert not panel.edit_scales_button.isVisibleTo(panel)


def test_edit_scales_of_the_array_shown(app, monkeypatch):
    browser = app.browser_widget
    panel = browser.info_panel
    # The browser shows Single, while the strip has another array selected
    app.selection_model.update(data=app.folder.data["RecordA0"], source="data")
    browser.show_data_of(app.folder, select="Single")
    asked = []

    def ask(x, y_rows, context, error):
        asked.append((x, y_rows, context, error))
        return {"xdelta": 0.05, "xunits": "s", "y": {"Y": {"label": "Im", "units": "pA"}}}

    monkeypatch.setattr(app.data_actions, "_ask_scales", ask)
    panel.edit_scales_button.click()
    x, y_rows, context, error = asked[0]
    assert x == {"xstart": "0", "xdelta": "0.1", "xlabel": "Time", "xunits": "ms"}
    assert y_rows == [("Y", "Vm", "mV")]
    assert context == ("Data", "Demo › Single")
    assert error is None

    data = app.folder.data["Single"]
    assert (data.xscale.delta, data.xscale.units) == (0.05, "s")
    assert (data.yscale.label, data.yscale.units) == ("Im", "pA")
    assert app.folder.data["RecordA0"].xscale.delta == 1  # the selected array is untouched
    # The panel shows the new scales
    panel = browser.info_panel
    assert [panel.row_text(label) for label in (
        "X Start", "X Delta", "X Label", "X Units", "Y Label", "Y Units",
    )] == ["0", "0.05", "Time", "s", "Im", "pA"]


def test_edit_scales_cancelled(app, monkeypatch):
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="Single")
    monkeypatch.setattr(app.data_actions, "_ask_scales", lambda *args: None)
    browser.info_panel.edit_scales_button.click()
    assert app.folder.data["Single"].xscale.delta == 0.1


def test_locked_array_has_no_edit_button(app):
    app.series.locked = True
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="RecordA1")
    panel = browser.info_panel
    assert not panel.edit_scales_button.isVisibleTo(panel)
    assert panel.row_text("Locked") == "Yes, with data series 'Record'"
    assert panel.row_text("X Delta") is not None  # the scales are still shown


def test_long_x_label_has_its_own_row(app):
    data = app.folder.data["Single"]
    data.xscale.label = "time after peak"
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="Single")
    assert browser.info_panel.row_text("X Label") == "time after peak"
    browser.show_object(app.series.channels["A"])  # channels too; empty ones read "—"
    assert browser.info_panel.row_text("Y Units") == "—"


@pytest.mark.parametrize("array, text", [
    (np.array([1, 2], dtype=np.int16), "int16 (16-bit integer)"),
    (np.array([1, 2], dtype=np.uint8), "uint8 (8-bit unsigned integer)"),
    (np.array([1.0], dtype=np.float32), "float32 (single)"),
    (np.array([1 + 2j]), "complex128 (complex, double)"),
    (np.array([True]), "bool (true/false)"),
    (np.array(["a", "b"]), "text"),
])
def test_data_array_type(app, array, text):
    app.folder.data.new("Typed", nparray=array)
    browser = app.browser_widget
    browser.refresh()
    browser.show_data_of(app.folder, select="Typed")
    panel = browser.info_panel
    assert panel.row_text("Type") == text
    if array.dtype.kind in "cOUSb":
        assert panel.row_text("Range") is None  # only for real numbers


def test_two_dimensional_points(app):
    app.folder.data.new("Matrix", nparray=np.zeros((100, 4)))
    browser = app.browser_widget
    browser.refresh()
    browser.show_data_of(app.folder, select="Matrix")
    assert browser.info_panel.row_text("Points") == "100 × 4"


def test_epochs_are_hidden_in_the_tree(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, kind="epochs")  # the Epochs row, expanded
    tree, model = browser.tree, browser.tree.model()
    epochs_row = tree.currentIndex()
    tree.expand(epochs_row)
    shown = [model.data(model.index(r, 0, epochs_row)) for r in range(model.rowCount(epochs_row))
             if not tree.isRowHidden(r, epochs_row)]
    assert shown == ["Sets", "Groups"]
    # The second column still lists them
    view = browser.detail
    listed = [model.data(model.index(r, 0, epochs_row)) for r in range(model.rowCount(epochs_row))
              if not view.isRowHidden(r)]
    assert listed == ["E0", "E1", "E2"]
    # Still hidden after a refresh
    browser.refresh()
    epochs_row = tree.currentIndex()
    assert all(tree.isRowHidden(r, epochs_row) for r in range(2, model.rowCount(epochs_row)))


def test_channels_still_listed_in_the_tree(app):
    browser = app.browser_widget
    browser.show_object(app.series.channels["B"])
    tree = browser.tree
    channel_row = tree.currentIndex()
    assert not tree.isRowHidden(channel_row.row(), channel_row.parent())


def test_edit_scales_of_a_channel(app, monkeypatch):
    browser = app.browser_widget
    browser.show_object(app.series.channels["B"])
    panel = browser.info_panel
    assert panel.edit_scales_button.isVisibleTo(panel)
    asked = []

    def ask(x, y_rows, context, error):
        asked.append((y_rows, context, error))
        return {"xunits": "ms", "y": {"Y": {"label": "Im", "units": "pA"}}}

    monkeypatch.setattr(app.series_actions, "_ask_scales", ask)
    panel.edit_scales_button.click()
    y_rows, context, error = asked[0]
    assert context == ("Channel", "Demo › Record › B") and error is None
    assert len(y_rows) == 1
    for name in ("RecordB0", "RecordB2"):  # its arrays follow
        data = app.folder.data[name]
        assert (data.xscale.units, data.yscale.label, data.yscale.units) == ("ms", "Im", "pA")
    assert app.folder.data["RecordA0"].yscale.units != "pA"  # channel A untouched
    assert browser.info_panel.row_text("Y Units") == "pA"
    channel = app.series.channels["B"]  # its template for new arrays too
    assert (channel.xscale.units, channel.yscale.label, channel.yscale.units) == ("ms", "Im", "pA")


def test_channel_shows_mixed_scales(app):
    app.folder.data["RecordA1"].xscale.delta = 0.5
    browser = app.browser_widget
    browser.show_object(app.series.channels["A"])
    assert browser.info_panel.row_text("X Delta") == "(mixed)"
    assert browser.info_panel.row_text("X Start") == "0"


def test_locked_channel_has_no_edit_button(app):
    app.series.locked = True
    browser = app.browser_widget
    browser.show_object(app.series.channels["A"])
    panel = browser.info_panel
    assert not panel.edit_scales_button.isVisibleTo(panel)
    assert panel.row_text("Locked") == "Yes, with data series 'Record'"


def _row_style(panel, label):
    """The style sheet of *label*'s value."""
    from PyQt6 import QtWidgets

    form = panel._form
    for row in range(form.rowCount()):
        name = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
        if name is not None and name.widget().text() == label:
            return form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole).widget().styleSheet()
    return None


def test_locked_is_red(app):
    from pyneuromatic.gui.info_panel import LOCKED_COLOR

    browser = app.browser_widget
    panel = browser.info_panel
    browser.show_object(app.series)
    assert _row_style(panel, "Locked") == ""  # "No"
    app.series.locked = True
    browser.refresh_info()
    assert panel.row_text("Locked") == "Yes"
    assert _row_style(panel, "Locked") == f"color: {LOCKED_COLOR};"
    browser.show_object(app.series.channels["A"])  # its channels and arrays too
    assert _row_style(panel, "Locked") == f"color: {LOCKED_COLOR};"
    browser.show_data_of(app.folder, select="RecordA1")
    assert _row_style(panel, "Locked") == f"color: {LOCKED_COLOR};"
