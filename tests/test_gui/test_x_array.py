"""Tests for Edit Scales' "X from": x values from another array."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tier_actions import EditScalesDialog, x_array_choices

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record: channels A, B x epochs E0, E1 (4 points, y = 1..4);
    arrays Times (0, 1.5, 2, 7), Short (2 points) and Back (not increasing)."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    folder.new_dataseries("Record", n_channels=2, n_epochs=2, n_points=4)
    for d in folder.dataseries["Record"].channels["A"].data:
        d.nparray = np.array([1.0, 2.0, 3.0, 4.0])
    folder.data.new("Times", nparray=np.array([0, 1.5, 2, 7]))
    folder.data.new("Short", nparray=np.array([0.0, 1.0]))
    folder.data.new("Back", nparray=np.array([0.0, 2.0, 1.0, 3.0]))
    manager.select_value_set(folder)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    series = folder.dataseries["Record"]
    window.selection_model.update(dataseries=series, channel=series.channels["A"])
    window.folder, window.series = folder, series
    return window


def _ask(app, monkeypatch, actions, answer):
    """Replace Edit Scales: keep what it was given, return *answer*."""
    seen = {}

    def ask(x, y_rows, context, error, **x_options):
        seen.update(x_options)
        return answer(x_options) if callable(answer) else answer

    monkeypatch.setattr(actions, "_ask_scales", ask)
    return seen


def test_choices_fit_the_arrays(app):
    arrays = [d for c in app.series.channels.values() for d in c.data]
    # Same number of points, numeric, not the series' own arrays
    assert x_array_choices(app.folder, arrays) == ["Times", "Back"]


def test_series_x_from_an_array(app, monkeypatch):
    actions = app.series_actions
    seen = _ask(app, monkeypatch, actions, {"xarray": "Times"})
    actions.scales_action.trigger()
    assert seen["x_arrays"] == ["Times", "Back"] and seen["x_array"] == ""
    assert seen["validate_x"]("Back") == "X array 'Back' must increase from point to point"
    assert seen["validate_x"]("Times") is None
    data = app.folder.data["RecordB1"]
    assert list(data.xarray) == [0.0, 1.5, 2.0, 7.0] and data.xarray_name == "Times"
    # Plotted at those x values
    items = app.plot_widget.plot_widget.getPlotItem().listDataItems()
    assert list(items[0].xData) == [0.0, 1.5, 2.0, 7.0]
    # Back to start and delta
    seen = _ask(app, monkeypatch, actions, {"xarray": ""})
    actions.scales_action.trigger()
    assert seen["x_array"] == "Times"
    assert data.xarray is None


def test_channel_x_from_an_array(app, monkeypatch):
    actions = app.series_actions
    _ask(app, monkeypatch, actions, {"xarray": "Times"})
    actions.edit_channel_scales(app.series.channels["B"])
    assert all(d.xarray_name == "Times" for d in app.series.channels["B"].data)
    assert all(d.xarray is None for d in app.series.channels["A"].data)
    seen = _ask(app, monkeypatch, actions, None)
    actions.scales_action.trigger()
    assert seen["x_array"] is None  # A and B differ: mixed


def test_times_is_the_x_array_itself(app, monkeypatch):
    times = app.folder.data["Times"]
    _ask(app, monkeypatch, app.series_actions, {"xarray": "Times"})
    app.series_actions.scales_action.trigger()
    data = app.folder.data["RecordA0"]
    assert data.xarray_source is times
    times.nparray[3] = 9.0  # editing Times moves the points
    app._on_data_edited(times)
    items = app.plot_widget.plot_widget.getPlotItem().listDataItems()
    assert list(items[0].xData) == [0.0, 1.5, 2.0, 9.0]


def test_x_array_in_use_cant_be_deleted(app, monkeypatch):
    app.series.set_xarray(app.folder.data["Times"])
    # Data ⋯ › Delete: refused in the dialog, with the reason
    asked = []
    monkeypatch.setattr(app.data_actions, "_choose",
                        lambda title, **options: asked.append(options) or None)
    app.data_actions.delete_data()
    error = asked[0]["validate"]("Times", None)
    assert error == ("'Times' is the x array of data series 'Record' and cannot be "
                     "deleted while in use.")
    # The browser: refused too, unless its users are deleted with it
    browser = app.browser_widget
    warned = []
    monkeypatch.setattr(browser, "_confirm", lambda *a: True)
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.warning",
                        lambda *a: warned.append(a[2]))
    data = app.folder.data
    browser._delete_objects(data, [data["Times"]])
    assert "Times" in data and "x array of data series 'Record'" in warned[0]
    browser._delete_objects(data, [data["Times"]] + [d for c in app.series.channels.values()
                                                    for d in c.data])
    assert "Times" not in data and "RecordA0" not in data


def test_locked_series_protects_its_x_array_in_the_table(app):
    from pyneuromatic.gui.context_views import ChannelTableModel

    times = app.folder.data["Times"]
    app.series.set_xarray(times)
    assert not ChannelTableModel.is_locked(times)
    app.series.locked = True
    assert ChannelTableModel.is_locked(times)  # its values can't be edited


def test_info_panel_names_the_x_array(app):
    browser = app.browser_widget
    browser.show_object(app.series.channels["A"])
    panel = browser.info_panel
    assert panel.row_text("X Array") is None  # start and delta
    app.series.set_xarray(app.folder.data["Times"], channel="A")
    browser.refresh_info()
    assert panel.row_text("X Array") == "Times"
    browser.show_object(app.series)
    assert panel.row_text("X Array") == "(mixed)"  # B uses start and delta
    browser.show_data_of(app.folder, select="RecordA1")
    assert panel.row_text("X Array") == "Times"
    browser.show_data_of(app.folder, select="Times")
    assert panel.row_text("Data Series") == "Record (x array)"  # why it can't be deleted
    # Times stops fitting: said, and start and delta are used meanwhile
    app.folder.data["Times"].nparray = np.array([0.0, 1.0])
    browser.show_data_of(app.folder, select="RecordA1")
    assert panel.row_text("X Array").startswith("Times: doesn't fit (x array 'Times' has 2")


# --- the dialog -----------------------------------------------------------------

def _dialog(qtbot, **options):
    dialog = EditScalesDialog({"xstart": "0", "xdelta": "1", "xlabel": "", "xunits": ""},
                              [("Y", "", "")], **options)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_x_from(qtbot):
    dialog = _dialog(qtbot, x_arrays=["Times", "Back"], x_array="",
                     validate_x=lambda name: "no" if name == "Back" else None)
    combo = dialog.x_from_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "None", "Times", "Back"]
    assert dialog.changes() == {}
    combo.setCurrentText("Times")
    assert not dialog.x_edits["xstart"].isEnabled()  # not used with an array
    assert dialog.changes() == {"xarray": "Times"}
    combo.setCurrentText("Back")
    dialog.accept()
    assert dialog.error_label.text() == "no"


def test_dialog_mixed_and_kept_copy(qtbot):
    mixed = _dialog(qtbot, x_arrays=["Times"], x_array=None)
    assert mixed.x_from_combo.currentText() == "(mixed)" and mixed.changes() == {}
    kept = _dialog(qtbot, x_arrays=["Times"], x_array="Gone")
    assert kept.x_from_combo.currentText() == "Gone"  # its source was deleted
    plain = _dialog(qtbot)
    assert plain.x_from_combo is None  # e.g. no x arrays offered


def test_browser_edit_scales_for_a_series(app, monkeypatch):
    other = app.folder.new_dataseries("Other", n_channels=1, n_epochs=1, n_points=4)
    seen = _ask(app, monkeypatch, app.series_actions, {"xunits": "s"})
    browser = app.browser_widget
    browser.show_object(other)  # not the strip's series
    panel = browser.info_panel
    assert panel.edit_scales_button.isVisibleTo(panel)
    panel.edit_scales_button.click()
    assert "x_arrays" in seen
    assert app.folder.data["OtherA0"].xscale.units == "s"
    assert app.folder.data["RecordA0"].xscale.units == ""  # the strip's series unchanged
    other.locked = True
    browser.refresh_info()
    assert not panel.edit_scales_button.isVisibleTo(panel)


# --- x label and units from the array ----------------------------------------

def test_dialog_fills_x_label_and_units_from_the_array(qtbot):
    dialog = _dialog(qtbot, x_arrays=["Times", "Bare"], x_array="",
                     x_array_scales={"Times": ("Time", "s"), "Bare": ("", "")})
    dialog.x_edits["xunits"].setText("ms")
    combo, note = dialog.x_from_combo, dialog.x_units_note
    combo.setCurrentText("Times")
    assert dialog.x_edits["xlabel"].text() == "Time" and dialog.x_edits["xunits"].text() == "s"
    assert note.isHidden()  # filled, so they match
    assert dialog.changes() == {"xarray": "Times", "xlabel": "Time", "xunits": "s"}
    dialog.x_edits["xunits"].setText("ms")  # edited back: noted, not blocked
    assert not note.isHidden() and note.text() == "Times is in s; X Units say ms."
    combo.setCurrentText("Bare")  # no label or units: the fields are kept
    assert dialog.x_edits["xlabel"].text() == "Time" and dialog.x_edits["xunits"].text() == "ms"
    assert note.isHidden()
    combo.setCurrentText("None")  # back: the label and units are kept
    assert dialog.x_edits["xunits"].text() == "ms" and note.isHidden()


def test_series_dialog_gets_the_arrays_scales(app, monkeypatch):
    app.folder.data["Times"].yscale.label = "Time"
    app.folder.data["Times"].yscale.units = "ms"
    seen = _ask(app, monkeypatch, app.series_actions, None)
    app.series_actions.scales_action.trigger()
    assert seen["x_array_scales"] == {"Times": ("Time", "ms"), "Back": ("", "")}


# --- Data ⋯ › Edit Scales: one array ---------------------------------------------

def test_data_edit_scales_x_array_for_a_plain_array(app, monkeypatch):
    single = app.folder.data.new("Single", nparray=np.array([5.0, 6.0, 7.0, 8.0]))
    actions = app.data_actions
    seen = _ask(app, monkeypatch, actions, {"xarray": "Times"})
    actions.edit_scales(single)
    assert "Times" in seen["x_arrays"] and "Single" not in seen["x_arrays"]
    assert seen["x_array"] == "" and seen["x_array_fixed"] is None
    assert single.xarray_source is app.folder.data["Times"]
    assert ("'Times' is the x array of 'Single' and cannot be deleted while in use"
            == app.folder.data.xarray_in_use_text("Times"))
    seen = _ask(app, monkeypatch, actions, {"xarray": ""})  # back to None
    actions.edit_scales(single)
    assert seen["x_array"] == "Times"
    assert single.xarray_source is None


def test_data_edit_scales_of_a_series_array_shows_the_series_choice(app, monkeypatch):
    app.series.set_xarray(app.folder.data["Times"])
    data = app.folder.data["RecordA0"]
    actions = app.data_actions
    seen = _ask(app, monkeypatch, actions, {"xarray": ""})  # ignored
    actions.edit_scales(data)
    assert seen["x_array"] == "Times" and seen["x_arrays"] == []
    assert seen["x_array_fixed"] == "Set by data series 'Record' (its Edit Scales)"
    assert data.xarray_source is app.folder.data["Times"]  # unchanged


def test_dialog_fixed_x_array(qtbot):
    dialog = _dialog(qtbot, x_arrays=[], x_array="Times",
                     x_array_fixed="Set by data series 'Record' (its Edit Scales)")
    assert not dialog.x_from_combo.isEnabled()
    assert dialog.x_from_combo.currentText() == "Times"
    assert dialog.x_array_fixed_note.text() == "Set by data series 'Record' (its Edit Scales)"
    assert not dialog.x_edits["xstart"].isEnabled()  # x from the array
    assert dialog.changes() == {}


def test_browser_info_for_a_plain_array_with_an_x_array(app):
    single = app.folder.data.new("Single", nparray=np.zeros(4))
    single.set_xarray_from(app.folder.data["Times"])
    browser = app.browser_widget
    browser.show_data_of(app.folder, select="Single")
    assert browser.info_panel.row_text("X Array") == "Times"
