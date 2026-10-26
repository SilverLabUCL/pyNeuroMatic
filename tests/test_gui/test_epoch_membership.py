"""Tests for the plot's Set1/Set2/SetX checkboxes and Group dropdown."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record of 6 epochs; Set1 = [E1]; E0-E3 in groups 0/1/0/1."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_epochs=6, n_points=3)
    epochs = series.epochs
    epochs.sets.add("Set1", [epochs["E1"]])
    epochs.groups.assign_cyclic(["E0", "E1", "E2", "E3"], n_groups=2)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series)
    window.series = series
    return window


def _checked(panel):
    return [name for name, box in panel.set_checkboxes.items() if box.isChecked()]


def test_boxes_follow_the_current_epoch(app):
    panel = app.plot_widget
    assert panel.membership_row.isVisibleTo(panel)
    assert list(panel.set_checkboxes) == ["Set0", "Set1", "SetX"]
    assert panel.current_epoch().name == "E0"
    assert _checked(panel) == [] and panel.group_combo.currentText() == "0"
    assert [panel.group_combo.itemText(i) for i in range(panel.group_combo.count())] == [
        "", "0", "1",
    ]

    panel._step_trace(1)
    assert panel.current_epoch().name == "E1"
    assert _checked(panel) == ["Set1"] and panel.group_combo.currentText() == "1"

    panel._step_trace(4)  # E5: no sets, no group
    assert _checked(panel) == [] and panel.group_combo.currentText() == ""


def test_ticking_adds_and_removes_the_current_epoch(app):
    panel, series = app.plot_widget, app.series
    panel._step_trace(2)  # E2
    panel.set_checkboxes["Set0"].setChecked(True)
    assert series.epochs.sets.get_items("Set0", get_keys=True) == ["E2"]
    panel.set_checkboxes["SetX"].setChecked(True)
    assert "E2" in series.epochs.excluded_names()
    panel.set_checkboxes["Set0"].setChecked(False)
    assert series.epochs.sets.get_items("Set0", get_keys=True) == []

    # The browser and the strip follow
    app.selection_model.update(epoch_set="SetX")
    assert app.selection_strip.epoch_count_label.text() == "n=1"
    assert panel.set_checkboxes["SetX"].isChecked()  # still on E2


def test_group_dropdown_moves_the_current_epoch(app):
    panel, groups = app.plot_widget, app.series.epochs.groups
    panel._step_trace(2)  # E2, group 0
    panel.group_combo.setCurrentIndex(panel.group_combo.findData(1))
    assert groups.get_group("E2") == 1
    panel.group_combo.setCurrentIndex(0)  # the blank: no group
    assert groups.get_group("E2") is None
    assert groups.group_numbers == [0, 1]

    # Moving a group's last epoch keeps the (empty) group
    panel._step_trace(-2)  # E0, the last epoch of group 0
    panel.group_combo.setCurrentIndex(panel.group_combo.findData(1))
    assert groups.get_items(0) == [] and groups.group_numbers == [0, 1]


def test_group_dropdown_hidden_without_groups(app):
    app.series.epochs.groups.clear()
    app.selection_model.refresh()
    app.plot_widget.update_selection(app.selection_model.selection)
    panel = app.plot_widget
    assert panel.membership_row.isVisibleTo(panel)
    assert not panel.group_combo.isVisibleTo(panel)


def test_usable_with_overlay_hidden_for_data_without_sets(app):
    panel = app.plot_widget
    panel.overlay_action.setChecked(True)  # still the current (blue) epoch
    assert panel.membership_row.isVisibleTo(panel)
    assert panel.membership_row.isEnabled()
    panel.set_checkboxes["Set0"].setChecked(True)
    assert app.series.epochs.sets.get_items("Set0", get_keys=True) == ["E0"]

    folder = app.selection_model.selection["folder"]
    folder.data.new("Single", nparray=np.array([1.0, 2.0]))
    app.selection_model.update(data=folder.data["Single"], source="data")
    assert not panel.membership_row.isVisibleTo(panel)


def test_combined_set_box_is_disabled(app):
    epochs = app.series.epochs
    del epochs.sets["Set0"]
    epochs.sets.define_or("Set0", "Set1", "SetX")
    app.plot_widget.update_selection(app.selection_model.selection)
    assert not app.plot_widget.set_checkboxes["Set0"].isEnabled()


def test_a_deleted_default_set_is_made_again(app):
    epochs = app.series.epochs
    del epochs.sets["Set0"]
    app.plot_widget.set_checkboxes["Set0"].setChecked(True)
    assert epochs.sets.get_items("Set0", get_keys=True) == ["E0"]


def test_browser_edit_mode_follows_the_plot(app):
    browser = app.browser_widget
    browser.show_data_of(app.series, sets=True, select="Set1", kind="epochs")
    browser.edit_checkbox.setChecked(True)
    model = browser.membership_model
    app.plot_widget.set_checkboxes["Set1"].setChecked(True)  # E0
    assert browser.membership_model is model
    state = model.data(model.index(0, 0), QtCore.Qt.ItemDataRole.CheckStateRole)
    assert state == QtCore.Qt.CheckState.Checked


def test_graph_menu(app):
    panel = app.plot_widget
    if panel.plot_widget is None:
        pytest.skip("needs pyqtgraph")
    menu = panel.graph_button.menu()
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts == ["Overlay Traces", "Grid", "Rescale", "Filter && Transform..."]  # shows "&"
    assert panel.overlay_action.isEnabled() and not panel.overlay_limit.isEnabled()
    panel.overlay_action.setChecked(True)
    assert panel.overlay_limit.isEnabled()
    assert panel.overlay_status_label.text() == "Showing 6 of 6 traces"

    panel.grid_action.setChecked(True)
    assert panel.plot_widget.getAxis("bottom").grid
    panel.grid_action.setChecked(False)
    assert not panel.plot_widget.getAxis("bottom").grid

    view = panel.plot_widget.getViewBox()
    panel.plot_widget.setXRange(0, 0.5)  # zooming turns autoscale off
    assert not view.state["autoRange"][0]
    panel.rescale_action.trigger()
    assert all(view.state["autoRange"])


def test_overlay_unavailable_for_data_arrays(app):
    folder = app.selection_model.selection["folder"]
    folder.data.new("Single", nparray=np.array([1.0, 2.0]))
    app.selection_model.update(data=folder.data["Single"], source="data")
    assert not app.plot_widget.overlay_action.isEnabled()
    assert not app.plot_widget.overlay_limit.isEnabled()


def test_selection_is_shown_in_the_window_title(app):
    app.plot_widget._step_trace(2)
    assert app.windowTitle() == "pyNeuroMatic — Demo / Record / A / All Epochs"
    app.selection_model.update(epoch_set="Current Epoch")
    assert app.windowTitle() == "pyNeuroMatic — Demo / Record / A / E2"
    app.selection_model.update(epoch_set="Set1")
    assert app.windowTitle() == "pyNeuroMatic — Demo / Record / A / Epoch Set1"
    assert not hasattr(app.plot_widget, "selection_label")  # more room to plot


def test_set_app_name():
    from PyQt6 import QtCore

    from pyneuromatic.gui.app_window import set_app_name

    set_app_name()
    assert QtCore.QCoreApplication.applicationName() == "pyNeuroMatic"


@pytest.fixture
def data_app(qtbot):
    """Folder with data A0-A3 and data sets Control=[A1], Drug, Washout, Mine."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for name in ("A0", "A1", "A2", "A3"):
        folder.data.new(name, nparray=np.array([1.0, 2.0]))
    folder.data.sets.add("Control", [folder.data["A1"]])
    for name in ("Drug", "Washout", "Mine"):
        folder.data.sets.add(name)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(data=folder.data["A0"], source="data")
    window.folder = folder
    return window


def test_data_set_boxes_are_the_first_three_sets(data_app):
    panel = data_app.plot_widget
    assert panel.membership_row.isVisibleTo(panel)
    assert list(panel.set_checkboxes) == ["Control", "Drug", "Washout"]
    assert not panel.group_combo.isVisibleTo(panel)  # groups are for epochs
    assert _checked(panel) == []
    panel._step_trace(1)  # A1
    assert _checked(panel) == ["Control"]


def test_the_strip_data_set_gets_a_box_too(data_app):
    panel = data_app.plot_widget
    data_app.selection_model.update(data_set="Mine")
    assert list(panel.set_checkboxes) == ["Control", "Drug", "Washout", "Mine"]
    data_app.selection_model.update(data_set="Drug")  # already shown
    assert list(panel.set_checkboxes) == ["Control", "Drug", "Washout"]
    # The old boxes are gone at once, not left drawn underneath
    boxes = panel.membership_row.findChildren(QtWidgets.QCheckBox)
    assert sorted(box.text() for box in boxes) == ["Control", "Drug", "Washout"]


def test_ticking_adds_the_current_data_array(data_app):
    panel, sets = data_app.plot_widget, data_app.folder.data.sets
    panel._step_trace(2)  # A2
    panel.set_checkboxes["Drug"].setChecked(True)
    assert sets.get_items("Drug", get_keys=True) == ["A2"]
    panel.set_checkboxes["Control"].setChecked(True)
    assert sets.get_items("Control", get_keys=True) == ["A1", "A2"]
    panel.set_checkboxes["Drug"].setChecked(False)
    assert sets.get_items("Drug", get_keys=True) == []
    assert panel.set_checkboxes["Control"].isChecked()  # still on A2


def test_no_data_sets_no_boxes(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    folder.data.new("A0", nparray=np.array([1.0]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(data=folder.data["A0"], source="data")
    assert not window.plot_widget.membership_row.isVisibleTo(window.plot_widget)

    # A new set (e.g. from the browser) appears
    folder.data.sets.add("Control")
    window._on_content_changed()
    assert list(window.plot_widget.set_checkboxes) == ["Control"]


def test_switching_between_data_and_epochs(data_app):
    panel, folder = data_app.plot_widget, data_app.folder
    series = folder.new_dataseries("Record", n_epochs=3, n_points=3)
    data_app.selection_model.update(dataseries=series, source="dataseries")
    assert list(panel.set_checkboxes) == ["Set0", "Set1", "SetX"]
    data_app.selection_model.update(source="data")
    assert list(panel.set_checkboxes) == ["Control", "Drug", "Washout"]


def test_the_strip_epoch_set_gets_a_box_too(app):
    panel, epochs = app.plot_widget, app.series.epochs
    epochs.sets.add("Mine", [epochs["E0"]])
    app.selection_model.update(epoch_set="Mine")
    assert list(panel.set_checkboxes) == ["Set0", "Set1", "SetX", "Mine"]
    assert panel.set_checkboxes["Mine"].isChecked()  # E0
    panel._step_trace(1)
    panel.set_checkboxes["Mine"].setChecked(True)
    assert epochs.sets.get_items("Mine", get_keys=True) == ["E0", "E1"]

    for chosen in ("All", "Current Epoch", "Set0"):  # nothing extra
        app.selection_model.update(epoch_set=chosen)
        assert list(panel.set_checkboxes) == ["Set0", "Set1", "SetX"]


def test_narrow_group_and_channel_dropdowns(app):
    strip = app.selection_strip
    assert strip.combo("group").width() == 66 and strip.combo("channel").width() == 66
    assert app.plot_widget.group_combo.width() == 56
