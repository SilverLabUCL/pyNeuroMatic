"""Tests for the trace navigator's jump buttons, step size and Selection only."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record of 10 epochs; Set1 = [E1, E3, E7]; groups 0/1 cyclic."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_epochs=10, n_points=3)
    epochs = series.epochs
    epochs.sets.add("Set1", [epochs["E1"], epochs["E3"], epochs["E7"]])
    epochs.groups.assign_cyclic(list(epochs.keys()), n_groups=2)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series)
    window.series = series
    return window


def _at(panel):
    return panel.current_item().name


def _enabled(panel):
    return [button.isEnabled() for button in (
        panel.trace_first_button, panel.trace_previous_button,
        panel.trace_next_button, panel.trace_last_button,
    )]


def test_jump_to_first_and_last(app):
    panel = app.plot_widget
    assert _at(panel) == "E0"
    assert _enabled(panel) == [False, False, True, True]
    panel.trace_last_button.click()
    assert _at(panel) == "E9"
    assert _enabled(panel) == [True, True, False, False]
    panel.trace_first_button.click()
    assert _at(panel) == "E0"


def test_step_size(app):
    panel = app.plot_widget
    panel.step_size.setValue(3)
    panel.trace_next_button.click()
    assert _at(panel) == "E3"
    panel.trace_next_button.click()
    panel.trace_next_button.click()
    assert _at(panel) == "E9"  # 6 + 3
    panel.trace_previous_button.click()
    assert _at(panel) == "E6"
    panel.step_size.setValue(5)
    panel.trace_next_button.click()  # past the end: goes to the last
    assert _at(panel) == "E9"
    panel.trace_index.setValue(2)
    panel.trace_previous_button.click()  # past the start: goes to the first
    assert _at(panel) == "E0"


def test_selection_only_steps_through_the_epoch_set(app):
    panel = app.plot_widget
    assert panel.selection_only_action.isEnabled()  # a series starts on All
    assert panel.selection_scope_label.text() == ""  # shown only while stepping it
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(All: 10)"
    app.selection_model.update(epoch_set="Set1")
    assert panel.selection_scope_label.text() == "(Set1: 3)"

    panel.trace_next_button.click()
    assert _at(panel) == "E1"
    panel.trace_next_button.click()
    assert _at(panel) == "E3"
    panel.trace_last_button.click()
    assert _at(panel) == "E7"
    assert _enabled(panel) == [True, True, False, False]
    panel.trace_first_button.click()
    assert _at(panel) == "E1"

    panel.step_size.setValue(2)
    panel.trace_next_button.click()
    assert _at(panel) == "E7"  # every other Set1 epoch

    # From an epoch outside the set, the nearest set epoch in that direction
    panel.step_size.setValue(1)
    panel.trace_index.setValue(5)
    panel.trace_previous_button.click()
    assert _at(panel) == "E3"
    panel.trace_index.setValue(5)
    panel.trace_next_button.click()
    assert _at(panel) == "E7"


def test_selection_only_with_a_group_and_and(app):
    panel = app.plot_widget
    app.selection_model.update(epoch_set=None, group=1)
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(Group 1: 5)"
    panel.trace_last_button.click()
    assert _at(panel) == "E9"
    panel.trace_previous_button.click()
    assert _at(panel) == "E7"

    # Set1 AND Group 1 = E1, E3, E7 (all odd)
    app.selection_model.update(epoch_set="Set1", group=1, group_operator="AND")
    assert panel.selection_scope_label.text() == "(Set1 AND Group 1: 3)"


def test_selection_only_skips_setx(app):
    panel, epochs = app.plot_widget, app.series.epochs
    epochs.sets.add("SetX", [epochs["E1"], epochs["E2"]])
    app.selection_model.refresh()
    app.plot_widget.update_selection(app.selection_model.selection)
    assert app.selection_model.selection["epoch_set"] == "All"
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(All: 8)"
    panel.trace_next_button.click()
    assert _at(panel) == "E3"  # E1, E2 excluded


def test_no_selection_to_step_through(app):
    panel = app.plot_widget
    app.selection_model.update(epoch_set="Current Epoch")
    assert not panel.selection_only_action.isEnabled()
    panel.selection_only_action.setChecked(True)  # ignored: steps every epoch
    assert panel.selection_scope_label.text() == ""
    panel.trace_next_button.click()
    assert _at(panel) == "E1"

    # A Set and a Group with no AND/OR chosen yet
    app.selection_model.update(epoch_set="Set1", group=0)
    assert not panel.selection_only_action.isEnabled()


def test_unticking_the_current_epoch_while_stepping_the_set(app):
    panel, epochs = app.plot_widget, app.series.epochs
    app.selection_model.update(epoch_set="Set1")
    panel.selection_only_action.setChecked(True)
    panel.trace_next_button.click()
    panel.trace_next_button.click()
    assert _at(panel) == "E3"
    panel.set_checkboxes["Set1"].setChecked(False)  # E3 out of Set1
    assert _at(panel) == "E3"  # still shown
    assert panel.selection_scope_label.text() == "(Set1: 2)"
    panel.trace_next_button.click()
    assert _at(panel) == "E7"
    panel.trace_previous_button.click()
    assert _at(panel) == "E1"


def test_selection_only_for_data_arrays(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for index in range(6):
        folder.data.new(f"A{index}", nparray=np.array([1.0, 2.0]))
    folder.data.sets.add("Mine", [folder.data["A2"], folder.data["A5"]])
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(data=folder.data["A0"], source="data")
    panel = window.plot_widget
    assert not panel.selection_only_action.isEnabled()  # no Data Set chosen

    window.selection_model.update(data_set="Mine")
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(Mine: 2)"
    panel.trace_next_button.click()
    assert _at(panel) == "A2"
    panel.trace_next_button.click()
    assert _at(panel) == "A5"
    assert not panel.trace_next_button.isEnabled()


def test_stepping_options_are_in_a_menu_beside_the_buttons(app):
    panel = app.plot_widget
    assert panel.step_button.menu() is panel.step_menu
    texts = [a.text() for a in panel.step_menu.actions() if a.text()]
    assert texts == ["Selection Only"]
    assert panel.step_size.window() is panel.step_menu
    # Turning Selection only off hides the reminder
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(All: 10)"
    panel.selection_only_action.setChecked(False)
    assert panel.selection_scope_label.text() == ""
