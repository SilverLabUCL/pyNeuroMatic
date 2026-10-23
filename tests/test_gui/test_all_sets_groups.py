"""Tests for "All Sets" and "All Groups": Stats runs once per set or group."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.selection_model import ALL_GROUPS, ALL_SETS, scoped_epoch_names

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record of 6 epochs (mean = epoch number): Set0 = E1, E3;
    Set1 = E2; SetX = E3; groups 0 / 1 cyclic."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(6):
        folder.data.new(f"RecordA{epoch}", nparray=np.array([float(epoch)] * 2),
                        xscale={"start": 0.0, "delta": 1.0})
    series = folder.sync_dataseries("Record", select=True)
    sets = series.epochs.sets
    sets.add("Set0", ["E1", "E3"])
    sets.add("Set1", ["E2"])
    sets.add("SetX", ["E3"])
    series.epochs.groups.assign_cyclic(list(series.epochs.keys()), n_groups=2)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.stats_tab.measurement_combo.setCurrentText("mean")
    window.folder, window.series = folder, series
    return window


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _means(folder):
    return list(folder.data["ST_w0_mean_y"].nparray)


def test_options_in_the_strip(app):
    strip = app.selection_strip
    assert "All Sets" in _items(strip.combo("epoch_set"))
    assert _items(strip.combo("group")) == ["", "All", "0", "1"]  # All = All Groups
    # AND / OR for All Sets with a group, as for a stored set
    app.selection_model.update(epoch_set=ALL_SETS, group=0)
    assert _items(strip.combo("group_operator")) == ["", "AND", "OR"]


def test_union_for_the_other_views(app):
    series = app.series
    selection = app.selection_model
    selection.update(epoch_set=ALL_SETS)
    # Set0 + Set1 = E1, E2, E3; SetX takes E3 out
    assert scoped_epoch_names(series, selection.selection) == ["E1", "E2"]
    label = app.selection_strip.epoch_count_label
    assert "n=3" in label.text()
    assert "Set0: 1, Set1: 1" in label.toolTip()  # per run, SetX left out
    assert app.windowTitle().endswith("All Sets")

    selection.update(epoch_set=None, group=ALL_GROUPS)
    assert scoped_epoch_names(series, selection.selection) == ["E0", "E1", "E2", "E4", "E5"]
    assert app.windowTitle().endswith("All Groups")


def test_run_all_sets(app):
    tab = app.stats_tab
    app.series.epochs.sets.add("Set2")  # empty: skipped
    app.selection_model.update(epoch_set=ALL_SETS)
    assert tab.run_info_label.text() == "n=2 runs"
    assert "Skipped (no epochs): Set2" in tab.run_info_label.toolTip()
    tab.run_button.click()

    folders = app.folder.toolfolders
    assert _means(folders["Stats_Record_A_S0_0"]) == [1.0]  # E3 excluded (SetX)
    assert _means(folders["Stats_Record_A_S1_0"]) == [2.0]
    assert "Stats_Record_A_S2_0" not in folders
    assert any("Skipped" in w and "Set2" in w for w in tab.warnings)
    # Stats2 shows the last run's folder, and lists each run's
    stats2 = tab.stats2
    assert stats2.folder_combo.currentText() == "Stats_Record_A_S1_0"
    assert {"Stats_Record_A_S0_0", "Stats_Record_A_S1_0"} <= set(_items(stats2.folder_combo))


def test_run_all_groups_with_a_set(app):
    tab = app.stats_tab
    app.selection_model.update(epoch_set=None, group=ALL_GROUPS)
    tab.run_button.click()
    folders = app.folder.toolfolders
    assert _means(folders["Stats_Record_A_G0_0"]) == [0.0, 2.0, 4.0]
    assert _means(folders["Stats_Record_A_G1_0"]) == [1.0, 5.0]  # E3 excluded

    # Set0 AND each group
    app.selection_model.update(epoch_set="Set0", group_operator="AND")
    assert app.selection_model.selection["group"] == ALL_GROUPS
    tab.run_button.click()
    assert _means(folders["Stats_Record_A_S0andG1_0"]) == [1.0]
    assert "Stats_Record_A_S0andG0_0" not in folders  # no Set0 epoch in group 0


def test_all_sets_with_all_groups_is_refused(app):
    tab = app.stats_tab
    app.selection_model.update(epoch_set=ALL_SETS, group=ALL_GROUPS, group_operator="AND")
    assert "isn't supported yet" in tab.run_info_label.text()
    before = list(app.folder.toolfolders.keys())
    tab.run_button.click()
    assert list(app.folder.toolfolders.keys()) == before


def test_all_groups_kept_until_the_groups_go(app):
    selection = app.selection_model
    selection.update(group=ALL_GROUPS)
    app.series.epochs.groups.delete_group(1)
    app._on_browser_content_changed()
    assert selection.selection["group"] == ALL_GROUPS  # group 0 is left
    app.series.epochs.groups.clear()
    app._on_browser_content_changed()
    assert selection.selection["group"] is None


def test_stepping_through_the_union(app):
    panel = app.plot_widget
    app.selection_model.update(epoch_set=ALL_SETS)
    panel.selection_only_action.setChecked(True)
    assert panel.selection_scope_label.text() == "(All Sets: 2)"
    panel.trace_next_button.click()
    assert panel.current_item().name == "E1"
    panel.trace_next_button.click()
    assert panel.current_item().name == "E2"
