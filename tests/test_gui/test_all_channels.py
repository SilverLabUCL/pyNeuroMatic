"""Tests for "All" channels: Stats runs once per channel."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.selection_model import ALL_CHANNELS, ALL_SETS

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Series Record: channels A (mean = epoch) and B (mean = 10 x epoch),
    4 epochs; Set0 = E1, E2; Set1 = E3."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(4):
        folder.data.new(f"RecordA{epoch}", nparray=np.array([float(epoch)] * 2),
                        xscale={"start": 0.0, "delta": 1.0}, yscale={"units": "pA"})
        folder.data.new(f"RecordB{epoch}", nparray=np.array([10.0 * epoch] * 2),
                        xscale={"start": 0.0, "delta": 1.0}, yscale={"units": "mV"})
    series = folder.sync_dataseries("Record", select=True)
    series.epochs.sets.add("Set0", ["E1", "E2"])
    series.epochs.sets.add("Set1", ["E3"])
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.stats_tab.measurement_combo.setCurrentText("mean")
    window.folder, window.series = folder, series
    return window


def _means(folder):
    return list(folder.data["ST_w0_mean_y"].nparray)


def test_all_in_the_channel_dropdown(app):
    combo = app.selection_strip.combo("channel")
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "All", "A", "B"]
    combo.setCurrentIndex(combo.findData(ALL_CHANNELS))
    selection = app.selection_model.selection
    assert selection["all_channels"] is True
    assert selection["channel"].name == "A"  # the current channel stays (Plot, Table)
    assert combo.currentText() == "All"
    assert "All Channels" in app.windowTitle()
    # Choosing one channel turns it off
    combo.setCurrentIndex(combo.findData("B"))
    assert app.selection_model.selection["all_channels"] is None
    assert app.selection_model.selection["channel"].name == "B"


def test_no_all_for_one_channel(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Single", n_epochs=2, n_points=2)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(dataseries=series)
    combo = window.selection_strip.combo("channel")
    assert "All" not in [combo.itemText(i) for i in range(combo.count())]


def test_run_once_per_channel(app):
    tab = app.stats_tab
    app.selection_model.update(epoch_set="Set0", all_channels=True)
    assert tab.run_info_label.text() == "n=2 runs"
    assert "A: 2, B: 2" in tab.run_info_label.toolTip()
    tab.run_button.click()
    folders = app.folder.toolfolders
    assert _means(folders["Stats_Record_A_S0_0"]) == [1.0, 2.0]
    assert _means(folders["Stats_Record_B_S0_0"]) == [10.0, 20.0]
    # Each channel's results keep its units
    assert folders["Stats_Record_B_S0_0"].data["ST_w0_mean_y"].yscale.units == "mV"


def test_channels_with_all_sets(app):
    tab = app.stats_tab
    app.selection_model.update(epoch_set=ALL_SETS, all_channels=True)
    assert tab.run_info_label.text() == "n=4 runs"
    assert "A Set0: 2, A Set1: 1, B Set0: 2, B Set1: 1" in tab.run_info_label.toolTip()
    tab.run_button.click()
    folders = app.folder.toolfolders
    for name in ("A_S0", "A_S1", "B_S0", "B_S1"):
        assert f"Stats_Record_{name}_0" in folders
    assert _means(folders["Stats_Record_B_S1_0"]) == [30.0]
    assert tab.stats2.folder_combo.currentText() == "Stats_Record_B_S1_0"  # the last


def test_another_series_starts_with_one_channel(app):
    app.selection_model.update(all_channels=True)
    other = app.folder.new_dataseries("Stim", n_channels=2, n_epochs=2, n_points=2)
    app.selection_model.update(dataseries=other)
    assert app.selection_model.selection["all_channels"] is None
