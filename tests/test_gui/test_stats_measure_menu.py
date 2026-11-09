"""Tests for Stats1's measurement menu: each measurement once, its sign a
choice (#478)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.stats_tab import split_measurement

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    data = folder.data.new("RecordA0", nparray=np.arange(20.0),
                           xscale={"start": 0.0, "delta": 1.0})
    manager.select_value_set(data)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.stats, window.manager = window.tool_tabs["Stats"], manager
    return window


def _items(tab):
    combo = tab.measurement_combo
    return [combo.itemData(i) for i in range(combo.count()) if combo.itemText(i)]


def _commands(app):
    return [e["command"] for e in app.manager.command_history.buffer]


def test_split_measurement():
    assert split_measurement("risetime-") == ("risetime", "-")
    assert split_measurement("fwhm+") == ("fwhm", "+")
    assert split_measurement("level") == ("level", "")
    assert split_measurement("level-") == ("level", "-")
    assert split_measurement("mean+std") == ("mean+std", "")  # not a sign
    assert split_measurement(None) == (None, "")


def test_each_measurement_once_and_no_counts(app):
    items = _items(app.stats)
    assert items[0] is None  # Choose measurement
    names = items[1:]
    assert len(names) == len(set(names))
    for base in ("risetime", "risetimeslope", "falltime", "falltimeslope",
                 "decaytime", "fwhm", "level"):
        assert base in names
    assert not any(name.endswith(("+", "-")) and not name.startswith("mean")
                   for name in names)  # no signed names (mean+std etc. aren't signs)
    assert not set(names) & {"count", "count_nans", "count_infs"}


@pytest.mark.parametrize("base, sign, core", [
    ("fwhm", "+", "fwhm+"), ("fwhm", "-", "fwhm-"),
    ("risetime", "-", "risetime-"), ("decaytime", "-", "decaytime-"),
])
def test_peak_sign_sets_the_core_name(app, base, sign, core):
    tab = app.stats
    tab.measurement_combo.setCurrentText(base)
    assert tab.peak_combo.isVisibleTo(tab) and not tab.crossing_combo.isVisibleTo(tab)
    count = len(_commands(app))
    tab.peak_combo.setCurrentIndex(tab.peak_combo.findData(sign))
    window = app.manager.stats.windows["w0"]
    assert window.func["name"] == core and tab.measurement() == core
    assert len(_commands(app)) - count == (0 if sign == "+" else 1)  # one command


def test_crossing_direction_sets_the_core_name(app):
    tab = app.stats
    tab.measurement_combo.setCurrentText("level")
    assert tab.crossing_combo.isVisibleTo(tab) and not tab.peak_combo.isVisibleTo(tab)
    window = app.manager.stats.windows["w0"]
    for shown, core in (("rising", "level+"), ("falling", "level-"), ("any", "level")):
        tab.crossing_combo.setCurrentText(shown)
        assert window.func["name"] == core
    assert _commands(app)[-1].startswith("nm.stats.windows['w0'].func = {'name': 'level'")


def test_no_sign_for_other_measurements(app):
    tab = app.stats
    tab.measurement_combo.setCurrentText("mean")
    assert not tab.peak_combo.isVisibleTo(tab) and not tab.crossing_combo.isVisibleTo(tab)
    assert app.manager.stats.windows["w0"].func["name"] == "mean"


def test_a_window_set_by_a_script_shows_its_sign(app):
    tab = app.stats
    app.manager.stats.windows["w0"].func = "fwhm-"
    tab._load_selected_window()
    assert tab.measurement_combo.currentData() == "fwhm"
    assert tab.peak_combo.currentData() == "-"
    app.manager.stats.windows["w0"].func = {"name": "level-", "ylevel": 3.0}
    tab._load_selected_window()
    assert (tab.measurement_combo.currentData(), tab.crossing_combo.currentText()) == (
        "level", "falling")


def test_a_count_from_a_script_still_shows(app):
    tab = app.stats
    app.manager.stats.windows["w0"].func = "count_nans"
    tab._load_selected_window()
    assert tab.measurement_combo.currentData() == "count_nans"  # added to the menu
    assert tab.measurement() == "count_nans"


def test_each_window_keeps_its_sign(app):
    tab = app.stats
    tab.set_measurement("risetime-")
    tab._add_window()  # w1
    tab.set_measurement("risetime+")
    tab.window_combo.setCurrentIndex(tab.window_combo.findData("w0"))
    assert tab.peak_combo.currentData() == "-"
    tab.window_combo.setCurrentIndex(tab.window_combo.findData("w1"))
    assert tab.peak_combo.currentData() == "+"
