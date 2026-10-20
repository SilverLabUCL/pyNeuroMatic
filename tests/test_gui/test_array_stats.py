"""Tests for the info panel's Details / Stats toggle for data arrays."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Folder Demo with data W = [1, 2, 3, 4, NaN, 10] (x 0.5 ms steps, mV)."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    folder.data.new("W", nparray=np.array([1.0, 2.0, 3.0, 4.0, np.nan, 10.0]),
                    xscale={"start": 0.0, "delta": 0.5, "units": "ms"},
                    yscale={"units": "mV"})
    folder.data.new("V", nparray=np.array([5.0, 7.0]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.folder = folder
    return window


def _show(app, name):
    browser = app.browser_widget
    browser.show_data_of(app.folder, select=name)
    return browser.info_panel


def test_toggle_only_for_data_arrays(app):
    panel = _show(app, "W")
    assert panel.details_button.isVisibleTo(panel) and panel.stats_button.isVisibleTo(panel)
    assert panel.details_button.isChecked() and not panel.showing_stats
    app.browser_widget.show_object(app.folder)
    assert not panel.stats_button.isVisibleTo(panel)


def test_stats(app):
    panel = _show(app, "W")
    panel.stats_button.click()
    assert panel.showing_stats
    assert panel.row_labels()[0] == "Created"
    assert panel.row_text("Points") == "6  (NaNs 1, INFs 0)"
    # Of 1, 2, 3, 4, 10 (the NaN is ignored)
    assert panel.row_text("Mean") == "4 mV"
    assert panel.row_text("Median") == "3 mV"
    assert panel.row_text("SD") == "3.53553 mV"
    assert panel.row_text("SEM") == "1.58114 mV"
    assert panel.row_text("RMS") == "5.09902 mV"
    assert panel.row_text("Min") == "1 mV at 0 ms"
    assert panel.row_text("Max") == "10 mV at 2.5 ms"
    # The details' rows and Edit Scales... are gone; notes stay
    assert panel.row_text("X Delta") is None
    assert not panel.edit_scales_button.isVisibleTo(panel)
    assert panel.notes_table.isVisibleTo(panel)


def test_back_to_details(app):
    panel = _show(app, "W")
    panel.stats_button.click()
    panel.details_button.click()
    assert panel.row_text("X Delta") == "0.5"
    assert panel.row_text("Mean") is None
    assert panel.edit_scales_button.isVisibleTo(panel)


def test_the_choice_is_kept_between_arrays(app):
    panel = _show(app, "W")
    panel.stats_button.click()
    panel = _show(app, "V")
    assert panel.showing_stats
    assert panel.row_text("Mean") == "6"  # no units
    assert panel.row_text("Max") == "7 at 1"  # x scale start 0, delta 1


def test_one_point_has_no_sd(app):
    app.folder.data.new("One", nparray=np.array([2.0]))
    app.browser_widget.refresh()
    panel = _show(app, "One")
    panel.stats_button.click()
    assert panel.row_text("SD") == "—" and panel.row_text("SEM") == "—"
    assert panel.row_text("Mean") == "2"


def test_all_nan(app):
    app.folder.data.new("Gaps", nparray=np.array([np.nan, np.nan]))
    app.browser_widget.refresh()
    panel = _show(app, "Gaps")
    panel.stats_button.click()
    assert panel.row_text("Points") == "2  (NaNs 2, INFs 0)"
    assert panel.row_text("Stats") == "Every point is NaN"


@pytest.mark.parametrize("array, kind", [
    (np.array(["a", "b"]), "text"),
    (np.array([True, False]), "bool"),
    (np.array([1 + 2j]), "complex128"),
])
def test_no_statistics_for_other_types(app, array, kind):
    app.folder.data.new("Other", nparray=array)
    app.browser_widget.refresh()
    panel = _show(app, "Other")
    panel.stats_button.click()
    assert panel.row_text("Stats") == f"No statistics for {kind} data"
    assert panel.row_text("Mean") is None


def test_integers_and_x_array(app):
    data = app.folder.data.new("Counts", nparray=np.array([3, 9, 1], dtype=np.int16))
    data.xarray = np.array([0.0, 0.25, 1.5])
    app.browser_widget.refresh()
    panel = _show(app, "Counts")
    panel.stats_button.click()
    assert panel.row_text("Mean") == "4.33333"
    assert panel.row_text("Min") == "1 at 1.5"  # from the x array
    assert panel.row_text("Max") == "9 at 0.25"


def test_two_dimensional_has_no_locations(app):
    app.folder.data.new("Matrix", nparray=np.array([[1.0, 5.0], [3.0, 2.0]]))
    app.browser_widget.refresh()
    panel = _show(app, "Matrix")
    panel.stats_button.click()
    assert panel.row_text("Max") == "5"
    assert panel.row_text("Points") == "4  (NaNs 0, INFs 0)"
