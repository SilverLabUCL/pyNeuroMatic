"""Tests for the Main tool's tab: Main2's Plot and Table of the selection (#550)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6 import QtCore

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.main_tab import MainToolTab, selected_arrays


@pytest.fixture
def app(qtbot):
    """Demo: series Record, channels A, B x 3 epochs of 4 points; Set1 = [E1, E2]."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    for channel in ("A", "B"):
        for epoch in range(3):
            data = series.get_data(channel, f"E{epoch}")
            data.nparray = np.arange(4.0) + 10 * epoch + (100 if channel == "B" else 0)
            data.xscale.delta = 0.5
            data.xscale.units = "ms"
    series.epochs.sets.add("Set1", [series.epochs["E1"], series.epochs["E2"]])
    folder.data.new("Single", nparray=np.array([1.0, 2.0]))
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.folder, window.series = folder, series
    window.tool_rail.select_tool("Main")
    window.main = window.tool_tabs["Main"]
    # Plot…'s prompt: answered with the options as they are
    window.main._ask_plot_options = lambda selected: window.main.plot_view.options
    return window


def _select(app, **changes):
    app.selection_model.update({"folder": app.folder, "dataseries": app.series,
                                "source": "dataseries", **changes})


def test_the_tab_has_main1_and_main2(app):
    tab = app.main
    assert isinstance(tab, MainToolTab)
    assert [tab.mode_tabs.tabText(i) for i in range(tab.mode_tabs.count())] == [
        "Main1", "Main2"]
    assert tab.parameters_group.isHidden()  # no box around them (#539)
    assert tab.run_row.isHidden()  # nothing to run yet
    assert "Main2" in [app.view_tabs.tabText(i) for i in range(app.view_tabs.count())]


def test_selected_arrays_follow_the_strip(app):
    series = app.series
    _select(app, channel=series.channels["A"], epoch=series.epochs["E0"], epoch_set="All")
    selected = selected_arrays(app.selection_model.selection)
    assert [g.label for g in selected.groups] == ["A"]
    assert [d.name for d in selected.arrays] == ["RecordA0", "RecordA1", "RecordA2"]
    assert selected.title == "Record · A · All Epochs · 3 epochs"
    _select(app, all_channels=True, epoch_set="Set1")
    selected = selected_arrays(app.selection_model.selection)
    assert [g.label for g in selected.groups] == ["A", "B"]
    assert [d.name for d in selected.groups[1].arrays] == ["RecordB1", "RecordB2"]
    assert selected.title == "Record · A, B · Epoch Set: Set1 · 2 epochs"
    _select(app, channel=series.channels["B"], epoch=series.epochs["E2"], epoch_set=None)
    selected = selected_arrays(app.selection_model.selection)  # no set: the current epoch
    assert [d.name for d in selected.arrays] == ["RecordB2"]
    # Data: the array
    app.selection_model.update(data=app.folder.data["Single"], source="data")
    selected = selected_arrays(app.selection_model.selection)
    assert [d.name for d in selected.arrays] == ["Single"]


def test_plot_draws_each_channel_with_its_epochs_overlaid(app):
    tab = app.main
    _select(app, all_channels=True, epoch_set="All")
    _answer(app, rainbow=False)  # one line of every epoch
    tab.mode_tabs.setCurrentWidget(tab.main2_page)
    assert tab.selection_label.text() == "Record · A, B · All Epochs · 3 epochs"
    tab.plot_button.click()
    plot = tab.plot_view
    assert len(plot.plots) == 2  # A above B, x linked
    assert plot.plots[1].getViewBox().linkedView(0) is plot.plots[0].getViewBox()
    x, y = plot.plots[0].listDataItems()[0].getData()
    finite = ~np.isnan(y)
    assert np.allclose(y[finite], np.concatenate([np.arange(4.0) + 10 * e for e in range(3)]))
    assert np.allclose(x[finite][:4], [0.0, 0.5, 1.0, 1.5])
    assert plot.plots[1].getAxis("bottom").labelUnits == "ms"  # not "kms"
    assert plot.title_label.text() == "Record · A, B · All Epochs · 3 epochs"
    # Shown: the Plot panel, at Main2's tab
    assert app.context_panel.currentWidget() is app.plot_widget
    assert app.view_tabs.current_view() == ("tool", "Main")


def test_plot_shows_a_channels_filters_and_transforms(app):
    from pyneuromatic.core.nm_transform import NMTransformInvert
    tab = app.main
    channel = app.series.channels["A"]
    channel.transform = [NMTransformInvert()]
    _select(app, channel=channel, epoch_set="All")
    _answer(app, rainbow=False)
    tab.plot_button.click()
    _x, y = tab.plot_view.plots[0].listDataItems()[0].getData()
    assert np.nanmax(y) <= 0  # inverted, as the Plot shows it


def test_table_lists_every_array_read_only(app):
    tab = app.main
    _select(app, all_channels=True, epoch_set="Set1")
    tab.table_button.click()
    model = tab.table_view.model
    headers = [model.headerData(c, QtCore.Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers == ["x (ms)", "RecordA1", "RecordA2", "RecordB1", "RecordB2"]
    assert model.data(model.index(2, 3)) == "112"  # RecordB1: 100 + 10 + 2
    assert not model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert app.context_panel.currentWidget() is app.table_panel


def test_drawn_as_clicked_and_without_deleted_arrays(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab.plot_button.click()
    _select(app, channel=app.series.channels["B"])  # not drawn until clicked
    assert tab.plot_view.title_label.text().startswith("Record · A ·")
    app.folder.data.pop("Single")
    tab.refresh()
    assert [d.name for d in tab.plot_view.shown.arrays] == [
        "RecordA0", "RecordA1", "RecordA2"]


def test_no_selection_disables_plot_and_table(qtbot):
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    tab = window.tool_tabs["Main"]
    assert not tab.plot_button.isEnabled() and not tab.table_button.isEnabled()
    assert tab.selection_label.text() == "Choose a data array or a data series"


def _answer(app, **changes):
    """Plot…'s prompt, answered with these options changed."""
    from dataclasses import replace
    app.main._ask_plot_options = lambda selected: replace(app.main.plot_view.options, **changes)


def test_offsets_move_each_epoch_from_the_one_before(app):
    """X and Y offsets: epoch i drawn at x + i*dx, y + i*dy; the data don't
    change (#550)."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _answer(app, rainbow=False, x_offset=10.0, y_offset=-1.0)
    tab.plot_button.click()
    x, y = tab.plot_view.plots[0].listDataItems()[0].getData()
    finite = ~np.isnan(y)
    epoch2 = slice(8, 12)  # the third epoch's 4 points
    assert np.allclose(x[finite][epoch2], [20.0, 20.5, 21.0, 21.5])
    assert np.allclose(y[finite][epoch2], np.arange(4.0) + 20 - 2)
    assert np.allclose(app.series.get_data("A", "E2").nparray, np.arange(4.0) + 20)
    assert tab.plot_view.options.x_offset == 10.0  # kept for next time


def test_reverse_order_starts_from_the_last_epoch(app):
    """Reverse: the last epoch has no offset and the first color; the first
    epoch is offset most, and drawn last (on top)."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _answer(app, reverse=True, y_offset=-1.0)
    tab.plot_button.click()
    curves = tab.plot_view.plots[0].listDataItems()
    first_drawn = curves[0].getData()[1]
    last_drawn = curves[-1].getData()[1]
    assert np.allclose(first_drawn[:4], np.arange(4.0) + 20)  # E2, not offset
    assert np.allclose(last_drawn[:4], np.arange(4.0) - 2)  # E0, offset twice
    hues = [c.opts["pen"].color().hsvHueF() for c in curves]
    assert hues == sorted(hues)  # E2 red ... E0 violet


def test_cancelled_prompt_plots_nothing(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab._ask_plot_options = lambda selected: None
    tab.plot_button.click()
    assert tab.plot_view.shown is None


def test_color_and_rainbow(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    assert tab.plot_view.options.rainbow  # by default (#550)
    _answer(app, rainbow=False, color="#1d4ed8")
    tab.plot_button.click()
    curves = tab.plot_view.plots[0].listDataItems()
    assert len(curves) == 1 and curves[0].opts["pen"].color().name() == "#1d4ed8"
    _answer(app, rainbow=True)  # an epoch each, red to violet
    tab.plot_button.click()
    curves = tab.plot_view.plots[0].listDataItems()
    hues = [c.opts["pen"].color().hsvHueF() for c in curves]
    assert len(curves) == 3 and hues == sorted(hues)


def test_plot_prompt(qtbot):
    from pyneuromatic.gui.main_tab import PlotOptions, PlotOptionsDialog
    dialog = PlotOptionsDialog(PlotOptions(color="#1d4ed8", rainbow=False, y_offset=2.0),
                               "Record · A")
    qtbot.addWidget(dialog)
    assert dialog.options() == PlotOptions(color="#1d4ed8", rainbow=False,
                                           y_offset=2.0)  # as it was
    chosen = []
    dialog._ask_color = lambda current: chosen.append(current.name())
    dialog.color_button.click()
    assert chosen == ["#1d4ed8"]  # the color picker starts at the color
    dialog.set_color("#dc2626")
    dialog.color_combo.setCurrentIndex(1)  # Rainbow: no one color to choose
    assert not dialog.color_button.isEnabled()
    dialog.x_offset_edit.setText("5")
    dialog.reverse_checkbox.setChecked(True)
    assert dialog.options() == PlotOptions(color="#dc2626", rainbow=True,
                                           x_offset=5.0, y_offset=2.0, reverse=True)
    dialog.y_offset_edit.setText("wide")  # not a number: the prompt stays
    dialog.accept()
    assert dialog.error_label.text() == "Y offset: 'wide' is not a number"
    assert dialog.result() != dialog.DialogCode.Accepted


def test_rainbow_draws_many_epochs_in_few_lines(app):
    from pyneuromatic.gui.main_tab import (
        RAINBOW_COLORS, ChannelArrays, PlotOptions, Selected, SelectionPlot)
    arrays = [app.folder.data.new(f"Many{i}", nparray=np.arange(3.0)) for i in range(200)]
    plot = SelectionPlot()
    plot.options = PlotOptions(rainbow=True)
    lines = plot._lines(ChannelArrays("Data", arrays))
    assert len(lines) == RAINBOW_COLORS
    assert sum(np.count_nonzero(~np.isnan(y)) for _x, y, _c in lines) == 600
    plot.show_selected(Selected("Many", [ChannelArrays("Data", arrays)]))
    plot.deleteLater()
