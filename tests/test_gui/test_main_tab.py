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
    assert not tab.run_row.isHidden()  # Main1's functions run (#553)
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


# --- Functions and Main2 (#553) ---------------------------------------------------

def _run(app, name="Average", **options):
    """Run function *name* from Main1, its prompt answered with *options*."""
    tab = app.main
    tab.function_combo.setCurrentText(name)
    asked = []
    tab._ask_function = lambda function: asked.append(function) or dict(options)
    tab.run_button.click()
    return asked


def test_average_of_all_channels_in_one_subfolder(app):
    """One run, every channel: MN_Avg_RecordA and MN_Avg_RecordB in a Main
    subfolder, made a data series; shown in Main2 and plotted."""
    _select(app, all_channels=True, epoch_set="All")
    asked = _run(app, "Average", compute_sem=True)
    assert asked == ["Average"]
    folder = app.folder.toolfolders["Main_Record_AB_All_0"]
    assert list(folder.data.keys()) == ["MN_Avg_RecordA", "MN_SEM_RecordA",
                                        "MN_Avg_RecordB", "MN_SEM_RecordB"]
    assert np.allclose(folder.data["MN_Avg_RecordB"].nparray, np.arange(4.0) + 110)
    series = folder.dataseries["MN_Avg_Record"]
    assert list(series.channels.keys()) == ["A", "B"]
    assert "MN_SEM_Record" in folder.dataseries
    assert folder.notes[0]["note"].startswith("NMMain: Average(")
    tab = app.main
    assert tab.mode_tabs.currentWidget() is tab.main2
    assert tab.main2.toolfolder() is folder
    assert tab.main2.selected_array() == "MN_Avg_RecordA"
    assert tab.plot_view.title_label.text() == "Main_Record_AB_All_0: MN_Avg_Record"
    assert len(tab.plot_view.plots) == 2  # A above B
    assert app.view_tabs.current_view() == ("tool", "Main")


def test_one_channel_and_the_command(app):
    manager = app.selection_model.manager
    manager.command_history.enabled = True
    _select(app, channel=app.series.channels["B"], epoch_set="Set1")
    _run(app, "Max")
    folder = app.folder.toolfolders["Main_Record_B_S1_0"]
    assert np.allclose(folder.data["MN_Max_RecordB"].nparray, np.arange(4.0) + 120)
    commands = [e["command"] for e in manager.command_history.buffer]
    assert commands[-2] == "nm.main.op = NMMainOpMax(ignore_nans=True)"
    assert commands[-1] == (
        "nm.main.run_selection(nm.folders['Demo'].dataseries['Record'], "
        "channel='B', epochs='Set1')")


def test_cancelled_function_runs_nothing(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab._ask_function = lambda name: None
    tab.run_button.click()
    assert len(app.folder.toolfolders) == 0


def test_function_options_are_kept_for_next_time(app, qtbot):
    from pyneuromatic.gui.main_tab import FunctionDialog
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average", compute_stdv=True)
    assert tab._function_options["Average"]["compute_stdv"]
    dialog = FunctionDialog("Average", tab._function_options["Average"], "Record · A")
    qtbot.addWidget(dialog)
    assert dialog.checkboxes["compute_stdv"].isChecked()
    assert list(dialog.checkboxes) == ["compute_stdv", "compute_var", "compute_sem"]
    concatenate = FunctionDialog("Concatenate", {}, "Record · A")
    qtbot.addWidget(concatenate)
    assert concatenate.checkboxes == {}  # nothing to choose


def test_overwrite_in_settings_reuses_the_subfolder(app):
    tab = app.main
    assert [o[:3] for o in tab.config_options()] == [
        ("ignore_nans", "Ignore NaNs", True), ("overwrite", "Overwrite results", False)]
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Sum")
    _run(app, "Sum")
    assert list(app.folder.toolfolders.keys()) == ["Main_Record_A_All_0",
                                                   "Main_Record_A_All_1"]
    tab.set_config_option("overwrite", True)
    _run(app, "Sum")
    assert len(app.folder.toolfolders) == 2  # _0 cleared and reused
    assert app.main._tool.toolfolder.name == "Main_Record_A_All_0"


def test_main2_table_lists_the_outputs(app):
    tab = app.main
    _select(app, all_channels=True, epoch_set="All")
    _run(app, "Average")
    tab.main2.select_array("MN_Avg_RecordB")
    tab.main2.table_action.trigger()
    model = tab.table_view.model
    headers = [model.headerData(c, QtCore.Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers[1:] == ["MN_Avg_RecordA", "MN_Avg_RecordB"]
    assert tab.table_view.selected_array().name == "MN_Avg_RecordB"
    assert app.context_panel.currentWidget() is app.table_panel


def test_ignore_nans_is_in_settings_not_the_prompt(app):
    """Ignore NaNs: Settings › Main, as for Stats and Spike; functions without
    other options run without a prompt."""
    tab = app.main
    app.series.get_data("A", "E1").nparray[0] = np.nan
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    asked = _run(app, "Sum")
    assert asked == []  # no prompt: nothing to ask
    folder = app.main._tool.toolfolder
    assert folder.data["MN_Sum_RecordA"].nparray[0] == 0 + 20  # the NaN left out
    tab.set_config_option("ignore_nans", False)
    _run(app, "Sum")
    assert np.isnan(app.main._tool.toolfolder.data["MN_Sum_RecordA"].nparray[0])
    assert app.main._tool.config.ignore_nans is False


def test_x_range_and_its_lines(app):
    """X begin / X end: what functions use; the Plot's range lines, dragged
    to set them (#553)."""
    tab = app.main
    assert tab.plot_ranges()[0].xbgn == -np.inf
    tab.xbgn_edit.setText("0.5")
    tab.xend_edit.setText("1")
    tab.xend_edit.editingFinished.emit()
    assert (tab._tool.xbgn, tab._tool.xend) == (0.5, 1.0)
    assert (tab.plot_ranges()[0].xbgn, tab.plot_ranges()[0].xend) == (0.5, 1.0)
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average")
    output = app.main._tool.toolfolder.data["MN_Avg_RecordA"]
    assert np.allclose(output.nparray, [11.0, 12.0])  # points at x 0.5 and 1
    assert output.xscale.start == 0.5
    tab.set_plot_range("range", 0.0, 1.5, final=True)  # dragged
    assert (tab._tool.xbgn, tab._tool.xend) == (0.0, 1.5)
    assert tab.xbgn_edit.text() == "0"
    tab.xbgn_edit.setText("2")  # not a range: kept, with a warning
    tab.xbgn_edit.editingFinished.emit()
    assert tab._tool.xbgn == 0.0


def test_average_plotted_over_its_data(app):
    """Main2: an average over the arrays it was made from, in the overlay
    color; Plot with its data off: the average alone."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average")
    group = tab.plot_view.shown.groups[0]
    assert [d.name for d in group.background] == ["RecordA0", "RecordA1", "RecordA2"]
    curves = tab.plot_view.plots[0].listDataItems()
    assert len(curves) == 2  # the data, then the average over it
    assert curves[0].opts["pen"].color().name() == tab.plot_view.background_color
    assert tab.main2.background_action.isEnabled()
    tab.main2.background_action.setChecked(False)
    assert len(tab.plot_view.plots[0].listDataItems()) == 1
    tab.main2.background_action.setChecked(True)
    _run(app, "Sum")  # not an average, min or max: no data behind
    assert not tab.main2.background_action.isEnabled()
    assert tab.plot_view.shown.groups[0].background == []



def test_average_with_a_stdv_or_sem_band(app):
    """Plot ± Stdv / ± SEM: a shaded band about the average, from its saved
    Stdv (or Var's root) or SEM; one at a time; greyed out without them."""
    import pyqtgraph as pg
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average", compute_var=True, compute_sem=True)  # no Stdv: from Var
    main2 = tab.main2
    main2.background_action.setChecked(False)
    stdv, sem = main2.error_actions["stdv"], main2.error_actions["sem"]
    assert stdv.isEnabled() and sem.isEnabled()

    def bands():
        return [i for i in tab.plot_view.plots[0].items if isinstance(i, pg.FillBetweenItem)]

    assert bands() == []  # off by default
    stdv.setChecked(True)
    assert len(bands()) == 1
    group = tab.plot_view.shown.groups[0]
    var = main2.toolfolder().data["MN_Var_RecordA"].nparray
    assert np.allclose(group.errors[0], np.sqrt(var))
    assert tab.plot_view.title_label.text().endswith("MN_Avg_Record ± Stdv")
    sem.setChecked(True)  # one band at a time
    assert not stdv.isChecked()
    assert np.allclose(tab.plot_view.shown.groups[0].errors[0],
                       main2.toolfolder().data["MN_SEM_RecordA"].nparray)
    main2.select_array("MN_SEM_RecordA")  # not an average: no band
    assert not sem.isEnabled() and not stdv.isEnabled()
    _run(app, "Average")  # nothing saved to make a band from
    assert not stdv.isEnabled() and not sem.isEnabled()


def test_main_range_lines_show_from_the_start(qtbot):
    """Main is the tool chosen at start: its range lines are on the Plot
    without choosing another tool and back."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    folder.new_dataseries("Record", n_channels=1, n_epochs=2, n_points=4)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.refresh()
    assert window.tool_rail.current_tool_name == "Main"
    assert "range" in window.plot_widget.range_overlay.regions



def test_ask_then_omit_leaves_epochs_out_of_the_average(app):
    """Progress › Ask after each: Omit leaves the epoch shown out of the
    function's outputs; the command says so, to repeat the run (#553)."""
    manager = app.selection_model.manager
    manager.command_history.enabled = True
    manager.settings.run_display = "ask"
    app._apply_settings()
    tab = app.main
    answers = ["continue", "omit", "all"]  # E1 omitted
    shown = []

    def prompt(_text):
        shown.append(tab.prompt_buttons["omit"].isVisibleTo(tab))
        answer = answers.pop(0) if answers else "cancel"
        QtCore.QTimer.singleShot(0, lambda: tab.answer_prompt(answer))

    tab.prompt_shown.connect(prompt)
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average")
    assert shown[0]  # Omit offered, for a function
    folder = tab._tool.toolfolder
    assert np.allclose(folder.data["MN_Avg_RecordA"].nparray, np.arange(4.0) + 10)  # E0, E2
    assert [d.name for d in folder.sources["MN_Avg_RecordA"]] == ["RecordA0", "RecordA2"]
    assert tab.status_label.text() == "Complete (omitted E1)"
    runs = [e["command"] for e in manager.command_history.buffer if "run_selection" in e["command"]]
    assert runs[-1].endswith("epochs='all', omit=['E1'])")


def test_main2_keeps_the_status_line(app):
    """As on Stats2's and Spike2's tabs: Run and Progress go, the status stays."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average")
    assert tab.mode_tabs.currentWidget() is tab.main2
    assert tab.run_row.isHidden() and tab.display_row.isHidden()
    assert not tab.status_label.isHidden()
    assert tab.status_label.text() == "Complete"


def test_overwrite_shows_each_runs_own_output(app):
    """Overwrite: runs on one selection share ..._0; after each, Main2 chooses
    and plots that run's first output, not the one chosen before."""
    tab = app.main
    tab.set_config_option("overwrite", True)
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    for name, output in (("Average", "MN_Avg_RecordA"), ("Max", "MN_Max_RecordA"),
                         ("Sum", "MN_Sum_RecordA"), ("Average", "MN_Avg_RecordA")):
        tab.mode_tabs.setCurrentWidget(tab.main1_page)
        _run(app, name)
        assert tab.main2.toolfolder().name == "Main_Record_A_All_0"
        assert tab.main2.selected_array() == output
        assert tab.plot_view.title_label.text() == f"Main_Record_A_All_0: {output[:-1]}"


def test_overwrite_shows_this_runs_outputs_not_an_earlier_stdv(app):
    """With Overwrite, an earlier Average's Stdv isn't a later run's output:
    Max chooses MN_Max_; an Average without Stdv makes MN_Avg_ alone."""
    tab = app.main
    tab.set_config_option("overwrite", True)
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average", compute_stdv=True)
    assert tab._tool.outputs == ["MN_Avg_RecordA", "MN_Stdv_RecordA"]
    tab.mode_tabs.setCurrentWidget(tab.main1_page)
    _run(app, "Max")
    assert tab._tool.outputs == ["MN_Max_RecordA"]
    assert tab.main2.selected_array() == "MN_Max_RecordA"
    tab.mode_tabs.setCurrentWidget(tab.main1_page)
    _run(app, "Average")
    assert tab._tool.outputs == ["MN_Avg_RecordA"]  # the Stdv is the earlier run's
    assert tab.main2.selected_array() == "MN_Avg_RecordA"

