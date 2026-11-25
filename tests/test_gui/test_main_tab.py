"""Tests for the Main tool's tab: Main2's Plot and Table of the selection (#550)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6 import QtCore, QtGui

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
    return window


def _entry(app, name):
    """Choose *name* in Main1's list (e.g. "Plot", "Table"), then Run."""
    app.main.function_combo.setCurrentText(name)
    app.main.run_button.click()


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
    assert tab.run_info_label.text() == "n=6"  # beside Run: 2 channels x 3 epochs, one run
    _entry(app, "Plot")
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
    _entry(app, "Plot")
    _x, y = tab.plot_view.plots[0].listDataItems()[0].getData()
    assert np.nanmax(y) <= 0  # inverted, as the Plot shows it


def test_table_lists_every_array_read_only(app):
    tab = app.main
    _select(app, all_channels=True, epoch_set="Set1")
    _entry(app, "Table")
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
    _entry(app, "Plot")
    _select(app, channel=app.series.channels["B"])  # not drawn until clicked
    assert tab.plot_view.title_label.text().startswith("Record · A ·")
    app.folder.data.pop("Single")
    tab.refresh()
    assert [d.name for d in tab.plot_view.shown.arrays] == [
        "RecordA0", "RecordA1", "RecordA2"]


def test_no_selection_plots_nothing_and_says_why(qtbot):
    window = NMAppWindow(NMManager(quiet=True))
    qtbot.addWidget(window)
    tab = window.tool_tabs["Main"]
    window.main = tab
    _entry(window, "Plot")
    assert tab.plot_view.shown is None
    assert tab.status_label.text().endswith("Choose a data array or a data series")


def _answer(app, rainbow=None, **values):
    """Plot's form set to *values* (rainbow: its Color)."""
    if rainbow is not None:
        values["colors"] = "rainbow" if rainbow else "one"
    app.main.forms["Plot"].set_values(values)


def test_offsets_move_each_epoch_from_the_one_before(app):
    """X and Y offsets: epoch i drawn at x + i*dx, y + i*dy; the data don't
    change (#550)."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _answer(app, rainbow=False, x_offset=10.0, y_offset=-1.0)
    _entry(app, "Plot")
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
    _entry(app, "Plot")
    curves = tab.plot_view.plots[0].listDataItems()
    first_drawn = curves[0].getData()[1]
    last_drawn = curves[-1].getData()[1]
    assert np.allclose(first_drawn[:4], np.arange(4.0) + 20)  # E2, not offset
    assert np.allclose(last_drawn[:4], np.arange(4.0) - 2)  # E0, offset twice
    hues = [c.opts["pen"].color().hsvHueF() for c in curves]
    assert hues == sorted(hues)  # E2 red ... E0 violet


def test_a_bad_plot_option_plots_nothing(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab.forms["Plot"].control("y_offset").setText("wide")
    _entry(app, "Plot")
    assert tab.plot_view.shown is None
    assert tab.status_label.text().endswith("Y offset: 'wide' is not a number")


def test_color_and_rainbow(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    assert tab.forms["Plot"].values()["colors"] == "rainbow"  # by default (#550)
    _answer(app, rainbow=False, color="#1d4ed8")
    _entry(app, "Plot")
    curves = tab.plot_view.plots[0].listDataItems()
    assert len(curves) == 1 and curves[0].opts["pen"].color().name() == "#1d4ed8"
    _answer(app, rainbow=True)  # an epoch each, red to violet
    _entry(app, "Plot")
    curves = tab.plot_view.plots[0].listDataItems()
    hues = [c.opts["pen"].color().hsvHueF() for c in curves]
    assert len(curves) == 3 and hues == sorted(hues)
    form = tab.forms["Plot"]
    assert form._rows["reverse_colors"][0].isEnabled()  # with Rainbow
    form.set_values({"reverse_colors": True})  # violet to red
    _entry(app, "Plot")
    curves = tab.plot_view.plots[0].listDataItems()
    assert [c.opts["pen"].color().hsvHueF() for c in curves] == sorted(hues, reverse=True)
    form.set_values({"colors": "one"})
    assert not form._rows["reverse_colors"][0].isEnabled()  # one color: none


def test_list_tips_dont_repeat_the_name(app):
    combo = app.main.function_combo
    tips = {combo.itemText(i): combo.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            for i in range(combo.count()) if combo.itemText(i)}
    assert tips["Average"] == (
        "Compute the average along each data point of the selected arrays")
    assert tips["Delete Points"].startswith("Delete points from the selected arrays")


def test_one_list_with_plot_and_its_form(app):
    """Main1's list, as Stats' measures (#555): Plot, Table, then the
    functions, divided; Plot's options are its form, Rainbow by default."""
    tab = app.main
    combo = tab.function_combo
    items = [combo.itemText(i) or "---" for i in range(combo.count())]
    assert items == ["Choose…", "---", "Plot", "Table", "---", "Average", "Sum",
                     "SumSqr", "Min", "Max", "Concatenate", "---", "Insert Points",
                     "Delete Points", "Delete NaNs", "Redimension",
                     "Reverse", "Rotate", "---", "Rescale X", "Set X Start", "Resample",
                     "Interpolate", "Align", "---", "Arithmetic", "Replace Values",
                     "Baseline", "Normalize", "dF/F", "Rescale Y", "Smooth", "Filter",
                     "Differentiate", "Integrate"]
    assert tab.entry() is None  # "Choose…", as Stats1's
    assert tab.form() is None and tab.plot_ranges() == []
    assert not tab.reset_button.isEnabled()
    tab.run_button.click()  # nothing chosen: nothing done, and why
    assert tab.status_label.text().endswith("Choose function: Plot, Average, Baseline, Filter…")
    combo.setCurrentText("Plot")
    form = tab.forms["Plot"]
    assert [p.name for p in form.params] == ["colors", "color", "reverse_colors", "x_offset", "y_offset",
                                             "reverse"]
    assert form.values()["colors"] == "rainbow"
    swatch = form.control("color")
    assert not swatch.isEnabled()  # Rainbow: no one color to choose
    form.set_values({"colors": "one"})
    assert swatch.isEnabled()
    form._ask_color = lambda current: QtGui.QColor("#dc2626")
    swatch.click()
    assert form.values()["color"] == "#dc2626"
    assert tab.display_row.isHidden()  # Plot draws: no run, no Progress
    tab.function_combo.setCurrentText("Average")
    assert tab.plot_ranges()[0].key == "x"  # a function's x range


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

def _run(app, name="Average", **values):
    """Run function *name* from Main1, its form set to *values* (by
    parameter name, e.g. compute_sem=True, x=(0.5, 1))."""
    tab = app.main
    tab.function_combo.setCurrentText(name)
    tab.forms[name].set_values(values)
    tab.run_button.click()


def test_average_of_all_channels_in_one_subfolder(app):
    """One run, every channel: MN_Avg_RecordA and MN_Avg_RecordB in a Main
    subfolder, made a data series; shown in Main2 and plotted."""
    _select(app, all_channels=True, epoch_set="All")
    _run(app, "Average", compute_sem=True)
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


def test_a_value_not_valid_runs_nothing(app):
    """The form's values are checked first: a bad one is said, with no run."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    begin, end = tab.forms["Average"].control("x")
    begin.setText("2")
    end.setText("1")
    tab.function_combo.setCurrentText("Average")
    tab.run_button.click()
    assert len(app.folder.toolfolders) == 0
    assert "begin must be less than end" in tab.status_label.text()


def test_each_function_shows_and_keeps_its_own_form(app):
    """The chosen function's form (#555), kept as set when another is chosen."""
    tab = app.main
    tab.function_combo.setCurrentText("Average")
    form = tab.form()
    assert tab.form_stack.currentWidget() is form
    assert [p.name for p in form.params] == ["x", "compute_stdv", "compute_var",
                                              "compute_sem"]
    form.control("compute_stdv").setChecked(True)
    assert tab.function_info.text().startswith(
        "Compute the average along each data point of the selected arrays")
    tab.function_combo.setCurrentText("Max")
    assert [p.name for p in tab.form().params] == ["x"]
    tab.function_combo.setCurrentText("Average")
    assert tab.form().values()["compute_stdv"] is True


def test_overwrite_in_settings_reuses_the_subfolder(app):
    tab = app.main
    assert [o[:3] for o in tab.config_options()] == [
        ("ignore_nans", "Ignore NaNs", True), ("overwrite", "Overwrite results", False),
        ("ask_before_changing", "Ask before changing arrays", True),
        ("prefix_outputs", "Prefix outputs with MN_", True)]
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


def test_ignore_nans_is_in_settings_not_the_form(app):
    """Ignore NaNs: Settings › Main, as for Stats and Spike."""
    tab = app.main
    app.series.get_data("A", "E1").nparray[0] = np.nan
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Sum")
    folder = app.main._tool.toolfolder
    assert folder.data["MN_Sum_RecordA"].nparray[0] == 0 + 20  # the NaN left out
    tab.set_config_option("ignore_nans", False)
    _run(app, "Sum")
    assert np.isnan(app.main._tool.toolfolder.data["MN_Sum_RecordA"].nparray[0])
    assert app.main._tool.config.ignore_nans is False


def test_x_range_and_its_lines(app):
    """The function's x range (#555): in its form, on the Plot as lines to
    drag, and in its command."""
    manager = app.selection_model.manager
    manager.command_history.enabled = True
    tab = app.main
    tab.function_combo.setCurrentText("Average")
    ranges = tab.plot_ranges()
    assert [(r.key, r.label, r.xbgn, r.xend) for r in ranges] == [
        ("x", "Average X range", -np.inf, np.inf)]
    begin, end = tab.form().control("x")
    begin.setText("0.5")
    end.setText("1")
    end.editingFinished.emit()
    assert (tab.plot_ranges()[0].xbgn, tab.plot_ranges()[0].xend) == (0.5, 1.0)
    assert "x" in app.plot_widget.range_overlay.regions
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average")
    output = app.main._tool.toolfolder.data["MN_Avg_RecordA"]
    assert np.allclose(output.nparray, [11.0, 12.0])  # points at x 0.5 and 1
    assert output.xscale.start == 0.5
    commands = [e["command"] for e in manager.command_history.buffer]
    assert any(c.startswith("nm.main.op = NMMainOpAverage(") and "xbgn=0.5, xend=1.0" in c
               for c in commands)
    tab.set_plot_range("x", 0.0, 1.5, final=True)  # dragged: the boxes follow
    assert (begin.text(), end.text()) == ("0", "1.5")
    tab.function_combo.setCurrentText("Max")  # its own range, all of each array
    assert (tab.plot_ranges()[0].label, tab.plot_ranges()[0].xbgn) == ("Max X range", -np.inf)


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
    _run(app, "Average", compute_var=False, compute_sem=False)  # no band to make
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
    window.tool_tabs["Main"].function_combo.setCurrentText("Average")  # has an x range
    assert "x" in window.plot_widget.range_overlay.regions  # its X range



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
    _run(app, "Average", compute_stdv=False)  # the form kept it: unticked
    assert tab._tool.outputs == ["MN_Avg_RecordA"]  # the Stdv is the earlier run's
    assert tab.main2.selected_array() == "MN_Avg_RecordA"



def test_outputs_without_the_mn_prefix(app):
    """Settings › Main, Prefix outputs with MN_ off: Avg_RecordA, its SEM
    and data series too; Main2's ± SEM and its data behind it still found."""
    tab = app.main
    tab.set_config_option("prefix_outputs", False)
    assert tab._tool.config.prefix_outputs is False
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Average", compute_sem=True)
    folder = app.folder.toolfolders["Main_Record_A_All_0"]
    assert list(folder.data.keys()) == ["Avg_RecordA", "SEM_RecordA"]
    assert "Avg_Record" in folder.dataseries
    main2 = tab.main2
    assert main2.selected_array() == "Avg_RecordA"
    assert main2.error_actions["sem"].isEnabled()
    assert not main2.error_actions["stdv"].isEnabled()  # not computed
    assert main2.background_action.isEnabled()
    tab.set_config_option("prefix_outputs", True)


def test_reset_sets_the_chosen_functions_form_back(app):
    """Reset: the chosen function's parameters back to their defaults; the
    other functions' forms keep theirs."""
    tab = app.main
    tab.forms["Max"].set_values({"x": (1.0, 2.0)})
    tab.function_combo.setCurrentText("Average")
    tab.form().set_values({"x": (0.5, 1.0), "compute_sem": True})
    tab.reset_button.click()
    assert tab.form().values() == {"x": (-np.inf, np.inf), "compute_stdv": False,
                                   "compute_var": False, "compute_sem": False}
    assert tab.plot_ranges()[0].xbgn == -np.inf  # its lines follow
    assert tab.forms["Max"].values()["x"] == (1.0, 2.0)


# --- Operations (#557) ---------------------------------------------------------

def test_baseline_changes_the_arrays_after_asking(app):
    """An operation changes the selected arrays themselves, asked first
    (Settings › Main); its window on the Plot in its own color."""
    manager = app.selection_model.manager
    manager.command_history.enabled = True
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab.function_combo.setCurrentText("Baseline")
    assert tab.function_info.text().startswith("Permanent operation")
    tab.form().set_values({"window": (0.0, 0.5)})
    (window,) = tab.plot_ranges()
    assert (window.label, window.xbgn, window.xend, window.color) == (
        "Baseline", 0.0, 0.5, "#16a34a")
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or False  # No
    tab.run_button.click()
    assert asked == ["Baseline changes 3 arrays of Record. Continue?"]
    assert np.allclose(app.series.get_data("A", "E1").nparray, np.arange(4.0) + 10)
    tab._confirm = lambda title, text: True
    tab.run_button.click()
    assert np.allclose(app.series.get_data("A", "E1").nparray, [-0.5, 0.5, 1.5, 2.5])
    commands = [e["command"] for e in manager.command_history.buffer]
    assert commands[-2].startswith("nm.main.op = NMMainOpBaseline(xbgn=0.0, xend=0.5")
    assert len(app.folder.toolfolders) == 0  # no subfolder: the arrays changed
    assert np.allclose(app.plot_widget.current_trace.nparray, [-0.5, 0.5, 1.5, 2.5])


def test_a_locked_series_is_refused(app):
    tab = app.main
    app.series.locked = True
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab.set_config_option("ask_before_changing", False)
    tab.function_combo.setCurrentText("Smooth")
    tab.run_button.click()
    assert tab.status_label.text().endswith(
        "Record is locked: unlock it, or copy it and run on the copy")
    assert np.allclose(app.series.get_data("A", "E1").nparray, np.arange(4.0) + 10)


def test_operation_forms(app):
    """Filter's and Smooth's settings apply by method; Rescale needs units."""
    tab = app.main
    tab.set_config_option("ask_before_changing", False)
    form = tab.forms["Filter"]
    high, q = form._rows["cutoff_high"][1], form._rows["q"][1]
    assert not high.isEnabled() and not q.isEnabled()  # low-pass Butterworth
    form.set_values({"btype": "bandpass"})
    assert high.isEnabled()
    form.set_values({"filter_type": "notch"})
    assert q.isEnabled() and not form._rows["order"][1].isEnabled()
    smooth = tab.forms["Smooth"]
    smooth.set_values({"method": "binomial"})
    assert not smooth._rows["window"][1].isEnabled()
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    tab.function_combo.setCurrentText("Rescale Y")
    tab.run_button.click()  # no units: nothing changed, and why
    assert tab.status_label.text().endswith("To units: enter it")
    tab.function_combo.setCurrentText("Arithmetic")
    tab.form().set_values({"op": "+", "factor": 1.0})
    tab.run_button.click()
    assert np.allclose(app.series.get_data("A", "E0").nparray, np.arange(4.0) + 1)


# --- Points operations (#559) ---------------------------------------------------

def _operate(app, name, **values):
    """Run operation *name* on channel A's epochs, not asked first (a
    question it still asks, e.g. channel B left as it is: Yes, unless the
    test answers)."""
    tab = app.main
    tab.set_config_option("ask_before_changing", False)
    if hasattr(tab._confirm, "__self__"):  # still the dialog: no one to answer it
        tab._confirm = lambda title, text: True
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, name, **values)


def _a1(app):
    return app.series.get_data("A", "E1").nparray


def test_points_operations_change_the_arrays_points(app):
    """Each Points operation from Main1 (A1 starts as 10, 11, 12, 13)."""
    _operate(app, "Insert Points", index=1, n_points=2, fill=np.nan)
    assert np.allclose(_a1(app), [10, np.nan, np.nan, 11, 12, 13], equal_nan=True)
    assert app.main.status_label.text() == "Complete"
    _operate(app, "Replace Values", old_value=np.nan, new_value=-1.0)
    assert np.allclose(_a1(app), [10, -1, -1, 11, 12, 13])
    _operate(app, "Delete Points", index=1, n_points=2)
    assert np.allclose(_a1(app), [10, 11, 12, 13])
    _operate(app, "Rotate", n_points=-1)
    assert np.allclose(_a1(app), [11, 12, 13, 10])
    _operate(app, "Reverse")
    assert np.allclose(_a1(app), [10, 13, 12, 11])
    _operate(app, "Redimension", n_points=6, fill=0.0)
    assert np.allclose(_a1(app), [10, 13, 12, 11, 0, 0])
    assert len(app.plot_widget.current_trace.nparray) == 6  # the Plot follows
    assert app.table_panel.channel_table.model.rowCount() == 6  # and the Table
    app.series.get_data("A", "E1").nparray = np.array([1.0, np.nan, np.inf, 2.0])
    _operate(app, "Delete NaNs", delete_nans=True, delete_infs=True)
    assert np.allclose(_a1(app), [1.0, 2.0])
    assert np.allclose(app.series.get_data("B", "E1").nparray,
                       np.arange(4.0) + 110)  # channel B wasn't selected


def test_points_operations_forms(app):
    """Points' own wording; what can't run says why, nothing changed."""
    tab = app.main
    tab.function_combo.setCurrentText("Delete Points")
    assert tab.function_info.text().startswith(
        "Permanent operation: delete points from the selected arrays")
    assert "Filter & Transform" not in tab.function_info.text()  # no virtual one
    tab.function_combo.setCurrentText("Smooth")
    assert tab.function_info.text().endswith(
        "See channel Filter & Transform to apply virtual operations.")
    _operate(app, "Delete NaNs", delete_nans=False, delete_infs=False)
    assert tab.status_label.text().endswith("Choose Delete NaNs, Delete ±inf, or both")
    _operate(app, "Redimension")  # no number of points yet
    assert tab.status_label.text().endswith("Points: enter a number")
    assert np.allclose(_a1(app), np.arange(4.0) + 10)
    assert tab.forms["Replace Values"].control("old_value").text() == "nan"


def test_points_in_x_follow_the_selection(app):
    """Rotate by 3 points: 1.5 ms, by the selected arrays' x scale (delta 0.5)."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    rotate = tab.forms["Rotate"]
    rotate.set_values({"n_points": 3})
    assert rotate.x_text("n_points") == "(= 1.5 ms)"
    assert tab.forms["Insert Points"].x_text("index") == "(x = 0 ms)"
    app.series.get_data("A", "E0").xscale.delta = 0.1  # the first selected
    tab.refresh()
    assert rotate.x_text("n_points") == "(= 0.3 ms)"


def test_redimension_says_how_many_points_now(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    form = tab.forms["Redimension"]
    assert form.note("n_points") == "currently 4 points"
    _operate(app, "Delete Points", index=0, n_points=1)  # A0, A1, A2: 3 points
    assert form.note("n_points") == "currently 3 points"
    app.series.get_data("A", "E1").nparray = np.arange(6.0)
    tab.refresh()
    assert form.note("n_points") == "currently 3 to 6 points"


def test_delete_and_insert_points_on_the_plot(app):
    """Delete Points' points as a range, Insert Points' place as a line
    (x by delta 0.5); dragged, their point numbers follow."""
    import pyqtgraph as pg
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch=app.series.epochs["E0"],
            epoch_set="All")
    tab.function_combo.setCurrentText("Delete Points")
    tab.form().set_values({"index": 1, "n_points": 2})
    (points,) = tab.plot_ranges()
    assert (points.key, points.label, points.xbgn, points.xend, points.line) == (
        "points", "Delete Points 1 to 2", 0.5, 1.0, False)
    tab.form().set_values({"n_points": 9})  # past the end: to its last point
    assert tab.plot_ranges()[0].xend == 1.5
    tab.set_plot_range("points", 0.0, 1.0, final=True)  # dragged
    assert (tab.form().values()["index"], tab.form().values()["n_points"]) == (0, 3)
    tab.form().set_values({"index": 4})  # no points there: nothing drawn
    assert tab.plot_ranges() == []
    tab.function_combo.setCurrentText("Insert Points")
    tab.form().set_values({"index": 2})
    (line,) = tab.plot_ranges()
    assert (line.label, line.xbgn, line.line) == ("Insert Points at point 2", 1.0, True)
    assert isinstance(app.plot_widget.range_overlay.regions["points"], pg.InfiniteLine)
    tab.set_plot_range("points", 1.5, 1.5, final=True)
    assert tab.form().values()["index"] == 3


def test_points_operations_warn_of_an_x_array_first(app):
    """Arrays with an x array: asked first, even without "Ask before
    changing arrays", as their x values aren't changed."""
    xs = app.folder.data.new("Xs", nparray=np.array([0.0, 1.0, 4.0, 9.0]))
    for epoch in ("E0", "E1", "E2"):
        app.series.get_data("A", epoch).set_xarray_from(xs)
    tab = app.main
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or False  # No
    _operate(app, "Delete Points", index=0, n_points=1)
    assert asked == ["Delete Points changes 3 arrays of Record. Their x array (Xs) "
                     "isn't changed: it won't fit their new number of points, so x "
                     "values will come from the x scale's start and delta. Continue?"]
    assert np.allclose(_a1(app), np.arange(4.0) + 10)  # No: nothing changed
    _operate(app, "Rotate")
    assert asked[-1].endswith("isn't changed: each point will be at a "
                              "different x. Continue?")
    _operate(app, "Arithmetic")  # not a Points operation: not asked
    assert len(asked) == 2


def test_an_x_array_that_no_longer_fits_is_said(app):
    """Points deleted from arrays with an x array: their x values are from
    start and delta, and the status says so."""
    xs = app.folder.data.new("Xs", nparray=np.array([0.0, 1.0, 4.0, 9.0]))
    for epoch in ("E0", "E1"):
        app.series.get_data("A", epoch).set_xarray_from(xs)
    app.main._confirm = lambda title, text: True  # Yes, despite the x array
    _operate(app, "Delete Points", index=0, n_points=1)
    assert app.main.status_label.text() == (
        "Warning: The x array of RecordA0 (and 1 more) no longer fits its points; "
        "x values come from the x scale's start and delta")
    app.series.get_data("A", "E2").set_xarray_from(None)
    _operate(app, "Reverse")  # still doesn't fit: still said
    assert app.main.status_label.text().startswith("Warning: The x array of RecordA0")


# --- X scale operations (#561) ---------------------------------------------------

def test_rescale_x_and_its_channel_template(app):
    """Rescale X: ms to s, each epoch; the channel's template follows a run
    on all its epochs. Its form says the arrays' units now."""
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    assert tab.forms["Rescale X"].note("from_units") == "currently ms"
    assert tab.forms["Rescale Y"].note("from_units") == "currently (none)"
    app.series.get_data("A", "E0").yscale.units = "pA"
    tab.refresh()
    assert tab.forms["Rescale Y"].note("from_units") == "currently (none), pA"
    _operate(app, "Rescale X", to_units="s")
    a1 = app.series.get_data("A", "E1")
    assert (a1.xscale.delta, a1.xscale.units) == (0.0005, "s")
    channel = app.series.channels["A"]
    assert (channel.xscale.delta, channel.xscale.units) == (0.0005, "s")
    assert app.series.get_data("B", "E1").xscale.units == "ms"  # B not run
    assert tab.forms["Rescale X"].note("from_units") == "currently s"


def test_x_scale_only_operations_arent_permanent(app):
    """Rescale X and Align change only the x scale: said, and not asked
    about (Ask before changing arrays); Resample changes the values."""
    tab = app.main
    tab.function_combo.setCurrentText("Align")
    assert tab.function_info.text().startswith("X scale only: shift the x")
    assert tab.function_info.text().endswith("The values aren't changed.")
    tab.function_combo.setCurrentText("Resample")
    assert tab.function_info.text().startswith("Permanent operation")
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or True
    _select(app, all_channels=True, epoch_set="All")  # (one channel: asked)
    _run(app, "Rescale X", to_units="s")  # Ask before changing arrays: on
    assert asked == [] and app.series.get_data("A", "E0").xscale.units == "s"
    _run(app, "Resample", delta=0.001)
    assert len(asked) == 1  # permanent: asked


def test_rescale_x_and_align_on_one_channel_ask_first(app):
    """Channel A of A and B: B keeps its x, so asked first (even without
    Ask before changing arrays); All channels: not asked."""
    tab = app.main
    asked = []
    tab._confirm = lambda title, text: asked.append(text) or False  # No
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    _run(app, "Rescale X", to_units="s")
    assert asked == ["Rescale X changes 3 arrays of Record. Channel B isn't changed, "
                     "so its x units will differ from A's (choose All channels to "
                     "change them together). Continue?"]
    assert app.series.get_data("A", "E0").xscale.units == "ms"  # No: unchanged
    _run(app, "Set X Start", value=1.0)
    assert "its epochs will no longer line up in time with A's" in asked[-1]
    _run(app, "Resample", delta=0.25)  # values kept at their x: not about B
    assert "Channel B" not in asked[-1]
    tab.set_config_option("ask_before_changing", False)
    _select(app, all_channels=True, epoch_set="All")
    _run(app, "Rescale X", to_units="s")
    assert app.series.get_data("B", "E0").xscale.units == "s" and len(asked) == 3


def test_rescale_x_refuses_an_x_array(app):
    xs = app.folder.data.new("Xs", nparray=np.array([0.0, 1.0, 4.0, 9.0]))
    app.series.get_data("A", "E1").set_xarray_from(xs)
    _operate(app, "Rescale X", to_units="s")
    assert app.main.status_label.text() == (
        "Warning: RecordA1 has an x array (Xs): Rescale X changes start and "
        "delta, not its values; change the x array itself")
    assert app.series.get_data("A", "E0").xscale.units == "ms"  # none changed


def test_resample_says_the_points_now_and_after(app):
    tab = app.main
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    form = tab.forms["Resample"]
    assert form.note("delta") == "currently 0.5 ms, 4 points"
    form.control("delta").setText("1")
    form.control("delta").editingFinished.emit()
    assert form.note("delta") == "currently 0.5 ms, 4 points; then about 2"
    _operate(app, "Resample", delta=1.0)
    a1 = app.series.get_data("A", "E1")
    assert a1.xscale.delta == 1.0 and a1.nparray.size == 2
    assert form.note("delta").startswith("currently 1 ms, 2 points")


def test_interpolate_onto_each_channels_common_x(app):
    """A common x axis per channel: A's epochs shifted apart are put onto
    their overlap; B (its own x) onto its own."""
    for epoch, start in (("E0", 0.0), ("E1", 0.5), ("E2", 1.0)):
        app.series.get_data("A", epoch).xscale.start = start
    _select(app, all_channels=True, epoch_set="All")
    _operate_all = app.main
    _operate_all.set_config_option("ask_before_changing", False)
    _run(app, "Interpolate")
    a0, b0 = app.series.get_data("A", "E0"), app.series.get_data("B", "E0")
    assert (a0.xscale.start, a0.nparray.size) == (1.0, 2)  # overlap: x 1 to 1.5
    assert np.allclose(a0.nparray, [2.0, 3.0])  # A0's points at x 1 and 1.5
    assert (b0.xscale.start, b0.nparray.size) == (0.0, 4)  # B: unchanged x
    # Onto a template's x: picked from the folder (or a subfolder)
    form = app.main.forms["Interpolate"]
    form.set_values({"x_source": "template"})
    names = [form.control("template_name").itemText(i)
             for i in range(form.control("template_name").count())]
    assert "Single" in names


def test_align_from_an_array_of_values(app):
    """Align: one value per array from an array (e.g. Stats2's), at x = 0;
    a NaN reference: that array left as it is, and said."""
    tab = app.main
    app.folder.data.new("Onsets", nparray=np.array([0.5, np.nan, 1.5]),
                        yscale={"units": "ms"})
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    form = tab.forms["Align"]
    assert [p.name for p in form.params] == ["values_array", "target"]  # an array only
    _operate(app, "Align", values_array="Onsets", target=0.0)
    starts = [app.series.get_data("A", e).xscale.start for e in ("E0", "E1", "E2")]
    assert starts == [-0.5, 0.0, -1.5]
    assert tab.status_label.text() == (
        "Warning: RecordA1 not aligned: its reference value is NaN")
    app.folder.data.new("Twos", nparray=np.array([1.0, 1.0, 1.0]), yscale={"units": "ms"})
    tab.refresh()  # as the app does when arrays are added: Twos to pick
    _operate(app, "Align", values_array="Twos", target="mean")  # mean 1: no shift
    assert app.series.get_data("A", "E0").xscale.start == -0.5


def test_set_x_start_sets_or_shifts(app):
    """Set X Start: every array starts at the value, or each is shifted by
    it (their differences kept); X scale only; refuses an x array."""
    tab = app.main
    for epoch, start in (("E0", 0.0), ("E1", 2.0), ("E2", 4.0)):
        app.series.get_data("A", epoch).xscale.start = start
    tab.function_combo.setCurrentText("Set X Start")
    assert tab.function_info.text().startswith("X scale only: set the x-scale start")
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    assert tab.forms["Set X Start"].note("value") == "ms"  # the arrays' x units
    _operate(app, "Set X Start", mode="shift", value=-10.0)
    starts = [app.series.get_data("A", e).xscale.start for e in ("E0", "E1", "E2")]
    assert starts == [-10.0, -8.0, -6.0]
    _operate(app, "Set X Start", mode="set", value=0.0)
    starts = [app.series.get_data("A", e).xscale.start for e in ("E0", "E1", "E2")]
    assert starts == [0.0, 0.0, 0.0]
    assert app.series.channels["A"].xscale.start == 0.0  # the template too
    xs = app.folder.data.new("Xs", nparray=np.array([0.0, 1.0, 4.0, 9.0]))
    app.series.get_data("A", "E1").set_xarray_from(xs)
    _operate(app, "Set X Start", mode="set", value=5.0)
    assert tab.status_label.text().endswith(
        "Set X Start changes start and delta, not its values; change the x array itself")


def test_align_needs_a_value_per_array(app):
    app.folder.data.new("Two", nparray=np.array([0.5, 1.0]), yscale={"units": "ms"})
    _operate(app, "Align", values_array="Two")
    assert app.main.status_label.text().endswith(
        "Align: Two has 2 values for 3 arrays (one per array, in order)")


def test_align_lists_only_x_values(app):
    """Align's Array: only arrays in the selected arrays' x units (ms),
    e.g. Stats2's ..._x, not their y values, a duration (_dx) or text."""
    folder = app.folder
    sub = folder.toolfolders.new(name="Stats_Record_A_All_0")
    sub.data.new("ST_w0_max_x", nparray=np.array([1.0, 1.5, 0.5]), yscale={"units": "ms"})
    sub.data.new("ST_w0_max_y", nparray=np.array([3.0, 13.0, 23.0]), yscale={"units": "mV"})
    sub.data.new("ST_w0_rt_p_dx", nparray=np.array([0.2, 0.3, 0.1]), yscale={"units": "ms"})
    sub.data.new("ST_data_names", nparray=np.array(["a", "b", "c"], dtype=object))
    spike = folder.toolfolders.new(name="Spike_Record_A_All_0")
    for name in ("SP_first", "SP_isi", "SP_RX"):  # not x positions, one per array
        spike.data.new(name, nparray=np.array([1.0, 2.0, 3.0]), yscale={"units": "ms"})
    sub.data.new("ST_w0_rt_p_p0_x", nparray=np.array([np.nan, 1.0, 1.2]))  # no units
    _select(app, channel=app.series.channels["A"], epoch_set="All")
    app.main.function_combo.setCurrentText("Align")
    combo = app.main.form().control("values_array")
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Stats_Record_A_All_0/ST_w0_max_x", "Stats_Record_A_All_0/ST_w0_rt_p_p0_x"]
    template = app.main.forms["Interpolate"].control("template_name")
    assert template.count() > 1  # Interpolate: any array's x will do
