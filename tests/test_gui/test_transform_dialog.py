"""Tests for Filter & Transform: a channel's filters and transforms in the GUI."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_transform import NMTransformBaseline, NMTransformInvert
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.transform_dialog import (
    MARK,
    MARK_COLOR,
    MARK_STYLE,
    applies,
    step_text,
)

pytestmark = pytest.mark.gui

Checked = QtCore.Qt.CheckState.Checked
Unchecked = QtCore.Qt.CheckState.Unchecked


@pytest.fixture
def app(qtbot):
    """Series Record: channels A, B x 3 epochs; y = 1, 2, 3, 4 + 10 * epoch."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    for channel in series.channels.values():
        for i, d in enumerate(channel.data):
            d.nparray = np.array([1.0, 2.0, 3.0, 4.0]) + 10 * i
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.series = series
    return window


def _open(app):
    panel = app.plot_widget
    panel.transform_action.trigger()
    return panel._transform_dialog


def _plotted(app):
    """The y values of the plotted curve."""
    items = app.plot_widget.plot_widget.getPlotItem().listDataItems()
    return list(items[0].yData)


def _add(dialog, name):
    dialog.add_actions[name].trigger()


def _tab_texts(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def _combo_texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


# --- the menu and window ----------------------------------------------------

def test_menu_item_for_a_series_channel(app):
    panel = app.plot_widget
    assert panel.transform_action.isEnabled()
    dialog = _open(app)
    assert dialog.isVisible() and dialog.channel is app.series.channels["A"]
    assert dialog.title_label.text() == "Record  channel A"
    assert dialog.step_list.count() == 0 and not dialog.remove_button.isEnabled()


def test_not_for_a_data_array(app):
    data = app.series.channels["A"].data[0]
    app.selection_model.update(source="data", data=data)
    assert not app.plot_widget.transform_action.isEnabled()


def test_add_a_step_redraws_and_marks(app):
    dialog = _open(app)
    assert _plotted(app) == [1.0, 2.0, 3.0, 4.0]
    _add(dialog, "Invert")
    channel = app.series.channels["A"]
    assert [t.type_str for t in channel.transform] == ["NMTransformInvert"]
    assert _plotted(app) == [-1.0, -2.0, -3.0, -4.0]
    assert list(channel.data[0].nparray) == [1.0, 2.0, 3.0, 4.0]  # unchanged
    # ƒ on the tab, the trace name and the strip's channel
    panel = app.plot_widget
    assert _tab_texts(panel.view_tabs)[:2] == [f"A {MARK}", "B"]
    assert "Invert" in panel.view_tabs.tabToolTip(0)
    label = panel.trace_name_label.text()
    assert label.startswith("RecordA0") and f'{MARK_STYLE}">{MARK}' in label  # red, bold ƒ
    assert app.selection_strip.transform_mark_label.styleSheet() == MARK_STYLE  # the same
    assert panel.view_tabs.tabTextColor(0) == QtGui.QColor(MARK_COLOR)
    strip = app.selection_strip
    assert not strip.transform_mark_label.isHidden()  # a red ƒ beside the channel
    assert "1. Invert" in strip.transform_mark_label.toolTip()
    combo = strip.combo("channel")
    assert "A" in _combo_texts(combo) and combo.currentData() == "A"
    assert "1. Invert" in combo.toolTip()
    assert _tab_texts(app.table_panel.view_tabs)[:2] == [f"A {MARK}", "B"]


def test_turn_a_step_off(app):
    dialog = _open(app)
    _add(dialog, "Invert")
    dialog.step_list.item(0).setCheckState(Unchecked)
    assert app.series.channels["A"].transform[0].on is False
    assert _plotted(app) == [1.0, 2.0, 3.0, 4.0]
    assert _tab_texts(app.plot_widget.view_tabs)[0] == "A"
    assert app.selection_strip.transform_mark_label.isHidden()
    assert MARK not in app.plot_widget.trace_name_label.text()
    dialog.step_list.item(0).setCheckState(Checked)
    assert _plotted(app) == [-1.0, -2.0, -3.0, -4.0]


def test_add_remove_and_undo(app):
    channel = app.series.channels["A"]
    channel.transform = [NMTransformBaseline(xbgn=0, xend=0)]
    dialog = _open(app)
    _add(dialog, "Invert")  # after the current step
    names = lambda: [t.type_str[11:] for t in channel.transform]  # noqa: E731
    assert names() == ["Baseline", "Invert"]
    assert _plotted(app) == [0.0, -1.0, -2.0, -3.0]
    assert dialog.undo_button.text() == "Undo"
    dialog.step_list.setCurrentRow(0)
    dialog.remove_button.click()
    assert names() == ["Invert"]
    assert dialog.undo_button.isEnabled()
    dialog.undo_button.click()
    assert names() == ["Baseline"]
    assert not dialog.undo_button.isEnabled()
    assert _plotted(app) == [0.0, 1.0, 2.0, 3.0]


def test_clear_removes_every_step(app):
    channel = app.series.channels["A"]
    channel.transform = [NMTransformInvert(), NMTransformBaseline(xbgn=0, xend=0)]
    dialog = _open(app)
    assert dialog.clear_button.isEnabled()
    dialog.clear_button.click()
    assert channel.transform == [] and dialog.step_list.count() == 0
    assert not dialog.clear_button.isEnabled()
    assert _plotted(app) == [1.0, 2.0, 3.0, 4.0]
    dialog.undo_button.click()  # puts them back
    assert len(channel.transform) == 2


def test_undo_goes_back_to_when_reopened(app):
    channel = app.series.channels["A"]
    dialog = _open(app)
    _add(dialog, "Invert")
    dialog.close()
    _open(app)  # opened again: Invert is where Undo goes back to
    assert not dialog.undo_button.isEnabled()
    _add(dialog, "Baseline")
    dialog.undo_button.click()
    assert [t.type_str for t in channel.transform] == ["NMTransformInvert"]


def test_settings_form(app):
    _rate(app, 0.05)  # 20 kHz
    dialog = _open(app)
    _add(dialog, "Filter")
    t = app.series.channels["A"].transform[0]
    editors = dialog._editors
    assert set(editors) == {"filter_type", "cutoff", "sample_rate", "order", "btype", "q"}
    assert not editors["q"].isEnabled() and editors["order"].isEnabled()
    editors["filter_type"].setCurrentText("notch")
    assert t.filter_type == "notch"
    assert editors["q"].isEnabled() and not editors["order"].isEnabled()
    assert dialog.step_list.item(0).text() == step_text(t) == "Filter: notch, 50 Hz"
    # A band-pass cutoff: low, high
    editors["filter_type"].setCurrentText("butterworth")
    editors["btype"].setCurrentText("bandpass")
    editors["cutoff"].setText("300, 3000")
    editors["cutoff"].editingFinished.emit()
    assert t.cutoff == [300.0, 3000.0]
    assert editors["order"].minimum() == 1
    # Not a number: the value is kept and the reason shown
    editors["sample_rate"].setText("abc")
    editors["sample_rate"].editingFinished.emit()
    assert t.sample_rate is None and editors["sample_rate"].text() == ""
    assert "not a number" in dialog.status_label.text()


def test_smooth_settings_follow_the_method(app):
    dialog = _open(app)
    _add(dialog, "Smooth")
    t = app.series.channels["A"].transform[0]
    assert applies(t, "window") and not applies(t, "polyorder")
    # The window is odd: its arrows step by 2, and an even number is refused
    window = dialog._editors["window"]
    assert window.minimum() == 3 and window.singleStep() == 2
    window.stepUp()
    assert t.window == 7 and dialog.status_label.text() == ""
    window.setValue(8)
    assert t.window == 7 and window.value() == 7
    assert "window" in dialog.status_label.text().lower()
    dialog._editors["method"].setCurrentText("savgol")
    assert dialog._editors["polyorder"].isEnabled()
    assert not dialog._editors["passes"].isEnabled()
    window.stepUp()  # savgol's window is odd too
    assert t.method == "savgol" and t.window == 9


def test_window_follows_the_plotted_channel(app):
    dialog = _open(app)
    panel = app.plot_widget
    panel.view_tabs.select_channel("B")
    assert dialog.channel is app.series.channels["B"]
    assert dialog.title_label.text() == "Record  channel B"


def test_a_step_that_fails_is_shown_raw(app):
    dialog = _open(app)
    _add(dialog, "Baseline")
    t = app.series.channels["A"].transform[0]
    dialog._editors["xbgn"].setText("2")
    dialog._editors["xbgn"].editingFinished.emit()  # xend 0 < xbgn 2
    assert t.xbgn == 2.0
    assert _plotted(app) == [1.0, 2.0, 3.0, 4.0]
    assert "not applied" in app.plot_widget.overlay_status_label.text()
    assert dialog.status_label.text().startswith("Can't apply to RecordA0")


# --- the Table ----------------------------------------------------------------

def test_table_show_transformed(app):
    panel = app.table_panel
    action = panel.transformed_action
    assert not action.isEnabled()  # no steps on
    app.series.channels["A"].transform = [NMTransformInvert()]
    app._on_transforms_changed()
    assert action.isEnabled() and not action.isChecked()
    model = panel.channel_table.model
    assert model.data(model.index(0, 1)) == "1"  # as stored, by default
    assert f"{MARK} not shown" in panel.channel_table.title_label.text()
    action.setChecked(True)
    assert model.data(model.index(0, 1)) == "-1"
    assert f"{MARK} transformed" in panel.channel_table.title_label.text()
    assert not (model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable)
    view = panel.channel_table.view
    view.clearSelection()
    view.selectColumn(1)
    panel.copy_action.trigger()
    assert QtWidgets.QApplication.clipboard().text().splitlines()[1] == "0\t-1"
    action.setChecked(False)
    assert model.data(model.index(0, 1)) == "1"
    assert model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable


# --- Stats ------------------------------------------------------------------

def test_stats_measures_what_is_plotted(app):
    app.series.channels["A"].transform = [NMTransformInvert()]
    tab = app.stats_tab
    tab.measurement_combo.setCurrentText("mean")
    app.selection_model.update(epoch_set="All")
    tab.run_button.click()
    folder = tab.stats_tool.toolfolder
    assert list(folder.data["ST_w0_mean_y"].nparray) == [-2.5, -12.5, -22.5]


# --- the strip's channel ----------------------------------------------------

def _shown(tabs):
    return tabs.current_view()


def test_plot_and_table_follow_the_strip_channel(app):
    plot, table = app.plot_widget, app.table_panel
    series = app.series
    assert _shown(plot.view_tabs) == ("channel", "A") == _shown(table.view_tabs)
    app.selection_model.update(channel=series.channels["B"])
    assert _shown(plot.view_tabs) == ("channel", "B") == _shown(table.view_tabs)
    assert plot.current_trace.name == "RecordB0"
    assert table.channel_table.model.arrays[0].name == "RecordB0"
    # A tab shows another channel, until the strip's channel changes again
    plot.view_tabs.select_channel("A")
    app.selection_model.update(epoch=series.epochs["E1"])
    assert _shown(plot.view_tabs) == ("channel", "A")
    assert _shown(table.view_tabs) == ("channel", "B")
    app.selection_model.update(channel=series.channels["A"])
    app.selection_model.update(channel=series.channels["B"])
    assert _shown(plot.view_tabs) == ("channel", "B")


# --- typing values, and filter messages ---------------------------------------

def test_return_keeps_the_step(app, qtbot):
    app.show()
    dialog = _open(app)
    _add(dialog, "Filter")
    _add(dialog, "Baseline")
    dialog.step_list.setCurrentRow(0)
    editor = dialog._editors["cutoff"]
    editor.setFocus()
    editor.setText("500")
    qtbot.keyClick(editor, QtCore.Qt.Key.Key_Return)
    steps = app.series.channels["A"].transform
    assert [t.type_str for t in steps] == ["NMTransformFilter", "NMTransformBaseline"]
    assert steps[0].cutoff == 500.0
    assert dialog.isVisible() and dialog.step_list.currentRow() == 0
    assert not any(b.autoDefault() for b in dialog.findChildren(QtWidgets.QPushButton))


def test_too_few_points_to_filter_says_so(app):
    dialog = _open(app)  # the arrays have 4 points
    _add(dialog, "Filter")
    assert "the array has 4 points, too few for this filter" in dialog.status_label.text()


def test_filter_messages_name_the_frequencies(app):
    for d in app.series.channels["A"].data:  # 0.05 ms steps: 20 kHz
        d.xscale.delta, d.xscale.units = 0.05, "ms"
    dialog = _open(app)
    _add(dialog, "Filter")
    assert dialog._editors["sample_rate"].placeholderText() == "20000 (x scale)"
    dialog._editors["cutoff"].setText("15000")
    dialog._editors["cutoff"].editingFinished.emit()
    text = dialog.status_label.text()
    assert "cutoff 15000 Hz must be above 0 and below the Nyquist frequency, 10000 Hz" in text
    assert "half the sample rate, 20000 Hz" in text


def test_means_ignore_nans_without_a_setting(app):
    dialog = _open(app)
    for name in ("Baseline", "dF/F"):
        _add(dialog, name)
        assert "ignore_nans" not in dialog._editors
        assert app.series.channels["A"].transform[-1].ignore_nans is True
    assert set(dialog._editors) == {"xbgn", "xend"}


def test_normalize_points_only_for_mean_at_min_or_max(app):
    dialog = _open(app)
    _add(dialog, "Normalize")
    editors = dialog._editors
    assert not editors["min_mean_n"].isEnabled() and not editors["max_mean_n"].isEnabled()
    editors["min_fxn"].setCurrentText("mean@min")
    assert editors["min_mean_n"].isEnabled() and not editors["max_mean_n"].isEnabled()
    editors["max_fxn"].setCurrentText("mean@max")
    assert editors["max_mean_n"].isEnabled()
    editors["min_fxn"].setCurrentText("min")
    assert not editors["min_mean_n"].isEnabled()


def test_dfof_with_f0_of_zero_says_why(app):
    for d in app.series.channels["A"].data:
        d.nparray[0] = 0.0  # F0 = the value at x = 0
    dialog = _open(app)
    _add(dialog, "dF/F")
    assert _plotted(app) == [0.0, 2.0, 3.0, 4.0]  # shown as it is, not blank
    assert "not applied" in app.plot_widget.overlay_status_label.text()
    assert "F0 (the mean over x 0 to 0) is 0" in dialog.status_label.text()


# --- the Browser --------------------------------------------------------------

def _tree_menu(app, monkeypatch, obj):
    browser = app.browser_widget
    browser.show_object(obj)
    index = browser.tree.currentIndex()
    captured = {}
    monkeypatch.setattr(QtWidgets.QMenu, "exec",
                        lambda menu, *a, **k: captured.setdefault("menu", menu))
    monkeypatch.setattr(browser.tree, "indexAt", lambda pos: index)
    browser._show_tree_context_menu(QtCore.QPoint(0, 0))
    return {a.text(): a for a in captured["menu"].actions() if a.text()}


def test_browser_info_shows_the_steps(app):
    browser = app.browser_widget
    channel = app.series.channels["A"]
    browser.show_object(channel)
    panel = browser.info_panel
    assert panel.row_text("Filter & Transform") == "—"
    assert panel.transforms_button.isVisibleTo(panel)
    channel.transform = [NMTransformInvert(), NMTransformBaseline(xbgn=0, xend=1)]
    app._on_transforms_changed()
    assert panel.row_text("Filter & Transform") == "1. Invert\n2. Baseline: x 0 to 1"
    browser.show_object(app.series)
    assert panel.row_text("Filter & Transform") == "A"  # the channels with steps on
    assert not panel.transforms_button.isVisibleTo(panel)


def test_browser_shows_transformed_channels_in_red(app):
    channel = app.series.channels["A"]
    channel.transform = [NMTransformInvert()]
    browser = app.browser_widget
    browser.show_object(channel)
    index = browser.tree.currentIndex()
    model = browser.tree.model()
    color = model.data(index, QtCore.Qt.ItemDataRole.ForegroundRole).color()
    assert color == QtGui.QColor(MARK_COLOR)
    assert "1. Invert" in model.data(index, QtCore.Qt.ItemDataRole.ToolTipRole)
    other = index.sibling(1, 0)  # B
    assert model.data(other, QtCore.Qt.ItemDataRole.ForegroundRole) is None


def test_browser_opens_the_window_for_that_channel(app, monkeypatch):
    channel = app.series.channels["B"]
    actions = _tree_menu(app, monkeypatch, channel)
    actions["Filter && Transform..."].trigger()
    dialog = app.plot_widget._transform_dialog
    assert dialog.isVisible() and dialog.channel is channel
    assert app.plot_widget.view_tabs.current_view() == ("channel", "B")
    assert app.context_panel.currentWidget() is app.plot_widget
    # The Info panel's button does the same
    app.browser_widget.show_object(app.series.channels["A"])
    app.browser_widget.info_panel.transforms_button.click()
    assert dialog.channel is app.series.channels["A"]


# --- the strip's Channel ⋯ menu -----------------------------------------------

def _texts(menu):
    return [a.text() for a in menu.actions() if a.text()]


def test_channel_menu_in_strip_and_menu_bar(app):
    button = app.selection_strip.action_button("channel")
    assert button.isEnabled()
    assert _texts(button.menu()) == ["Add Channels…", "Remove Channel…", "Filter && Transform...", "Show in Browser"]
    series_menu = {a.text(): a.menu() for a in app.menuBar().actions()}["Series"]
    channel_menu = {a.text(): a.menu() for a in series_menu.actions()}["Channel"]
    assert _texts(channel_menu) == ["Add Channels…", "Remove Channel…", "Filter && Transform...", "Show in Browser"]


def test_channel_menu_opens_the_window_for_the_strip_channel(app):
    app.selection_model.update(channel=app.series.channels["B"])
    app.channel_actions.transform_action.trigger()
    dialog = app.plot_widget._transform_dialog
    assert dialog.isVisible() and dialog.channel is app.series.channels["B"]


def test_channel_menu_shows_the_channel_in_the_browser(app):
    app.selection_model.update(channel=app.series.channels["B"])
    app.channel_actions.browse_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    assert app.browser_widget.tree.currentIndex().internalPointer() is app.series.channels["B"]


def test_channel_menu_needs_a_series_channel(app):
    data = app.series.channels["A"].data[0]
    app.selection_model.update(source="data", data=data, dataseries=None)
    actions = app.channel_actions
    assert not actions.transform_action.isEnabled()
    assert not actions.browse_action.isEnabled()


# --- starting cutoffs -------------------------------------------------------

def _rate(app, delta_ms):
    for d in app.series.channels["A"].data:
        d.xscale.delta, d.xscale.units = delta_ms, "ms"


def test_start_cutoff_is_below_nyquist():
    from pyneuromatic.gui.transform_dialog import start_cutoff
    assert start_cutoff(20000) == 1000.0  # the usual cutoff fits
    assert start_cutoff(2000) == 200.0  # 1000 Hz is Nyquist: a fifth of it
    assert start_cutoff(500) == 50.0
    assert start_cutoff(100) == 10.0
    assert start_cutoff(None) == 1000.0  # no x scale: as before
    assert start_cutoff(20000, 50) == 50.0
    assert start_cutoff(80, 50) == 5.0  # a notch at 50 Hz would be above 40 Hz


def test_added_filter_starts_in_range(app):
    for d in app.series.channels["A"].data:
        d.nparray = np.sin(np.arange(200) / 5.0)  # long enough to filter
    _rate(app, 0.5)  # 2 kHz: 1000 Hz would be at Nyquist
    dialog = _open(app)
    _add(dialog, "Filter")
    t = app.series.channels["A"].transform[0]
    assert t.cutoff == 200.0 and dialog._editors["cutoff"].text() == "200"
    assert dialog.status_label.text() == ""  # applied, no error
    assert "not applied" not in app.plot_widget.overlay_status_label.text()


def test_notch_starts_at_50_hz(app):
    _rate(app, 0.05)  # 20 kHz
    dialog = _open(app)
    _add(dialog, "Filter")
    t = app.series.channels["A"].transform[0]
    assert t.cutoff == 1000.0
    dialog._editors["filter_type"].setCurrentText("notch")
    assert t.cutoff == 50.0 and dialog._editors["cutoff"].text() == "50"
    dialog._editors["filter_type"].setCurrentText("bessel")
    assert t.cutoff == 1000.0  # back to the usual cutoff
    # A typed cutoff is kept
    dialog._editors["cutoff"].setText("300")
    dialog._editors["cutoff"].editingFinished.emit()
    dialog._editors["filter_type"].setCurrentText("notch")
    assert t.cutoff == 300.0


def test_cutoff_tip_follows_the_filter_type(app):
    _rate(app, 0.05)  # 20 kHz
    dialog = _open(app)
    _add(dialog, "Filter")
    editors = dialog._editors
    assert "passes frequencies below" in editors["cutoff"].toolTip()
    editors["btype"].setCurrentText("bandpass")
    assert editors["cutoff"].toolTip() == "Low, high, in Hz (e.g. 300, 3000)"
    editors["filter_type"].setCurrentText("notch")
    assert "e.g. 50 for mains" in editors["cutoff"].toolTip()
    # Two cutoffs for a notch: said plainly
    editors["cutoff"].setText("300, 3000")
    editors["cutoff"].editingFinished.emit()
    assert "a notch filter removes one frequency" in dialog.status_label.text()


def test_bandpass_example_fits_the_sample_rate(app):
    _rate(app, 0.5)  # 2 kHz: 3000 Hz is above Nyquist
    dialog = _open(app)
    _add(dialog, "Filter")
    dialog._editors["btype"].setCurrentText("bandpass")
    assert dialog._editors["cutoff"].toolTip() == "Low, high, in Hz (e.g. 20, 200)"
    assert "(e.g. 20, 200)" in dialog.status_label.text()  # one cutoff isn't enough
