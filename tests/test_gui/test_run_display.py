"""Tests for showing each epoch in the Plot while a tool runs (#468)."""

import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6 import QtCore, QtTest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.selection_model import ALL_EPOCHS
from pyneuromatic.gui.settings_dialog import SettingsDialog

pytestmark = pytest.mark.gui

N = 6


@pytest.fixture
def app(qtbot):
    """Channel A x 6 epochs, y = epoch number (so the mean of En is n);
    Stats mean, on All epochs, E0 selected."""
    manager = NMManager(quiet=True, command_history=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=N, n_points=5)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = np.full(5, float(i))
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(epoch_set=ALL_EPOCHS)
    tab = window.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    window.tool_rail.select_tool("Stats")
    window.stats, window.manager, window.series = tab, manager, series
    return window


def _display(app, mode, delay=0.0):
    app.manager.settings.run_display = mode
    app.manager.settings.run_delay = delay
    app._apply_settings()


def _record(app):
    """What the Plot shows each time a run shows an epoch."""
    seen = []

    def shown(data, _channel):
        if data is not None:
            plot = app.plot_widget
            seen.append((data.name, plot.current_trace.name, plot.trace_index.value(),
                         [m.y[0] for m in plot.mark_overlay.marks]))

    app.stats.run_trace_shown.connect(shown)
    return seen


def _commands(app):
    return [e["command"] for e in app.manager.command_history.buffer]


def test_each_epoch_is_the_default():
    assert NMManager(quiet=True).settings.run_display == "each"
    assert NMManager(quiet=True).settings.run_delay == 0.0


def test_off_shows_nothing(app):
    _display(app, "off")
    seen = _record(app)
    app.stats.run_button.click()
    assert seen == []
    assert app.stats.status_label.text() == "Complete"


def test_each_epoch_is_shown_with_the_runs_results(app):
    _display(app, "each")
    seen = _record(app)
    count = len(_commands(app))
    app.stats.run_button.click()
    assert [s[:3] for s in seen] == [(f"RecordA{i}", f"RecordA{i}", i) for i in range(N)]
    assert [s[3] for s in seen] == [[float(i)] for i in range(N)]  # each epoch's mean
    # After the run: the Plot as before, the selection unchanged
    plot = app.plot_widget
    assert not plot.run_display
    assert plot.current_trace.name == "RecordA0" and plot.trace_index.value() == 0
    assert app.selection_model.selection["epoch"].name == "E0"
    # One command: no selection commands for the epochs shown
    commands = _commands(app)[count:]
    assert [c for c in commands if not c.startswith("nm.stats.windows")] == [
        c for c in commands if c.startswith("nm.stats.run_selection(")]
    assert sum(c.startswith("nm.stats.run_selection(") for c in commands) == 1
    assert not any(c.startswith("nm.select_keys") for c in commands)


def test_the_plot_is_shown_for_the_run(app):
    _display(app, "each")
    app.context_panel.setCurrentWidget(app.table_panel)
    panels = []
    app.stats.run_trace_shown.connect(
        lambda data, _c: data is not None and panels.append(app.context_panel.currentWidget()))
    app.stats.run_button.click()
    assert panels and all(p is app.plot_widget for p in panels)


def test_delay_and_the_window_responds(app):
    _display(app, "each", delay=0.05)
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    started = time.monotonic()
    app.stats.run_button.click()
    elapsed = time.monotonic() - started
    timer.stop()
    assert elapsed >= N * 0.05 * 0.9
    assert len(ticks) >= 3  # events handled while waiting


def test_stop_during_the_delay(app):
    _display(app, "each", delay=5.0)
    app.stats.run_trace_shown.connect(
        lambda data, _c: data is not None and QtCore.QTimer.singleShot(20, app.stats.stop))
    started = time.monotonic()
    app.stats.run_button.click()
    assert time.monotonic() - started < 3.0  # not waited out
    assert app.stats.status_label.text() == f"Cancelled after 1 of {N}"


def _answer(app, answers):
    """Answer the prompt in turn; returns the prompts' texts."""
    prompts = []

    def shown(text):
        prompts.append(text)
        answer = answers.pop(0) if answers else "cancel"
        QtCore.QTimer.singleShot(0, lambda: app.stats.answer_prompt(answer))

    app.stats.prompt_shown.connect(shown)
    return prompts


def test_ask_continue_and_continue_all(app):
    _display(app, "ask")
    prompts = _answer(app, ["continue", "all"])
    seen = _record(app)
    app.stats.run_button.click()
    assert prompts == ["E0 (1 of 6)", "E1 (2 of 6)"]  # the result: in Details (#471)
    assert len(seen) == N  # every epoch still shown after Continue All
    assert app.stats.status_label.text() == "Complete"
    assert not any(b.isVisibleTo(app.stats) for b in app.stats.prompt_buttons.values())


def test_ask_cancel_stops_there(app):
    _display(app, "ask")
    _answer(app, ["continue", "continue", "cancel"])
    app.stats.run_button.click()
    assert app.stats.status_label.text() == f"Cancelled after 3 of {N}"
    values = app.stats.stats_tool.toolfolder.data.get("ST_w0_mean_y").nparray
    assert list(values[~np.isnan(values)]) == [0.0, 1.0, 2.0]
    assert app.plot_widget.current_trace.name == "RecordA0"  # back as before


def test_ask_return_continues_and_esc_cancels(app):
    _display(app, "ask")
    keys = [QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Escape]

    def shown(_text):
        key = keys.pop(0) if keys else QtCore.Qt.Key.Key_Escape
        QtCore.QTimer.singleShot(0, lambda: QtTest.QTest.keyClick(app, key))

    app.stats.prompt_shown.connect(shown)
    app.stats.run_button.click()
    assert app.stats.status_label.text() == f"Cancelled after 2 of {N}"


def test_overlay_traces_are_not_shown_during_a_run(app):
    _display(app, "each")
    plot = app.plot_widget
    plot.overlay_count.setValue(2)
    curves = []
    app.stats.run_trace_shown.connect(
        lambda data, _c: data is not None and curves.append(
            len(plot.plot_widget.getPlotItem().listDataItems())
            - len(plot.mark_overlay.items)))
    app.stats.run_button.click()
    assert curves and all(n == 1 for n in curves)
    assert plot.overlay_count.value() == 2
    assert plot.rendered_trace_count > 1  # overlays back after the run


def test_the_plot_takes_clicks_during_a_run(app):
    assert app.plot_widget.plot_widget in app.stats.allowed_during_run()


def test_display_controls_and_settings_agree(app, qtbot):
    tab = app.stats
    assert tab.display_row.isVisible() or not tab.isVisible()
    assert tab.display_combo.currentData() == "each"  # the default
    tab.display_combo.setCurrentIndex(tab.display_combo.findData("off"))
    assert not tab.delay_box.isVisibleTo(tab)  # no delay when off
    tab.display_combo.setCurrentIndex(tab.display_combo.findData("each"))
    tab.delay_box.setValue(0.5)
    assert app.manager.settings.run_display == "each"
    assert app.manager.settings.run_delay == 0.5
    assert tab.delay_box.isVisibleTo(tab)
    dialog = app.settings_dialog()
    qtbot.addWidget(dialog)
    assert dialog.run_display_combo.currentData() == "each"
    assert dialog.run_delay_box.value() == 0.5
    assert dialog.run_delay_box.isVisibleTo(dialog.run_delay_box.parentWidget())
    dialog.run_display_combo.setCurrentIndex(dialog.run_display_combo.findData("ask"))
    assert tab.display_combo.currentData() == "ask"  # the panel follows
    # Ask: the delay is for after Continue All, so it shows
    assert dialog.run_delay_box.isVisibleTo(dialog.run_delay_box.parentWidget())
    assert tab.delay_box.isVisibleTo(tab)
    dialog.run_display_combo.setCurrentIndex(dialog.run_display_combo.findData("off"))
    assert not dialog.run_delay_box.isVisibleTo(dialog.run_delay_box.parentWidget())
    assert not tab.delay_box.isVisibleTo(tab)
    assert dialog.run_display_combo.itemText(0) == "Off (fastest)"


def test_continue_all_uses_the_delay(app):
    _display(app, "ask", delay=0.05)
    _answer(app, ["all"])  # after E0: the other 5 shown, the delay apart
    started = time.monotonic()
    app.stats.run_button.click()
    assert time.monotonic() - started >= (N - 1) * 0.05 * 0.9
    assert app.stats.status_label.text() == "Complete"


def test_table_settings_have_their_own_page(qtbot):
    manager = NMManager(quiet=True)
    dialog = SettingsDialog(manager.settings)
    qtbot.addWidget(dialog)
    assert dialog.page_titles()[:3] == ["General", "Plot", "Table"]
    dialog.show_page("Table")
    page = dialog.pages.currentWidget()
    assert all(page.isAncestorOf(button) for button in dialog.nan_buttons.values())
    general = dialog.pages.widget(0)
    assert not any(general.isAncestorOf(button) for button in dialog.nan_buttons.values())


def test_settings_dialog_without_a_window():
    manager = NMManager(quiet=True)
    dialog = SettingsDialog(manager.settings)
    dialog.run_delay_box.setValue(1.5)
    assert manager.settings.run_delay == 1.5


# --- clicking the prompt as a mouse does (through the window) -------------------

@pytest.fixture
def shown_app(app, qtbot):
    app.resize(1300, 760)
    app.show()
    qtbot.waitExposed(app)
    return app


@pytest.mark.parametrize("answers, status", [
    (["continue", "all"], "Complete"),
    (["continue", "cancel"], f"Cancelled after 2 of {N}"),
])
def test_prompt_buttons_take_mouse_clicks(shown_app, click, answers, status):
    tab = shown_app.stats
    _display(shown_app, "ask")
    answered = []

    def shown(_text):
        QtCore.QTimer.singleShot(2000, tab.stop)  # if the click doesn't land: fail, not hang
        if answers:
            answer = answers.pop(0)
            button = tab.run_button if answer == "cancel" else tab.prompt_buttons[answer]
            QtCore.QTimer.singleShot(0, lambda: (answered.append(button.text()), click(button)))
        else:
            QtCore.QTimer.singleShot(0, tab.stop)

    tab.prompt_shown.connect(shown)
    tab.run_button.click()
    assert tab.status_label.text() == status
    assert len(answered) == 2
    assert list(tab.prompt_buttons) == ["continue", "all", "omit"]  # Cancel: the Run button


def test_stop_takes_a_mouse_click_during_a_delay(shown_app, click):
    tab = shown_app.stats
    _display(shown_app, "each", delay=5.0)
    tab.run_trace_shown.connect(
        lambda data, _c: data is not None
        and QtCore.QTimer.singleShot(20, lambda: click(tab.run_button)))
    started = time.monotonic()
    tab.run_button.click()
    assert time.monotonic() - started < 3.0
    assert tab.status_label.text() == f"Cancelled after 1 of {N}"


def test_the_rest_of_the_window_ignores_mouse_clicks(shown_app, click):
    tab = shown_app.stats
    _display(shown_app, "ask")
    on = tab.window_enabled.isChecked()

    def shown(_text):
        QtCore.QTimer.singleShot(2000, tab.stop)  # if Cancel doesn't land: fail, not hang
        click(tab.window_enabled)
        click(shown_app.tool_rail._buttons["Browser"])
        shown_app.close()  # refused during a run
        QtCore.QTimer.singleShot(0, lambda: click(tab.run_button))  # Cancel

    tab.prompt_shown.connect(shown)
    tab.run_button.click()
    assert tab.window_enabled.isChecked() == on  # On didn't change
    assert shown_app.current_tool_name == "Stats"
    assert shown_app.isVisible()
    assert tab.status_label.text() == f"Cancelled after 1 of {N}"
    click(shown_app.tool_rail._buttons["Browser"])  # after the run: works again
    assert shown_app.current_tool_name == "Browser"


def test_details_show_the_runs_result_at_the_prompt(app):
    _display(app, "ask")
    details = []
    app.stats.prompt_shown.connect(
        lambda _text: (details.append(app.stats.details.toPlainText()),
                       QtCore.QTimer.singleShot(0, lambda: app.stats.answer_prompt("continue"))))
    app.stats.run_button.click()
    assert details[2].splitlines() == ["RecordA2  w0 mean  (x 0 to 4)",
                                       "mean = 2"]  # E2's, from the run
    # After the run: E0 again (the Plot's), measured as a preview
    assert app.stats.details.toPlainText().splitlines() == [
        "RecordA0  w0 mean  (x 0 to 4)", "mean = 0"]


# --- Omit the epoch shown (#472) --------------------------------------------------

def test_omit_the_epoch_shown(app):
    _display(app, "ask")
    tab = app.stats
    buttons = []
    tab.prompt_shown.connect(lambda _t: buttons.append(
        (tab.prompt_buttons["omit"].text(), tab.prompt_buttons["omit"].isVisibleTo(tab))))
    _answer(app, ["continue", "omit", "omit", "all"])  # E1 and E2 omitted
    count = len(_commands(app))
    tab.run_button.click()
    assert buttons[:3] == [("Omit E0", True), ("Omit E1", True), ("Omit E2", True)]
    values = tab.stats_tool.toolfolder.data.get("ST_w0_mean_y").nparray
    assert list(values) == [0.0, 3.0, 4.0, 5.0]
    assert tab.status_label.text() == "Complete (omitted E1, E2)"
    assert "Stats: Complete (omitted E1, E2)" in app.history_panel.text()
    runs = [c for c in _commands(app)[count:] if c.startswith("nm.stats.run_selection(")]
    assert len(runs) == 1 and runs[0].endswith("epochs='all', omit=['E1', 'E2'])")


def test_omit_then_cancel(app):
    _display(app, "ask")
    _answer(app, ["omit", "continue", "cancel"])
    app.stats.run_button.click()
    assert app.stats.status_label.text() == f"Cancelled after 3 of {N} (omitted E0)"


def test_omit_takes_a_mouse_click(shown_app, click):
    tab = shown_app.stats
    _display(shown_app, "ask")
    answers = ["omit"]

    def shown(_text):
        QtCore.QTimer.singleShot(2000, tab.stop)  # if the click doesn't land: fail, not hang
        button = tab.prompt_buttons[answers.pop(0)] if answers else tab.prompt_buttons["all"]
        QtCore.QTimer.singleShot(0, lambda: click(button))

    tab.prompt_shown.connect(shown)
    tab.run_button.click()
    assert tab.status_label.text() == "Complete (omitted E0)"



def test_the_prompt_is_in_the_run_row(app):
    """Cancel, Continue, Continue All, Omit, then the count at the right: one
    row, so the panel (and the History below it) doesn't change height."""
    _display(app, "ask")
    tab = app.stats
    seen = []

    def shown(_text):
        layout = tab.run_row.layout()
        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        visible = [w.text() for w in widgets if w is not None and w.isVisibleTo(tab)]
        seen.append((visible, tab.run_info_label.alignment() & QtCore.Qt.AlignmentFlag.AlignRight,
                     tab.sizeHint().height()))
        QtCore.QTimer.singleShot(0, tab.stop)

    height = tab.sizeHint().height()
    tab.prompt_shown.connect(shown)
    tab.run_button.click()
    (visible, right, asking_height), = seen
    assert visible == ["Cancel", "Continue", "Continue All", "Omit E0", f"1 / {N}"]
    assert right
    assert asking_height == height
    assert not any(b.isVisibleTo(tab) for b in tab.prompt_buttons.values())  # after



def test_after_a_run_the_plot_is_back_on_its_channel(qtbot):
    """The run shows its channel's epochs; afterwards the Plot is on the
    channel (and panel) shown before."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=2, n_epochs=2, n_points=5)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    tab = window.tool_tabs["Stats"]
    tab.measurement_combo.setCurrentText("mean")
    tab.set_config_option("results_to_numpy", False)  # no Stats plot after it
    window.view_tabs.select_channel("B")  # looking at B; the run is on A
    window.context_panel.setCurrentWidget(window.table_panel)
    channels = []
    tab.run_trace_shown.connect(
        lambda data, _c: data is not None and channels.append(window.view_tabs.current_view()))
    tab.run_button.click()
    assert channels and all(view == ("channel", "A") for view in channels)
    assert window.view_tabs.current_view() == ("channel", "B")
    assert window.context_panel.currentWidget() is window.table_panel
    assert window.plot_widget.current_trace is series.get_data("B", "E0")


def test_the_choice_is_labelled_progress(app, qtbot):
    assert app.stats.display_label.text() == "Progress"
    dialog = app.settings_dialog()
    qtbot.addWidget(dialog)
    form = dialog.run_display_combo.parentWidget().layout()
    assert form.labelForField(dialog.run_display_combo).text() == "Progress"
