"""Reusable base widgets for analysis-tool tabs."""
from __future__ import annotations

import json
import time
from typing import Any

from PyQt6 import QtCore, QtWidgets

# The Display control (tool panel) and its choices, as in Settings (#468)
RUN_DISPLAY_LABELS = {"off": "Off", "each": "Each epoch", "ask": "Ask after each"}


class _RunGuard(QtCore.QObject):
    """While a tool runs: the window ignores clicks, keys and closing, except
    on Cancel, the run's prompt and the Plot (to zoom an epoch shown, #468),
    and menus opened from it. Esc stops the run (#450); Return answers the
    prompt with Continue."""

    _BLOCKED = {
        QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease,
        QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QEvent.Type.KeyPress,
        QtCore.QEvent.Type.KeyRelease, QtCore.QEvent.Type.Wheel,
        QtCore.QEvent.Type.ContextMenu, QtCore.QEvent.Type.Shortcut,
        QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.Close,
    }

    def __init__(self, tab: "ToolTabWidget") -> None:
        super().__init__(tab)
        self._tab = tab

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        kind = event.type()
        if kind not in self._BLOCKED:
            return False
        if kind == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self._tab.stop()
                return True
            if (event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
                    and self._tab.asking):
                self._tab.answer_prompt("continue")
                return True
        if kind in (QtCore.QEvent.Type.Shortcut, QtCore.QEvent.Type.ShortcutOverride):
            return True  # e.g. Quit: they go to actions, not widgets
        if kind == QtCore.QEvent.Type.Close:
            if obj is self._tab.window():
                event.ignore()  # refused, not just not delivered
                return True
            return False
        if not isinstance(obj, QtWidgets.QWidget):
            # The window's QWindow: it passes the click (or key) on to the
            # widget under it, which is checked then
            return False
        if isinstance(obj, QtWidgets.QMenu):
            return False  # e.g. the Plot's right-click menu
        for allowed in self._tab.allowed_during_run():
            if obj is allowed or allowed.isAncestorOf(obj):
                return False
        return True


class ResultPanel(QtWidgets.QWidget):
    """Read-only, reusable display area for tool results."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.output = QtWidgets.QPlainTextEdit(self)
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    def set_result(self, result: Any) -> None:
        if isinstance(result, (dict, list, tuple)):
            text = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        else:
            text = "" if result is None else str(result)
        self.output.setPlainText(text)

    def clear(self) -> None:
        self.output.clear()


class ToolTabWidget(QtWidgets.QWidget):
    """Shared title, parameter, execution, status, and result layout.

    Subclasses populate :attr:`parameter_layout` and implement
    :meth:`run_tool`. The method's return value is displayed in the standard
    results panel; subclasses can also update status and progress while work
    is running.

    What a tool shows in the app window is optional, and the app window
    registers every tool tab the same way (#454):

    - :attr:`plot_view`, :attr:`table_view`: its pages in the Plot and Table
      panels, shown as a tab there while :meth:`has_results` is True
    - :meth:`config_options` / :meth:`set_config_option`: its page in Settings
    - :meth:`refresh`, :meth:`apply_settings`, :meth:`close_views`: called when
      the data, the general settings or the window change
    - :meth:`plot_ranges` / :meth:`set_plot_range`: its X ranges, drawn on the
      Plot with draggable edges while it is the tool chosen (#436)
    - :meth:`plot_marks`: its measurements of the trace shown, drawn over it
      while it is the tool chosen (#463)
    - :attr:`results_changed`, :attr:`view_requested`, :attr:`ranges_changed`,
      :attr:`marks_changed`: signals to the window
    """

    run_finished = QtCore.pyqtSignal(bool)
    run_failed = QtCore.pyqtSignal(str)
    selection_updated = QtCore.pyqtSignal(dict)
    # The ⋯ button: open Settings at this tool's page (its title)
    settings_requested = QtCore.pyqtSignal(str)
    # A run, for the History panel: (kind, text), e.g. ("tool", "Stats: Complete")
    history_entry = QtCore.pyqtSignal(str, str)
    # Its results changed (a run, a deleted subfolder): its tabs and the Subfolder list
    results_changed = QtCore.pyqtSignal()
    # Show its view: "plot" or "table"
    view_requested = QtCore.pyqtSignal(str)
    # A run's progress, as shown: (done, total)
    progress_changed = QtCore.pyqtSignal(int, int)
    # Display while running: show this epoch's data (and its channel's name)
    # in the Plot; None when the run ends (#468)
    run_trace_shown = QtCore.pyqtSignal(object, str)
    # Its Display or delay changed (the settings are shared by all tools)
    run_display_changed = QtCore.pyqtSignal()
    # The prompt is up, waiting for an answer (its text)
    prompt_shown = QtCore.pyqtSignal(str)

    # Run in green; Cancel (while running) in red, the app's error red
    RUN_COLOR = "#15803d"
    CANCEL_COLOR = "#c62828"

    # Progress is shown once a run takes longer than this (s), so quick runs
    # don't flash, and updated at most this often (s)
    SHOW_PROGRESS_AFTER = 0.5
    PROGRESS_INTERVAL = 0.1
    # Its X ranges changed (e.g. a typed X begin): the Plot moves them
    ranges_changed = QtCore.pyqtSignal()
    # What it would measure changed (e.g. its function): the Plot redraws the marks
    marks_changed = QtCore.pyqtSignal()

    # Its pages in the Plot and Table panels, if any
    plot_view: QtWidgets.QWidget | None = None
    table_view: QtWidgets.QWidget | None = None

    def __init__(
        self,
        title: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._running = False
        self._stop_requested = False
        self._settings: Any = None  # run_display and run_delay (set_settings)
        self._run_allowed: list[QtWidgets.QWidget] = []
        self._wait_loop: QtCore.QEventLoop | None = None
        self._prompt_answer = ""
        self._asking = False
        self.warnings: list[str] = []
        self._selection_model: Any = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # The title, then ⋯ for the tool's settings: the same place on every tool
        title_row = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel(title, self)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_row.addWidget(self.title_label)
        self.settings_button = QtWidgets.QToolButton(self)
        self.settings_button.setText("⋯")
        self.settings_button.setToolTip(f"{title} settings (opens Settings at {title})")
        self.settings_button.clicked.connect(lambda: self.settings_requested.emit(title))
        title_row.addWidget(self.settings_button)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.parameters_group = QtWidgets.QGroupBox("Parameters", self)
        self.parameter_layout = QtWidgets.QVBoxLayout(self.parameters_group)
        layout.addWidget(self.parameters_group)

        self.preview_group = QtWidgets.QGroupBox("Preview", self)
        self.preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_group.hide()
        layout.addWidget(self.preview_group)

        # Run button with a label subclasses can use, e.g. for what will run
        self.run_row = QtWidgets.QWidget(self)
        run_layout = QtWidgets.QHBoxLayout(self.run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        self.run_button = QtWidgets.QPushButton("Run", self.run_row)
        self.run_button.setToolTip("Run (while running: Cancel, or Esc)")
        self._show_run_button(running=False)
        self.run_button.clicked.connect(self._on_run_clicked)
        run_layout.addWidget(self.run_button)
        self.run_info_label = QtWidgets.QLabel("", self.run_row)
        run_layout.addWidget(self.run_info_label, stretch=1)
        layout.addWidget(self.run_row)

        # Display while running (#468): shared by all tools, saved in Settings;
        # a row of its own, so the panel is no wider
        self.display_row = QtWidgets.QWidget(self)
        display_layout = QtWidgets.QHBoxLayout(self.display_row)
        display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_label = QtWidgets.QLabel("Display", self.display_row)
        display_layout.addWidget(self.display_label)
        self.display_combo = QtWidgets.QComboBox(self.display_row)
        for key, label in RUN_DISPLAY_LABELS.items():
            self.display_combo.addItem(label, key)
        self.display_combo.setToolTip(
            "While running, show each epoch in the Plot, then continue after "
            "a delay, or ask to continue (Off: fastest)")
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        display_layout.addWidget(self.display_combo)
        self.delay_box = QtWidgets.QDoubleSpinBox(self.display_row)
        self.delay_box.setRange(0.0, 10.0)
        self.delay_box.setSingleStep(0.1)
        self.delay_box.setDecimals(1)
        self.delay_box.setSuffix(" s")
        self.delay_box.setToolTip(
            "Wait after showing each epoch")
        self.delay_box.valueChanged.connect(self._on_display_changed)
        display_layout.addWidget(self.delay_box)
        display_layout.addStretch(1)
        self.display_row.hide()  # until set_settings
        layout.addWidget(self.display_row)

        # Ask after each epoch: Continue or Continue All; Cancel is the Run
        # button (Cancel while running) or Esc (#468)
        self.prompt_bar = QtWidgets.QFrame(self)
        self.prompt_bar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        prompt_layout = QtWidgets.QHBoxLayout(self.prompt_bar)
        prompt_layout.setContentsMargins(6, 4, 6, 4)
        self.prompt_label = QtWidgets.QLabel("", self.prompt_bar)
        self.prompt_label.setWordWrap(True)
        prompt_layout.addWidget(self.prompt_label, stretch=1)
        self.prompt_buttons: dict[str, QtWidgets.QPushButton] = {}
        for answer, text, tip in (
                ("continue", "Continue", "Analyse the next epoch (Return)"),
                ("all", "Continue All", "Finish the run without asking again")):
            button = QtWidgets.QPushButton(text, self.prompt_bar)
            button.setAutoDefault(False)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, a=answer: self.answer_prompt(a))
            prompt_layout.addWidget(button)
            self.prompt_buttons[answer] = button
        self.prompt_bar.hide()
        layout.addWidget(self.prompt_bar)

        self.results_group = QtWidgets.QGroupBox("Results", self)
        self.results_layout = QtWidgets.QVBoxLayout(self.results_group)
        self.results_panel = ResultPanel(self.results_group)
        self.results_layout.addWidget(self.results_panel)
        layout.addWidget(self.results_group, stretch=1)

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready", self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.hide()
        status_layout.addWidget(self.status_label, stretch=1)
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout)

    # --- progress and Cancel (#450) ---------------------------------------------

    def analysis_tool(self) -> Any:
        """The NMTool a run uses, which reports its progress (None: none)."""
        return None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def stop(self) -> None:
        """Cancel the run after the epoch (or data array) it's on. What it has
        done is kept."""
        if self._running:
            self._stop_requested = True
            self.set_status("Cancelling")
            if self._wait_loop is not None:  # waiting: the delay or the prompt
                self._wait_loop.quit()

    # --- display while running (#468) -----------------------------------------

    def set_settings(self, settings: Any) -> None:
        """The settings with run_display and run_delay (shared by all tools):
        shows the Display control."""
        self._settings = settings
        self.display_row.show()
        self._show_run_display()

    def allow_during_run(self, widget: QtWidgets.QWidget) -> None:
        """*widget* (e.g. the Plot) still takes clicks and keys during a run."""
        self._run_allowed.append(widget)

    def allowed_during_run(self) -> list[QtWidgets.QWidget]:
        return [self.run_button, self.prompt_bar, *self._run_allowed]

    def run_display(self) -> str:
        """"off", "each" or "ask" (Settings: run_display)."""
        return getattr(self._settings, "run_display", "off")

    def _show_run_display(self) -> None:
        """The Display control and delay show the settings."""
        if self._settings is None:
            return
        for widget in (self.display_combo, self.delay_box):
            widget.blockSignals(True)
        self.display_combo.setCurrentIndex(self.display_combo.findData(self.run_display()))
        self.delay_box.setValue(self._settings.run_delay)
        for widget in (self.display_combo, self.delay_box):
            widget.blockSignals(False)
        # The delay is for "each", and "ask" after Continue All
        self.delay_box.setVisible(self.run_display() != "off")

    def _on_display_changed(self, *_args) -> None:
        if self._settings is None:
            return
        self._settings.run_display = self.display_combo.currentData()
        self._settings.run_delay = float(self.delay_box.value())
        self._settings.save_user()  # only if loaded from the user's file
        self.delay_box.setVisible(self.run_display() != "off")
        self.run_display_changed.emit()

    def set_trace(self, data) -> None:
        """The trace the Plot shows (as plotted; None for none), e.g. to
        describe its measurement (Stats' Results, #471)."""

    @property
    def asking(self) -> bool:
        """True while the prompt waits for an answer."""
        return self._asking

    def answer_prompt(self, answer: str) -> None:
        """The prompt's answer: "continue", "all" (Continue All) or "cancel"."""
        self._prompt_answer = answer
        if answer == "cancel":
            self.stop()
        elif self._wait_loop is not None:
            self._wait_loop.quit()

    def _wait(self, seconds: float) -> None:
        """Wait, with the window responding (and Cancel working)."""
        self._wait_loop = QtCore.QEventLoop(self)
        QtCore.QTimer.singleShot(int(seconds * 1000), self._wait_loop.quit)
        self._wait_loop.exec()
        self._wait_loop = None

    def _ask(self, name: str, done: int, total: int) -> str:
        """Ask after an epoch: "continue", "all" or "cancel"."""
        # Its result is in the tool's own display (e.g. Stats' Results, #471)
        self.prompt_label.setText(f"{name} ({done} of {total})")
        self._prompt_answer = ""
        self._asking = True
        self.prompt_bar.show()
        self.prompt_buttons["continue"].setFocus()
        self._wait_loop = QtCore.QEventLoop(self)
        self.prompt_shown.emit(self.prompt_label.text())
        if not self._stop_requested and not self._prompt_answer:  # not answered yet
            self._wait_loop.exec()
        self._wait_loop = None
        self._asking = False
        self.prompt_bar.hide()
        return "cancel" if self._stop_requested else (self._prompt_answer or "continue")

    def _display_step(self, done: int, total: int) -> None:
        """Show the epoch just analysed, then wait (the delay) or ask."""
        tool = self.analysis_tool()
        data = getattr(tool, "data", None)
        channel = getattr(getattr(tool, "channel", None), "name", "") or ""
        self.run_trace_shown.emit(data, channel)
        QtWidgets.QApplication.processEvents()  # draw it
        if self.run_display() == "ask" and not self._continue_all:
            epoch = getattr(tool, "epoch", None)
            name = getattr(epoch, "name", None) or getattr(data, "name", "")
            answer = self._ask(name, done, total)
            if answer == "all":
                self._continue_all = True  # as "each", with its delay
        elif self._settings is not None and self._settings.run_delay > 0:
            self._wait(self._settings.run_delay)

    def set_run_total(self, total: int) -> None:
        """A run made of several tool runs (e.g. All Sets): their targets in
        all, so progress counts them together."""
        self._run_total = total

    def _show_run_button(self, running: bool) -> None:
        """Run (green), or Cancel (red) while running."""
        self.run_button.setText("Cancel" if running else "Run")
        color = self.CANCEL_COLOR if running else self.RUN_COLOR
        self.run_button.setStyleSheet(f"QPushButton {{ color: {color}; }}")

    def _on_run_clicked(self) -> None:
        if self._running:
            self.stop()
        else:
            self._execute()

    def _run_progress(self, done: int, total: int) -> bool:
        """The tool's progress callback: shows how far the run is (once it's
        long enough to see), lets the window handle Cancel, and says whether
        to go on."""
        if self._stop_requested:
            return False
        overall = self._progress_offset + done
        overall_total = self._run_total or total
        if done >= total:  # this tool run is done; the next one adds on
            self._progress_offset += total
        self._progress_done = (overall, overall_total)
        now = time.monotonic()
        displayed = self.run_display() != "off"
        if displayed or (now - self._run_started >= self.SHOW_PROGRESS_AFTER
                         and (now - self._progress_shown >= self.PROGRESS_INTERVAL
                              or overall >= overall_total)):
            self._progress_shown = now
            self.progress_bar.setRange(0, overall_total)
            self.progress_bar.setValue(overall)
            self.progress_bar.show()
            self.run_info_label.setText(f"{overall} / {overall_total}")
            self.progress_changed.emit(overall, overall_total)
            QtWidgets.QApplication.processEvents()  # repaint, and Cancel
        if displayed and not self._stop_requested:
            self._display_step(overall, overall_total)
        return not self._stop_requested

    def run_tool(self) -> Any:
        """Run this tool and return a value for the shared results panel."""
        raise NotImplementedError("ToolTabWidget subclasses must implement run_tool()")

    def run_summary(self) -> str:
        """A finished run, for the History panel (e.g. where results went)."""
        return self.status_label.text()

    # --- what the app window asks of a tool (all optional) --------------------

    def has_results(self) -> bool:
        """Whether its views have something to show (their tabs appear)."""
        return False

    def config_options(self) -> list[tuple[str, str, bool, str]]:
        """(key, label, value, tip) of each option in Settings; none: no page."""
        return []

    def set_config_option(self, key: str, value: bool) -> None:
        """Set an option from its Settings page."""
        raise KeyError(f"{self.title_label.text()} has no option {key!r}")

    def plot_marks(self, data) -> list:
        """Its measurements of *data* (the trace shown) to draw over it
        (:class:`~pyneuromatic.gui.plot_marks.PlotMark`); never writes a
        command or changes data."""
        return []

    def plot_ranges(self) -> list:
        """Its X ranges to draw on the Plot (:class:`~pyneuromatic.gui.plot_ranges.PlotRange`)."""
        return []

    def set_plot_range(self, key: str, xbgn: float, xend: float, final: bool) -> None:
        """Range *key* was dragged to *xbgn*, *xend*: shown while dragging,
        saved when released (*final*)."""

    def refresh(self) -> None:
        """Folders, data or sets changed."""

    def apply_settings(self) -> None:
        """A general setting changed (e.g. how tables show NaN, or the run
        display)."""
        self._show_run_display()

    def close_views(self) -> None:
        """The window is closing: tear its pyqtgraph plot down before Qt
        deletes it, so a queued paint event never reaches a half-deleted plot."""
        view = getattr(self.plot_view, "plot_widget", None)
        if view is not None and not getattr(view, "closed", False):
            view.close()

    def _log_run(self, succeeded: bool) -> None:
        """The run in the History panel: one entry, then its warnings."""
        title = self.title_label.text()
        if succeeded:
            self.history_entry.emit("tool", f"{title}: {self.run_summary()}")
            for warning in self.warnings:
                self.history_entry.emit("warning", f"{title}: {warning}")
        else:
            status = self.status_label.text()
            cancelled = status.startswith("Failed: Cancelled")
            self.history_entry.emit("message" if cancelled else "error", f"{title}: {status}")

    @property
    def selection(self) -> dict[str, Any]:
        """Current shared selection, or an empty mapping before binding."""
        if self._selection_model is None:
            return {}
        return self._selection_model.selection

    def bind_selection_model(self, model: Any) -> None:
        if self._selection_model is model:
            return
        if self._selection_model is not None:
            self._selection_model.selection_changed.disconnect(self._on_selection_changed)
        self._selection_model = model
        model.selection_changed.connect(self._on_selection_changed)
        self._on_selection_changed(model.selection)

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        self.selection_updated.emit(selection)

    def set_preview_widget(self, widget: QtWidgets.QWidget | None) -> None:
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            old_widget = item.widget()
            if old_widget is not None:
                old_widget.setParent(None)
        if widget is not None:
            self.preview_layout.addWidget(widget)
        self.preview_group.setVisible(widget is not None)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def show_warning(self, message: str) -> None:
        self.warnings.append(message)
        self.set_status(f"Warning: {message}")

    def set_progress(self, value: int | None = None, maximum: int = 100) -> None:
        if value is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(value)
        self.progress_bar.show()

    def _execute(self) -> None:
        if self._running:
            return

        self._running = True
        self._stop_requested = False
        self._continue_all = False
        self._run_total = 0
        self._progress_offset = 0
        self._progress_done: tuple[int, int] | None = None
        self._run_started = time.monotonic()
        self._progress_shown = 0.0
        info = self.run_info_label.text()
        self.warnings.clear()
        self.results_panel.clear()
        self._show_run_button(running=True)
        self.set_status("Running")
        tool = self.analysis_tool()
        if tool is not None:
            tool.progress = self._run_progress
        guard = _RunGuard(self)
        QtWidgets.QApplication.instance().installEventFilter(guard)
        succeeded = False
        try:
            result = self.run_tool()
            if result is not None:
                self.results_panel.set_result(result)
            if self._stop_requested and self._progress_done is not None:
                self.set_status("Cancelled after %d of %d" % self._progress_done)
            else:
                self.set_status("Complete with warnings" if self.warnings else "Complete")
            succeeded = True
        except Exception as error:
            self.set_status(f"Failed: {error}")
            self.run_failed.emit(str(error))
        finally:
            QtWidgets.QApplication.instance().removeEventFilter(guard)
            guard.deleteLater()
            if tool is not None:
                tool.progress = None
            self._running = False
            self._stop_requested = False
            self._show_run_button(running=False)
            self.run_info_label.setText(info)
            self.progress_bar.hide()
            self.prompt_bar.hide()
            if self.run_display() != "off":
                self.run_trace_shown.emit(None, "")  # the Plot as before the run
            self._log_run(succeeded)
            self.run_finished.emit(succeeded)


__all__ = ["ResultPanel", "ToolTabWidget"]