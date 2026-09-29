"""Tests for the reusable analysis-tool tab framework."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.gui.tool_tab import ToolTabWidget

pytestmark = pytest.mark.gui


class _WorkingToolTab(ToolTabWidget):
    def __init__(self):
        super().__init__("Stats")
        self.run_count = 0
        self.parameter = QtWidgets.QLineEdit("mean")
        self.parameter_layout.addWidget(self.parameter)

    def run_tool(self):
        self.run_count += 1
        return {"function": self.parameter.text(), "value": 12.5}


class _FailingToolTab(ToolTabWidget):
    def __init__(self):
        super().__init__("Failing")

    def run_tool(self):
        raise ValueError("no data selected")


class _WarningToolTab(ToolTabWidget):
    def __init__(self):
        super().__init__("Warning")

    def run_tool(self):
        self.show_warning("some values were skipped")
        self.set_progress(1, maximum=1)
        return "finished"


def test_tool_tab_runs_subclass_and_displays_results(qtbot):
    tab = _WorkingToolTab()
    qtbot.addWidget(tab)
    finished = []
    tab.run_finished.connect(finished.append)

    tab.run_button.click()

    assert tab.run_count == 1
    assert '"function": "mean"' in tab.results_panel.output.toPlainText()
    assert '"value": 12.5' in tab.results_panel.output.toPlainText()
    assert tab.status_label.text() == "Complete"
    assert finished == [True]
    assert tab.run_button.isEnabled()
    assert not tab.progress_bar.isVisible()


def test_tool_tab_reports_run_errors(qtbot):
    tab = _FailingToolTab()
    qtbot.addWidget(tab)
    failures = []
    finished = []
    tab.run_failed.connect(failures.append)
    tab.run_finished.connect(finished.append)

    tab.run_button.click()

    assert failures == ["no data selected"]
    assert finished == [False]
    assert tab.status_label.text() == "Failed: no data selected"
    assert tab.run_button.isEnabled()


def test_tool_tab_supports_preview_warnings_and_progress(qtbot):
    tab = _WarningToolTab()
    qtbot.addWidget(tab)
    preview = QtWidgets.QLabel("Preview")

    tab.set_preview_widget(preview)
    assert not tab.preview_group.isHidden()

    tab.run_button.click()

    assert tab.warnings == ["some values were skipped"]
    assert tab.status_label.text() == "Complete with warnings"
    assert tab.results_panel.output.toPlainText() == "finished"
    assert tab.progress_bar.isHidden()

    tab.set_preview_widget(None)
    assert tab.preview_group.isHidden()