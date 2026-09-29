"""Reusable base widgets for analysis-tool tabs."""
from __future__ import annotations

import json
from typing import Any

from PyQt6 import QtCore, QtWidgets


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
            text = json.dumps(result, indent=2, default=str)
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
    """

    run_finished = QtCore.pyqtSignal(bool)
    run_failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        title: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._running = False
        self.warnings: list[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.title_label = QtWidgets.QLabel(title, self)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.parameters_group = QtWidgets.QGroupBox("Parameters", self)
        self.parameter_layout = QtWidgets.QVBoxLayout(self.parameters_group)
        layout.addWidget(self.parameters_group)

        self.preview_group = QtWidgets.QGroupBox("Preview", self)
        self.preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_group.hide()
        layout.addWidget(self.preview_group)

        self.run_button = QtWidgets.QPushButton("Run", self)
        self.run_button.clicked.connect(self._execute)
        layout.addWidget(self.run_button, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        results_group = QtWidgets.QGroupBox("Results", self)
        results_layout = QtWidgets.QVBoxLayout(results_group)
        self.results_panel = ResultPanel(results_group)
        results_layout.addWidget(self.results_panel)
        layout.addWidget(results_group, stretch=1)

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready", self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.hide()
        status_layout.addWidget(self.status_label, stretch=1)
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout)

    def run_tool(self) -> Any:
        """Run this tool and return a value for the shared results panel."""
        raise NotImplementedError("ToolTabWidget subclasses must implement run_tool()")

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
        self.warnings.clear()
        self.results_panel.clear()
        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.set_status("Running")
        succeeded = False
        try:
            result = self.run_tool()
            if result is not None:
                self.results_panel.set_result(result)
            self.set_status("Complete with warnings" if self.warnings else "Complete")
            succeeded = True
        except Exception as error:
            self.set_status(f"Failed: {error}")
            self.run_failed.emit(str(error))
        finally:
            self._running = False
            self.run_button.setEnabled(True)
            self.progress_bar.hide()
            self.run_finished.emit(succeeded)


__all__ = ["ResultPanel", "ToolTabWidget"]