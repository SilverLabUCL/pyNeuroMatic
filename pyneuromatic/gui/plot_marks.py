"""A tool's measurements drawn over the current trace (#463).

A tool tab reports what to draw for the trace shown
(:meth:`ToolTabWidget.plot_marks`), e.g. the level Stats measured across its
window, or the point where it found the max. The Plot shows only the marks
of the tool chosen in the tool rail, and draws them again when the trace or
the tool's settings change. Drawing them never writes a command or changes
data: it's a preview of what Run would measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PyQt6 import QtCore

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None


@dataclass
class PlotMark:
    """A line through *x*, *y* ("line"), or markers at them ("points")."""

    kind: str
    x: list[float]
    y: list[float]
    color: str
    dashed: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("line", "points"):
            raise ValueError(f"kind must be 'line' or 'points', not {self.kind!r}")


class MarkOverlay(QtCore.QObject):
    """Draws a tool's marks over a pyqtgraph plot."""

    def __init__(self, plot_widget: Any, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._plot_widget = plot_widget
        self._source: Any = None
        self._data: Any = None
        self._visible = True
        self.items: list[Any] = []  # the plot items drawn
        self.marks: list[PlotMark] = []  # what they show

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self.redraw()

    def set_source(self, tool: Any) -> None:
        """Show the marks of *tool* (a tool tab), or none."""
        if tool is self._source:
            return
        if self._source is not None:
            self._source.marks_changed.disconnect(self.redraw)
        self._source = tool
        if tool is not None:
            tool.marks_changed.connect(self.redraw)
        self.redraw()

    def set_data(self, data: Any) -> None:
        """The trace shown (as plotted: transformed, if the channel is)."""
        self._data = data

    def redraw(self) -> None:
        for item in self.items:
            if item.scene() is not None:
                self._plot_widget.removeItem(item)
        self.items, self.marks = [], []
        if (self._plot_widget is None or pg is None or not self._visible
                or self._source is None or self._data is None):
            return
        try:
            marks = list(self._source.plot_marks(self._data))
        except Exception:  # a preview: a measurement that fails shows nothing
            marks = []
        for mark in marks:
            # A solid level stands out over a noisy trace; a dashed one is a guide
            pen = pg.mkPen(mark.color, width=1.5 if mark.dashed else 2.5,
                           style=QtCore.Qt.PenStyle.DashLine if mark.dashed
                           else QtCore.Qt.PenStyle.SolidLine)
            if mark.kind == "points":
                item = pg.PlotDataItem(mark.x, mark.y, pen=None, symbol="o", symbolSize=9,
                                       symbolPen=pg.mkPen(mark.color, width=1.5),
                                       symbolBrush=None)
            else:
                item = pg.PlotDataItem(mark.x, mark.y, pen=pen)
            item.setZValue(10)  # over the traces
            self._plot_widget.addItem(item, ignoreBounds=True)
            self.items.append(item)
            self.marks.append(mark)


__all__ = ["MarkOverlay", "PlotMark"]
