"""A tool's Y levels on the Plot: horizontal lines that can be dragged (#527).

A tool tab reports its levels for the trace shown
(:meth:`ToolTabWidget.plot_levels`), e.g. a Stats window's level for its
level measures, or Spike's threshold, and takes dragged values back
(:meth:`ToolTabWidget.set_plot_level`): shown in its field while dragging,
saved (one command) when released. The Plot shows only the levels of the
tool chosen in the tool rail, draggable, locked or hidden with its X ranges
(Plot ⋯ › Tool Ranges & Levels).

Dragged levels are rounded to a step that suits the Y axis shown (4
significant digits of its span), so the fields don't fill with noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PyQt6 import QtCore, QtGui

from pyneuromatic.gui.plot_ranges import MODES

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None


@dataclass
class PlotLevel:
    """One Y level of a tool, e.g. ("level", "w0 Y level", -40.0, "#2563eb")."""

    key: str
    label: str
    y: float
    color: str


def round_level(value: float, span: float) -> float:
    """*value* rounded to 4 significant digits of *span* (the Y axis shown),
    e.g. -41.2367 in a span of 80 -> -41.24."""
    if not math.isfinite(value) or not math.isfinite(span) or span <= 0:
        return value
    step = 10.0 ** (math.floor(math.log10(span)) - 3)
    return float(f"{round(value / step) * step:.12g}")  # no float noise


class LevelOverlay(QtCore.QObject):
    """Draws a tool's levels over a pyqtgraph plot and sends drags back."""

    # As the X ranges' edges: easy to grab, wider under the pointer
    LINE_WIDTH = 2
    LINE_HOVER_WIDTH = 4

    def __init__(self, plot_widget: Any, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._plot_widget = plot_widget
        self._source: Any = None
        self._data: Any = None
        self._mode = "on"
        # key -> (line item, its level)
        self._items: dict[str, tuple[Any, PlotLevel]] = {}

    @property
    def lines(self) -> dict[str, Any]:
        """The line items drawn, by level key."""
        return {key: item[0] for key, item in self._items.items()}

    @property
    def levels(self) -> list[PlotLevel]:
        """The levels drawn."""
        return [item[1] for item in self._items.values()]

    def set_source(self, tool: Any) -> None:
        """Show the levels of *tool* (a tool tab), or none."""
        if tool is self._source:
            return
        if self._source is not None:
            self._source.levels_changed.disconnect(self.update)
        self._source = tool
        if tool is not None:
            tool.levels_changed.connect(self.update)
        self.redraw()

    def set_data(self, data: Any) -> None:
        """The trace shown (as plotted: transformed, if the channel is)."""
        self._data = data

    @property
    def mode(self) -> str:
        """"on" (draggable), "frozen" (shown, locked) or "off" (hidden)."""
        return self._mode

    @property
    def _movable(self) -> bool:
        return self._mode == "on"

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self._mode = mode
        self.redraw()

    def redraw(self) -> None:
        """Draw the levels again (e.g. after the plot was cleared)."""
        for line, _level in self._items.values():
            if line.scene() is not None:
                self._plot_widget.removeItem(line)
        self._items = {}
        if self._plot_widget is None or pg is None or self._mode == "off":
            return
        for level in self._levels():
            line = pg.InfiniteLine(angle=0, movable=self._movable)
            line.setZValue(5)  # over the traces, under the marks
            self._plot_widget.addItem(line, ignoreBounds=True)
            self._items[level.key] = (line, level)
            self._show(level)
            line.sigPositionChanged.connect(
                lambda _l, key=level.key: self._on_dragged(key, final=False))
            line.sigPositionChangeFinished.connect(
                lambda _l, key=level.key: self._on_dragged(key, final=True))

    def update(self) -> None:
        """The tool's levels changed (e.g. a typed value): move the lines."""
        levels = self._levels()
        if [level.key for level in levels] != list(self._items):
            self.redraw()
            return
        for level in levels:
            self._show(level)

    def _levels(self) -> list[PlotLevel]:
        if self._source is None or self._data is None:
            return []
        try:
            levels = list(self._source.plot_levels(self._data))
        except Exception:  # e.g. a baseline that can't be measured: none shown
            return []
        return [level for level in levels if math.isfinite(level.y)]

    def _show(self, level: PlotLevel) -> None:
        line = self._items[level.key][0]
        color = QtGui.QColor(level.color)
        line.blockSignals(True)
        line.setValue(level.y)
        line.setPen(pg.mkPen(color, width=self.LINE_WIDTH))
        line.setHoverPen(pg.mkPen(color, width=self.LINE_HOVER_WIDTH))
        line.setToolTip(f"{level.label}: drag to change" if self._movable else level.label)
        line.blockSignals(False)
        self._items[level.key] = (line, level)

    def _span(self) -> float:
        """The Y axis shown: its span."""
        try:
            low, high = self._plot_widget.getPlotItem().getViewBox().viewRange()[1]
        except AttributeError:
            return math.nan
        return float(high - low)

    def _on_dragged(self, key: str, final: bool) -> None:
        if self._source is None or key not in self._items:
            return
        line = self._items[key][0]
        y = round_level(float(line.value()), self._span())
        self._source.set_plot_level(key, y, final)
        if final:
            self.update()  # rounded, and as the tool saved it


__all__ = ["LevelOverlay", "PlotLevel", "round_level"]
