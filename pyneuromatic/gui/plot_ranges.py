"""A tool's X ranges on the Plot: shaded, with draggable edges (#436).

A tool tab reports its ranges (:meth:`ToolTabWidget.plot_ranges`), e.g. a
Stats window's X begin / X end and its baseline range, and takes dragged
values back (:meth:`ToolTabWidget.set_plot_range`). The Plot shows only the
ranges of the tool chosen in the tool rail.

The ranges are on (draggable), frozen (shown, locked) or off (hidden).

An unbounded edge (-inf or inf) is drawn dashed at the trace's start or
end, and stays unbounded unless it is dragged. Dragged edges snap to the
nearest sample point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI dependency
    pg = None


@dataclass
class PlotRange:
    """One X range of a tool, e.g. ("window", "w0", 10.0, 20.0, "#2563eb")."""

    key: str
    label: str
    xbgn: float
    xend: float
    color: str


def snap(x_values: np.ndarray, value: float) -> float:
    """The sample x in *x_values* (ascending) nearest *value*."""
    if x_values.size == 0:
        return value
    index = int(np.searchsorted(x_values, value))
    neighbours = x_values[max(index - 1, 0):index + 1]
    nearest = float(neighbours[np.argmin(np.abs(neighbours - value))])
    return float(f"{nearest:.12g}")  # no float noise in the fields


# How the ranges are shown: draggable, shown but locked, or hidden
MODES = ("on", "frozen", "off")


class RangeOverlay(QtCore.QObject):
    """Draws a tool's ranges over a pyqtgraph plot and sends drags back."""

    BRUSH_ALPHA = 40
    # Edges: wide enough to grab easily; wider under the pointer, which also
    # widens where they can be grabbed (pyqtgraph uses the wider pen)
    EDGE_WIDTH = 2
    EDGE_HOVER_WIDTH = 4

    def __init__(self, plot_widget: Any, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._plot_widget = plot_widget
        self._source: Any = None
        self._x = np.array([])
        self._mode = "on"
        # key -> (region item, its range, the edges as drawn)
        self._items: dict[str, tuple[Any, PlotRange, tuple[float, float]]] = {}

    @property
    def regions(self) -> dict[str, Any]:
        """The region items drawn, by range key."""
        return {key: item[0] for key, item in self._items.items()}

    def set_source(self, tool: Any) -> None:
        """Show the ranges of *tool* (a tool tab), or none."""
        if tool is self._source:
            return
        if self._source is not None:
            self._source.ranges_changed.disconnect(self.update)
        self._source = tool
        if tool is not None:
            tool.ranges_changed.connect(self.update)
        self.redraw()

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

    def set_x(self, x_values) -> None:
        """The current trace's x values (where unbounded edges sit, and what
        edges snap to)."""
        x = np.asarray(x_values, dtype=float) if x_values is not None else np.array([])
        self._x = x[np.isfinite(x)]

    def redraw(self) -> None:
        """Draw the ranges again (e.g. after the plot was cleared)."""
        for region, _range, _drawn in self._items.values():
            if region.scene() is not None:
                self._plot_widget.removeItem(region)
        self._items = {}
        if self._plot_widget is None or pg is None or self._x.size == 0 or self._mode == "off":
            return
        for plot_range in self._ranges():
            region = pg.LinearRegionItem(swapMode="block", movable=self._movable)
            region.setZValue(-10)  # behind the traces
            self._plot_widget.addItem(region, ignoreBounds=True)
            self._items[plot_range.key] = (region, plot_range, (0.0, 0.0))
            self._show(plot_range)
            region.sigRegionChanged.connect(
                lambda _r, key=plot_range.key: self._on_dragged(key, final=False))
            region.sigRegionChangeFinished.connect(
                lambda _r, key=plot_range.key: self._on_dragged(key, final=True))

    def update(self) -> None:
        """The tool's ranges changed (e.g. typed values): move the edges."""
        ranges = self._ranges()
        if [r.key for r in ranges] != list(self._items):
            self.redraw()
            return
        for plot_range in ranges:
            self._show(plot_range)

    def _ranges(self) -> list[PlotRange]:
        return list(self._source.plot_ranges()) if self._source is not None else []

    def _edges(self, plot_range: PlotRange) -> tuple[float, float]:
        """Where its edges are drawn: an unbounded one at the trace's end."""
        first, last = float(self._x[0]), float(self._x[-1])
        bgn = first if math.isinf(plot_range.xbgn) else plot_range.xbgn
        end = last if math.isinf(plot_range.xend) else plot_range.xend
        return bgn, end

    def _show(self, plot_range: PlotRange) -> None:
        region = self._items[plot_range.key][0]
        drawn = self._edges(plot_range)
        color = QtGui.QColor(plot_range.color)
        fill = QtGui.QColor(color)
        fill.setAlpha(self.BRUSH_ALPHA)
        region.blockSignals(True)
        region.setRegion(drawn)
        region.setBrush(pg.mkBrush(fill))
        for line, unbounded in zip(region.lines,
                                   (math.isinf(plot_range.xbgn), math.isinf(plot_range.xend))):
            style = QtCore.Qt.PenStyle.DashLine if unbounded else QtCore.Qt.PenStyle.SolidLine
            line.setPen(pg.mkPen(color, width=self.EDGE_WIDTH, style=style))
            line.setHoverPen(pg.mkPen(color, width=self.EDGE_HOVER_WIDTH, style=style))
        region.setToolTip(f"{plot_range.label}: drag an edge, or the middle to move both"
                          if self._movable else plot_range.label)
        region.blockSignals(False)
        self._items[plot_range.key] = (region, plot_range, drawn)

    def _on_dragged(self, key: str, final: bool) -> None:
        if self._source is None or key not in self._items:
            return
        region, plot_range, drawn = self._items[key]
        low, high = region.getRegion()
        # An edge not dragged keeps its value (so -inf / inf stay unbounded)
        xbgn = snap(self._x, low) if low != drawn[0] else plot_range.xbgn
        xend = snap(self._x, high) if high != drawn[1] else plot_range.xend
        self._source.set_plot_range(key, xbgn, xend, final)
        if final:
            self.update()  # snapped, and as the tool saved it


__all__ = ["MODES", "PlotRange", "RangeOverlay", "snap"]
