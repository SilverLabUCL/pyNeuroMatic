# -*- coding: utf-8 -*-
"""
Parameter descriptions (#555): what a function's parameters are, as data, so
one form builder can make their controls (pyneuromatic.gui.param_form).

A function (e.g. the Main tool's Average) lists its parameters as
:class:`NMParam` descriptions; the form shows a control for each (a number
box, checkbox, dropdown, array picker, or an x range's two boxes, also drawn
on the Plot as draggable lines), and returns their values as the function's
keyword arguments.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# The kinds of parameter
NUMBER = "number"  # a float
INTEGER = "integer"
BOOLEAN = "boolean"  # yes/no
CHOICE = "choice"  # one of :attr:`NMParam.choices`
ARRAY = "array"  # a data array's name
XRANGE = "xrange"  # an x range: two values, :attr:`NMParam.names` (begin, end)
COLOR = "color"  # e.g. "#1d4ed8"
KINDS = (NUMBER, INTEGER, BOOLEAN, CHOICE, ARRAY, XRANGE, COLOR)
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class NMParam:
    """One parameter of a function, e.g. Average's ``compute_stdv``::

        NMParam("compute_stdv", BOOLEAN, "Standard deviation", default=False)

    An x range has two values, its :attr:`names` (e.g. ``("xbgn", "xend")``),
    and a :attr:`color` for its lines on the Plot.

    Attributes:
        name: The keyword argument (for an x range, a key for the pair).
        kind: One of :data:`KINDS`.
        label: Shown beside the control.
        default: Its value to start with (an x range: a (begin, end) pair).
        tip: The control's tooltip.
        choices: A choice's ``(value, shown)`` pairs.
        minimum, maximum: A number's limits (inclusive), or None.
        units: Shown after a number (e.g. "ms"); "x" for the data's x units.
        names: An x range's two keyword arguments, (begin, end).
        color: An x range's lines on the Plot.
        finite: A number must be finite (not inf or -inf).
        when: ``(name, value)``: shown enabled only while parameter *name*
            has *value* (e.g. a color only with "One color").
    """

    name: str
    kind: str
    label: str
    default: Any = None
    tip: str = ""
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    units: str = ""
    names: tuple[str, str] = ()
    color: str = "#2563eb"
    finite: bool = False
    when: tuple = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("kind must be one of %s, not %r" % (KINDS, self.kind))
        if self.kind == XRANGE and len(self.names) != 2:
            raise ValueError("an x range needs names=(begin, end)")
        if self.kind == CHOICE and not self.choices:
            raise ValueError("a choice needs choices")

    def check(self, value: Any) -> Any:
        """*value* as this parameter's (e.g. "2.5" -> 2.5); raises
        ValueError, naming the parameter, if it isn't one."""
        label = self.label or self.name
        if self.kind == BOOLEAN:
            return bool(value)
        if self.kind == CHOICE:
            values = [v for v, _shown in self.choices]
            if value not in values:
                raise ValueError("%s: %r is not one of %s" % (label, value, values))
            return value
        if self.kind == ARRAY:
            if not isinstance(value, str) or not value:
                raise ValueError("%s: choose an array" % label)
            return value
        if self.kind == COLOR:
            if not isinstance(value, str) or not _COLOR.match(value):
                raise ValueError("%s: %r is not a color (#rrggbb)" % (label, value))
            return value.lower()
        if self.kind == XRANGE:
            begin, end = (_number(label + " begin", value[0]),
                          _number(label + " end", value[1]))
            if not begin < end:
                raise ValueError("%s: begin must be less than end" % label)
            return begin, end
        number = _number(label, value, whole=self.kind == INTEGER)
        if self.finite and not math.isfinite(number):
            raise ValueError("%s: %s is not a finite number" % (label, value))
        if self.minimum is not None and number < self.minimum:
            raise ValueError("%s: at least %g" % (label, self.minimum))
        if self.maximum is not None and number > self.maximum:
            raise ValueError("%s: at most %g" % (label, self.maximum))
        return number

    def keywords(self, value: Any) -> dict[str, Any]:
        """The function's keyword arguments for *value* (an x range's two)."""
        if self.kind == XRANGE:
            return {self.names[0]: value[0], self.names[1]: value[1]}
        return {self.name: value}


def _number(label: str, value: Any, whole: bool = False) -> float | int:
    """*value* as a number (text too: "inf", "-inf"); raises ValueError."""
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError("%s: %r is not a number" % (label, value)) from None
    if math.isnan(number):
        raise ValueError("%s: NaN is not a number" % label)
    if whole:
        if not number.is_integer():
            raise ValueError("%s: %r is not a whole number" % (label, value))
        return int(number)
    return number


def keywords(params: tuple[NMParam, ...], values: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments of *values* (by parameter name), each checked."""
    kwargs: dict[str, Any] = {}
    for param in params:
        if param.name in values:
            kwargs.update(param.keywords(param.check(values[param.name])))
    return kwargs


# An x range of the whole of each array, as functions use by default
X_RANGE = NMParam(
    "x", XRANGE, "X range", default=(-math.inf, math.inf), names=("xbgn", "xend"),
    tip="The part of each array used (-inf, inf: all of it); or drag its lines on the Plot")
