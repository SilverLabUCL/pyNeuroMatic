"""How tables show values, following the general settings (e.g. NaN).

The app sets the mode from ``manager.settings.table_nan``
(:func:`set_nan_mode`); table models ask :func:`cell` for each value's text
and look.
"""
from __future__ import annotations

import math

from PyQt6 import QtGui

from pyneuromatic.core.nm_settings import TABLE_NAN_BLANK, TABLE_NAN_GREY, TABLE_NAN_TEXT

NAN_GREY = QtGui.QColor("#b0b0b0")

_nan_mode = TABLE_NAN_TEXT


def nan_mode() -> str:
    return _nan_mode


def set_nan_mode(mode: str) -> None:
    """How tables show NaN: "NaN", "NaN (grey)" or "blank"."""
    global _nan_mode
    if mode not in (TABLE_NAN_TEXT, TABLE_NAN_GREY, TABLE_NAN_BLANK):
        raise ValueError(f"unknown table NaN mode {mode!r}")
    _nan_mode = mode


def cell(value) -> tuple[str, bool]:
    """(text, is_nan) for a table cell; None is an empty cell."""
    if value is None:
        return "", False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value), False
    if math.isnan(number):
        return ("" if _nan_mode == TABLE_NAN_BLANK else "NaN"), True
    return f"{number:.6g}", False


def nan_foreground() -> QtGui.QColor | None:
    """The text colour of a NaN cell (grey mode), else None."""
    return NAN_GREY if _nan_mode == TABLE_NAN_GREY else None


def nan_tooltip() -> str | None:
    """A blank NaN cell still says what it is when hovered."""
    return "NaN" if _nan_mode == TABLE_NAN_BLANK else None


__all__ = ["cell", "nan_foreground", "nan_mode", "nan_tooltip", "set_nan_mode"]
