"""How tables show values, following the general settings (e.g. NaN).

The app sets the mode from ``manager.settings.table_nan``
(:func:`set_nan_mode`); table models ask :func:`cell` for each value's text
and look.
"""
from __future__ import annotations

import math

from PyQt6 import QtCore, QtGui

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


def copy_value(value) -> str:
    """A value for the clipboard: full precision, NaN always as "NaN"."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NaN" if math.isnan(number) else f"{number:.15g}"


def selection_text(view, value, header, row_label=None, row_label_header: str = "",
                   names: bool = True) -> str:
    """The selected cells of *view* as tab-separated text (for ⌘C).

    Whole columns copy every row, with the column names first (*names*);
    other selections copy the block they span (cells not selected are
    empty). *value(row, column)* and *header(column)* give the text;
    *row_label(row)* (e.g. the x value) adds a first column, headed
    *row_label_header*.
    """
    selection = view.selectionModel()
    model = view.model()
    if selection is None or model is None:
        return ""
    chosen = {(i.row(), i.column()) for i in selection.selectedIndexes()}
    if not chosen:
        return ""
    columns = sorted({column for _row, column in chosen})
    whole = all(selection.isColumnSelected(c, QtCore.QModelIndex()) for c in columns)
    if whole:
        rows = list(range(model.rowCount()))
    else:
        rows = list(range(min(r for r, _ in chosen), max(r for r, _ in chosen) + 1))
        columns = list(range(columns[0], columns[-1] + 1))
    lines = []
    if whole and names:
        lead = [row_label_header] if row_label is not None else []
        lines.append("\t".join(lead + [header(c) for c in columns]))
    for row in rows:
        cells = [value(row, c) if whole or (row, c) in chosen else "" for c in columns]
        lead = [row_label(row)] if row_label is not None else []
        lines.append("\t".join(lead + cells))
    return "\n".join(lines)


__all__ = ["cell", "copy_value", "nan_foreground", "nan_mode", "nan_tooltip",
           "selection_text", "set_nan_mode"]
