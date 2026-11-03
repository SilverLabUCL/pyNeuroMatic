# -*- coding: utf-8 -*-
"""
Base utilities for I/O operations.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
from __future__ import annotations
import dataclasses
import math
import re
from typing import NamedTuple


class ParsedUnits(NamedTuple):
    """Result of parsing a label string for units."""
    label: str
    units: str
    scale: float


# Standard unit conversions to preferred electrophysiology units
UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    # Time: convert to milliseconds
    "s": ("ms", 1e3),
    "sec": ("ms", 1e3),
    "second": ("ms", 1e3),
    "seconds": ("ms", 1e3),
    # Voltage: convert to millivolts
    "V": ("mV", 1e3),
    "volt": ("mV", 1e3),
    "volts": ("mV", 1e3),
    # Current: convert to picoamperes
    "A": ("pA", 1e12),
    "amp": ("pA", 1e12),
    "amps": ("pA", 1e12),
    "ampere": ("pA", 1e12),
    "amperes": ("pA", 1e12),
    # Already in preferred units - no conversion
    "ms": ("ms", 1.0),
    "mV": ("mV", 1.0),
    "pA": ("pA", 1.0),
    "nA": ("nA", 1.0),
}


def parse_units_from_label(label: str) -> ParsedUnits:
    """Parse units from a label string.

    Looks for units in parentheses at the end of the label, e.g.:
    - "Current (pA)" -> ParsedUnits("Current", "pA", 1.0)
    - "Voltage (V)" -> ParsedUnits("Voltage", "mV", 1000.0)
    - "Time (s)" -> ParsedUnits("Time", "ms", 1000.0)

    Args:
        label: The label string to parse.

    Returns:
        ParsedUnits with label, units, and scale factor.
    """
    if not label:
        return ParsedUnits("", "", 1.0)

    # Match pattern: "Label (units)" or "Label(units)"
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", label)

    if match:
        parsed_label = match.group(1).strip()
        units = match.group(2).strip()

        # Check for unit conversion
        if units in UNIT_CONVERSIONS:
            new_units, scale = UNIT_CONVERSIONS[units]
            return ParsedUnits(parsed_label, new_units, scale)

        return ParsedUnits(parsed_label, units, 1.0)

    # No units found in parentheses
    return ParsedUnits(label, "", 1.0)


def make_data_name(prefix: str, channel: int, epoch: int) -> str:
    """Create a NeuroMatic-style data name.

    Args:
        prefix: The data prefix (e.g., "Record").
        channel: Channel number (0=A, 1=B, etc.).
        epoch: Epoch/wave number.

    Returns:
        Name string like "RecordA0", "RecordB1", etc.
    """
    from pyneuromatic.core.nm_utilities import channel_char
    ch = channel_char(channel)
    if not ch:
        ch = chr(ord('A') + channel)  # Fallback
    return f"{prefix}{ch}{epoch}"


@dataclasses.dataclass
class FileInfo:
    """What a data file holds, for choosing what to import (#456).

    Attributes:
        prefix: The default prefix of the imported arrays (e.g. "Record").
        channels: ``(letter, label, units)`` of each channel, e.g.
            ``("A", "Vmem", "mV")``.
        epochs: The epoch numbers in the file (e.g. 0..18).
        n_points: Points per array (the longest).
        x_start: X value of the first point.
        x_delta: Sample interval.
        x_units: X units (e.g. "ms").
        file_format: The file's format, e.g. "Igor Pro packed experiment".
    """
    prefix: str
    channels: list[tuple[str, str, str]]
    epochs: list[int]
    n_points: int
    x_start: float
    x_delta: float
    x_units: str
    file_format: str = ""


def check_options(channels, epochs, xrange) -> None:
    """Raise if the import options aren't valid."""
    if channels is not None and (isinstance(channels, str)
                                 or not all(isinstance(c, str) for c in channels)):
        raise TypeError("channels: a list of channel letters, e.g. ['A', 'C']")
    if epochs is not None and not all(
            isinstance(e, int) and not isinstance(e, bool) for e in epochs):
        raise TypeError("epochs: epoch numbers, e.g. range(0, 10) or [0, 2, 5]")
    if xrange is not None:
        if len(xrange) != 2:
            raise ValueError("xrange: (xbgn, xend); None for an open end")
        xbgn, xend = xrange
        if xbgn is not None and xend is not None and not xbgn < xend:
            raise ValueError("xrange: xbgn must be less than xend")


def import_numbers(file_channels, file_epochs, channels, epochs):
    """The channels and epochs to import, renumbered from A and 0 in file
    order: ({file letter: new channel number}, {file epoch: new epoch}).

    Imported arrays are named by the new numbers (e.g. channel C, epoch 5
    alone become RecordA0), as a data series' channels and epochs are
    always A, B... and E0, E1... (#456). *channels* / *epochs* None: all.
    """
    if channels is not None:
        chosen = {c.upper() for c in channels}
        missing = chosen - {c.upper() for c in file_channels}
        if missing:
            raise ValueError("channels not in the file: %s" % ", ".join(sorted(missing)))
        file_channels = [c for c in file_channels if c.upper() in chosen]
    if epochs is not None:
        chosen_epochs = set(epochs)
        file_epochs = [e for e in file_epochs if e in chosen_epochs]
        if not file_epochs:
            raise ValueError("none of the epochs chosen are in the file")
    return ({c: i for i, c in enumerate(file_channels)},
            {e: i for i, e in enumerate(file_epochs)})


def note_import(folder, channel_numbers, epoch_numbers, channels, epochs, xrange) -> None:
    """Keep in the folder's metadata what part of the file was imported
    (its channels, epochs and X range), when not all of it."""
    if channels is None and epochs is None and xrange is None:
        return
    folder.metadata["import"] = {
        "channels": list(channel_numbers),  # the file's letters, in import order
        "epochs": list(epoch_numbers),  # the file's epoch numbers
        "xrange": list(xrange) if xrange is not None else None,
    }


def crop(y, start: float, delta: float, xrange):
    """*y* cropped to *xrange* (``(xbgn, xend)``, None for an open end):
    the points from xbgn to xend, inclusive, and the new X start."""
    if xrange is None:
        return y, start
    xbgn, xend = xrange
    n = len(y)
    first = 0 if xbgn is None else min(n, max(0, math.ceil((xbgn - start) / delta - 1e-9)))
    last = n if xend is None else min(n, max(0, math.floor((xend - start) / delta + 1e-9) + 1))
    last = max(first, last)
    return y[first:last], start + first * delta
