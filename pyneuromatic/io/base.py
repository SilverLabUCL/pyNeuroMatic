# -*- coding: utf-8 -*-
"""
Base utilities for I/O operations.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
from __future__ import annotations
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
