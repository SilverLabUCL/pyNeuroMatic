# -*- coding: utf-8 -*-
"""
I/O module for importing and exporting data.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

Public API:
    read_axograph: Read Axograph files (.axgx, .axgd)

Example:
    >>> from pyneuromatic.io import read_axograph
    >>> folder = read_axograph("data.axgx")
"""
from pyneuromatic.io.axograph import read_axograph
from pyneuromatic.io.igor_text import write_itx

__all__ = ["read_axograph", "write_itx"]
