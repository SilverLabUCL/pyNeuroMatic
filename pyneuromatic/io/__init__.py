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
from pyneuromatic.io.abf import read_abf
from pyneuromatic.io.axograph import read_axograph
from pyneuromatic.io.igor_text import write_itx
from pyneuromatic.io.pxp import read_pxp

__all__ = ["read_abf", "read_axograph", "read_pxp", "write_itx"]
