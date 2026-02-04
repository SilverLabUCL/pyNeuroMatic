# -*- coding: utf-8 -*-
"""
NMToolMain - Main tool that is always loaded.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiological Data. Front. Neuroinform. 12:14.
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

from pyneuromatic.analysis.nm_tool import NMTool


class NMToolMain(NMTool):
    """
    Main NM Tool - always loaded by default.

    Provides core functionality that is always available regardless
    of which other tools are enabled in the workspace.
    """

    def __init__(self) -> None:
        super().__init__()
