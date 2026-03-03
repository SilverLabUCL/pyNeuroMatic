# -*- coding: utf-8 -*-
"""
NM Configurations - Global constants for pyNeuroMatic.

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

# Global behaviour flags used as default parameter values throughout the
# codebase.

DATASERIES_SET_LIST = ["all", "set1", "set2", "setX"]
QUIET = False       # suppress history/log output
GUI = False         # GUI mode active
NAN_EQ_NAN = True  # treat NaN == NaN (Python default is NaN != NaN)
