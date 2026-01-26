# -*- coding: utf-8 -*-
"""
[Module description].

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
DATASERIES_SET_LIST = ["all", "set1", "set2", "setX"]
DELETE_CONFIRM = True
QUIET = False
GUI = False
NAN_EQ_NAN = True  # in Python nan != nan, use this flag so that nan == nan


class Configs(object):
    """
    NM Configs class
    """

    def __init__(self):
        self.__quiet = QUIET
        self.__gui = GUI

    @property
    def quiet(self):
        return self.__quiet

    @quiet.setter
    def quiet(self, quiet):
        if isinstance(quiet, bool):
            self.__quiet = quiet

    @property
    def gui(self):
        return self.__gui

    @gui.setter
    def gui(self, on):
        if isinstance(on, bool):
            self.__qui = on
