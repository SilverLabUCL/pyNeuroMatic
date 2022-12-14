# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
DATA_PREFIX = 'Record'
DATASERIES_SET_PREFIX = 'Set'
DATASERIES_SET_LIST = ['All', 'Set1', 'Set2', 'SetX']
CHANNEL_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z']
HISTORY_TREEPATH_SKIP = ['Project']
NAME_SYMBOLS_OK = ['_']
BAD_NAMES = ['', 'all', 'default', 'none', 'select', 'self',
             'nan', 'inf', '-inf']
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
