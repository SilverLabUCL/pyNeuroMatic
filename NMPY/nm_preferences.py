# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
PROJECT_NAME = 'NMProject'
FOLDER_PREFIX = 'NMFolder'
DATA_PREFIX = 'Data'
ESET_PREFIX = 'Set'
ESET_LIST = ['All', 'Set1', 'Set2', 'SetX']
CHAN_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
HISTORY_TREEPATH_SKIP = ['Project']
QUIET = False
GUI = False
NAN_EQ_NAN = True  # in Python nan != nan, use this flag so nan == nan
S0 = ' -> '


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
        self.__quiet = quiet

    @property
    def gui(self):
        return self.__gui

    @gui.setter
    def gui(self, on):
        self.__qui = on
