# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
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
