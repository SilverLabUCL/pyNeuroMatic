#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import Container
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error


class WaveSet(object):
    """
    NM WaveSet class
    """

    def __init__(self, name):
        self.__name = name
        self.__waveset = set()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        error("use waveset rename function")

    @property
    def waveset(self):
        return self.__waveset


class WaveSetContainer(Container):
    """
    Container for NM WaveSet
    """

    def __init__(self, thewaves):
        super().__init__(prefix=nmconfig.SET_PREFIX)
        self.__thewaves = thewaves  # 2D matrix, i = channel #, j = wave #
        self.count_from = 1

    def object_new(self, name):  # override, do not call super
        return WaveSet(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, WaveSet)

    def rename(self, name, newname):  # override
        o = self.get(name)
        if not o:
            return False
        if o.name.casefold() == "All".casefold():
            error("cannot rename 'All' set")
            return False
        if o.name.casefold() == "SetX".casefold():
            error("cannot rename SetX")
            return False
        return super().rename(name, newname)

    def addWave(self, name, wavename):
        o = self.get(name)
        if not o:
            return False
        return False