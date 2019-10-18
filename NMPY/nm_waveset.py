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
from nm_utilities import history


class WaveSet(object):
    """
    NM WaveSet class
    """

    def __init__(self, name):
        self.__name = name
        self.__theset = set()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        error("use waveset rename function")

    @property
    def theset(self):
        return self.__theset
    
    def add_wave(self, wave):
        self.__theset.add(wave)
        history("added " + wave.name + " to " + self.__name)


class WaveSetContainer(Container):
    """
    Container for NM WaveSet
    """

    def __init__(self, wave_prefix):
        super().__init__(prefix=nmconfig.SET_PREFIX)
        self.count_from = 1
        self.__wave_prefix = wave_prefix
        self.__set_select = "All"

    def object_new(self, name):  # override, do not call super
        return WaveSet(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, WaveSet)
    
    @property
    def select(self):  # override, do not call super
        return self.__set_select

    @select.setter
    def select(self, set_eq):
        self.__set_select = set_eq
        
    def rename(self, name, newname):  # override
        s = self.get(name)
        if not s:
            return False
        if s.name.casefold() == "all":
            error("cannot rename 'All' set")
            return False
        if s.name.casefold() == "setx":
            error("cannot rename SetX")
            return False
        return super().rename(name, newname)

    def add(self, name, wave_num):
        s = self.get(name)
        if not s:
            return False
        if wave_num == -1:
            wave_num = self.__wave_prefix.wave_select
        for chan in self.__wave_prefix.thewaves:
            if wave_num < 0 or wave_num >= len(chan):
                return False
            s.add_wave(chan[wave_num])
        return True
    
    def getSelected(self):
        return []

