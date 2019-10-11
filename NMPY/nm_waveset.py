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
        if not name_ok(name):
            return error("bad name " + quotes(name))
        self.__name = name
        return True

    @property
    def waveset(self):
        return self.__waveset


class WaveSetContainer(Container):
    """
    Container for NM WaveSet
    """

    def __init__(self):
        super().__init__()
        self.prefix = nmconfig.SET_PREFIX

    def object_new(self, name):  # override, do not call super
        return WaveSet(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, WaveSet)
