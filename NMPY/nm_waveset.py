#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_utilities import name_ok
from nm_utilities import error

SET_PREFIX = "Set"

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
        if name_ok(name):
            self.__name = name

    @property
    def waveset(self):
        return self.__waveset


class WaveSetContainer(Container):
    """
    Container for NM WaveSet
    """

    def __init__(self):
        super().__init__()
        self.prefix = SET_PREFIX

    def object_new(self, name):
        return WaveSet(name)

    def instance_ok(self, obj):
        return isinstance(obj, WaveSet)
