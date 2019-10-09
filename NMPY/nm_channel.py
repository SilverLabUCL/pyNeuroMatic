#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_waveset import WaveSetContainer
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import chan_char

CHAN_PREFIX = "Chan"

class Channel(object):
    """
    NM Channel class
    """

    def __init__(self, name):
        self.__name = name
        self.__waveset = WaveSetContainer()
        self.__waveset.new("Set1")
        self.__waveset.new("Set2")
        self.__waveset.new("SetX")
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

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


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self):
        super().__init__()
        self.prefix = CHAN_PREFIX

    def object_new(self, name):
        return Channel(name)

    def instance_ok(self, obj):
        return isinstance(obj, Channel)

    def name_next(self):
        """Get next default channel name."""
        if self.prefix:
            prefix = self.prefix
        else:
            prefix = "Chan"
        n = 10 + len(self.getAll())
        for i in range(0, n):
            name = prefix + chan_char(i)
            if not self.exists(name):
                return name
        return prefix + "Z"

    def rename(self, name, newname):
        error("cannot rename Channel object")
        return False
