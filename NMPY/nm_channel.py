#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_sets import SetsContainer
from nm_utilities import error

SETS_PREFIX = "Set"


class Channel(object):
    """
    NM Channel class
    """

    def __init__(self, name):
        self.__name = name
        self.__sets = SetsContainer(SETS_PREFIX)
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

    @property
    def name(self):
        return self.__name

    @property
    def sets(self):
        return self.__sets


class ChannelContainer(Container):
    """
    Container for NM Channels
    """

    def object_new(self, name):
        return Channel(name)

    def instance_ok(self, obj):
        return isinstance(obj, Channel)

    def name_next(self):
        """Get next default channel name."""
        chan_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        if self.prefix:
            prefix = self.prefix
        else:
            prefix = "Chan"
        n = 10 + len(self.getAll())
        for i in range(0, n):
            name = prefix + chan_chars[i]
            if not self.exists(name):
                return name
        return prefix + "Z"

    def rename(self, name, newname):
        error("cannot rename channel object")
        return False