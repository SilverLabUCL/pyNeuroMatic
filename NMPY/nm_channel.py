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
from nm_utilities import channel_char


class Channel(object):
    """
    NM Channel class
    """

    def __init__(self, name):
        self.__name = name
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        error("cannot rename Channel object")

    @property
    def waveset(self):
        return self.__waveset


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self):
        super().__init__()
        self.prefix = nmconfig.CHAN_PREFIX

    def object_new(self, name):  # override, do not call super
        return Channel(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Channel)

    def name_next(self):  # override, do not call super
        """Get next default channel name."""
        if self.prefix:
            prefix = self.prefix
        else:
            prefix = "Chan"
        n = 10 + len(self.getAll())
        for i in range(0, n):
            name = prefix + channel_char(i)
            if not self.exists(name):
                return name
        return prefix + "Z"

    def rename(self, name, newname):  # override, do not call super
        error("cannot rename Channel object")
        return False

    def kill(self, name, quiet=False):  # override, do not call super
        error("cannot kill Channel object")
        return False
