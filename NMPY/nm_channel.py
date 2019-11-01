#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Channel(NMObject):
    """
    NM Channel class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self, parent, name):
        super().__init__(parent, name, prefix='Chan')

    def object_new(self, name):  # override, do not call super
        return Channel(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Channel)

    @property
    def select(self):  # override, do not call super
        return 'NOT USED. See nm.dataprefix.select.channel_select'

    @select.setter
    def select(self, name):
        nmu.alert('NOT USED. See nm.dataprefix.select.channel_select')

    def name_next(self):  # override, do not call super
        """Get next default channel name."""
        if self.prefix:
            prefix = self.prefix
        else:
            prefix = "Chan"
        n = 10 + len(self.getAll())
        for i in range(0, n):
            name = prefix + nmu.channel_char(i)
            if not self.exists(name):
                return name
        return prefix + "Z"

    def rename(self, name, newname):  # override, do not call super
        nmu.error("cannot rename Channel object")
        return False

    def kill(self, name, quiet=False):  # override, do not call super
        nmu.error("cannot kill Channel object")
        return False
