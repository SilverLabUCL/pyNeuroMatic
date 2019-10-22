#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import NMObject
from nm_container import Container
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import alert
from nm_utilities import error
from nm_utilities import channel_char


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

    def __init__(self, parent):
        super().__init__(parent, prefix=nmconfig.CHAN_PREFIX)

    def object_new(self, name):  # override, do not call super
        return Channel(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Channel)

    @property
    def select(self):  # override, do not call super
        return 'NOT USED. See nm.dataprefix.select.channel_select'

    @select.setter
    def select(self, name):
        alert('NOT USED. See nm.dataprefix.select.channel_select')

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
