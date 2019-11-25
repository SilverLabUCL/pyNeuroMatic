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
        super().__init__(parent, name, rename=False)
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

    @property
    def content(self):  # override, no super
        return {'channel': self.name}


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """
    __prefix = ''  # no prefix, channel objects will be names 'A', 'B'...
    __select_alert = 'NOT USED. See nm.channel_select.'

    def __init__(self, parent, name='NMChannelContainer'):
        super().__init__(parent, name, prefix=self.__prefix,
                         select_alert=self.__select_alert, rename=False,
                         duplicate=False, kill=False)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        return {'channel': self.names}
    
    def new(self, name='default', select=True, quiet=False, nmobj=None):
        # override
        o = Channel(self.__parent, 'temp')
        return super().new(name=name, select=select, quiet=quiet, nmobj=o)

    def name_default(self, quiet=False):  # override, no super
        """Get next default channel name."""
        n = 10 + len(self.get_all())
        for i in range(0, n):
            name = self.prefix + nmu.channel_char(i)
            if not self.exists(name):
                return name
        return ''
