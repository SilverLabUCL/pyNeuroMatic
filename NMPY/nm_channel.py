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
        self._NMObject__rename = False

    @property
    def content(self):  # override, no super
        return {'channel': self.name}


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self, parent, name='NMChannelContainer'):
        o = Channel(parent, 'temp')
        super().__init__(parent, name=name, nmobj=o, prefix='')
        # NO PREFIX, Channel names are 'A', 'B'...
        self.__parent = parent
        self._Container__rename = False
        self._Container__duplicate = False
        self._Container__kill = False

    @property
    def content(self):  # override, no super
        return {'channel': self.names}

    """
    @property
    def select(self):
        nmu.alert('NOT USED. See nm.channel_select.')
        return super().select

    @select.setter
    def select(self, name):
        nmu.alert('NOT USED. See nm.channel_select.')
        return self.select_set(name)
    """

    def new(self, name='default', select=True, quiet=False):  # override
        o = Channel(self.__parent, name)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    def name_default(self, first=0, quiet=False):  # override, no super
        i = self.name_next_seq(first=first, quiet=quiet)
        if i >= 0:
            return str(i)
        return ''

    def name_next_seq(self, prefix='', first=0, quiet=False):
        # override, no super
        # NO PREFIX, Channel names are 'A', 'B'...
        first = 0  # enforce
        n = 10 + len(self.get_all())
        for i in range(first, n):
            # name = self.prefix + nmu.channel_char(i)
            name = nmu.channel_char(i)
            if self.exists(name):
                return i
        return -1
