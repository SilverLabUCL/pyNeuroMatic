#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Channel(NMObject):
    """
    NM Channel class
    """

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns, rename=False)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

    @property  # override, no super
    def content(self):
        return {'channel': self.name}


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns, type_='Channel',
                         prefix='', rename=False, duplicate=False)
        # NO PREFIX, Channel names are 'A', 'B'...
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        return {'channels': self.names}

    # override
    def new(self, name='default', select=True, quiet=nmc.QUIET):
        o = Channel(self.__manager, self.__parent, name, self.__fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override, no super
    def name_next(self, first=0, quiet=nmc.QUIET):
        i = self.name_next_seq(first=first, quiet=quiet)
        if i >= 0:
            return nmu.channel_char(i)
        return ''

    # override, no super
    def name_next_seq(self, first=0, quiet=nmc.QUIET):
        # NO PREFIX, Channel names are 'A', 'B'...
        if not isinstance(first, int):
            first = 0
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        first = 0  # enforce
        n = 10 + self.count
        for i in range(first, n):
            # name = self.prefix + nmu.channel_char(i)
            name = nmu.channel_char(i)
            if not self.exists(name):
                return i
        return -1
