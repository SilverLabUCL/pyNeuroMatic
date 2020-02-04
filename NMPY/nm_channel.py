#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_dimension as nmd
import nm_preferences as nmp
import nm_utilities as nmu


class Channel(NMObject):
    """
    NM Channel class
    """

    def __init__(self, parent, name, fxns={}, xdim={}, ydim={}):
        super().__init__(parent, name, fxns=fxns, rename=False)
        self.__x = nmd.XDimension(self, 'xdim', fxns=fxns, dim=xdim)
        self.__y = nmd.Dimension(self, 'ydim', fxns=fxns, dim=ydim)
        # self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        # self.__transform = []
        # self._param_list += ['graphXY', 'transform']

    # override
    @property
    def parameters(self):
        k = super().parameters
        # k.update({'graphXY': self.__graphXY})
        # k.update({'transform': self.__transform})
        return k

    # override, no super
    @property
    def content(self):
        return {'channel': self.name}

    # override, no super
    def copy(self):
        return Channel(self._parent, self.name, fxns=self._fxns,
                       xdim=self.__x.dim, ydim=self.__y.dim)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """

    def __init__(self, parent, name, fxns={}):
        t = Channel(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t, prefix='',
                         rename=False, duplicate=False)
        # NO PREFIX, Channel names are 'A', 'B'...

    # override, no super
    @property
    def content(self):
        return {'channels': self.names}

    # override
    def copy(self):
        c = ChannelContainer(self._parent, self.__name, self._fxns)
        super().copy(container=c)
        return c

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = Channel(self._parent, 'temp', self._fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override, no super
    def name_next(self, quiet=nmp.QUIET):
        i = self.name_next_seq(quiet=quiet)
        if i >= 0:
            return nmu.channel_char(i)
        return ''

    # override, no super
    def name_next_seq(self, quiet=nmp.QUIET):
        # NO PREFIX, Channel names are 'A', 'B'...
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        n = 10 + self.count
        for i in range(0, n):
            # name = self.prefix + nmu.channel_char(i)
            name = nmu.channel_char(i)
            if not self.exists(name):
                return i
        return -1
