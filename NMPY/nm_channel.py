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

    def __init__(self, parent, name, fxns={}, xdims={}, ydims={}):
        super().__init__(parent, name, fxns=fxns, rename=False)
        self.__x = nmd.XDimension(self, 'xdim', fxns=fxns, notes=None)
        self.__y = nmd.Dimension(self, 'ydim', fxns=fxns, notes=None)
        if xdims:
            self.__x._dims_set(xdims, quiet=True)
        if ydims:
            self.__y._dims_set(ydims, quiet=True)
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'graphXY': self.__graphXY})
        k.update({'transform': self.__transform})
        return k

    # override, no super
    @property
    def content(self):
        return {'channel': self.name}

    # override
    def _copy(self, channel, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(channel, Channel):
            raise TypeError(nmu.type_error(channel, 'Channel'))
        if not super()._copy(channel, copy_name=copy_name, quiet=True):
            return False
        self.__x._copy(channel.x)
        self.__y._copy(channel.y)
        self.__graphXY = channel._Channel__graphXY
        self.__transform = channel._Channel__transform
        h = ('copied Channel ' + nmu.quotes(channel.name) + ' to ' +
             nmu.quotes(name))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

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
