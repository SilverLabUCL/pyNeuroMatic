#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject
from nm_object import NMObjectContainer
import nm_dimension as nmd
import nm_preferences as nmp
import nm_utilities as nmu


class Channel(NMObject):
    """
    NM Channel class
    """

    def __init__(self, parent, name, xdim={}, ydim={}, **copy):
        super().__init__(parent, name)
        self.__x = nmd.XDimension(self, 'xdim', dim=xdim)
        self.__y = nmd.Dimension(self, 'ydim', dim=ydim)
        # self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        # self.__transform = []
        self._param_list += ['xdim', 'ydim']  # ['graphXY', 'transform']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'xdim': self.__x.dim})
        k.update({'ydim': self.__y.dim})
        # k.update({'graphXY': self.__graphXY})
        # k.update({'transform': self.__transform})
        return k

    # override, no super
    def copy(self):
        return Channel(self._parent, self.name, xdim=self.__x.dim,
                       ydim=self.__y.dim)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y


class ChannelContainer(NMObjectContainer):
    """
    Container for NM Channel objects
    """

    def __init__(self, parent, name, **copy):
        t = Channel(None, 'empty').__class__.__name__
        super().__init__(parent, name, type_=t, prefix='', rename=False,
                         **copy)
        # NO PREFIX, Channel names are 'A', 'B'...

    # override, no super
    def copy(self):
        return ChannelContainer(self._parent, self.name, c_prefix=self.prefix,
                                c_rename=self.parameters['rename'],
                                c_thecontainer=self._thecontainer_copy())

    # override
    def new(self, xdim={}, ydim={}, select=True, quiet=nmp.QUIET):
        o = Channel(None, 'iwillberenamed', xdim=xdim, ydim=ydim)
        return super().new(name='default', nmobject=o, select=select,
                           quiet=quiet)

    # override, no super
    def name_next(self, quiet=nmp.QUIET):
        i = self.name_next_seq(quiet=quiet)
        if i >= 0:
            return nmu.chan_char(i)
        return ''

    # override, no super
    def name_next_seq(self, quiet=nmp.QUIET):
        # NO PREFIX, Channel names are 'A', 'B'...
        n = 10 + self.count
        for i in range(0, n):
            # name = self.prefix + nmu.chan_char(i)
            name = nmu.chan_char(i)
            if not self.exists(name):
                return i
        return -1

    # override, no super
    def duplicate(self):
        raise RuntimeError('channels cannot be duplicated')
