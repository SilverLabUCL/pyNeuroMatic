#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject, NMobject
from nm_object_container import NMObjectContainer, NMobjectContainer
from nm_scale import NMScaleX, NMscaleX, NMScale, NMscale
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, NewType, Union

NMchannel = NewType('NMChannel', NMobject)
NMchannelContainer = NewType('NMChannelContainer', NMobjectContainer)


class NMChannel(NMObject):
    """
    NM Channel class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMChannel',
        xscale: Union[dict, NMscaleX] = {},
        yscale: Union[dict, NMscale] = {},
        copy: NMchannel = None
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__thedata = []  # list of references to NMData

        if isinstance(xscale, NMScaleX):
            self.__x = xscale
        elif isinstance(xscale, dict):
            self.__x = NMScaleX(self, 'xscale', scale=xscale)
        else:
            e = self._type_error('xscale', 'dictionary or NMScaleX')
            raise TypeError(e)

        if isinstance(yscale, NMScale):
            self.__y = yscale
        elif isinstance(yscale, dict):
            self.__y = NMScale(self, 'yscale', scale=yscale)
        else:
            e = self._type_error('yscale', 'dictionary or NMScale')
            raise TypeError(e)

        # self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        # self.__transform = []
        # TODO: copy

    # override
    @property
    def parameters(self) -> Dict[str, str]:
        k = super().parameters
        k.update({'xscale': self.__x.scale})
        k.update({'yscale': self.__y.scale})
        # k.update({'graphXY': self.__graphXY})
        # k.update({'transform': self.__transform})
        return k

    # override, no super
    def copy(self) -> NMchannel:
        c = NMChannel(copy=self)
        c.note = 'this is a copy of ' + str(self)
        return c

    @property
    def x(self) -> Union[dict, NMscaleX]:
        return self.__x

    @property
    def y(self) -> Union[dict, NMscaleX]:
        return self.__y


class NMChannelContainer(NMObjectContainer):
    """
    Container for NM Channel objects
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMChannelContainer',
        copy: NMchannelContainer = None
    ) -> None:
        o = NMChannel(None, 'empty')
        prefix = ''  # no prefix, channel names are 'A', 'B'...
        super().__init__(parent=parent, name=name, nmobject=o,
                         prefix=prefix, rename=False, copy=copy)
        # TODO: copy

    # override, no super
    def copy(self) -> NMchannelContainer:
        c = NMChannelContainer(copy=self)
        c.note = 'this is a copy of ' + str(self)
        return c

    # override
    def new(self, xscale={}, yscale={}, select=True, quiet=nmp.QUIET):
        o = NMChannel(None, 'iwillberenamed', xscale=xscale, yscale=yscale)
        return super().new(name='default', nmobject=o, select=select,
                           quiet=quiet)

    # override, no super
    def name_next(self, quiet=nmp.QUIET):
        i = self.name_next_seq(quiet=quiet)
        if i >= 0:
            return nmu.channel_char(i)
        return ''

    # override, no super
    def name_next_seq(self, quiet=nmp.QUIET):
        # NO PREFIX, Channel names are 'A', 'B'...
        n = 10 + self.count
        for i in range(0, n):
            # name = self.prefix + nmu.channel_char(i)
            name = nmu.channel_char(i)
            if not self.exists(name):
                return i
        return -1

    # override, no super
    def duplicate(self):
        e = self._error('channels cannot be duplicated')
        raise RuntimeError(e)
