#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject
from nm_object_container import NMObjectContainer
from nm_scale import NMScale, NMScaleX
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, Union


class NMChannel(NMObject):
    """
    NM Channel class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMChannel',
        xscale: Union[dict, nmu.NMScaleXType] = {},
        yscale: Union[dict, nmu.NMScaleType] = {},
        copy: nmu.NMChannelType = None
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__x = None
        self.__y = None
        self.__thedata = []  # list of NMData references

        # self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        # self.__transform = []

        if isinstance(copy, NMChannel):

            xscale = copy._NMChannel__x.scale
            self.__x = NMScaleX(self, 'xscale', scale=xscale)
            yscale = copy._NMChannel__y.scale
            self.__y = NMScale(self, 'yscale', scale=yscale)

            if isinstance(copy._NMChannel__thedata, list):
                self.__thedata = list(copy._NMChannel__thedata)

        else:

            if isinstance(xscale, NMScaleX):
                self.__x = xscale
            elif xscale is None:
                self.__x = NMScaleX(self, 'xscale')
            elif isinstance(xscale, dict):
                self.__x = NMScaleX(self, 'xscale', scale=xscale)
            else:
                e = self._type_error('xscale', 'dictionary or NMScaleX')
                raise TypeError(e)

            if isinstance(yscale, NMScale):
                self.__y = yscale
            elif yscale is None:
                self.__y = NMScale(self, 'yscale')
            elif isinstance(yscale, dict):
                self.__y = NMScale(self, 'yscale', scale=yscale)
            else:
                e = self._type_error('yscale', 'dictionary or NMScale')
                raise TypeError(e)

        if not isinstance(self.__x, NMScaleX):
            self.__x = NMScaleX(self, 'xscale')

        if not isinstance(self.__y, NMScale):
            self.__y = NMScale(self, 'yscale')

    # override
    def __eq__(
        self,
        other: nmu.NMObjectType
    ) -> bool:
        if not super().__eq__(other):
            return False
        if self.x.scale != other.x.scale:
            return False
        if self.y.scale != other.y.scale:
            return False
        for a, b in zip(self.data, other.data):
            if a != b:  # __ne__()
                return False
        return True

    # override, no super
    def copy(self) -> nmu.NMChannelType:
        return NMChannel(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'xscale': self.__x.scale})
        k.update({'yscale': self.__y.scale})
        # k.update({'graphXY': self.__graphXY})
        # k.update({'transform': self.__transform})
        return k

    @property
    def data(self) -> List[nmu.NMDataType]:
        return self.__thedata

    @property
    def x(self) -> nmu.NMScaleXType:
        return self.__x

    @property
    def y(self) -> nmu.NMScaleType:
        return self.__y


class NMChannelContainer(NMObjectContainer):
    """
    Container for NM Channel objects
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMChannelContainer',
        copy: nmu.NMChannelContainerType = None
    ) -> None:
        o = NMChannel(parent=parent, name='ContainerUtility')
        prefix = ''  # no prefix, channel names are 'A', 'B'...
        super().__init__(parent=parent, name=name, nmobject=o,
                         prefix=prefix, rename=False, copy=copy)

    # override, no super
    def copy(self) -> nmu.NMChannelContainerType:
        return NMChannelContainer(copy=self)

    # override
    def new(
        self,
        # name: str = 'A',  use next_name()
        xscale: Union[dict, nmu.NMScaleXType] = {},
        yscale: Union[dict, nmu.NMScaleType] = {},
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> nmu.NMChannelType:
        name = self.name_next()
        o = NMChannel(parent=self._parent, name=name, xscale=xscale,
                      yscale=yscale)
        if super().append(nmobject=o, select=select, quiet=quiet):
            return o
        return None

    # override, no super
    def append(self):
        e = self._error('use ' + nmu.quotes('new') +
                        'function to create a new channel.')
        raise RuntimeError(e)

    # override, no super
    def name_next(self):
        clist = nmp.CHANNEL_LIST
        if isinstance(clist, list) and len(clist) > 0:
            for c in clist:
                if not self.exists(c):
                    return c
        # failed to find new channel name, try another method...
        i = self.name_next_seq()
        if i >= 0:
            return nmu.channel_char(i)
        return None

    # override, no super
    def name_next_seq(self):
        # NMChannel names are 'A', 'B'...
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
