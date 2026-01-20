#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from __future__ import annotations

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProjectContainer
        NMProject (project0, project1...)
            NMFolderContainer
                NMFolder (folder0, folder1...)
                    NMDataContainer
                        NMData (recordA0, recordA1... avgA0, avgB0)
                    NMDataSeriesContainer
                        NMDataSeries (record, avg...)
                            NMChannelContainer
                                NMChannel (A, B, C...)
                            NMEpochContainer
                                NMEpoch (E0, E1, E2...)
DataSeries:
      E0  E1  E2... (epochs)
Ch A  A0  A1  A2...
Ch B  B0  B1  B2...
.
.
.
"""


class NMChannel(NMObject):
    """
    NM Channel class
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMChannel0",
        xscale: dict | NMDimensionX = {},
        yscale: dict | NMDimension = {},
        copy: NMChannel | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__x = None
        self.__y = None
        self.__thedata = []  # list of NMData refs for this channel

        # self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        # self.__transform = []

        if copy is None:
            pass
        elif isinstance(copy, NMChannel):
            xscale = copy._NMChannel__x.scale
            yscale = copy._NMChannel__y.scale
            if self._folder is None:
                # direct copy
                self.__thedata = list(copy._NMChannel__thedata)
            else:
                # grab NMData refs from copied NMDataContainer
                # NMDataContainer should be copied before this copy
                data = self._folder.data
                for d in copy._NMChannel__thedata:
                    o = data.get(d.name)
                    self.__thedata.append(o)
        else:
            e = nmu.typeerror(copy, "copy", NMChannel)
            raise TypeError(e)

        if xscale is None:
            pass
        elif isinstance(xscale, NMDimensionX):
            self.__x = xscale
        elif isinstance(xscale, dict):
            self.__x = NMDimensionX(parent=self, name="xscale", scale=xscale)
        else:
            e = nmu.typeerror(xscale, "xscale", "dictionary or NMDimensionX")
            raise TypeError(e)

        if yscale is None:
            pass
        elif isinstance(yscale, NMDimension):
            self.__y = yscale
        elif isinstance(yscale, dict):
            self.__y = NMDimension(parent=self, name="yscale", scale=yscale)
        else:
            e = nmu.typeerror(yscale, "yscale", "dictionary or NMDimension")
            raise TypeError(e)

        if not isinstance(self.__x, NMDimensionX):
            self.__x = NMDimensionX(parent=self, name="xscale")

        if not isinstance(self.__y, NMDimension):
            self.__y = NMDimension(parent=self, name="yscale")

        return None

    # override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if self.x.scale != other.x.scale:
            return False
        if self.y.scale != other.y.scale:
            return False
        if len(self.__thedata) != len(other.data):
            return False
        if not all(a == b for a, b in zip(self.__thedata, other.data)):
            return False
        return True

    # override, no super
    def copy(self) -> NMChannel:
        return NMChannel(copy=self)

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"xscale": self.__x.scale})
        k.update({"yscale": self.__y.scale})
        # k.update({'graphXY': self.__graphXY})
        # k.update({'transform': self.__transform})
        return k

    @property
    def data(self) -> list[NMData]:
        return self.__thedata

    @property
    def x(self) -> NMDimensionX:
        return self.__x

    @property
    def y(self) -> NMDimension:
        return self.__y


class NMChannelContainer(NMObjectContainer):
    """
    Container of NMChannels
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMChannelContainer0",
        rename_on: bool = False,
        name_prefix: str = "",  # default is no prefix
        name_seq_format: str = "A",
        copy: NMChannelContainer = None,
    ) -> None:
        return super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )

    # override, no super
    def copy(self) -> NMChannelContainer:
        return NMChannelContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMChannel.__name__

    # override
    def new(
        self,
        # name: str = 'A',  use name_next()
        xscale: dict | NMDimensionX = {},
        yscale: dict | NMDimension = {},
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMChannel:
        name = self.name_next()
        c = NMChannel(parent=self._parent, name=name, xscale=xscale, yscale=yscale)
        super().new(c, select=select)
        return c
