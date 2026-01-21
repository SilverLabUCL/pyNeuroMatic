# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from __future__ import annotations
import numpy
# import numpy.typing as npt # No module named 'numpy.typing

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


NP_ORDER = "C"
NP_DTYPE = numpy.float64
NP_FILL_VALUE = numpy.nan

"""
NM class tree:

NMManager
    NMProjectContainer
        NMProject (project0, project1...)
            NMFolderContainer
                NMFolder (folder0, folder1...)
                    NMDataContainer
                        NMData (record0, record1... avg0, avg1)
                    NMDataSeriesContainer
                        NMDataSeries (record, avg...)
                            NMChannelContainer
                                NMChannel (A, B, C... or 0, 1, 2)
                            NMSetContainer
                                NMSet (all, set0, set1...)
"""


class NMData(NMObject):
    """
    NM Data class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMData0",
        xdim: NMDimensionX | None = None,
        ydim: NMDimension | None = None,
        # dataseries: NMDataSeries | None = None,
        dataseries_channel: NMChannel | None = None,
        dataseries_epoch: NMEpoch | None = None,
        copy: NMData = None  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)  # NMObject

        # self.__dataseries = None
        self.__dataseries_channel = None
        self.__dataseries_epoch = None

        if copy is None:
            pass
        elif isinstance(copy, NMData):
            xdim = copy.x.copy()
            ydim = copy.y.copy()
            # TODO
            # dataseries = copy._NMData__dataseries
            # dataseries_channel = copy._NMData__dataseries_channel
            # dataseries_epoch = copy._NMData__dataseries_epoch
        else:
            e = nmu.typeerror(copy, "copy", NMData)
            raise TypeError(e)

        if xdim is None:
            self.__x = NMDimensionX(self, "xscale")
            self.__x.start = 0
            self.__x.delta = 1
        elif isinstance(xdim, NMDimensionX):
            self.__x = xdim
        else:
            e = nmu.typeerror(xdim, "xdim", "NMDimensionX")
            raise TypeError(e)

        if ydim is None:
            self.__y = NMDimension(self, "yscale")
        elif isinstance(ydim, NMDimension):
            self.__y = ydim
        else:
            e = nmu.typeerror(ydim, "ydim", "NMDimension")
            raise TypeError(e)

        self.__x.ypair = self.__y.nparray

        self._dataseries_set(dataseries_channel, dataseries_epoch)

        # TODO: if dataseries exist, then use this as x-y scale master
        # TODO: turn off scale notes and divert here?
        # TODO: option that x-scale is an array (ref to NMData)

    # override, no super
    def copy(self) -> NMData:
        return NMData(copy=self)

    # override
    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not super().__eq__(other):
            return False

        if self.__x != other.x:
            return False
        if self.__y != other.y:
            return False
        if self._dataseries_channel != other._dataseries_channel:
            return False
        if self._dataseries_epoch != other._dataseries_epoch:
            return False

        return True

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"x": self.x.parameters})
        k.update({"y": self.y.parameters})
        ds = self._dataseries
        if isinstance(ds, NMDataSeries):
            k.update({"dataseries": ds.name})
        else:
            k.update({"dataseries": None})
        if isinstance(self.__dataseries_channel, NMChannel):
            k.update({"dataseries channel": self.__dataseries_channel.name})
        else:
            k.update({"dataseries channel": None})
        if isinstance(self.__dataseries_epoch, NMEpoch):
            k.update({"dataseries epoch": self.__dataseries_epoch.name})
        else:
            k.update({"dataseries epoch": None})
        return k

    @property
    def x(self) -> NMDimensionX:
        if isinstance(self.__dataseries_channel, NMChannel):
            if isinstance(self.__dataseries_channel.x, NMDimensionX):
                return self.__dataseries_channel.x
        return self.__x

    @property
    def y(self) -> NMDimension:
        if isinstance(self.__dataseries_channel, NMChannel):
            if isinstance(self.__dataseries_channel.y, NMDimension):
                return self.__dataseries_channel.y
        return self.__y

    @property
    def _dataseries(self) -> NMDataSeries | None:
        if not isinstance(self.__dataseries_channel, NMChannel):
            return None
        if not isinstance(self.__dataseries_epoch, NMEpoch):
            return None
        dataseries_c = self.__dataseries_channel._parent
        dataseries_e = self.__dataseries_epoch._parent
        if dataseries_c is dataseries_e:
            return dataseries_c
        else:
            e = (
                "data channel and epoch are not from the same dataseries: "
                "'%s' vs '%s'" % (dataseries_c.name, dataseries_e.name)
            )
            raise ValueError(e)

    @property
    def _dataseries_channel(self) -> NMChannel | None:
        return self.__dataseries_channel

    @property
    def _dataseries_epoch(self) -> NMEpoch | None:
        return self.__dataseries_epoch

    def _dataseries_set(
        self,
        channel: NMChannel | None,
        epoch: NMEpoch | None,
    ) -> NMDataSeries | None:
        if channel is None or isinstance(channel, NMChannel):
            self.__dataseries_channel = channel
        else:
            e = nmu.typeerror(channel, "channel", "NMChannel")
            raise TypeError(e)
        if epoch is None or isinstance(epoch, NMEpoch):
            self.__dataseries_epoch = epoch
        else:
            e = nmu.typeerror(epoch, "epoch", "NMEpoch")
            raise TypeError(e)
        return self._dataseries


class NMDataContainer(NMObjectContainer):
    """
    Container of NMData
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMDataContainer0",
        rename_on: bool = True,
        name_prefix: str = "data",
        name_seq_format: str = "0",
        copy: NMDataContainer = None,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )

    # override, no super
    def copy(self) -> NMDataContainer:
        return NMDataContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMData.__name__

    # override
    def new(
        self,
        name: str = "default",
        xdim: NMDimensionX | None = None,
        ydim: NMDimension | None = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMData:
        name = self._newkey(name)
        d = NMData(
            parent=self._parent,
            name=name,
            xdim=xdim,
            ydim=ydim
        )
        super().new(d, select=select)
        return d

    # @property
    # def dataseries(self):  # use self._folder.dataseries
    #     if self._parent.__class__.__name__ == 'NMFolder':
    #         return self._parent.dataseries
    #     return None

    # override
    # TODO: this no longer works
    def remove(self, names=[], indexes=[], confirm=True, quiet=nmp.QUIET):
        rlist = super().remove(
            names=names, indexes=indexes, confirm=confirm, quiet=quiet
        )
        dsc = self.dataseries
        if not dsc or not isinstance(dsc, NMDataSeriesContainer):
            return rlist
        for d in rlist:  # remove data refs from data series and sets
            for i in range(0, dsc.count):
                ds = dsc.getitem(index=i)
                if not ds or not ds.thedata:
                    continue
                for c, cdata in ds.thedata.items():
                    if d in cdata:
                        cdata.remove(d)
                for j in range(0, ds.sets.count):
                    s = ds.sets.getitem(index=j)
                    if d in s.theset:
                        s.discard(d)
        return rlist
