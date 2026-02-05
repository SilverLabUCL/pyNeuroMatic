# -*- coding: utf-8 -*-
"""
[Module description].

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source 
Software Toolkit for Acquisition, Analysis and Simulation of 
Electrophysiological Data. Front. Neuroinform. 12:14. 
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations
import copy
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
    NMProject (root)
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
"""


class NMData(NMObject):
    """
    NM Data class
    """

    # Extend NMObject's special attrs with NMData's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMData__x",
        "_NMData__y",
        "_NMData__dataseries_channel",
        "_NMData__dataseries_epoch",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMData0",
        xdim: NMDimensionX | None = None,
        ydim: NMDimension | None = None,
        dataseries_channel: NMChannel | None = None,
        dataseries_epoch: NMEpoch | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)  # NMObject

        self.__dataseries_channel: NMChannel | None = None
        self.__dataseries_epoch: NMEpoch | None = None

        if xdim is None:
            self.__x = NMDimensionX(self, "xscale")
            self.__x.start = 0
            self.__x.delta = 1
        elif isinstance(xdim, NMDimensionX):
            self.__x = xdim
        else:
            e = nmu.type_error_str(xdim, "xdim", "NMDimensionX")
            raise TypeError(e)

        if ydim is None:
            self.__y = NMDimension(self, "yscale")
        elif isinstance(ydim, NMDimension):
            self.__y = ydim
        else:
            e = nmu.type_error_str(ydim, "ydim", "NMDimension")
            raise TypeError(e)

        self.__x.ypair = self.__y.nparray

        self._dataseries_set(dataseries_channel, dataseries_epoch)

        # TODO: if dataseries exist, then use this as x-y scale master
        # TODO: turn off scale notes and divert here?
        # TODO: option that x-scale is an array (ref to NMData)

    # override
    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, NMData):
            return NotImplemented
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

    def __deepcopy__(self, memo: dict) -> NMData:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMData by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMData
        """
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMData's special attributes

        # __x and __y: deep copy (they're NMDimension objects)
        result._NMData__x = copy.deepcopy(self._NMData__x, memo)
        result._NMData__x._parent = result  # update parent
        result._NMData__y = copy.deepcopy(self._NMData__y, memo)
        result._NMData__y._parent = result  # update parent

        # Update ypair reference
        result._NMData__x.ypair = result._NMData__y.nparray

        # __dataseries_channel and __dataseries_epoch: copy references
        # If copying within a folder context, try to resolve to copied objects
        if result._folder is not None:
            # Try to resolve channel reference
            if self._NMData__dataseries_channel is not None:
                if id(self._NMData__dataseries_channel) in memo:
                    result._NMData__dataseries_channel = memo[id(self._NMData__dataseries_channel)]
                else:
                    result._NMData__dataseries_channel = self._NMData__dataseries_channel
            else:
                result._NMData__dataseries_channel = None

            # Try to resolve epoch reference
            if self._NMData__dataseries_epoch is not None:
                if id(self._NMData__dataseries_epoch) in memo:
                    result._NMData__dataseries_epoch = memo[id(self._NMData__dataseries_epoch)]
                else:
                    result._NMData__dataseries_epoch = self._NMData__dataseries_epoch
            else:
                result._NMData__dataseries_epoch = None
        else:
            # Direct copy: keep references
            result._NMData__dataseries_channel = self._NMData__dataseries_channel
            result._NMData__dataseries_epoch = self._NMData__dataseries_epoch

        return result

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
        if self.__dataseries_channel is None:
            return None
        if not isinstance(self.__dataseries_channel, NMChannel):
            return None
        if self.__dataseries_epoch is None:
            return None
        if not isinstance(self.__dataseries_epoch, NMEpoch):
            return None
        dataseries_c = self.__dataseries_channel._parent
        if not isinstance(dataseries_c, NMDataSeries):
            return None
        dataseries_e = self.__dataseries_epoch._parent
        if not isinstance(dataseries_e, NMDataSeries):
            return None
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
            e = nmu.type_error_str(channel, "channel", "NMChannel")
            raise TypeError(e)
        if epoch is None or isinstance(epoch, NMEpoch):
            self.__dataseries_epoch = epoch
        else:
            e = nmu.type_error_str(epoch, "epoch", "NMEpoch")
            raise TypeError(e)
        return self._dataseries


class NMDataContainer(NMObjectContainer):
    """
    Container of NMData
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataContainer0",
        rename_on: bool = True,
        name_prefix: str = "data",
        name_seq_format: str = "0",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            auto_name_prefix=name_prefix,
            auto_name_seq_format=name_seq_format,
        )

    # override, no super
    def content_type(self) -> str:
        return NMData.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        xdim: NMDimensionX | None = None,
        ydim: NMDimension | None = None,
        # quiet: bool = nmp.QUIET
    ) -> NMData | None:
        actual_name = self._newkey(name)
        # Use self._parent (NMFolder) to skip container in parent chain,
        # consistent with NMFolder, NMDataSeries, NMChannel, NMEpoch
        d = NMData(
            parent=self._parent,
            name=actual_name,
            xdim=xdim,
            ydim=ydim
        )
        if super()._new(d, select=select):
            return d
        return None

    # @property
    # def dataseries(self):  # use self._folder.dataseries
    #     if self._parent.__class__.__name__ == 'NMFolder':
    #         return self._parent.dataseries
    #     return None

    # override
    # TODO: this no longer works
    def remove(self, names=[], indexes=[], confirm=True, quiet=False):
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
