# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import math
import numpy

# import numpy.typing as npt # No module named 'numpy.typing
from typing import Dict, Union

from pyneuromatic.nm_channel import NMChannel
from pyneuromatic.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.nm_epoch import NMEpoch
from pyneuromatic.nm_object import NMObject
from pyneuromatic.nm_object_container import NMObjectContainer
import pyneuromatic.nm_preferences as nmp
from pyneuromatic.nm_scale import NMScale, NMScaleX
import pyneuromatic.nm_utilities as nmu


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

    np_array: NumPy N-dimensional array (ndarray)
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMData",
        np_array=None,  # TODO: typing
        xscale: Union[dict, nmu.NMScaleXType] = {},
        yscale: Union[dict, nmu.NMScaleType] = {},
        # pass dictionary for independent scale
        # pass reference to NMscale (master)
        # dataseries: Union[nmu.NMDataSeriesType, None] = None,
        dataseries_channel: Union[nmu.NMChannelType, None] = None,
        dataseries_epoch: Union[nmu.NMEpochType, None] = None,
        copy: nmu.NMDataType = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)  # NMObject

        self.__np_array = None
        # self.__dataseries = None
        self.__dataseries_channel = None
        self.__dataseries_epoch = None

        if copy is None:
            pass
        elif isinstance(copy, NMData):
            if isinstance(copy.np_array, numpy.ndarray):
                np_array = copy.np_array.copy()
            xscale = copy._NMData__x.scale
            yscale = copy._NMData__y.scale
            # TODO
            # dataseries = copy._NMData__dataseries
            # dataseries_channel = copy._NMData__dataseries_channel
            # dataseries_epoch = copy._NMData__dataseries_epoch
        else:
            e = nmu.typeerror(copy, "copy", "NMData")
            raise TypeError(e)

        if np_array is None:
            pass
        elif isinstance(np_array, numpy.ndarray):
            self.__np_array = np_array
        else:
            e = nmu.typeerror(np_array, "np_array", "NumPy.ndarray")
            raise TypeError(e)

        if isinstance(xscale, NMScaleX):
            self.__x = xscale
        elif xscale is None:
            self.__x = NMScaleX(self, "xscale")
        elif isinstance(xscale, dict):
            self.__x = NMScaleX(self, "xscale", scale=xscale)
        else:
            e = nmu.typeerror(xscale, "xscale", "dictionary or NMScaleX")
            raise TypeError(e)

        if isinstance(yscale, NMScale):
            self.__y = yscale
        elif yscale is None:
            self.__y = NMScale(self, "yscale")
        elif isinstance(yscale, dict):
            self.__y = NMScale(self, "yscale", scale=yscale)
        else:
            e = nmu.typeerror(yscale, "yscale", "dictionary or NMScale")
            raise TypeError(e)

        self._dataseries_set(dataseries_channel, dataseries_epoch)

        # TODO: if dataseries exist, then use this as x-y scale master
        # TODO: turn off scale notes and divert here?
        # TODO: option that x-scale is an array (ref to NMData)

    # override, no super
    def copy(self) -> nmu.NMDataType:
        return NMData(copy=self)

    # override
    def __eq__(
        self,
        other: nmu.NMDataType,
    ) -> bool:
        if not super().__eq__(other):
            return False

        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        # print(isinstance(self.__np_array, numpy.ndarray))
        if isinstance(self.__np_array, numpy.ndarray):
            if isinstance(other.__np_array, numpy.ndarray):
                if self.__np_array.dtype != other.np_array.dtype:
                    return False
                if self.__np_array.shape != other.np_array.shape:
                    return False
                if self.__np_array.nbytes != other.np_array.nbytes:
                    return False
                if not numpy.array_equal(self.__np_array, other.np_array):
                    # array_equal returns false if both arrays filled with NANs
                    if nmp.NAN_EQ_NAN:
                        # compare array elements within a tolerance
                        if not numpy.allclose(
                            self.__np_array,
                            other.np_array,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        ):
                            return False
                    else:
                        return False
            else:
                return False
        elif isinstance(other.__np_array, numpy.ndarray):
            return False

        if self._dataseries_channel != other._dataseries_channel:
            return False
        if self._dataseries_epoch != other._dataseries_epoch:
            return False

        return True

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({"xscale": self.x.scale})
        k.update({"yscale": self.y.scale})
        if isinstance(self.__np_array, numpy.ndarray):
            k.update({"np_array": self.__np_array.dtype})
        else:
            k.update({"np_array": None})
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
    def x(self) -> nmu.NMScaleXType:
        if isinstance(self.__dataseries_channel, NMChannel):
            if isinstance(self.__dataseries_channel.x, NMScaleX):
                return self.__dataseries_channel.x
        return self.__x

    @property
    def y(self) -> nmu.NMScaleType:
        if isinstance(self.__dataseries_channel, NMChannel):
            if isinstance(self.__dataseries_channel.y, NMScale):
                return self.__dataseries_channel.y
        return self.__y

    @property
    def np_array(self):
        return self.__np_array

    @np_array.setter
    def np_array(self, np_array) -> None:
        return self._np_array_set(np_array)

    def _np_array_set(
        self,
        np_array,
        # quiet=nmp.QUIET
    ) -> None:
        if np_array is None:
            pass  # ok
        elif not isinstance(np_array, numpy.ndarray):
            e = nmu.typeerror(np_array, "np_array", "NumPy.ndarray")
            raise TypeError(e)
        if self.__np_array is None:
            old = None
        else:
            old = self.__np_array.__array_interface__["data"][0]
        self.__np_array = np_array
        self.modified()
        if self.__np_array is None:
            new = None
        else:
            new = self.__np_array.__array_interface__["data"][0]
        h = nmu.history_change("np_array reference", old, new)
        # self.__notes_container.new(h)
        # self._history(h, quiet=quiet)
        return None

    def np_array_make(
        self,
        shape,
        fill_value=NP_FILL_VALUE,
        dtype=NP_DTYPE,
        order=NP_ORDER,
        # quiet=nmp.QUIET
    ):
        # wrapper for NumPy.full
        self.__np_array = numpy.full(shape, fill_value, dtype=dtype, order=order)
        self.modified()
        if not isinstance(self.__np_array, numpy.ndarray):
            raise RuntimeError("failed to create numpy array")
        n = (
            "created numpy array (numpy.full): shape="
            + str(shape)
            + ", fill_value="
            + str(fill_value)
            + ", dtype="
            + str(dtype)
        )
        # self.__notes_container.new(n)
        # self._history(n, quiet=quiet)
        return True

    def np_array_make_random_normal(
        self,
        shape,
        mean=0,
        stdv=1,
        # quiet=nmp.QUIET
    ):
        # wrapper for NumPy.random.normal
        # dtype = float64
        self.__np_array = numpy.random.normal(mean, stdv, shape)
        self.modified()
        if not isinstance(self.__np_array, numpy.ndarray):
            raise RuntimeError("failed to create numpy array")
        n = (
            "created data array (numpy.random.normal): shape="
            + str(shape)
            + ", mean="
            + str(mean)
            + ", stdv="
            + str(stdv)
        )
        # self.__notes_container.new(n)
        # self._history(n, quiet=quiet)
        return True

    @property
    def _dataseries(self) -> Union[nmu.NMDataSeriesType, None]:
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
    def _dataseries_channel(self) -> Union[nmu.NMChannelType, None]:
        return self.__dataseries_channel

    @property
    def _dataseries_epoch(self) -> Union[nmu.NMEpochType, None]:
        return self.__dataseries_epoch

    def _dataseries_set(
        self,
        channel: Union[nmu.NMChannelType, None],
        epoch: Union[nmu.NMEpochType, None],
    ) -> Union[nmu.NMDataSeriesType, None]:
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
        name: str = "NMDataContainer",
        rename_on: bool = True,
        name_prefix: str = "data",
        name_seq_format: str = "0",
        copy: nmu.NMDataContainerType = None,
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
    def copy(self) -> nmu.NMDataContainerType:
        return NMDataContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMData.__name__

    # override
    def new(
        self,
        name: str = "default",
        np_array=None,  # TODO: typing
        xscale: Union[dict, nmu.NMScaleXType] = {},
        # TODO: can also be NMData?
        yscale: Union[dict, nmu.NMScaleType] = {},
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMDataType:
        name = self._newkey(name)
        d = NMData(
            parent=self._parent,
            name=name,
            np_array=np_array,
            xscale=xscale,
            yscale=yscale,
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
