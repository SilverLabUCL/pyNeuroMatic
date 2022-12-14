
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import math
import numpy
# import numpy.typing as npt # No module named 'numpy.typing

from nm_object import NMObject
from nm_object_container import NMObjectContainer
from nm_dataseries import NMDataSeries, NMDataSeriesContainer
from nm_scale import NMScale, NMScaleX
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, Optional, Union

NP_ORDER = 'C'
NP_DTYPE = numpy.float64
NP_FILL_VALUE = numpy.nan


class NMData(NMObject):
    """
    NM Data class

    np_array: NumPy N-dimensional array (ndarray)
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMData',
        np_array=None,  # TODO: typing
        xscale: Union[dict, nmu.NMScaleXType] = {},  # TODO: also NMData?
        yscale: Union[dict, nmu.NMScaleType] = {},
        # pass dictionary for independent scale
        # pass reference to NMscale (master)
        dataseries: Optional[dict] = None,  # master dataseries
        # {'dataseries': NMDataSeries, 'channel': 'A', 'epoch': 0}
        copy: nmu.NMDataType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        if isinstance(copy, NMData):
            if isinstance(copy.np_array, numpy.ndarray):
                np_array = copy.np_array.copy()
            else:
                np_array = None
            xscale = copy._NMData__x.scale
            yscale = copy._NMData__y.scale
            dataseries = copy._NMData__dataseries

        if np_array is None:
            self.__np_array = None
        elif isinstance(np_array, numpy.ndarray):
            self.__np_array = np_array
        else:
            e = self._type_error('np_array', 'numpy.ndarray')
            raise TypeError(e)

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

        if isinstance(dataseries, dict) and dataseries:
            if 'dataseries' not in dataseries:
                e = ('missing key ' + nmu.quotes('dataseries'))
                raise KeyError(e)
            if 'channel' not in dataseries:
                e = ('missing key ' + nmu.quotes('channel'))
                raise KeyError(e)
            if 'epoch' not in dataseries:
                e = ('missing key ' + nmu.quotes('epoch'))
                raise KeyError(e)
            self._dataseries_set(dataseries['dataseries'],
                                 dataseries['channel'],
                                 dataseries['epoch'])

        # TODO: if dataseries exist, then use this as x-y scale master
        # TODO: turn off scale notes and divert here?
        # TODO: option that x-scale is an array (ref to NMData)

    # override, no super
    def copy(self) -> nmu.NMDataType:
        return NMData(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'xscale': self.__x.scale})
        k.update({'yscale': self.__y.scale})
        k.update({'dataseries': self.dataseries})
        # need nmobject names for isequivalent() to work
        return k

    # override
    def _isequivalent(self, data, alert=False):
        nan_eq_nan = nmp.NAN_EQ_NAN  # argument?
        if not super()._isequivalent(data, alert=alert):
            return False
        ue = 'unequivalent '
        if self.__np_array is None and data.np_array is not None:
            if alert:
                self._alert(ue + 'np_array NoneType')
            return False
        if self.__np_array is not None and data.np_array is None:
            if alert:
                self._alert(ue + 'np_array NoneType')
            return False
        if self.__np_array is not None and data.np_array is not None:
            if self.__np_array.dtype != data.np_array.dtype:
                if alert:
                    self._alert(ue + 'np_array dtype')
                return False
            if self.__np_array.shape != data.np_array.shape:
                if alert:
                    self._alert(ue + 'np_array shape')
                return False
            if self.__np_array.nbytes != data.np_array.nbytes:
                if alert:
                    self._alert(ue + 'np_array nbytes')
                return False
            if not numpy.array_equal(self.__np_array, data.np_array):
                # array_equal returns false if both arrays filled with NANs
                if nan_eq_nan:
                    if not numpy.allclose(self.__np_array, data.np_array,
                                          rtol=0, atol=0, equal_nan=True):
                        if alert:
                            self._alert(ue + 'np_array')
                        return False
        # TODO self.__dataseries
        return True

    @property
    def dataseries(self):
        if isinstance(self.__dataseries, dict) and self.__dataseries:
            if (
                'dataseries' in self.__dataseries and
                'channel' in self.__dataseries and
                'epoch' in self.__dataseries
            ):
                name = self.__dataseries['dataseries'].name
                channel = self.__dataseries['channel']
                epoch = self.__dataseries['epoch']
                return {'name': name, 'channel': channel, 'epoch': epoch}
        return None

    def _dataseries_set(
        self,
        dataseries: nmu.NMDataSeriesType,
        channel: str,
        epoch: int
    ) -> bool:
        if dataseries is None or not dataseries:
            self.__dataseries = None
            return True
        if not isinstance(dataseries, NMDataSeries):
            e = self._type_error('dataseries', 'NMDataSeries')
            raise TypeError(e)
        if not isinstance(channel, str):
            e = self._type_error('channel', 'character')
            raise TypeError(e)
        # TODO: check channel and epoch are OK for this dataseries?
        cc = nmu.channel_char_check(channel)
        if not cc:
            e = self._value_error('channel')
            raise ValueError(e)
        if not isinstance(epoch, int):
            e = self._type_error('int', 'integer')
            raise TypeError(e)

        self.__dataseries = {'dataseries': dataseries, 'channel': channel,
                             'epoch': epoch}
        self._modified()
        # TODO: history
        return True

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def np_array(self):
        return self.__np_array

    @np_array.setter
    def np_array(self, np_array) -> None:
        self._np_array_set(np_array)

    def _np_array_set(self, np_array, quiet=nmp.QUIET):
        if np_array is None:
            pass  # ok
        elif not isinstance(np_array, numpy.ndarray):
            e = self._type_error('np_array', 'numpy.ndarray')
            raise TypeError(e)
        if self.__np_array is None:
            old = None
        else:
            old = self.__np_array.__array_interface__['data'][0]
        self.__np_array = np_array
        self._modified()
        if self.__np_array is None:
            new = None
        else:
            new = self.__np_array.__array_interface__['data'][0]
        h = nmu.history_change('np_array reference', old, new)
        # self.__notes_container.new(h)
        self._history(h, quiet=quiet)
        return True

    def np_array_make(self, shape, fill_value=NP_FILL_VALUE, dtype=NP_DTYPE,
                      order=NP_ORDER, quiet=nmp.QUIET):
        # wrapper for NumPy.full
        self.__np_array = numpy.full(shape, fill_value, dtype=dtype,
                                     order=order)
        self._modified()
        if not isinstance(self.__np_array, numpy.ndarray):
            e = self._error('failed to create numpy array')
            raise RuntimeError(e)
        n = ('created numpy array (numpy.full): shape=' + str(shape) +
             ', fill_value=' + str(fill_value) + ', dtype=' + str(dtype))
        # self.__notes_container.new(n)
        self._history(n, quiet=quiet)
        return True

    def np_array_make_random_normal(self, shape, mean=0, stdv=1,
                                    quiet=nmp.QUIET):
        # wrapper for NumPy.random.normal
        # dtype = float64
        self.__np_array = numpy.random.normal(mean, stdv, shape)
        self._modified()
        if not isinstance(self.__np_array, numpy.ndarray):
            e = self._error('failed to create numpy array')
            raise RuntimeError(e)
        n = ('created data array (numpy.random.normal): shape=' +
             str(shape) + ', mean=' + str(mean) + ', stdv=' + str(stdv))
        # self.__notes_container.new(n)
        self._history(n, quiet=quiet)
        return True


class NMDataContainer(NMObjectContainer):
    """
    Container for NM Data objects
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMDataContainer',
        prefix: str = nmp.DATA_PREFIX,  # for generating names of NMData
        copy: nmu.NMDataContainerType = None
    ) -> None:
        o = NMData(parent=parent, name='ContainerUtility')
        super().__init__(parent=parent, name=name, nmobject=o, prefix=prefix,
                         rename=True, copy=copy)

    # override, no super
    def copy(self) -> nmu.NMDataContainerType:
        return NMDataContainer(copy=self)

    # override
    def new(
        self,
        name: str = 'default',
        np_array=None,  # TODO: typing
        xscale: Union[dict, nmu.NMScaleXType] = {},  # TODO: can also be NMData?
        yscale: Union[dict, nmu.NMScaleType] = {},
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> nmu.NMDataType:
        if isinstance(name, str) and name == 'default':
            name = self.name_next()
        o = NMData(parent=self._parent, name=name, np_array=np_array,
                   xscale=xscale, yscale=yscale)
        if super().append(nmobject=o, select=select, quiet=quiet):
            return o
        return None

    # @property
    # def dataseries(self):  # use self._folder.dataseries
    #     if self._parent.__class__.__name__ == 'NMFolder':
    #         return self._parent.dataseries
    #     return None

    # override
    def remove(self, names=[], indexes=[], confirm=True, quiet=nmp.QUIET):
        # wrapper for NMObjectContainer.remove()
        rlist = super().remove(names=names, indexes=indexes, confirm=confirm,
                               quiet=quiet)
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
                        ds._modified()
                        dsc._modified()
                for j in range(0, ds.sets.count):
                    s = ds.sets.getitem(index=j)
                    if d in s.theset:
                        s.discard(d)
                        s._modified()
                        ds.sets._modified()
        return rlist
