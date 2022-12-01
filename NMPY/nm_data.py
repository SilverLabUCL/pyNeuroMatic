
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

from nm_object import NMObject
from nm_object import NMobject
from nm_object_container import NMObjectContainer
from nm_object_container import NMobjectContainer
from nm_dataseries import DataSeries
from nm_dataseries import DataSeriesContainer
import nm_dimension as nmd
from nm_note import NMNoteContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, NewType

# NMdata = NewType('NMData', NMobject)

NP_ORDER = 'C'
NP_DTYPE = np.float64
NP_FILL_VALUE = np.nan


class Data(NMObject):
    """
    NM Data class

    np_array: NumPy N-dimensional array (ndarray)
    """

    def __init__(self, parent, name, np_array=None, xdim={}, ydim={},
                 dataseries={}, **copy):
        super().__init__(parent, name)
        if np_array is None:
            self.__np_array = None
        elif isinstance(np_array, np.ndarray):
            # self.__np_array = np_array.copy()  # COPY ARRAY?
            self.__np_array = np_array
        else:
            e = self._type_error('np_array', 'numpy.ndarray')
            raise TypeError(e)
        self.__notes_container = None
        for k, v in copy.items():
            if k.lower() == 'c_notes' and isinstance(v, NMNoteContainer):
                self.__notes_container = v
        if not isinstance(self.__notes_container, NMNoteContainer):
            self.__notes_container = NMNoteContainer(self, 'Notes')
        self.__x = nmd.XDimension(self, 'xdim', dim=xdim,
                                  notes=self.__notes_container)
        self.__y = nmd.Dimension(self, 'ydim', dim=ydim,
                                 notes=self.__notes_container)
        self.__dataseries = {}
        if dataseries is None:
            pass
        elif isinstance(dataseries, dict):
            for ds, cc in dataseries.items():
                if ds is None:
                    pass
                elif isinstance(ds, DataSeries):
                    if cc is None:
                        pass
                    elif isinstance(cc, str):
                        if cc in ds.channel_list:
                            self.__dataseries.update({ds: cc})
                        else:
                            channel = cc
                            e = self._value_error('channel')
                            raise ValueError(e)
                    else:
                        channel = cc
                        e = self._type_error('channel', 'channel character')
                        raise TypeError(e)
                else:
                    data_series = ds
                    e = self._type_error('data_series', 'DataSeries')
                    raise TypeError(e)
        else:
            e = self._type_error('dataseries', 'dictionary')
            raise TypeError(e)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'xdim': self.__x.dim})
        k.update({'ydim': self.__y.dim})
        k.update({'dataseries': self._dataseries_str()})
        # need nmobject names for isequivalent() to work
        return k

    # override
    @property
    def content(self):
        k = super().content
        if self.__notes_container:
            k.update(self.__notes_container.content)
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
            if not np.array_equal(self.__np_array, data.np_array):
                # array_equal returns false if both arrays filled with NANs
                if nan_eq_nan:
                    if not np.allclose(self.__np_array, data.np_array, rtol=0,
                                       atol=0, equal_nan=True):
                        if alert:
                            self._alert(ue + 'np_array')
                        return False
        c = self.__notes_container
        c2 = data._Data__notes_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        # TODO self.__dataseries
        return True

    # override, no super
    def copy(self):
        if self.__np_array is None:
            a = None
        else:
            a = self.__np_array.copy()
        nc = self.__notes_container.copy()
        nc.off = True  # block notes during class creation
        c = Data(self._parent, self.name, np_array=a, xdim=self.__x.dim,
                 ydim=self.__y.dim, dataseries=self.__dataseries, c_notes=nc)
        nc.off = False
        return c

    def _dataseries_str(self):
        d = {}
        for ds, c in self.__dataseries.items():
            d.update({ds.name: c})
        return d

    def _dataseries_add(self, dataseries, chan_char):
        if not isinstance(dataseries, DataSeries):
            e = self._type_error('dataseries', 'DataSeries')
            raise TypeError(e)
        if not isinstance(chan_char, str):
            e = self._type_error('chan_char', 'channel character')
            raise TypeError(e)
        cc = nmu.channel_char_check(chan_char)
        if not cc:
            channel = chan_char
            e = self._value_error('channel')
            raise ValueError(e)
        self.__dataseries.update({dataseries: cc})
        self._modified()
        return True

    def _dataseries_remove(self, dataseries):
        if not isinstance(dataseries, DataSeries):
            e = self._type_error('dataseries', 'DataSeries')
            raise TypeError(e)
        if dataseries in self.__dataseries:
            del self.__dataseries[dataseries]
            self._modified()
        return True

    def _dataseries_alert(self):
        count = len(self.__dataseries)
        if count == 0:
            return ''
        if count == 1:
            if not isinstance(self.__dataseries[0], DataSeries):
                return ''
            dsn = nmu.quotes(self.__dataseries[0].name)
            a = ('dimensions are superceded by those of data-series ' + dsn +
                 '.' + '\n' + 'do you want to continue?')
            return a
        dsn = [d.name for d in self.__dataseries if isinstance(d, DataSeries)]
        a = ('dimensions are superceded by those of the following ' +
             'data-series: ' + str(dsn) + '.' + '\n' +
             'do you want to continue?')
        return a

    @property
    def note(self):
        return self.__notes_container

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
    def np_array(self, np_array):
        return self._np_array_set(np_array)

    def _np_array_set(self, np_array, quiet=nmp.QUIET):
        if np_array is None:
            pass  # ok
        elif not isinstance(np_array, np.ndarray):
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
        self.__notes_container.new(h)
        self._history(h, quiet=quiet)
        return True

    def np_array_make(self, shape, fill_value=NP_FILL_VALUE, dtype=NP_DTYPE,
                      order=NP_ORDER, quiet=nmp.QUIET):
        # wrapper for NumPy.full
        self.__np_array = np.full(shape, fill_value, dtype=dtype, order=order)
        self._modified()
        if not isinstance(self.__np_array, np.ndarray):
            e = self._error('failed to create numpy array')
            raise RuntimeError(e)
        n = ('created numpy array (numpy.full): shape=' + str(shape) +
             ', fill_value=' + str(fill_value) + ', dtype=' + str(dtype))
        self.__notes_container.new(n)
        self._history(n, quiet=quiet)
        return True

    def np_array_make_random_normal(self, shape, mean=0, stdv=1,
                                    quiet=nmp.QUIET):
        # wrapper for NumPy.random.normal
        # dtype = float64
        self.__np_array = np.random.normal(mean, stdv, shape)
        self._modified()
        if not isinstance(self.__np_array, np.ndarray):
            e = self._error('failed to create numpy array')
            raise RuntimeError(e)
        n = ('created data array (numpy.random.normal): shape=' +
             str(shape) + ', mean=' + str(mean) + ', stdv=' + str(stdv))
        self.__notes_container.new(n)
        self._history(n, quiet=quiet)
        return True


class DataContainer(NMObjectContainer):
    """
    Container for NM Data objects
    """

    def __init__(self, parent, name, **copy):
        d = Data(None, 'empty')
        super().__init__(parent, name, nmobject=d, prefix=nmp.DATA_PREFIX,
                         rename=True, **copy)

    # override, no super
    def copy(self):
        return DataContainer(self._parent, self.name, c_prefix=self.prefix,
                             c_rename=self.parameters['rename'],
                             c_thecontainer=self._thecontainer_copy())

    # override
    def new(self, name='default', np_array=None, xdim={}, ydim={},
            dataseries={}, select=True, quiet=nmp.QUIET):
        o = Data(None, 'iwillberenamed', np_array=np_array, xdim=xdim,
                 ydim=ydim, dataseries=dataseries)
        return super().new(name=name, nmobject=o, select=select, quiet=quiet)

    # wrapper
    def add(self, data, select=True, quiet=nmp.QUIET):
        if not isinstance(data, Data):
            e = self._type_error('data', 'Data')
            raise TypeError(e)
        return super().new(name=data.name, nmobject=data, select=select,
                           quiet=quiet)

    @property
    def dataseries(self):
        if self._parent.__class__.__name__ == 'Folder':
            return self._parent.dataseries
        return None

    # override
    def remove(self, names=[], indexes=[], confirm=True, quiet=nmp.QUIET):
        # wrapper for NMObjectContainer.remove()
        rlist = super().remove(names=names, indexes=indexes, confirm=confirm,
                               quiet=quiet)
        dsc = self.dataseries
        if not dsc or not isinstance(dsc, DataSeriesContainer):
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
