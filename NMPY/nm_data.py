
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

from nm_container import NMObject
from nm_container import Container
from nm_dataseries import DataSeries
from nm_dataseries import DataSeriesContainer
import nm_dimension as nmd
from nm_note import NoteContainer
import nm_preferences as nmp
import nm_utilities as nmu

NP_ORDER = 'C'
NP_DTYPE = np.float64
NP_FILL_VALUE = np.nan


class Data(NMObject):
    """
    NM Data class

    np_array: NumPy N-dimensional array (ndarray)
    """

    def __init__(self, parent, name, fxns={}, np_array=None,
                 xdim={}, ydim={}, dataseries={}, **copy):
        super().__init__(parent, name, fxns=fxns)
        self._content_name = 'data'
        if np_array is None:
            self.__np_array = None
        elif isinstance(np_array, np.ndarray):
            # self.__np_array = np_array.copy()  # COPY ARRAY?
            self.__np_array = np_array
        else:
            raise TypeError(nmu.type_error(np_array, 'numpy.ndarray'))
        self.__note_container = None
        for k, v in copy.items():
            if k.lower() == 'notes' and isinstance(v, NoteContainer):
                self.__note_container = v
        if not isinstance(self.__note_container, NoteContainer):
            self.__note_container = NoteContainer(self, 'Notes', fxns=fxns)
        self.__x = nmd.XDimension(self, 'xdim', fxns=fxns, dim=xdim,
                                  notes=self.__note_container)
        self.__y = nmd.Dimension(self, 'ydim', fxns=fxns, dim=ydim,
                                 notes=self.__note_container)
        self.__dataseries = {}
        if dataseries is None:
            pass
        elif isinstance(dataseries, dict):
            for ds, c in dataseries.items():
                if ds is None:
                    pass
                elif isinstance(ds, DataSeries):
                    if c is None:
                        pass
                    elif isinstance(c, str):
                        if c in ds.channel_list:
                            self.__dataseries.update({ds: c})
                        else:
                            e = 'unknown dataseries channel: ' + c
                            raise ValueError(e)
                    else:
                        raise TypeError(nmu.type_error(c, 'string'))
                else:
                    raise TypeError(nmu.type_error(ds, 'DataSeries'))
        else:
            raise TypeError(nmu.type_error(dataseries, 'dict'))
        self._param_list += ['xdim', 'ydim', 'dataseries']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'xdim': self.__x.dim})
        k.update({'ydim': self.__y.dim})
        k.update({'dataseries': self._dataseries_str()})
        # need dataseries names for equal() to work
        return k

    # override
    @property
    def content(self):
        k = super().content
        if self.__note_container:
            k.update(self.__note_container.content)
        return k

    # override
    def _equal(self, data, alert=False):
        if not super()._equal(data, alert=alert):
            return False
        if self.__np_array is None and data.np_array is not None:
            if alert:
                self._alert('unequal np_array NoneType', tp=self._tp)
            return False
        if self.__np_array is not None and data.np_array is None:
            if alert:
                self._alert('unequal np_array NoneType', tp=self._tp)
            return False
        if self.__np_array is not None and data.np_array is not None:
            if self.__np_array.dtype != data.np_array.dtype:
                if alert:
                    self._alert('unequal np_array dtype', tp=self._tp)
                return False
            if self.__np_array.shape != data.np_array.shape:
                if alert:
                    self._alert('unequal np_array shape', tp=self._tp)
                return False
            if self.__np_array.nbytes != data.np_array.nbytes:
                if alert:
                    self._alert('unequal np_array nbytes', tp=self._tp)
                return False
            if not np.array_equal(self.__np_array, data.np_array):
                # array_equal returns false if both arrays filled with NANs
                if nmp.NAN_EQ_NAN:
                    if not np.allclose(self.__np_array, data.np_array, rtol=0,
                                       atol=0, equal_nan=True):
                        if alert:
                            self._alert('unequal np_array', tp=self._tp)
                        return False
        if self.__note_container:
            if not self.__note_container._equal(data._Data__note_container,
                                                alert=alert):
                return False
        # self.__dataseries
        return True

    # override, no super
    def copy(self):
        if self.__np_array is None:
            a = None
        else:
            a = self.__np_array.copy()
        nc = self.__note_container.copy()
        nc.off = True  # block notes during class creation
        c = Data(self._parent, self.name, fxns=self._fxns, np_array=a,
                 xdim=self.__x.dim, ydim=self.__y.dim,
                 dataseries=self.__dataseries, notes=nc)
        self._copy_extra(c)
        nc.off = False
        return c

    def _name_set(self, name, quiet=nmp.QUIET):
        raise RuntimeError('use container rename()')

    def _dataseries_str(self):
        d = {}
        for ds, c in self.__dataseries.items():
            d.update({ds.name: c})
        return d

    def _dataseries_add(self, dataseries, chan_char):
        if not isinstance(dataseries, DataSeries):
            raise TypeError(nmu.type_error(dataseries, 'DataSeries'))
        if chan_char not in nmp.CHAN_LIST:
            raise ValueError('bad chan_char: ' + chan_char)
        self.__dataseries.update({dataseries: chan_char})
        self._modified()
        return True

    def _dataseries_remove(self, dataseries):
        if not isinstance(dataseries, DataSeries):
            raise TypeError(nmu.type_error(dataseries, 'DataSeries'))
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
            return ('dimensions are superceded by those of data-series ' + dsn + '.'
                    + '\n' + 'do you want to continue?')
        dsn = [d.name for d in self.__dataseries if isinstance(d, DataSeries)]
        return ('dimensions are superceded by those of the following data-series: ' +
                str(dsn) + '.' + '\n' + 'do you want to continue?')

    @property
    def note(self):
        return self.__note_container

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
            raise TypeError(nmu.type_error(np_array, 'numpy.ndarray'))
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
        self.__note_container.new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def np_array_make(self, shape, fill_value=NP_FILL_VALUE, dtype=NP_DTYPE,
                      order=NP_ORDER, quiet=nmp.QUIET):
        self.__np_array = np.full(shape, fill_value, dtype=dtype, order=order)
        self._modified()
        if not isinstance(self.__np_array, np.ndarray):
            raise RuntimeError('failed to create numpy array')
        n = ('created numpy array (numpy.full): shape=' + str(shape) +
             ', fill_value=' + str(fill_value) + ', dtype=' + str(dtype))
        self.__note_container.new(n)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    def np_array_make_random_normal(self, shape, mean=0, stdv=1,
                                    quiet=nmp.QUIET):
        self.__np_array = np.random.normal(mean, stdv, shape)
        self._modified()
        # dtype = float64
        if not isinstance(self.__np_array, np.ndarray):
            raise RuntimeError('failed to create numpy array')
        n = ('created data array (numpy.random.normal): shape=' +
             str(shape) + ', mean=' + str(mean) + ', stdv=' + str(stdv))
        self.__note_container.new(n)
        self._history(n, tp=self._tp, quiet=quiet)
        return True


class DataContainer(Container):
    """
    Container for NM Data objects
    """

    def __init__(self, parent, name, fxns={}, **copy):
        t = Data(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t,
                         prefix=nmp.DATA_PREFIX, rename=True, **copy)
        self._content_name = 'data'

    # override, no super
    def copy(self):
        c = DataContainer(self._parent, self.name, fxns=self._fxns,
                          thecontainer=self._thecontainer_copy())
        self._copy_extra(c)
        return c

    # override
    def new(self, name='default', np_array=None, xdim={}, ydim={},
            dataseries={}, select=True, quiet=nmp.QUIET):
        o = Data(self._parent, 'temp', fxns=self._fxns, np_array=np_array,
                 xdim=xdim, ydim=ydim, dataseries=dataseries)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    @property
    def dataseries(self):
        if self._parent.__class__.__name__ == 'Folder':
            return self._parent.dataseries
        return None

    # override
    def kill(self, name, all_=False, confirm=True, quiet=nmp.QUIET):
        klist = super().kill(name=name, all_=all_, confirm=confirm,
                             quiet=quiet)
        dsc = self.dataseries
        if not dsc or not isinstance(dsc, DataSeriesContainer):
            return klist
        for d in klist:  # remove data refs from data series and sets
            for i in range(0, dsc.count):
                ds = dsc.getitem(index=i)
                if not ds or not ds.thedata:
                    continue
                for c, cdata in ds.thedata.items():
                    if d in cdata:
                        cdata.remove(d)
                        ds._modified()
                        dsc._modified()
                for j in range(0, ds.eset.count):
                    s = ds.eset.getitem(index=j)
                    if d in s.theset:
                        s.discard(d)
                        s._modified()
                        ds.eset._modified()
        return klist
