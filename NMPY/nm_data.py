
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
    """

    def __init__(self, parent, name, fxns={}, shape=[],
                 fill_value=NP_FILL_VALUE, xdim={}, ydim={}, dataseries={}):
        super().__init__(parent, name, fxns=fxns)
        self._note_container = NoteContainer(self, 'Notes', fxns=fxns)
        self.__np_array = None  # NumPy N-dimensional array
        self.__dataseries = {}
        if dataseries and isinstance(dataseries, dict):
            for ds, c in dataseries.items():
                if isinstance(ds, DataSeries) and c in nmp.CHAN_LIST:
                    self.__dataseries.update({ds: c})
        # self.__size = 0
        if shape:
            self.__np_array_make(shape, fill_value=fill_value)
        self.__x = nmd.XDimension(self, 'xdim', fxns=fxns,
                                  notes=self._note_container)
        self.__y = nmd.Dimension(self, 'ydim', fxns=fxns,
                                 notes=self._note_container)
        if xdim:
            self.__x._dim_set(xdim, quiet=True)
        if ydim:
            self.__y._dim_set(ydim, quiet=True)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'xdim': self.__x.dim})
        k.update({'ydim': self.__y.dim})
        return k

    # override, no super
    @property
    def content(self):
        k = {'data': self.name}
        if self._note_container:
            k.update(self._note_container.content)
        return k

    # override
    def _equal(self, data, ignore_name=False, alert=False):
        if not super()._equal(data, ignore_name=ignore_name, alert=alert):
            return False
        if self._note_container:
            return self._note_container._equal(data._Data__note_container,
                                               alert=alert)
        return True

    # override
    def _copy(self, data, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(data, Data):
            raise TypeError(nmu.type_error(data, 'Data'))
        if not super()._copy(data, copy_name=copy_name, quiet=True):
            return False
        c = data._Data__note_container
        if self._note_container:
            if not self._note_container._copy(c, quiet=True):
                return False
        self.__x._copy(data.x)
        self.__y._copy(data.y)
        self._modified()
        h = 'copied Data ' + nmu.quotes(data.name) + ' to ' + nmu.quotes(name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def _add_dataseries(self, dataseries, chan_char):
        if not isinstance(dataseries, DataSeries):
            raise TypeError(nmu.type_error(dataseries, 'DataSeries'))
        if chan_char not in nmp.CHAN_LIST:
            raise ValueError('bad chan_char: ' + chan_char)
        self.__dataseries.update({dataseries: chan_char})
        return True

    def _remove_dataseries(self, dataseries):
        if not isinstance(dataseries, DataSeries):
            raise TypeError(nmu.type_error(dataseries, 'DataSeries'))
        if dataseries in self.__dataseries:
            del self.__dataseries[dataseries]
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
    def notes(self):
        return self._note_container

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
    def np_array(self, np_ndarray):
        return self._np_array_set(np_ndarray)

    def _np_array_set(self, np_ndarray, quiet=nmp.QUIET):
        if not isinstance(np_ndarray, np.ndarray):
            e = nmu.type_error(np_ndarray, 'numpy.ndarray')
            raise TypeError(e)
        old = self.__np_array
        self.__np_array = np_ndarray
        self._modified()
        h = nmu.history_change('np_array', old, self.__np_array)
        self._note_container.new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def np_array_make_random_normal(self, shape, mean=0, stdv=1,
                                    quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not nmu.number_ok(shape, no_neg=True):
            raise ValueError('bad shape: ' + str(shape))
        if not nmu.number_ok(mean):
            raise ValueError('bad mean: ' + str(mean))
        if not nmu.number_ok(stdv, no_neg=True):
            raise ValueError('bad stdv: ' + str(stdv))
        self.__np_array = np.random.normal(mean, stdv, shape)
        self._modified()
        # dtype = float64
        n = ('created data array (numpy.random.normal): shape=' +
             str(shape) + ', mean=' + str(mean) + ', stdv=' + str(stdv))
        self._note_container.new(n)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    __np_array_make_random_normal = np_array_make_random_normal

    def np_array_make(self, shape, fill_value=NP_FILL_VALUE, dtype=NP_DTYPE,
                      order=NP_ORDER, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not nmu.number_ok(shape, no_neg=True):
            raise ValueError('bad shape: ' + str(shape))
        self.__np_array = np.full(shape, fill_value, dtype=dtype, order=order)
        self._modified()
        n = ('created numpy array (numpy.full): shape=' + str(shape) +
             ', fill_value=' + str(fill_value) + ', dtype=' + str(dtype))
        self._note_container.new(n)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    __np_array_make = np_array_make


class DataContainer(Container):
    """
    Container for NM Data objects
    """
    __select_alert = ('NOT USED. See nm.dataseries.select, ' +
                      'nm.channel_select, nm.eset.select and nm.epoch_select.')

    def __init__(self, parent, name, fxns={}, dataseries_container=None):
        t = Data(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t,
                         prefix=nmp.DATA_PREFIX)
        if isinstance(dataseries_container, DataSeriesContainer):
            self.__dataseries_container = dataseries_container

    # override, no super
    @property
    def content(self):
        return {'data': self.names}

    # override
    def new(self, name='default', shape=[], fill_value=np.nan, xdim={},
            ydim={}, select=True, quiet=nmp.QUIET):
        o = Data(self._parent, 'temp', self._fxns, shape=shape,
                 fill_value=fill_value, xdim=xdim, ydim=ydim)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def kill(self, name, all_=False, confirm=True, quiet=nmp.QUIET):
        klist = super().kill(name=name, all_=all_, confirm=confirm,
                             quiet=quiet)
        if not self.__dataseries_container:
            return klist
        dsc = self.__dataseries_container
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
