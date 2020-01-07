
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

from nm_container import NMObject
from nm_container import Container
from nm_dataseries import DataSeriesContainer
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
                 fill_value=NP_FILL_VALUE, dims={}):
        super().__init__(parent, name, fxns=fxns)
        self.__note_container = NoteContainer(self, 'Notes', fxns=fxns)
        self.__np_array = None  # NumPy N-dimensional array
        self.__xdata = None
        self.__xstart = 0
        self.__xdelta = 1
        self.__xlabel = ''
        self.__xunits = ''
        self.__ylabel = ''
        self.__yunits = ''
        self.__size = 0
        self._dims_set(dims, quiet=True)
        if shape:
            self.__np_array_make(shape, fill_value=fill_value)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update(self.dims)
        return k

    # override, no super
    @property
    def content(self):
        k = {'data': self.name}
        k.update(self.__note_container.content)
        return k

    # override
    def _equal(self, data, ignore_name=False, alert=False):
        if not super()._equal(data, ignore_name=ignore_name, alert=alert):
            return False
        return self.__note_container._equal(data._Data__note_container,
                                            alert=alert)

    # override
    def _copy(self, data, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(data, Data):
            raise TypeError(nmu.type_error(data, 'Data'))
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not super()._copy(data, copy_name=copy_name, quiet=True):
            return False
        c = data._Data__note_container
        if not self.__note_container._copy(c, quiet=True):
            return False
        self.__xdata = data._Data__xdata
        self.__xstart = data._Data__xstart
        self.__xdelta = data._Data__xdelta
        self.__xlabel = data._Data__xlabel
        self.__xunits = data._Data__xunits
        self.__ylabel = data._Data__ylabel
        self.__yunits = data._Data__yunits
        self._modified()
        h = 'copied Data ' + nmu.quotes(data.name) + ' to ' + nmu.quotes(name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def note(self):
        return self.__note_container

    @property
    def dims(self):
        if self.__xdata:
            d = {'xdata': self.__xdata.name}
        else:
            d = {'xstart': self.xstart, 'xdelta': self.xdelta}
        d.update({'xlabel': self.xlabel, 'xunits': self.xunits})
        d.update({'ylabel': self.ylabel, 'yunits': self.yunits})
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, quiet=nmp.QUIET):
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dictionary of dimensions')
            raise TypeError(e)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        for k in dims.keys():
            if k not in nmu.DIM_LIST:
                raise KeyError('unknown dimension key: ' + k)
        k = dims.keys()
        if 'xdata' in k:
            self._xdata_set(dims['xdata'], quiet=quiet)
        if 'xstart' in k:
            self._xstart_set(dims['xstart'], quiet=quiet)
        if 'xdelta' in k:
            self._xdelta_set(dims['xdelta'], quiet=quiet)
        if 'xlabel' in k:
            self._xlabel_set(dims['xlabel'], quiet=quiet)
        if 'xunits' in k:
            self._xunits_set(dims['xunits'], quiet=quiet)
        if 'ylabel' in k:
            self._ylabel_set(dims['ylabel'], quiet=quiet)
        if 'yunits' in k:
            self._yunits_set(dims['yunits'], quiet=quiet)
        return True

    @property
    def xdata(self):
        return self.__xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if xdata is None:
            pass  # ok
        elif not isinstance(xdata, Data):
            raise TypeError(nmu.type_error(xdata, 'Data'))
        if xdata == self.__xdata:
            return True
        if self.__xdata:
            old = self.__xdata.name
        else:
            old = 'None'
        self.__xdata = xdata
        self._modified()
        h = nmu.history_change('xdata', old, self.__xdata.name)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xstart(self):
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        return self._xstart_set(xstart)

    def _xstart_set(self, xstart, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(xstart, float) and not isinstance(xstart, int):
            raise TypeError(nmu.type_error(xstart, 'number'))
        if not nmu.number_ok(xstart):
            raise ValueError('bad xstart: ' + str(xstart))
        if xstart == self.__xstart:
            return True
        old = self.__xstart
        self.__xstart = xstart
        self._modified()
        h = nmu.history_change('xstart', old, self.__xstart)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xdelta(self):
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        return self._xdelta_set(xdelta)

    def _xdelta_set(self, xdelta, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(xdelta, float) and not isinstance(xdelta, int):
            raise TypeError(nmu.type_error(xdelta, 'number'))
        if not nmu.number_ok(xdelta, no_zero=True):
            raise ValueError('bad xdelta: ' + str(xdelta))
        if xdelta == self.__xdelta:
            return True
        old = self.__xdelta
        self.__xdelta = xdelta
        self._modified()
        h = nmu.history_change('xdelta', old, self.__xdelta)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xlabel(self):
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        return self._xlabel_set(xlabel)

    def _xlabel_set(self, xlabel, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(xlabel, str):
            raise TypeError(nmu.type_error(xlabel, 'string'))
        if xlabel == self.__xlabel:
            return True
        old = self.__xlabel
        self.__xlabel = xlabel
        self._modified()
        h = nmu.history_change('xlabel', old, self.__xlabel)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xunits(self):
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        return self._xunits_set(xunits)

    def _xunits_set(self, xunits, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(xunits, str):
            raise TypeError(nmu.type_error(xunits, 'string'))
        if xunits == self.__xunits:
            return True
        old = self.__xunits
        self.__xunits = xunits
        self._modified()
        h = nmu.history_change('xunits', old, self.__xunits)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def ylabel(self):
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        return self._ylabel_set(ylabel)

    def _ylabel_set(self, ylabel, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(ylabel, str):
            raise TypeError(nmu.type_error(ylabel, 'string'))
        if ylabel == self.__ylabel:
            return True
        old = self.__ylabel
        self.__ylabel = ylabel
        self._modified()
        h = nmu.history_change('ylabel', old, self.__ylabel)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def yunits(self):
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        return self._yunits_set(yunits)

    def _yunits_set(self, yunits, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(yunits, str):
            raise TypeError(nmu.type_error(yunits, 'string'))
        if yunits == self.__yunits:
            return True
        old = self.__yunits
        self.__yunits = yunits
        self._modified()
        h = nmu.history_change('yunits', old, self.__yunits)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def np_array(self):
        return self.__np_array

    @np_array.setter
    def np_array(self, np_ndarray):
        return self._np_array_set(np_ndarray)

    def _np_array_set(self, np_ndarray, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(np_ndarray, np.ndarray):
            e = nmu.type_error(np_ndarray, 'numpy.ndarray')
            raise TypeError(e)
        old = self.__np_array
        self.__np_array = np_ndarray
        self._modified()
        h = nmu.history_change('np_array', old, self.__np_array)
        self.note.new(note=h, quiet=True)
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
        self.note.new(note=n, quiet=True)
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
        self.note.new(note=n, quiet=True)
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
    def new(self, name='default', shape=[], fill_value=np.nan, dims={},
            select=True, quiet=nmp.QUIET):
        o = Data(self._parent, 'tempname', self._fxns, shape=shape,
                 fill_value=fill_value, dims=dims)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def kill(self, name, all_=False, ask=True, quiet=nmp.QUIET):
        klist = super().kill(name=name, all_=all_, ask=ask, quiet=quiet)
        if not self.__dataseries_container:
            return klist
        dsc = self.__dataseries_container
        for d in klist:  # remove data refs from data series and sets
            for i in range(0, dsc.count):
                ds = dsc.get(item_num=i)
                if not ds or not ds.thedata:
                    continue
                for cdata in ds.thedata:
                    if d in cdata:
                        cdata.remove(d)
                        ds._modified()
                        dsc._modified()
                for j in range(0, ds.eset.count):
                    s = ds.eset.get(item_num=j)
                    if d in s.theset:
                        s.discard(d)
                        s._modified()
                        ds.eset._modified()
        return klist
