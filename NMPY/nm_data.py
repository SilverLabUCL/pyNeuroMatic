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

DIMS = {'xstart': 0, 'xdelta': 1, 'xlabel': '', 'xunits': '', 'ylabel': '',
        'yunits': '', 'dtype': np.float64}


class Data(NMObject):
    """
    NM Data class
    """

    def __init__(self, parent, name, fxns={}, samples=-1, fill_value=np.nan,
                 noise=[], dims=DIMS):
        super().__init__(parent, name, fxns=fxns)
        self.__note_container = NoteContainer(self, 'Notes', fxns=fxns)
        self.__ndarray = None  # NumPy N-dimensional array
        self.__xdata = None
        self.__xstart = 0
        self.__xdelta = 1
        self.__xlabel = ''
        self.__xunits = ''
        self.__ylabel = ''
        self.__yunits = ''
        self.__dtype = np.float64
        self.__size = 0
        self._dims_set(dims, quiet=True)
        if samples >= 0:
            if isinstance(noise, list) and len(noise) == 2:
                self.__make_np_array_random_normal(samples=samples,
                                                   mean=noise[0],
                                                   stdv=noise[1])
            else:
                self.__make_np_array(samples=samples, fill_value=fill_value,
                                     dtype=self.__dtype)

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
        # self.__dtype
        h = 'copied Data ' + nmu.quotes(data.name) + ' to ' + nmu.quotes(name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def ndarray(self):
        return self.__ndarray

    @ndarray.setter
    def ndarray(self, numpy_ndarray):
        if not isinstance(numpy_ndarray, np.ndarray):
            e = 'bad np_ndarray arg: expected type NumPy.ndarray'
            self._error(e, tp=self._tp)
        self.__ndarray = numpy_ndarray

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
        d.update({'dtype': self.__ndarray.dtype})
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, quiet=nmp.QUIET):
        if not isinstance(dims, dict):
            e = 'bad dims arg: expected a dictionary of dimensions'
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if 'xdata' in dims.keys():
            self._xdata_set(dims['xdata'], quiet=quiet)
        if 'xstart' in dims.keys():
            self._xstart_set(dims['xstart'], quiet=quiet)
        if 'xdelta' in dims.keys():
            self._xdelta_set(dims['xdelta'], quiet=quiet)
        if 'xlabel' in dims.keys():
            self._xlabel_set(dims['xlabel'], quiet=quiet)
        if 'xunits' in dims.keys():
            self._xunits_set(dims['xunits'], quiet=quiet)
        if 'ylabel' in dims.keys():
            self._ylabel_set(dims['ylabel'], quiet=quiet)
        if 'yunits' in dims.keys():
            self._yunits_set(dims['yunits'], quiet=quiet)
        if 'dtype' in dims.keys():
            self._dtype_set(dims['dtype'], quiet=quiet)
        return True

    @property
    def xdata(self):
        return self.__xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, quiet=nmp.QUIET):
        if xdata is None:
            pass  # ok
        elif not isinstance(xdata, Data):
            self._error('expected type Data', tp=self._tp)
            return False
        if xdata == self.__xdata:
            return True
        if self.__xdata:
            old = self.__xdata.name
        else:
            old = 'None'
        self.__xdata = xdata
        n = ('changed xdata from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xdata.name)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def xstart(self):
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        return self._xstart_set(xstart)

    def _xstart_set(self, xstart, quiet=nmp.QUIET):
        if np.isinf(xstart) or np.isnan(xstart):
            e = 'xstart: bad value: ' + str(xstart)
            self._error(e, tp=self._tp)
            return False
        if xstart == self.__xstart:
            return True
        old = self.__xstart
        self.__xstart = xstart
        n = ('changed xstart from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xstart)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def xdelta(self):
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        return self._xdelta_set(xdelta)

    def _xdelta_set(self, xdelta, quiet=nmp.QUIET):
        if np.isinf(xdelta) or np.isnan(xdelta):
            e = 'xdelta: bad value: ' + str(xdelta)
            self._error(e, tp=self._tp)
            return False
        if xdelta == self.__xdelta:
            return True
        old = self.__xdelta
        self.__xdelta = xdelta
        n = ('changed xdelta from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xdelta)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def xlabel(self):
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        return self._xlabel_set(xlabel)

    def _xlabel_set(self, xlabel, quiet=nmp.QUIET):
        if not isinstance(xlabel, str):
            e = 'bad xlabel: expected string'
            self._error(e, tp=self._tp)
            return False
        if xlabel == self.__xlabel:
            return True
        old = self.__xlabel
        self.__xlabel = xlabel
        n = ('changed xlabel from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xlabel)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def xunits(self):
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        return self._xunits_set(xunits)

    def _xunits_set(self, xunits, quiet=nmp.QUIET):
        if not isinstance(xunits, str):
            e = 'bad xunits: expected string'
            self._error(e, tp=self._tp)
            return False
        if xunits == self.__xunits:
            return True
        old = self.__xunits
        self.__xunits = xunits
        n = ('changed xunits from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xunits)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def ylabel(self):
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        return self._ylabel_set(ylabel)

    def _ylabel_set(self, ylabel, quiet=nmp.QUIET):
        if not isinstance(ylabel, str):
            e = 'bad ylabel: expected string'
            self._error(e, tp=self._tp)
            return False
        if ylabel == self.__ylabel:
            return True
        old = self.__ylabel
        self.__ylabel = ylabel
        n = ('changed ylabel from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__ylabel)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def yunits(self):
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        return self._yunits_set(yunits)

    def _yunits_set(self, yunits, quiet=nmp.QUIET):
        if not isinstance(yunits, str):
            e = 'bad yunits: expected string'
            self._error(e, tp=self._tp)
            return False
        if yunits == self.__yunits:
            return True
        old = self.__yunits
        self.__yunits = yunits
        n = ('changed yunits from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__yunits)))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        return self._dtype_set(dtype)

    def _dtype_set(self, dtype, quiet=nmp.QUIET):
        # check dtype is OK
        # change np array to new dtype ()
        # arr = arr.astype('float64')
        self.__dtype = dtype

    def make_np_array_random_normal(self, samples=0, mean=0, stdv=1,
                                    quiet=nmp.QUIET):
        if not nmu.number_ok(samples, no_neg=True):
            e = 'bad samples argument: ' + str(samples)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if not nmu.number_ok(mean):
            e = 'bad mean argument: ' + str(mean)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if not nmu.number_ok(stdv, no_neg=True):
            e = 'bad stdv argument: ' + str(stdv)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        self.__ndarray = np.random.normal(mean, stdv, samples)
        # dtype = float64
        n = ('created data array (numpy.random.normal): samples=' +
             str(samples) + ', mean=' + str(mean) + ', stdv=' + str(stdv))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    __make_np_array_random_normal = make_np_array_random_normal

    def make_np_array(self, samples=0, fill_value=np.nan, dtype=np.float64,
                      quiet=nmp.QUIET):
        if not nmu.number_ok(samples, no_neg=True):
            e = 'bad samples argument: ' + str(samples)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        self.__ndarray = np.full(samples, fill_value, dtype=dtype)
        n = ('created data array (numpy.full): samples=' + str(samples) +
             ', fill_value=' + str(fill_value) + ', dtype=' + str(dtype))
        self.note.new(note=n, quiet=True)
        self._history(n, tp=self._tp, quiet=quiet)
        return True

    __make_np_array = make_np_array


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
        else:
            e = 'dataseries_container arg: expected DataSeriesContainer type'
            raise TypeError(e)

    # override, no super
    @property
    def content(self):
        return {'data': self.names}

    # override
    def new(self, name='default', samples=0, fill_value=np.nan, noise=[],
            dims=DIMS, select=True, quiet=nmp.QUIET):
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        o = Data(self._parent, name, self._fxns, samples=samples,
                 fill_value=fill_value, noise=noise, dims=dims)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def kill(self, name, all_=False, ask=True, quiet=nmp.QUIET):
        klist = super().kill(name=name, all_=all_, ask=ask, quiet=quiet)
        for d in klist:  # remove data refs from data series and sets
            for i in range(0, self.__dataseries_container.count):
                ds = self.__dataseries_container.get(item_num=i)
                if not ds or not ds.thedata:
                    continue
                for cdata in ds.thedata:
                    if d in cdata:
                        cdata.remove(d)
                for j in range(0, ds.eset.count):
                    s = ds.eset.get(item_num=j)
                    if d in s.theset:
                        s.discard(d)
        return klist
