
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
                 fill_value=NP_FILL_VALUE, dims={}, dataseries={}):
        super().__init__(parent, name, fxns=fxns)
        self.__note_container = NoteContainer(self, 'Notes', fxns=fxns)
        self.__np_array = None  # NumPy N-dimensional array
        self.__dims = {'xdata': None, 'xstart': 0, 'xdelta': 1, 'xlabel': '',
                       'xunits': '', 'ylabel': '', 'yunits': ''}
        self.__dataseries = {}
        if dataseries and isinstance(dataseries, dict):
            for ds, c in dataseries.items():
                if isinstance(ds, DataSeries) and c in nmp.CHAN_LIST:
                    self.__dataseries.update({ds: c})
        # self.__size = 0
        if dims:
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
        self.dims = data.dims
        self._modified()
        h = 'copied Data ' + nmu.quotes(data.name) + ' to ' + nmu.quotes(name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def note(self):
        return self.__note_container

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
            return ('dims are superceded by those of data-series ' + dsn + '.'
                    + '\n' + 'do you want to continue?')
        dsn = [d.name for d in self.__dataseries if isinstance(d, DataSeries)]
        return ('dims are superceded by those of the following data-series: ' +
                str(dsn) + '.' + '\n' + 'do you want to continue?')

    @property
    def dims(self):
        d = {}
        d.update({'xdata': self.xdata})
        d.update({'xstart': self.xstart, 'xdelta': self.xdelta})
        d.update({'xlabel': self.xlabel, 'xunits': self.xunits})
        d.update({'ylabel': self.ylabel, 'yunits': self.yunits})
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dictionary of dimensions')
            raise TypeError(e)
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
        if 'xdata' in self.__dims.keys():
            return self.__dims['xdata']
        return None

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if xdata is None:
            pass  # ok
        elif not isinstance(xdata, Data):
            raise TypeError(nmu.type_error(xdata, 'Data'))
        old = self.xdata
        if xdata == old:
            return True
        self.__dims['xdata'] = xdata
        self._modified()
        if old:
            oldname = old.name
        else:
            oldname = 'None'
        if xdata:
            newname = xdata.name
        else:
            newname = 'None'
        h = nmu.history_change('xdata', oldname, newname)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def _xdata_alert(self):
        if not isinstance(self.xdata, Data):
            return ''
        xn = nmu.quotes(self.xdata.name)
        return ('x-dims are superceded by xdata ' + xn + '.' + '\n' +
                'do you want to continue?')

    @property
    def xstart(self):
        if 'xstart' in self.__dims.keys():
            return self.__dims['xstart']
        return 0

    @xstart.setter
    def xstart(self, xstart):
        return self._xstart_set(xstart)

    def _xstart_set(self, xstart, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xstart, float) and not isinstance(xstart, int):
            raise TypeError(nmu.type_error(xstart, 'number'))
        if not nmu.number_ok(xstart):
            raise ValueError('bad xstart: ' + str(xstart))
        old = self.xstart
        if xstart == old:
            return True
        self.__dims['xstart'] = xstart
        self._modified()
        h = nmu.history_change('xstart', old, xstart)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xdelta(self):
        if 'xdelta' in self.__dims.keys():
            return self.__dims['xdelta']
        return 1

    @xdelta.setter
    def xdelta(self, xdelta):
        return self._xdelta_set(xdelta)

    def _xdelta_set(self, xdelta, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xdelta, float) and not isinstance(xdelta, int):
            raise TypeError(nmu.type_error(xdelta, 'number'))
        if not nmu.number_ok(xdelta, no_zero=True):
            raise ValueError('bad xdelta: ' + str(xdelta))
        old = self.xdelta
        if xdelta == old:
            return True
        self.__dims['xdelta'] = xdelta
        self._modified()
        h = nmu.history_change('xdelta', old, xdelta)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xlabel(self):
        if 'xlabel' in self.__dims.keys():
            return self.__dims['xlabel']
        return ''

    @xlabel.setter
    def xlabel(self, xlabel):
        return self._xlabel_set(xlabel)

    def _xlabel_set(self, xlabel, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xlabel, str):
            raise TypeError(nmu.type_error(xlabel, 'string'))
        old = self.xlabel
        if xlabel == old:
            return True
        self.__dims['xlabel'] = xlabel
        self._modified()
        h = nmu.history_change('xlabel', old, xlabel)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xunits(self):
        if 'xunits' in self.__dims.keys():
            return self.__dims['xunits']
        return ''

    @xunits.setter
    def xunits(self, xunits):
        return self._xunits_set(xunits)

    def _xunits_set(self, xunits, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xunits, str):
            raise TypeError(nmu.type_error(xunits, 'string'))
        old = self.xunits
        if xunits == old:
            return True
        self.__dims['xunits'] = xunits
        self._modified()
        h = nmu.history_change('xunits', old, xunits)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def ylabel(self):
        if 'ylabel' in self.__dims.keys():
            return self.__dims['ylabel']
        return ''

    @ylabel.setter
    def ylabel(self, ylabel):
        return self._ylabel_set(ylabel)

    def _ylabel_set(self, ylabel, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(ylabel, str):
            raise TypeError(nmu.type_error(ylabel, 'string'))
        old = self.ylabel
        if ylabel == old:
            return True
        self.__dims['ylabel'] = ylabel
        self._modified()
        h = nmu.history_change('ylabel', old, ylabel)
        self.note.new(note=h, quiet=True)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def yunits(self):
        if 'yunits' in self.__dims.keys():
            return self.__dims['yunits']
        return ''

    @yunits.setter
    def yunits(self, yunits):
        return self._yunits_set(yunits)

    def _yunits_set(self, yunits, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.__dataseries and alert:
            if nmu.input_yesno(self._dataseries_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(yunits, str):
            raise TypeError(nmu.type_error(yunits, 'string'))
        old = self.yunits
        if yunits == old:
            return True
        self.__dims['yunits'] = yunits
        self._modified()
        h = nmu.history_change('yunits', old, yunits)
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
        o = Data(self._parent, 'temp', self._fxns, shape=shape,
                 fill_value=fill_value, dims=dims)
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
