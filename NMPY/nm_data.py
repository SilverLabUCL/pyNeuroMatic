# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_dataseries import DataSeriesContainer
from nm_note import NoteContainer
import nm_utilities as nmu

DIMS = {'xstart': 0, 'xdelta': 1, 'xlabel': '', 'xunits': '', 'ylabel': '',
        'yunits': ''}
DTYPE = np.dtype(float)


class Data(NMObject):
    """
    NM Data class
    """

    def __init__(self, parent, name, fxns, samples=0,
                 fill_value=0, dtype=DTYPE, noise=[], dims=DIMS):
        super().__init__(parent, name, fxns)
        self.make(samples=samples, fill_value=fill_value, dtype=dtype,
                  noise=noise)
        self.__note_container = NoteContainer(self, 'Notes', fxns)
        self.__xdata = None
        self.__xstart = 0
        self.__xdelta = 1
        self.__xlabel = ''
        self.__xunits = ''
        self.__ylabel = ''
        self.__yunits = ''
        self.dims = dims

    @property
    def __parent(self):
        return self._NMObject__parent

    @property
    def __fxns(self):
        return self._NMObject__fxns

    @property
    def __quiet(self):
        return self._NMObject__quiet

    @property
    def __alert(self):
        return self._NMObject__alert

    @property
    def __error(self):
        return self._NMObject__error

    @property
    def __history(self):
        return self._NMObject__history

    @property
    def __rename(self):
        return self._NMObject__rename

    @property
    def __tp(self):
        return self.tree_path(history=True)

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
    def equal(self, data, ignore_name=False, alert=False):
        if not super().equal(data, ignore_name=ignore_name, alert=alert):
            return False
        return self.__note_container.equal(data._Data__note_container,
                                           alert=alert)

    # override
    def copy(self, data, copy_name=True, quiet=nmc.QUIET):
        name = self.name
        if not super().copy(data, copy_name=copy_name, quiet=True):
            return False
        c = data._Data__note_container
        if not self.__note_container.copy(c, quiet=quiet):
            return False
        self.__xdata = data._Data__xdata
        self.__xstart = data._Data__xstart
        self.__xdelta = data._Data__xdelta
        self.__xlabel = data._Data__xlabel
        self.__xunits = data._Data__xunits
        self.__ylabel = data._Data__ylabel
        self.__yunits = data._Data__yunits
        h = 'copied Data ' + nmu.quotes(data.name) + ' to ' + nmu.quotes(name)
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

    @property
    def data(self):
        return self.__parent.data

    @property
    def thedata(self):
        return self.__thedata

    @thedata.setter
    def thedata(self, np_array):
        self.__thedata = np_array

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
        if 'xdata' in dims.keys():
            self.xdata = dims['xdata']
        if 'xstart' in dims.keys():
            self.xstart = dims['xstart']
        if 'xdelta' in dims.keys():
            self.xdelta = dims['xdelta']
        if 'xlabel' in dims.keys():
            self.xlabel = dims['xlabel']
        if 'xunits' in dims.keys():
            self.xunits = dims['xunits']
        if 'ylabel' in dims.keys():
            self.ylabel = dims['ylabel']
        if 'yunits' in dims.keys():
            self.yunits = dims['yunits']

    @property
    def xdata(self):
        return self.__xdata

    @xdata.setter
    def xdata(self, xdata):
        if not isinstance(xdata, Data):
            e = 'xdata: expected type Data'
            self.__error(e, tp=self.__tp)
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
        return True

    @property
    def xstart(self):
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        if np.isinf(xstart) or np.isnan(xstart):
            e = 'xstart: bad value: ' + str(xstart)
            self.__error(e, tp=self.__tp)
            return False
        if xstart == self.__xstart:
            return True
        old = self.__xstart
        self.__xstart = xstart
        n = ('changed xstart from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xstart)))
        self.note.new(note=n, quiet=True)
        return True

    @property
    def xdelta(self):
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        if np.isinf(xdelta) or np.isnan(xdelta):
            e = 'xdelta: bad value: ' + str(xdelta)
            self.__error(e, tp=self.__tp)
            return False
        if xdelta == self.__xdelta:
            return True
        old = self.__xdelta
        self.__xdelta = xdelta
        n = ('changed xdelta from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xdelta)))
        self.note.new(note=n, quiet=True)
        return True

    @property
    def xlabel(self):
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        if not isinstance(xlabel, str):
            e = 'bad xlabel: expected string'
            self.__error(e, tp=self.__tp)
            return False
        if xlabel == self.__xlabel:
            return True
        old = self.__xlabel
        self.__xlabel = xlabel
        n = ('changed xlabel from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xlabel)))
        self.note.new(note=n, quiet=True)
        return True

    @property
    def xunits(self):
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        if not isinstance(xunits, str):
            e = 'bad xunits: expected string'
            self.__error(e, tp=self.__tp)
            return False
        if xunits == self.__xunits:
            return True
        old = self.__xunits
        self.__xunits = xunits
        n = ('changed xunits from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__xunits)))
        self.note.new(note=n, quiet=True)
        return True

    @property
    def ylabel(self):
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        if not isinstance(ylabel, str):
            e = 'bad ylabel: expected string'
            self.__error(e, tp=self.__tp)
            return False
        if ylabel == self.__ylabel:
            return True
        old = self.__ylabel
        self.__ylabel = ylabel
        n = ('changed ylabel from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__ylabel)))
        self.note.new(note=n, quiet=True)
        return True

    @property
    def yunits(self):
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        if not isinstance(yunits, str):
            e = 'bad yunits: expected string'
            self.__error(e, tp=self.__tp)
            return False
        if yunits == self.__yunits:
            return True
        old = self.__yunits
        self.__yunits = yunits
        n = ('changed yunits from ' + nmu.quotes(str(old)) + ' to ' +
             nmu.quotes(str(self.__yunits)))
        self.note.new(note=n, quiet=True)
        return True

    def make(self, samples=0, fill_value=0, dtype=DTYPE, noise=[],
             quiet=nmc.QUIET):
        if not nmu.number_ok(samples, no_neg=True):
            e = 'bad samples argument: ' + str(samples)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return False
        if isinstance(noise, list) and len(noise) == 2:
            n_mean = noise[0]
            n_stdv = noise[1]
            numok = nmu.number_ok(n_mean)
            if not numok or not nmu.number_ok(n_stdv, no_neg=True):
                e = 'bad noise argument: ' + str(noise)
                self.__error(e, tp=self.__tp, quiet=quiet)
                return False
            self.__thedata = np.random.normal(n_mean, n_stdv, samples)
        else:
            self.__thedata = np.full(samples, fill_value, dtype=dtype)
        return True


class DataContainer(Container):
    """
    Container for NM Data objects
    """
    __select_alert = ('NOT USED. See nm.dataseries.select, ' +
                      'nm.channel_select, nm.eset.select and nm.epoch_select.')

    def __init__(self, parent, name, fxns, dataseries_container):
        t = Data(parent, 'empty', fxns).__class__.__name__
        super().__init__(parent, name, fxns, type_=t, prefix=nmc.DATA_PREFIX)
        if isinstance(dataseries_container, DataSeriesContainer):
            self.__dataseries_container = dataseries_container
        else:
            e = 'dataseries_container arg: expected DataSeriesContainer type'
            raise TypeError(e)

    @property
    def __parent(self):
        return self._NMObject__parent

    @property
    def __fxns(self):
        return self._NMObject__fxns

    # override, no super
    @property
    def content(self):
        return {'data': self.names}

    # override
    def new(self, name='default', samples=0, fill_value=0, dtype=DTYPE,
            noise=[], dims=DIMS, select=True, quiet=nmc.QUIET):
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        o = Data(self.__parent, name, self.__fxns, samples=samples,
                 fill_value=fill_value, dtype=DTYPE, noise=noise, dims=dims)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def kill(self, name, all_=False, ask=True, quiet=nmc.QUIET):
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
