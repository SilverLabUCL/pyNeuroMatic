# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_note import NoteContainer
import nm_utilities as nmu

DIMS = {'xstart': 0, 'xdelta': 1, 'xlabel': '', 'xunits': '', 'ylabel': '',
        'yunits': ''}
DTYPE = np.dtype(float)


class Data(NMObject):
    """
    NM Data class
    """

    def __init__(self, manager, parent, name, fxns, samples=0,
                 fill_value=0, dtype=DTYPE, noise=[], dims=DIMS):
        super().__init__(manager, parent, name, fxns)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.make(samples=samples, fill_value=fill_value, dtype=dtype,
                  noise=noise)
        n = NoteContainer(manager, self, 'NMNoteContainer', fxns)
        self.__note_container = n
        self.__xdata = None
        self.__xstart = 0
        self.__xdelta = 1
        self.__xlabel = ''
        self.__xunits = ''
        self.__ylabel = ''
        self.__yunits = ''
        self.dims = dims

    @property  # override, no super
    def content(self):
        k = {'data': self.name}
        k.update(self.__note_container.content)
        return k

    @property
    def data(self):
        folder = self.__parent
        return folder.data

    @property
    def thedata(self):
        return self.__thedata

    @thedata.setter
    def thedata(self, np_array):
        self.__thedata = np_array

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
        self.__xdata = xdata
        return True

    @property
    def xstart(self):
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        if np.isinf(xstart) or np.isnan(xstart):
            return False
        self.__xstart = xstart
        return True

    @property
    def xdelta(self):
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        if np.isinf(xdelta) or np.isnan(xdelta):
            return False
        self.__xdelta = xdelta
        return True

    @property
    def xlabel(self):
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        if isinstance(xlabel, str):
            self.__xlabel = xlabel
            return True
        return False

    @property
    def xunits(self):
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        if isinstance(xunits, str):
            self.__xunits = xunits
            return True
        return False

    @property
    def ylabel(self):
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        if isinstance(ylabel, str):
            self.__ylabel = ylabel
            return True
        return False

    @property
    def yunits(self):
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        if isinstance(yunits, str):
            self.__yunits = yunits
            return True
        return False

    def make(self, samples=0, fill_value=0, dtype=DTYPE, noise=[],
             quiet=nmc.QUIET):
        if not nmu.num_ok(samples, no_neg=True):
            self.__error('bad samples argument: ' + str(samples), quiet=quiet)
            return False
        if isinstance(noise, list) and len(noise) == 2:
            n_mean = noise[0]
            n_stdv = noise[1]
            if not nmu.num_ok(n_mean) or not nmu.num_ok(n_stdv, no_neg=True):
                self.__error('bad noise argument: ' + str(noise), quiet=quiet)
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

    def __init__(self, manager, parent, name, fxns):
        o = Data(manager, parent, 'temp', fxns)
        super().__init__(manager, parent, name, fxns, nmobj=o,
                         prefix=nmc.DATA_PREFIX)
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        return {'data': self.names}

    # override
    def new(self, name='default', samples=0, fill_value=0, dtype=DTYPE,
            noise=[], dims=DIMS, select=True, quiet=nmc.QUIET):
        o = Data(self.__manager, self.__parent, name, self.__fxns,
                 samples=samples, fill_value=fill_value, dtype=DTYPE,
                 noise=noise, dims=dims)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)
