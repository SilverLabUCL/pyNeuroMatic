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


class Data(NMObject):
    """
    NM Data class
    """

    def __init__(self, parent, name, xstart=0, xdelta=1, xlabel='', xunits='',
                 ylabel='', yunits=''):
        super().__init__(parent, name)
        self.__thedata = np.array([])
        self.__note_container = NoteContainer(self)
        self.__xstart = xstart
        self.__xdelta = xdelta
        self.__xlabel = xlabel  # e.g. 'Time'
        self.__xunits = xunits  # e.g. 'ms' for milliseconds
        self.__ylabel = ylabel  # e.g. 'Membrane current'
        self.__yunits = yunits  # e.g. 'nA' for nano-amperes

    @property
    def content(self):  # override, no super
        k = {'data': self.name}
        k.update(self.__note_container.content)
        return k

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


class DataContainer(Container):
    """
    Container for NM Data objects
    """
    __select_alert = ('NOT USED. See nm.dataseries.select, ' +
                      'nm.channel_select, nm.eset.select and nm.epoch_select.')

    def __init__(self, parent, name='NMDataContainer'):
        super().__init__(parent, name, nmc.DATA_PREFIX,
                         select_alert=self.__select_alert)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        return {'data': self.names}

    def new(self, name='default', select=True, quiet=False):  # override
        o = Data(self.__parent, 'temp')
        return super().new(name=name, select=select, quiet=quiet, nmobj=o)
