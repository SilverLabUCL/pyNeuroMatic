# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

import nm_configs as nmc
from nm_container import NMObject
from nm_dimensions import Dimensions
from nm_container import Container
from nm_note import NoteContainer
import nm_utilities as nmu


class Data(NMObject):
    """
    NM Data class
    """

    def __init__(self, parent, name, dimensions=None):
        super().__init__(parent, name)
        self.__thedata = np.array([])
        self.__note_container = NoteContainer(self)
        if dimensions:
            self.__dimensions = dimensions
        else:
            self.__dimensions = Dimensions()

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


class DataContainer(Container):
    """
    Container for NM Data objects
    """
    __select_alert = ('NOT USED. See nm.dataseries.select, ' +
                      'nm.channel_select, nm.eset.select and nm.epoch_select.')

    def __init__(self, parent, name='NMDataContainer'):
        o = Data(parent, 'temp')
        super().__init__(parent, name=name, nmobj=o, prefix=nmc.DATA_PREFIX)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        return {'data': self.names}

    """
    @property
    def select(self):
        nmu.alert(self.__select_alert)
        return super().select

    @select.setter
    def select(self, name):
        nmu.alert(self.__select_alert)
        return self.select_set(name)
    """

    def new(self, name='default', dimensions=None, select=True, quiet=False):
        # override
        o = Data(self.__parent, name, dimensions=dimensions)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)
