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
    Include: time-series properties (xstart, dx...) and notes
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__thedata = np.array([])
        self.__note_container = NoteContainer(self)
        # self.__thedata = np.array([], dtype=np.float64)

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
    __select_alert = ('NOT USED. See nm.dataprefix.select, ' +
                      'nm.channel_select, nm.eset.select and nm.epoch_select.')

    def __init__(self, parent, name='NMDataContainer'):
        super().__init__(parent, name, nmc.DATA_PREFIX,
                         select_alert=self.__select_alert)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        return {'data': self.names}

    def object_new(self, name):  # override, no super
        return Data(self.__parent, name)

    def make(self, prefix='default', channels=1, epochs=3, samples=10,
             noise=False, select=True, quiet=False):
        noise_mu, noise_sigma = 0, 0.1  # mean and standard deviation
        if not prefix or prefix.casefold() == 'default':
            prefix = self.prefix
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix), quiet=quiet)
            return False
        if channels <= 0 or epochs <= 0 or samples <= 0:
            return False
        seq_start = []
        for ci in range(0, channels):  # look for existing data
            cc = nmu.channel_char(ci)
            si = self.name_next_seq(prefix + cc, quiet=quiet)
            if si >= 0:
                seq_start.append(si)
        ss = max(seq_start)
        se = ss + epochs
        htxt = []
        tree_path = ''
        for ci in range(0, channels):
            cc = nmu.channel_char(ci)
            for j in range(ss, se):
                name = prefix + cc + str(j)
                ts = self.new(name, quiet=True)
                tree_path = self.__parent.tree_path(history=True)
                if not ts:
                    a = 'failed to create ' + nmu.quotes(name)
                    nmu.alert(a, quiet=quiet)
                if noise:
                    ts.thedata = np.random.normal(noise_mu, noise_sigma,
                                                  samples)
                else:
                    # ts.thedata = np.zeros(samples)
                    ts.thedata = np.empty(samples)
            htxt.append('ch=' + cc + ', ep=' + str(ss) + '-' + str(se-1))
        for h in htxt:
            path = prefix
            if len(tree_path) > 0:
                path = tree_path + "." + path
            nmu.history('created' + nmc.S0 + path + ', ' + h, quiet=quiet)
        if select:
            self.__parent.dataprefix.new(prefix, quiet=quiet)
        return True
