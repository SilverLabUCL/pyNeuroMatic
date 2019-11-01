# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Data(NMObject):
    """
    NM Data class
    Include: time-series properties (xstart, dx...) and notes
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__thedata = np.array([])
        # self.__thedata = np.array([], dtype=np.float64)

    @property
    def thedata(self):
        return self.__thedata

    @thedata.setter
    def thedata(self, np_array):
        self.__thedata = np_array


class DataContainer(Container):
    """
    Container for NM Data objects
    """
    def __init__(self, parent, name):
        super().__init__(parent, name, nmc.DATA_PREFIX)

    def object_new(self, name):  # override, do not call super
        return Data(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Data)

    @property
    def select(self):  # override
        nmu.alert('Use nm.dataprefix.select')
        return super().select

    @select.setter
    def select(self, name, quiet=False):  # override, do not call super
        nmu.alert('Use nm.dataprefix.select')
        return False

    def make(self, prefix='default', channels=1, epochs=3, samples=10,
             noise=False, select=True, quiet=False):
        noise_mu, noise_sigma = 0, 0.1  # mean and standard deviation
        if not prefix or prefix.casefold() == 'default':
            prefix = self.prefix
        if not nmu.name_ok(prefix):
            return nmu.error('bad prefix ' + nmu.quotes(prefix))
        if channels <= 0 or epochs <= 0 or samples <= 0:
            return False
        seq_start = []
        for ci in range(0, channels):  # look for existing data
            si = self.name_next_seq(prefix + nmu.channel_char(ci))
            if si >= 0:
                seq_start.append(si)
        ss = max(seq_start)
        se = ss + epochs
        htxt = []
        for ci in range(0, channels):
            cc = nmu.channel_char(ci)
            for j in range(ss, se):
                name = prefix + cc + str(j)
                ts = self.new(name, quiet=True)
                tree_path = ts.parent.tree_path
                if not ts:
                    nmu.alert('failed to create ' + nmu.quotes(name))
                if noise:
                    ts.thedata = np.random.normal(noise_mu, noise_sigma,
                                                  samples)
                else:
                    ts.thedata = np.zeros(samples)
            htxt.append('ch=' + cc + ', ep=' + str(ss) + '-' + str(se-1))
        if not quiet:
            for h in htxt:
                path = prefix
                if nmc.TREE_PATH_LONG:
                    path = tree_path + "." + path
                nmu.history('created' + nmc.S0 + path + ', ' + h)
        if select:
            self.parent.dataprefix_container.new(prefix, quiet=quiet)
        return True
