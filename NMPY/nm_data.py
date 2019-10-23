# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import NMObject
from nm_container import Container
from nm_utilities import channel_char
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import history
from nm_utilities import error
import numpy as np


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
    def __init__(self, parent):
        super().__init__(parent, prefix=nmconfig.DATA_PREFIX)

    def object_new(self, name):  # override, do not call super
        return Data(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Data)

    def make(self, prefix='default', channels=1, epochs=3, samples=10,
             noise=False, select=True, quiet=False):
        noise_mu, noise_sigma = 0, 0.1  # mean and standard deviation
        if not prefix or prefix.casefold() == 'default':
            prefix = self.prefix
        if not name_ok(prefix):
            return error('bad prefix ' + quotes(prefix))
        if channels <= 0 or epochs <= 0 or samples <= 0:
            return False
        seq_start = []
        for ci in range(0, channels):
            sn = self.name_next_seq(prefix + channel_char(ci))
            if sn >= 0:
                seq_start.append(sn)
        ss = max(seq_start)
        ss = max(ss, 0)
        for ci in range(0, channels):
            seq = []
            cc = channel_char(ci)
            for j in range(ss, ss+epochs):
                name = prefix + cc + str(j)
                ts = self.new(name, quiet=True)
                if ts:
                    seq.append(j)
                    if noise:
                        ts.thedata = np.random.normal(noise_mu, noise_sigma, 
                                                      samples)
                    else:
                        ts.thedata = np.zeros(samples)
            if not quiet:
                history(prefix + ", Ch " + cc + ", #=" + str(seq))
        if select:
            self.parent.dataprefix_container.new(prefix)
        return True
