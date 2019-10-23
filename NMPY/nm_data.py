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
from nm_utilities import error
import numpy as np


class Data(NMObject):
    """
    NM Data class
    Include: time-series properties (xstart, dx...) and notes
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__thedata = np.array([], dtype=np.float64)

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

    def make(self, prefix="", channels=1, epochs=5, samples=10, noise=False,
             select=True, quiet=False):
        if not prefix:
            prefix = self.prefix
        if not name_ok(prefix):
            return error("bad prefix " + quotes(prefix))
        if channels <= 0 or epochs <= 0 or samples <= 0:
            return False
        mu, sigma = 0, 0.1  # mean and standard deviation
        for i in range(0, channels):
            cc = channel_char(i)
            for j in range(0, epochs):
                name = self.name_next(prefix=prefix + cc)
                ts = self.new(name, quiet=True)
                if ts and noise:
                    ts.thedata = np.random.normal(mu, sigma, samples)
        if select:
            self.parent.dataprefix_container.new(prefix)
        return True
