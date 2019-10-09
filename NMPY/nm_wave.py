# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_utilities import name_ok
from nm_utilities import chan_char
import numpy as np

WAVE_PREFIX = "Record"

class Wave(object):
    """
    NM Wave class
    Include: wave properties (xstart, dx...) and wave notes
    """

    def __init__(self, name):
        self.__name = name
        self.__data = np.array([], dtype=np.float64)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name
            
    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def numwaves(self):
        return len(self.__data)


class WaveContainer(Container):
    """
    Container for NM Experimnents
    """
    def __init__(self):
        super().__init__()
        self.prefix = WAVE_PREFIX

    def object_new(self, name):
        return Wave(name)

    def instance_ok(self, obj):
        return isinstance(obj, Wave)
    
    def make(self, prefix="", numchan=1, numwaves=5, points=10):
        if not prefix:
            prefix = self.prefix
        mu, sigma = 0, 0.1 # mean and standard deviation
        for i in range(0, numchan):
            cc = chan_char(i)
            for j in range(0, numwaves):
                wname = prefix + cc + str(j)
                w = self.new(wname)
                if w:
                    w.data = np.random.normal(mu, sigma, points)
                    print(w.data)
