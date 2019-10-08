# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_utilities import name_ok
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
