# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np
from nm_container import Container
from nm_wave_prefix import WavePrefix


class Folder(Container):
    """
    NM Folder class
    Container for NM WavePrefixes and Data
    """
    def __init__(self, name):
        super().__init__(name)
        self.OBJECT_NAME_PREFIX = ""  # not used
        self.new("Record")

    def object_new(self, name):
        return WavePrefix(name)

    def instance_ok(self, obj):
        return isinstance(obj, WavePrefix)
