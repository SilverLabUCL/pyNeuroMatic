# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_utilities import quotes


class WavePrefix(Container):
    """
    NM Wave Prefix class
    Container for NM Channels
    Sets and Groups
    """

    def __init__(self, name):
        super().__init__(name)
        self.OBJECT_NAME_PREFIX = ""  # not used
        self.wave_names = []  # 2D matrix, i = channel #, j = wave #
        self.channels = 0
        self.waves = 0
        self.channel = 0
        self.wavenum = 0
        self.wave_names_mock(channels=2, waves=5)
        # self.wave_names_search()
        self.print_details()

    def object_new(self, name):
        return Channel(name)

    def instance_ok(self, obj):
        return isinstance(obj, Channel)

    def wave_names_mock(self, channels, waves):
        if channels == 0 or waves == 0:
            return 0
        self.channels = channels
        self.waves = waves
        chan_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for i in range(0, channels):
            chan_char = chan_chars[i]
            channel_wave_names = []
            for j in range(0, waves):
                channel_wave_names.append(self.name + chan_char + str(j))
            self.wave_names.append(channel_wave_names)

    def wave_names_search(self, channels = 0):
        # search for data array names that begin with prefix (e.g. RecordA0, RecordA1...)
        pass
    
    def print_details(self) -> None:
        print("WavePrefix = " + quotes(self.name))
        print("channels = " + str(self.channels))
        print("waves = " + str(self.waves))
        print("wave list = " + str(self.wave_names))
