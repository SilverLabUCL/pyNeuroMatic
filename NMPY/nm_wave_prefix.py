# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_channel import ChannelContainer
from nm_utilities import quotes
from nm_utilities import error

CHANNEL_PREFIX = "Chan"


class WavePrefix(object):
    """
    NM WavePrefix class
    """

    def __init__(self, name):
        self.__name = name
        self.__channel = ChannelContainer(CHANNEL_PREFIX)
        self.wave_names = []  # 2D matrix, i = channel #, j = wave #
        self.channels = 0
        # self.waves = 0
        # self.chanNum = 0
        # self.wavenum = 0
        # self.wave_names_mock(channels=2, waves=5)
        # self.wave_names_search()
        # sself.print_details()

    @property
    def name(self):
        return self.__name

    @property
    def channel(self):
        return self.__channel

    def wave_names_mock(self, channels, waves):
        if channels == 0 or waves == 0:
            return 0
        self.channels = channels
        self.waves = waves
        chan_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for i in range(0, channels):
            chan_char = chan_chars[i]
            self.channel.new("")
            channel_wave_names = []
            for j in range(0, waves):
                channel_wave_names.append(self.name + chan_char + str(j))
            self.wave_names.append(channel_wave_names)

    def wave_names_search(self, channels=0):
        # search for waves
        pass

    def print_details(self) -> None:
        print("WavePrefix = " + quotes(self.name))
        print("channels = " + str(self.channels))
        # print("waves = " + str(self.waves))
        # print("wave list = " + str(self.wave_names))


class WavePrefixContainer(Container):
    """
    Container for NM WavePrefixes
    """

    def object_new(self, name):
        return WavePrefix(name)

    def instance_ok(self, obj):
        return isinstance(obj, WavePrefix)

    def rename(self, name, newname):
        error("cannot rename wave-prefix object")
        return False
