# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from NM_Utilities import quotes

# things to include: Sets, Groups

class WavePrefix(object):
    """
    NM Wave Prefix class
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self.wave_names = []  # 2D matrix, i = channel #, j = wave #
        self.num_channels = 0
        self.num_waves = 0
        self.channel_select = 0
        self.wave_select = 0
        self.wave_names_mock(num_channels=2, num_waves=5)
        self.print_details()

    def wave_names_mock(self, num_channels, num_waves):
        if num_channels == 0 or num_waves == 0:
            return 0
        self.num_channels = num_channels
        self.num_waves = num_waves
        channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for i in range(0, num_channels):
            channel = channels[i]
            channel_wave_names = []
            for j in range(0, num_waves):
                channel_wave_names.append(self.prefix + channel + str(j))
            self.wave_names.append(channel_wave_names)

    def print_details(self) -> None:
        print("WavePrefix = " + quotes(self.prefix))
        print("# channels = " + str(self.num_channels))
        print("# waves = " + str(self.num_waves))
        print("wave list = " + str(self.wave_names))
