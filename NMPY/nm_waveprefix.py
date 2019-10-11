# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_channel import ChannelContainer
from nm_utilities import chan_char
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error
from nm_utilities import history
from nm_utilities import get_items


class WavePrefix(object):
    """
    NM WavePrefix class
    """

    def __init__(self, name):
        self.__name = name
        self.__channel = ChannelContainer()
        self.__waves = []  # 2D matrix, i = channel #, j = wave #
        self.__channels = 0
        # self.waves = 0
        # self.chanNum = 0
        # self.wavenum = 0
        # self.wave_names_mock(channels=2, waves=5)
        # self.wave_names_search()
        # sself.print_details()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not name_ok(name):
            return error("bad name " + quotes(name))
        self.__name = name
        return True

    @property
    def channel(self):
        return self.__channel

    @property
    def waves(self):
        return self.__waves

    @waves.setter
    def waves(self, waves):
        self.__waves = waves

    def wave_names_mock(self, nchan, nwaves):
        if nchan == 0 or nwaves == 0:
            return False
        self.channels = nchan
        self.waves = nwaves
        for i in range(0, nchan):
            cc = chan_char(i)
            self.channel.new("Chan" + cc)
            channel_wave_names = []
            for j in range(0, nwaves):
                channel_wave_names.append(self.name + cc + str(j))
            self.wave_names.append(channel_wave_names)
        return True

    def print_details(self) -> None:
        print("WavePrefix = " + quotes(self.name))
        print("channels = " + str(self.channels))
        # print("waves = " + str(self.waves))
        # print("wave list = " + str(self.wave_names))


class WavePrefixContainer(Container):
    """
    Container for NM WavePrefixes
    """

    def __init__(self, wave_container):
        super().__init__()
        self.prefix = ""  # not used
        self.__thewaves = wave_container.getAll()

    def object_new(self, name):  # override, do not call super
        return WavePrefix(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, WavePrefix)

    def name_next(self):  # override, do not call super
        error("WavePrefix objects do not have common names")
        return ""

    def new(self, name="", select=True):  # override, do not call super
        prefix = name  # name is actually a prefix
        if not name_ok(prefix):
            error("bad prefix " + quotes(prefix))
            return None
        p = WavePrefix(prefix)
        foundsomething = False
        for i in range(0, 25):  # look for names with channel-seq format
            cc = chan_char(i)
            olist = get_items(self.__thewaves, prefix, chan_char=cc)
            if len(olist) > 0:
                p.waves.append(olist)
                foundsomething = True
                history(prefix + ", Ch " + cc + ", waves = " + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            olist = get_items(self.__thewaves, prefix)  # try without chanchar
            if len(olist) > 0:
                p.waves.append(olist)
                foundsomething = True
                history(prefix + ", Ch " + cc + ", waves = " + str(len(olist)))
        if foundsomething:
            self.add(p)
            return p
        error("failed to find waves beginning with " + quotes(prefix))
        return None
    
    def rename(self, name, newname):  # override, do not call super
        error("cannot rename WavePrefix objects")
        return False
    
    def duplicate(self, name, newname, select=False):  # override
        error("cannot duplicate WavePrefix objects")
        return None  # not sued
