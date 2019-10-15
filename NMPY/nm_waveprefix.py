# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import Container
from nm_channel import ChannelContainer
from nm_waveset import WaveSetContainer
from nm_utilities import channel_char
from nm_utilities import channel_num
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import alert
from nm_utilities import error
from nm_utilities import history
from nm_utilities import get_items


class WavePrefix(object):
    """
    NM WavePrefix class
    """

    def __init__(self, name):
        self.__name = name  # name is actually a prefix
        self.__thewaves = []  # 2D matrix, i = channel #, j = wave #
        self.__channel_container = ChannelContainer()
        self.__waveset_container = WaveSetContainer(self.__thewaves)
        self.__waveset_container.new("All", select=True, quiet=True)
        self.__waveset_container.new("Set1", select=False, quiet=True)
        self.__waveset_container.new("Set2", select=False, quiet=True)
        self.__waveset_container.new("SetX", select=False, quiet=True)
        self.__chanselect = 'A'
        self.__waveselect = 0
        # self.print_details()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        error("cannot rename WavePrefix objects")

    @property
    def channel_container(self):
        return self.__channel_container

    @property
    def waveset_container(self):
        return self.__waveset_container

    @property
    def numchannels(self):
        if not self.__thewaves:
            return 0
        return len(self.__thewaves)

    @property
    def chanlist(self):
        if not self.__thewaves:
            return []
        clist = []
        channels = len(self.__thewaves)
        for i in range(0, channels):
            c = channel_char(i)
            clist.append(c)
        return clist

    @property
    def numwaves(self):  # waves per channel
        if not self.__thewaves:
            return 0
        nlist = []
        for row in self.__thewaves:
            nlist.append(len(row))
        return nlist

    @property
    def channel_select(self):
        return self.__chanselect

    @channel_select.setter
    def channel_select(self, chan_char):  # e.g 'A'
        if nmconfig.CHAN_LIST.count(chan_char) == 1:
            self.__chanselect = chan_char
            return True
        else:
            error("unknown channel letter: " + chan_char)
            return False

    @property
    def select(self):
        return [self.__chanselect, self.__waveselect]

    @select.setter
    def select(self, chan_wave_list):  # e.g ['A', 3]
        chan_char = chan_wave_list[0]
        wave = chan_wave_list[1]
        clist = nmconfig.CHAN_LIST
        if clist.count(chan_char) == 1:
            self.__chanselect = chan_char
        else:
            error("unknown channel letter: " + chan_char)
            return False
        chan_num = channel_num(chan_char)
        # nwaves = len(self.__thewaves[chan])
        nwaves = 0
        if wave >= 0 and wave < nwaves:
            self.__waveselect = wave
        else:
            error("wave # out of range: " + str(wave))
            return False
        return True

    @property
    def thewaves(self):
        return self.__thewaves

    def details(self):
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
        self.__wave_container = wave_container

    def object_new(self, name):  # override, do not call super
        return WavePrefix(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, WavePrefix)

    def name_next(self):  # override, do not call super
        error("WavePrefix objects do not have names with a common prefix")
        return ""

    def new(self, name="", select=True):  # override, do not call super
        prefix = name  # name is actually a prefix
        if not name_ok(prefix):
            error("bad prefix " + quotes(prefix))
            return None
        p = WavePrefix(prefix)
        foundsomething = False
        thewaves = self.__wave_container.getAll()
        for i in range(0, 25):  # try prefix+chan+seq format
            cc = channel_char(i)
            olist = get_items(thewaves, prefix, chan_char=cc)
            if len(olist) > 0:
                p.thewaves.append(olist)
                foundsomething = True
                history(prefix + ", Ch " + cc + ", waves=" + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:  # try without chan
            olist = get_items(thewaves, prefix)
            if len(olist) > 0:
                p.thewaves.append(olist)
                foundsomething = True
                history(prefix + ", Ch " + cc + ", waves=" + str(len(olist)))
        if not foundsomething:
            alert("failed to find waves beginning with " + quotes(prefix))
            return None
        self.add(p)
        # print("channels=" + str(p.numchannels))
        # print("waves=" + str(p.numwaves))
        return p

    def rename(self, name, newname):  # override, do not call super
        error("cannot rename WavePrefix objects")
        return False

    def duplicate(self, name, newname, select=False):  # override
        error("cannot duplicate WavePrefix objects")
        return None  # not sued
