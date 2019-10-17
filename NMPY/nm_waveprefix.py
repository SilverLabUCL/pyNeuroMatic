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
        self.__waves_all = []  # 2D matrix, i = channel #, j = wave #
        self.__waves_selected = []  # 2D matrix, i = channel #, j = wave #
        self.__channel_container = ChannelContainer()
        self.__waveset_container = WaveSetContainer(self.__waves_all)
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
    def waveset_container(self):
        return self.__waveset_container

    @property
    def waveset_select(self):
        if self.__waveset_container:
            return self.__waveset_container.select
        return None

    @property
    def channel_container(self):
        return self.__channel_container

    @property
    def channel_count(self):
        if not self.__waves_all:
            return 0
        return len(self.__waves_all)

    def channel_list(self, includeAll=False):
        n = self.channel_count
        if n == 0:
            return []
        if includeAll and n > 1:
            clist = ['All']
        else:
            clist = []
        for i in range(0, n):
            cc = channel_char(i)
            clist.append(cc)
        return clist

    def channel_ok(self, chan_char):
        if not chan_char:
            return False
        if chan_char.casefold() == "all":
            return True  # always ok
        if self.channel_list().count(chan_char) == 1:
            return True
        return False

    @property
    def channel_select(self):
        return self.__chanselect

    @channel_select.setter
    def channel_select(self, chan_char):  # e.g 'A' or 'All'
        if self.channel_ok(chan_char):
            self.__chanselect = chan_char
            history("selected channel " + chan_char)
            return True
        error("bad channel select: " + chan_char)
        return False

    @property
    def wave_count(self):  # waves per channel
        if not self.__waves_all:
            return [0]
        nlist = []
        for row in self.__waves_all:
            nlist.append(len(row))
        return nlist

    def wave_ok(self, wave_num):
        if wave_num >= 0 and wave_num < max(self.wave_count):
            return True
        return False

    @property
    def wave_select(self):
        return self.__waveselect

    @wave_select.setter
    def wave_select(self, wave_num):
        if self.wave_ok(wave_num):
            self.__waveselect = wave_num
            history("selected wave #" + str(wave_num))
            return True
        error("bad wave number: " + str(wave_num))
        return False

    @property
    def select(self):
        c = self.__chanselect
        w = self.__waveselect
        if self.__waveset_container and self.__waveset_container.select:
            ws = self.__waveset_container.select.name
        else:
            ws = "None"
        s = {}
        s['channel'] = c
        s['wave'] = w
        s['waveset'] = ws
        return s

    @property
    def thewaves(self):
        return self.__waves_all

    @property
    def details(self):
        print("WavePrefix = " + quotes(self.name))
        print("channels = " + str(self.channel_count))
        print("waves = " + str(self.wave_count))
        # print("wave list = " + str(self.wave_names))

    def wave_list(self, chan_char='selected', wave_num=-1):
        if not chan_char or chan_char.casefold() == 'selected':
            chan_char = self.__chanselect
        if wave_num == -1:
            wave_num = self.__waveselect
        wlist = []
        if chan_char.casefold() == 'all':
            for chan in self.__waves_all:
                if wave_num >= 0 and wave_num < len(chan):
                    wlist.append(chan[wave_num])
            return wlist
        chan_num = channel_num(chan_char)
        if chan_num >= 0 and chan_num < len(self.__waves_all):
            chan = self.__waves_all[chan_num]
            if wave_num >= 0 and wave_num < len(chan):
                return chan[wave_num]
        return []


class WavePrefixContainer(Container):
    """
    NM Container for WavePrefix objects
    """

    def __init__(self, wave_container):
        super().__init__(prefix="")
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
                p.channel_container.new(quiet=True)
                history(prefix + ", Ch " + cc + ", waves=" + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:  # try without chan
            olist = get_items(thewaves, prefix)
            if len(olist) > 0:
                p.thewaves.append(olist)
                foundsomething = True
                p.channel_container.new(quiet=True)
                history(prefix + ", Ch " + cc + ", waves=" + str(len(olist)))
        if not foundsomething:
            alert("failed to find waves beginning with " + quotes(prefix))
            return None
        self.add(p)
        # print("channels=" + str(p.channel_count))
        # print("waves=" + str(p.wave_count))
        return p

    def rename(self, name, newname):  # override, do not call super
        error("cannot rename WavePrefix objects")
        return False

    def duplicate(self, name, newname, select=False):  # override
        error("cannot duplicate WavePrefix objects")
        return None  # not sued
