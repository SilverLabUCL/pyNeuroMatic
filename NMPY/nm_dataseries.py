# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_channel import ChannelContainer
from nm_eset import EpochSetContainer
import nm_utilities as nmu


class DataSeries(NMObject):
    """
    NM DataSeries class
    """

    def __init__(self, parent, name):
        # name is the prefix
        super().__init__(parent, name, rename=False)
        self.__thedata = []  # 2D list, i = chan #, j = seq #
        self.__channel_container = ChannelContainer(self)
        self.__eset_container = EpochSetContainer(self)
        self.__channel_select = ['A']
        self.__epoch_select = [0]
        self.__data_select = []
        self.__eset_init()
        # self.details()

    @property
    def content(self):  # override, no super
        k = {'dataseries': self.name}
        k.update(self.__channel_container.content)
        k.update({'channel_select': self.channel_select})
        k.update({'epochs': self.epoch_count})
        k.update({'epoch_select': self.epoch_select})
        k.update(self.__eset_container.content)
        return k

    @property
    def channel(self):
        return self.__channel_container

    @property
    def channel_count(self):
        return len(self.__thedata)

    def channel_list(self, includeAll=False):
        n = len(self.__thedata)
        if n == 0:
            return []
        if includeAll and n > 1:
            clist = ['ALL']
        else:
            clist = []
        for i in range(0, n):
            cc = nmu.channel_char(i)
            clist.append(cc)
        return clist

    def channel_ok(self, chan_char):
        if not chan_char:
            return False
        if chan_char.lower() == 'all':
            return True  # always ok
        for c in self.channel_list():
            if c.upper() == chan_char.upper():
                return True
        return False

    @property
    def channel_select(self):
        return self.__channel_select

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'ALL' or ['A', 'B']
        return self.channel_select_set(chan_char_list)

    def channel_select_set(self, chan_char_list, quiet=False):
        # e.g 'A', 'ALL' or ['A', 'B']
        if type(chan_char_list) is not list:
            chan_char_list = [chan_char_list]
        clist = []
        for cc in chan_char_list:
            if not self.channel_ok(cc):
                nmu.error('bad channel: ' + cc, quiet=quiet)
                return False
            clist.append(cc.upper())
        self.__channel_select = clist
        h = self.tree_path(history=True) + nmc.S0 + 'ch=' + str(clist)
        nmu.history(h, quiet=quiet)
        return True

    @property
    def all_channels(self):
        if not self.channel_select:
            return False
        for c in self.channel_select:
            if c.lower() == 'all':
                return True
        return False

    @property
    def eset(self):
        return self.__eset_container

    def eset_init(self):
        if not nmc.ESETS_LIST:
            return []
        r = []
        for s in nmc.ESETS_LIST:
            select = s.lower() == 'all'
            if self.__eset_container.new(s, select=select, quiet=True):
                r.append(s)
        if not self.__eset_container.select:
            self.__eset_container.select = nmc.ESETS_LIST[0]
        return r

    __eset_init = eset_init

    @property
    def epoch_count(self):  # epochs per channel
        if not self.__thedata:
            return [0]
        elist = []
        for cdata in self.__thedata:
            elist.append(len(cdata))
        return elist

    def epoch_ok(self, epoch_list):
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        emax = max(self.epoch_count)
        for e in epoch_list:
            if e >= 0 and e < emax:
                continue
            else:
                return False
        return True

    @property
    def epoch_select(self):
        return self.__epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch_list):
        return self.epoch_select_set(epoch_list)

    def epoch_select_set(self, epoch_list, quiet=False):
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        for e in epoch_list:
            if not self.epoch_ok(e):
                nmu.error('bad epoch: ' + str(e), quiet=quiet)
                return False
        self.__epoch_select = epoch_list
        h = self.tree_path(history=True) + nmc.S0 + 'epoch=' + str(epoch_list)
        nmu.history(h, quiet=quiet)
        return True

    @property
    def thedata(self):
        return self.__thedata

    def data_list(self, chan_char_list=['select'], epoch_list=[-1],
                  names=False, quiet=False):
        if type(chan_char_list) is not list:
            chan_char_list = [chan_char_list]
        for cc in chan_char_list:
            if cc.lower() == 'select':
                chan_char_list = self.__channel_select
                break
        all_chan = False
        for cc in chan_char_list:
            if cc.lower() == 'all':
                all_chan = True
                break
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        for e in epoch_list:
            if e == -1:
                epoch_list = self.__epoch_select
        dlist = []
        if all_chan:
            for cdata in self.__thedata:
                for e in epoch_list:
                    if e >= 0 and e < len(cdata):
                        if names:
                            dlist.append(cdata[e].name)
                        else:
                            dlist.append(cdata[e])
                    else:
                        nmu.error('bad epoch: ' + str(e), quiet=quiet)
                        return []
            return dlist
        for cc in chan_char_list:
            cn = nmu.channel_num(cc)
            if cn >= 0 and cn < len(self.__thedata):
                cdata = self.__thedata[cn]
                for e in epoch_list:
                    if e >= 0 and e < len(cdata):
                        if names:
                            dlist.append(cdata[e].name)
                        else:
                            dlist.append(cdata[e])
                    else:
                        nmu.error('bad epoch: ' + str(e), quiet=quiet)
                        return []
            else:
                nmu.error('bad channel: ' + cc, quiet=quiet)
                return []
        return dlist

    @property
    def data_select(self):
        self.__data_select = self.get_selected()  # update
        return self.__data_select

    @property
    def data_select_names(self):
        return self.get_selected(names=True)

    def get_selected(self, names=False, quiet=False):
        channels = len(self.__thedata)
        cs = self.__channel_select
        ss = self.eset.select.name
        eset = self.eset.select.theset
        setx = self.eset.get('SetX').theset
        if channels == 0:
            return []
        if ss.lower() == 'all':
            all_epochs = True
        else:
            all_epochs = False
            eset = eset.difference(setx)
        if self.all_channels and channels > 1:
            clist = []
            for cdata in self.__thedata:
                dlist = []
                for d in cdata:
                    if all_epochs:
                        if d not in setx:
                            if names:
                                dlist.append(d.name)
                            else:
                                dlist.append(d)
                    elif d in eset:
                        if names:
                            dlist.append(d.name)
                        else:
                            dlist.append(d)
                clist.append(dlist)
            return clist
        dlist = []
        cnum_list = []
        if channels == 1:
            cnum_list.append(0)
        else:
            for c in cs:
                cnum = nmu.channel_num(c)
                if cnum >= 0 and cnum < channels:
                    cnum_list.append(cnum)
        for c in cnum_list:
            chan = self.__thedata[c]
            for d in chan:
                if all_epochs:
                    if d not in setx:
                        if names:
                            dlist.append(d.name)
                        else:
                            dlist.append(d)
                elif d in eset:
                    if names:
                        dlist.append(d.name)
                    else:
                        dlist.append(d)
        return dlist


class DataSeriesContainer(Container):
    """
    NM Container for DataSeries objects
    """

    def __init__(self, parent, data_container, name='NMDataSeriesContainer'):
        super().__init__(parent, name, prefix='', select_new=True,
                         rename=False, duplicate=False)
        self.__parent = parent
        self.__data_container = data_container

    @property
    def content(self):  # override, no super
        k = {'dataseries': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'dataseries_select': s})
        return k

    def new(self, name='', select=True, quiet=False):  # override, no super
        if not name:
            nmu.error('must specify name', quiet=quiet)
            return None
        prefix = name  # name is actually a prefix
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix), quiet=quiet)
            return None
        pexists = self.exists(prefix)
        if pexists:
            p = self.get(prefix, quiet=quiet)
        else:
            p = DataSeries(self.__parent, prefix)
            self.add(p, select=select, quiet=True)
        #if select:
            #self.select_set(prefix, quiet=True)
        t = ''
        if not pexists:
            t = 'created'
        if select:
            if len(t) == 0:
                t = 'selected'
            else:
                t += '/selected'
        nmu.history(t + nmc.S0 + p.tree_path(history=True), quiet=quiet)
        self.search(p)
        return p

    def search(self, p, quiet=False):
        if not p:
            return False
        prefix = p.name
        foundsomething = False
        thedata = self.__data_container.get_all()
        htxt = []
        for i in range(0, 25):  # try prefix+chan+seq format
            cc = nmu.channel_char(i)
            olist = nmu.get_items(thedata, prefix, chan_char=cc)
            if len(olist) > 0:
                p.thedata.append(olist)
                foundsomething = True
                p.channel.new(quiet=True)
                htxt.append('ch=' + cc + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:  # try without chan
            olist = nmu.get_items(thedata, prefix)
            if len(olist) > 0:
                p.thedata.append(olist)
                foundsomething = True
                p.channel.new(quiet=True)
                htxt.append('n=' + str(len(olist)))
        if not foundsomething:
            a = 'failed to find data with prefix ' + nmu.quotes(prefix)
            nmu.alert(a, quiet=quiet)
        for h in htxt:
            tp = p.tree_path(history=True)
            nmu.history('found' + nmc.S0 + tp + ', ' + h, quiet=quiet)
        return True

    def make(self, prefix='default', channels=1, epochs=3, samples=10,
             fill_value=0, noise=[], xstart=0, xdelta=1, xlabel='',
             xunits='', ylabel='', yunits='', select=True, quiet=False):
        n_ok = False
        if len(noise) == 2:
            n_mean = noise[0]
            n_stdv = abs(noise[1])
            if not np.isinf(n_mean * n_stdv) and not np.isnan(n_mean * n_stdv):
                n_ok = True
        if not prefix or prefix.casefold() == 'default':
            prefix = self.__data_container.prefix
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix), quiet=quiet)
            return False
        if channels <= 0 or epochs <= 0 or samples <= 0:
            return False
        seq_start = []
        for ci in range(0, channels):  # look for existing data
            cc = nmu.channel_char(ci)
            si = self.__data_container.name_next_seq(prefix + cc, quiet=quiet)
            if si >= 0:
                seq_start.append(si)
        ss = max(seq_start)
        se = ss + epochs
        htxt = []
        tree_path = ''
        for ci in range(0, channels):
            cc = nmu.channel_char(ci)
            for j in range(ss, se):
                name = prefix + cc + str(j)
                ts = self.__data_container.new(name, quiet=True)
                tree_path = self.__parent.tree_path(history=True)
                if not ts:
                    a = 'failed to create ' + nmu.quotes(name)
                    nmu.alert(a, quiet=quiet)
                if n_ok:
                    ts.thedata = np.random.normal(n_mean, n_stdv, samples)
                else:
                    ts.thedata = np.full(samples, fill_value)
            htxt.append('ch=' + cc + ', ep=' + str(ss) + '-' + str(se-1))
        for h in htxt:
            path = prefix
            if len(tree_path) > 0:
                path = tree_path + "." + path
            nmu.history('created' + nmc.S0 + path + ', ' + h, quiet=quiet)
        if select:
            self.new(prefix, quiet=quiet)
        return True
