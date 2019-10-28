# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import NMObject
from nm_container import Container
from nm_channel import ChannelContainer
from nm_epoch_set import EpochSetContainer
from nm_utilities import channel_char
from nm_utilities import channel_num
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import alert
from nm_utilities import error
from nm_utilities import history
from nm_utilities import get_items


class DataPrefix(NMObject):
    """
    NM DataPrefix class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)  # name is actually a prefix
        self.__thedata = []  # 2D list, i = chan #, j = seq #
        self.__channel_container = ChannelContainer(self, 'NMChannels')
        self.__eset_container = EpochSetContainer(self, 'NMEpochSets')
        for s in nmconfig.ESETS_DEFAULT:
            select = s.upper() == 'ALL'
            self.__eset_container.new(s, select=select, quiet=True)
        self.__channel_select = ['A']
        self.__epoch_select = 0
        self.__data_select = []
        # self.print_details()

    @property
    def eset_container(self):
        return self.__eset_container

    @property
    def eset_select(self):
        if self.__eset_container:
            return self.__eset_container.select
        return None

    @property
    def channel_container(self):
        return self.__channel_container

    @property
    def channel_count(self):
        if not self.__thedata:
            return 0
        return len(self.__thedata)

    def channel_list(self, includeAll=False):
        n = self.channel_count
        if n == 0:
            return []
        if includeAll and n > 1:
            clist = ['ALL']
        else:
            clist = []
        for i in range(0, n):
            cc = channel_char(i)
            clist.append(cc)
        return clist

    def channel_ok(self, chan_char):
        if not chan_char:
            return False
        if chan_char.upper() == 'ALL':
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
        if type(chan_char_list) is not list:
            chan_char_list = [chan_char_list]
        clist = []
        for cc in chan_char_list:
            if not self.channel_ok(cc):
                error('bad channel: ' + cc)
                return False
            clist.append(cc.upper())
        self.__channel_select = clist
        history(self.tree_path + nmconfig.HD0 + 'ch=' + str(clist))
        return True

    @property
    def all_channels(self):
        if not self.channel_select:
            return False
        for c in self.channel_select:
            if c.upper() == 'ALL':
                return True
        return False

    @property
    def epoch_count(self):  # epochs per channel
        if not self.__thedata:
            return [0]
        elist = []
        for chan in self.__thedata:
            elist.append(len(chan))
        return elist

    def epoch_ok(self, epoch):
        if epoch >= 0 and epoch < max(self.epoch_count):
            return True
        return False

    @property
    def epoch_select(self):
        return self.__epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch):
        if self.epoch_ok(epoch):
            self.__epoch_select = epoch
            history(self.tree_path + nmconfig.HD0 + 'epoch=' + str(epoch))
            return True
        error('bad epoch number: ' + str(epoch))
        return False

    @property
    def select(self):
        s = {}
        s['channel'] = self.__channel_select
        s['epoch'] = self.__epoch_select
        if self.__eset_container and self.__eset_container.select:
            s['eset'] = self.__eset_container.select.name
        else:
            s['eset'] = 'None'
        return s

    @property
    def thedata(self):
        return self.__thedata

    @property
    def details(self):
        print('data prefix = ' + quotes(self.name))
        print('channels = ' + str(self.channel_count))
        print('epochs = ' + str(self.epoch_count))
        # print("data list = " + str(self.data_names))

    def data_list(self, channel='selected', epoch=-1):
        if type(channel) is not list:
            channel = [channel]
        for cc in channel:
            if not cc or cc.lower() == 'selected':
                channel = self.__channel_select
                break
        all_chan = False
        for cc in channel:
            if cc.upper() == 'ALL':
                all_chan = True
                break
        if epoch == -1:
            epoch = self.__epoch_select
        dlist = []
        if all_chan:
            for chan in self.__thedata:
                if epoch >= 0 and epoch < len(chan):
                    dlist.append(chan[epoch])
            return dlist
        for cc in channel:
            cn = channel_num(cc)
            if cn >= 0 and cn < len(self.__thedata):
                chan = self.__thedata[cn]
                if epoch >= 0 and epoch < len(chan):
                    dlist.append(chan[epoch])
        return dlist

    @property
    def data_select(self):
        self.__data_select = self.get_selected(names=True)
        return self.__data_select

    def get_selected(self, names=False):
        channels = len(self.__thedata)
        cs = self.__channel_select
        ss = self.eset_select.name
        eset = self.eset_select.theset
        setx = self.eset_container.get('SetX').theset
        if channels == 0 or len(cs) == 0:
            return []
        if ss.upper() == 'ALL':
            all_epochs = True
        else:
            all_epochs = False
            eset = eset.difference(setx)
        if self.all_channels and channels > 1:
            clist = []
            for chan in self.__thedata:
                dlist = []
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
                clist.append(dlist)
            return clist
        dlist = []
        cnum_list = []
        if channels == 1:
            cnum_list.append(0)
        else:
            for c in cs:
                cnum = channel_num(c)
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


class DataPrefixContainer(Container):
    """
    NM Container for DataPrefix objects
    """

    def __init__(self, parent, name, data_container):
        super().__init__(parent, name, prefix='')
        self.__data_container = data_container

    def object_new(self, name):  # override, do not call super
        return DataPrefix(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, DataPrefix)

    def name_next(self):  # override, do not call super
        error('DataPrefix objects do not have names with a common prefix')
        return ''

    def select_set(self, name, quiet=False, new=False):  # override
        return super().select_set(name, quiet=quiet, new=True)

    def new(self, name='', select=True, quiet=False):  # override, dont call super
        if not name:
            return None
        prefix = name  # name is actually a prefix
        if not name_ok(prefix):
            error('bad prefix ' + quotes(prefix))
            return None
        pexists = self.exists(prefix)
        if pexists:
            p = self.get(prefix)
        else:
            p = DataPrefix(self.parent, prefix)
            self.add(p, select=select, quiet=True)
        if select:
            self.select_set(prefix, quiet=True)
        if not quiet:
            t = ''
            if not pexists:
                t = 'created'
            if select:
                if len(t) == 0:
                    t = 'selected'
                else:
                    t += '/selected'
                history(t + nmconfig.HD0 + p.tree_path)
        self.search(p)
        return p

    def search(self, p, quiet=False):
        if not p:
            return False
        prefix = p.name
        foundsomething = False
        thedata = self.__data_container.getAll()
        htxt = []
        for i in range(0, 25):  # try prefix+chan+seq format
            cc = channel_char(i)
            olist = get_items(thedata, prefix, chan_char=cc)
            if len(olist) > 0:
                p.thedata.append(olist)
                foundsomething = True
                p.channel_container.new(quiet=True)
                htxt.append('ch=' + cc + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:  # try without chan
            olist = get_items(thedata, prefix)
            if len(olist) > 0:
                p.thedata.append(olist)
                foundsomething = True
                p.channel_container.new(quiet=True)
                htxt.append('n=' + str(len(olist)))
        if not foundsomething:
            alert('failed to find data with prefix ' + quotes(prefix))
        if not quiet:
            for h in htxt:
                history('found' + nmconfig.HD0 + p.tree_path + ', ' + h)
        return True

    def rename(self, name, newname):  # override, do not call super
        error('cannot rename DataPrefix objects')
        return False

    def duplicate(self, name, newname, select=False):  # override
        error('cannot duplicate DataPrefix objects')
        return None  # not sued
