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
            if s.casefold() == 'all':
                select = True
            else:
                select = False
            self.__eset_container.new(s, select=select, quiet=True)
        self.__chan_select = 'A'
        self.__epoch_select = 0
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
        if chan_char.casefold() == 'all':
            return True  # always ok
        if self.channel_list().count(chan_char) == 1:
            return True
        return False

    @property
    def channel_select(self):
        return self.__chan_select

    @channel_select.setter
    def channel_select(self, chan_char):  # e.g 'A' or 'All'
        if self.channel_ok(chan_char):
            self.__chan_select = chan_char
            history('selected channel ' + chan_char)
            return True
        error('bad channel select: ' + chan_char)
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
            history('selected epoch #' + str(epoch))
            return True
        error('bad epoch number: ' + str(epoch))
        return False

    @property
    def select(self):
        s = {}
        s['channel'] = self.__chan_select
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

    def data_list(self, chan_char='selected', epoch=-1):
        if not chan_char or chan_char.casefold() == 'selected':
            chan_char = self.__chan_select
        if epoch == -1:
            epoch = self.__epoch_select
        dlist = []
        if chan_char.casefold() == 'all':
            for chan in self.__thedata:
                if epoch >= 0 and epoch < len(chan):
                    dlist.append(chan[epoch])
            return dlist
        chan_num = channel_num(chan_char)
        if chan_num >= 0 and chan_num < len(self.__thedata):
            chan = self.__thedata[chan_num]
            if epoch >= 0 and epoch < len(chan):
                return chan[epoch]
        return []


class DataPrefixContainer(Container):
    """
    NM Container for DataPrefix objects
    """

    def __init__(self, parent, name, data_container):
        super().__init__(parent, name, prefix="")
        self.__data_container = data_container

    def object_new(self, name):  # override, do not call super
        return DataPrefix(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, DataPrefix)

    def name_next(self):  # override, do not call super
        error("DataPrefix objects do not have names with a common prefix")
        return ""

    def select_setx(self, name, quiet=False):  # override, do not call super
        if not name_ok(name):
            return error('bad name ' + quotes(name))
        if not self.getAll():
            alert('nothing to select')
            return False
        for o in self.getAll():
            if name.casefold() == o.name.casefold():
                #self.select = o  # BAD >> CALLING ITSELF
                if not quiet:
                    history('selected --> ' + o.tree_path)
                return True
        p = self.new(name=name, quiet = quiet)
        return p is not None

    def new(self, name="", select=True, quiet=False):  # override, do not call super
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
                history(t + ' --> ' + p.tree_path)
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
            alert('failed to find data with prefix name ' + quotes(prefix))
        if not quiet:
            for h in htxt:
                history(' --> ' + p.tree_path + ', ' + h)
        return True

    def rename(self, name, newname):  # override, do not call super
        error('cannot rename DataPrefix objects')
        return False

    def duplicate(self, name, newname, select=False):  # override
        error('cannot duplicate DataPrefix objects')
        return None  # not sued
