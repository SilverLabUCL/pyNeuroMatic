# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_data import Data
from nm_channel import ChannelContainer
from nm_eset import EpochSetContainer
import nm_utilities as nmu

DIMS = {'xstart': 0, 'xdelta': 1, 'xlabel': '', 'xunits': '', 'ylabel': '',
        'yunits': ''}


class DataSeries(NMObject):
    """
    NM DataSeries class
    """

    def __init__(self, manager, parent, name, fxns):
        # name is the prefix
        super().__init__(manager, parent, name, fxns, rename=False)
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__thedata = []  # 2D list, i = chan #, j = seq #
        cc = ChannelContainer(manager, self, 'NMChannelContainer', fxns)
        self.__channel_container = cc
        ec = EpochSetContainer(manager, self, 'NMEpochSetContainer', fxns)
        self.__eset_container = ec
        self.__channel_select = []
        self.__epoch_select = []
        self.__data_select = []
        self.__eset_init()

    @property  # override, no super
    def content(self):
        k = {'dataseries': self.name}
        k.update(self.__channel_container.content)
        k.update({'channel_select': self.channel_select})
        k.update({'epochs': self.epoch_count})
        k.update({'epoch_select': self.epoch_select})
        k.update(self.__eset_container.content)
        return k

    @property
    def data(self):
        return self.__parent.data

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
        if len(self.__channel_select) == 0 and len(self.__thedata) > 0:
            self.__channel_select = ['A']
        return self.__channel_select

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'ALL' or ['A', 'B']
        return self.channel_select_set(chan_char_list)

    def channel_select_set(self, chan_char_list, quiet=nmc.QUIET):
        # e.g 'A', 'ALL' or ['A', 'B']
        if type(chan_char_list) is not list:
            chan_char_list = [chan_char_list]
        clist = []
        for cc in chan_char_list:
            if not self.channel_ok(cc):
                self.__error('bad channel: ' + cc, quiet=quiet)
                return False
            clist.append(cc.upper())
        self.__channel_select = clist
        h = self.tree_path(history=True) + nmc.S0 + 'ch=' + str(clist)
        self.__history(h, quiet=quiet)
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
        if not nmc.ESET_LIST:
            return []
        r = []
        for s in nmc.ESET_LIST:
            select = s.lower() == 'all'
            if self.__eset_container.new(s, select=select, quiet=True):
                r.append(s)
        if not self.__eset_container.select:
            self.__eset_container.select = nmc.ESET_LIST[0]
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
        if len(self.__epoch_select) == 0 and len(self.__thedata) > 0:
            self.__epoch_select = [0]
        return self.__epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch_list):
        return self.epoch_select_set(epoch_list)

    def epoch_select_set(self, epoch_list, quiet=nmc.QUIET):
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        for e in epoch_list:
            if not self.epoch_ok(e):
                self.__error('bad epoch: ' + str(e), quiet=quiet)
                return False
        self.__epoch_select = epoch_list
        h = self.tree_path(history=True) + nmc.S0 + 'epoch=' + str(epoch_list)
        self.__history(h, quiet=quiet)
        return True

    @property
    def thedata(self):
        return self.__thedata

    def thedata_clear(self, quiet=nmc.QUIET):
        if not quiet:
            q = ('are you sure you want to clear dataseries ' +
                 nmu.quotes(self.name) + '?')
            yn = nmu.input_yesno(q)
            if not yn == 'y':
                self.__history('abort')
                return False
        self.__thedata = []
        return True

    def data_list(self, chan_char_list=['all'], epoch_list=[-2], names=False,
                  quiet=nmc.QUIET):
        if chan_char_list == 'select':
            chan_char_list = self.channel_select
        if type(chan_char_list) is not list:
            chan_char_list = [chan_char_list]
        all_chan = False
        for cc in chan_char_list:
            if cc.lower() == 'all':
                all_chan = True
                break
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        all_epochs = False
        for e in epoch_list:
            if e == -1:
                epoch_list = self.__epoch_select
                break
            if e == -2:
                all_epochs = True
                break
        dlist = []
        if all_chan:
            for cdata in self.__thedata:
                if all_epochs:
                    epoch_list = list(range(0, len(cdata)))
                for e in epoch_list:
                    if e >= 0 and e < len(cdata):
                        if names:
                            dlist.append(cdata[e].name)
                        else:
                            dlist.append(cdata[e])
                    else:
                        self.__error('bad epoch: ' + str(e), quiet=quiet)
                        return []
            return dlist
        for cc in chan_char_list:
            cn = nmu.channel_num(cc)
            if cn >= 0 and cn < len(self.__thedata):
                cdata = self.__thedata[cn]
                if all_epochs:
                    epoch_list = list(range(0, len(cdata)))
                for e in epoch_list:
                    if e >= 0 and e < len(cdata):
                        if names:
                            dlist.append(cdata[e].name)
                        else:
                            dlist.append(cdata[e])
                    else:
                        self.__error('bad epoch: ' + str(e), quiet=quiet)
                        return []
            else:
                self.__error('bad channel: ' + cc, quiet=quiet)
                return []
        return dlist

    @property
    def data_select(self):
        self.__data_select = self.get_selected()  # update
        return self.__data_select

    @property
    def data_select_names(self):
        return self.get_selected(names=True)

    def get_selected(self, names=False, quiet=nmc.QUIET):
        channels = len(self.__thedata)
        cs = self.channel_select
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

    @property
    def dims(self):
        x = self.xdata
        if x:
            d = {'xdata': x.name}
        else:
            d = {'xstart': self.xstart, 'xdelta': self.xdelta}
        d.update({'xlabel': self.xlabel, 'xunits': self.xunits})
        d.update({'ylabel': self.ylabel, 'yunits': self.yunits})
        return d

    @dims.setter
    def dims(self, dims):
        if 'xdata' in dims.keys():
            self.xdata = dims['xdata']
        if 'xstart' in dims.keys():
            self.xstart = dims['xstart']
        if 'xdelta' in dims.keys():
            self.xdelta = dims['xdelta']
        if 'xlabel' in dims.keys():
            self.xlabel = dims['xlabel']
        if 'xunits' in dims.keys():
            self.xunits = dims['xunits']
        if 'ylabel' in dims.keys():
            self.ylabel = dims['ylabel']
        if 'yunits' in dims.keys():
            self.yunits = dims['yunits']

    def xdata_make(self, name, samples=0, dims=DIMS, quiet=nmc.QUIET):
        dims.update({'xstart': 0, 'xdelta': 1})  # enforce
        if 'xlabel' in dims.keys():  # switch x and y
            dims.update({'ylabel': dims['xlabel']})
            dims.update({'xlabel': ''})
        if 'xunits' in dims.keys():  # switch x and y
            dims.update({'yunits': dims['xunits']})
            dims.update({'xunits': ''})
        x = self.data.new(name=name, samples=samples, dims=dims, quiet=quiet)
        if not x:
            return None
        for i in range(0, samples):
            x.thedata[i] = i
        self.xdata = x
        return x

    @property
    def xdata(self):
        x = set()
        for cdata in self.__thedata:
            for d in cdata:
                x.add(d.xdata)
        x = list(x)
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x[0]
        n = []
        for xx in x:
            n.append(xx.name)
        self.__alert('encountered multiple xdata: ' + str(n))
        return None
        

    @xdata.setter
    def xdata(self, xdata):
        for cdata in self.__thedata:
            for d in cdata:
                d.xdata = xdata
        return True

    @property
    def xstart(self):
        x = set()
        for cdata in self.__thedata:
            for d in cdata:
                x.add(d.xstart)
        x = list(x)
        if len(x) == 0:
            return 0
        if len(x) == 1:
            return x[0]
        self.__alert('encountered multiple xstarts: ' + str(x))
        return 0
        

    @xstart.setter
    def xstart(self, xstart):
        if np.isinf(xstart) or np.isnan(xstart):
            return False
        for cdata in self.__thedata:
            for d in cdata:
                d.xstart = xstart
        return True

    @property
    def xdelta(self):
        x = set()
        for cdata in self.__thedata:
            for d in cdata:
                x.add(d.xdelta)
        x = list(x)
        if len(x) == 0:
            return 1
        if len(x) == 1:
            return x[0]
        self.__alert('encountered multiple xdeltas: ' + str(x))
        return 1

    @xdelta.setter
    def xdelta(self, xdelta):
        if np.isinf(xdelta) or np.isnan(xdelta):
            return False
        for cdata in self.__thedata:
            for d in cdata:
                d.xdelta = xdelta
        return True

    @property
    def xlabel(self):
        x = set()
        for cdata in self.__thedata:
            for d in cdata:
                x.add(d.xlabel)
        x = list(x)
        if len(x) == 0:
            return ''
        if len(x) == 1:
            return x[0]
        self.__alert('encountered multiple xlabels: ' + str(x))
        return ''

    @xlabel.setter
    def xlabel(self, xlabel):
        if not isinstance(xlabel, str):
            return False
        for cdata in self.__thedata:
            for d in cdata:
                d.xlabel = xlabel
        return True

    @property
    def xunits(self):
        x = set()
        for cdata in self.__thedata:
            for d in cdata:
                x.add(d.xunits)
        x = list(x)
        if len(x) == 0:
            return ''
        if len(x) == 1:
            return x[0]
        self.__alert('encountered multiple xunits: ' + str(x))
        return ''

    @xunits.setter
    def xunits(self, xunits):
        if not isinstance(xunits, str):
            return False
        for cdata in self.__thedata:
            for d in cdata:
                d.xunits = xunits
        return True

    @property
    def ylabel(self):
        y = []
        for cdata in self.__thedata:
            i = 0
            ys = set()
            for d in cdata:
                ys.add(d.ylabel)
            yl = list(ys)
            if len(yl) == 0:
                y.append('')
            elif len(yl) == 1:
                y.append(yl[0])
            else:
                self.__alert('ch=' + nmu.channel_char(i) +
                          ', encountered multiple ylabels: ' + str(yl))
                y.append('')
            i += 1
        return y

    @ylabel.setter
    def ylabel(self, ylabel):
        if not isinstance(ylabel, list):
            if not isinstance(ylabel, str):
                return False
            ylabel = [ylabel]
        i = 0
        for cdata in self.__thedata:
            if i >= len(ylabel):
                break
            y = ylabel[i]
            if isinstance(y, str):
                for d in cdata:
                    d.ylabel = y
            i += 1
        return True

    def ylabel_set(self, chan_char, ylabel, quiet=nmc.QUIET):
        if not isinstance(chan_char, str):
            self.__error('bad channel', quiet=quiet)
            return False
        if not isinstance(ylabel, str):
            self.__error('bad ylabel', quiet=quiet)
            return False
        cn = nmu.channel_num(chan_char)
        if cn >= 0 and cn < len(self.__thedata):
            cdata = self.__thedata[cn]
            for d in cdata:
                d.ylabel = ylabel
            return True
        self.__error('bad channel: ' + chan_char, quiet=quiet)
        return False

    @property
    def yunits(self):
        y = []
        for cdata in self.__thedata:
            i = 0
            ys = set()
            for d in cdata:
                ys.add(d.yunits)
            yl = list(ys)
            if len(yl) == 0:
                y.append('')
            elif len(yl) == 1:
                y.append(yl[0])
            else:
                self.__alert('ch=' + nmu.channel_char(i) +
                          ', encountered multiple yunits: ' + str(yl))
                y.append('')
            i += 1
        return y

    @yunits.setter
    def yunits(self, yunits):
        if not isinstance(yunits, list):
            if not isinstance(yunits, str):
                return False
            yunits = [yunits]
        i = 0
        for cdata in self.__thedata:
            if i >= len(yunits):
                break
            y = yunits[i]
            if isinstance(y, str):
                for d in cdata:
                    d.yunits = y
            i += 1
        return True

    def yunits_set(self, chan_char, yunits, quiet=nmc.QUIET):
        if not isinstance(chan_char, str):
            self.__error('bad channel', quiet=quiet)
            return False
        if not isinstance(yunits, str):
            self.__error('bad yunits', quiet=quiet)
            return False
        cn = nmu.channel_num(chan_char)
        if cn >= 0 and cn < len(self.__thedata):
            cdata = self.__thedata[cn]
            for d in cdata:
                d.yunits = yunits
            return True
        self.__error('bad channel: ' + chan_char, quiet=quiet)
        return False


class DataSeriesContainer(Container):
    """
    NM Container for DataSeries objects
    """

    def __init__(self, manager, parent, name, fxns):
        o = DataSeries(manager, parent, 'temp', fxns)
        super().__init__(manager, parent, name, fxns, nmobj=o, prefix='',
                         rename=False, duplicate=False)
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        k = {'dataseries': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'dataseries_select': s})
        return k

    @property
    def data(self):
        folder = self.__parent
        return folder.data

    # override
    def new(self, name='', select=True, quiet=nmc.QUIET):
        # name is the data prefix name
        o = DataSeries(self.__manager, self.__parent, name, self.__fxns)
        ds = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if ds:
            self.update(name=ds.name, quiet=quiet)
            return ds
        return None

    def update(self, name='select', quiet=nmc.QUIET):
        # name is data prefix
        ds = self.get(name=name, quiet=quiet)
        if not ds:
            return False
        name = ds.name
        foundsomething = False
        # thedata = self.data.get_all()
        thedata = self.data._Container__thecontainer
        htxt = []
        ds.thedata_clear(quiet=True)
        for i in range(0, 25):
            cc = nmu.channel_char(i)
            olist = nmu.get_items(thedata, name, chan_char=cc)
            if len(olist) > 0:
                ds.thedata.append(olist)
                foundsomething = True
                ds.channel.new(name=cc, quiet=True)
                htxt.append('ch=' + cc + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            a = 'failed to find data with prefix name ' + nmu.quotes(name)
            self.__alert(a, quiet=quiet)
        for h in htxt:
            tp = ds.tree_path(history=True)
            self.__history('found' + nmc.S0 + tp + ', ' + h, quiet=quiet)
        return True

    def make(self, name='default', channels=1, epochs=1, samples=0,
             fill_value=0, noise=[], dims={}, select=True, quiet=nmc.QUIET):
        # name is data prefix
        if not name or name.lower() == 'default':
            name = self.data.prefix
        if not nmu.name_ok(name):
            self.__error('bad data prefix name ' + nmu.quotes(name), quiet=quiet)
            return None
        if not nmu.num_ok(channels, no_neg=True, no_zero=True):
            self.__error('bad channels argument: ' + str(channels), quiet=quiet)
            return None
        if not nmu.num_ok(epochs, no_neg=True, no_zero=True):
            self.__error('bad epochs argument: ' + str(epochs), quiet=quiet)
            return None
        if not nmu.num_ok(samples, no_neg=True):
            self.__error('bad samples argument: ' + str(samples), quiet=quiet)
            return None
        ds = self.get(name=name, quiet=True)
        if ds and ds.channel_count != channels:
            e = ('data series ' + nmu.quotes(name) + ' already exists, and ' +
                 'requires channels=' + str(ds.channel_count))
            self.__error(e, quiet=quiet)
            return None
        seq_start = []
        for ci in range(0, channels):  # look for existing data
            cc = nmu.channel_char(ci)
            si = self.data.name_next_seq(name + cc, quiet=quiet)
            if si >= 0:
                seq_start.append(si)
        ss = max(seq_start)
        se = ss + epochs
        htxt = []
        tree_path = ''
        for ci in range(0, channels):
            cc = nmu.channel_char(ci)
            for j in range(ss, se):
                name2 = name + cc + str(j)
                ts = self.data.new(name=name2, samples=samples,
                                   fill_value=fill_value, noise=noise,
                                   quiet=True)
                tree_path = self.__parent.tree_path(history=True)
                if not ts:
                    a = 'failed to create ' + nmu.quotes(name2)
                    self.__alert(a, quiet=quiet)
            htxt.append('ch=' + cc + ', ep=' + str(ss) + '-' + str(se-1))
        for h in htxt:
            path = name
            if len(tree_path) > 0:
                path = tree_path + "." + path
            self.__history('created' + nmc.S0 + path + ', ' + h, quiet=quiet)
        if ds:
            if select:
                self.select = name
            else:
                self.update(name=name, quiet=quiet)
        else:
            ds = self.new(name=name, select=select, quiet=quiet)
        ds.dims = dims
        return ds
