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

DIMS = {'xstart': 0, 'xdelta': 1, 'xlabel': '', 'xunits': '', 'ylabel': '',
        'yunits': ''}


class DataSeries(NMObject):
    """
    NM DataSeries class
    """

    def __init__(self, manager, parent, name, fxns):
        # name is data-series prefix
        super().__init__(manager, parent, name, fxns, rename=False)
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__thedata = []  # 2D list, i = chan #, j = seq #
        cc = ChannelContainer(manager, self, 'Channels', fxns)
        self.__channel_container = cc
        ec = EpochSetContainer(manager, self, 'EpochSets', fxns)
        self.__eset_container = ec
        self.__channel_select = []
        self.__epoch_select = []
        self.__data_select = []
        self.__eset_init(quiet=True)

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

    def channel_list(self, include_all=False):
        if not isinstance(include_all, bool):
            include_all = False
        n = len(self.__thedata)
        if n == 0:
            return []
        if include_all and n > 1:
            clist = ['ALL']
        else:
            clist = []
        for i in range(0, n):
            c = nmu.channel_char(i)
            clist.append(c)
        return clist

    def channel_ok(self, chan_char):
        if not chan_char or not isinstance(chan_char, str):
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
            self.__channel_select = ['A']  # default value
        return self.__channel_select

    @channel_select.setter
    def channel_select(self, chan_list):  # e.g 'A', 'ALL' or ['A', 'B']
        tp = self.tree_path(history=True)
        if not isinstance(chan_list, list):
            chan_list = [chan_list]
        clist = []
        for c in chan_list:
            if not self.channel_ok(c):
                e = 'bad channel: ' + nmu.quotes(c)
                self.__error(e, tp=tp)
                return False
            clist.append(c.upper())
        self.__channel_select = clist
        self.__history('channel = ' + str(clist), tp=tp)
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

    def eset_init(self, eset_list=nmc.ESET_LIST, select=True, quiet=nmc.QUIET):
        if not eset_list:
            return []
        if not isinstance(eset_list, list):
            eset_list = [eset_list]
        if not isinstance(select, bool):
            select = True
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        r = []
        init_select = select or self.__eset_container.select is None
        for s in nmc.ESET_LIST:
            if not s or not isinstance(s, str):
                continue
            select = init_select and s.lower() == 'all'
            if self.__eset_container.new(name=s, select=select, quiet=quiet):
                r.append(s)
        if init_select and self.__eset_container.select is None:
            self.__eset_container.select = nmc.ESET_LIST[0]
        return r

    __eset_init = eset_init  # called within __init__

    @property
    def epoch_count(self):  # epochs per channel
        if not self.__thedata:
            return [0]
        elist = []
        for cdata in self.__thedata:
            elist.append(len(cdata))
        return elist

    def epoch_ok(self, epoch_list):
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        emax = max(self.epoch_count)
        for e in epoch_list:
            if not isinstance(e, int):
                return False
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
        tp = self.tree_path(history=True)
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        for e in epoch_list:
            if not self.epoch_ok(e):
                self.__error('bad epoch: ' + str(e), tp=tp)
                return False
        self.__epoch_select = epoch_list
        self.__history('epoch = ' + str(epoch_list), tp=tp)
        return True

    @property
    def thedata(self):
        return self.__thedata

    def thedata_clear(self, quiet=nmc.QUIET):
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not quiet:
            q = ('are you sure you want to clear data references for ' +
                 'dataseries ' + nmu.quotes(self.name) + '?')
            yn = nmu.input_yesno(q)
            if not yn == 'y':
                self.__history('cancel', tp=tp, quiet=quiet)
                return False
        self.__thedata = []
        return True

    def data_list(self, chan_list=['all'], epoch_list=[-2], names=False,
                  quiet=nmc.QUIET):
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if chan_list == 'select':
            chan_list = self.channel_select
        if not isinstance(chan_list, list):
            chan_list = [chan_list]
        all_chan = False
        for c in chan_list:
            if not self.channel_ok(c):
                e = 'bad channel: ' + nmu.quotes(c)
                self.__error(e, tp=tp, quiet=quiet)
                return []
            if c.lower() == 'all':
                all_chan = True
                break
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        all_epochs = False
        for ep in epoch_list:
            if not isinstance(ep, int):
                self.__error('bad epoch: ' + str(ep), tp=tp, quiet=quiet)
                return []
            if ep == -1:
                epoch_list = self.__epoch_select
                break
            if ep == -2:
                all_epochs = True
                break
        if not isinstance(names, bool):
            names = False
        dlist = []
        if all_chan:
            for cdata in self.__thedata:
                if all_epochs:
                    epoch_list = list(range(0, len(cdata)))
                for ep in epoch_list:
                    if ep >= 0 and ep < len(cdata):
                        if names:
                            dlist.append(cdata[ep].name)
                        else:
                            dlist.append(cdata[ep])
                    else:
                        e = 'bad epoch: ' + str(ep)
                        self.__error(e, tp=tp, quiet=quiet)
                        return []
            return dlist
        for c in chan_list:
            cn = nmu.channel_num(c)
            if cn >= 0 and cn < len(self.__thedata):
                cdata = self.__thedata[cn]
                if all_epochs:
                    epoch_list = list(range(0, len(cdata)))
                for ep in epoch_list:
                    if ep >= 0 and ep < len(cdata):
                        if names:
                            dlist.append(cdata[ep].name)
                        else:
                            dlist.append(cdata[ep])
                    else:
                        e = 'bad epoch: ' + str(ep)
                        self.__error(e, tp=tp, quiet=quiet)
                        return []
            else:
                e = 'bad channel: ' + nmu.quotes(c)
                self.__error(e, tp=tp, quiet=quiet)
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
        if not isinstance(names, bool):
            names = False
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
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
        if not isinstance(dims, dict):
            dims = DIMS
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
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
        tp = self.tree_path(history=True)
        self.__alert('encountered multiple xdata: ' + str(n), tp=tp)
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
        tp = self.tree_path(history=True)
        self.__alert('encountered multiple xstarts: ' + str(x), tp=tp)
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
        tp = self.tree_path(history=True)
        self.__alert('encountered multiple xdeltas: ' + str(x), tp=tp)
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
        tp = self.tree_path(history=True)
        self.__alert('encountered multiple xlabels: ' + str(x), tp=tp)
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
        tp = self.tree_path(history=True)
        self.__alert('encountered multiple xunits: ' + str(x), tp=tp)
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
        tp = self.tree_path(history=True)
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
                a = ('ch=' + nmu.channel_char(i) + ', ' +
                     'encountered multiple ylabels: ' + str(yl))
                self.__alert(a, tp=tp)
                y.append('')
            i += 1
        return y

    @ylabel.setter
    def ylabel(self, ylabel):
        tp = self.tree_path(history=True)
        if not isinstance(ylabel, list):
            ylabel = [ylabel]
        i = 0
        for y in ylabel:
            if not isinstance(y, str):
                e = 'ylabel is not a string type: ' + nmu.quotes(y)
                self.__alert(e, tp=tp)
                return False
        for cdata in self.__thedata:
            if i >= len(ylabel):
                break
            y = ylabel[i]
            for d in cdata:
                d.ylabel = y
            i += 1
        return True

    def ylabel_set(self, chan_char, ylabel, quiet=nmc.QUIET):
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not isinstance(chan_char, str):
            e = 'bad channel: ' + nmu.quotes(chan_char)
            self.__error(e, tp=tp, quiet=quiet)
            return False
        if not isinstance(ylabel, str):
            e = 'bad ylabel: ' + nmu.quotes(ylabel)
            self.__error(e, tp=tp, quiet=quiet)
            return False
        cn = nmu.channel_num(chan_char)
        if cn >= 0 and cn < len(self.__thedata):
            cdata = self.__thedata[cn]
            for d in cdata:
                d.ylabel = ylabel
            return True
        e = 'bad channel: ' + nmu.quotes(chan_char)
        self.__error(e, tp=tp, quiet=quiet)
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
                tp = self.tree_path(history=True)
                a = ('ch=' + nmu.channel_char(i) + ', ' +
                     'encountered multiple yunits: ' + str(yl))
                self.__alert(a, tp=tp)
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
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not isinstance(chan_char, str):
            e = 'bad channel: ' + nmu.quotes(chan_char)
            self.__error(e, tp=tp, quiet=quiet)
            return False
        if not isinstance(yunits, str):
            e = 'bad yunits: ' + nmu.quotes(yunits)
            self.__error(e, tp=tp, quiet=quiet)
            return False
        cn = nmu.channel_num(chan_char)
        if cn >= 0 and cn < len(self.__thedata):
            cdata = self.__thedata[cn]
            for d in cdata:
                d.yunits = yunits
            return True
        e = 'bad channel: ' + nmu.quotes(chan_char)
        self.__error(e, tp=tp, quiet=quiet)
        return False


class DataSeriesContainer(Container):
    """
    NM Container for DataSeries objects
    """

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns, type_='DataSeries',
                         prefix='', rename=False, duplicate=False)
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
        return self.__parent.data

    # override
    def new(self, name='', select=True, quiet=nmc.QUIET):
        # name is the data-series name
        o = DataSeries(self.__manager, self.__parent, name, self.__fxns)
        ds = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if ds:
            self.update(name=ds.name, quiet=quiet)
            return ds
        return None

    def update(self, name='select', quiet=nmc.QUIET):
        # name is data-series prefix
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not self.exists(name):
            e = 'failed to find ' + nmu.quotes(name)
            e += '\n' + 'acceptable names: ' + str(self.names)
            self.__error(e, tp=tp, quiet=quiet)
        ds = self.get(name, quiet=quiet)
        if not ds:
            return False
        foundsomething = False
        # thedata = self.data.get_all()
        htxt = []
        ds.thedata_clear(quiet=True)
        for i in range(0, 25):
            c = nmu.channel_char(i)
            olist = self.__get_items(ds.name, c)
            if len(olist) > 0:
                ds.thedata.append(olist)
                foundsomething = True
                ds.channel.new(name=c, quiet=True)
                htxt.append('ch=' + c + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            a = 'failed to find data with prefix ' + nmu.quotes(ds.name)
            self.__alert(a, tp=tp, quiet=quiet)
        for h in htxt:
            h = 'found data with prefix ' + nmu.quotes(ds.name) + ': ' + h
            self.__history(h, tp=tp, quiet=quiet)
        return True

    def __get_items(self, name, chan_char):
        thedata = self.data._Container__thecontainer  # mangled
        olist = []
        i = len(name)
        for o in thedata:
            if name.casefold() == o.name[:i].casefold():
                if chan_char:
                    if nmu.channel_char_exists(o.name[i:], chan_char):
                        olist.append(o)
                    else:
                        pass
                else:
                    olist.append(o)
        return olist

    def make(self, name='default', channels=1, epochs=1, samples=0,
             fill_value=0, noise=[], dims={}, select=True, quiet=nmc.QUIET):
        # name is data-series prefix
        tp = self.tree_path(history=True)
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not nmu.name_ok(name):
            e = 'bad name arg: ' + nmu.quotes(name)
            self.__error(e, tp=tp, quiet=quiet)
            return None
        if not name or name.lower() == 'default':
            name = self.data.prefix
        if not nmu.number_ok(channels, no_neg=True, no_zero=True):
            e = 'bad channels argument: ' + str(channels)
            self.__error(e, tp=tp, quiet=quiet)
            return None
        if not nmu.number_ok(epochs, no_neg=True, no_zero=True):
            e = 'bad epochs argument: ' + str(epochs)
            self.__error(e, tp=tp, quiet=quiet)
            return None
        if not nmu.number_ok(samples, no_neg=True):
            e = 'bad samples argument: ' + str(samples)
            self.__error(e, tp=tp, quiet=quiet)
            return None
        ds = self.get(name, quiet=True)
        if ds and ds.channel_count != channels:
            e = ('data series ' + nmu.quotes(name) + ' already exists, and ' +
                 'requires channels=' + str(ds.channel_count))
            self.__error(e, tp=tp, quiet=quiet)
            return None
        epoch_start = []
        for ci in range(0, channels):  # look for existing data
            c = nmu.channel_char(ci)
            si = self.data.name_next_seq(name + c, quiet=quiet)
            if si >= 0:
                epoch_start.append(si)
        e_bgn = max(epoch_start)
        e_end = e_bgn + epochs
        n = nmu.quotes(name)
        for ci in range(0, channels):
            c = nmu.channel_char(ci)
            elist = []
            for j in range(e_bgn, e_end):
                name2 = name + c + str(j)
                d = self.data.new(name=name2, samples=samples,
                                  fill_value=fill_value, noise=noise,
                                  quiet=True)
                if d:
                    elist.append(j)
                else:
                    a = 'failed to create ' + nmu.quotes(name2)
                    self.__alert(a, tp=tp, quiet=quiet)
            h = nmu.int_list_to_seq_str(elist, space=False)
            h = 'created ' + n + ', ' + 'ch=' + c + ', ep=' + h
            self.__history(h, tp=tp, quiet=quiet)
        if ds:
            if select:
                self.select = name
            else:
                self.update(name=name, quiet=quiet)
        else:
            ds = self.new(name=name, select=select, quiet=quiet)
        ds.dims = dims
        return ds
