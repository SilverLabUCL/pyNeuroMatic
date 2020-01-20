# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

from nm_container import NMObject
from nm_container import Container
from nm_channel import ChannelContainer
from nm_dimensions import Dimensions
from nm_dimensions import XDimensions
from nm_eset import EpochSetContainer
import nm_preferences as nmp
import nm_utilities as nmu


class DataSeries(NMObject):
    """
    NM DataSeries class
    """

    def __init__(self, parent, name, fxns={}, dims={}):
        # name is data-series prefix
        super().__init__(parent, name, fxns=fxns, rename=False)
        cc = ChannelContainer(self, 'Channels', fxns=fxns)
        self.__channel_container = cc
        ec = EpochSetContainer(self, 'EpochSets', fxns=fxns)
        self.__eset_container = ec
        self.__thedata = {}  # dict, {channel: data-list}
        if dims:
            self._dims_set(dims, quiet=True)
        else:
            self.__dims = {}
        self.__dims_master_on = True
        self.__data_select = {}  # dict, {channel: data-list}
        self.__channel_select = []
        self.__epoch_select = []
        self.__eset_init(quiet=True)

    # override
    @property
    def parameters(self):
        k = super().parameters
        # k.update({'thedata': self.__thedata})
        k.update({'channel_select': self.__channel_select})
        k.update({'epoch_select': self.__epoch_select})
        # k.update({'data_select': self.__data_select})
        return k

    # override, no super
    @property
    def content(self):
        k = {'dataseries': self.name}
        k.update(self.__channel_container.content)
        k.update({'channel_select': self.channel_select})
        k.update({'epochs': self.epoch_count})
        k.update({'epoch_select': self.epoch_select})
        k.update(self.__eset_container.content)
        return k

    # override
    def _equal(self, dataseries, ignore_name=False, alert=False):
        if not super()._equal(dataseries, ignore_name=ignore_name,
                              alert=alert):
            return False
        c = dataseries._DataSeries__channel_container
        if not self.__channel_container._equal(c, alert=alert):
            return False
        c = dataseries._DataSeries__eset_container
        return self.__eset_container._equal(c, alert=alert)

    # override
    def _copy(self, dataseries, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(dataseries, DataSeries):
            e = nmu.type_error(dataseries, 'DataSeries')
            raise TypeError(e)
        if not super()._copy(dataseries, copy_name=copy_name, quiet=True):
            return False
        c = dataseries._DataSeries__channel_container
        if not self.__channel_container._copy(c, quiet=True):
            return False
        c = dataseries._DataSeries__eset_container
        if not self.__eset_container._copy(c, quiet=True):
            return False
        self.__channel_select = list(dataseries._DataSeries__channel_select)
        self.__epoch_select = list(dataseries._DataSeries__epoch_select)
        # COPY thedata
        # COPY data_select
        # self.__data_select.clear()  # RESET
        h = ('copied DataSeries ' + nmu.quotes(dataseries.name) + ' to ' +
             nmu.quotes(name))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def data(self):
        # cannot import Folder class
        if self._parent.__class__.__name__ == 'Folder':
            return self._parent.data
        return None

    @property
    def thedata(self):
        return self.__thedata

    @property
    def dims_master_on(self):
        return self.__dims_master_on

    @dims_master_on.setter
    def dims_master_on(self, on):
        on = nmu.check_bool(on, True)
        self.__dims_master_on = on
        return on

    # override
    @property
    def dims(self):
        if self.__dims_master_on:
            return self.__dims
        return self._dims_of_thedata

    def _dims_of_thedata(self):
        xdata = []
        xstart = []
        xdelta = []
        xlabel = []
        xunits = []
        ylabel = {}
        yunits = {}
        for c, cdata in self.__thedata.items():
            yl = []
            yu = []
            for d in cdata:
                if d.xdata not in xdata:
                    xdata.append(d.xdata)
                if d.xstart not in xstart:
                    xstart.append(d.xstart)
                if d.xdelta not in xdelta:
                    xdelta.append(d.xdelta)
                if d.xlabel not in xlabel:
                    xlabel.append(d.xlabel)
                if d.xunits not in xunits:
                    xunits.append(d.xunits)
                if d.ylabel not in yl:
                    yl.append(d.ylabel)
                if d.yunits not in yu:
                    yu.append(d.yunits)
            ylabel.update({c: yl})
            yunits.update({c: yu})
        dims = {'xdata': xdata}
        dims.update({'xstart': xstart, 'xdelta': xdelta})
        dims.update({'xlabel': xlabel, 'xunits': xunits})
        dims.update({'ylabel': ylabel, 'yunits': yunits})
        return dims

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, quiet=nmp.QUIET):
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        for k in dims.keys():
            if k not in nmu.DIM_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        k = dims.keys()
        if 'xdata' in k:
            self._xdata_set(dims['xdata'], quiet=quiet)
        if 'xstart' in k:
            self._xstart_set(dims['xstart'], quiet=quiet)
        if 'xdelta' in k:
            self._xdelta_set(dims['xdelta'], quiet=quiet)
        if 'xlabel' in k:
            self._xlabel_set(dims['xlabel'], quiet=quiet)
        if 'xunits' in k:
            self._xunits_set(dims['xunits'], quiet=quiet)
        if 'ylabel' in k:
            self._ylabel_set(dims['ylabel'], quiet=quiet)
        if 'yunits' in k:
            self._yunits_set(dims['yunits'], quiet=quiet)
        return True

    @property
    def xdata(self):
        k = 'xdata'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return None

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, quiet=nmp.QUIET):
        if xdata is None:
            pass  # ok
        elif xdata.__class__.__name__ != 'Data':  # cannot import Data class
            raise TypeError(nmu.type_error(xdata, 'Data'))
        old = self.xdata
        # if xdata == old:
        #    return True
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d._xdata_set(xdata, alert=False, quiet=True)
        self.__dims = {}  # reset
        new = self.xdata
        h = nmu.history_change('xdata', old, new)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xstart(self):
        k = 'xstart'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return 0

    @xstart.setter
    def xstart(self, xstart):
        return self._xstart_set(xstart)

    def _xstart_set(self, xstart, quiet=nmp.QUIET):
        if not isinstance(xstart, float) and not isinstance(xstart, int):
            raise TypeError(nmu.type_error(xstart, 'number'))
        if not nmu.number_ok(xstart):
            raise ValueError('bad xstart: ' + str(xstart))
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d._xstart_set(xstart, alert=False, quiet=True)
        return True

    @property
    def xdelta(self):
        k = 'xdelta'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return 1

    @xdelta.setter
    def xdelta(self, xdelta):
        return self._xdelta_set(xdelta)

    def _xdelta_set(self, xdelta, quiet=nmp.QUIET):
        if np.isinf(xdelta) or np.isnan(xdelta):
            return False
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xdelta = xdelta
        return True

    @property
    def xlabel(self):
        k = 'xlabel'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ''

    @xlabel.setter
    def xlabel(self, xlabel):
        return self._xlabel_set(xlabel)

    def _xlabel_set(self, xlabel, quiet=nmp.QUIET):
        if not isinstance(xlabel, str):
            return False
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xlabel = xlabel
        return True

    @property
    def xunits(self):
        k = 'xunits'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ''

    @xunits.setter
    def xunits(self, xunits):
        return self._xunits_set(xunits)

    def _xunits_set(self, xunits, quiet=nmp.QUIET):
        if not isinstance(xunits, str):
            return False
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xunits = xunits
        return True

    @property
    def ylabel(self):
        k = 'ylabel'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ''

    @ylabel.setter
    def ylabel(self, chan_ylabel):
        return self._ylabel_set(chan_ylabel)

    def _ylabel_set(self, chan_ylabel, quiet=nmp.QUIET):
        if isinstance(chan_ylabel, str):
            chan_ylabel = {'A': chan_ylabel}
        elif not isinstance(chan_ylabel, dict):
            e = 'chan_ylabel is not a dictionary'
            self._error(e, tp=self._tp)
            return False
        for c, ylabel in chan_ylabel.items():
            if not isinstance(ylabel, str):
                e = 'ylabel is not a string type: ' + nmu.quotes(ylabel)
                self._error(e, tp=self._tp)
            # elif c not in self.__thedata.keys():
            elif c not in nmp.CHAN_LIST:
                e = 'bad channel: ' + nmu.quotes(c)
                self._error(e, tp=self._tp)
            if c in self.__thedata.keys():
                cdata = self.__thedata[c]
                for d in cdata:
                    d.ylabel = ylabel
        return True

    @property
    def _ylabel_of_thedata(self):
        yy = {}
        for c, cdata in self.__thedata.items():
            y = []
            ylower = []
            for d in cdata:
                if d.ylabel.lower() not in ylower:
                    y.append(d.ylabel)
                    ylower.append(d.ylabel.lower())
            yy.update({c: y})
        return yy

    @property
    def yunits(self):
        k = 'yunits'
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ''

    @yunits.setter
    def yunits(self, chan_yunits):
        return self._yunits_set(chan_yunits)

    def _yunits_set(self, chan_yunits, quiet=nmp.QUIET):
        if isinstance(chan_yunits, str):
            chan_yunits = {'A': chan_yunits}
        elif not isinstance(chan_yunits, dict):
            e = 'chan_yunits is not a dictionary'
            self._error(e, tp=self._tp)
            return False
        for c, yunits in chan_yunits.items():
            if not isinstance(yunits, str):
                e = 'yunits is not a string type: ' + nmu.quotes(yunits)
                self._error(e, tp=self._tp)
            # elif c not in self.__thedata.keys():
            elif c not in nmp.CHAN_LIST:
                e = 'bad channel: ' + nmu.quotes(c)
                self._error(e, tp=self._tp)
            if c in self.__thedata.keys():
                cdata = self.__thedata[c]
                for d in cdata:
                    d.yunits = yunits
        return True

    @property
    def channel(self):
        return self.__channel_container

    @property
    def channel_count(self):
        if not self.__thedata:
            return 0
        return len(self.__thedata)

    @property
    def channel_list(self):
        if not self.__thedata:
            return []
        return [c.upper() for c in self.__thedata.keys()]  # force UPPER

    def channel_ok(self, chan_list):
        if not isinstance(chan_list, list):
            chan_list = [chan_list]
        for c in chan_list:
            if not isinstance(c, str):
                return False
            if c.upper() == 'ALL':
                return True  # always OK
        clist = self.channel_list
        for c in chan_list:
            if c.upper() not in clist:
                return False
        return True

    @property
    def channel_select(self):
        if not self.__channel_select:
            clist = self.channel_list
            if len(clist) > 0:
                cfirst = clist[0].upper()
                self.__channel_select = [cfirst]  # first is default
            else:
                self.__channel_select = []
            return self.__channel_select
        if not isinstance(self.__channel_select, list):
            self.__channel_select = [self.__channel_select]
        clist = [c.upper() for c in self.__channel_select]  # force UPPER
        self.__channel_select = clist
        return self.__channel_select

    @channel_select.setter
    def channel_select(self, chan_list):  # e.g 'A', 'ALL' or ['A', 'C']
        if not isinstance(chan_list, list):
            chan_list = [chan_list]
        clist = []
        for c in chan_list:
            if self.channel_ok(c):
                clist.append(c.upper())  # force UPPER
            else:
                self._error('bad channel: ' + nmu.quotes(c), tp=self._tp)
        self.__channel_select = clist
        self._history('channel = ' + str(clist), tp=self._tp)
        return True

    @property
    def eset(self):
        return self.__eset_container

    def eset_init(self, eset_list=nmp.ESET_LIST, select=True, quiet=nmp.QUIET):
        if not eset_list:
            return []
        if not isinstance(eset_list, list):
            eset_list = [eset_list]
        select = nmu.check_bool(select, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        r = []
        init_select = select or not self.__eset_container.select
        for s in nmp.ESET_LIST:
            if not s or not isinstance(s, str):
                continue
            select = init_select and s.upper() == 'ALL'
            if self.__eset_container.new(name=s, select=select, quiet=quiet):
                r.append(s)
        if init_select and not self.__eset_container.select:
            self.__eset_container.select = eset_list[0]
        return r

    __eset_init = eset_init

    @property
    def epoch_count(self):  # epochs per channel
        if not self.__thedata:
            return {}
        e = {}
        for c, cdata in self.__thedata.items():
            e.update({c: len(cdata)})
        return e

    def epoch_ok(self, epoch_list):
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        ep_max = max(self.epoch_count)
        for ep in epoch_list:
            if not isinstance(ep, int):
                return False
            if ep < 0 or ep >= ep_max:
                return False
        return True

    @property
    def epoch_select(self):
        if not self.__epoch_select:
            for c, n in self.epoch_count:
                if n > 0:
                    self.__epoch_select = [0]  # first is default
                    return self.__epoch_select
            self.__epoch_select = []
            return self.__epoch_select
        if not isinstance(self.__epoch_select, list):
            self.__epoch_select = [self.__epoch_select]
        self.__epoch_select = [e for e in self.__epoch_select if
                               isinstance(e, int)]
        return self.__epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch_list):
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        elist = []
        for e in epoch_list:
            if self.epoch_ok(e):
                elist.append(e)
            else:
                self._error('bad epoch: ' + str(e), tp=self._tp)
        self.__epoch_select = elist
        self._history('epoch = ' + str(elist), tp=self._tp)
        return True

    def _get_channel_list(self, chan_list=['select']):
        if not isinstance(chan_list, list):
            chan_list = [chan_list]
        clist = []
        for c in chan_list:
            if not isinstance(c, str):
                continue
            if c.upper() == 'ALL':
                return self.channel_list
            if c.upper() == 'SELECT':
                clist = self.channel_select
                break
            clist.append(c)
        return [c for c in clist if self.channel_ok(c)]

    def get_data_names(self, chan_list=['ALL'], epoch_list=[-2],
                       quiet=nmp.QUIET):
        d = self.get_data(chan_list=chan_list, epoch_list=epoch_list,
                          quiet=quiet)
        n = {}
        for c, cdata in d.items():
            nlist = [d.name for d in cdata]
            n.update({c: nlist})
        return n

    def get_data(self, chan_list=['ALL'], epoch_list=[-2], quiet=nmp.QUIET):
        if not self.__thedata:
            return {}
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        clist = self._get_channel_list(chan_list=chan_list)
        if not clist:
            return {}
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        all_epochs = False
        elist = []
        for ep in epoch_list:
            if not isinstance(ep, int):
                self._error('bad epoch: ' + str(ep), tp=self._tp, quiet=quiet)
                continue
            if ep == -1:
                elist = self.__epoch_select
                break
            if ep == -2:
                elist = []
                all_epochs = True
                break
            elist.append(ep)
        if not elist and not all_epochs:
            return {}
        dd = {}
        for c in clist:
            cdata = self.__thedata[c]
            dlist = []
            if all_epochs:
                dlist = cdata
            else:
                for ep in elist:
                    if ep >= 0 and ep < len(cdata):
                        dlist.append(cdata[ep])
                    else:
                        e = 'epoch out of range: ' + str(ep)
                        self._error(e, tp=self._tp, quiet=quiet)
            dd.update({c: dlist})
        return dd

    @property
    def data_select_names(self):
        d = self.data_select
        n = {}
        for c, cdata in d.items():
            nlist = [d.name for d in cdata]
            n.update({c: nlist})
        return n

    @property
    def data_select(self):
        clist = self._get_channel_list(chan_list=['select'])
        if not clist:
            return {}
        if not self.eset.select or not self.eset.select.name:
            return {}
        if not self.eset.select.theset:
            return {}
        sname = self.eset.select.name
        eset = self.eset.select.theset
        sx = self.eset.getitem('SetX')
        if sx:
            setx = sx.theset
        else:
            setx = None
        all_epochs = sname.upper() == 'ALL'
        dd = {}
        for c in clist:
            cdata = self.__thedata[c]
            dlist = []
            for d in cdata:
                if setx and d in setx:
                    continue
                if all_epochs or d in eset:
                    dlist.append(d)
            dd.update({c: dlist})
        return dd

    def _getitems(self, chan_char):
        thedata = self.data._Container__thecontainer  # mangled
        dlist = []
        i = len(self.name)
        for o in thedata:
            if o.name[:i].casefold() == self.name.casefold():
                if chan_char:
                    if nmu.channel_char_exists(o.name[i:], chan_char):
                        dlist.append(o)
                else:
                    dlist.append(o)
        return dlist

    def update(self, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        foundsomething = False
        htxt = []
        self.__thedata = {}
        for i in range(0, 25):
            c = nmu.channel_char(i)
            olist = self._getitems(c)
            if len(olist) > 0:
                self.__thedata.append(olist)
                foundsomething = True
                self.channel.new(name=c, quiet=True)
                htxt.append('ch=' + c + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            a = 'failed to find data with prefix ' + nmu.quotes(self.name)
            self._alert(a, tp=self._tp, quiet=quiet)
        for h in htxt:
            h = 'found data with prefix ' + nmu.quotes(self.name) + ': ' + h
            self._history(h, tp=self._tp, quiet=quiet)
        return True

    def make(self, channels=1, epochs=1, shape=[], fill_value=0, dims={},
             quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not nmu.number_ok(channels, no_neg=True, no_zero=True):
            e = 'bad channels argument: ' + str(channels)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if not nmu.number_ok(epochs, no_neg=True, no_zero=True):
            e = 'bad epochs argument: ' + str(epochs)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if not nmu.number_ok(shape, no_neg=True):
            e = 'bad shape argument: ' + str(shape)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if self.channel_count > 0 and channels != self.channel_count:
            e = ('data series ' + nmu.quotes(self.name) +
                 'requires channels=' + str(self.channel_count))
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        self.__thedata = {}
        epoch_bgn = []
        for i in range(0, channels):
            c = nmu.channel_char(i)
            dlist = self._getitems(c)  # search for existing data
            epoch_bgn.append(len(dlist))
            self.__thedata.update({c: dlist})
            if not self.channel.exists(c):
                self.channel.new(name=c, quiet=True)
        e_bgn = max(epoch_bgn)
        e_end = e_bgn + epochs
        for i in range(0, channels):
            c = nmu.channel_char(i)
            elist = []
            dlist = []
            for j in range(e_bgn, e_end):
                name2 = self.name + c + str(j)
                d = self.data.new(name=name2, shape=shape,
                                  fill_value=fill_value, quiet=True)
                if d:
                    elist.append(j)
                    dlist.append(d)
                else:
                    a = 'failed to create ' + nmu.quotes(name2)
                    self._alert(a, tp=self._tp, quiet=quiet)
            dlist2 = self.__thedata[c]
            dlist2.extend(dlist)
            self.__thedata[c] = dlist2
            h = ('created ' + nmu.quotes(self.name) + ', ' + 'ch=' + c +
                 ', ep=' + nmu.int_list_to_seq_str(elist, space=False))
            self._history(h, tp=self._tp, quiet=quiet)
        if dims:
            self.dims = dims
        return True

    def xdata_make(self, name, shape=[], dims={}, quiet=nmp.QUIET):
        if not isinstance(dims, dict):
            dims = {'xdata': None, 'xstart': 0, 'xdelta': 1, 'xlabel': '',
                    'xunits': '', 'ylabel': '', 'yunits': ''}
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        dims.update({'xstart': 0, 'xdelta': 1})  # enforce
        if 'xlabel' in dims.keys():  # switch x and y
            dims.update({'ylabel': dims['xlabel']})  # NOT DICT TYPE
            dims.update({'xlabel': ''})
        if 'xunits' in dims.keys():  # switch x and y
            dims.update({'yunits': dims['xunits']})  # NOT DICT TYPE
            dims.update({'xunits': ''})
        d = self.data.new(name=name, shape=shape, dims=dims, quiet=quiet)
        if not d:
            return None
        for i in range(0, shape):  # CHECK THIS WORKS WITH SHAPE
            d.np_array[i] = i
        self.xdata = d
        return d


class DataSeriesContainer(Container):
    """
    NM Container for DataSeries objects
    """

    def __init__(self, parent, name, fxns={}):
        t = DataSeries(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t, prefix='',
                         rename=False, duplicate=False)

    # override, no super
    @property
    def content(self):
        k = {'dataseries': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'dataseries_select': s})
        return k

    # override
    def new(self, name='', select=True, quiet=nmp.QUIET):
        # name is the data-series name
        o = DataSeries(self._parent, 'temp', self._fxns)
        ds = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if ds:
            ds.update(quiet=quiet)
            return ds
        return None
