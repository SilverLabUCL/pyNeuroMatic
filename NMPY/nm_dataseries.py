# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np

from nm_object import NMObject
from nm_object_container import NMObjectContainer
from nm_channel import NMChannelContainer
from nm_dataseries_set import NMDataSeriesSetContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, Union

ALLSTR = 'ALL'


class NMDataSeries(NMObject):
    """
    NM DataSeries class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMDataSeries',  # dataseries name/prefix
        copy: nmu.NMDataSeriesType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__channel_container = None
        self.__set_container = None
        self.__channel_scale_lock = None  # NMdata share channel x-y scales
        self.__xscale_lock = None  # all NMdata share x-scale
        self.__channel_select = None  # 'A', ['A', 'C'], 'ALL'
        self.__epoch_select = None  # 0, [0, 5], 'ALL', range(2, 6)
        self.__data_select = None  # {['A']: [NMData]}

        # self.__thedata = {}  # dict, {channel: data-list}
        # TODO: data list contained in each NMChannel

        if isinstance(copy, NMDataSeries):

            if isinstance(copy.channel, NMChannelContainer):
                self.__channel_container = copy.channel.copy()

            if isinstance(copy.sets, NMDataSeriesSetContainer):
                self.__set_container = copy.sets.copy()

            self.__channel_scale_lock = copy.channel_scale_lock
            self.__xscale_lock = copy.xscale_lock

            if isinstance(copy.channel_select, str):
                self.__channel_select = copy.channel_select
            elif isinstance(copy.channel_select, list):
                self.__channel_select = list(copy.channel_select)

            if isinstance(copy.epoch_select, int) or isinstance(
                    copy.epoch_select, str):
                self.__epoch_select = copy.epoch_select
            elif isinstance(copy.epoch_select, list):
                self.__epoch_select = list(copy.epoch_select)
            elif isinstance(copy.epoch_select, range):
                start = copy.epoch_select.start
                stop = copy.epoch_select.stop
                step = copy.epoch_select.step
                self.__epoch_select = range(start, stop, step)

            # TODO: function to create this dictionary of selected data
            # based on channel and epoch select
            # self.__data_select = {}  # new refs

        if not isinstance(self.__channel_container, NMChannelContainer):
            self.__channel_container = NMChannelContainer(
                parent=self,
                name='Channels')
            self.__channel_select = None
            self.__epoch_select = None
            self.__data_select = None

        if not isinstance(self.__set_container, NMDataSeriesSetContainer):
            self.__set_container = NMDataSeriesSetContainer(
                parent=self,
                name='DataSeriesSets')

        if not isinstance(self.__channel_scale_lock, bool):
            self.__channel_scale_lock = True

        if not isinstance(self.__xscale_lock, bool):
            self.__xscale_lock = True

        self._sets_init(quiet=True)

    # override
    def __eq__(
        self,
        other: nmu.NMObjectType
    ) -> bool:
        if not super().__eq__(other):
            return False
        # TODO: finish
        return True

    # override, no super
    def copy(self) -> nmu.NMDataSeriesType:
        return NMDataSeries(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'channel_select': self.__channel_select})
        k.update({'epoch_select': self.__epoch_select})
        # k.update({'data_select': self.__data_select})
        return k

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__channel_container.content)
        k.update({'channel_select': self.channel_select})
        k.update({'epochs': self.epoch_count})
        k.update({'epoch_select': self.epoch_select})
        k.update(self.__set_container.content)
        return k

    # override
    def _isequivalent(self, dataseries, alert=False):
        if not super()._isequivalent(dataseries, alert=alert):
            return False
        c = self.__channel_container
        c2 = dataseries._NMDataSeries__channel_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        c = self.__set_container
        c2 = dataseries._NMDataSeries__set_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        return True

#    @property
#    def folder_data(self):  # use self._folder.data
        # cannot import NMFolder class
#        if not isinstance(self._parent, NMObject):
#            return None
#        if self._parent.__class__.__name__ == 'NMFolder':
#            return self._parent.data
#       return None

    @property
    def thedata(self) -> Dict[str, List[nmu.NMDataType]]:
        if not self.__channel_container:
            return {}
        data = {}
        for i in range(0, self.__channel_container.count):
            c = self.__channel_container.getitem(i)
            data.update({c.name.upper(): len(c.data)})
        return data

    @property
    def channel_scale_lock(self) -> bool:
        if isinstance(self.__channel_scale_lock, bool):
            return self.__channel_scale_lock
        else:
            return True

    @channel_scale_lock.setter
    def channel_scale_lock(self, on: bool) -> None:
        if isinstance(on, bool):
            self.__channel_scale_lock = on
        else:
            e = self._type_error('channel_scale_lock', 'boolean')
            raise TypeError(e)

    @property
    def xscale_lock(self) -> bool:
        if isinstance(self.__xscale_lock, bool):
            return self.__xscale_lock
        else:
            return True

    @xscale_lock.setter
    def xscale_lock(self, on: bool) -> None:
        if isinstance(on, bool):
            self.__xscale_lock = on
        else:
            e = self._type_error('xscale_lock', 'boolean')
            raise TypeError(e)

    @property
    def channel(self) -> NMChannelContainer:
        return self.__channel_container

    @property
    def channel_count(self) -> int:
        if isinstance(self.__channel_container, NMChannelContainer):
            return self.__channel_container.count
        return 0

    @property
    def channel_list(self) -> List[str]:  # UPPER
        if isinstance(self.__channel_container, NMChannelContainer):
            return [c.upper() for c in self.__channel_container.names]
        else:
            return []

    def channel_ok(
        self,
        channel: Union[str, List[str]]
    ) -> bool:
        if not isinstance(channel, list):
            channel = [channel]
        for c in channel:
            if not isinstance(c, str):
                return False
            if c.upper() == ALLSTR:
                return True  # OK
            if c.upper() not in self.channel_list:  # UPPER
                return False
        return True

    @property
    def channel_select(self) -> Union[str, List[str]]:
        if isinstance(self.__channel_select, str):
            return self.__channel_select.upper()
        if isinstance(self.__channel_select, list):
            return [c.upper() for c in self.__channel_select]  # UPPER
        return None

    @channel_select.setter
    def channel_select(
        self,
        channel: Union[str, List[str]]  # 'A', 'ALL', ['A', 'C'], None
    ) -> None:
        self._channel_select_set(channel=channel)

    def _channel_select_set(
        self,
        channel: Union[str, List[str]]  # 'A', 'ALL', ['A', 'C'], None
    ) -> bool:
        clist = self.channel_list
        select = None
        if channel is None:
            pass  # OK
        elif isinstance(channel, str):
            if channel.upper == ALLSTR:
                select = ALLSTR
            elif channel.upper in clist:
                select = channel.upper()
            else:
                e = self._value_error('channel')
                raise ValueError(e)
        elif isinstance(channel, list):
            select = []
            for c in channel:
                if c.upper() == ALLSTR:
                    select = ALLSTR
                    break
                elif c.upper in clist:
                    select.append(c.upper())  # UPPER
                else:
                    channel = c
                    e = self._value_error('channel')
                    raise ValueError(e)
        else:
            e = self._type_error('channel', 'string or list')
            raise TypeError(e)
        self.__channel_select = select
        self._modified()
        self._history('channel select = ' + str(select), quiet=nmp.QUIET)
        return True

    @property
    def channel_select_List(self) -> List[str]:
        return self._channel_list(self.__channel_select)

    def _channel_list(
        self,
        channel  # 'A', 'ALL', ['A', 'C'], None
    ) -> List[str]:
        if channel is None:
            return []
        if isinstance(channel, str):
            if channel.upper() == ALLSTR:
                return self.channel_list
        if isinstance(channel, list):
            clist = self.channel_list
            clist2 = []
            for c in channel:
                if c.upper() == ALLSTR:
                    return clist
                if c.upper in clist:
                    clist2.append(c.upper())  # UPPER
            return clist2
        return []

    @property
    def epoch_count(self) -> Dict[str, int]:  # epochs per channel
        if not self.__channel_container:
            return {}
        e = {}
        for i in range(0, self.__channel_container.count):
            c = self.__channel_container.getitem(i)
            e.update({c.name.upper(): len(c.data)})
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
        self.__epoch_select = [ep for ep in self.__epoch_select if
                               isinstance(ep, int)]
        return self.__epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch: Union[int, List[int]]) -> None:
        self._epoch_select_set(epoch)

    def _epoch_select_set(
        self,
        epoch: Union[int, List[int]]
    ) -> bool:
        select = None
        if epoch is None:
            pass  # OK
        elif isinstance(epoch, int):
            pass
        elif not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        elist = []
        for ep in epoch_list:
            if self.epoch_ok(ep):
                elist.append(ep)
            else:
                epoch = ep
                e = self._value_error('epoch')
                raise ValueError(e)
        self.__epoch_select = elist
        self._modified()
        self._history('epoch select = ' + str(elist), quiet=nmp.QUIET)
        return True
    # override

    @property
    def dims(self):
        if self.__channel_scale_lock:
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
            e = self._type_error('dims', 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in nmd.DIM_LIST + ['channel']:
                raise KeyError('unknown dimensions key: ' + k)
        if 'xdata' in keys:
            self._xdata_set(dims['xdata'], quiet=quiet)
        if 'xstart' in keys:
            self._xstart_set(dims['xstart'], quiet=quiet)
        if 'xdelta' in keys:
            self._xdelta_set(dims['xdelta'], quiet=quiet)
        if 'xlabel' in keys:
            self._xlabel_set(dims['xlabel'], quiet=quiet)
        if 'xunits' in keys:
            self._xunits_set(dims['xunits'], quiet=quiet)
        if 'ylabel' in keys:
            self._ylabel_set(dims['ylabel'], quiet=quiet)
        if 'yunits' in keys:
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
        elif xdata.__class__.__name__ != 'NMData':  # cannot import Data class
            e = self._type_error('xdata', 'NMData')
            raise TypeError(e)
        old = self.xdata
        # if xdata == old:
        #    return True
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d._xdata_set(xdata, alert=False, quiet=True)
        self.__dims = {}  # reset
        new = self.xdata
        h = nmu.history_change('xdata', old, new)
        self._history(h, quiet=quiet)
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
            e = self._type_error('xstart', 'number')
            raise TypeError(e)
        if not nmu.number_ok(xstart):
            e = self._value_error('xstart')
            raise ValueError(e)
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
            e = self._type_error('xlable', 'string')
            raise TypeError(e)
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
            e = self._type_error('xunits', 'string')
            raise TypeError(e)
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
            e = self._type_error('chan_ylabel', 'channel dictionary')
            raise TypeError(e)
        for cc, ylabel in chan_ylabel.items():
            if not isinstance(ylabel, str):
                e = self._type_error('ylabel', 'string')
                raise TypeError(e)
            # elif c not in self.__thedata.keys():
            if not isinstance(cc, str):
                channel = cc
                e = self._type_error('channel', 'character')
                raise TypeError(e)
            cc2 = nmu.chanel_char_check(cc)
            if not cc2:
                channel = cc
                e = self._value_error('channel')
                raise ValueError(e)
            if cc2 in self.__thedata.keys():
                cdata = self.__thedata[cc2]
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
            e = self._type_error('chan_yunits', 'channel dictionary')
            raise TypeError(e)
        for c, yunits in chan_yunits.items():
            if not isinstance(yunits, str):
                e = self._type_error('yunits', 'string')
                raise TypeError(e)
            # elif c not in self.__thedata.keys():
            elif not nmu.channel_char_check(c):
                channel = c
                e = self._value_error('channel')
                raise ValueError(e)
            if c in self.__thedata.keys():
                cdata = self.__thedata[c]
                for d in cdata:
                    d.yunits = yunits
        return True

    @property
    def sets(self):
        return self.__set_container

    def _sets_init(self, set_list=nmp.DATASERIES_SET_LIST, select=True,
                   quiet=nmp.QUIET):
        if not set_list:
            return []
        if not isinstance(set_list, list):
            set_list = [set_list]
        r = []
        init_select = select or not self.__set_container.select
        for s in nmp.DATASERIES_SET_LIST:
            if not s or not isinstance(s, str):
                continue
            select = init_select and s.upper() == ALLSTR
            if self.__set_container.new(name=s, select=select, quiet=quiet):
                r.append(s)
        if init_select and not self.__set_container.select:
            self.__set_container.select = set_list[0]
        return r

    def get_data_names(self, chan_list=ALLSTR, epoch_list=[-2],
                       quiet=nmp.QUIET):
        d = self.get_data(chan_list=chan_list, epoch_list=epoch_list,
                          quiet=quiet)
        n = {}
        for c, cdata in d.items():
            nlist = [d.name for d in cdata]
            n.update({c: nlist})
        return n

    def get_data(self, chan_list=ALLSTR, epoch_list=[-2], quiet=nmp.QUIET):
        if not self.__thedata:
            return {}
        clist = self._channel_list(chan_list=chan_list)
        if not clist:
            return {}
        if not isinstance(epoch_list, list):
            epoch_list = [epoch_list]
        all_epochs = False
        elist = []
        for ep in epoch_list:
            if not isinstance(ep, int):
                epoch = ep
                e = self._type_error('epoch', 'integer')
                raise TypeError(e)
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
                        epoch = ep
                        e = self._value_error('epoch')
                        raise ValueError(e)
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
        clist = self.channel_select_list
        if not clist:
            return {}
        if not self.sets.select or not self.sets.select.name:
            return {}
        if not self.sets.select.theset:
            return {}
        sname = self.sets.select.name
        theset = self.sets.select.theset
        sx = self.sets.getitem('SetX')
        if sx:
            setx = sx.theset
        else:
            setx = None
        all_epochs = sname.upper() == ALLSTR
        dd = {}
        for c in clist:
            cdata = self.__thedata[c]
            dlist = []
            for d in cdata:
                if setx and d in setx:
                    continue
                if all_epochs or d in theset:
                    dlist.append(d)
            dd.update({c: dlist})
        return dd

    def _getitems(self, chan_char):
        thedata = self._folder.data._NMObjectContainer__container  # mangled
        dlist = []
        i = len(self.name)
        for o in thedata:
            if o.name[:i].casefold() == self.name.casefold():
                if chan_char:
                    if nmu.channel_char_search(o.name[i:], chan_char) >= 0:
                        dlist.append(o)
                else:
                    dlist.append(o)
        return dlist

    def update(self, quiet=nmp.QUIET):
        foundsomething = False
        htxt = []
        self.__thedata = {}
        for i in range(0, 25):
            cc = nmu.channel_char(i)
            olist = self._getitems(cc)
            if len(olist) > 0:
                self.__thedata.append(olist)
                foundsomething = True
                if not self.channel.exists(cc):
                    self.channel.new(name=cc, quiet=True)
                htxt.append('ch=' + cc + ', n=' + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            a = 'failed to find data with prefix ' + nmu.quotes(self.name)
            self._alert(a, quiet=quiet)
        for h in htxt:
            h = 'found data with prefix ' + nmu.quotes(self.name) + ': ' + h
            self._history(h, quiet=quiet)
        return True

    def make(self, channels=1, epochs=1, shape=[], fill_value=0, dims={},
             quiet=nmp.QUIET):
        if not nmu.number_ok(channels, no_neg=True, no_zero=True):
            e = self._value_error('channels')
            raise ValueError(e)
        if not nmu.number_ok(epochs, no_neg=True, no_zero=True):
            e = self._value_error('epochs')
            raise ValueError(e)
        if not nmu.number_ok(shape, no_neg=True):
            e = self._value_error('shape')
            raise ValueError(e)
        if self.channel_count > 0 and channels != self.channel_count:
            e = self._error('data series ' + nmu.quotes(self.name) +
                            'requires channels=' + str(self.channel_count))
            raise ValueError(e)
        self.__thedata = {}
        epoch_bgn = []
        for i in range(0, channels):
            cc = nmu.channel_char(i)
            dlist = self._getitems(cc)  # search for existing data
            epoch_bgn.append(len(dlist))
            self.__thedata.update({cc: dlist})
            if not self.channel.exists(cc):
                self.channel.new(name=cc, quiet=True)
        e_bgn = max(epoch_bgn)
        e_end = e_bgn + epochs
        for i in range(0, channels):
            cc = nmu.channel_char(i)
            elist = []
            dlist = []
            for j in range(e_bgn, e_end):
                name2 = self.name + cc + str(j)
                d = self._folder.data.new(name=name2,
                                         shape=shape,
                                         fill_value=fill_value,
                                         quiet=True)
                if d:
                    elist.append(j)
                    dlist.append(d)
                else:
                    a = 'failed to create ' + nmu.quotes(name2)
                    self._alert(a, quiet=quiet)
            dlist2 = self.__thedata[cc]
            dlist2.extend(dlist)
            self.__thedata[cc] = dlist2
            h = ('created ' + nmu.quotes(self.name) + ', ' + 'ch=' + cc +
                 ', ep=' + nmu.int_list_to_seq_str(elist, seperator=','))
            self._history(h, quiet=quiet)
        if dims:
            self.dims = dims
        return True

    def xdata_make(self, name, shape=[], dims={}, quiet=nmp.QUIET):
        if not isinstance(dims, dict):
            e = self._type_error('dims', 'dimensions dictionary')
            raise TypeError(e)
        dims.update({'xstart': 0, 'xdelta': 1})  # enforce
        if 'xlabel' in dims.keys():  # switch x and y
            dims.update({'ylabel': dims['xlabel']})  # NOT DICT TYPE
            dims.update({'xlabel': ''})
        if 'xunits' in dims.keys():  # switch x and y
            dims.update({'yunits': dims['xunits']})  # NOT DICT TYPE
            dims.update({'xunits': ''})
        d = self._folder.data.new(name=name, shape=shape, dims=dims,
                                 quiet=quiet)
        if not d:
            return None
        for i in range(0, shape):  # CHECK THIS WORKS WITH SHAPE
            d.np_array[i] = i
        self.xdata = d
        return d


class NMDataSeriesContainer(NMObjectContainer):
    """
    NM Container for DataSeries objects
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMDataSeriesContainer',
        copy: nmu.NMDataSeriesContainerType = None
    ):
        o = NMDataSeries(parent=parent, name='ContainerUtility')
        prefix = ''  # no prefix
        super().__init__(parent=parent, name=name, nmobject=o, prefix=prefix,
                         rename=False, copy=copy)
        # TODO: copy

    # override, no super
    def copy(self) -> nmu.NMDataSeriesContainerType:
        return NMDataSeriesContainer(copy=self)

    # override
    def new(
        self,
        name: str = '',  # dataseries name/prefix
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> NMDataSeries:
        ds = super().new(name=name, select=select, quiet=quiet)
        # TODO: change to super.add?
        if isinstance(ds, NMDataSeries):
            ds.update(quiet=quiet)
            return ds
        return None

    # @property
    # def data(self):  # use self._folder.data
    #    if self._parent.__class__.__name__ == 'NMFolder':
    #        return self._parent.data
    #    return None

    # override, no super
    def duplicate(self):
        e = self._error('dataseries cannot be duplicated')
        raise RuntimeError(e)
