# -*- coding: utf-8 -*-
"""
[Module description].

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source 
Software Toolkit for Acquisition, Analysis and Simulation of 
Electrophysiological Data. Front. Neuroinform. 12:14. 
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations
import copy
import numpy as np

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_channel import NMChannelContainer
from pyneuromatic.core.nm_epoch import NMEpochContainer
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


ALLSTR = "all".upper()

"""
NM class tree:

NMManager
    NMProject (root)
        NMFolderContainer
            NMFolder (folder0, folder1...)
                NMDataContainer
                    NMData (recordA0, recordA1... avgA0, avgB0)
                NMDataSeriesContainer
                    NMDataSeries (record, avg...)
                        NMChannelContainer
                            NMChannel (A, B, C...)
                        NMEpochContainer
                            NMEpoch (E0, E1, E2...)
DataSeries:
      E0  E1  E2... (epochs)
Ch A  A0  A1  A2...
Ch B  B0  B1  B2...
.
.
.
"""


class NMDataSeries(NMObject):
    """
    NM DataSeries class
    """

    # Extend NMObject's special attrs with NMDataSeries's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMDataSeries__channel_container",
        "_NMDataSeries__epoch_container",
        "_NMDataSeries__channel_scale_lock",
        "_NMDataSeries__xscale_lock",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataSeries0",  # dataseries name/prefix
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__channel_container: NMChannelContainer = NMChannelContainer(parent=self)
        self.__epoch_container: NMEpochContainer = NMEpochContainer(parent=self)
        self.__channel_scale_lock: bool = True  # NMdata share channel x-y scales
        self.__xscale_lock: bool = True  # all NMdata share x-scale

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMDataSeries):
            return NotImplemented
        # TODO
        return super().__eq__(other)

    def __deepcopy__(self, memo: dict) -> NMDataSeries:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMDataSeries by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMDataSeries
        """
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMDataSeries's special attributes

        # __channel_container: deep copy and update parent
        if isinstance(self._NMDataSeries__channel_container, NMChannelContainer):
            result._NMDataSeries__channel_container = copy.deepcopy(
                self._NMDataSeries__channel_container, memo
            )
            result._NMDataSeries__channel_container._parent = result
        else:
            result._NMDataSeries__channel_container = NMChannelContainer(parent=result)

        # __epoch_container: deep copy and update parent
        if isinstance(self._NMDataSeries__epoch_container, NMEpochContainer):
            result._NMDataSeries__epoch_container = copy.deepcopy(
                self._NMDataSeries__epoch_container, memo
            )
            result._NMDataSeries__epoch_container._parent = result
        else:
            result._NMDataSeries__epoch_container = NMEpochContainer(parent=result)

        # __channel_scale_lock: simple bool copy
        result._NMDataSeries__channel_scale_lock = self._NMDataSeries__channel_scale_lock

        # __xscale_lock: simple bool copy
        result._NMDataSeries__xscale_lock = self._NMDataSeries__xscale_lock

        return result

    # TODO: replace with __eq__()
    def _isequivalent(self, dataseries, alert=False):
        if not super()._isequivalent(dataseries, alert=alert):
            return False
        c = self.channels
        c2 = dataseries._NMDataSeries__channel_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        # c = self.__set_container
        # c2 = dataseries._NMDataSeries__set_container
        # if c and not c._isequivalent(c2, alert=alert):
        #    return False
        return True

    # override
    # TODO: finish
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        # k.update({'channel_select': self.__channel_select})
        # k.update({'epoch_select': self.__epoch_select})
        # k.update({'data_select': self.__data_select})
        return k

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        k.update(self.channels.content)
        k.update(self.epochs.content)
        return k

    @property
    def channels(self) -> NMChannelContainer:
        return self.__channel_container

    @property
    def epochs(self) -> NMEpochContainer:
        return self.__epoch_container

    def get_select(
        self, 
        get_keys: bool = False
    ) -> list[NMObject] | list[str]:
        if not self.channels.selected_name:
            return []
        c = self.channels.selected_value
        if c is None:
            return []
        if not self.epochs.selected_name:
            return []
        e = self.epochs.selected_value
        if e is None:
            return []

        dlist = []
        for d in e.data:
            if d in c.data:
                if get_keys:
                    dlist.append(d.name)
                else:
                    dlist.append(d)
        return dlist

    @property
    def channel_scale_lock(self) -> bool:
        if isinstance(self.__channel_scale_lock, bool):
            return self.__channel_scale_lock
        else:
            return True

    @channel_scale_lock.setter
    def channel_scale_lock(
        self,
        on: bool
    ) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "channel_scale_lock", "boolean")
            raise TypeError(e)
        self.__channel_scale_lock = on

    @property
    def xscale_lock(self) -> bool:
        if isinstance(self.__xscale_lock, bool):
            return self.__xscale_lock
        else:
            return True

    @xscale_lock.setter
    def xscale_lock(
        self, 
        on: bool
    ) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "xscale_lock", "boolean")
            raise TypeError(e)
        self.__xscale_lock = on

    @property
    def dims(self):
        if self.__channel_scale_lock:
            return self.__dims
        return self._dims_of_thedata

    @dims.setter
    def dims(
        self, 
        dims
    ):
        return self._dims_set(dims)

    def _dims_set(
        self, 
        dims, 
        quiet=False
    ):
        if not isinstance(dims, dict):
            e = nmu.type_error_str(dims, "dims", "dimensions dictionary")
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in ["xdata", "xstart", "xdelta", "xlabel", "xunits", "ylabel", "yunits"]:
                raise KeyError("unknown dimensions key: " + k)
        if "xdata" in keys:
            self._xdata_set(dims["xdata"], quiet=quiet)
        if "xstart" in keys:
            self._xstart_set(dims["xstart"], quiet=quiet)
        if "xdelta" in keys:
            self._xdelta_set(dims["xdelta"], quiet=quiet)
        if "xlabel" in keys:
            self._xlabel_set(dims["xlabel"], quiet=quiet)
        if "xunits" in keys:
            self._xunits_set(dims["xunits"], quiet=quiet)
        if "ylabel" in keys:
            self._ylabel_set(dims["ylabel"], quiet=quiet)
        if "yunits" in keys:
            self._yunits_set(dims["yunits"], quiet=quiet)
        return True

    @property
    def xdata(self):
        k = "xdata"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return None

    @xdata.setter
    def xdata(
        self, 
        xdata
    ):
        return self._xdata_set(xdata)

    def _xdata_set(
        self, 
        xdata, 
        quiet=False
    ):
        if xdata is None:
            pass  # ok
        elif xdata.__class__.__name__ != "NMData":  # cannot import Data class
            e = nmu.type_error_str(xdata, "xdata", "NMData")
            raise TypeError(e)
        old = self.xdata
        # if xdata == old:
        #    return True
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d._xdata_set(xdata, alert=False, quiet=True)
        self.__dims = {}  # reset
        new = self.xdata
        # h = nmh.history_change_str("xdata", old, new)
        # nmh.history(h, quiet=quiet)
        return True

    @property
    def xstart(self):
        k = "xstart"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return 0

    @xstart.setter
    def xstart(
        self, 
        xstart
    ):
        return self._xstart_set(xstart)

    def _xstart_set(
        self, 
        xstart, 
        quiet=False
    ):
        if not isinstance(xstart, float) and not isinstance(xstart, int):
            e = nmu.type_error_str(xstart, "xstart", "number")
            raise TypeError(e)
        if not nmu.number_ok(xstart):
            raise ValueError("xstart: %s" % xstart)
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d._xstart_set(xstart, alert=False, quiet=True)
        return True

    @property
    def xdelta(self):
        k = "xdelta"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return 1

    @xdelta.setter
    def xdelta(
        self, 
        xdelta
    ):
        return self._xdelta_set(xdelta)

    def _xdelta_set(
        self, 
        xdelta, 
        quiet=False
    ):
        if np.isinf(xdelta) or np.isnan(xdelta):
            return False
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xdelta = xdelta
        return True

    @property
    def xlabel(self):
        k = "xlabel"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ""

    @xlabel.setter
    def xlabel(
        self, 
        xlabel
    ):
        return self._xlabel_set(xlabel)

    def _xlabel_set(
        self, 
        xlabel, 
        quiet=False
    ):
        if not isinstance(xlabel, str):
            e = nmu.type_error_str(xlabel, "xlabel", "string")
            raise TypeError(e)
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xlabel = xlabel
        return True

    @property
    def xunits(self):
        k = "xunits"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ""

    @xunits.setter
    def xunits(
        self, 
        xunits
    ):
        return self._xunits_set(xunits)

    def _xunits_set(
        self, 
        xunits, 
        quiet=False
    ):
        if not isinstance(xunits, str):
            e = nmu.type_error_str(xunits, "xunits", "string")
            raise TypeError(e)
        for c, cdata in self.__thedata.items():
            for d in cdata:
                d.xunits = xunits
        return True

    @property
    def ylabel(self):
        k = "ylabel"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ""

    @ylabel.setter
    def ylabel(
        self, 
        chan_ylabel
    ):
        return self._ylabel_set(chan_ylabel)

    def _ylabel_set(
        self, 
        chan_ylabel, 
        quiet=False
    ):
        if isinstance(chan_ylabel, str):
            chan_ylabel = {"A": chan_ylabel}
        elif not isinstance(chan_ylabel, dict):
            e = nmu.type_error_str(chan_ylabel, "chan_ylabel", "channel dictionary")
            raise TypeError(e)
        for cc, ylabel in chan_ylabel.items():
            if not isinstance(ylabel, str):
                e = nmu.type_error_str(ylabel, "ylabel", "string")
                raise TypeError(e)
            # elif c not in self.__thedata.keys():
            if not isinstance(cc, str):
                e = nmu.type_error_str(cc, "channel", "character")
                raise TypeError(e)
            cc2 = nmu.chanel_char_check(cc)
            if not cc2:
                raise ValueError("channel: %s" % cc)
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
        k = "yunits"
        if k in self.__dims.keys():
            return self.__dims[k]
        self.__dims = self._dims_of_thedata()
        if k in self.__dims.keys():
            return self.__dims[k]
        return ""

    @yunits.setter
    def yunits(
        self, 
        chan_yunits
    ):
        return self._yunits_set(chan_yunits)

    def _yunits_set(
        self, 
        chan_yunits, 
        quiet=False
    ):
        if isinstance(chan_yunits, str):
            chan_yunits = {"A": chan_yunits}
        elif not isinstance(chan_yunits, dict):
            e = nmu.type_error_str(chan_yunits, "chan_yunits", "channel dictionary")
            raise TypeError(e)
        for c, yunits in chan_yunits.items():
            if not isinstance(yunits, str):
                e = nmu.type_error_str(yunits, "yunits", "string")
                raise TypeError(e)
            # elif c not in self.__thedata.keys():
            elif not nmu.channel_char_check(c):
                raise ValueError("channel: %s" % c)
            if c in self.__thedata.keys():
                cdata = self.__thedata[c]
                for d in cdata:
                    d.yunits = yunits
        return True

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
        dims = {"xdata": xdata}
        dims.update({"xstart": xstart, "xdelta": xdelta})
        dims.update({"xlabel": xlabel, "xunits": xunits})
        dims.update({"ylabel": ylabel, "yunits": yunits})
        return dims
    
    # @property
    # def sets(self):
    #    return self.__set_container
    """
    def _sets_init(self, set_list=nmp.DATASERIES_SET_LIST, select=True,
                   quiet=False):
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
    """

    def get_data_names(
        self, 
        chan_list=ALLSTR, 
        epoch_list=[-2], 
        quiet=False
    ):
        d = self.get_data(chan_list=chan_list, epoch_list=epoch_list, quiet=quiet)
        n = {}
        for c, cdata in d.items():
            nlist = [d.name for d in cdata]
            n.update({c: nlist})
        return n

    def get_data(
        self, 
        chan_list=ALLSTR, 
        epoch_list=[-2], 
        quiet=False
    ):
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
                e = nmu.type_error_str(ep, "epoch", "integer")
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
                        raise ValueError("epoch: %s" % ep)
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
        # TODO: remove sets.select
        clist = self.channel_select_list
        if not clist:
            return {}
        if not self.sets.select or not self.sets.select.name:
            return {}
        if not self.sets.select.theset:
            return {}
        sname = self.sets.select.name
        theset = self.sets.select.theset
        sx = self.sets.getitem("SetX")
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

    def _getitems(
        self, 
        chan_char
    ):
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

    def update(
        self, 
        quiet=False
    ):
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
                htxt.append("ch=" + cc + ", n=" + str(len(olist)))
            else:
                break  # no more channels
        if not foundsomething:
            a = "failed to find data with prefix '%s'" % self.name
            nmh.alert(a, quiet=quiet)
        # for h in htxt:
            # h = "found data with prefix '%s' : %s" % (self.name, h)
            # nmh.history(h, quiet=quiet)
        return True

    def make(
        self, 
        channels=1, 
        epochs=1, 
        shape=[], 
        fill_value=0, 
        dims={}, 
        quiet=False
    ):
        if not nmu.number_ok(channels, no_neg=True, no_zero=True):
            raise ValueError("channels: %s" % channels)
        if not nmu.number_ok(epochs, no_neg=True, no_zero=True):
            raise ValueError("epochs: %s" % epochs)
        if not nmu.number_ok(shape, no_neg=True):
            raise ValueError("shape: %s" % shape)
        if self.channel_count > 0 and channels != self.channel_count:
            e = "data series '%s' requires channels = %s" % (
                self.name,
                self.channel_count,
            )
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
                d = self._folder.data.new(
                    name=name2, shape=shape, fill_value=fill_value, quiet=True
                )
                if d:
                    elist.append(j)
                    dlist.append(d)
                else:
                    a = "failed to create '%s'" % name2
                    nmh.alert(a, quiet=quiet)
            dlist2 = self.__thedata[cc]
            dlist2.extend(dlist)
            self.__thedata[cc] = dlist2
            ep = nmu.int_list_to_seq_str(elist, seperator=",")
            # h = "created '%s', ch=%s, ep=%s" % (self.name, cc, ep)
            # nmh.history(h, quiet=quiet)
        if dims:
            self.dims = dims
        return True

    def xdata_make(
        self, 
        name, 
        shape=[], 
        dims={}, 
        quiet=False
    ):
        if not isinstance(dims, dict):
            e = nmu.type_error_str(dims, "dims", "dimensions dictionary")
            raise TypeError(e)
        dims.update({"xstart": 0, "xdelta": 1})  # enforce
        if "xlabel" in dims.keys():  # switch x and y
            dims.update({"ylabel": dims["xlabel"]})  # NOT DICT TYPE
            dims.update({"xlabel": ""})
        if "xunits" in dims.keys():  # switch x and y
            dims.update({"yunits": dims["xunits"]})  # NOT DICT TYPE
            dims.update({"xunits": ""})
        d = self._folder.data.new(name=name, shape=shape, dims=dims, quiet=quiet)
        if not d:
            return None
        for i in range(0, shape):  # CHECK THIS WORKS WITH SHAPE
            d.np_array[i] = i
        self.xdata = d
        return d


class NMDataSeriesContainer(NMObjectContainer):
    """
    Container of NMDataSeries
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataSeriesContainer0",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=False,
            auto_name_prefix="",  # no prefix
            auto_name_seq_format="",
        )

    # override, no super
    def content_type(self) -> str:
        return NMDataSeries.__name__

    # override
    def new(
        self,
        name: str = "",  # dataseries name/prefix
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMDataSeries | None:
        name = self._newkey(name)
        s = NMDataSeries(parent=self, name=name)
        if super()._new(s, select=select):
            return s
        return None
    # @property
    # def data(self):  # use self._folder.data
    #    if self._parent.__class__.__name__ == 'NMFolder':
    #        return self._parent.data
    #    return None

    # override, no super
    def duplicate(self):
        raise RuntimeError("dataseries cannot be duplicated")
