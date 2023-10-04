#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from pyneuromatic.nm_object import NMObject
from pyneuromatic.nm_object_container import NMObjectContainer
import pyneuromatic.nm_preferences as nmp
import pyneuromatic.nm_utilities as nmu
from typing import Dict, List

SET_SYMBOLS = ["&", "|", "-", "^"]


class NMSet(NMObject):
    """
    NM DataSeriesSet class
    Behaves like a Python set, but is a list
    Using lists instead of sets because sets have no order
    For data series, want to keep items in order
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMSet",
        copy: nmu.NMSetType = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        if isinstance(copy, NMSet):  # TODO copy
            self.__theset = {}
            self.__sort_template = None
            self.__eq_lock = []
        else:
            self.__theset = {}  # dictionary of sets, keys are chan-char
            self.__sort_template = None
            self.__eq_lock = []

    # override, no super
    def copy(self) -> nmu.NMSetType:
        return NMSet(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        if isinstance(self.__sort_template, NMSet):
            k.update({"sort_template": self.__sort_template.name})
            # need name for isequivalent() to work
        else:
            k.update({"sort_template": str(None)})
        if self.__eq_lock:
            k.update({"eq_lock": self.equation_lock_str})
            # need nmobject names for isequivalent() to work
        else:
            k.update({"eq_lock": str(None)})
        return k

    # override
    @property
    def _bad_names(self):
        bn = []
        if isinstance(nmp.BADNAMES, list):
            bn = list(nmp.BADNAMES)
            bn.remove("all")
        return bn

    # TODO: replace with __eq__()
    def _isequivalent(self, dataseriesset, alert=False):
        if not super()._isequivalent(dataseriesset, alert=alert):
            return False
        ue = "unequivalent "
        ks = self.__theset.keys()
        ko = dataseriesset._theset_keys()
        if ks != ko:
            if alert:
                a = ue + "channels: " + str(ks) + " vs " + str(ko)
                self._alert(a)
            return False
        for cc, dlist_o in dataseriesset._theset_items():
            dlist_s = self.__theset[cc]
            ns = len(dlist_s)
            no = len(dlist_o)
            if ns != no:
                if alert:
                    a = (
                        ue
                        + "data items in ch "
                        + cc
                        + ": n = "
                        + str(ns)
                        + " vs "
                        + str(no)
                    )
                    self._alert(a)
                return False
            for i in range(0, ns):
                if not dlist_s[i]._isequivalent(dlist_o[i], alert=alert):
                    if alert:
                        nms = dlist_s[i].name
                        nmo = dlist_o[i].name
                        a = (
                            ue
                            + "ch "
                            + cc
                            + ", item #"
                            + str(i)
                            + ": "
                            + nms
                            + " vs "
                            + nmo
                        )
                        self._alert(a)
                    return False
        if self.__sort_template != dataseriesset.sort_template:
            if alert:
                if isinstance(self.__sort_template, NMSet):
                    nms = self.__sort_template.name
                else:
                    nms = "None"
                if isinstance(dataseriesset.sort_template, NMSet):
                    nmo = dataseriesset.sort_template.name
                else:
                    nmo = "None"
                a = ue + "sort templates: " + nms + " vs " + nmo
                self._alert(a)
            return False
        elock_o = dataseriesset.equation_lock
        if not self.__eq_lock and not elock_o:
            return True
        ns = len(self.__eq_lock)
        no = len(elock_o)
        if ns != no:
            a = ue + "items in equation lock: " + str(ns) + " vs " + str(no)
            self._alert(a)
            return False
        for i in range(0, ns, 2):
            ss = self.__eq_lock[i]
            so = elock_o[i]
            if not ss._isequivalent(so, alert=alert):
                if alert:
                    a = (
                        ue
                        + "eq. lock item #"
                        + str(i)
                        + ": "
                        + str(ss)
                        + " vs "
                        + str(so)
                    )
                    self._alert(a)
                return False
        for i in range(1, ns, 2):
            ss = self.__eq_lock[i]
            so = elock_o[i]
            if ss != so:
                if alert:
                    a = ue + "eq. lock item #" + str(i) + ": " + ss + " vs " + so
                    self._alert(a)
                return False
        return True

    @property
    def sort_template(self):
        return self.__sort_template

    @sort_template.setter
    def sort_template(self, dataseriesset) -> None:
        self._sort_template_set(dataseriesset)

    def _sort_template_set(self, dataseriesset, quiet=nmp.QUIET):
        if isinstance(dataseriesset, NMSet):
            self.__sort_template = dataseriesset
            h = "sort template = " + dataseriesset._tp
            self._history(h, quiet=quiet)
            return True
        else:
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)

    def _sort_update(self):
        if isinstance(self.__sort_template, NMSet):
            return self.sort(self.__sort_template)
        return False

    def sort(self, dataseriesset_template):
        if not isinstance(dataseriesset_template, NMSet):
            e = self._type_error("dataseriesset_template", "NMSet")
            raise TypeError(e)
        s = {}
        for cc, dlist in dataseriesset_template._theset_items():
            if cc in self.__theset:
                dlist_s = []
                for d in dlist:
                    if d in self.__theset[cc]:
                        dlist_s.append(d)
                for d in self.__theset[cc]:
                    if d not in dlist:
                        dlist_s.append(d)
                s.update({cc: dlist_s})
        modified = False
        for cc, dlist in s.items():
            self.__theset.update({cc: dlist})
            modified = True
        return modified

    @property
    def equation(self):
        return []  # nothing to return as this is one-off

    @equation.setter
    def equation(self, eq_list) -> None:  # one-off execution
        self._equation(eq_list, lock=False)

    @property
    def equation_lock(self):
        return self.__eq_lock

    @equation_lock.setter
    def equation_lock(self, eq_list) -> None:
        self._equation(eq_list, lock=True)

    @property
    def equation_lock_str(self):
        elist = []
        for i in self.__eq_lock:
            if isinstance(i, NMSet):
                elist.append(i.name)
            else:
                elist.append(i)
        return " ".join(elist)

    @property
    def _eq_lock_error(self):
        e = "this set is locked to equation: " + self.equation_lock_str
        return self._error(e)

    def _eq_lock_update(self):
        if self.__eq_lock:
            return self._equation(self.__eq_lock, lock=True)
        return False

    @property
    def locked(self):
        if self.__eq_lock:
            return True
        return False

    @locked.setter
    def locked(self, lock) -> None:
        if lock:
            e = self._error(
                "create a locked equation via " + nmu.quotes("equation_lock")
            )
            raise ValueError(e)
        else:
            self.__eq_lock = []

    def _eq_list_check(self, eq_list, lock):
        if eq_list is None:
            return []
        if isinstance(eq_list, list):
            if not eq_list:
                return []
        elif isinstance(eq_list, NMSet):
            eq_list = [eq_list]
        else:
            e = self._type_error("eq_list", "list")
            raise TypeError(e)
        if len(eq_list) % 2 == 0:
            e = self._error("equation list should have odd number of items")
            raise ValueError(e)
        for i in range(0, len(eq_list), 2):
            s = eq_list[i]
            if not isinstance(s, NMSet):
                e = self._error("bad equation list: even items should be " + "a NMSet")
                raise TypeError(e)
            if lock and s == self:
                e = self._error(
                    "sets cannot have locked equations that " + "contain themself"
                )
                raise ValueError(e)
        for i in range(1, len(eq_list), 2):
            symbol = eq_list[i]
            if not isinstance(symbol, str):
                e = self._type_error("symbol", "string")
                raise TypeError(e)
            if symbol not in SET_SYMBOLS:
                slist = [nmu.quotes(ss) for ss in SET_SYMBOLS]
                e = self._error(
                    "bad equation list: odd items should be "
                    + "one of the following: "
                    + ", ".join(slist)
                )
                raise ValueError(e)
        return eq_list  # ok

    def _equation(self, eq_list, lock=False, quiet=nmp.QUIET):
        eq_list = self._eq_list_check(eq_list, lock)
        if not eq_list:
            self.__eq_lock = []
            h = "unlocked"
            self._history(h, quiet=quiet)
            return True
        if not lock and self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        c = eq_list[0].copy()
        hlist = []
        hlist.append(c.name)
        for i in range(1, len(eq_list), 2):
            s2 = eq_list[i + 1]
            if eq_list[i] == "&":
                c.intersection(s2, quiet=quiet)
            elif eq_list[i] == "|":
                c.union(s2, quiet=quiet)
            elif eq_list[i] == "-":
                c.difference(s2, quiet=quiet)
            elif eq_list[i] == "^":
                c.symmetric_difference(s2, quiet=quiet)
            hlist.append(eq_list[i])
            hlist.append(s2.name)
        self.__theset.clear()
        for cc, dlist in c._theset_items():
            self.__theset.update({cc: dlist})
        self._sort_update()
        if lock:
            self.__eq_lock = eq_list
        else:
            self.__eq_lock = []
        h = self.name + " = " + " ".join(hlist)
        if lock:
            h += " (locked)"
        else:
            h += " (not locked)"
        self._history(h, quiet=quiet)
        return True

    @property
    def channel_count(self):
        return len(self.__theset)

    @property
    def epoch_count(self):
        count = 0
        for dlist in self.__theset.values():
            count = max(count, len(dlist))
        return count

    @property
    def data_names(self):
        if self.name.upper() == "ALL":
            return {"ALL"}
        n = {}
        for cc, dlist in self.__theset.items():
            nlist = [d.name for d in dlist]
            n.update({cc: nlist})
        return n

    @property
    def isempty(self):
        if not self.__theset:
            return True
        for dlist in self.__theset.values():
            if len(dlist) > 0:
                return False
        return True

    def _theset_clear_if_empty(self):
        for dlist in self.__theset.values():
            if len(dlist) > 0:
                return False
        self.__theset.clear()
        return True

    def _theset_copy(self):
        # cannot do simple dictionary copy as data list is not a copy
        c = {}
        for cc, dlist in self.__theset.items():
            c.update({cc: dlist.copy()})  # enforce copy of data list
        return c

    def _theset_items(self):
        return self.__theset.items()

    def _theset_keys(self):
        return self.__theset.keys()

    def _theset_values(self):
        return self.__theset.values()

    def _data_dict_check(self, data_dict, chan_default=nmu.CHANNEL_LIST[0]):
        # data_dict = MyData -converted-> {'A': [MyData]}
        # data_dict = [MyData0, MyData1] -converted-> {'A': [MyData0, MyData1]}
        # data_dict = {'A': MyDataA0, 'B': MyDataB0} -converted->
        #        {'A': [MyDataA0], 'B': [MyDataB0]}
        # data_dict = {'A': [MyDataA0, MyDataA1], 'B': [MyDataB0, MyDataB1]}
        # data_dict format = {chan_char: [Data List]}
        if not isinstance(chan_default, str):
            e = self._type_error("chan_default", "channel character")
            raise TypeError(e)
        dnew = {}
        edict = "dictionary with channel keys"
        if data_dict.__class__.__name__ == "NMData":  # no channel
            if chan_default.upper() == "ALL_EXISTING":
                for cc in self.__theset.keys():
                    dnew.update({cc.upper(): [data_dict]})
                return dnew
            if len(self.__theset.keys()) > 1:
                e = self._type_error("data_dict", edict)
                raise TypeError(e)
            cc = nmu.channel_char_check(chan_default)
            if not cc:
                channel = chan_default
                e = self._value_error("channel")
                raise ValueError(e)
            dnew.update({cc: [data_dict]})
            return dnew
        if isinstance(data_dict, list):
            if len(data_dict) == 0:
                return {}
            for data_item in data_dict:
                if data_item.__class__.__name__ != "NMData":
                    e = self._type_error("data_item", "NMData")
                    raise TypeError(e)
            if chan_default.upper() == "ALL_EXISTING":
                for cc in self.__theset.keys():
                    dnew.update({cc.upper(): data_dict})
                return dnew
            if len(self.__theset.keys()) > 1:
                e = self._type_error("data_dict", edict)
                raise TypeError(e)
            cc = nmu.channel_char_check(chan_default)
            if not cc:
                channel = chan_default
                e = self._value_error("channel")
                raise ValueError(e)
            dnew.update({cc: data_dict})
            return dnew
        if not isinstance(data_dict, dict):
            e = self._type_error("data_dict", edict)
            raise TypeError(e)
        for cc, dlist in data_dict.items():
            cc2 = nmu.channel_char_check(cc)
            if not cc2:
                channel = cc
                e = self._value_error("channel")
                raise ValueError(e)
            if dlist.__class__.__name__ == "NMData":
                dlist = [dlist]
            elif not isinstance(dlist, list):
                data_list = dlist
                e = self._type_error("data_list", "list of Data")
                raise TypeError(e)
            if len(dlist) == 0:
                continue
            for data in dlist:
                if data.__class__.__name__ != "NMData":
                    e = self._type_error("data", "NMData")
                    raise TypeError(e)
            dnew.update({cc2: dlist})
        return dnew

    def _chan_list_check(self, chan_list):
        # chan_list = ['A', 'B']
        if isinstance(chan_list, str):
            chan_list = [chan_list]
        elif not isinstance(chan_list, list):
            e = self._type_error("chan_list", "list of channel characters")
            raise TypeError(e)
        clist = []
        for cc in chan_list:
            if isinstance(cc, str) and cc.upper() == "ALL":
                return list(self.__theset.keys())
            cc2 = nmu.channel_char_check(cc)
            if not cc2:
                channel = cc
                e = self._value_error("channel")
                raise ValueError(e)
            clist.append(cc2)
        return clist

    def _chan_check(self, dataseriesset):
        ks = self.__theset.keys()
        kd = dataseriesset._theset_keys()
        if ks != kd:
            q = (
                "unequal channels: "
                + str(ks)
                + " vs "
                + str(kd)
                + ". "
                + "Do you want to continue?"
            )
            yn = nmu.input_yesno(q)
            return yn.lower() == "y"
        return False

    def add(self, data_dict, quiet=nmp.QUIET):
        # data_dict = {'A': [DataA0, DataA1...]}
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        dd = self._data_dict_check(data_dict)
        modified = False
        for cc, dlist in dd.items():
            hlist = []
            for d in dlist:
                if cc in self.__theset.keys():
                    if d in self.__theset[cc]:
                        pass  # do nothing
                    else:
                        self.__theset[cc].append(d)
                        hlist.append(d.name)
                        modified = True
                else:
                    self.__theset.update({cc: [d]})
                    hlist.append(d.name)
                    modified = True
            if hlist:
                h = "added to ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        self._sort_update()
        if modified:
            self._modified()
        return modified

    def discard(self, data_dict, quiet=nmp.QUIET):
        # data_dict = {'A': [DataA0, DataA1...]}
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        dd = self._data_dict_check(data_dict, chan_default="ALL_EXISTING")
        modified = False
        for cc, dlist in dd.items():
            hlist = []
            if cc in self.__theset.keys():
                for d in dlist:
                    if d in self.__theset[cc]:
                        self.__theset[cc].remove(d)
                        hlist.append(d.name)
                        modified = True
            if hlist:
                h = "removed from ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        if modified:
            self._theset_clear_if_empty()
            self._modified()
        return modified

    def contains(self, data_dict, alert=False):
        # data_dict = {'A': [DataA0, DataA1...]}
        dd = self._data_dict_check(data_dict)
        failure = False
        for cc, dlist in dd.items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d not in self.__theset[cc]:
                        if alert:
                            a = "ch " + cc + ": failed to find " + d.name
                            self._alert(a)
                        failure = True
            else:
                if alert:
                    a = "failed to find channel " + cc
                    self._alert(a)
                failure = True
        return not failure

    def get_channel(self, chan_char):
        cc = nmu.channel_char_check(chan_char)
        if not cc:
            channel = chan_char
            e = self._value_error("channel")
            raise ValueError(e)
        if cc in self.__theset.keys():
            return self.__theset[cc].copy()  # copy, enforce private set
        return []

    def get(self, chan_list=["ALL"]):
        clist = self._chan_list_check(chan_list)
        s = {}
        for cc in clist:
            if cc in self.__theset.keys():
                s.update({cc: self.__theset[cc].copy()})
                # copy, enforce private set
        return s

    def clear(self, chan_list=["ALL"], confirm=True, quiet=nmp.QUIET):
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        clist = self._chan_list_check(chan_list)
        if not clist:
            return True  # nothing to do
        if confirm:
            if len(clist) > 1:
                ctxt = ", channels " + ", ".join(clist) + "?"
            else:
                ctxt = ", channel " + ", ".join(clist) + "?"
            q = "are you sure you want to clear " + self.name + ctxt
            yn = nmu.input_yesno(q)
            if not yn.lower() == "y":
                self._history("cancel", quiet=quiet)
                return False
        modified = False
        hlist = []
        for cc in clist:
            if cc in self.__theset.keys():
                self.__theset[cc].clear()
                hlist.append(cc)
                modified = True
        self._theset_clear_if_empty()
        if modified:
            self._modified()
        if hlist:
            h = "cleared ch " + ", ".join(hlist)
            self._history(h, quiet=quiet)
        return True

    def reverse(self, chan_list=["ALL"], quiet=nmp.QUIET):
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        clist = self._chan_list_check(chan_list)
        if not clist:
            return True  # nothing to do
        modified = False
        hlist = []
        for cc in clist:
            if cc in self.__theset.keys():
                self.__theset[cc].reverse()
                hlist.append(cc)
                modified = True
        self._theset_clear_if_empty()
        if modified:
            self._modified()
        if hlist:
            h = "reversed ch " + ", ".join(hlist)
            self._history(h, quiet=quiet)
        return True

    def difference(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        # symbol: -
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        if alert and self._chan_check(dataseriesset):
            self._history("cancel", quiet=quiet)
            return False
        modified = False
        for cc, dlist in dataseriesset._theset_items():
            if cc not in self.__theset.keys():
                continue
            hlist = []
            for d in dlist:
                if d in self.__theset[cc]:
                    self.__theset[cc].remove(d)
                    hlist.append(d.name)
                    modified = True
            if hlist:
                h = "removed from ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        self._theset_clear_if_empty()
        if modified:
            self._modified()
        return modified

    def intersection(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        # symbol: &
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        if alert and self._chan_check(dataseriesset):
            self._history("cancel", quiet=quiet)
            return False
        remove = {}
        for cc_s, dlist_s in self.__theset.items():
            for cc_o, dlist_o in dataseriesset._theset_items():
                if cc_s.upper() == cc_o.upper():
                    for d in dlist_s:
                        if d not in dlist_o:
                            if cc_s in remove.keys():
                                remove[cc_s].append(d)
                            else:
                                remove.update({cc_s: [d]})
        if len(remove) == 0:
            return False
        for cc, dlist in remove.items():
            hlist = []
            for d in dlist:
                self.__theset[cc].remove(d)
                hlist.append(d.name)
            if hlist:
                h = "removed from ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        self._theset_clear_if_empty()
        self._modified()
        return True

    def symmetric_difference(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        # symbol: ^
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        if alert and self._chan_check(dataseriesset):
            self._history("cancel", quiet=quiet)
            return False
        add = {}
        remove = {}
        for cc, dlist in dataseriesset._theset_items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d in self.__theset[cc]:
                        if cc in remove.keys():
                            remove[cc].append(d)
                        else:
                            remove.update({cc: [d]})
                    else:
                        if cc in add.keys():
                            add[cc].append(d)
                        else:
                            add.update({cc: [d]})
        if len(remove) == 0 and len(add) == 0:
            return False
        for cc, dlist in remove.items():
            hlist = []
            for d in dlist:
                self.__theset[cc].remove(d)
                hlist.append(d.name)
            if hlist:
                h = "removed from ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        for cc, dlist in add.items():
            hlist = []
            for d in dlist:
                self.__theset[cc].append(d)
                hlist.append(d.name)
            if hlist:
                h = "added to ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        self._sort_update()
        self._theset_clear_if_empty()
        self._modified()
        return True

    def union(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        # symbol: |
        if self.__eq_lock:
            raise RuntimeError(self._eq_lock_error)
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        if alert and self._chan_check(dataseriesset):
            self._history("cancel", quiet=quiet)
            return False
        add = {}
        for cc, dlist in dataseriesset._theset_items():
            if cc in self.__theset.keys() and isinstance(dlist, list):
                for d in dlist:
                    if d not in self.__theset[cc]:
                        if cc in add.keys():
                            add[cc].append(d)
                        else:
                            add.update({cc: [d]})
        if len(add) == 0:
            return False
        for cc, dlist in add.items():
            hlist = []
            for d in dlist:
                self.__theset[cc].append(d)
                hlist.append(d.name)
            if hlist:
                h = "added to ch " + cc + ": " + ", ".join(hlist)
                self._history(h, quiet=quiet)
        self._sort_update()
        self._theset_clear_if_empty()
        self._modified()
        return True

    def isdisjoint(self, dataseriesset):
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        for cc, dlist in dataseriesset._theset_items():
            if cc in self.__theset.keys() and isinstance(dlist, list):
                for d in dlist:
                    if d in self.__theset[cc]:
                        return False
        return True

    def isequal(self, dataseriesset, alert=True):
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        ks = self.__theset.keys()
        ko = dataseriesset._theset_keys()
        if ks != ko:
            if alert:
                a = "unequal channels: " + str(ks) + " vs " + str(ko)
                self._alert(a)
            return False
        for cc, dlist_o in dataseriesset._theset_items():
            dlist_s = self.__theset[cc]
            ns = len(dlist_s)
            no = len(dlist_o)
            if ns != no:
                a = (
                    "unequal data items in channel "
                    + cc
                    + ": "
                    + str(ns)
                    + " vs "
                    + str(no)
                )
                self._alert(a)
                return False
            for d in dlist_s:
                if d not in dlist_o:
                    a = "missing data item " + d.name + ": " + str(d)
                    self._alert(a)
                    return False
        return True

    def issubset(self, dataseriesset):
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        for cc_s, dlist_s in self.__theset.items():
            cc_found = False
            for cc_o, dlist_o in dataseriesset._theset_items():
                if cc_o.upper() == cc_s.upper():
                    cc_found = True
                    for d in dlist_s:
                        if d not in dlist_o:
                            return False
            if not cc_found:
                return False
        return True

    def issuperset(self, dataseriesset):
        if not isinstance(dataseriesset, NMSet):
            e = self._type_error("dataseriesset", "NMSet")
            raise TypeError(e)
        for cc, dlist in dataseriesset._theset_items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d not in self.__theset[cc]:
                        return False
            else:
                return False
        return True


class NMSetContainer(NMObjectContainer):
    """
    Container of NMSets
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMSetContainer",
        rename_on: bool = False,
        name_prefix: str = "set",
        # for generating names of NMSet
        name_seq_format: str = "0",
        copy: nmu.NMSetContainerType = None,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )

    # override, no super
    def copy(self) -> nmu.NMSetContainerType:
        return NMSetContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMSet.__name__

    # override
    def new(
        self,
        name: str = "default",
        # select: bool = True,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMSetType:
        name = self._newkey_check(name, ok="default")
        if name.lower() == "default":
            name = self.name_next()
        s = NMSet(parent=self, name=name)
        super().new(s)
        return s

    # override
    @property
    def _bad_names(self):  # names not allowed
        bn = []
        if isinstance(nmp.BADNAMES, list):
            bn = list(nmp.BADNAMES)
            bn.remove("all")
        return bn

    # @property  # override, no super
    # def select(self):
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    # override
    def rename(self, name, newname, quiet=nmp.QUIET):
        if name.lower() == "all":
            e = self._error("cannot rename 'All' set")
            raise RuntimeError(e)
        if name.lower() == "setx":
            e = self._error("cannot rename SetX")
            raise RuntimeError(e)
        return super().rename(name, newname, quiet=quiet)

    # override, note first=1
    def name_next(self, first=1):
        return super().name_next(first=first)

    # override, note first=1
    def name_next_seq(self, prefix="default", first=1):
        return super().name_next_seq(prefix=prefix, first=first)

    def add_epoch(self, name, epoch, quiet=nmp.QUIET):
        if not self._parent.thedata:
            e = self._error("no selected data for dataseries " + self._parent._tp)
            raise RuntimeError(e)
        if not isinstance(name, list):
            if name.lower() == "all":
                name = self.names
                name.remove("All")
                name.remove("SetX")
            else:
                name = [name]
        if not name:
            return False
        if not isinstance(epoch, list):
            epoch = [epoch]
        for n in name:
            if n.lower() == "all":
                a = "cannot edit 'All' set"
                self._alert(a, quiet=quiet)
                continue
            s = self.getitem(n, quiet=quiet)
            if not s:
                continue
            added = set()
            oor = set()
            for ep in epoch:
                if ep == -1:
                    ep = self._parent.epoch_select
                for chan in self._parent.thedata:
                    if ep >= 0 and ep < len(chan):
                        d = chan[ep]
                        if s.add(d):
                            added.add(ep)
                    else:
                        oor.add(ep)
            if len(added) > 0:
                added = list(added)
                added.sort()
                h = "added -> " + s._tp + ", ep=" + str(added)
                self._history(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = "out of range -> " + self._tp + ", ep=" + str(oor)
                self._error(h, quiet=quiet)
        return True

    def remove_epoch(self, name, epoch, quiet=nmp.QUIET):
        if len(self._parent.thedata) == 0:
            tp = self._parent.treepath()
            a = "no selected data for dataseries " + tp
            self._alert(a, quiet=quiet)
            return False
        if not isinstance(name, list):
            if name.lower() == "all":
                name = self.names
                name.remove("All")
                name.remove("SetX")
            else:
                name = [name]
        if len(name) == 0:
            return False
        if not isinstance(epoch, list):
            epoch = [epoch]
        for n in name:
            if n.lower() == "all":
                a = "cannot edit 'All' set"
                self._alert(a, quiet=quiet)
                continue
            s = self.getitem(n, quiet=quiet)
            if not s:
                continue
            removed = set()
            nis = set()
            oor = set()
            for ep in epoch:
                if ep == -1:
                    ep = self._parent.epoch_select
                for chan in self._parent.thedata:
                    if ep >= 0 and ep < len(chan):
                        d = chan[ep]
                        if s.contains(d):
                            if s.discard(d):
                                removed.add(ep)
                        else:
                            nis.add(ep)
                    else:
                        oor.add(ep)
            if len(removed) > 0:
                removed = list(removed)
                removed.sort()
                h = "removed -> " + s._tp + ", ep=" + str(removed)
                self._history(h, quiet=quiet)
            if len(nis) > 0:
                nis = list(nis)
                nis.sort()
                h = "not in set -> " + s._tp + ", ep=" + str(nis)
                self._error(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = "out of range -> " + s._tp + ", ep=" + str(oor)
                self._error(h, quiet=quiet)
        return True

    def equation(self, name, eq_list, lock=False, quiet=nmp.QUIET):
        """eq_list = ['Set1', '|', 'Set2']"""
        if self.exists(name):
            s = self.getitem(name, quiet=quiet)
        else:
            s = self.new(name, quiet=quiet)
        eq_list2 = []
        for i in eq_list:  # check equation is OK
            if i in SET_SYMBOLS:
                eq_list2.append(i)
            elif not self.exists(i):
                e = self._exists_error(i)
                raise ValueError(e)
            if lock and i.lower() == name.lower():
                e = self._error(
                    "locked set equation cannot contain itself: " + nmu.quotes(name)
                )
                raise ValueError(e)
            s2 = self.getitem(i, quiet=quiet)
            eq_list2.append(s2)
        return s._equation_execute(eq_list2, lock=lock)

    def clear(self, name, quiet=nmp.QUIET):
        if not isinstance(name, list):
            if name.lower() == "all":
                name = self.names
                name.remove("All")
                name.remove("SetX")
            else:
                name = [name]
        if len(name) == 0:
            return False
        if not quiet:
            n = ", ".join(name)
            q = "are you sure you want to clear " + n + "?"
            yn = nmu.input_yesno(q)
            if not yn.lower() == "y":
                self._history("cancel")
                return False
        for n in name:
            if n.lower() == "all":
                e = "cannot clear 'All' set"
                self._error(e, quiet=quiet)
                continue
            s = self.getitem(n, quiet=quiet)
            if s and s.clear():
                h = "cleared -> " + s._tp
                self._history(h, quiet=quiet)
        return True
