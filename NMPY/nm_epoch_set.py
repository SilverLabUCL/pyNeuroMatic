#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import NMObject
from nm_container import Container
from nm_utilities import channel_num
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error
from nm_utilities import history


class EpochSet(NMObject):
    """
    NM EpochSet class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__theset = set()

    @property
    def theset(self):
        return self.__theset


class EpochSetContainer(Container):
    """
    Container for NM EpochSet objects
    """

    def __init__(self, parent, name):
        super().__init__(parent, name, prefix=nmconfig.ESET_PREFIX)
        self.count_from = 1
        # self.__set_select = "All"

    def object_new(self, name):  # override, do not call super
        return EpochSet(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, EpochSet)

    # @property
    # def select(self):  # override, do not call super
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    def rename(self, name, newname):  # override
        s = self.get(name)
        if not s:
            return False
        if s.name.casefold() == "all":
            error("cannot rename 'All' set")
            return False
        if s.name.casefold() == "setx":
            error("cannot rename SetX")
            return False
        return super().rename(name, newname)

    def add(self, name, epoch, quiet=True):
        s = self.get(name)
        if not s:
            return {}
        r = {'Set': name}
        if epoch == -1:
            epoch = self.parent.epoch_select
        dnames = []
        for chan in self.parent.thedata:
            if epoch < 0 or epoch >= len(chan):
                return []
            w = chan[epoch]
            s.theset.add(w)
            dnames.append(w.name)
        r['added'] = dnames
        if not quiet:
            print(r)
        return r

    def remove(self, name, epoch, quiet=True):
        s = self.get(name)
        if not s:
            return {}
        r = {'Set': name}
        if epoch == -1:
            epoch = self.parent.epoch_select
        dnames = []
        for chan in self.parent.thedata:
            if epoch < 0 or epoch >= len(chan):
                return []
            d = chan[epoch]
            s.theset.discard(d)
            dnames.append(d.name)
        r['removed'] = dnames
        if not quiet:
            print(r)
        return r

    def clear(self, name):
        s = self.get(name)
        if not s:
            return False
        s.theset.clear()
        return True

    def get_selected(self):
        channels = len(self.parent.thedata)
        cs = self.parent.channel_select
        ss = self.parent.eset_select.name
        eset = self.parent.eset_select.theset
        setx = self.get("SetX").theset
        if ss.casefold() == "all":
            alldata = True
        else:
            alldata = False
            eset = eset.difference(setx)
        if cs.casefold() == "all" and channels > 1:
            clist = []
            for chan in self.parent.thedata:
                dlist = []
                for d in chan:
                    if alldata:
                        if d not in setx:
                            dlist.append(d)
                    elif d in eset:
                        dlist.append(d)
                clist.append(dlist)
            return clist
        dlist = []
        if channels == 1:
            cnum = 0
        else:
            cnum = channel_num(cs)
        if cnum >= 0 and cnum < channels:
            chan = self.parent.thedata[cnum]
            for d in chan:
                if alldata:
                    if d not in setx:
                        dlist.append(d)
                elif d in eset:
                    dlist.append(d)
        return dlist
