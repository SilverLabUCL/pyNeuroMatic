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
        if s.name.upper() == 'ALL':
            error("cannot rename 'All' set")
            return False
        if s.name.upper() == 'SETX':
            error("cannot rename SetX")
            return False
        return super().rename(name, newname)

    def add(self, name, epoch_list, quiet=False):
        s = self.get(name)
        if not s:
            return {}
        r = {'Set': name}
        dnames = []
        i = []
        for e in epoch_list:
            if e == -1:
                e = self.parent.epoch_select
            found_something = False
            for chan in self.parent.thedata:
                if e < 0 or e >= len(chan):
                    continue
                d = chan[e]
                s.theset.add(d)
                dnames.append(d.name)
                found_something = True
            if found_something:
                i.append(str(e))
        r['added'] = dnames
        if not quiet:
            history(s.tree_path + ' -> ep=' + ', '.join(i))
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

    def get_selected(self, names=False):
        channels = len(self.parent.thedata)
        cs = self.parent.channel_select
        all_chan = self.parent.all_channels
        ss = self.parent.eset_select.name
        eset = self.parent.eset_select.theset
        setx = self.get('SetX').theset
        if channels == 0 or len(cs) == 0:
            return []
        if ss.upper() == 'ALL':
            all_epochs = True
        else:
            all_epochs = False
            eset = eset.difference(setx)
        if all_chan and channels > 1:
            clist = []
            for chan in self.parent.thedata:
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
            chan = self.parent.thedata[c]
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
