#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


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
        super().__init__(parent, name, nmc.ESET_PREFIX, seq_start=1)

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

    def rename(self, name, newname, quiet=False):  # override
        s = self.get(name, quiet=quiet)
        if not s:
            return False
        if s.name.upper() == 'ALL':
            nmu.error("cannot rename 'All' set", quiet=quiet)
            return False
        if s.name.upper() == 'SETX':
            nmu.error("cannot rename SetX", quiet=quiet)
            return False
        return super().rename(name, newname, quiet=quiet)

    def __add_remove(self, name, epoch_list, add=False, remove=False,
                     quiet=False):
        if not add and not remove:
            return {}
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        if len(self.parent.thedata) == 0:
            a = 'no selected data for dataprefix ' + self.parent.tree_path
            nmu.alert(a, quiet=quiet)
            return {}
        s = self.get(name, quiet=quiet)
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
                if e >= 0 and e < len(chan):
                    d = chan[e]
                    s.theset.add(d)
                    dnames.append(d.name)
                    found_something = True
                else:
                    nmu.alert('out of range epoch: ' + str(e))
            if found_something:
                i.append(e)
        r['added'] = dnames
        nmu.history(s.tree_path + nmc.S0 + 'ep=' + str(i), quiet=quiet)
        return r

    def add_epochs(self, name, epoch_list, quiet=False):
        return self.__add_remove(name, epoch_list, add=True, quiet=quiet)

    def remove_epochs(self, name, epoch_list, quiet=False):
        if type(epoch_list) is not list:
            epoch_list = [epoch_list]
        s = self.get(name, queit=quiet)
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
                if e >= 0 and e < len(chan):
                    d = chan[e]
                    s.theset.discard(d)
                    dnames.append(d.name)
                    found_something = True
                else:
                    nmu.alert('out of range epoch: ' + str(e))
            if found_something:
                i.append(e)
        r['removed'] = dnames
        nmu.history(s.tree_path + nmc.S0 + 'ep=' + str(i), quiet=quiet)
        return r

    def clear(self, name, quiet=False):
        s = self.get(name, quiet=quiet)
        if not s:
            return False
        s.theset.clear()
        return True
