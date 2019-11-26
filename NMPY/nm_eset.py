#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_data import Data
import nm_utilities as nmu


class EpochSet(NMObject):
    """
    NM EpochSet class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__theset = set()
        self.__eq_list = []
        self.__eq_lock = True

    @property
    def content(self):  # override, no super
        return {'eset': self.name}

    @property
    def theset(self):
        return self.__theset

    @property
    def data_names(self):
        if self.name.lower() == 'all':
            return ['All']
        n = []
        for d in self.__theset:
            n.append(d.name)
        n.sort()
        return n

    @property
    def eq_list(self):
        return self.__eq_list

    @eq_list.setter
    def eq_list(self, eq_list):
        nmu.alert('see nm.eset.equation')

    @property
    def eq_lock(self):
        return self.__eq_lock

    @eq_lock.setter
    def eq_lock(self, eq_lock):
        self.__eq_lock = eq_lock

    def contains(self, data):
        return data in self.__theset

    def add(self, data):
        if isinstance(data, Data):
            self.__theset.add(data)
            return True
        return False

    def discard(self, data):
        if isinstance(data, Data):
            self.__theset.discard(data)
            return True
        return False

    def clear(self):
        if self.name.lower() == 'all':
            return False
        self.__theset.clear()
        return True

    def union(self, eset):
        if type(eset) is not list:
            eset = [eset]
        for s in eset:
            if isinstance(s, EpochSet):
                self.__theset = self.__theset.union(s.__theset)


class EpochSetContainer(Container):
    """
    Container for NM EpochSet objects
    """

    def __init__(self, parent, name='NMEpochSetContainer'):
        super().__init__(parent, name, nmc.ESET_PREFIX, seq_start=1)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        k = {'eset': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'eset_select': s})
        return k

    def new(self, name='default', select=True, quiet=False):  # override
        o = EpochSet(self.__parent, 'temp')
        return super().new(name=name, select=select, quiet=quiet, nmobj=o)

    # @property
    # def select(self):  # override, no super
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    def rename(self, name, newname, quiet=False):  # override, call super
        if name.lower() == 'all':
            nmu.error("cannot rename 'All' set", quiet=quiet)
            return False
        if name.lower() == 'setx':
            nmu.error('cannot rename SetX', quiet=quiet)
            return False
        return super().rename(name, newname, quiet=quiet)

    def add_epoch(self, name, epoch, quiet=False):
        if len(self.__parent.thedata) == 0:
            tp = self.__parent.tree_path(history=True)
            e = 'no selected data for dataseries ' + tp
            nmu.error(e, quiet=quiet)
            return False
        if type(name) is not list:
            if name.lower() == 'all':
                name = self.names
                name.remove('All')
                name.remove('SetX')
            else:
                name = [name]
        if len(name) == 0:
            return False
        if type(epoch) is not list:
            epoch = [epoch]
        for n in name:
            if n.lower() == 'all':
                nmu.alert("cannot edit 'All' set")
                continue
            s = self.get(n, quiet=quiet)
            if not s:
                continue
            added = set()
            oor = set()
            for e in epoch:
                if e == -1:
                    e = self.__parent.epoch_select
                for chan in self.__parent.thedata:
                    if e >= 0 and e < len(chan):
                        d = chan[e]
                        if s.add(d):
                            added.add(e)
                    else:
                        oor.add(e)
            if len(added) > 0:
                added = list(added)
                added.sort()
                h = ('added' + nmc.S0 + s.tree_path(history=True) +
                     ', ep=' + str(added))
                nmu.history(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = ('out of range' + nmc.S0 + s.tree_path(history=True) +
                     ', ep=' + str(oor))
                nmu.error(h, quiet=quiet)
        return True

    def remove_epoch(self, name, epoch, quiet=False):
        if len(self.__parent.thedata) == 0:
            tp = self.__parent.tree_path(history=True)
            e = 'no selected data for dataseries ' + tp
            nmu.alert(e, quiet=quiet)
            return False
        if type(name) is not list:
            if name.lower() == 'all':
                name = self.names
                name.remove('All')
                name.remove('SetX')
            else:
                name = [name]
        if len(name) == 0:
            return False
        if type(epoch) is not list:
            epoch = [epoch]
        for n in name:
            if n.lower() == 'all':
                nmu.alert("cannot edit 'All' set")
                continue
            s = self.get(n, quiet=quiet)
            if not s:
                continue
            removed = set()
            nis = set()
            oor = set()
            for e in epoch:
                if e == -1:
                    e = self.__parent.epoch_select
                for chan in self.__parent.thedata:
                    if e >= 0 and e < len(chan):
                        d = chan[e]
                        if s.contains(d):
                            if s.discard(d):
                                removed.add(e)
                        else:
                            nis.add(e)
                    else:
                        oor.add(e)
            tp = s.tree_path(history=True)
            if len(removed) > 0:
                removed = list(removed)
                removed.sort()
                h = 'removed' + nmc.S0 + tp + ', ep=' + str(removed)
                nmu.history(h, quiet=quiet)
            if len(nis) > 0:
                nis = list(nis)
                nis.sort()
                h = 'not in set' + nmc.S0 + tp + ', ep=' + str(nis)
                nmu.error(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = 'out of range' + nmc.S0 + tp + ', ep=' + str(oor)
                nmu.error(h, quiet=quiet)
        return True

    def equation(self, name, eq_list, lock=True, quiet=False):
        """eq_list=[Set1', '|', 'Set2']"""
        if self.exists(name):
            s = self.get(name, quiet=quiet)
        else:
            s = self.new(name, quiet=quiet)
        for i in eq_list:  # check equation is OK
            if i == '|' or i == '&':
                continue
            elif self.exists(i):
                continue
            else:
                e = 'unrecognized set equation item: ' + i
                nmu.error(e, quiet=quiet)
                return False
        s.eq_list = eq_list

    def clear(self, name, quiet=False):
        if type(name) is not list:
            if name.lower() == 'all':
                name = self.names
                name.remove('All')
                name.remove('SetX')
            else:
                name = [name]
        if len(name) == 0:
            return False
        if not quiet:
            n = ', '.join(name)
            q = 'are you sure you want to clear ' + n + '?'
            yn = nmu.input_yesno(q)
            if not yn == 'y':
                nmu.history('abort')
                return False
        for n in name:
            if n.lower() == 'all':
                nmu.error("cannot clear 'All' set", quiet=quiet)
                continue
            s = self.get(n, quiet=quiet)
            if s and s.clear():
                tp = s.tree_path(history=True)
                nmu.history('cleared' + nmc.S0 + tp, quiet=quiet)
        return True
