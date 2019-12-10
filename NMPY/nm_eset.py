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

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__theset = set()
        self.__eq_list = []
        self.__eq_lock = True

    @property  # override, no super
    def content(self):
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
        self.__alert('see nm.eset.equation')

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

    def __init__(self, manager, parent, name, fxns):
        o = EpochSet(manager, parent, 'temp', fxns)
        super().__init__(manager, parent, name, fxns, nmobj=o,
                         prefix=nmc.ESET_PREFIX)
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        k = {'esets': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'eset_select': s})
        return k

    # @property  # override, no super
    # def select(self):
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    # override
    def new(self, name='default', select=True, quiet=nmc.QUIET):
        o = EpochSet(self.__manager, self.__parent, name, self.__fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def rename(self, name, newname, quiet=nmc.QUIET):
        if name.lower() == 'all':
            self.__error("cannot rename 'All' set", quiet=quiet)
            return False
        if name.lower() == 'setx':
            self.__error('cannot rename SetX', quiet=quiet)
            return False
        return super().rename(name, newname, quiet=quiet)

    # override, change default first to 1
    def name_next(self, first=1, quiet=nmc.QUIET):
        return super().name_next(first=first, quiet=quiet)

    # override, change default first to 1
    def name_next_seq(self, prefix='default', first=1, quiet=nmc.QUIET):
        return super().name_next_seq(prefix=prefix, first=first, quiet=quiet)

    def add_epoch(self, name, epoch, quiet=nmc.QUIET):
        if len(self.__parent.thedata) == 0:
            tp = self.__parent.tree_path(history=True)
            e = 'no selected data for dataseries ' + tp
            self.__error(e, quiet=quiet)
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
                self.__alert("cannot edit 'All' set")
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
                self.__history(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = ('out of range' + nmc.S0 + s.tree_path(history=True) +
                     ', ep=' + str(oor))
                self.__error(h, quiet=quiet)
        return True

    def remove_epoch(self, name, epoch, quiet=nmc.QUIET):
        if len(self.__parent.thedata) == 0:
            tp = self.__parent.tree_path(history=True)
            e = 'no selected data for dataseries ' + tp
            self.__alert(e, quiet=quiet)
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
                self.__alert("cannot edit 'All' set")
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
                self.__history(h, quiet=quiet)
            if len(nis) > 0:
                nis = list(nis)
                nis.sort()
                h = 'not in set' + nmc.S0 + tp + ', ep=' + str(nis)
                self.__error(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = 'out of range' + nmc.S0 + tp + ', ep=' + str(oor)
                self.__error(h, quiet=quiet)
        return True

    def equation(self, name, eq_list, lock=True, quiet=nmc.QUIET):
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
                self.__error(e, quiet=quiet)
                return False
        s.eq_list = eq_list

    def clear(self, name, quiet=nmc.QUIET):
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
                self.__history('abort')
                return False
        for n in name:
            if n.lower() == 'all':
                self.__error("cannot clear 'All' set", quiet=quiet)
                continue
            s = self.get(n, quiet=quiet)
            if s and s.clear():
                tp = s.tree_path(history=True)
                self.__history('cleared' + nmc.S0 + tp, quiet=quiet)
        return True
