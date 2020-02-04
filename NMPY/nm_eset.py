#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_preferences as nmp
import nm_utilities as nmu


class EpochSet(NMObject):
    """
    NM EpochSet class
    """

    def __init__(self, parent, name, fxns={}, theset=None, eq_list=[],
                 eq_lock=True):
        super().__init__(parent, name, fxns=fxns)
        if theset is None:
            self.__theset = set()
        elif isinstance(theset, set):
            self.__theset = theset
        else:
            raise TypeError(nmu.type_error(theset, 'set'))
        if not eq_list:
            self.__eq_list = []
        elif isinstance(eq_list, list):
            self.__eq_list = eq_list
        else:
            raise TypeError(nmu.type_error(eq_list, 'list'))
        self.__eq_lock = nmu.check_bool(eq_lock, True)
        self._param_list += ['eq_list', 'eq_lock']

    # override
    @property
    def _bad_names(self):
        bn = super()._bad_names
        bn.remove('all')
        return bn

    # override
    @property
    def parameters(self):
        k = super().parameters
        # k.update({'theset': self.__theset})
        k.update({'eq_list': self.__eq_list})
        k.update({'eq_lock': self.__eq_lock})
        return k

    # override, no super
    @property
    def content(self):
        return {'eset': self.name}

    # override, no super
    def copy(self):
        c = EpochSet(self._parent, self.name, fxns=self._fxns,
                     theset=self.__theset.copy(),
                     eq_list=self.__eq_list.copy(),
                     eq_lock=self.__eq_lock)
        return c

    @property
    def theset(self):
        return self.__theset

    @property
    def data_names(self):
        if self.name.lower() == 'all':
            return ['All']
        n = [d.name for d in self.__theset]
        return n.sort()

    @property
    def eq_list(self):
        return self.__eq_list

    @eq_list.setter
    def eq_list(self, eq_list):
        self._alert('see nm.eset.equation')

    @property
    def eq_lock(self):
        return self.__eq_lock

    @eq_lock.setter
    def eq_lock(self, eq_lock):
        self.__eq_lock = eq_lock

    def contains(self, data):
        return data in self.__theset

    def add(self, data):
        if data.__class__.__name__ == 'Data':
            self.__theset.add(data)
            return True
        return False

    def discard(self, data):
        if data.__class__.__name__ == 'Data':
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

    def __init__(self, parent, name, fxns={}):
        t = EpochSet(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t,
                         prefix=nmp.ESET_PREFIX)

    # override
    @property
    def _bad_names(self):  # names not allowed
        bn = super()._bad_names
        bn.remove('all')
        return bn

    # override, no super
    @property
    def content(self):
        return {'esets': self.names}

    # override
    def copy(self):
        c = EpochSetContainer(self._parent, self.name, fxns=self._fxns)
        super().copy(container=c)
        return c

    # @property  # override, no super
    # def select(self):
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = EpochSet(self._parent, 'temp', self._fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    # override
    def rename(self, name, newname, quiet=nmp.QUIET):
        if name.lower() == 'all':
            self._error("cannot rename 'All' set", quiet=quiet)
            return False
        if name.lower() == 'setx':
            self._error('cannot rename SetX', quiet=quiet)
            return False
        return super().rename(name, newname, quiet=quiet)

    # override, change default first to 1
    def name_next(self, first=1, quiet=nmp.QUIET):
        return super().name_next(first=first, quiet=quiet)

    # override, change default first to 1
    def name_next_seq(self, prefix='default', first=1, quiet=nmp.QUIET):
        return super().name_next_seq(prefix=prefix, first=first, quiet=quiet)

    def add_epoch(self, name, epoch, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if len(self._parent.thedata) == 0:
            tp = self._parent._tp
            e = 'no selected data for dataseries ' + tp
            self._error(e, quiet=quiet)
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
                self._alert("cannot edit 'All' set")
                continue
            s = self.getitem(n, quiet=quiet)
            if not s:
                continue
            added = set()
            oor = set()
            for e in epoch:
                if e == -1:
                    e = self._parent.epoch_select
                for chan in self._parent.thedata:
                    if e >= 0 and e < len(chan):
                        d = chan[e]
                        if s.add(d):
                            added.add(e)
                    else:
                        oor.add(e)
            if len(added) > 0:
                added = list(added)
                added.sort()
                h = ('added' + nmp.S0 + s._tp + ', ep=' + str(added))
                self._history(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = ('out of range' + nmp.S0 + self._tp + ', ep=' + str(oor))
                self._error(h, quiet=quiet)
        return True

    def remove_epoch(self, name, epoch, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if len(self._parent.thedata) == 0:
            tp = self._parent.treepath(history=True)
            e = 'no selected data for dataseries ' + tp
            self._alert(e, quiet=quiet)
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
                self._alert("cannot edit 'All' set")
                continue
            s = self.getitem(n, quiet=quiet)
            if not s:
                continue
            removed = set()
            nis = set()
            oor = set()
            for e in epoch:
                if e == -1:
                    e = self._parent.epoch_select
                for chan in self._parent.thedata:
                    if e >= 0 and e < len(chan):
                        d = chan[e]
                        if s.contains(d):
                            if s.discard(d):
                                removed.add(e)
                        else:
                            nis.add(e)
                    else:
                        oor.add(e)
            if len(removed) > 0:
                removed = list(removed)
                removed.sort()
                h = 'removed' + nmp.S0 + s._tp + ', ep=' + str(removed)
                self._history(h, quiet=quiet)
            if len(nis) > 0:
                nis = list(nis)
                nis.sort()
                h = 'not in set' + nmp.S0 + s._tp + ', ep=' + str(nis)
                self._error(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = 'out of range' + nmp.S0 + s._tp + ', ep=' + str(oor)
                self._error(h, quiet=quiet)
        return True

    def equation(self, name, eq_list, lock=True, quiet=nmp.QUIET):
        """eq_list=[Set1', '|', 'Set2']"""
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.exists(name):
            s = self.getitem(name, quiet=quiet)
        else:
            s = self.new(name, quiet=quiet)
        for i in eq_list:  # check equation is OK
            if i == '|' or i == '&':
                continue
            elif self.exists(i):
                continue
            else:
                e = 'unrecognized set equation item: ' + i
                self._error(e, quiet=quiet)
                return False
        s.eq_list = eq_list

    def clear(self, name, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
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
                self._history('cancel')
                return False
        for n in name:
            if n.lower() == 'all':
                self._error("cannot clear 'All' set", quiet=quiet)
                continue
            s = self.getitem(n, quiet=quiet)
            if s and s.clear():
                self._history('cleared' + nmp.S0 + s._tp, quiet=quiet)
        return True
