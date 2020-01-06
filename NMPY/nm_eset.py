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

    def __init__(self, parent, name, fxns={}):
        super().__init__(parent, name, fxns=fxns)
        self.__theset = set()
        self.__eq_list = []
        self.__eq_lock = True

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

    def _copy(self, epochset, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(epochset, EpochSet):
            raise TypeError(nmu.type_error(epochset, 'epochset', 'EpochSet'))
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not super()._copy(epochset, copy_name=copy_name, quiet=True):
            return False
        # COPY theset
        # self.__theset.clear()  # RESET
        self.__eq_list = list(epochset._EpochSet__eq_list)
        self.__eq_lock = epochset._EpochSet__eq_lock
        h = ('copied EpochSet ' + nmu.quotes(epochset.name) + ' to ' +
             nmu.quotes(name))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

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
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = EpochSet(self._parent, 'tempname', self._fxns)
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
            tp = self._parent.tree_path(history=True)
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
            s = self.get(n, quiet=quiet)
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
                h = ('added' + nmp.S0 + s.tree_path(history=True) +
                     ', ep=' + str(added))
                self._history(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = ('out of range' + nmp.S0 + s.tree_path(history=True) +
                     ', ep=' + str(oor))
                self._error(h, quiet=quiet)
        return True

    def remove_epoch(self, name, epoch, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if len(self._parent.thedata) == 0:
            tp = self._parent.tree_path(history=True)
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
            s = self.get(n, quiet=quiet)
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
            tp = s.tree_path(history=True)
            if len(removed) > 0:
                removed = list(removed)
                removed.sort()
                h = 'removed' + nmp.S0 + tp + ', ep=' + str(removed)
                self._history(h, quiet=quiet)
            if len(nis) > 0:
                nis = list(nis)
                nis.sort()
                h = 'not in set' + nmp.S0 + tp + ', ep=' + str(nis)
                self._error(h, quiet=quiet)
            if len(oor) > 0:
                oor = list(oor)
                oor.sort()
                h = 'out of range' + nmp.S0 + tp + ', ep=' + str(oor)
                self._error(h, quiet=quiet)
        return True

    def equation(self, name, eq_list, lock=True, quiet=nmp.QUIET):
        """eq_list=[Set1', '|', 'Set2']"""
        quiet = nmu.check_bool(quiet, nmp.QUIET)
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
            s = self.get(n, quiet=quiet)
            if s and s.clear():
                tp = s.tree_path(history=True)
                self._history('cleared' + nmp.S0 + tp, quiet=quiet)
        return True
