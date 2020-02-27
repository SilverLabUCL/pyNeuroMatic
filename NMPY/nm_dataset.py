#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject
from nm_object import NMObjectContainer
import nm_preferences as nmp
import nm_utilities as nmu


class DataSet(NMObject):
    """
    NM DataSet class
    """

    def __init__(self, parent, name, **copy):
        super().__init__(parent, name)
        self.__theset = {}
        self.__eq_list = []
        self.__eq_lock = True
        for k, v in copy.items():
            if k.lower() == 'c_theset' and isinstance(v, dict):
                self.__theset = v
            if k.lower() == 'c_eq_list' and isinstance(v, list):
                self.__eq_list = v
            if k.lower() == 'c_eq_lock' and isinstance(v, bool):
                self.__eq_lock = v
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
        k.update({'eq_list': self.__eq_list})
        k.update({'eq_lock': self.__eq_lock})
        return k

    # override, no super
    def copy(self):
        thesetcopy = self.__theset.copy()  # TODO, refs need copying
        c = DataSet(self._parent, self.name, c_theset=thesetcopy,
                    c_eq_list=self.__eq_list.copy(),
                    c_eq_lock=self.__eq_lock)
        return c

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
        return self._eq_lock_set(eq_lock)

    def _eq_lock_set(self, eq_lock, quiet=nmp.QUIET):
        self.__eq_lock = eq_lock
        self._modified()
        return True

    @property
    def data_names(self):
        if self.name.upper() == 'ALL':
            return ['ALL']
        n = {}
        for cc, s in self.__theset.items():
            nlist = [d.name for d in s]
            nlist.sort()
            n.update({cc: nlist})
        return n

    def contains(self, data, chan_char=''):
        if data.__class__.__name__ != 'Data':
            raise TypeError(nmu.type_error(data, 'Data'))
        if not isinstance(chan_char, str):
            raise TypeError(nmu.type_error(chan_char, 'string'))
        if chan_char == '':
            pass  # search all channels
        elif nmu.channel_num(chan_char) < 0:
            e = 'bad channel character: ' + nmu.quotes(chan_char)
            raise ValueError(e)
        if chan_char == '':
            for cc, s in self.__theset.index():
                if data in s:
                    return True
            return False
        cc = chan_char.upper()
        if cc in self.__theset.keys():
            return data in self.__theset[cc]
        return False

    def _data_dict(self, data):
        # data = MyData -converted-> {'A': [MyData]}
        # data = [MyData0, MyData1] -converted-> {'A': [MyData0, MyData1]}
        # data = {'A': MyDataA0, 'B': MyDataB0} -converted->
        #        {'A': [MyDataA0], 'B': [MyDataB0]}
        # data = {'A': [MyDataA0, MyDataA1], 'B': [MyDataB0, MyDataB1]}
        # data format = {chan_char: [Data List]}
        if data.__class__.__name__ == 'Data':
            if len(self.__theset.keys()) > 1:
                e = 'dictionary with channel keys'
                TypeError(nmu.type_error(data, e))
            return {'A': [data]}  # assume channel A
        if isinstance(data, list):
            if len(data) == 0:
                return {}
            if len(self.__theset.keys()) > 1:
                e = 'dictionary with channel keys'
                TypeError(nmu.type_error(data, e))
            data = {'A': data}  # assume channel A
        elif not isinstance(data, dict):
            raise TypeError(nmu.type_error(data, 'dictionary'))
        for cc, dlist in data.items():
            if nmu.channel_num(cc) < 0:
                e = 'bad channel character: ' + nmu.quotes(cc)
                raise ValueError(e)
            if dlist.__class__.__name__ == 'Data':
                dlist = [dlist]
                data[cc] = dlist
            elif not isinstance(dlist, list):
                raise TypeError(nmu.type_error(dlist, 'Data list'))
            for d in dlist:
                if d.__class__.__name__ != 'Data':
                    raise TypeError(nmu.type_error(d, 'Data'))
        return data

    # add/update
    # clear
    # copy
    # difference
    # difference_update
    # discard/remove
    # intersection
    # intersection_update
    # isdisjoint
    # issubset
    # issuperset
    # pop
    # symmetric_difference
    # symmetric_difference_update
    # union
    # TODO 'All' set

    def add(self, data, quiet=nmp.QUIET):
        # more like 'update' but call 'add'
        dd = self._data_dict(data)
        if not isinstance(dd, dict):
            return False
        modified = False
        for cc, dlist in dd.items():
            cc = cc.upper()
            for d in dlist:
                if cc in self.__theset.keys():
                    if d not in self.__theset[cc]:
                        self.__theset[cc].add(d)
                        modified = True
                else:
                    self.__theset.update({cc: set([d])})
                    modified = True
        if modified:
            self._modified()
        return True

    def clear(self, chan_char='', quiet=nmp.QUIET):
        if not isinstance(chan_char, str):
            raise TypeError(nmu.type_error(chan_char, 'string'))
        if chan_char == '':
            pass  # clear all channels
        elif nmu.channel_num(chan_char) < 0:
            e = 'bad channel character: ' + nmu.quotes(chan_char)
            raise ValueError(e)
        if chan_char == '':
            for cc in self.__theset.keys():
                self.__theset[cc].clear()
                self._modified()
            return True
        cc = chan_char.upper()
        if cc in self.__theset.keys():
            self.__theset[cc].clear()
            self._modified()
        return True

    # difference
    # difference_update (one function, with 'update' argument?)

    def difference(self, data, quiet=nmp.QUIET):
        dd = self._data_dict(data)
        if not isinstance(dd, dict):
            return False
        for cc, dlist in dd.items():
            cc = cc.upper()
            if cc not in self.__theset.keys():
                continue
            for d in dlist:
                if d in self.__theset[cc]:
                    self.__theset[cc].discard(d)
                    modified = True
        if modified:
            self._modified()
        return True

    def discard(self, data, quiet=nmp.QUIET):  # not in set? # remove
        dd = self._data_dict(data)
        if not isinstance(dd, dict):
            return False
        modified = False
        for cc, dlist in dd.items():
            cc = cc.upper()
            if cc not in self.__theset.keys():
                continue
            for d in dlist:
                if d in self.__theset[cc]:
                    self.__theset[cc].discard(d)
                    modified = True
        if modified:
            self._modified()
        return True

    # intersection
    # intersection_update
    # isdisjoint
    # issubset
    # issuperset
    # symmetric_difference
    # symmetric_difference_update

    def union(self, eset, quiet=nmp.QUIET):
        if not isinstance(eset, list):
            eset = [eset]
        for s in eset:
            if isinstance(s, DataSet):
                self.__theset = self.__theset.union(s.__theset)


class DataSetContainer(NMObjectContainer):
    """
    Container for NM DataSet objects
    """

    def __init__(self, parent, name, **copy):
        t = DataSet(None, 'empty').__class__.__name__
        super().__init__(parent, name, type_=t, prefix=nmp.DATASET_PREFIX,
                         rename=True, **copy)

    # override
    @property
    def _bad_names(self):  # names not allowed
        bn = super()._bad_names
        bn.remove('all')
        return bn

    # override, no super
    def copy(self):
        return DataSetContainer(self._parent, self.name, c_prefix=self.prefix,
                                 c_rename=self.parameters['rename'],
                                 c_thecontainer=self._thecontainer_copy())

    # @property  # override, no super
    # def select(self):
    #     return self.__set_select

    # @select.setter
    # def select(self, set_eq):
    #     self.__set_select = set_eq

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = DataSet(None, 'iwillberenamed')
        return super().new(name=name, nmobject=o, select=select, quiet=quiet)

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
        if len(self._parent.thedata) == 0:
            tp = self._parent._tp
            e = 'no selected data for dataseries ' + tp
            self._error(e, quiet=quiet)
            return False
        if not isinstance(name, list):
            if name.lower() == 'all':
                name = self.names
                name.remove('All')
                name.remove('SetX')
            else:
                name = [name]
        if len(name) == 0:
            return False
        if not isinstance(epoch, list):
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
        if len(self._parent.thedata) == 0:
            tp = self._parent.treepath(history=True)
            e = 'no selected data for dataseries ' + tp
            self._alert(e, quiet=quiet)
            return False
        if not isinstance(name, list):
            if name.lower() == 'all':
                name = self.names
                name.remove('All')
                name.remove('SetX')
            else:
                name = [name]
        if len(name) == 0:
            return False
        if not isinstance(epoch, list):
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
        """eq_list=['Set1', '|', 'Set2']"""
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
        if not isinstance(name, list):
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
