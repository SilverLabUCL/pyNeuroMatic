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


class DataSeriesSet(NMObject):  # rename as DictSet
    """
    NM DataSeriesSet class
    """

    def __init__(self, parent, name, **copy):
        super().__init__(parent, name)
        self.__theset = {}  # dictionary of sets, keys are chan-char
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

    # override
    def _iscopy(self, dataseriesset, alert=False):
        if not isinstance(dataseriesset, DataSeriesSet):
            if alert:
                a = nmu.type_error(dataseriesset, 'DataSeriesSet')
                self._alert(a, tp=self._tp)
            return False
        if not super()._iscopy(dataseriesset, alert=alert):
            return False
        ks = self.__theset.keys()
        ko = dataseriesset._keys()
        if ks != ko:
            if alert:
                a = 'unequal channels: ' + str(ks) + ' vs ' + str(ko)
                self._alert(a, tp=self._tp)
            return False
        for cc, dlist_o in dataseriesset._items():
            dlist_s = self.__theset[cc]
            ns = len(dlist_s)
            no = len(dlist_o)
            if ns != no:
                a = ('unequal data items in channel ' + cc + ': ' + str(ns) +
                     ' vs ' + str(no))
                self._alert(a, tp=self._tp)
                return False
            for i in range(0, ns):
                if not dlist_s[i]._iscopy(dlist_o[i], alert=alert):
                    return False
        # TODO eq_list
        # TODO eq_lock
        return True

    # override, no super
    def copy(self):
        thesetcopy = self.__theset.copy()  # TODO, refs need copying
        c = DataSeriesSet(self._parent, self.name, c_theset=thesetcopy,
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
    def count(self):
        return len(self.__theset)

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

    @property
    def theset(self):
        return self.__theset.copy()  # copy, enforce private set

    @theset.setter
    def theset(self, data):
        # data = {'A': [DataA0, DataA1...]}
        dd = self._data_dict_check(data)
        self.__theset.clear()
        return self.add(dd)

    def _theset_empty(self):
        for s in self.__theset.values():
            if len(s) > 0:
                return False
        self.__theset.clear()
        return True

    def _keys(self):
        return self.__theset.keys()

    def _items(self):
        return self.__theset.items()

    def _values(self):
        return self.__theset.values()

    def _data_dict_check(self, data, chan_default=nmp.CHANNEL_LIST[0]):
        # data = MyData -converted-> {'A': [MyData]}
        # data = [MyData0, MyData1] -converted-> {'A': [MyData0, MyData1]}
        # data = {'A': MyDataA0, 'B': MyDataB0} -converted->
        #        {'A': [MyDataA0], 'B': [MyDataB0]}
        # data = {'A': [MyDataA0, MyDataA1], 'B': [MyDataB0, MyDataB1]}
        # data format = {chan_char: [Data List]}
        if not isinstance(chan_default, str):
            raise TypeError(nmu.type_error(chan_default, 'channel character'))
        dnew = {}
        if data.__class__.__name__ == 'Data':  # no channel
            if chan_default.upper() == 'ALL':
                for cc in self.__theset.keys():
                    dnew.update({cc.upper(): [data]})
                return dnew
            if len(self.__theset.keys()) > 1:
                e = 'dictionary with channel keys'
                raise TypeError(nmu.type_error(data, e))
            cc = nmu.chan_char_check(chan_default)
            if len(cc) == 0:
                e = 'bad channel character: ' + nmu.quotes(chan_default)
                raise ValueError(e)
            dnew.update({cc: [data]})
            return dnew
        if isinstance(data, list):
            if len(data) == 0:
                return {}
            for d in data:
                if d.__class__.__name__ != 'Data':
                    raise TypeError(nmu.type_error(d, 'Data'))
            if chan_default.upper() == 'ALL':
                for cc in self.__theset.keys():
                    dnew.update({cc.upper(): data})
                return dnew
            if len(self.__theset.keys()) > 1:
                e = 'dictionary with channel keys'
                raise TypeError(nmu.type_error(data, e))
            cc = nmu.chan_char_check(chan_default)
            if len(cc) == 0:
                e = 'bad channel character: ' + nmu.quotes(chan_default)
                raise ValueError(e)
            dnew.update({cc: data})
            return dnew
        if not isinstance(data, dict):
            e = 'dictionary with channel keys'
            raise TypeError(nmu.type_error(data, e))
        for cc, dlist in data.items():
            cc2 = nmu.chan_char_check(cc)
            if len(cc2) == 0:
                e = 'bad channel character: ' + nmu.quotes(cc)
                raise ValueError(e)
            if dlist.__class__.__name__ == 'Data':
                dlist = [dlist]
            elif not isinstance(dlist, list):
                raise TypeError(nmu.type_error(dlist, 'Data list'))
            if len(dlist) == 0:
                continue
            for d in dlist:
                if d.__class__.__name__ != 'Data':
                    raise TypeError(nmu.type_error(d, 'Data'))
            dnew.update({cc2: dlist})
        return dnew

    def _chan_check(self, dataseriesset):
        ks = self.__theset.keys()
        kd = dataseriesset._keys()
        if ks != kd:
            q = ('unequal channels: ' + str(ks) + ' vs ' + str(kd) + '. ' +
                 'Do you want to continue?')
            yn = nmu.input_yesno(q, tp=self._tp)
            return yn == 'y'
        return False

    def get_channel(self, chan_char):
        cc = nmu.chan_char_check(chan_char)
        if len(cc) == 0:
            e = 'bad channel character: ' + nmu.quotes(chan_char)
            raise ValueError(e)
        if cc in self.__theset.keys():
            return self.__theset[cc].copy()  # copy, enforce private set
        return []

    def set_channel(self, chan_char, data_list):
        cc = nmu.chan_char_check(chan_char)
        if len(cc) == 0:
            e = 'bad channel character: ' + nmu.quotes(chan_char)
            raise ValueError(e)
        dd = self._data_dict_check({cc: data_list})
        if cc in self.__theset.keys():
            self.__theset[cc].clear()
        self.add(dd)
        return True

    def add(self, data, quiet=nmp.QUIET):
        # data = {'A': [DataA0, DataA1...]}
        dd = self._data_dict_check(data)
        modified = False
        for cc, dlist in dd.items():
            for d in dlist:
                if cc in self.__theset.keys():
                    if d in self.__theset[cc]:
                        pass  # do nothing
                    else:
                        self.__theset[cc].append(d)
                        modified = True
                else:
                    self.__theset.update({cc: [d]})
                    modified = True
        if modified:
            self._modified()
        return modified

    def discard(self, data, quiet=nmp.QUIET):
        # data = {'A': [DataA0, DataA1...]}
        dd = self._data_dict_check(data, chan_default='ALL')
        modified = False
        for cc, dlist in dd.items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d in self.__theset[cc]:
                        self.__theset[cc].remove(d)
                        modified = True
        if modified:
            self._theset_empty()
            self._modified()
        return modified

    def contains(self, data, alert=False):
        # data = {'A': [DataA0, DataA1...]}
        alert = nmu.bool_check(alert, False)
        dd = self._data_dict_check(data)
        for cc, dlist in dd.items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d not in self.__theset[cc]:
                        if alert:
                            a = ('failed to find ' + d.name + ' in channel ' +
                                 cc)
                            self._alert(a)
                        return False
            else:
                if alert:
                    self._alert('failed to find channel ' + cc)
                return False
        return True

    def clear(self, chan_list=['ALL'], confirm=True, quiet=nmp.QUIET):
        # chan_list = ['A', 'B']
        if isinstance(chan_list, str):
            chan_list = [chan_list]
        elif not isinstance(chan_list, list):
            raise TypeError(nmu.type_error(chan_list, 'list'))
        clist = []
        for cc in chan_list:
            if cc.upper() == 'ALL':
                clist = list(self.__theset.keys())
                break
            cc2 = nmu.chan_char_check(cc)
            if len(cc2) == 0:
                e = 'bad channel character: ' + nmu.quotes(cc)
                raise ValueError(e)
            clist.append(cc2)
        if len(clist) == 0:
            return False
        if nmu.bool_check(confirm, True):
            if len(clist) > 1:
                ctxt = ', channels ' + ', '.join(clist) + '?'
            else:
                ctxt = ', channel ' + ', '.join(clist) + '?'
            q = 'are you sure you want to clear ' + self.name + ctxt
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn == 'y':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        modified = False
        for cc in clist:
            if cc in self.__theset.keys():
                self.__theset[cc].clear()
                modified = True
        self._theset_empty()
        if modified:
            self._modified()
        return modified

    def difference(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        if nmu.bool_check(alert, True) and self._chan_check(dataseriesset):
            self._history('cancel', tp=self._tp, quiet=quiet)
            return False
        modified = False
        for cc, dlist in dataseriesset._items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d in self.__theset[cc]:
                        self.__theset[cc].remove(d)
                        modified = True
        self._theset_empty()
        if modified:
            self._modified()
        return modified

    def intersection(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        if nmu.bool_check(alert, True) and self._chan_check(dataseriesset):
            self._history('cancel', tp=self._tp, quiet=quiet)
            return False
        remove = {}
        for cc_s, dlist_s in self.__theset.items():
            for cc_o, dlist_o in dataseriesset._items():
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
            for d in dlist:
                self.__theset[cc].remove(d)
        self._theset_empty()
        self._modified()
        return True

    def symmetric_difference(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        if nmu.bool_check(alert, True) and self._chan_check(dataseriesset):
            self._history('cancel', tp=self._tp, quiet=quiet)
            return False
        add = {}
        remove = {}
        for cc, dlist in dataseriesset._items():
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
            for d in dlist:
                self.__theset[cc].remove(d)
        for cc, dlist in add.items():
            for d in dlist:
                self.__theset[cc].append(d)
        self._theset_empty()
        self._modified()
        return True

    def union(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        if nmu.bool_check(alert, True) and self._chan_check(dataseriesset):
            self._history('cancel', tp=self._tp, quiet=quiet)
            return False
        add = {}
        for cc, dlist in dataseriesset._items():
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
            for d in dlist:
                self.__theset[cc].append(d)
        self._theset_empty()
        self._modified()
        return True

    def isdisjoint(self, dataseriesset, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        for cc, dlist in dataseriesset._items():
            if cc in self.__theset.keys() and isinstance(dlist, list):
                for d in dlist:
                    if d in self.__theset[cc]:
                        return False
        return True

    def isequal(self, dataseriesset, alert=True, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        ks = self.__theset.keys()
        ko = dataseriesset._keys()
        if ks != ko:
            if alert:
                a = 'unequal channels: ' + str(ks) + ' vs ' + str(ko)
                self._alert(a, tp=self._tp)
            return False
        for cc, dlist_o in dataseriesset._items():
            dlist_s = self.__theset[cc]
            ns = len(dlist_s)
            no = len(dlist_o)
            if ns != no:
                a = ('unequal data items in channel ' + cc + ': ' + str(ns) +
                     ' vs ' + str(no))
                self._alert(a, tp=self._tp)
                return False
            for d in dlist_s:
                if d not in dlist_o:
                    a = 'missing data item ' + d.name + ': ' + str(d)
                    self._alert(a, tp=self._tp)
                    return False
        return True

    def issubset(self, dataseriesset, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        for cc_s, dlist_s in self.__theset.items():
            cc_found = False
            for cc_o, dlist_o in dataseriesset._items():
                if cc_o.upper() == cc_s.upper():
                    cc_found = True
                    for d in dlist_s:
                        if d not in dlist_o:
                            return False
            if not cc_found:
                return False
        return True

    def issuperset(self, dataseriesset, quiet=nmp.QUIET):
        if not isinstance(dataseriesset, DataSeriesSet):
            raise TypeError(nmu.type_error(dataseriesset, 'DataSeriesSet'))
        for cc, dlist in dataseriesset._items():
            if cc in self.__theset.keys():
                for d in dlist:
                    if d not in self.__theset[cc]:
                        return False
            else:
                return False
        return True


class DataSeriesSetContainer(NMObjectContainer):
    """
    Container for NM DataSeriesSet objects
    """

    def __init__(self, parent, name, **copy):
        t = DataSeriesSet(None, 'empty').__class__.__name__
        super().__init__(parent, name, type_=t,
                         prefix=nmp.DATASERIESSET_PREFIX, rename=True, **copy)

    # override
    @property
    def _bad_names(self):  # names not allowed
        bn = super()._bad_names
        bn.remove('all')
        return bn

    # override, no super
    def copy(self):
        return DataSeriesSetContainer(self._parent, self.name,
                                      c_prefix=self.prefix,
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
        o = DataSeriesSet(None, 'iwillberenamed')
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
