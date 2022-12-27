#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:25:27 2022

@author: jason
"""
from collections.abc import MutableMapping
from typing import Dict, List, Optional, Tuple, Union

from nm_object import NMObject
import nm_utilities as nmu


class NMObjectMapping(NMObject, MutableMapping):

    # MutableMapping:
    # __contains__, keys, items, values, get, __eq__, __ne__
    # pop, popitem, clear, update, and setdefault
    # dict is ordered for Python 3.7

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMObjectMap',
        nmobjects: Union[
            nmu.NMObjectType,
            List[nmu.NMObjectType],
            nmu.NMObjectMappingType] = [],
        allow_renaming: bool = True,  # rename NMObjects
        default_prefix: str = 'NMObject',  # auto name generation of NMObjects
        copy: nmu.NMObjectMappingType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        select = []  # key names of selected NMObjects

        if isinstance(copy, NMObjectMapping):
            allow_renaming = copy._NMObjectMapping__allow_renaming
            default_prefix = copy._NMObjectMapping__default_prefix
            select = copy._NMObjectMapping__select
            nmobjects = copy.map_copy()

        if isinstance(allow_renaming, bool):
            self.__allow_renaming = allow_renaming
        else:
            e = self._type_error('allow_renaming', 'boolean', tp='')
            raise TypeError(e)

        if default_prefix is None:
            default_prefix = ''
        elif not isinstance(default_prefix, str):
            e = self._type_error('default_prefix', 'string', tp='')
            raise TypeError(e)
        elif not self.name_ok(default_prefix, ok=['', 'default']):
            e = self._value_error('default_prefix', tp='')
            raise ValueError(e)
        if default_prefix.lower() == 'default':
            self.__default_prefix = 'NMObject'
        else:
            self.__default_prefix = default_prefix

        self.__map = {}
        self.update(nmobjects)  # add NMObjects to map

        self.__select = []
        if isinstance(select, list):
            for s in select:
                key = self._getkey(s)
                if key:
                    self.__select.append(key)

    # override, children should override
    def copy(self) -> nmu.NMObjectMappingType:
        return NMObjectMapping(copy=self)

    def map_copy(self) -> Dict:
        new_map = {}
        for k, v in self.__map.items():
            c = v.copy()
            new_map[c.name] = c
        return new_map

    # children should override
    @property
    def nmobject_class(self) -> nmu.NMObjectType:
        return NMObject  # can use with isinstance()

    # MutableMapping required
    def __setitem__(  # NOT ALLOWED
        self,
        key,  # NOT USED
        value
    ) -> None:
        # called by setdefault()
        # e.g. mymap['recorda0'] = mynmobject
        # return self.update(value)
        e = self._error('use method ' + nmu.quotes('update') + ' and/or ' +
                        nmu.quotes('delete') + ' to change mappings.')
        raise RuntimeError(e)

    # MutableMapping required
    def __getitem__(
        self,
        key
    ) -> nmu.NMObjectType:
        # called by in, get(), pop(), popitem(), items(), clear()
        # print('__getitem__ ' + key)
        key = self._getkey(key)
        if key:
            return self.__map[key]
        return None

    # MutableMapping required
    def __delitem__(self, key):
        # called by 'del', pop(), popitem(), clear()
        # print('__delitem__ ' + key)
        # del self.__map[key]
        e = self._error('use method ' + nmu.quotes('pop') +
                        ' to delete mappings.')
        raise RuntimeError(e)

    # MutableMapping required
    def __iter__(self):
        return iter(self.__map)

    # MutableMapping required
    def __len__(self):
        return len(self.__map)

    def __contains__(
        self,
        key: str
    ) -> bool:
        if self._getkey(key):
            return True
        return False

    def _getkey(
        self,
        key: str
    ) -> str:
        if not isinstance(key, str):
            e = self._type_error('key', 'string')
            raise TypeError(e)
        for k in self.__map.keys():
            if k.lower() == key.lower():  # keys are case insensitive
                return k
        return None

    # def __repr__(self):
    #    return f"{type(self).__name__}({self.mapping})"

    # override
    def __eq__(
        self,
        other: nmu.NMObjectMappingType,
    ) -> bool:
        if not super().__eq__(other):
            return False
        # if not isinstance(other.nmobject_class, type(self.nmobject_class)):
        #    return False
        if self.__default_prefix != other._NMObjectMapping__default_prefix:
            return False
        if self.__allow_renaming != other._NMObjectMapping__allow_renaming:
            return False
        if self.__select != other._NMObjectMapping__select:
            return False
        if len(self) != len(other):
            return False
        for k in other.keys():
            s = self.__getitem__(k)
            if s is None:
                return False
            o.get(k)
            if not s.__eq__(o):  # compare NMObjects
                return False
        return True

    # MutableMapping override
    def update(
        self,
        nmobjects: Union[
            nmu.NMObjectType,
            List[nmu.NMObjectType],
            nmu.NMObjectMappingType] = [],
    ) -> None:
        if isinstance(nmobjects, self.nmobject_class):
            olist = [nmobjects]
        elif isinstance(nmobjects, list):
            olist = nmobjects
        elif isinstance(nmobjects, NMObjectMapping):
            olist = list(nmobjects._NMObjectMapping__map.values())
        else:
            # dict not accepted to avoid confusion with keys
            # since keys are set equal to the NMObject names
            cn = self.nmobject_class.__name__
            e = (cn + ', List[' + cn + '] or NMObjectMapping')
            e = self._type_error('nmobjects', e)
            raise TypeError(e)
        for o in olist:  # test if all is OK
            if not isinstance(o, self.nmobject_class):
                e = self._type_error('nmobjects', self.nmobject_class.__name__)
                raise TypeError(e)
            if self._getkey(o.name):
                raise KeyError('key name ' + nmu.quotes(o.name) +
                               ' already exists.')
        for o in olist:
            self.__map[o.name] = o
        self.__update_nmobject_references()
        self._modified()

    def __update_nmobject_references(self):
        for o in self.__map.values():
            o._rename_fxnref_set(self.rename)  # reference of 'rename' fxn
            o._NMObject__parent = self._parent

    # MutableMapping override
    def clear(
        self,
        confirm: bool = True
    ) -> None:
        if len(self) == 0:
            return ()
        if confirm:
            q = ('are you sure you want to delete the following?' +
                 '\n' + ', '.join(self.__map.keys()))
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                return None
        self.__map.clear()
        self._modified()

    # MutableMapping override
    def popitem(
        self,
        confirm: bool = True
    ) -> Tuple[str, nmu.NMObjectType]:
        if len(self) == 0:
            return ()
        key = list(self.__map.keys())[-1]  # last item
        if confirm:
            q = ('are you sure you want to delete ' + nmu.quotes(key) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                return ()
        rtuple = self.__map.popitem()
        if rtuple:
            self._modified()
        return rtuple

    # MutableMapping override
    def pop(
        self,
        key,
        confirm: bool = True
    ) -> Tuple[str, nmu.NMObjectType]:
        key2 = self._getkey(key)
        if not key2:
            raise KeyError(nmu.quotes(key) + ' does not exist.')
        if confirm:
            q = ('are you sure you want to delete ' + nmu.quotes(key2) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                return ()
        rtuple = self.__map.pop(key2)
        if rtuple:
            self._modified()
        return rtuple

    def rename(
        self,
        key: str,
        newkey: str
    ) -> None:
        if not self.__allow_renaming:
            e = self._error('key names are locked in this mapping.')
            raise RuntimeError(e)
        key2 = self._getkey(key)
        if not key2:
            raise KeyError(nmu.quotes(key) + ' does not exist.')
        if not isinstance(newkey, str):
            e = self._type_error('newkey', 'string')
            raise TypeError(e)
        if not newkey or not self.name_ok(newkey, ok='default'):
            e = self._value_error('newkey')
            raise ValueError(e)
        if newkey.lower() == 'default':
            newkey = self.next_default_name()
        if self._getkey(newkey):
            e = self._error('key name ' + nmu.quotes(newkey) +
                            ' is already in use.')
            raise KeyError(e)
        new_map = {}
        for k in self.__map.keys():
            o = self.__map[k]
            if k == key2:
                # o.name = newkey  # double history
                o._NMObject__name = newkey  # mangled, avoid double history
            new_map[o.name] = o
        self.__map = new_map
        # TODO: check reference change is not a problem
        self._modified()

    def reorder(
        self,
        newkeys: List[str]
    ) -> None:
        if not isinstance(newkeys, list):
            e = self._type_error('newkeys', 'List[string]')
            raise TypeError(e)
        n_new = len(newkeys)
        n_old = len(self)
        if n_new != n_old:
            e = ('number of keys mismatch: ' + str(n_new) + ' != ' +
                 str(n_old))
            raise KeyError(e)
        new_map = {}
        for k in newkeys:
            o = self.__getitem__(k)
            if not o:
                raise KeyError(nmu.quotes(k) + ' does not exist.')
            new_map[o.name] = o
        self.__map = new_map
        # TODO: check reference change is not a problem
        self._modified()

    def duplicate(
        self,
        key: str,
        newkey: str,
    ) -> nmu.NMObjectType:
        if not isinstance(newkey, str):
            e = self._type_error('newkey', 'string')
            raise TypeError(e)
        if not newkey or not self.name_ok(newkey, ok='default'):
            e = self._value_error('newkey')
            raise ValueError(e)
        if newkey.lower() == 'default':
            newkey = self.next_default_name()
        if self._getkey(newkey):
            e = self._error('key name ' + nmu.quotes(newkey) +
                            ' is already in use.')
            raise KeyError(e)
        o = self.__getitem__(key)
        if o is None:
            raise KeyError(nmu.quotes(key) + ' does not exist.')
        c = o.copy()
        # c.name = newkey  # double history
        c._NMObject__name = newkey  # mangled, avoid double history
        self.__map[c.name] = c
        self.__update_nmobject_references()
        self._modified()
        return c

    @property
    def default_prefix(self) -> str:  # see next_default_name())
        return self.__default_prefix

    @default_prefix.setter
    def default_prefix(self, newprefix: str) -> None:
        self._default_prefix_set(newprefix)

    def _prefix_set(
        self,
        newprefix: Optional[str] = None,
        # quiet: bool = nmp.QUIET
    ) -> None:
        if newprefix is None:
            pass  # ok
        elif not isinstance(newprefix, str):
            e = self._type_error('newprefix', 'string')
            raise TypeError(e)
        elif not self.name_ok(newprefix, ok=''):
            e = self._value_error('newprefix')
            raise ValueError(e)
        elif newprefix == '':
            newprefix = None
        if newprefix.lower() == self.__default_prefix.lower():
            return None  # nothing to do
        oldprefix = self.__default_prefix
        self.__default_prefix = newprefix
        self._modified()
        '''
        if not self.__default_prefix:
            h = 'prefix = None'
        else:
            h = 'prefix = ' + nmu.quotes(self.__prefix)
        self.note = h
        self._modified()
        h = nmu.history_change('prefix', oldprefix, self.__prefix)
        self._history(h, quiet=quiet)
        '''

    def next_default_name(self) -> str:
        inext = nmu.name_next_seq(names=self.__map.keys(),
                                  prefix=self.__default_prefix)
        return self.__default_prefix + str(inext)

    @property
    def select(self) -> List[nmu.NMObjectType]:
        return self.__select

    @select.setter
    def select(
        self,
        keys: Union[str, List[str]]
    ) -> None:
        self._select_set(keys)

    def select_set(
        self,
        keys: Union[str, List[str]] = None,
        # failure_alert: bool = True,
        # notes: bool = False,    # too many notes for select?
        # quiet: bool = nmp.QUIET
    ) -> None:
        if keys is None:
            keys = []
        elif isinstance(keys, str):
            if not keys or keys.lower() == 'none':
                keys = []
            else:
                keys = [keys]
        elif not isinstance(keys, list):
            e = self._type_error('keys', 'string or List[string]')
            raise TypeError(e)
        if len(keys) == 0:
            self.__select.clear()
            self._modified()
            return None
        found = []
        for k in keys:
            k2 = self._getkey(k)
            if not k2:
                raise KeyError(nmu.quotes(k) + ' does not exist.')
            found.append(k2)
        if len(found) > 0:
            self.__select.clear()
            self.__select.append(found)
            self._modified()

    def select_append(
        self,
        keys: Union[str, List[str]],
        # failure_alert: bool = True,
        # notes: bool = False,    # too many notes for select?
        # quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, list):
            e = self._type_error('keys', 'string or List[string]')
            raise TypeError(e)
        found = []
        for k in keys:
            k2 = self._getkey(k)
            if not k2:
                raise KeyError(nmu.quotes(k) + ' does not exist.')
            found.append(k2)
        self.__select.append(found)
        self._modified()

    def select_clear(
        self,
    ) -> None:
        self.__select.clear()
        self._modified()


if __name__ == '__main__':
    o0 = NMObject(parent=None, name='F0')
    o1 = NMObject(parent=None, name='F1')
    o2 = NMObject(parent=None, name='F2')
    o3 = NMObject(parent=None, name='F3')
    o4 = NMObject(parent=None, name='F4')
    # test = NMObjectDictionary([o0, o1, o2])
    test = NMObjectMapping(nmobjects=[o0, o1, o2])
    print('f1' in test.keys())
    # test['a'] = o3  # key is not used
    test.update(o4)
    test.update(o3)
    o = test.get('F2')
    print(o.name)
    print(test.setdefault('F2'))
    print(test.popitem())
    # del test['F2'] NOT ALLOWED
    o = test.pop('F2')
    print(str(o))
    # test['F0'] = o0  NOT ALLOWED
    test.rename(key='F4', newkey='F5')
    for k, v in test.items():
        print(k, ' -> ', v)
    # test.reorder(['F1', 'F3', 'F4'])
    # for k, v in test.items():
    #    print(k, ' -> ', v)
    print(test.clear())
    print(dict(test))
    # print(test.nmobject_class.__name__)
