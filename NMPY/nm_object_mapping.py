#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:25:27 2022

@author: jason
"""
from collections.abc import MutableMapping
from typing import Dict, List, Tuple, Union

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
        rename_on: bool = True,  # rename NMObjects
        name_prefix: str = 'NMObject',  # auto name generation of NMObjects
        copy: nmu.NMObjectMappingType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__name_prefix = 'NMObject'
        select = []  # key names of selected NMObjects

        if isinstance(copy, NMObjectMapping):
            rename_on = copy._NMObjectMapping__rename_on
            name_prefix = copy._NMObjectMapping__name_prefix
            select = copy._NMObjectMapping__select
            nmobjects = copy.map_copy()

        if isinstance(rename_on, bool):
            self.__rename_on = rename_on
        else:
            e = self._type_error('rename_on', 'boolean', tp='')
            raise TypeError(e)

        self._name_prefix_set(name_prefix)

        self.__map = {}
        self.update(nmobjects)  # add NMObjects to map

        self.__select = []
        if isinstance(select, list):  # will be from copy
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

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'content_type': self.nmobject_class.__name__})
        k.update({'rename_on': self.__rename_on})
        k.update({'name_prefix': self.__name_prefix})
        k.update({'select': self.__select})
        return k

    @property
    def parameters_content(self) -> List[Dict]:
        plist = []
        for o in self.__container:
            plist.append(o.parameters)
        return plist

    # override, no super
    @property
    def content(self) -> Dict[str, str]:
        return {self.nmobject_class.__name__: self.keys()}

    # MutableMapping required
    def __setitem__(  # NOT ALLOWED
        self,
        key,  # NOT USED
        value
    ) -> None:
        # called by '=' and update()
        # override below: update()
        # e.g. mymap['recorda0'] = mynmobject
        # return self.update(value)
        # print('__setitem__, ' + str(key) + ', ' + str(value))
        e = self._error('use method ' + nmu.quotes('update') + ' and/or ' +
                        nmu.quotes('delete') + ' to change mappings.')
        raise RuntimeError(e)

    # MutableMapping required
    def __getitem__(
        self,
        key,
    ) -> nmu.NMObjectType:
        # called by:
        #     get(), setdefault(), items(), values(), pop(), popitem(), clear()
        # override below: setdefault(), pop(), popitem(), clear()
        # print('__getitem__ ' + str(key))
        key2 = self._getkey(key)
        if key2:
            return self.__map[key2]
        return None

    # MutableMapping required
    def __delitem__(self, key):  # NOT ALLOWED
        # called by 'del', pop(), popitem(), clear()
        # override below: pop(), popitem(), clear()
        # print('__delitem__ ' + str(key))
        # del self.__map[key]
        e = self._error('use method ' + nmu.quotes('pop') +
                        ' to delete mappings.')
        raise RuntimeError(e)

    # MutableMapping required
    def __iter__(self):
        # called by keys(), items(), values(), popitem(), clear()
        # print('__iter__ ')
        return iter(self.__map)

    # MutableMapping required
    def __len__(self):
        return len(self.__map)

    def __contains__(
        self,
        key: str
    ) -> bool:
        # called by 'in'
        # print('__contains__ ' + str(key))
        if self._getkey(key):
            return True
        return False

    def _getkey(
        self,
        key: str
    ) -> str:
        # need this function since keys (names) are case insensitive
        if not isinstance(key, str):
            e = self._type_error('key', 'string')
            raise TypeError(e)
        for k in self.__map.keys():
            if k.lower() == key.lower():
                return k
        return None

    # def __repr__(self):
    #    return f'{type(self).__name__}()'

    # override
    def __eq__(
        self,
        other: nmu.NMObjectMappingType,
    ) -> bool:
        if not super().__eq__(other):
            return False
        if self.__rename_on != other._NMObjectMapping__rename_on:
            return False
        if self.__name_prefix != other._NMObjectMapping__name_prefix:
            return False
        if self.__select != other._NMObjectMapping__select:
            return False
        if len(self) != len(other):
            return False
        for k in other.keys():
            s = self.__getitem__(k)
            if s is None:
                return False
            o = other.get(k)
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
                e = self._error('key ' + nmu.quotes(o.name) +
                                ' already exists.')
                raise KeyError(e)
        for o in olist:
            self.__map[o.name] = o
        self.__update_nmobject_references()
        self._modified()

    def __update_nmobject_references(self):
        for o in self.__map.values():
            o._rename_fxnref_set(self.rename)  # reference of 'rename' fxn
            o._NMObject__parent = self._parent

    # MutableMapping override.
    # have to override to get this function to work.
    # should be called get_value_or_default
    def setdefault(
        self,
        key,
        default=None
    ):
        o = self.__getitem__(key)
        if isinstance(o, NMObject):
            return o
        return default

    # MutableMapping override
    def pop(
        self,
        key,
        default=None,
        confirm: bool = True
    ) -> nmu.NMObjectType:
        key2 = self._getkey(key)
        if not key2:
            if default is None:
                e = self._error(nmu.quotes(key) + ' does not exist.')
                raise KeyError(e)
            else:
                return default
        if confirm:
            q = ('are you sure you want to delete ' + nmu.quotes(key2) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                return None
        self._modified()
        return self.__map.pop(key2)

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
        self._modified()
        return self.__map.popitem()

    # MutableMapping override
    def clear(
        self,
        confirm: bool = True
    ) -> None:
        if len(self) == 0:
            return None
        if confirm:
            q = ('are you sure you want to delete the following?' +
                 '\n' + ', '.join(self.__map.keys()))
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                return None
        self.__map.clear()
        self._modified()

    def rename(
        self,
        key: str,
        newkey: str
    ) -> None:
        if not self.__rename_on:
            e = self._error('key names are locked in this mapping.')
            raise RuntimeError(e)
        key2 = self._getkey(key)
        if not key2:
            e = self._error(nmu.quotes(key) + ' does not exist.')
            raise KeyError(e)
        if not isinstance(newkey, str):
            e = self._type_error('newkey', 'string')
            raise TypeError(e)
        if not newkey or not self.name_ok(newkey, ok='default'):
            e = self._value_error('newkey')
            raise ValueError(e)
        if newkey.lower() == 'default':
            newkey = self.next_default_name()
        if self._getkey(newkey):
            e = self._error('key ' + nmu.quotes(newkey) + ' already exists.')
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
            e = self._error('key mismatch: ' + str(n_new) + ' != ' +
                            str(n_old))
            raise KeyError(e)
        new_map = {}
        for k in newkeys:
            o = self.__getitem__(k)
            if not o:
                e = self._error(nmu.quotes(k) + ' does not exist.')
                raise KeyError(e)
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
            e = self._error('key ' + nmu.quotes(newkey) + ' already exists.')
            raise KeyError(e)
        o = self.__getitem__(key)
        if o is None:
            e = self._error(nmu.quotes(key) + ' does not exist.')
            raise KeyError(e)
        c = o.copy()
        # c.name = newkey  # double history
        c._NMObject__name = newkey  # mangled, avoid double history
        self.__map[c.name] = c
        self.__update_nmobject_references()
        self._modified()
        return c

    @property
    def name_prefix(self) -> str:  # see next_default_name())
        return self.__name_prefix

    @name_prefix.setter
    def name_prefix(
        self,
        prefix: str = 'NMObject'
    ) -> None:
        self._name_prefix_set(prefix)

    def _name_prefix_set(
        self,
        prefix: str = 'NMObject'
        # quiet: bool = nmp.QUIET
    ) -> None:
        if prefix is None:
            prefix = ''
        elif not isinstance(prefix, str):
            e = self._type_error('prefix', 'string')
            raise TypeError(e)
        elif not self.name_ok(prefix, ok=['', 'default']):
            e = self._value_error('prefix')
            raise ValueError(e)
        if prefix.lower() == 'default':
            prefix = 'NMObject'
        if prefix.lower() == self.__name_prefix.lower():
            return None  # nothing to do
        oldprefix = self.__name_prefix
        self.__name_prefix = prefix
        self._modified()
        '''
        if self.__name_prefix is None:
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
                                  prefix=self.__name_prefix)
        return self.__name_prefix + str(inext)

    @property
    def select(self) -> List[nmu.NMObjectType]:
        return self.__select

    @select.setter
    def select(
        self,
        keys: Union[str, List[str]]  # None is ok
    ) -> None:
        self._select_set(keys)

    def select_set(
        self,
        keys: Union[str, List[str]] = None,  # None is ok
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
                e = self._error(nmu.quotes(k) + ' does not exist.')
                raise KeyError(e)
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
                e = self._error(nmu.quotes(k) + ' does not exist.')
                raise KeyError(e)
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
    print(test.__repr__())
    '''
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
    '''
