#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 07:45:13 2023

@author: jason
"""
from collections.abc import MutableMapping
from typing import Callable, Dict, List, Tuple, Union

from pyneuromatic.nm_object import NMObject
import pyneuromatic.nm_preferences as nmp
import pyneuromatic.nm_utilities as nmu


class NMSets(NMObject, MutableMapping):
    """
    A NM set behaves like a Python set, but is a list and therefore can have
    ordered content.

    MutableMapping
    https://docs.python.org/3/library/collections.abc.html
    Abstract Base Classes for Containers

    Required abstract methods:
        __getitem__, __setitem__, __delitem__, __iter__, __len__

    Inherits from Mapping (Mixin Methods):
        __contains__, keys, items, values, get, __eq__, __ne__

    MutableMapping Mixin Methods:
        pop, popitem, clear, update, setdefault
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: Union[str, None] = 'NMSets0',
        nmobjects_fxnref: Union[Callable, None] = None,
        nmobjects: Union[Dict[str, nmu.NMObjectType], None] = None,
        # it is safer to use nmobjects_fxnref than nmobjects
        # to avoid problems of reference change of dictionary
        # e.g. NMObjectContainer rename() or reorder()
        copy: Union[nmu.NMSetsType, None] = None  # see copy()
    ) -> None:

        super().__init__(
            parent=parent,
            name=name,
            notes_on=False,  # turn notes off during __init__
            copy=copy
        )  # NMObject

        self.__map = {}  # {key: [NMObject]} or {key: [Equation]}
        self.__nmobjects = {}  # {keys: [NMObjects]}
        self.__nmobjects_fxnref = self._nmobjects_dict_default

        # self._eq_list.append('nmobjects')
        # probably should not test self.__nmobjects
        # self.__nmobjects is reference to external {}

        # check arguments for nmobjects_fxnref and nmobjects
        # create self.__nmobjects
        if (
            nmobjects_fxnref and
            nmobjects_fxnref.__class__.__name__ == 'method'
           ):
            if isinstance(nmobjects, dict):
                raise ValueError("found arguments for '%s' and '%s' "
                                 ' but only one argument should be passed'
                                 % ('nmobjects_fxnref', 'nmobjects'))
            self.__nmobjects_fxnref = nmobjects_fxnref
        elif isinstance(nmobjects, dict):
            for k, v in nmobjects.items():
                if not isinstance(k, str):
                    e = nmu.typeerror(k, 'nmobjects: key', 'string')
                    raise TypeError(e)
                if not isinstance(v, NMObject):
                    e = "nmobjects: '%s' value" % k
                    e = nmu.typeerror(v, e, 'NMObject')
                    raise TypeError(e)
            self.__nmobjects = nmobjects
            # reference to an existing dictionary. DO NOT COPY.
        else:
            raise ValueError("expected argument for '%s' or '%s' but got "
                             'neither' % ('nmobjects_fxnref', 'nmobjects'))

        if copy is None:
            pass
        elif isinstance(copy, NMSets):
            # DO NOT COPY self.__nmobjects
            # Use keys to grab objects from self.__nmobjects
            # Hence, self.__nmobjects must exist before this copy

            # do not use copy.values() in for-loop
            # values() does not return set equation
            for key in copy.keys():
                if not isinstance(key, str):
                    e = nmu.typeerror(key, 'copy: key', 'string')
                    raise TypeError(e)
                for k in self.__map.keys():
                    if key.lower() == k.lower():
                        raise KeyError("copy: found multiple keys for '%s'"
                                       % k)
                olist = copy.get(key, get_equation=True)
                if not isinstance(olist, list):
                    e = "copy: '%s' value" % key
                    e = nmu.typeerror(olist, e, 'list')
                    raise TypeError(e)
                if NMSets.listisequation(olist):
                    self.__map[key] = olist.copy()
                    continue
                # else copy normal set
                olist_new = []
                finished = olist.copy()
                for k, o1 in self._nmobjects_dict.items():
                    # maintain list order of self._nmobjects_dict
                    if not isinstance(o1, NMObject):
                        e = "self._nmobjects_dict: '%s' value" % k
                        e = nmu.typeerror(o1, e, 'NMObject')
                        raise TypeError(e)
                    found = False
                    for o2 in olist:
                        # want matching type, so do not use isinstance()
                        if type(o2) != type(o1):
                            e = "copy: '%s' value" % key
                            e = nmu.typeerror(o2, e, type(o1).__name__)
                            raise TypeError(e)
                        if o1.name.lower() == o2.name.lower():
                            found = True
                            break
                    if found:
                        olist_new.append(o1)
                        finished.remove(o2)
                if len(finished) > 0:
                    nlist = []
                    for o in finished:
                        nlist.append(o.name)
                    raise KeyError("copy: '%s': the following list "
                                   'items do not exist: ' + str(nlist) % key)
                self.__map[key] = olist_new
        else:
            e = nmu.typeerror(copy, 'copy', 'NMSets')
            raise TypeError(e)

        self.notes_on = True

        return None

    @property
    def _nmobjects_dict(self) -> Dict[str, nmu.NMObjectType]:
        return self.__nmobjects_fxnref()

    def _nmobjects_dict_default(self) -> Dict[str, nmu.NMObjectType]:
        return self.__nmobjects

    # override, no super
    def copy(
        self,
        nmobjects: Union[Dict[str, nmu.NMObjectType], None] = None,
        nmobjects_fxnref: Union[Callable, None] = None,
    ) -> nmu.NMSetsType:
        return NMSets(nmobjects=nmobjects, nmobjects_fxnref=nmobjects_fxnref,
                      copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'sets': list(self.__map.keys())})
        return k

    # MutableMapping Required Abstract Methods:
    # __getitem__, __setitem__, __delitem__, __iter__, __len__

    # MutableMapping required abstract method
    def __getitem__(
        self,
        key: str
    ) -> List[nmu.NMObjectType]:
        """
        called by:
            get(), setdefault(), items(), values(), pop(), popitem(), clear()
        """
        return self.get(key)  # return list of nmobjects

    # MutableMapping required abstract method
    def __setitem__(
        self,
        key: str,
        olist: Union[str, List[str], nmu.NMObjectType, List[nmu.NMObjectType]],
        add: bool = False  # extra parameter
        # NMObject names or set equation or NMObjects
    ) -> None:
        # print('__setitem__: ' + key + ', ' + str(olist))
        """
        called by '=' and update()
        e.g. sets['Set1'] = ['RecordA0', 'RecordA3']
        e.g. sets['Set3'] = ['Set1', '&', 'Set2']
        ['&', '|', '-', '^']
        """

        k = self._getkey(key, error2=False)
        if k is None:  # new key
            key = self._newkey(key)
            new = True
            olist_old = None
            add = False
        else:  # key exists
            key = k
            new = False
            olist_old = self.__map[key]
            if not isinstance(olist_old, list):
                e = "self.__map['%s']: value" % key
                e = nmu.typeerror(olist_old, e, 'list')
                raise TypeError(e)
            if add:
                if NMSets.listisequation(olist_old):
                    raise ValueError("cannot 'add' to an existing equation")
                if len(olist_old) == 0:
                    add = False

        if isinstance(olist, str) or isinstance(olist, NMObject):
            olist = [olist]
        elif not isinstance(olist, list):
            e = nmu.typeerror(olist, 'olist', 'string or NMObject or list')
            raise TypeError(e)

        if NMSets.listisequation(olist):
            # equation is saved, not NMObjects
            # this is important since sets can change
            if not new:  # can only overwrite an existing equation
                if add:
                    raise ValueError("cannot 'add' to an existing equation")
                if olist_old and not NMSets.listisequation(olist_old):
                    raise ValueError("set '%s' exists and is not an equation"
                                     % key)
            olist_new = []
            for istr in olist:
                if istr in ['&', '|', '-', '^']:
                    pass  # ok
                else:
                    istr = self._getkey(istr)
                olist_new.append(istr)
            if new:
                self.__map[key] = olist_new
            else:
                olist_old.clear()
                olist_old.extend(olist_new)
            self.modified()
            return None  # finished equation

        allkeys = True
        for o in olist:
            if not isinstance(o, str):
                allkeys = False
                break

        if allkeys:  # convert keys to NMObjects
            if add:
                for o in olist_old:
                    olist.append(o.name)
            klist = self.__match_to_nmobject_keys(olist)
            olist_new = []
            for k in klist:
                o = self._nmobjects_dict.get(k)
                if o not in olist_new:
                    olist_new.append(o)
            if new:
                self.__map[key] = olist_new
            else:
                olist_old.clear()
                olist_old.extend(olist_new)
            self.modified()
            return None  # finished keys

        # else: olist contains NMObjects

        for o in olist:
            if not isinstance(o, NMObject):
                e = nmu.typeerror(o, 'olist: list item', 'NMObject')
                raise TypeError(e)
            found = False
            for o2 in self._nmobjects_dict.values():
                if o is o2:
                    found = True
                    break
            if not found:
                raise ValueError("olist: list item '%s' does not exist"
                                 % o.name)

        if add:
            olist += olist_old
        olist_new = []  # match order to self._nmobjects_dict
        finished = olist.copy()
        for o in self._nmobjects_dict.values():
            if o in olist and o not in olist_new:
                olist_new.append(o)
                finished.remove(o)
        for o in finished:  # finished may contain duplicate NMObjects
            if o not in olist_new:
                raise ValueError('olist: the following list items do not '
                                 'exist: ' + str(finished))
        if new:
            self.__map[key] = olist_new
        else:
            olist_old.clear()
            olist_old.extend(olist_new)
        self.modified()
        return None

    # MutableMapping required abstract method
    def __delitem__(
        self,
        key: str
    ) -> None:
        """
        called by: 'del', pop(), popitem(), clear()
        # e.g. del self.__map[key]
        """
        return self.pop(key)  # allows delete confirmation

    # MutableMapping required abstract method
    def __iter__(self):
        # called by keys(), items(), values(), popitem(), clear()
        # print('__iter__ ')
        return iter(self.__map)

    # MutableMapping required abstract method
    def __len__(self):
        return len(self.__map)

    # MutableMapping Mixin Methods inherited from Mapping:
    # __contains__, keys, items, values, get, __eq__, __ne__
    # keys() NO OVERRIDE
    # items() NO OVERRIDE
    # values() NO OVERRIDE
    # __ne__() NO OVERRIDE
    # no way to obtain set equation via items() or values()

    # override MutableMapping mixin method
    def __contains__(
        self,
        key: str
    ) -> bool:
        # called by 'in'
        # print('__contains__ ' + str(key))
        key = self._getkey(key, error1=False, error2=False)
        return key is not None

    # override MutableMapping mixin method
    def get(
        self,
        key: str,
        default=None,
        get_equation: bool = False,  # extra parameter
        # return equation if one exists, otherwise returns NMObject list
        get_keys: bool = False  # extra parameter
    ) -> Union[List[nmu.NMObjectType], List[str]]:

        key = self._getkey(key, error1=False, error2=False)
        if key is None:
            return default

        olist = self.__map[key]

        if NMSets.listisequation(olist):

            if get_equation:
                return olist  # equation is saved in map

            set0 = set()
            for i, eq_item in enumerate(olist):
                if i == 0:
                    key0 = self._getkey(eq_item)
                    set0 = set()
                    for nmobject in self.__map[key0]:
                        set0.add(nmobject.name)
                    op = ''  # reset
                elif eq_item in ['&', '|', '-', '^']:
                    op = eq_item
                elif op:
                    key1 = self._getkey(eq_item)
                    set1 = set()
                    for nmobject in self.__map[key1]:
                        set1.add(nmobject.name)
                    if op == '&':
                        set0 = set0 & set1
                    elif op == '|':
                        set0 = set0 | set1
                    elif op == '-':
                        set0 = set0 - set1
                    elif op == '^':
                        set0 = set0 ^ set1
                    op = ''  # reset
            klist = list(set0)
            klist = self.__match_to_nmobject_keys(klist)

            if get_keys:
                return klist

            olist_new = []
            for k in klist:
                nmobject = self._nmobjects_dict.get(k)
                if nmobject and nmobject not in olist_new:
                    olist_new.append(nmobject)
            return olist_new  # nmobjects

        # else set is not an equation

        # if get_equation:
        #    return default

        if get_keys:
            klist = []
            for nmobject in olist:
                klist.append(nmobject.name)
            klist = self.__match_to_nmobject_keys(klist)
            return klist

        return olist

    # override MutableMapping mixin method
    # children should override if new parameters are declared
    def __eq__(
        self,
        other: nmu.NMSetsType
    ) -> bool:
        if not super().__eq__(other):
            return False
        # first compare self.__map
        s_keys = self.keys()
        if not nmu.keys_are_equal(s_keys, other.keys()):
            return False
        for s_key in s_keys:
            s_olist = self.__map[s_key]
            if NMSets.listisequation(s_olist):
                o_olist = other.get(s_key, get_equation=True)
                if not NMSets.listisequation(o_olist):
                    return False
                for i, sstr in enumerate(s_olist):
                    # items need to be in same sequence
                    # set names are case insensitive
                    ostr = o_olist[i]
                    if sstr.lower() != ostr.lower():
                        return False
            else:
                o_olist = other.get(s_key)
                if not NMObject.lists_are_equal(s_olist, o_olist):
                    return False
        if 'nmobjects' in self._eq_list:
            for s_key in self._nmobjects_dict.keys():
                o_key = other._get_nmobject_key(s_key, error1=False,
                                                error2=False)
                if o_key is None:
                    return False
                s = self._nmobjects_dict[s_key]
                o = other._nmobjects_dict[o_key]
                if s != o:
                    return False
        return True

    # MutableMapping Mixin Methods: pop, popitem, clear, update, setdefault

    # override MutableMapping mixin method
    def pop(
        self,
        key: str,
        confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> List[nmu.NMObjectType]:
        # removed 'default' parameter
        key = self._getkey(key)
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_LIST:
                ync = confirm_answer
            else:
                q = ("are you sure you want to delete '%s'?" % key)
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == 'y' or ync.lower() == 'yes':
                pass
            else:
                print("cancel pop '%s'" % key)
                return None
        o = self.__map.pop(key)
        self.modified()
        return o

    # override MutableMapping mixin method
    def popitem(  # delete last item
        self,
        confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> Tuple[str, List[nmu.NMObjectType]]:
        """
        Must override, otherwise first item is deleted rather than last.
        Consider deprecating to prevent accidental deletes.
        """
        if len(self.__map) == 0:
            return ()
        klist = list(self.__map.keys())
        key = klist[-1]  # last key
        olist = self.pop(key=key, confirm_answer=confirm_answer)
        if olist:
            return (key, olist)
        return ()  # return tuple to be consistent with Python

    # override MutableMapping mixin method
    # override so there is only a single delete confirmation
    def clear(
        self,
        confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> None:
        if len(self) == 0:
            return None
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_LIST:
                ync = confirm_answer
            else:
                q = ('are you sure you want to delete the following?\n' +
                     ', '.join(self.__map.keys()))
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == 'y' or ync.lower() == 'yes':
                pass
            else:
                print('cancel delete all')
                return None
        self.__map.clear()
        self.modified()
        return None

    # update() NO OVERRIDE

    # override MutableMapping mixin method
    def setdefault(
        self,
        key,
        default=None
    ):
        # have to override to get default option to work
        k = self._getkey(key, error1=False, error2=False)
        if k is None:
            if default is None:
                return None
            self.__setitem__(key, default)
            return default
        return self.get(key)

    # NMSets methods...

    def _getkey(
        self,
        key: str,  # set key
        error1: bool = True,
        error2: bool = True
    ) -> str:
        # wrapper function for input parameter key
        # forces keys/names to be case insensitive
        if not isinstance(key, str):
            if error1:
                e = nmu.typeerror(key, 'key', 'string')
                raise TypeError(e)
            return None
        for k in self.__map.keys():
            if k.lower() == key.lower():  # keys are case insensitive
                return k  # return key from self.__map
        if error2:
            raise KeyError("key '%s' does not exist" % key)
        return None

    def _newkey(
        self,
        newkey: str,
    ) -> str:
        if not isinstance(newkey, str):
            e = nmu.typeerror(newkey, 'newkey', 'string')
            raise TypeError(e)
        if newkey.lower() == 'default':
            return self.name_next()
        if not newkey or not nmu.name_ok(newkey, ok_names=['all']):
            raise ValueError("newkey: %s" % newkey)
        for k in self.__map.keys():
            if k.lower() == newkey.lower():  # keys are case insensitive
                raise KeyError("key '%s' already exists" % newkey)
        return newkey

    def name_next(
        self,
        prefix: str = "set",
        trials: int = 100
    ) -> str:
        if not isinstance(prefix, str):
            e = nmu.typeerror(prefix, 'prefix', 'string')
            raise TypeError(e)
        for i in range(trials):
            newkey = prefix + str(i)
            key = self._getkey(newkey, error2=False)
            if key is None:
                newkey = self._newkey(newkey)
                return newkey
        raise ValueError("failed to find next name for NMSets '%s'"
                         % self.name)

    def _get_nmobject_key(
        self,
        nmobject_key: str,
        error1: bool = True,
        error2: bool = True
    ) -> str:
        # wrapper function for nmobject_key
        # forces keys/names to be case insensitive
        if not isinstance(nmobject_key, str):
            if error1:
                e = nmu.typeerror(nmobject_key, 'nmobject_key', 'string')
                raise TypeError(e)
            return None
        for k in self._nmobjects_dict.keys():
            if k.lower() == nmobject_key.lower():
                return k  # return key of nmobjects_dict
        if error2:
            raise KeyError("nmobject_key '%s' does not exist" % nmobject_key)
        return None

    def contains(
        self,
        key: str,
        olist: Union[str, List[str], nmu.NMObjectType, List[nmu.NMObjectType]]
    ) -> bool:
        key = self._getkey(key, error1=False, error2=False)
        if key is None:
            return False
        if isinstance(olist, str) or isinstance(olist, NMObject):
            olist = [olist]
        elif not isinstance(olist, list):
            return False
        klist = self.get(key, get_keys=True)
        for o in olist:
            if isinstance(o, NMObject):
                k1 = o.name
            elif isinstance(o, str):
                k1 = o
            else:
                return False
            found = False
            for k2 in klist:
                if k1.lower() == k2.lower():
                    found = True
                    break
            if not found:
                return False
        return True

    def __match_to_nmobject_keys(
        self,
        nmobject_keys: List[str]
    ) -> List[str]:
        # match key list to self.__nmobjects
        if isinstance(nmobject_keys, str):
            nmobject_keys = [nmobject_keys]
        elif not isinstance(nmobject_keys, list):
            e = nmu.typeerror(nmobject_keys, 'nmobject_keys', 'list')
            raise TypeError(e)
        klist = []
        finished = nmobject_keys.copy()
        for k1 in self._nmobjects_dict.keys():
            if not isinstance(k1, str):
                e = nmu.typeerror(k1, 'self._nmobjects_dict: key', 'string')
                raise TypeError(e)
            for k2 in nmobject_keys:
                if not isinstance(k2, str):
                    e = nmu.typeerror(k2, 'nmobject_keys: list item', 'string')
                    raise TypeError(e)
                if k1.lower() == k2.lower():
                    if k1 not in klist:
                        klist.append(k1)  # use keys of self.__nmobjects
                    if k2 in finished:
                        finished.remove(k2)
                    # break  # do not break, removes duplicates
        if len(finished) > 0:
            raise KeyError('nmobject_keys: the following keys do not exist: ' +
                           str(finished))
        return klist

    def isequation(
        self,
        key: str,
    ) -> bool:
        key = self._getkey(key, error1=False, error2=False)
        if not key:
            return False
        return NMSets.listisequation(self.__map[key])

    def listisequation(
        equation: List[str],  # e.g. ['set1', '&', 'set2']
    ) -> bool:
        # checks format of list items
        # does not check if sets exist
        if not isinstance(equation, list):
            return False
        n = len(equation)
        if n < 3:
            return False  # need at least 3 items
        found_symbol = False
        for i in range(1, n, 2):  # odd items should be symbol
            if equation[i] in ['&', '|', '-', '^']:
                found_symbol = True
            else:
                return False
        for i in range(0, len(equation), 2):  # even items should be names
            if not isinstance(equation[i], str):
                return False
        return found_symbol

    def new(
        self,
        newkey: str = 'default'
    ) -> str:
        newkey = self._newkey(newkey)
        olist = []
        self.__map[newkey] = olist  # empty set
        self.modified()
        return (newkey, olist)

    def duplicate(
        self,
        key: str,
        newkey: str = 'default'
    ) -> Dict[str, List[nmu.NMObjectType]]:
        key = self._getkey(key)
        newkey = self._newkey(newkey)
        clist = []
        for o in self.__map[key]:
            if o not in clist:
                clist.append(o)
        self.__map[newkey] = clist
        self.modified()
        return (newkey, clist)

    def rename(
        self,
        key: str,
        newkey: str = 'default'
    ) -> str:
        key = self._getkey(key)
        newkey = self._newkey(newkey)
        new_map = {}
        for k, olist in self.__map.items():
            if k.lower() == key.lower():
                new_map.update({newkey: olist})
            else:
                new_map.update({k: olist})
        self.__map.clear()
        self.__map.update(new_map)
        self.modified()
        return newkey

    def reorder(
        self,
        newkeyorder: List[str]
    ) -> None:
        """
        Cannot change map key names.
        TODO: order by name, creation date, modified date
        """
        if not isinstance(newkeyorder, list):
            e = nmu.typeerror(newkeyorder, 'newkeyorder', 'list')
            raise TypeError(e)
        for k in newkeyorder:
            if not isinstance(k, str):
                e = nmu.typeerror(k, 'newkeyorder: list item', 'string')
                raise TypeError(e)
        n_new = len(newkeyorder)
        n_old = len(self.__map)
        if n_new != n_old:
            raise KeyError("number of keys mismatch: '%s' != '%s'"
                           % (n_new, n_old))
        if newkeyorder == list(self.__map.keys()):
            return None  # nothing to do
        new_map = {}
        for k in newkeyorder:
            k = self._getkey(k)
            new_map[k] = self.__map[k]
        self.__map.clear()
        self.__map.update(new_map)
        self.modified()
        return None

    def empty(
        self,
        key: str,
        confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> None:
        key = self._getkey(key)
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_LIST:
                ync = confirm_answer
            else:
                q = ("are you sure you want to empty '%s'?" % key)
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == 'y' or ync.lower() == 'yes':
                pass
            else:
                print("cancel empty '%s'" % key)
                return None
        olist = self.__map[key]
        olist.clear()
        self.modified()
        return None

    def empty_all(
        self,
        confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> None:
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_LIST:
                ync = confirm_answer
            else:
                q = ('are you sure you want to empty the following?\n' +
                     ', '.join(self.__map.keys()))
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == 'y' or ync.lower() == 'yes':
                pass
            else:
                print('cancel empty all')
                return None
        for olist in self.__map.values():
            olist.clear()
            self.modified()
        return None

    def add(
        self,
        key: str,
        olist: Union[str, List[str], nmu.NMObjectType,
                     List[nmu.NMObjectType]] = []
    ) -> None:
        self.__setitem__(key, olist, add=True)
        return None

    def remove(
        self,
        key: str,
        olist: Union[str, List[str], nmu.NMObjectType, List[nmu.NMObjectType]],
        error: bool = True
    ) -> List[nmu.NMObjectType]:
        key = self._getkey(key)
        if isinstance(olist, str) or isinstance(olist, NMObject):
            olist = [olist]
        elif not isinstance(olist, list):
            if error:
                e = nmu.typeerror(olist, 'olist', 'string or NMObject or list')
                raise TypeError(e)
            return []
        olist_old = self.__map[key]
        if NMSets.listisequation(olist_old):
            if error:
                raise ValueError('cannot remove NMObject items from '
                                 "'%s' since it is an equation" % key)
            return []
        remove_list = []
        for o1 in olist:
            found = False
            for o2 in olist_old:
                if not isinstance(o2, NMObject):
                    if error:
                        e = "self.__map['%s']: list item" % key
                        e = nmu.typeerror(o2, e, 'NMObject')
                        raise TypeError(e)
                    return []
                if isinstance(o1, str):
                    if o1.lower() == o2.name.lower():
                        remove_list.append(o2)
                        found = True
                        break
                elif isinstance(o1, NMObject):
                    if o1 is o2:
                        remove_list.append(o2)
                        found = True
                        break
                else:
                    if error:
                        e = 'olist: list item'
                        e = nmu.typeerror(o1, e, 'string or NMObject')
                        raise TypeError(e)
                    return []
            if not found:
                if error:
                    if isinstance(o1, NMObject):
                        oname = o1.name
                    else:
                        oname = o1
                    raise ValueError("set '%s' does not contain '%s'"
                                     % (key, oname))
                return []
        for o in remove_list:
            olist_old.remove(o)
            self.modified()
        return remove_list

    def remove_from_all(
        self,
        olist: Union[str, List[str], nmu.NMObjectType, List[nmu.NMObjectType]],
        error: bool = False
    ) -> None:
        if isinstance(olist, str) or isinstance(olist, NMObject):
            olist = [olist]
        elif not isinstance(olist, list):
            e = nmu.typeerror(olist, 'olist', 'string or NMObject or list')
            raise TypeError(e)
        for o in olist:
            if isinstance(o, str) or isinstance(o, NMObject):
                pass
            else:
                e = 'olist: list item'
                e = nmu.typeerror(o, e, 'string or NMObject')
                raise TypeError(e)
            for key, olist in self.__map.items():
                if not NMSets.listisequation(olist):
                    self.remove(key, o, error=error)
        return None
