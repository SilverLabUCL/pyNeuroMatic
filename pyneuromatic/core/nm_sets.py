# -*- coding: utf-8 -*-
"""
[Module description].

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source 
Software Toolkit for Acquisition, Analysis and Simulation of 
Electrophysiological Data. Front. Neuroinform. 12:14. 
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations
import copy
from collections.abc import MutableMapping
from typing import Callable, Any

from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


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

    # Extend NMObject's special attrs with NMSets's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMSets__map",
        "_NMSets__nmobjects",
        "_NMSets__nmobjects_fxnref",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str | None = "NMSets0",
        nmobjects_fxnref: Callable | None = None,
        nmobjects: dict[str, NMObject] | None = None,
        # it is safer to use nmobjects_fxnref than nmobjects
        # to avoid problems of reference change of dictionary
        # e.g. NMObjectContainer rename() or reorder()
    ) -> None:
        actual_name = name if name is not None else "NMSets0"
        super().__init__(
            parent=parent,
            name=actual_name,
        )  # NMObject

        self.__map: dict[str, Any] = {}  # {key: [NMObject]} or {key: [Equation]}
        self.__nmobjects: dict[str, NMObject] = {}  # {keys: [NMObjects]}
        self.__nmobjects_fxnref = self._nmobjects_dict_default

        # check arguments for nmobjects_fxnref and nmobjects
        # create self.__nmobjects
        if nmobjects_fxnref and nmobjects_fxnref.__class__.__name__ == "method":
            if isinstance(nmobjects, dict):
                raise ValueError(
                    "found arguments for '%s' and '%s' "
                    " but only one argument should be passed"
                    % ("nmobjects_fxnref", "nmobjects")
                )
            self.__nmobjects_fxnref = nmobjects_fxnref
        elif isinstance(nmobjects, dict):
            for k, v in nmobjects.items():
                if not isinstance(k, str):
                    e = nmu.type_error_str(k, "nmobjects: key", "string")
                    raise TypeError(e)
                if not isinstance(v, NMObject):
                    e = "nmobjects: '%s' value" % k
                    e = nmu.type_error_str(v, e, "NMObject")
                    raise TypeError(e)
            self.__nmobjects = nmobjects
            # reference to an existing dictionary. DO NOT COPY.
        else:
            raise ValueError(
                "expected argument for '%s' or '%s' but got "
                "neither" % ("nmobjects_fxnref", "nmobjects")
            )

    def __deepcopy__(self, memo: dict) -> NMSets:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMSets by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMSets
        """
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMSets's special attributes

        # __nmobjects: keep as empty dict initially
        # (will be populated by the function reference or set externally)
        result._NMSets__nmobjects = {}

        # __nmobjects_fxnref: point to the new instance's default method
        result._NMSets__nmobjects_fxnref = result._nmobjects_dict_default

        # __map: deep copy the structure, resolving NMObject references through memo
        result._NMSets__map = {}
        for key, olist in self._NMSets__map.items():
            if NMSets.list_is_equation(olist):
                # Equations are just strings, deep copy directly
                result._NMSets__map[key] = copy.deepcopy(olist, memo)
            else:
                # List of NMObjects - try to resolve through memo
                new_olist = []
                for obj in olist:
                    if isinstance(obj, NMObject):
                        if id(obj) in memo:
                            # Object was already copied, use the copy
                            new_olist.append(memo[id(obj)])
                        else:
                            # Object not in memo, keep original reference
                            new_olist.append(obj)
                    else:
                        new_olist.append(obj)
                result._NMSets__map[key] = new_olist

        return result

    @property
    def _nmobjects_dict(self) -> dict[str, NMObject]:
        return self.__nmobjects_fxnref()

    def _nmobjects_dict_default(self) -> dict[str, NMObject]:
        return self.__nmobjects

    # override NMObject.copy()
    def copy(self) -> NMSets:
        """Create a copy of this NMSets using deepcopy."""
        return copy.deepcopy(self)

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"sets": list(self.__map.keys())})
        return k

    # MutableMapping Required Abstract Methods:
    # __getitem__, __setitem__, __delitem__, __iter__, __len__

    # MutableMapping required abstract method
    def __getitem__(
        self, 
        key: str
    ) -> list[NMObject]:
        """
        called by:
            get(), setdefault(), items(), values(), pop(), popitem(), clear()
        """
        return self.get(key)  # type: ignore[return-value]  # returns list of nmobjects

    # MutableMapping required abstract method
    def __setitem__(
        self,
        key: str,
        olist: str | list[str] | NMObject | list[NMObject],
    ) -> None:
        """
        called by '=' and update()
        e.g. sets['Set1'] = ['RecordA0', 'RecordA3']
        e.g. sets['Set3'] = ['Set1', '&', 'Set2']
        ['&', '|', '-', '^']
        """
        self._setitem(key, olist, add=False)

    def _setitem(
        self,
        key: str,
        olist: str | list[str] | NMObject | list[NMObject],
        add: bool = False,
    ) -> None:
        actual_key = self._getkey(key)
        if actual_key is None:  # new key
            actual_key = self._newkey(key)
            new = True
            olist_old: list[Any] | None = None
            add = False
        else:  # key exists
            new = False
            olist_old = self.__map[actual_key]
            if not isinstance(olist_old, list):
                e = "self.__map['%s']: value" % actual_key
                e = nmu.type_error_str(olist_old, e, "list")
                raise TypeError(e)
            if add:
                if NMSets.list_is_equation(olist_old):
                    raise ValueError("cannot 'add' to an existing equation")
                if len(olist_old) == 0:
                    add = False

        items: list[Any]
        if isinstance(olist, (str, NMObject)):
            items = [olist]
        elif isinstance(olist, list):
            items = list(olist)
        else:
            e = nmu.type_error_str(olist, "olist", "string or NMObject or list")
            raise TypeError(e)

        if NMSets.list_is_equation(items):
            # equation is saved, not NMObjects
            # this is important since sets can change
            if not new:  # can only overwrite an existing equation
                if add:
                    raise ValueError("cannot 'add' to an existing equation")
                if olist_old and not NMSets.list_is_equation(olist_old):
                    raise ValueError("set '%s' exists and is not an equation" % key)
            olist_new: list[Any] = []
            for istr in items:
                if isinstance(istr, str) and istr in ["&", "|", "-", "^"]:
                    pass  # ok
                elif isinstance(istr, str):
                    istr = self._getkey(istr)
                    if istr is None:
                        raise KeyError(
                            "olist: equation item '%s' does not exist" % istr
                        )
                olist_new.append(istr)
            if new:
                self.__map[actual_key] = olist_new
            else:
                assert olist_old is not None
                olist_old.clear()
                olist_old.extend(olist_new)
            return None  # finished equation

        allkeys = True
        for o in items:
            if not isinstance(o, str):
                allkeys = False
                break

        if allkeys:  # convert keys to NMObjects
            str_items: list[str] = [
                s for s in items if isinstance(s, str)
            ]
            if add:
                assert olist_old is not None
                for o in olist_old:
                    str_items.append(o.name)
            klist = self.__match_to_nmobject_keys(str_items)
            olist_new = []
            for k in klist:
                nmobj = self._nmobjects_dict.get(k)
                if nmobj not in olist_new:
                    olist_new.append(nmobj)
            if new:
                self.__map[actual_key] = olist_new
            else:
                assert olist_old is not None
                olist_old.clear()
                olist_old.extend(olist_new)
            return None  # finished keys

        # else: items contains NMObjects
        obj_items: list[NMObject] = []
        for o in items:
            if not isinstance(o, NMObject):
                e = nmu.type_error_str(o, "olist: list item", "NMObject")
                raise TypeError(e)
            found = False
            for o2 in self._nmobjects_dict.values():
                if o is o2:
                    found = True
                    break
            if not found:
                raise ValueError("olist: list item '%s' does not exist" % o.name)
            obj_items.append(o)

        if add:
            assert olist_old is not None
            obj_items += olist_old
        olist_new = []  # match order to self._nmobjects_dict
        finished = obj_items.copy()
        for o in self._nmobjects_dict.values():
            if o in obj_items and o not in olist_new:
                olist_new.append(o)
                finished.remove(o)
        for o in finished:  # finished may contain duplicate NMObjects
            if o not in olist_new:
                raise ValueError(
                    "olist: the following list items do not " "exist: " + str(finished)
                )
        if new:
            self.__map[actual_key] = olist_new
        else:
            assert olist_old is not None
            olist_old.clear()
            olist_old.extend(olist_new)
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
        self.pop(key)  # allows delete confirmation

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
        key: object
    ) -> bool:
        # called by 'in'
        # print('__contains__ ' + str(key))
        if not isinstance(key, str):
            return False
        actual_key = self._getkey(key)
        return actual_key != None

    # override MutableMapping mixin method
    def get(
        self,
        key: str,
        default=None,
        get_equation: bool = False,  # extra parameter
        # return equation if one exists, otherwise returns NMObject list
        get_keys: bool = False,  # extra parameter
    ) -> list[NMObject] | list[str] | None:
        actual_key = self._getkey(key)
        if actual_key is None:
            return default

        olist = self.__map[actual_key]

        if NMSets.list_is_equation(olist):
            if get_equation:
                return olist  # equation is saved in map

            set0: set[str] = set() # set0 = set()
            for i, eq_item in enumerate(olist):
                if i == 0:
                    key0 = self._getkey(eq_item)
                    set0 = set()
                    for nmobject in self.__map[key0]:
                        set0.add(nmobject.name)
                    op = ""  # reset
                elif eq_item in ["&", "|", "-", "^"]:
                    op = eq_item
                elif op:
                    key1 = self._getkey(eq_item)
                    set1: set[str] = set()
                    for nmobject in self.__map[key1]:
                        set1.add(nmobject.name)
                    if op == "&":
                        set0 = set0 & set1
                    elif op == "|":
                        set0 = set0 | set1
                    elif op == "-":
                        set0 = set0 - set1
                    elif op == "^":
                        set0 = set0 ^ set1
                    op = ""  # reset
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
        other: object
    ) -> bool:
        ignore_parameters = []
        if not isinstance(other, NMSets):
            return NotImplemented
        if not super().__eq__(other):
            return False
        # first compare self.__map
        s_keys = self.keys()
        if not nmu.keys_are_equal(list(s_keys), list(other.keys())):
            return False
        for s_key in s_keys:
            s_olist = self.__map[s_key]
            if NMSets.list_is_equation(s_olist):
                o_olist_eq = other.get(s_key, get_equation=True)
                if not NMSets.list_is_equation(o_olist_eq):
                    return False
                if not isinstance(s_olist, list) or not isinstance(o_olist_eq, list):
                    return False
                for i, sstr in enumerate(s_olist):
                    # items need to be in same sequence
                    # set names are case insensitive
                    ostr = o_olist_eq[i]
                    if isinstance(sstr, str) and isinstance(ostr, str):
                        if sstr.lower() != ostr.lower():
                            return False
            else:
                o_olist_obj = other.get(s_key)
                if not isinstance(o_olist_obj, list):
                    return False
                if not NMObject.lists_are_equal(
                    s_olist,
                    o_olist_obj,  # type: ignore[arg-type]
                ):
                    return False
        if "nmobjects" not in ignore_parameters:
            for s_key in self._nmobjects_dict.keys():
                o_key = other._get_nmobject_key(s_key)
                if o_key is None:
                    return False
                s = self._nmobjects_dict[s_key]
                o = other._nmobjects_dict[o_key]
                if s != o:
                    return False
        return True

    # MutableMapping Mixin Methods: pop, popitem, clear, update, setdefault

    # override MutableMapping mixin method
    def pop(  # type: ignore[override]
        self,
        key: str
    ) -> list[NMObject] | None:
        # removed 'default' parameter
        key = self._getkey(key)
        o = self.__map.pop(key)
        return o

    # override MutableMapping mixin method
    # type: ignore[override]
    # # delete last item
    def popitem(self) -> tuple[str, list[NMObject]] | None:
        """
        Must override, otherwise first item is deleted rather than last.
        Consider deprecating to prevent accidental deletes.
        """
        if len(self.__map) == 0:
            return None
        klist = list(self.__map.keys())
        key = klist[-1]  # last key
        olist = self.pop(key=key)
        if olist:
            return (key, olist)
        return None

    # override MutableMapping mixin method
    # override so there is only a single delete confirmation
    def clear(self) -> None:
        if len(self) == 0:
            return None
        self.__map.clear()
        return None

    # update() NO OVERRIDE

    # override MutableMapping mixin method
    def setdefault(
            self, 
            key, 
            default=None
        ):
        # have to override to get default option to work
        k = self._getkey(key)
        if k is None:
            if default is None:
                return None
            self.__setitem__(key, default)
            return default
        return self.get(key)

    # NMSets methods...

    def _getkey(
        self,
        key: str | None = None,
    ) -> str | None:
        # wrapper function for input parameter key
        # forces keys/names to be case insensitive
        if key is None:
            return None
        if not isinstance(key, str):
            e = nmu.type_error_str(key, "key", "string or None")
            raise TypeError(e)
        for k in self.__map.keys():
            if k.lower() == key.lower():  # keys are case insensitive
                return k  # return key from self.__map
        return None
    
    def _newkey(
        self,
        newkey: str | None = None,
    ) -> str:
        if newkey is None:
            return self.auto_name_next()
        if not isinstance(newkey, str):
            e = nmu.type_error_str(newkey, "newkey", "string")
            raise TypeError(e)
        if newkey.strip() == "":
            return self.auto_name_next()
        if not nmu.name_ok(newkey, ok_names=["all"]):
            raise ValueError("newkey: %s" % newkey)
        for k in self.__map.keys():
            if k.lower() == newkey.lower():  # keys are case insensitive
                raise KeyError("key '%s' already exists" % newkey)
        return newkey # return new key

    def auto_name_next(
        self, 
        prefix: str = "set", 
        trials: int = 100
    ) -> str:
        if not isinstance(prefix, str):
            e = nmu.type_error_str(prefix, "prefix", "string")
            raise TypeError(e)
        for i in range(trials):
            newkey = prefix + str(i)
            key = self._getkey(newkey)
            if key is None:
                newkey = self._newkey(newkey)
                return newkey
        raise ValueError("failed to find next name for NMSets '%s'" % self.name)

    def _get_nmobject_key(
        self, 
        nmobject_key: str, 
        error1: bool = True, 
        error2: bool = True
    ) -> str | None:
        # wrapper function for nmobject_key
        # forces keys/names to be case insensitive
        if not isinstance(nmobject_key, str):
            if error1:
                e = nmu.type_error_str(nmobject_key, "nmobject_key", "string")
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
        olist: str | list[str] | NMObject | list[NMObject]
    ) -> bool:
        key = self._getkey(key)
        if key is None:
            return False
        items: list[Any]
        if isinstance(olist, (str, NMObject)):
            items = [olist]
        elif isinstance(olist, list):
            items = list(olist)
        else:
            return False
        klist = self.get(key, get_keys=True)
        if not isinstance(klist, list):
            return False
        for o in items:
            if isinstance(o, NMObject):
                k1 = o.name
            elif isinstance(o, str):
                k1 = o
            else:
                return False
            found = False
            for k2 in klist:
                if isinstance(k2, str) and k1.lower() == k2.lower():
                    found = True
                    break
            if not found:
                return False
        return True

    def __match_to_nmobject_keys(
        self, 
        nmobject_keys: list[str]
    ) -> list[str]:
        # match key list to self.__nmobjects
        if isinstance(nmobject_keys, str):
            nmobject_keys = [nmobject_keys]
        elif not isinstance(nmobject_keys, list):
            e = nmu.type_error_str(nmobject_keys, "nmobject_keys", "list")
            raise TypeError(e)
        klist = []
        finished = nmobject_keys.copy()
        for k1 in self._nmobjects_dict.keys():
            if not isinstance(k1, str):
                e = nmu.type_error_str(k1, "self._nmobjects_dict: key", "string")
                raise TypeError(e)
            for k2 in nmobject_keys:
                if not isinstance(k2, str):
                    e = nmu.type_error_str(k2, "nmobject_keys: list item", "string")
                    raise TypeError(e)
                if k1.lower() == k2.lower():
                    if k1 not in klist:
                        klist.append(k1)  # use keys of self.__nmobjects
                    if k2 in finished:
                        finished.remove(k2)
                    # break  # do not break, removes duplicates
        if len(finished) > 0:
            raise KeyError(
                "nmobject_keys: the following keys do not exist: " + str(finished)
            )
        return klist

    def is_equation(
        self,
        key: str,
    ) -> bool:
        key = self._getkey(key)
        if key is None:
            return False
        return NMSets.list_is_equation(self.__map[key])

    @staticmethod
    def list_is_equation(
        equation: object  # e.g. ['set1', '&', 'set2']
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
            if equation[i] in ["&", "|", "-", "^"]:
                found_symbol = True
            else:
                return False
        for i in range(0, len(equation), 2):  # even items should be names
            if not isinstance(equation[i], str):
                return False
        return found_symbol

    def new(
        self, 
        name: str | None = None
    ) -> tuple[str, list[NMObject]]:
        key = self._newkey(name)
        olist: list[Any] = []
        self.__map[key] = olist  # empty set
        return (key, olist)

    def duplicate(
        self, 
        name: str, 
        newname: str | None = None
    ) -> tuple[str, list[NMObject]]:
        if name is None:
            raise ValueError("key name cannot be None")
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        newkey = self._newkey(newname)
        clist = []
        for o in self.__map[key]:
            if o not in clist:
                clist.append(o)
        self.__map[newkey] = clist
        return (newkey, clist)

    def rename(
            self, 
            name: str, 
            newname: str | None = None
    ) -> str:
        key = self._getkey(name)
        newkey = self._newkey(newname)
        new_map = {}
        for k, olist in self.__map.items():
            if k.lower() == key.lower():
                new_map.update({newkey: olist})
            else:
                new_map.update({k: olist})
        self.__map.clear()
        self.__map.update(new_map)
        return newkey

    def reorder(
        self, 
        name_order: list[str]
    ) -> None:
        """
        Cannot change map key names.
        TODO: order by name, creation date, modified date
        """
        if not isinstance(name_order, list):
            e = nmu.type_error_str(name_order, "newkeyorder", "list")
            raise TypeError(e)
        for name in name_order:
            if not isinstance(name, str):
                e = nmu.type_error_str(name, "newkeyorder: list item", "string")
                raise TypeError(e)
        n_new = len(name_order)
        n_old = len(self.__map)
        if n_new != n_old:
            raise KeyError("number of keys mismatch: '%s' != '%s'" % (n_new, n_old))
        if name_order == list(self.__map.keys()):
            return None  # nothing to do
        new_map = {}
        for name in name_order:
            name = self._getkey(name)
            new_map[name] = self.__map[name]
        self.__map.clear()
        self.__map.update(new_map)
        return None

    def empty(
        self,
        key: str
    ) -> None:
        key = self._getkey(key)
        olist = self.__map[key]
        olist.clear()
        return None

    def empty_all(self) -> None:
        for olist in self.__map.values():
            olist.clear()    
        return None

    def add(
        self,
        key: str,
        olist: str | list[str] | NMObject | list[NMObject] | None = None,
    ) -> None:
        if olist is None:
            olist = []
        self._setitem(key, olist, add=True)
        return None

    def remove(
        self,
        key: str,
        olist: str | list[str] | NMObject | list[NMObject],
        error: bool = True,
    ) -> list[NMObject]:
        key = self._getkey(key)
        items: list[Any]
        if isinstance(olist, (str, NMObject)):
            items = [olist]
        elif isinstance(olist, list):
            items = list(olist)
        else:
            if error:
                e = nmu.type_error_str(olist, "olist", "string or NMObject or list")
                raise TypeError(e)
            return []
        olist_old = self.__map[key]
        if NMSets.list_is_equation(olist_old):
            if error:
                raise ValueError(
                    "cannot remove NMObject items from "
                    "'%s' since it is an equation" % key
                )
            return []
        remove_list: list[NMObject] = []
        for o1 in items:
            found = False
            for o2 in olist_old:
                if not isinstance(o2, NMObject):
                    if error:
                        e = "self.__map['%s']: list item" % key
                        e = nmu.type_error_str(o2, e, "NMObject")
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
                        e = "olist: list item"
                        e = nmu.type_error_str(o1, e, "string or NMObject")
                        raise TypeError(e)
                    return []
            if not found:
                if error:
                    if isinstance(o1, NMObject):
                        oname = o1.name
                    else:
                        oname = o1
                    raise ValueError("set '%s' does not contain '%s'" % (key, oname))
                return []
        for o in remove_list:
            olist_old.remove(o)
        return remove_list

    def remove_from_all(
        self,
        olist: str | list[str] | NMObject | list[NMObject],
        error: bool = False,
    ) -> None:
        items: list[Any]
        if isinstance(olist, (str, NMObject)):
            items = [olist]
        elif isinstance(olist, list):
            items = list(olist)
        else:
            e = nmu.type_error_str(olist, "olist", "string or NMObject or list")
            raise TypeError(e)
        for o in items:
            if isinstance(o, (str, NMObject)):
                pass
            else:
                e = "olist: list item"
                e = nmu.type_error_str(o, e, "string or NMObject")
                raise TypeError(e)
            for map_key, map_val in self.__map.items():
                if not NMSets.list_is_equation(map_val):
                    self.remove(map_key, o, error=error)
        return None
