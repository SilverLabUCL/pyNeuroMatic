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
from enum import Enum, auto

from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.core.nm_sets import NMSets
import pyneuromatic.core.nm_utilities as nmu


class ExecuteMode(Enum):
    """Specifies which NMObjects to operate on in a container."""
    SELECTED = auto()  # the currently selected NMObject
    ALL = auto()       # all NMObjects in the container
    NAME = auto()      # a specific NMObject by name (use execute_target_name)
    SET = auto()       # all NMObjects in a named set (use execute_target_name)


class NMObjectContainer(NMObject, MutableMapping):
    """
    NM container (Mapping) of NMObjects. Allows creation of the NM class tree.

    NM class tree:

    NMManager
        NMProject (root)
            NMFolderContainer
                NMFolder (folder0, folder1...)
                    NMDataContainer
                        NMData (recordA0, recordA1... avgA0, avgB0)
                    NMDataSeriesContainer
                        NMDataSeries (record, avg...)
                            NMChannelContainer
                                NMChannel (A, B, C...)
                            NMEpochContainer
                                NMEpoch (E0, E1, E2...)

    Known children of NMObjectContainer:
        NMChannelContainer, NMDataContainer, NMDataSeriesContainer,
        NMEpochContainer, NMFolderContainer

    A NMObjectContainer behaves like a Python dictionary, where each
    NMObject is entered with a key (k) and value (v). The NMObject name
    is the key and the NMObject reference is the value. Keys/names
    must be unique and are case-insensitive.

    Unique NMObject keys/names can be automatically generated using the
    'name_prefix' and 'name_seq_format' attributes. See name_next().

    One key can be chosen as the 'select_key'. This is used for a 'select_key'
    (focused) navigation through the NM class tree for data display and
    analysis.

    NMObjects within a NMObjectContainer can be added to one or more NM sets.
    The NM sets can then be used for execution of NM functions via the
    NMManager. See NMManager.execute_values().

    See Attributes and Properties of NMObject.

    Attributes:
        __rename_on (bool):
        __name_prefix (str):
        __name_seq_format (str)
        __name_seq_counter (str)
        __select (str):
        __sets (Dict{NMObjects})
        __sets_select (str):
        __map (Dict{NMObjects})

    Properties: (@property)
        parameters
        content
        content_parameters
        name_prefix
        name_seq_format
        _name_seq_counter
        select
        sets_select

    Children of this class should override:
        copy()
        content_type()
        new()

    Children of this class should override if new parameters declared:
        parameters()
        __eq__()

    MutableMapping
    https://docs.python.org/3/library/collections.abc.html
    Abstract Base Classes for Containers

    Required abstract methods:
        __getitem__, __setitem__, __delitem__, __iter__, __len__

    Inherits from Mapping (Mixin Methods):
        __contains__, keys, items, values, get, __eq__, __ne__

    MutableMapping Mixin Methods:
        pop, popitem, clear, update, setdefault

    Dict is ordered for Python 3.7
    """

    # Extend NMObject's special attrs with NMObjectContainer's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMObjectContainer__map",
        "_NMObjectContainer__sets",
    })

    def __init__(
        self,
        parent: object | None = None,  # for creating NM class tree
        name: str = "NMObjectContainer0",
        rename_on: bool = True,  # allow renaming of NMobjects
        auto_name_prefix: str = "NMObject",
        # used for auto-name-generation of NMObjects
        auto_name_seq_format: str = "0",
        # used for auto-name-generation of NMObjects
        # e.g. '0'  ->  0, 1, 2, 3...
        # e.g. '00' ->  00, 01, 02, 03...
        # e.g. 'A'  ->  A, B, C, D...
        # e.g. 'AA' ->  AA, AB, AC, AD...
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
        )  # NMObject

        self.__rename_on = True
        self.__auto_name_prefix = "NMObject"
        self.__auto_name_seq_format = "0"
        self.__auto_name_seq_counter = "0"
        self.__sets: NMSets

        # selected_name: the selected/focused NMObject in this container
        # used for navigating the NM class tree (e.g., selected project, folder, data)
        # only one NMObject can be selected at a time; None if nothing selected
        self.__selected_name: str | None = None

        # execute_mode: specifies which NMObjects to operate/execute on (see ExecuteMode enum)
        # execute_target_name: used with NAME or SET modes to specify which object/set
        # see execute_targets property for resolved NMObject list
        self.__execute_mode: ExecuteMode = ExecuteMode.SELECTED
        self.__execute_target_name: str | None = None

        if not isinstance(rename_on, bool):
            e = nmu.type_error_str(rename_on, "rename_on", "boolean")
            raise TypeError(e)

        self.__rename_on = rename_on

        self._auto_name_prefix_set(auto_name_prefix, quiet=True)
        # self.__name_prefix

        self._auto_name_seq_format_set(auto_name_seq_format, quiet=True)
        # self.__name_seq_format

        self.__map: dict[str, NMObject] = {}  # where NMObjects are stored/mapped

        self.__sets = NMSets(
            name="NMObjectContainerSets",
            nmobjects_fxnref=self._get_map,
        )


    def _get_map(self) -> dict[str, NMObject]:  # see __init__()
        return self.__map

    # override NMObject method
    # children should override if new parameters are declared
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"content_type": self.content_type().lower()})
        k.update({"rename_on": self.__rename_on})
        k.update({"auto_name_prefix": self.__auto_name_prefix})
        k.update({"auto_name_seq_format": self.__auto_name_seq_format})
        k.update({"selected_name": self.__selected_name})
        k.update({"execute_mode": self.__execute_mode.name})
        k.update({"execute_target_name": self.__execute_target_name})
        if self.__sets:
            k.update({"sets": list(self.__sets.keys())})
        return k

    # override NMObject method, no super
    @property
    def content(self) -> dict[str, str]:
        cname = self.__class__.__name__.lower()
        c = {cname: self.name}
        for k in self.__map.keys():
            c[k] = self.__map[k].name # key == name, so redundant
        return c

    # children should override
    def content_type(self) -> str:
        # self not used
        return NMObject.__name__

    def content_type_ok(
        self, 
        nmobject: NMObject | None
    ) -> bool:
        if isinstance(nmobject, NMObject):
            return nmobject.__class__.__name__ == self.content_type()
        return False
        # return isinstance(nmobject, NMObject)
        # want same type NOT same instance

    @property
    def content_parameters(self) -> list[dict]:
        plist = []
        for o in self.__map.values():
            plist.append(o.parameters)
        return plist

    # MutableMapping Required Abstract Methods:
    # __getitem__, __setitem__, __delitem__, __iter__, __len__

    # MutableMapping required abstract method
    def __getitem__(
        self,
        key: str,
    ) -> NMObject | None:
        """
        called by:
            get(), setdefault(), items(), values(), pop(), popitem(), clear()
        override below: setdefault(), pop(), popitem(), clear()
        """
        # print('__getitem__ ' + str(key))
        actual_key = self._getkey(key)
        if actual_key is None:
            raise KeyError(key)
        return self.__map[actual_key]

    # MutableMapping required abstract method
    def __setitem__(
        self,
        key: str,
        nmobject: NMObject  # key is equal to NMObject name
    ) -> None:
        """
        called by '=' and update()
        override below: update()
        e.g. mymap['recorda0'] = mynmobject
        """
        # print('__setitem__, ' + str(key) + ', ' + str(nmobject))
        self.update({key: nmobject})  # key is not used

    # MutableMapping required abstract method
    def __delitem__(
        self,
        key: str,
    ) -> None:
        """
        called by: 'del', pop(), popitem(), clear()
        override below: pop(), popitem(), clear()
        # e.g. del self.__map[key]
        """
        # print('__delitem__ ' + str(key))
        actual_key = self._getkey(key)
        if actual_key is None:
            raise KeyError(key)
        self.pop(actual_key)  # allows delete confirmation

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

    # override MutableMapping mixin method
    def __contains__(
        self,
        key: object
    ) -> bool:
        # called by 'in' operator
        # supports both string keys (case-insensitive) and NMObject values (by identity)
        if isinstance(key, str):
            actual_key = self._getkey(key)
            return actual_key is not None
        if isinstance(key, NMObject):
            return self.contains_value(key)
        return False

    def contains_value(
        self,
        nmobject: NMObject
    ) -> bool:
        # Check if NMObject is in container
        if isinstance(nmobject, NMObject):
            for o in self.__map.values():
                if nmobject is o:
                    return True
        return False

    # keys() NO OVERRIDE
    # items() NO OVERRIDE
    # values() NO OVERRIDE
    # get() NO OVERRIDE

    # override MutableMapping mixin method
    # children should override if new parameters are declared
    def __eq__(
        self,
        other: object,
    ) -> bool:
        # called by '==' and '!=' operators
        # print("__eq__" + other.name)
        # first compare self.__map
        ignore_parameters = []
        if not isinstance(other, NMObjectContainer):
            return NotImplemented
        if not super().__eq__(other):
            return False
        self_keys = self.__map.keys()
        if not nmu.keys_are_equal(list(self_keys), list(other.keys())):
            return False
        for k in self_keys:
            s = self.__getitem__(k)
            if s is None:
                return False
            o = other.__getitem__(k)
            if o is None:
                return False
            if s != o:
                return False
        if "rename_on" not in ignore_parameters:
            if self.__rename_on != other.__rename_on:
                return False
        if "auto_name_prefix" not in ignore_parameters:
            self_prefix = self.__auto_name_prefix.lower()
            other_prefix = other.__auto_name_prefix.lower()
            if self_prefix != other_prefix:
                return False
        if "auto_name_seq_format" not in ignore_parameters:
            self_format = self.__auto_name_seq_format
            other_format = other.__auto_name_seq_format
            if self_format != other_format:
                return False
        if "selected_name" not in ignore_parameters:
            if self.selected_name != other.selected_name:
                return False
            if self.selected_value != other.selected_value:
                return False
        if "execute_mode" not in ignore_parameters:
            if self.execute_mode != other.execute_mode:
                return False
            if self.execute_target_name != other.execute_target_name:
                return False
            if len(self.execute_targets) != len(other.execute_targets):
                return False
            for i in range(len(self.execute_targets)):
                if self.execute_targets[i] != other.execute_targets[i]:
                    return False
        if "sets" not in ignore_parameters:
            if self.sets != other.sets:
                return False
        return True

    # __ne__() NO OVERRIDE
    # '!=' operator

    def __deepcopy__(self, memo: dict) -> NMObjectContainer:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMObjectContainer by bypassing __init__ and
        directly setting attributes. Handles special cases:
        - Deep copies all contained NMObjects in __map
        - Creates new NMSets with function reference to new container

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMObjectContainer
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
        result._container = None
        result._NMObject__copy_of = self

        # Now handle NMObjectContainer's special attributes

        # __map: deep copy all contained NMObjects
        result._NMObjectContainer__map = {}
        for key, nmobj in self._NMObjectContainer__map.items():
            copied_obj = copy.deepcopy(nmobj, memo)
            copied_obj._parent = result  # update parent to new container
            copied_obj._container = result  # link back to new container
            result._NMObjectContainer__map[key] = copied_obj

        # __sets: deep copy the sets and update resolve function
        result._NMObjectContainer__sets = copy.deepcopy(
            self._NMObjectContainer__sets, memo
        )
        result._NMObjectContainer__sets._resolve_fxn = result._get_map

        return result

    # MutableMapping Mixin Methods: pop, popitem, clear, update, setdefault

    # Sentinel value to distinguish "no default provided" from "default is None"
    _POPDEFAULT = object()

    # override MutableMapping mixin method
    def pop(  # type: ignore[override]
        self,
        key: str,
        default: object = _POPDEFAULT,
        quiet: bool = nmp.QUIET,
    ) -> NMObject | None:
        # Check if key exists, raise KeyError if not and no default provided
        actual_key = self._getkey(key)
        if actual_key is None:
            if default is self._POPDEFAULT:
                raise KeyError(key)
            return default  # type: ignore
        if isinstance(self.selected_name, str):
            if self.selected_name.lower() == actual_key.lower():
                self.selected_name = None
        self.sets.remove_from_all(actual_key)
        o = self.__map.pop(actual_key)
        o._container = None
        nmh.history("removed '%s'" % actual_key, path=self.path_str, quiet=quiet)
        return o

    # override MutableMapping mixin method
    # type: ignore[override]
    # # delete last item
    def popitem(self) -> tuple[str, NMObject] | tuple[()]:
        """
        Must override, otherwise first item is deleted rather than last.
        Returns empty tuple () if cancelled or failed to pop.
        Raises KeyError if the mapping is empty (per Python contract).
        Consider deprecating to prevent accidental deletes.
        """
        if len(self.__map) == 0:
            raise KeyError("popitem(): mapping is empty")
        klist = list(self.__map.keys())
        key = klist[-1]  # last key
        o = self.pop(key=key, default=None)
        if o:
            return (key, o)
        return ()  # Return empty tuple if cancelled

    # override MutableMapping mixin method
    # override so there is only a single delete confirmation
    def clear(self, quiet: bool = nmp.QUIET) -> None:
        if len(self) == 0:
            return
        names = list(self.__map.keys())
        for o in self.__map.values():
            o._container = None
        self.selected_name = None
        self.sets.empty_all()
        self.__map.clear()
        nmh.history("cleared all: %s" % names, path=self.path_str, quiet=quiet)

    # override MutableMapping mixin method
    # add/update NMObject to map
    def update(  # type: ignore[override]
        self,
        nmobjects: NMObject | list[NMObject] | dict[str, NMObject] | NMObjectContainer | None = None
    ) -> None:
        olist: list[NMObject]
        if nmobjects is None:
            nmobjects = []
        if isinstance(nmobjects, NMObject) and self.content_type_ok(nmobjects):
            olist = [nmobjects]
        elif isinstance(nmobjects, list):
            olist = nmobjects
        elif isinstance(nmobjects, NMObjectContainer):
            olist = list(nmobjects.__map.values())
        elif isinstance(nmobjects, dict):
            olist = []
            for k, o in nmobjects.items():
                # key is not used, key is NMObject name
                if not self.content_type_ok(o):
                    e = "nmobjects: '%s' value" % k
                    e = nmu.type_error_str(o, e, self.content_type())
                    raise TypeError(e)
                if k is None:
                    k = o.name
                if not isinstance(k, str):
                    e = nmu.type_error_str(k, "nmobjects: key", "string")
                    raise TypeError(e)
                if k.lower() != o.name.lower():
                    raise KeyError("key and name mismatch: '%s' != '%s'" % (k, o.name))
                olist.append(o)
        else:
            e = (
                self.content_type()
                + " or list or dictionary or "
                + self.__class__.__name__
            )
            e = nmu.type_error_str(nmobjects, "nmobjects", e)
            raise TypeError(e)
        update = False
        for o in olist:
            if self.content_type_ok(o):
                key = self._getkey(o.name)
                if key is None:
                    key = self._newkey(o.name)
                self.__map[key] = o
                update = True
            else:
                e = "nmobjects: list item"
                e = nmu.type_error_str(o, e, self.content_type())
                raise TypeError(e)
        if update:
            self.__update_nmobject_references()

    def __update_nmobject_references(self):
        for o in self.__map.values():
            o._container = self
            o._parent = self._parent

    # override MutableMapping mixin method
    # Sentinel value to distinguish "no default provided" from "default is None"
    _SETDEFAULT_NODEFAULT = object()

    def setdefault(
        self,
        key: str,
        default: object = _SETDEFAULT_NODEFAULT
    ) -> object:
        """
        Override to handle the case where no default is provided (raise KeyError).
        If default is provided, returns it without necessarily adding to dict.
        This function should arguably be called get_value_or_default().
        Consider deprecating to avoid confusion with standard dict behavior.
        """
        actual_key = self._getkey(key)
        if actual_key is not None:
            return self.__map[actual_key]
        # Key doesn't exist - raise KeyError if no default, else return default
        if default is self._SETDEFAULT_NODEFAULT:
            raise KeyError(key)
        # Try to set if it's a valid NMObject, otherwise just return the default
        if default is not None and self.content_type_ok(default):  # type: ignore
            self.__setitem__(key, default)  # type: ignore
        return default

    # NMObjectContainer methods:
    # _getkey, _newkey
    # rename, reorder, duplicate, new,
    # name_prefix (property), name_prefix_set, name_next,
    # select_key (property), select_value, select_item, is_selected

    def _getkey(
        self,
        key: str | None = None,
    ) -> str | None:
        """Returns the actual key from __map for a given name (case-insensitive lookup)."""
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
        if not newkey or not nmu.name_ok(newkey):
            raise ValueError("newkey: %s" % newkey)
        for k in self.__map.keys():
            if k.lower() == newkey.lower():  # keys are case insensitive
                raise KeyError("key name '%s' already exists" % newkey)
        return newkey

    def rename(
        self,
        name: str,
        newname: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> bool:
        """
        Cannot change map key names.
        """
        if not self.__rename_on:
            raise RuntimeError("key names are locked.")
        if name is None:
            e = nmu.type_error_str(name, "name", "string")
            raise TypeError(e)
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        actual_newname = self._newkey(newname)
        new_map = {}
        for k in self.__map.keys():
            o = self.__map[k]
            if k == key:
                o._name_set(newname=actual_newname, quiet=True)  # no history
            new_map[o.name] = o
        # self.__map = new_map  # reference change
        self.__map.clear()
        self.__map.update(new_map)
        self.__sets.rename_item(key, actual_newname)
        nmh.history(
            "renamed '%s' as '%s'" % (key, actual_newname),
            path=self.path_str, quiet=quiet,
        )
        return True

    def reorder(
        self,
        name_order: list[str],
        quiet: bool = nmp.QUIET,
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
        n_old = len(self)
        if n_new != n_old:
            raise KeyError("number of keys mismatch: '%s' != '%s'" % (n_new, n_old))
        if name_order == list(self.__map.keys()):
            return  # nothing to do
        new_map = {}
        for name in name_order:
            key = self._getkey(name)
            if key is None:
                raise KeyError("key name '%s' does not exist" % name)
            new_map[key] = self.__map[key]
        # self.__map = new_map  # reference change
        self.__map.clear()
        self.__map.update(new_map)
        nmh.history("reordered items", path=self.path_str, quiet=quiet)

    def duplicate(
        self,
        name: str,
        newname: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> NMObject | None:
        if name is None:
            e = nmu.type_error_str(name, "name", "string")
            raise TypeError(e)
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        o = self.__getitem__(key)
        if o is None:
            return None
        c = o.copy()
        newkey = self._newkey(newname)
        # c.name = newkey  # double history
        c._name_set(newname=newkey, quiet=True)  # no history
        self.__map[c.name] = c
        self.__update_nmobject_references()
        nmh.history(
            "duplicated '%s' as '%s'" % (key, c.name),
            path=self.path_str, quiet=quiet,
        )
        return c

    # children should override
    # and call super()._new()
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMObject | None:
        actual_name = self._newkey(name)
        o = NMObject(parent=self, name=actual_name)
        if self._new(o, select=select):
            return o
        return None
    
    def _new(
        self,
        nmobject: NMObject,
        select: bool = False,
        quiet: bool = nmp.QUIET,
    ) -> bool:
        if not self.content_type_ok(nmobject):
            e = nmu.type_error_str(nmobject, "nmobject", self.content_type())
            raise TypeError(e)
        newkey = self._newkey(nmobject.name)
        if not isinstance(newkey, str) or len(newkey) == 0:
            return False
        self.__map[newkey] = nmobject
        self.__update_nmobject_references()
        if len(self.__map) == 1:
            select = True  # select first entry
        if isinstance(select, bool) and select:
            self.__selected_name = newkey
        nmh.history("new '%s'" % newkey, path=self.path_str, quiet=quiet)
        return True

    @property
    def auto_name_prefix(self) -> str:  # see name_next())
        return self.__auto_name_prefix

    @auto_name_prefix.setter
    def auto_name_prefix(
        self, 
        prefix: str | None = "NMObject",
    ) -> None:
        return self._auto_name_prefix_set(prefix)

    def _auto_name_prefix_set(
        self, 
        prefix: str | None = "NMObject", 
        quiet: bool = nmp.QUIET
    ) -> None:
        if prefix is None:
            prefix = ""
        if not isinstance(prefix, str):
            e = nmu.type_error_str(prefix, "prefix", "string")
            raise TypeError(e)
        if not nmu.name_ok(prefix, ok_names=[""]):
            # '' empty string OK for channel names
            raise ValueError("prefix: %s" % prefix)
        if prefix.lower() == self.__auto_name_prefix.lower():
            return  # nothing to do
        oldprefix = self.__auto_name_prefix
        self.__auto_name_prefix = prefix
        h = nmh.history_change_str("prefix", oldprefix, self.__auto_name_prefix)
        nmh.history(h, path=self.path_str, quiet=quiet)

    @property
    def auto_name_seq_format(self) -> str:
        return self.__auto_name_seq_format

    @auto_name_seq_format.setter
    def auto_name_seq_format(
        self,
        seq_format: str = "0",
    ) -> None:
        return self._auto_name_seq_format_set(seq_format)

    def _auto_name_seq_format_set(
        self, seq_format: str = "0",
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(seq_format, int) and seq_format == 0:
            seq_format = "0"
        elif not isinstance(seq_format, str):
            e = nmu.type_error_str(seq_format, "seq_format", "string")
            raise TypeError(e)
        slist = ""
        for char in seq_format:
            if char == "0":
                slist += "0"
            elif char.upper() == "A":
                slist += "A"
            else:
                raise ValueError("seq format item should be '0' or 'A'")
        if "0" in slist and "A" in slist:
            raise ValueError("encounted mixed seq format types '0' and 'A'")
        if self.__auto_name_seq_format == slist:
            return  # no change
        old_format = self.__auto_name_seq_format
        self.__auto_name_seq_format = slist
        self._auto_name_seq_counter_reset()
        h = nmh.history_change_str("seq_format", old_format, slist)
        nmh.history(h, path=self.path_str, quiet=quiet)

    @staticmethod
    def __auto_name_seq_char_list() -> list[str]:
        return [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z",
        ]

    def _auto_name_seq_next_str(self) -> str:
        seq_num = len(self)
        if "0" in self.__auto_name_seq_format:
            padding = len(self.__auto_name_seq_format)
            seq_str = "{:0" + str(padding) + "d}"
            return seq_str.format(seq_num)
        elif "A" in self.__auto_name_seq_format:
            slist = ""
            clist = NMObjectContainer.__auto_name_seq_char_list()
            num_items = len(clist)
            for i in range(len(self.__auto_name_seq_format)):
                if num_items > 0:
                    quotient = seq_num // num_items
                    remainder = seq_num % num_items
                    seq_str = clist[remainder]
                    num_items = quotient
                else:
                    seq_str = clist[0]
                    num_items = 0
                slist = seq_str + slist  # inverts sequence
            return slist
        else:
            return str(seq_num)

    def _auto_name_seq_counter(self) -> str:
        return self.__auto_name_seq_counter

    def _auto_name_seq_counter_reset(self) -> None:
        self.__auto_name_seq_counter = self.__auto_name_seq_format

    def _auto_name_seq_counter_increment(self) -> str:
        ilist = [str(x) for x in range(10)]
        clist = NMObjectContainer.__auto_name_seq_char_list()
        increment_next = True  # increment first place
        seq_next = ""
        for char in reversed(self.__auto_name_seq_counter):
            if char in ilist:
                slist = ilist  # '0', '1', '2'...
            elif char in clist:
                slist = clist  # 'A', 'B', 'C'
            else:
                raise RuntimeError("seq format char should be '0-9' or 'A-Z'")
            i = slist.index(char)
            j = i
            if increment_next:
                if i == len(slist) - 1:
                    j = 0
                    increment_next = True  # increment next place
                else:
                    j = i + 1
                    increment_next = False
            seq_next = slist[j] + seq_next  # inverts sequence
        if seq_next == self.__auto_name_seq_format:
            raise RuntimeError("name sequence reached upper limit")
        self.__auto_name_seq_counter = seq_next
        return seq_next

    def auto_name_next(
        self, 
        use_counter: bool = False, 
        trials: int = 100
    ) -> str:
        for i in range(trials):
            if use_counter:
                seq_str = self._auto_name_seq_counter()
            else:
                seq_str = self._auto_name_seq_next_str()
            name = self.__auto_name_prefix + seq_str
            if name not in self:
                return name
            if use_counter:
                self._auto_name_seq_counter_increment()
            else:
                use_counter = True
        raise RuntimeError("failed to find next name")

    @property
    def selected_value(self) -> NMObject | None:
        key = self._getkey(self.__selected_name)
        if key is None:
            return None
        return self.__map[key]

    @property
    def selected_name(self) -> str | None:
        return self.__selected_name

    @selected_name.setter
    def selected_name(
        self, 
        name: str | None
    ) -> None:
        self._selected_name_set(name)

    def _selected_name_set(
        self,
        name: str | None,
        quiet: bool = nmp.QUIET,
    ) -> None:
        old_selected = self.__selected_name
        if name is None or (isinstance(name, str) and
                            (name == "" or name.lower() == "none")):
            self.__selected_name = None
        elif not isinstance(name, str):
            e = nmu.type_error_str(name, "key", "string")
            raise TypeError(e)
        else:
            key = self._getkey(name)
            if key is None:
                raise KeyError("key name '%s' does not exist" % name)
            self.__selected_name = key
        if old_selected != self.__selected_name:
            h = nmh.history_change_str("selected", old_selected,
                                       self.__selected_name)
            nmh.history(h, path=self.path_str, quiet=quiet)

    def is_selected(
        self, 
        name: str | None
    ) -> bool:
        if name is None:
            return self.__selected_name is None
        if not isinstance(name, str):
            return False
        if name == "" or name.lower() == "none":
            return self.__selected_name is None
        if self.__selected_name is None:
            return False
        return name.lower() == self.__selected_name.lower()

    @property
    def execute_targets(self) -> list[NMObject]:
        """Returns the list of NMObjects to operate on based on execute_mode."""
        if self.__execute_mode == ExecuteMode.SELECTED:
            value = self.selected_value
            if value is None:
                return []
            return [value]
        if self.__execute_mode == ExecuteMode.ALL:
            return list(self.values())
        if self.__execute_mode == ExecuteMode.NAME:
            if self.__execute_target_name is None:
                return []
            key = self._getkey(self.__execute_target_name)
            if key is None:
                return []
            return [self.__map[key]]
        if self.__execute_mode == ExecuteMode.SET:
            if self.__execute_target_name is None:
                return []
            key = self.sets._getkey(self.__execute_target_name)
            if key is None:
                return []
            s = self.sets.get(key)
            if s is None:
                return []
            return list(s)  # s is already a list[NMObject] from NMSets.get()
        return []

    @property
    def execute_mode(self) -> ExecuteMode:
        return self.__execute_mode

    @execute_mode.setter
    def execute_mode(self, mode: ExecuteMode) -> None:
        self._execute_mode_set(mode)

    @property
    def execute_target_name(self) -> str | None:
        return self.__execute_target_name

    @execute_target_name.setter
    def execute_target_name(self, name: str | None) -> None:
        self._execute_mode_set(self.__execute_mode, name)

    @property
    def execute_target(self) -> str:
        """Returns string representation of current execute target."""
        if self.__execute_mode == ExecuteMode.SELECTED:
            return "selected"
        if self.__execute_mode == ExecuteMode.ALL:
            return "all"
        if self.__execute_target_name is not None:
            return self.__execute_target_name
        return "selected"

    @execute_target.setter
    def execute_target(self, target: str | None) -> None:
        """Set execute mode from a string value.

        Args:
            target: One of:
                - "select" or "selected": use the currently selected item
                - "all": use all items in the container
                - a specific name: use that named item
                - a set name: use all items in that set
        """
        if target is None:
            self._execute_mode_set(ExecuteMode.SELECTED)
            return
        if not isinstance(target, str):
            e = nmu.type_error_str(target, "target", "string")
            raise TypeError(e)
        target_lower = target.lower()
        if target_lower in ("select", "selected"):
            self._execute_mode_set(ExecuteMode.SELECTED)
        elif target_lower == "all":
            self._execute_mode_set(ExecuteMode.ALL)
        elif self._getkey(target) is not None:
            # It's a valid name in the container
            self._execute_mode_set(ExecuteMode.NAME, target)
        elif self.sets._getkey(target) is not None:
            # It's a valid set name
            self._execute_mode_set(ExecuteMode.SET, target)
        else:
            raise ValueError("unknown execute target: %s" % target)

    def _execute_mode_set(
        self,
        mode: ExecuteMode = ExecuteMode.SELECTED,
        target_name: str | None = None,
    ) -> None:
        if not isinstance(mode, ExecuteMode):
            e = nmu.type_error_str(mode, "mode", "ExecuteMode")
            raise TypeError(e)
        if mode in (ExecuteMode.SELECTED, ExecuteMode.ALL):
            self.__execute_mode = mode
            self.__execute_target_name = None
            return
        if target_name is None:
            # raise ValueError("target_name required for NAME or SET mode")
            self.__execute_mode = mode
            self.__execute_target_name = None
            return
        if not isinstance(target_name, str):
            e = nmu.type_error_str(target_name, "target_name", "string")
            raise TypeError(e)
        if mode == ExecuteMode.NAME:
            key = self._getkey(target_name)
            if key is None:
                raise KeyError("name '%s' does not exist" % target_name)
            self.__execute_mode = mode
            self.__execute_target_name = key
        elif mode == ExecuteMode.SET:
            key = self.sets._getkey(target_name)
            if key is None:
                raise KeyError("set '%s' does not exist" % target_name)
            self.__execute_mode = mode
            self.__execute_target_name = key

    def is_execute_target(
        self,
        target: str
    ) -> bool:
        """Check if the given target matches the current execute configuration."""
        if target is None:
            return False
        if not isinstance(target, str):
            return False
        if self.__execute_mode == ExecuteMode.SELECTED:
            key = self.__selected_name
            if key is None:
                return False
            return target.lower() == key.lower()
        if self.__execute_mode == ExecuteMode.ALL:
            if target.lower() == "all":
                return True
            return target in self
        if self.__execute_mode == ExecuteMode.NAME:
            if self.__execute_target_name is None:
                return False
            return target.lower() == self.__execute_target_name.lower()
        if self.__execute_mode == ExecuteMode.SET:
            if self.__execute_target_name is None:
                return False
            if target.lower() == self.__execute_target_name.lower():
                return True
            # Check if target is in the set
            s = self.sets.get(self.__execute_target_name)
            if s is not None:
                return target in s
        return False

    @property
    def sets(self) -> NMSets:
        return self.__sets


if __name__ == "__main__":
    """
    o0 = NMObject(parent=None, name='F0')
    o1 = NMObject(parent=None, name='F1')
    o2 = NMObject(parent=None, name='F2')
    o3 = NMObject(parent=None, name='F3')
    o4 = NMObject(parent=None, name='F4')
    # test = NMObjectDictionary([o0, o1, o2])
    test = NMObjectContainer(nmobjects=[o0, o1, o2])
    print(test.__repr__())
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
    """
