# -*- coding: utf-8 -*-
"""
Named sets of NMObjects within a container, with AND/OR set algebra.

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
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_object import NMObject

import pyneuromatic.core.nm_utilities as nmu


# Valid equation operators for set combinations
EQUATION_OPERATORS = ("and", "or")


class NMSets(MutableMapping):
    """Named groups of items within a container, with AND/OR set algebra.

    Internally stores string keys (NMObject names), resolving to NMObject
    references on-demand via a resolve function.

    Sets can be combined using equations with AND/OR operators:
        sets.define_and("set3", "set1", "set2")  # set3 = set1 AND set2
        sets.define_or("set3", "set1", "set2")   # set3 = set1 OR set2

    Equations are evaluated dynamically - changes to source sets are
    automatically reflected when getting the equation set.
    """

    def __init__(
        self,
        name: str = "NMSets0",
        nmobjects_fxnref: Callable | None = None,
        nmobjects: dict | None = None,
    ) -> None:
        self._name = name
        self._map: dict[str, list[str] | tuple[str, str, str]] = {}
        self._nmobjects: dict = {}
        self._resolve_fxn: Callable = self._nmobjects_default

        if nmobjects_fxnref is not None and nmobjects is not None:
            raise ValueError(
                "pass either 'nmobjects_fxnref' or 'nmobjects', not both"
            )
        if nmobjects_fxnref is not None:
            self._resolve_fxn = nmobjects_fxnref
        elif isinstance(nmobjects, dict):
            self._nmobjects = nmobjects  # reference, not copy
        else:
            raise ValueError(
                "expected argument for 'nmobjects_fxnref' or 'nmobjects'"
            )

    def _nmobjects_default(self) -> dict:
        return self._nmobjects

    @property
    def _nmobjects_dict(self) -> dict:
        return self._resolve_fxn()

    @property
    def name(self) -> str:
        return self._name

    def copy(self) -> NMSets:
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> NMSets:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._name = self._name
        result._map = copy.deepcopy(self._map, memo)
        result._nmobjects = {}
        result._resolve_fxn = result._nmobjects_default
        return result

    # MutableMapping abstract methods

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(
        self,
        key: str,
        olist: str | list[str] | tuple[str, str, str],
    ) -> None:
        self._setitem(key, olist, add=False)

    def __delitem__(self, key: str) -> None:
        self.pop(key)

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    # MutableMapping mixin overrides

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._getkey(key) is not None

    def get(
        self,
        key: str,
        default=None,
        get_equation: bool = False,
        get_keys: bool = False,
    ) -> Any:
        actual_key = self._getkey(key)
        if actual_key is None:
            return default

        value = self._map[actual_key]

        if NMSets.tuple_is_equation(value):
            if get_equation:
                return value

            assert isinstance(value, tuple)
            operator, set1_key, set2_key = value
            set1_items = self.get(set1_key, get_keys=True)
            set2_items = self.get(set2_key, get_keys=True)
            if not isinstance(set1_items, list) or not isinstance(set2_items, list):
                return []

            set1 = set(set1_items)
            set2 = set(set2_items)
            if operator == "and":
                result_set = set1 & set2
            elif operator == "or":
                result_set = set1 | set2
            else:
                return []

            klist = list(result_set)
            if klist:
                klist = self._match_keys(klist)

            if get_keys:
                return klist

            return self._resolve_keys(klist)

        # value is a list of string keys
        assert isinstance(value, list)
        if get_keys:
            return list(value)  # return a copy

        return self._resolve_keys(value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMSets):
            return NotImplemented
        if self._name != other._name:
            return False
        s_keys = list(self.keys())
        if not nmu.keys_are_equal(s_keys, list(other.keys())):
            return False
        for s_key in s_keys:
            s_value = self._map[s_key]
            if NMSets.tuple_is_equation(s_value):
                o_eq = other.get(s_key, get_equation=True)
                if not NMSets.tuple_is_equation(o_eq):
                    return False
                assert isinstance(s_value, tuple) and isinstance(o_eq, tuple)
                if s_value[0] != o_eq[0]:
                    return False
                if s_value[1].lower() != o_eq[1].lower():
                    return False
                if s_value[2].lower() != o_eq[2].lower():
                    return False
            else:
                o_keys = other.get(s_key, get_keys=True)
                if not isinstance(o_keys, list):
                    return False
                assert isinstance(s_value, list)
                if not nmu.keys_are_equal(s_value, o_keys):
                    return False
        return True

    def pop(self, key: str) -> Any:  # type: ignore[override]
        key = self._getkey(key)
        return self._map.pop(key)

    def popitem(self) -> tuple[str, Any] | None:
        if len(self._map) == 0:
            return None
        klist = list(self._map.keys())
        key = klist[-1]
        value = self.pop(key=key)
        if value is not None:
            return (key, value)
        return None

    def clear(self) -> None:
        self._map.clear()

    def setdefault(self, key, default=None):
        k = self._getkey(key)
        if k is None:
            if default is None:
                return None
            self.__setitem__(key, default)
            return default
        return self.get(key)

    # Key lookup helpers

    def _getkey(self, key: str | None = None) -> str | None:
        if key is None:
            return None
        if not isinstance(key, str):
            raise TypeError(nmu.type_error_str(key, "key", "string or None"))
        for k in self._map.keys():
            if k.lower() == key.lower():
                return k
        return None

    def _newkey(self, newkey: str | None = None) -> str:
        if newkey is None:
            return self.auto_name_next()
        if not isinstance(newkey, str):
            raise TypeError(nmu.type_error_str(newkey, "newkey", "string"))
        if newkey.strip() == "":
            return self.auto_name_next()
        if not nmu.name_ok(newkey, ok_names=["all"]):
            raise ValueError("newkey: %s" % newkey)
        for k in self._map.keys():
            if k.lower() == newkey.lower():
                raise KeyError("key '%s' already exists" % newkey)
        return newkey

    def auto_name_next(self, prefix: str = "set", trials: int = 100) -> str:
        for i in range(trials):
            newkey = prefix + str(i)
            if self._getkey(newkey) is None:
                return self._newkey(newkey)
        raise ValueError("failed to find next name for NMSets '%s'" % self._name)

    def _match_keys(self, keys: list[str]) -> list[str]:
        """Match keys to the canonical casing in _nmobjects_dict, preserving order."""
        nmobjects = self._nmobjects_dict
        matched = []
        remaining = list(keys)
        for canonical_key in nmobjects.keys():
            for k in keys:
                if canonical_key.lower() == k.lower():
                    if canonical_key not in matched:
                        matched.append(canonical_key)
                    if k in remaining:
                        remaining.remove(k)
        if remaining:
            raise KeyError(
                "nmobject_keys: the following keys do not exist: " + str(remaining)
            )
        return matched

    def _resolve_keys(self, keys: list[str]) -> list:
        """Resolve string keys to NMObject references."""
        nmobjects = self._nmobjects_dict
        result = []
        for k in keys:
            obj = nmobjects.get(k)
            if obj is not None and obj not in result:
                result.append(obj)
        return result

    def _to_key(self, item: Any) -> str:
        """Extract string key from a string or NMObject."""
        if isinstance(item, str):
            return item
        if hasattr(item, "name"):
            return item.name
        raise TypeError(
            nmu.type_error_str(item, "item", "string or NMObject")
        )

    # Core set operations

    def _setitem(
        self,
        key: str,
        olist: Any,
        add: bool = False,
    ) -> None:
        actual_key = self._getkey(key)
        if actual_key is None:
            actual_key = self._newkey(key)
            new = True
            old_keys: list[str] | None = None
            add = False
        else:
            new = False
            old_value = self._map[actual_key]
            if add:
                if NMSets.tuple_is_equation(old_value):
                    raise ValueError("cannot 'add' to an existing equation")
                assert isinstance(old_value, list)
                old_keys = list(old_value)
                if len(old_keys) == 0:
                    add = False
            else:
                old_keys = None

        # Handle equation tuples
        if NMSets.tuple_is_equation(olist):
            if not new:
                if add:
                    raise ValueError("cannot 'add' to an existing equation")
                old_value = self._map[actual_key]
                if not NMSets.tuple_is_equation(old_value):
                    raise ValueError("set '%s' exists and is not an equation" % key)
            assert isinstance(olist, tuple)
            operator, set1_name, set2_name = olist
            set1_key = self._getkey(set1_name)
            if set1_key is None:
                raise KeyError(f"equation set '{set1_name}' does not exist")
            set2_key = self._getkey(set2_name)
            if set2_key is None:
                raise KeyError(f"equation set '{set2_name}' does not exist")
            self._map[actual_key] = (operator, set1_key, set2_key)
            return

        # Normalize items to string keys
        items: list[str]
        if isinstance(olist, list):
            items = [self._to_key(o) for o in olist]
        elif isinstance(olist, str) or hasattr(olist, "name"):
            items = [self._to_key(olist)]
        else:
            raise TypeError(
                nmu.type_error_str(olist, "olist", "string or NMObject or list or equation tuple")
            )

        if add and old_keys:
            items = items + old_keys

        # Validate and match to canonical keys, dedup preserving order
        matched = self._match_keys(items)
        deduped: list[str] = []
        for k in matched:
            if k not in deduped:
                deduped.append(k)

        self._map[actual_key] = deduped

    def add(
        self,
        key: str,
        olist: Any = None,
    ) -> None:
        if olist is None:
            olist = []
        self._setitem(key, olist, add=True)

    def define_and(self, name: str, set1: str, set2: str) -> str:
        """Define an equation set as the AND (intersection) of two sets."""
        self._setitem(name, ("and", set1, set2), add=False)
        return self._getkey(name) or name

    def define_or(self, name: str, set1: str, set2: str) -> str:
        """Define an equation set as the OR (union) of two sets."""
        self._setitem(name, ("or", set1, set2), add=False)
        return self._getkey(name) or name

    def remove(
        self,
        key: str,
        olist: Any,
        error: bool = True,
    ) -> list:
        key = self._getkey(key)
        items: list[str]
        if isinstance(olist, str) or hasattr(olist, "name"):
            items = [self._to_key(olist)]
        elif isinstance(olist, list):
            items = [self._to_key(o) for o in olist]
        else:
            if error:
                raise TypeError(
                    nmu.type_error_str(olist, "olist", "string or NMObject or list")
                )
            return []

        value = self._map[key]
        if NMSets.tuple_is_equation(value):
            if error:
                raise ValueError(
                    "cannot remove items from '%s' since it is an equation" % key
                )
            return []

        assert isinstance(value, list)
        removed = []
        for item_key in items:
            found = False
            for stored_key in value:
                if item_key.lower() == stored_key.lower():
                    removed.append(stored_key)
                    found = True
                    break
            if not found:
                if error:
                    raise ValueError("set '%s' does not contain '%s'" % (key, item_key))
                return []
        for r in removed:
            value.remove(r)

        # Resolve removed keys to NMObjects for return value
        return self._resolve_keys(removed)

    def rename_item(self, old_name: str, new_name: str) -> None:
        """Rename an item across all sets (used when container renames an object)."""
        for key, value in self._map.items():
            if isinstance(value, list):
                self._map[key] = [
                    new_name if k.lower() == old_name.lower() else k
                    for k in value
                ]
            # Equations reference set names, not item names — no change needed

    def remove_from_all(
        self,
        olist: Any,
        error: bool = False,
    ) -> None:
        if isinstance(olist, str) or hasattr(olist, "name"):
            items = [self._to_key(olist)]
        elif isinstance(olist, list):
            items = [self._to_key(o) for o in olist]
        else:
            raise TypeError(
                nmu.type_error_str(olist, "olist", "string or NMObject or list")
            )
        for item_key in items:
            for map_key, map_val in self._map.items():
                if not NMSets.tuple_is_equation(map_val):
                    self.remove(map_key, item_key, error=error)

    def contains(self, key: str, olist: Any) -> bool:
        key = self._getkey(key)
        if key is None:
            return False
        if isinstance(olist, str) or hasattr(olist, "name"):
            items = [self._to_key(olist)]
        elif isinstance(olist, list):
            items = [self._to_key(o) for o in olist]
        else:
            return False
        klist = self.get(key, get_keys=True)
        if not isinstance(klist, list):
            return False
        for item_key in items:
            if not any(item_key.lower() == k.lower() for k in klist):
                return False
        return True

    def new(self, name: str | None = None) -> tuple[str, list]:
        key = self._newkey(name)
        self._map[key] = []
        return (key, [])

    def duplicate(
        self, name: str, newname: str | None = None
    ) -> tuple[str, Any]:
        """Duplicate a set with a new name."""
        if name is None:
            raise ValueError("key name cannot be None")
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        newkey = self._newkey(newname)
        value = self._map[key]
        if NMSets.tuple_is_equation(value):
            self._map[newkey] = value
            return (newkey, value)
        else:
            assert isinstance(value, list)
            self._map[newkey] = list(value)
            return (newkey, list(value))

    def rename(self, name: str, newname: str | None = None) -> str:
        key = self._getkey(name)
        newkey = self._newkey(newname)
        new_map: dict[str, Any] = {}
        for k, v in self._map.items():
            if k.lower() == key.lower():
                new_map[newkey] = v
            else:
                new_map[k] = v
        # Update equation tuples that reference the old set name
        for k, v in new_map.items():
            if NMSets.tuple_is_equation(v):
                assert isinstance(v, tuple)
                op, s1, s2 = v
                s1 = newkey if s1.lower() == key.lower() else s1
                s2 = newkey if s2.lower() == key.lower() else s2
                new_map[k] = (op, s1, s2)
        self._map.clear()
        self._map.update(new_map)
        return newkey

    def reorder(self, name_order: list[str]) -> None:
        if not isinstance(name_order, list):
            raise TypeError(nmu.type_error_str(name_order, "newkeyorder", "list"))
        if len(name_order) != len(self._map):
            raise KeyError(
                "number of keys mismatch: '%s' != '%s'"
                % (len(name_order), len(self._map))
            )
        new_map: dict[str, Any] = {}
        for n in name_order:
            actual = self._getkey(n)
            new_map[actual] = self._map[actual]
        self._map.clear()
        self._map.update(new_map)

    def empty(self, key: str) -> None:
        """Clear the contents of a set, converting equations to empty lists."""
        key = self._getkey(key)
        self._map[key] = []

    def empty_all(self) -> None:
        """Clear all sets, converting equations to empty lists."""
        for key in list(self._map.keys()):
            self.empty(key)

    def is_equation(self, key: str) -> bool:
        key = self._getkey(key)
        if key is None:
            return False
        return NMSets.tuple_is_equation(self._map[key])

    @staticmethod
    def tuple_is_equation(value: object) -> bool:
        """Check if value is a valid equation tuple: (operator, set1, set2)."""
        if not isinstance(value, tuple):
            return False
        if len(value) != 3:
            return False
        operator, set1, set2 = value
        if not isinstance(operator, str) or operator not in EQUATION_OPERATORS:
            return False
        if not isinstance(set1, str) or not isinstance(set2, str):
            return False
        return True
