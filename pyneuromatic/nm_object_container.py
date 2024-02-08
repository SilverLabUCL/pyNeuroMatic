# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:25:27 2022

@author: jason
"""
from collections.abc import MutableMapping
from typing import Dict, List, Tuple, Union

from pyneuromatic.nm_object import NMObject
import pyneuromatic.nm_preferences as nmp
from pyneuromatic.nm_sets import NMSets
import pyneuromatic.nm_utilities as nmu


class NMObjectContainer(NMObject, MutableMapping):
    """
    NM container (Mapping) of NMObjects. Allows creation of the NM class tree.

    NM class tree:

    NMManager
        NMProjectContainer
            NMProject (project0, project1...)
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
        NMEpochContainer, NMFolderContainer, NMProjectContainer

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

    def __init__(
        self,
        parent: Union[object, None] = None,  # for creating NM class tree
        name: str = "NMObjectContainer0",
        rename_on: bool = True,  # allow renaming of NMobjects
        name_prefix: str = "NMObject",
        # used for auto-name-generation of NMObjects
        name_seq_format: str = "0",
        # used for auto-name-generation of NMObjects
        # e.g. '0'  ->  0, 1, 2, 3...
        # e.g. '00' ->  00, 01, 02, 03...
        # e.g. 'A'  ->  A, B, C, D...
        # e.g. 'AA' ->  AA, AB, AC, AD...
        copy: Union[nmu.NMObjectContainerType, None] = None
        # see copy()
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            notes_on=False,  # turn notes off during __init__
            copy=copy,
        )  # NMObject

        self.__name_prefix = "NMObject"
        self.__name_seq_format = "0"
        self.__name_seq_counter = "0"

        # key of selected NMObject
        # 'current' project, folder, data, dataseries, channel, epoch...
        # only one value
        self.__select_key = None

        # key of selected NMObject or set
        # for function execution, e.g. computing stats
        # if 'select', then equals self.__select_key
        # if 'all', then all NMObjects are executed
        self.__execute_key = "select"

        nmobjects = []
        select_key = None
        execute_key = None
        sets_copy = None

        if copy is None:
            pass
        elif isinstance(copy, NMObjectContainer):
            rename_on = copy._NMObjectContainer__rename_on
            name_prefix = copy._NMObjectContainer__name_prefix
            name_seq_format = copy._NMObjectContainer__name_seq_format
            select_key = copy._NMObjectContainer__select_key
            execute_key = copy._NMObjectContainer__execute_key
            for v in copy.values():
                oc = v.copy()
                nmobjects.append(oc)
            sets_copy = copy._NMObjectContainer__sets
        else:
            e = nmu.typeerror(copy, "copy", "NMObjectContainer")
            raise TypeError(e)

        self._eq_list.append("rename_on")
        self._eq_list.append("name_prefix")
        self._eq_list.append("name_seq_format")
        self._eq_list.append("select")
        self._eq_list.append("execute")
        self._eq_list.append("sets")

        if not isinstance(rename_on, bool):
            e = nmu.typeerror(rename_on, "rename_on", "boolean")
            raise TypeError(e)

        self.__rename_on = rename_on

        self._name_prefix_set(name_prefix, quiet=True)
        # self.__name_prefix

        self._name_seq_format_set(name_seq_format, quiet=True)
        # self.__name_seq_format

        self.__map = {}  # where NMObjects are stored/mapped

        if len(nmobjects) > 0:
            self.update(nmobjects)  # add NMObjects to self.__map

        if select_key:
            self._select_key_set(select_key, quiet=True)
            # self.__select_key

        if execute_key:
            self._execute_key_set(execute_key, quiet=True)
            # self.__execute_key

        self.__sets = NMSets(
            parent=self,
            name="NMObjectContainerSets",
            nmobjects_fxnref=self._get_map,
            copy=sets_copy,
        )

        self.notes_on = True

        return None

    def _get_map(self) -> Dict[str, nmu.NMObjectType]:  # see __init__()
        return self.__map

    # override NMObject method
    # children should override
    def copy(self) -> nmu.NMObjectContainerType:
        return NMObjectContainer(copy=self)

    # override NMObject method
    # children should override if new parameters are declared
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({"content_type": self.content_type().lower()})
        k.update({"rename_on": self.__rename_on})
        k.update({"name_prefix": self.__name_prefix})
        k.update({"name_seq_format": self.__name_seq_format})
        k.update({"select": self.__select_key})
        k.update({"execute": self.__execute_key})
        k.update({"sets": list(self.__sets.keys())})
        return k

    # override NMObject method, no super
    @property
    def content(self) -> Dict[str, str]:
        cname = self.__class__.__name__.lower()
        return {cname: list(self.__map.keys())}

    # children should override
    def content_type(self) -> str:
        # self not used
        return NMObject.__name__

    def content_type_ok(self, nmobject: nmu.NMObjectType) -> bool:
        if isinstance(nmobject, NMObject):
            return nmobject.__class__.__name__ == self.content_type()
        return False
        # return isinstance(nmobject, NMObject)
        # want same type NOT same instance

    @property
    def content_parameters(self) -> List[Dict]:
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
    ) -> nmu.NMObjectType:
        """
        called by:
            get(), setdefault(), items(), values(), pop(), popitem(), clear()
        override below: setdefault(), pop(), popitem(), clear()
        """
        # print('__getitem__ ' + str(key))
        key = self._getkey(key)
        return self.__map[key]

    # MutableMapping required abstract method
    def __setitem__(
        self, key: str, nmobject: nmu.NMObjectType  # key is equal to NMObject name
    ) -> None:
        """
        called by '=' and update()
        override below: update()
        e.g. mymap['recorda0'] = mynmobject
        """
        # print('__setitem__, ' + str(key) + ', ' + str(nmobject))
        return self.update({key: nmobject})  # key is not used

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
        return self.pop(key)  # allows detele confirmation

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
    def __contains__(self, key_or_value: Union[str, nmu.NMObjectType]) -> bool:
        # called by 'in'
        # print('__contains__ ' + str(key))
        if isinstance(key_or_value, NMObject):
            for o in self.__map.values():
                if key_or_value is o:
                    return True
        elif isinstance(key_or_value, str):
            if key_or_value.lower() == "select":
                return False
            key = self._getkey(key_or_value, error1=False, error2=False)
            return key is not None
        return False

    # keys() NO OVERRIDE
    # items() NO OVERRIDE
    # values() NO OVERRIDE
    # get() NO OVERRIDE

    # override MutableMapping mixin method
    # children should override if new parameters are declared
    def __eq__(
        self,
        other: nmu.NMObjectContainerType,
    ) -> bool:
        # called by '==' and '!=' operators
        # print("__eq__" + other.name)
        # first compare self.__map
        if not super().__eq__(other):
            return False
        s_keys = self.__map.keys()
        if not nmu.keys_are_equal(s_keys, other.keys()):
            return False
        for k in s_keys:
            s = self.__getitem__(k)
            if s is None:
                return False
            o = other.__getitem__(k)
            if o is None:
                return False
            if s != o:
                return False
        if "rename_on" in self._eq_list:
            if self.__rename_on != other._NMObjectContainer__rename_on:
                return False
        if "name_prefix" in self._eq_list:
            s = self.__name_prefix.lower()
            o = other._NMObjectContainer__name_prefix.lower()
            if s != o:
                return False
        if "name_seq_format" in self._eq_list:
            s = self.__name_seq_format
            o = other._NMObjectContainer__name_seq_format
            if s != o:
                return False
        if "select" in self._eq_list:
            if self.select_key != other.select_key:
                return False
            if self.select_value != other.select_value:
                return False
        if "execute" in self._eq_list:
            if self.execute_key != other.execute_key:
                return False
            if len(self.execute_values) != len(other.execute_values):
                return False
            for i in range(len(self.execute_values)):
                if self.execute_values[i] != other.execute_values[i]:
                    return False
        if "sets" in self._eq_list:
            if self.sets != other.sets:
                return False
        return True

    # __ne__() NO OVERRIDE
    # '!=' operator

    # MutableMapping Mixin Methods: pop, popitem, clear, update, setdefault

    # override MutableMapping mixin method
    def pop(
        self,
        key: str,
        confirm_answer: Union[str, None] = None,  # to skip confirm prompt
    ) -> nmu.NMObjectType:
        # removed 'default' parameter
        key = self._getkey(key)
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_YNC:
                ync = confirm_answer
            else:
                prompt = "are you sure you want to delete '%s'?" % key
                ync = nmu.input_yesno(prompt, treepath=self.treepath())
            if ync.lower() == "y" or ync.lower() == "yes":
                pass
            else:
                print("cancel pop '%s'" % key)
                return None
        if isinstance(self.select_key, str):
            if self.select_key.lower() == key.lower():
                self.select_key = None
        self.sets.remove_from_all(key)
        o = self.__map.pop(key)
        return o

    # override MutableMapping mixin method
    def popitem(  # delete last item
        self, confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> Tuple[str, nmu.NMObjectType]:
        """
        Must override, otherwise first item is deleted rather than last.
        Consider deprecating to prevent accidental deletes.
        """
        if len(self.__map) == 0:
            return ()
        klist = list(self.__map.keys())
        key = klist[-1]  # last key
        o = self.pop(key=key, confirm_answer=confirm_answer)
        if o:
            return (key, o)
        return ()  # return tuple to be consistent with Python

    # override MutableMapping mixin method
    # override so there is only a single delete confirmation
    def clear(
        self, confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> None:
        if len(self) == 0:
            return None
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_YNC:
                ync = confirm_answer
            else:
                q = "are you sure you want to delete the following?\n" + ", ".join(
                    self.__map.keys()
                )
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == "y" or ync.lower() == "yes":
                pass
            else:
                print("cancel delete all")
                return None
        self.select_key = None
        self.sets.empty_all(confirm_answer="y")
        self.__map.clear()
        return None

    # override MutableMapping mixin method
    # add/update NMObject to map
    def update(
        self,
        nmobjects: Union[
            nmu.NMObjectType,
            List[nmu.NMObjectType],
            Dict[str, nmu.NMObjectType],
            nmu.NMObjectContainerType,
        ] = [],
    ) -> None:
        if self.content_type_ok(nmobjects):
            olist = [nmobjects]
        elif isinstance(nmobjects, list):
            olist = nmobjects
        elif isinstance(nmobjects, NMObjectContainer):
            olist = list(nmobjects._NMObjectContainer__map.values())
        elif isinstance(nmobjects, dict):
            olist = []
            for k, o in nmobjects.items():
                # key is not used, key is NMObject name
                if not self.content_type_ok(o):
                    e = "nmobjects: '%s' value" % k
                    e = nmu.typeerror(o, e, self.content_type())
                    raise TypeError(e)
                if k is None:
                    k = o.name
                if not isinstance(k, str):
                    e = nmu.typeerror(k, "nmobjects: key", "string")
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
            e = nmu.typeerror(nmobjects, "nmobjects", e)
            raise TypeError(e)
        update = False
        for o in olist:
            if self.content_type_ok(o):
                key = self._getkey(o.name, error2=False)
                if key is None:
                    key = self._newkey(o.name)
                self.__map[key] = o
                update = True
            else:
                e = "nmobjects: list item"
                e = nmu.typeerror(o, e, self.content_type())
                raise TypeError(e)
        if update:
            self.__update_nmobject_references()
        return None

    def __update_nmobject_references(self):
        for o in self.__map.values():
            o._rename_fxnref_set(self.rename)  # reference of 'rename' fxn
            o._parent = self._parent

    # override MutableMapping mixin method
    def setdefault(self, key, default=None):
        """
        Have to override to get this function to work.
        This functuion should be called get_value_or_default().
        Consider deprecating.
        """
        if default is None:
            return self.__getitem__(key)
        key = self._getkey(key, error1=False, error2=False)
        if key is None:
            return default
        return self.__map[key]

    # NMObjectContainer methods:
    # _key_check, _newkey_check
    # rename, reorder, duplicate, new,
    # name_prefix (property), name_prefix_set, name_next,
    # select_key (property), select_value, select_item, is_selected

    def _getkey(
        self,
        key: str,  # key or 'select' for selected key
        error1: bool = True,
        error2: bool = True,
    ) -> str:
        if not isinstance(key, str):
            if error1:
                e = nmu.typeerror(key, "key", "string")
                raise TypeError(e)
            return None
        if key.lower() == "select":
            if self.__select_key is None and error2:
                raise KeyError("select key is None")
            return self.__select_key
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
            e = nmu.typeerror(newkey, "newkey", "string")
            raise TypeError(e)
        if newkey.lower() == "default":
            return self.name_next()
        if not newkey or not nmu.name_ok(newkey):
            raise ValueError("newkey: %s" % newkey)
        for k in self.__map.keys():
            if k.lower() == newkey.lower():  # keys are case insensitive
                raise KeyError("key '%s' already exists" % newkey)
        return newkey

    def rename(self, key: str, newkey: str = "default") -> None:
        """
        Cannot change map key names.
        """
        if not self.__rename_on:
            raise RuntimeError("key names are locked.")
        key = self._getkey(key)
        newkey = self._newkey(newkey)
        new_map = {}
        for k in self.__map.keys():
            o = self.__map[k]
            if k == key:
                o._name_set(newname=newkey, quiet=True)  # no history
            new_map[o.name] = o
        # self.__map = new_map  # reference change
        self.__map.clear()
        self.__map.update(new_map)
        return None

    def reorder(self, newkeyorder: List[str]) -> None:
        """
        Cannot change map key names.
        TODO: order by name, creation date, modified date
        """
        if not isinstance(newkeyorder, list):
            e = nmu.typeerror(newkeyorder, "newkeyorder", "list")
            raise TypeError(e)
        for k in newkeyorder:
            if not isinstance(k, str):
                e = nmu.typeerror(k, "newkeyorder: list item", "string")
                raise TypeError(e)
            if k.lower() == "select":
                raise KeyError("key 'select' is invalid for this function")
        n_new = len(newkeyorder)
        n_old = len(self)
        if n_new != n_old:
            raise KeyError("number of keys mismatch: '%s' != '%s'" % (n_new, n_old))
        if newkeyorder == list(self.__map.keys()):
            return None  # nothing to do
        new_map = {}
        for k in newkeyorder:
            k = self._getkey(k)
            new_map[k] = self.__map[k]
        # self.__map = new_map  # reference change
        self.__map.clear()
        self.__map.update(new_map)
        return None

    def duplicate(
        self,
        key: str,
        newkey: str = "default",
    ) -> nmu.NMObjectType:
        key = self._getkey(key)
        newkey = self._newkey(newkey)
        o = self.__getitem__(key)
        c = o.copy()
        # c.name = newkey  # double history
        c._name_set(newname=newkey, quiet=True)  # no history
        self.__map[c.name] = c
        self.__update_nmobject_references()
        return c

    # children should override
    # and call super().new()
    def new(
        self,
        nmobject: nmu.NMObjectType = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMObjectType:
        if not self.content_type_ok(nmobject):
            e = nmu.typeerror(nmobject, "nmobject", self.content_type())
            raise TypeError(e)
        newkey = self._newkey(nmobject.name)
        self.__map[newkey] = nmobject
        self.__update_nmobject_references()
        if len(self.__map) == 1:
            select = True  # select first entry
        if isinstance(select, bool) and select:
            self.__select_key = newkey
        return nmobject

    @property
    def name_prefix(self) -> str:  # see name_next())
        return self.__name_prefix

    @name_prefix.setter
    def name_prefix(self, prefix: str = "default") -> None:
        return self._name_prefix_set(prefix)

    def _name_prefix_set(self, prefix: str, quiet: bool = nmp.QUIET) -> None:
        if not isinstance(prefix, str):
            e = nmu.typeerror(prefix, "prefix", "string")
            raise TypeError(e)
        if not nmu.name_ok(prefix, ok_names=[""]):
            # '' empty string OK for channel names
            raise ValueError("prefix: %s" % prefix)
        if prefix.lower() == self.__name_prefix.lower():
            return True  # nothing to do
        oldprefix = self.__name_prefix
        self.__name_prefix = prefix
        """
        if self.__name_prefix is None:
            h = 'prefix = None'
        else:
            h = "prefix = '%s'" % self.__prefix
        self.note = h
        h = nmu.history_change('prefix', oldprefix, self.__prefix)
        self._history(h, quiet=quiet)
        """
        return True

    @property
    def name_seq_format(self) -> str:
        return self.__name_seq_format

    @name_seq_format.setter
    def name_seq_format(
        self,
        seq_format: str = "0",
    ) -> None:
        return self._name_seq_format_set(seq_format)

    def _name_seq_format_set(
        self, seq_format: str = "0", quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(seq_format, int) and seq_format == 0:
            seq_format = "0"
        elif not isinstance(seq_format, str):
            e = nmu.typeerror(seq_format, "seq_format", "string")
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
        if self.__name_seq_format == slist:
            return None  # no change
        self.__name_seq_format = slist
        self._name_seq_counter_reset()
        return None

    def __name_seq_char_list() -> List[str]:
        return [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]

    def _name_seq_next_str(self) -> str:
        seq_num = len(self)
        if "0" in self.__name_seq_format:
            padding = len(self.__name_seq_format)
            seq_str = "{:0" + str(padding) + "d}"
            return seq_str.format(seq_num)
        elif "A" in self.__name_seq_format:
            slist = ""
            clist = NMObjectContainer.__name_seq_char_list()
            num_items = len(clist)
            for i in range(len(self.__name_seq_format)):
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

    def _name_seq_counter(self) -> str:
        return self.__name_seq_counter

    def _name_seq_counter_reset(self) -> None:
        self.__name_seq_counter = self.__name_seq_format
        return None

    def _name_seq_counter_increment(self) -> List[str]:
        ilist = [str(x) for x in range(10)]
        clist = NMObjectContainer.__name_seq_char_list()
        increment_next = True  # increment first place
        seq_next = ""
        for char in reversed(self.__name_seq_counter):
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
        if seq_next == self.__name_seq_format:
            raise RuntimeError("name sequence reached upper limit")
        self.__name_seq_counter = seq_next
        return seq_next

    def name_next(self, use_counter: bool = False, trials: int = 100) -> str:
        for i in range(trials):
            if use_counter:
                seq_str = self._name_seq_counter()
            else:
                seq_str = self._name_seq_next_str()
            name = self.__name_prefix + seq_str
            if name not in self.__map:
                return name
            if use_counter:
                self._name_seq_counter_increment()
            else:
                use_counter = True
        raise RuntimeError("failed to find next name")

    @property
    def select_value(self) -> nmu.NMObjectType:
        if self.__select_key is None:
            return None
        key = self._getkey(self.__select_key)
        return self.__map[key]

    @property
    def select_key(self) -> str:
        return self.__select_key

    @select_key.setter
    def select_key(self, key: str) -> None:
        self._select_key_set(key)
        return None

    def _select_key_set(self, key: str, quiet: bool = nmp.QUIET) -> str:
        if key is None or key == "":
            if self.__select_key is not None:
                self.__select_key = None
            return None
        if not isinstance(key, str):
            e = nmu.typeerror(key, "key", "string")
            raise TypeError(e)
        if key.lower() == "select":
            raise KeyError("invalid key 'select'")
        key = self._getkey(key)
        self.__select_key = key
        return key

    def is_select_key(self, key: str) -> bool:
        if key is None and self.__select_key is None:
            return True
        if not isinstance(key, str) or not isinstance(self.__select_key, str):
            return False
        if self.__select_key.lower() == "select":
            if key.lower() == "select":
                return True
            k = self._getkey("select", error2=False)
            return key.lower() == k.lower()
        return key.lower() == self.__select_key.lower()

    @property
    def execute_values(self) -> List[nmu.NMObjectType]:
        if not isinstance(self.__execute_key, str):
            return []
        if self.__execute_key.lower() == "select":
            key = self._getkey("select", error2=False)
            if key:
                o = self.__map[key]
                return [o]
            return []
        if self.__execute_key.lower() == "all":
            return list(self.values())
        key = self._getkey(self.__execute_key, error2=False)
        if key:
            o = self.__map[key]
            return [o]
        key = self.sets._getkey(self.__execute_key, error2=False)  # try sets
        if key:
            return self.sets.get(key)
        return []

    @property
    def execute_key(self) -> str:
        return self.__execute_key

    @execute_key.setter
    def execute_key(self, key: str) -> None:
        return self._execute_key_set(key)

    def _execute_key_set(self, key: str, quiet: bool = nmp.QUIET) -> str:
        if key is None or key == "":
            if self.__execute_key is not None:
                self.__execute_key = None
            return None
        elif not isinstance(key, str):
            e = nmu.typeerror(key, "key", "string")
            raise TypeError(e)
        if key.lower() == "select":
            if self.__execute_key.lower() != "select":
                self.__execute_key = "select"
            return "select"
        if key.lower() == "all":
            if self.__execute_key.lower() != "all":
                self.__execute_key = "all"
            return "all"
        k = self._getkey(key, error2=False)
        if k is None:
            k = self.sets._getkey(key)  # try sets, throw error
        self.__execute_key = k
        return k

    def is_execute_key(self, key: str) -> bool:
        if key is None:
            return self.__execute_key is None
        if not isinstance(key, str) or not isinstance(self.__execute_key, str):
            return False
        if self.__execute_key.lower() == "select":
            if key.lower() == "select":
                return True
            k = self._getkey("select", error2=False)
            if isinstance(k, str):
                return key.lower() == k.lower()
            return False
        if self.__execute_key.lower() == "all":
            if key.lower() == "all":
                return True
            return key in self
        return key.lower() == self.__execute_key.lower()

    @property
    def sets(self) -> nmu.NMSetsType:
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
