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
import datetime
from tkinter.font import names
import types
from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_folder import NMFolder
    from pyneuromatic.core.nm_manager import NMManager
    from pyneuromatic.core.nm_project import NMProject

import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMObject(object):
    """
    Foundation NeuroMatic Object that allows creation of the NM class tree.
    Most NMObjects reside in a container (see NMObjectContainer).

    NM class tree:

    NMManager (not an NMObject)
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

    Each NMObject has a path in the hierarchy:
        - path: list of names, e.g. ['root', 'folder0', 'recordA0']
        - path_str: dotted string, e.g. 'root.folder0.recordA0'
        - path_objects: list of NMObject references

    Known children of NMObject:
        NMChannel, NMData, NMDataSeries, NMEpoch, NMFolder, NMObjectContainer,
        NMProject, NMSets

    Attributes:
        __created (str): creation date of NMObject.
        __parent (NMObject or any object): parent of NMObject.
        __name (str): name of NMObject.
        __rename_fxnref: reference of function that renames NMObject,
            e.g. NMObject._name_set or NMObjectContainer.rename.
        __copy_of (NMObject): if NMObject is a copy of another NMObject, this
            attribute holds the reference of the other NMObject.

    Properties (@property):
        parameters
        _parent
        content
        content_tree
        path
        path_str
        path_objects
        name
        _manager
        _project
        _folder

    Children of this class should override:
        __deepcopy__()
        __eq__()
        parameters()
    """

    # Attributes that need special handling in __deepcopy__ (name-mangled).
    # Subclasses should extend this by unioning with their own special attrs:
    #   _DEEPCOPY_SPECIAL_ATTRS = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({...})
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = frozenset({
        "_NMObject__created",
        "_NMObject__parent",
        "_NMObject__name",
        "_NMObject__rename_fxnref",
        "_NMObject__copy_of",
    })

    def __init__(
        self,
        parent: object | None = None,  # for creating NM class tree
        name: str = "NMObject0",  # name of this NMObject
    ) -> None:
        """Initialise a NMObject.

        :param parent: parent of this NMObject (see NM class tree)
        :type parent: object, optional
        :param name: name of this NMObject
        :type name: str, optional
        :return: None
        :rtype: None
        """

        self.__created = datetime.datetime.now().isoformat(" ", "seconds")
        self.__parent: object | None = parent
        self.__name: str = "NMObject0"
        self.__rename_fxnref = self._name_set
        self.__copy_of: NMObject | None = None

        if not isinstance(name, str):
            e = nmu.type_error_str(name, "name", "string")
            raise TypeError(e)

        self._name_set(newname=name, quiet=True)

    # children should override __deepcopy__ instead of copy()
    def copy(self) -> NMObject:
        """Create a copy of this NMObject.

        Convenience method that calls copy.deepcopy(self).
        Subclasses should override __deepcopy__ to customize copy behavior.

        Returns:
            A deep copy of this NMObject
        """
        return copy.deepcopy(self)

    def __copy__(self) -> NMObject:
        """Support Python's copy.copy() protocol.

        For NMObject, shallow copy delegates to deep copy.

        Returns:
            A copy of this NMObject (same as deepcopy)
        """
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> NMObject:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMObject by bypassing __init__ and directly
        setting attributes. This avoids the complexity of the copy parameter
        in __init__ and provides a cleaner separation of concerns.

        Subclasses should override this method to handle their own special
        attributes. Use _DEEPCOPY_SPECIAL_ATTRS class attribute to define
        which attributes need special handling.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMObject
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (allows subclass extension)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # First, deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        # __created: NOT copied, gets new timestamp
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        # __parent: copied (maintains reference to same parent)
        result._NMObject__parent = self._NMObject__parent
        # __name: copied
        result._NMObject__name = self._NMObject__name
        # __rename_fxnref: NOT copied, set to new instance's _name_set
        result._NMObject__rename_fxnref = result._name_set
        # __copy_of: set to reference the original object
        result._NMObject__copy_of = self

        return result

    # children should override
    def __eq__(
        self,
        other: object,
    ) -> bool:
        # executed with '==' but not 'is'
        # can use 'is' to test if objects are the same
        if not isinstance(other, NMObject):
            return NotImplemented
        if self.name.lower() != other.name.lower():  # case insensitive
            return False
        return True

    @staticmethod
    def lists_are_equal(
        nmobject_list1: list[NMObject], nmobject_list2: list[NMObject]
    ) -> bool:
        """Compare lists of NMObjects.

        :param nmobject_list1: first list of NMObjects
        :type nmobject_list1: list[NMObject]
        :param nmobject_list2: second list of NMObjects
        :type nmobject_list2: list[NMObject]
        :return: true if lists of NMObjects are equal, otherwise false
        :rtype: bool
        """
        if nmobject_list1 is None:
            return nmobject_list2 is None
        elif nmobject_list2 is None:
            return False
        if not isinstance(nmobject_list1, list):
            return False
        if not isinstance(nmobject_list2, list):
            return False
        if len(nmobject_list1) != len(nmobject_list2):
            return False
        for s in nmobject_list1:
            if not isinstance(s, NMObject):
                return False
            found = False
            for o in nmobject_list2:
                if not isinstance(o, NMObject):
                    return False
                if s.name.lower() == o.name.lower():
                    if s != o:
                        return False
                    found = True
                    break
            if not found:
                return False
        return True

    # children should override, call super() and add class parameters
    # similar to __dict__
    @property
    def parameters(self) -> dict[str, object]:
        p: dict[str, object] = {"name": self.__name} # Tell mypy the correct type
        p.update({"created": self.__created})
        if isinstance(self.__copy_of, type(self)):
            p.update({"copy of": self.__copy_of.path_str})
        else:
            p.update({"copy of": None})
        return p

    @property
    def _parent(self) -> object:
        return self.__parent

    @_parent.setter
    def _parent(self, parent: object) -> None:
        self.__parent = parent

    # @property
    # def parameter_list(self) -> List[str]:
    #    return list(self.parameters.keys())

    @property
    def content(self) -> dict[str, str]:
        cname = self.__class__.__name__.lower()
        return {cname: self.__name}

    @property
    def content_tree(self) -> dict[str, str]:
        if isinstance(self.__parent, NMObject):
            k = {}
            k.update(self.__parent.content_tree)  # goes up NM class tree
            k.update(self.content)
            return k
        return self.content

    @property
    def path(self) -> list[str]:
        """Hierarchy path as list of names.

        Example: ['root', 'folder0', 'recordA0']

        Returns:
            List of NMObject names from root to this object.
        """
        result: list[str] = []
        if isinstance(self.__parent, NMObject):
            result.extend(self.__parent.path)
        result.append(self.__name)
        return result

    @property
    def path_str(self) -> str:
        """Hierarchy path as dotted string.

        Example: 'root.folder0.recordA0'

        Returns:
            Dotted string path from root to this object.
        """
        return ".".join(self.path) if self.path else self.__name

    @property
    def path_objects(self) -> list[NMObject]:
        """Hierarchy path as list of NMObject references.

        Example: [<NMProject>, <NMFolder>, <NMData>]

        Returns:
            List of NMObject references from root to this object.
        """
        result: list[NMObject] = []
        if isinstance(self.__parent, NMObject):
            result.extend(self.__parent.path_objects)
        result.append(self)
        return result
        
    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, newname: str) -> None:
        # Name setter is called via function reference self.__rename_fxnref
        # By default, self.__rename_fxnref points to
        # NMObject._name_set(name, newname) (see below)
        # Otherwise, it may point to
        # NMObjectContainer.rename(key, newkey)
        self.__rename_fxnref(self.__name, newname)

    def _name_set(
        self,
        name_notused: str | None = None,
        # name_notused, dummy argument to be consistent with
        # NMObjectContainer.rename(key, newkey)
        newname: str | None = None,
        # coding newname as optional (None)
        # since preceding param name_notused is optional
        quiet: bool = nmp.QUIET,
    ) -> None:
        """Set the name of the this NMObject.

        :param name_notused: name of this NMObject, but param is NOT USED
            since name is known.
        :type name_notused: str, optional
        :param newname: a new name for this NMObject
        :type newname: str
        :raises TypeError: If newname is not a string
        :raises ValueError: If newname is invalid
        :return: None
        :rtype: None
        """
        if not isinstance(newname, str):
            e = nmu.type_error_str(newname, "newname", "string")
            raise TypeError(e)
        if not newname or not nmu.name_ok(newname):
            raise ValueError("newname: %s" % newname)
        oldname = self.__name
        self.__name = newname
        h = nmh.history_change_str("name", oldname, self.__name)
        nmh.history(h, path=self.path_str, quiet=quiet)

    def _rename_fxnref_set(self, rename_fxnref) -> None:
        """Set the rename function reference for this NMObject.

        The rename function must have the following format:
            fxn(oldname, newname)
        See NMObject._name_set(name, newname)
        See NMObjectContainer.rename(key, newkey)
        """
        if not isinstance(rename_fxnref, types.MethodType):
            e = nmu.type_error_str(rename_fxnref, "rename_fxnref", "MethodType")
            raise TypeError(e)
        # TODO: test if function has 2 arguments?
        self.__rename_fxnref = rename_fxnref

    @property
    def _manager(self) -> NMManager | None:  # find NMManager of this NMObject
        return self._find_parent("NMManager")

    @property
    def _project(self) -> NMProject | None:  # find NMProject of this NMObject
        return self._find_parent("NMProject")

    @property
    def _folder(self) -> NMFolder | None:  # find NMFolder of this NMObject
        return self._find_parent("NMFolder")

    def _find_parent(self, classname: str) -> object:
        if self.__parent is None or not isinstance(classname, str):
            return None
        if self.__parent.__class__.__name__ == classname:
            return self.__parent
        if isinstance(self.__parent, NMObject):
            # go up the ancestry tree
            return self.__parent._find_parent(classname)
        return None

    def save(self, path: str = "", quiet: bool = nmp.QUIET):
        # TODO
        raise RuntimeError("save under construction")