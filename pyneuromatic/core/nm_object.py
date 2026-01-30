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

import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMObject(object):
    """
    Foundation NeuroMatic Object that allows creation of the NM class tree.
    Most NMObjects reside in a container (see NMObjectContainer).

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

    Each NMObject has a treepath (tp), e.g. ['project0', 'folder0', 'recordA0']
    or 'project0.folder0.recordA0'

    Known children of NMObject:
        NMChannel, NMData, NMDataSeries, NMEpoch, NMFolder, NMObjectContainer,
        NMProject, NMDimension, NMSets

    Attributes:
        __created (str): creation date of NMObject.
        __parent (NMObject or any object): parent of NMObject.
        __name (str): name of NMObject.
        __rename_fxnref: reference of function that renames NMObject,
            e.g. NMObject._name_set or NMObjectContainer.rename.
        __copy_of (NMObject): if NMObject is a copy of another NMObject, this
            attribute holds the reference of the other NMObject.
        __notes_on (bool):
        __notes (List[str]):

    Properties (@property):
        parameters
        _parent
        content
        content_tree
        name
        notes
        note
        notes_on
        _manager
        _project
        _folder

    Children of this class should override:
        __deepcopy__()
        __eq__()
        parameters()
    """

    def __init__(
        self,
        parent: object | None = None,  # for creating NM class tree
        name: str = "NMObject0",  # name of this NMObject
        notes_on: bool = True,
        copy: NMObject | None = None,  # see copy()
    ) -> None:
        """Initialise a NMObject.

        :param parent: parent of this NMObject (see NM class tree)
        :type parent: object, optional
        :param name: name of this NMObject
        :type name: str, optional
        :param notes_on: turn notes on/off for this NMObect
        :type notes_on: bool, optional
        :param copy: pass a NMObect here to create a copy of it (see copy())
        :type copy: NMObject, optional
        :return: None
        :rtype: None
        """

        date_time = str(datetime.datetime.now())
        self.__created = date_time  # NOT COPIED
        self.__parent: object | None = None
        self.__name: str = "NMObject0"
        self.__notes_on: bool = False  # turn off during __init__
        self.__notes: list[dict] = []  # [{'date': 'date', 'note': 'note'}]
        self.__rename_fxnref = self._name_set  # NOT COPIED
        # fxn ref for name setter
        self.__copy_of: NMObject | None = None
        # self._eq_list = ['parent', notes']
        # self._eq_list: list[str] = []

        # Determine actual values to use
        actual_parent = parent
        actual_name = name
        actual_notes_on = notes_on

        if copy is None:
            pass
        elif isinstance(copy, NMObject):
            # When copying, use values from the copy object
            actual_parent = copy._parent
            actual_name = copy.name
            actual_notes_on = copy.notes_on
            if NMObject.notes_ok(copy.notes):
                for n in copy.notes:
                    self.__notes.append(dict(n))  # append a copy
            self.__copy_of = copy
        else:
            e = nmu.typeerror(copy, "copy", "NMObject")
            raise TypeError(e)

        self.__parent = actual_parent  # family tree 'parent' and 'child'
        # nothing to test, parent can be any object

        if not isinstance(actual_name, str):
            e = nmu.typeerror(actual_name, "name", "string")
            raise TypeError(e)

        self._name_set(newname=actual_name, quiet=True)

        if isinstance(actual_notes_on, bool):
            self.__notes_on = actual_notes_on
        else:
            self.__notes_on = True

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

        For NMObject, shallow copy delegates to deep copy since
        we need to copy mutable attributes like notes.

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
        attributes, calling super().__deepcopy__(memo) to copy NMObject
        attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMObject
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # NMObject attributes that need special handling (name-mangled)
        nmobject_special_attrs = {
            "_NMObject__created",
            "_NMObject__parent",
            "_NMObject__name",
            "_NMObject__notes_on",
            "_NMObject__notes",
            "_NMObject__rename_fxnref",
            "_NMObject__copy_of",
        }

        # First, deep copy all attributes that aren't NMObject's special attrs
        for attr, value in self.__dict__.items():
            if attr not in nmobject_special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Now set NMObject's attributes with custom handling
        # __created: NOT copied, gets new timestamp
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        # __parent: copied (maintains reference to same parent)
        result._NMObject__parent = self._NMObject__parent
        # __name: copied
        result._NMObject__name = self._NMObject__name
        # __notes_on: copied
        result._NMObject__notes_on = self._NMObject__notes_on
        # __notes: deep copied (mutable list of dicts)
        result._NMObject__notes = copy.deepcopy(self._NMObject__notes, memo)
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

        # if not super().__eq__(other):  # not sure this is needed (object)
        #    return False
        ignore_parameters = ["parent", "notes"]
        if not isinstance(other, NMObject):
            return NotImplemented
        if "parent" not in ignore_parameters:
            if not isinstance(other._parent, type(self.__parent)):
                return False
        if self.name.lower() != other.name.lower():  # case insensitive
            return False
        if "notes" not in ignore_parameters:
            if self.notes_on != other.notes_on:
                return False
            if len(self.notes) != len(other.notes):
                return False
            if not all(s == o for s, o in zip(self.notes, other.notes)):
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
            p.update({"copy of": self.__copy_of._treepath_str()})
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

    def treepath(  # NM class tree path
        self,
        names: bool = True,  # True: names, False: NMObjects
    ) -> list[str] | list[NMObject]:
        """returns the NM tree path of this NMObject.

        The NM tree path can be a list of NMObject names or references.
        Example of list of names: ['nm', 'project0', 'folder0']

        :param names: return NMObject names, otherwise return NMObject refs
        :type names: bool, optional
        :return: list of names or NMObjects
        :rtype: list[str], list[NMObject]
        """
        if names:
            tplist_names: list[str] = []
            
            if isinstance(self.__parent, NMObject):
                parent_path = self.__parent.treepath(names=True)
                if isinstance(parent_path, list):
                    tplist_names.extend(parent_path)  # type: ignore[arg-type]
            
            tplist_names.append(self.__name)
        
            return tplist_names
                
        else:
            tplist_objects: list[NMObject] = []
            
            if isinstance(self.__parent, NMObject):
                parent_path = self.__parent.treepath(names=False)
                if isinstance(parent_path, list):
                    tplist_objects.extend(parent_path)  # type: ignore[arg-type]
            
            tplist_objects.append(self)

            return tplist_objects
        
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
            e = nmu.typeerror(newname, "newname", "string")
            raise TypeError(e)
        if not newname or not nmu.name_ok(newname):
            raise ValueError("newname: %s" % newname)
        oldname = self.__name
        self.__name = newname
        self.note = "renamed to '%s'" % self.__name
        h = nmu.history_change("name", oldname, self.__name)
        self._history(h, tp=self._treepath_str(), quiet=quiet)

    def _rename_fxnref_set(self, rename_fxnref) -> None:
        """Set the rename function reference for this NMObject.

        The rename function must have the following format:
            fxn(oldname, newname)
        See NMObject._name_set(name, newname)
        See NMObjectContainer.rename(key, newkey)
        """
        if not isinstance(rename_fxnref, types.MethodType):
            e = nmu.typeerror(rename_fxnref, "rename_fxnref", "MethodType")
            raise TypeError(e)
        # TODO: test if function has 2 arguments?
        self.__rename_fxnref = rename_fxnref

    @property
    def notes(self) -> list[dict]:
        return self.__notes

    def notes_print(self) -> None:
        note_seperator = "  "
        if isinstance(self.__notes, list):
            for n in self.__notes:
                keys = n.keys()
                if isinstance(n, dict) and "date" in keys and "note" in keys:
                    print(n["date"] + note_seperator + n["note"])

    @property
    def note(self) -> str:
        if isinstance(self.__notes, list) and len(self.__notes) > 0:
            last_note = self.__notes[-1]
            if isinstance(last_note, dict) and "note" in last_note:
                return last_note["note"]
        return ""

    @note.setter
    def note(self, thenote: str) -> None:
        self._notes_append(thenote)

    def _notes_append(self, thenote: str) -> bool:
        if not self.__notes_on:
            # self._alert('notes are off')
            return False
        if not isinstance(self.__notes, list):
            self.__notes = []
        if thenote is None:
            return False
        if not isinstance(thenote, str):
            thenote = str(thenote)
        n = {"date": str(datetime.datetime.now())}
        n.update({"note": thenote})
        self.__notes.append(n)
        return True

    def _notes_delete(
        self, auto_confirm: str | None = None  # to skip confirm prompt
    ) -> None:
        """Delete all notes for this object.
        
        :param auto_confirm: pre-answer to confirmation prompt to skip UI
        :type auto_confirm: str, optional
        :raises RuntimeError: If user cancels the deletion
        :return: None
        :rtype: None
        """
        if nmp.DELETE_CONFIRM:
            if auto_confirm in nmu.CONFIRM_YNC:
                ync = auto_confirm
            else:
                q = "are you sure you want to delete all notes for '%s'?" % self.__name
                ync = nmu.input_yesno(q, treepath=self._treepath_str())
            if not isinstance(ync, str) or (ync.lower() != "y" and ync.lower() != "yes"):
                print("cancel delete all notes")
                raise RuntimeError("User cancelled note deletion")
        self.__notes = []

    @property
    def notes_on(self) -> bool:
        return self.__notes_on

    @notes_on.setter
    def notes_on(self, on: bool) -> None:
        if isinstance(on, bool):
            self.__notes_on = on
        else:
            self.__notes_on = True

    @staticmethod
    def notes_ok(notes: list[dict]) -> bool:
        # test notes type format
        if not isinstance(notes, list):
            return False
        for n in notes:
            if not isinstance(n, dict):
                return False
            keys = n.keys()
            if len(keys) == 2 and "date" in keys and "note" in keys:
                pass  # ok
            else:
                return False
            for k, v in n.items():
                if not isinstance(v, str):
                    return False
        return True

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

    def _alert(
        self,
        message: str,
        tp: str = "NONE",  # history treepath
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message,
            title="ALERT",
            treepath=tp,
            frame=frame,
            red=True,
            quiet=self._quiet(quiet),
        )

    def _error(
        self,
        message: str,
        tp: str = "NONE",  # history treepath
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message,
            title="ERROR",
            treepath=tp,
            frame=frame,
            red=True,
            quiet=self._quiet(quiet),
        )

    def _history(
        self,
        message: str,
        tp: str = "NONE",  # history treepath
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message, treepath=tp, frame=frame, red=False, quiet=self._quiet(quiet)
        )

    def _type_error(
        self,
        obj_name: str,  # name of object that is of the wrong type
        type_expected: str,  # expected type of the object
        tp: str = "NONE",  # history treepath
        quiet: bool = False,  # history quiet
        frame: int = 2,
    ) -> str:
        raise RuntimeError("function NMObject._type_error " "has been deprecated")
        """
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        found_variable = False
        for var_name, var_val in callers_local_vars:  # loop thru dict_items
            if var_name == obj_name:
                obj_val = var_val
                found_variable = True
                break
        if (found_variable):
            t = str(type(obj_val))
            t = t.replace('<class ', '').replace('>', '').replace("'", "")
        else:
            t = 'NMObject_TypeError_FailedToFindVariableType'
        e = 'bad ' + obj_name + ': expected ' + type_expected + ' but got ' + t
        return nmu.history(e, title='ERROR', tp=tp,
                           frame=frame, red=True, quiet=self._quiet(quiet))
        """

    def _value_error(
        self,
        obj_name: str,  # name of function variable with bad value
        tp: str = "NONE",  # history treepath
        quiet: bool = False,  # history quiet
        frame: int = 2,
    ) -> str:
        raise RuntimeError("function NMObject._value_error " "has been deprecated")
        """
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        found_variable = False
        for var_name, var_val in callers_local_vars:  # loop thru dict_items
            if var_name == obj_name:
                obj_val = var_val
                found_variable = True
                break
        if (found_variable):
            if isinstance(obj_val, str):
                v = "'%s'" % obj_val
            else:
                v = str(obj_val)
        else:
            v = 'NMObject_TypeError_FailedToFindVariableValue'
        e = 'bad ' + obj_name + ': ' + v
        return nmu.history(e, title='ERROR', tp=tp,
                           frame=frame, red=True, quiet=self._quiet(quiet))
        """

    def _quiet(self, quiet: bool) -> bool:
        # m = self._manager
        # if m and m.__class__.__name__ == "NMManager":
        #    return m._quiet(quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return quiet

    def _treepath_str(self) -> str:
        # NM treepath list of names is concatenated via '.'
        # Concatenated list of names: 'nm.project0.folder0'
        tp = self.treepath(names=True)
        tp_strs = [item for item in tp if isinstance(item, str)]
        return ".".join(tp_strs) if tp_strs else self.__name