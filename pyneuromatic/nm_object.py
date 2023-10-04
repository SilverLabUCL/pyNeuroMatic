#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime
import inspect
import types
from typing import Dict, List, Union

import pyneuromatic.nm_preferences as nmp
import pyneuromatic.nm_utilities as nmu


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
        NMProject, NMScale, NMSets

    Attributes:
        __created (str): creation date of NMObject.
        __parent (NMObject or any object): parent of NMObject.
        __name (str): name of NMObject.
        __rename_fxnref: reference of function that renames NMObject,
            e.g. NMObject._name_set or NMObjectContainer.rename.
        __copy_of (NMObject): if NMObject is a copy of another NMObject, this
            attribute holds the reference of the other NMObject.
        __eq_list (List[str]): names of objects to test in equality; see
            function __eq__().
        __notes_on (bool):
        __notes (List[str]):

    Properties (@property):
        parameters
        _parent
        content
        content_tree
        _tp (tree path)
        name
        notes
        note
        notes_on
        _manager
        _project
        _folder

    Children of this class should override:
        copy()
        __eq__()
        parameters()
    """

    def __init__(
        self,
        parent: Union[object, None] = None,  # for creating NM class tree
        name: str = "NMObject0",  # name of this NMObject
        notes_on: bool = True,
        copy: Union[nmu.NMObjectType, None] = None,  # see copy()
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
        self.__modified = date_time  # NOT COPIED
        self.__parent = None
        self.__name = None
        self.__notes_on = False  # turn off during __init__
        self.__notes = []  # [{'date': 'date', 'note': 'note'}]
        self.__rename_fxnref = self._name_set  # NOT COPIED
        # fxn ref for name setter
        self.__copy_of = None
        # self._eq_list = ['parent', notes']
        self._eq_list = []

        if copy is None:
            pass
        elif isinstance(copy, NMObject):
            parent = copy._parent
            name = copy.name
            notes_on = copy.notes_on
            if NMObject.notes_ok(copy._NMObject__notes):
                for n in copy._NMObject__notes:
                    self.__notes.append(dict(n))  # append a copy
            self.__copy_of = copy
        else:
            e = nmu.typeerror(copy, "copy", "NMObject")
            raise TypeError(e)

        self.__parent = parent  # family tree 'parent' and 'child'
        # nothing to test, parent can be any object

        if not isinstance(name, str):
            e = nmu.typeerror(name, "name", "string")
            raise TypeError(e)

        self._name_set(newname=name, quiet=True)

        if isinstance(notes_on, bool):
            self.__notes_on = notes_on
        else:
            self.__notes_on = True

        return None

    # children should override
    def copy(self) -> nmu.NMObjectType:
        return NMObject(copy=self)

    # children should override
    def __eq__(
        self,
        other: nmu.NMObjectType,
    ) -> bool:
        # executed with '==' but not 'is'
        # can use 'is' to test if objects are the same

        # if not super().__eq__(other):  # not sure this is needed (object)
        #    return False
        if not isinstance(other, type(self)):
            return False
        if "parent" in self._eq_list:
            if not isinstance(other._parent, type(self.__parent)):
                return False
        if self.name.lower() != other.name.lower():  # case insensitive
            return False
        if "notes" in self._eq_list:
            if self.notes_on != other.notes_on:
                return False
            if len(self.notes) != len(other.notes):
                return False
            if not all(s == o for s, o in zip(self.notes, other.notes)):
                return False
        return True

    def lists_are_equal(
        nmobject_list1: List[nmu.NMObjectType], nmobject_list2: List[nmu.NMObjectType]
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
    def parameters(self) -> Dict[str, object]:
        p = {"name": self.__name}
        p.update({"created": self.__created})
        p.update({"modified": self.__modified})
        if isinstance(self.__copy_of, type(self)):
            p.update({"copy of": self.__copy_of.treepath()})
        else:
            p.update({"copy of": None})
        return p

    @property
    def _parent(self) -> object:
        return self.__parent

    @_parent.setter
    def _parent(self, parent: object) -> None:
        self.__parent = parent
        self.modified()
        return None

    # @property
    # def parameter_list(self) -> List[str]:
    #    return list(self.parameters.keys())

    @property
    def content(self) -> Dict[str, str]:
        cname = self.__class__.__name__.lower()
        return {cname: self.__name}

    @property
    def content_tree(self) -> Dict[str, str]:
        if isinstance(self.__parent, NMObject):
            k = {}
            k.update(self.__parent.content_tree)  # goes up NM class tree
            k.update(self.content)
            return k
        return self.content

    def treepath(  # NM class tree path
        self,
        names: bool = True,  # True: names, False: NMObjects
        list_format: bool = False
        # True: list of names or NMObjects, e.g. ['nm', 'project0', 'folder0']
        # False: concatenated names, e.g. 'nm.project0.folder0'
    ) -> Union[str, List[nmu.NMObjectType]]:
        """returns the NM tree path of this NMObject.

        The NM tree path can be a list of NMObject names or references.
        By default, the list of names is concatenated via '.'
        Example of list of names: ['nm', 'project0', 'folder0']
        Concatenated list of names: 'nm.project0.folder0'

        :param names: return NMObject names, otherwise return NMObject refs
        :type names: bool, optional
        :param list_format: return list of names, otherwise return '.' concat
        :type list_format: bool, optional
        :return: list of names or NMObjects
        :rtype: str, list[str], list[NMObject]
        """
        if not names:
            list_format = True
        # create treepath list
        if isinstance(self.__parent, NMObject):
            tplist = self.__parent.treepath(names=names, list_format=True)
            # goes up NM class tree
            if names:
                tplist.append(self.__name)
            else:
                tplist.append(self)
        else:
            if names:
                tplist = [self.__name]
            else:
                tplist = [self]
        if list_format:
            return tplist
        # concat list using '.' seperator
        if len(tplist) > 0:
            tpstr = ".".join(tplist)
        else:
            tpstr = self.__name
        return tpstr

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
        return self.__rename_fxnref(self.__name, newname)

    def _name_set(
        self,
        name_notused: Union[str, None] = None,
        # name_notused, dummy argument to be consistent with
        # NMObjectContainer.rename(key, newkey)
        newname: Union[str, None] = None,
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
        self.modified()
        self.note = "renamed to '%s'" % self.__name
        h = nmu.history_change("name", oldname, self.__name)
        self._history(h, tp=self.treepath(), quiet=quiet)
        return True

    def _rename_fxnref_set(self, rename_fxnref) -> None:  # fxn reference
        # rename fxn must have this format: fxn(oldname, newname)
        if not isinstance(rename_fxnref, types.MethodType):
            e = nmu.typeerror(rename_fxnref, "rename_fxnref", "MethodType")
            raise TypeError(e)
        # TODO: test if function has 2 arguments?
        self.__rename_fxnref = rename_fxnref
        return None

    def modified(self, date_time: Union[str, None] = None) -> str:
        if not isinstance(date_time, str):
            date_time = str(datetime.datetime.now())
        if isinstance(self.__parent, NMObject):
            self.__parent.modified(date_time=date_time)
            # goes up NM class tree
        self.__modified = date_time
        return date_time

    @property
    def notes(self) -> List[Dict]:
        return self.__notes

    def notes_print(self) -> None:
        note_seperator = "  "
        if isinstance(self.__notes, list):
            for n in self.__notes:
                keys = n.keys()
                if isinstance(n, dict) and "date" in keys and "note" in keys:
                    print(n["date"] + note_seperator + n["note"])
        return None

    @property
    def note(self) -> str:
        if isinstance(self.__notes, list) and len(self.__notes) > 0:
            return self.__notes[-1]
        return ""

    @note.setter
    def note(self, thenote: str) -> None:
        self._notes_append(thenote)
        return None

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
        self, confirm_answer: Union[str, None] = None  # to skip confirm prompt
    ) -> bool:
        if nmp.DELETE_CONFIRM:
            if confirm_answer in nmu.CONFIRM_LIST:
                ync = confirm_answer
            else:
                q = "are you sure you want to delete all notes for '%s'?" % self.__name
                ync = nmu.input_yesno(q, treepath=self.treepath())
            if ync.lower() == "y" or ync.lower() == "yes":
                pass
            else:
                print("cancel delete all notes")
                return False
        self.__notes = []
        return True

    @property
    def notes_on(self) -> bool:
        return self.__notes_on

    @notes_on.setter
    def notes_on(self, on: bool) -> None:
        if isinstance(on, bool):
            self.__notes_on = on
        else:
            self.__notes_on = True
        return None

    def notes_ok(notes: List[Dict]) -> bool:
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
    def _manager(self) -> nmu.NMManagerType:  # find NMManager of this NMObject
        return self._find_parent("NMManager")

    @property
    def _project(self) -> nmu.NMProjectType:  # find NMProject of this NMObject
        return self._find_parent("NMProject")

    @property
    def _folder(self) -> nmu.NMFolderType:  # find NMFolder of this NMObject
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
        tp: Union[str, None] = None,
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message,
            title="ALERT",
            tp=tp,
            frame=frame,
            red=True,
            quiet=self._quiet(quiet),
        )

    def _error(
        self,
        message: str,
        tp: Union[str, None] = None,
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message,
            title="ERROR",
            tp=tp,
            frame=frame,
            red=True,
            quiet=self._quiet(quiet),
        )

    def _history(
        self,
        message: str,
        tp: Union[str, None] = None,
        quiet: bool = False,
        frame: int = 2,
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(
            message, tp=tp, frame=frame, red=False, quiet=self._quiet(quiet)
        )

    def _type_error(
        self,
        obj_name: str,  # name of object that is of the wrong type
        type_expected: str,  # expected type of the object
        # TODO: can pass type
        tp: Union[str, None] = None,  # history treepath
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
        tp: Union[str, None] = None,  # history treepath
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
        m = self._manager
        if m and m.__class__.__name__ == "NMManager":
            return m._quiet(quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return quiet
