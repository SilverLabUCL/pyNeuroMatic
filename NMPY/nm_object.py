#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime
import inspect
import math
import types

import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List


class NMObject(object):
    """
    NeuroMatic object to be stored in a Container class (NMObjectContainer).

    Known children:
        Channel, Data, DataSeries, Dimension, DataSeriesSet, Folder, not,
        Project

    Attributes:
        _parent (NMObject or any object):
        __name (str):
        __rename_fxnref (ref):
        __created (str):
        __modified (str):
        __notes (List[Dict])
    """

    # tp: treepath of the NMObject
    # children of this class should override:
    #   parameters()
    #   _isequivalent()
    #   copy()

    def __init__(
        self,
        parent: object = None,  # parent of NMObject
        name: str = 'NMObject',  # name of this NMObject
        copy: nmu.NMObjectType = None  # see copy()
    ) -> None:

        self.__created = str(datetime.datetime.now())
        self.__modified = None

        if isinstance(copy, NMObject):
            self.__parent = copy._parent
            self.__name = copy._NMObject__name
            # self.__created = copy._NMObject__created
            # self.__modified = copy._NMObject__modified
            self.__notes_on = copy.notes_on
            self.__notes = []  # [{'date': 'date', 'note': 'note'}]
            self.__copy_of = copy
            if isinstance(copy.notes, list):
                for n in copy.notes:
                    if isinstance(n, dict):
                        self.__notes.append(dict(n))
        else:
            self.__parent = parent
            self.__name = name
            self.__notes_on = True
            self.__notes = []
            self.__copy_of = None

        self.__rename_fxnref = self._name_set  # fxn ref for name.setter

        if not isinstance(self.__name, str):
            e = self._type_error('name', 'string', tp='')  # no tp yet
            raise TypeError(e)
        if not self.__name or not self.name_ok(self.__name):
            e = self._value_error('name', tp='')  # no tp yet
            raise ValueError(e)

    def __eq__(
        self,
        other: nmu.NMObjectType
    ) -> bool:
        # executed with '==' but not 'is'
        # can use 'is' to test if objects are the same
        # if not super().__eq__(other):  # not sure this is needed (object)
        #    return False
        if not isinstance(other, type(self)):
            return False
        if self.name.lower() != other.name.lower():
            # names are case insensitive
            return False
        for a, b in zip(self.notes, other.notes):
            if a != b:
                return False
        return True

    def __ne__(
        self,
        other: nmu.NMObjectType
    ) -> bool:
        return not self.__eq__(other)

    # children need to override copy()
    def copy(self) -> nmu.NMObjectType:
        return NMObject(copy=self)

    def _isequivalent(  # compare this NMObject to another NMObject
        self,
        other: nmu.NMObjectType,  # the other NMObject
        alert: bool = False  # write alert to NM history
    ) -> bool:
        self_is_equiv = False  # make this an argument?
        nan_eq_nan = nmp.NAN_EQ_NAN  # make this an argument?
        # TODO: REPLACE WITH __EQ__
        ue = 'unequivalent '

        if other is self:
            if self_is_equiv:
                return True
            if alert:
                a = 'equivalence with self is False'
                self._alert(a)
            return False

        scn = self.__class__.__name__
        ocn = other.__class__.__name__
        if ocn != scn:
            if alert:
                a = ue + 'NMObject types: ' + scn + ' vs ' + ocn
                self._alert(a)
            return False
        # if nmobject._parent != self.__parent:
            # problematic for copying containers
            # compare parent name?
            # a = (ue + 'parents: ' + str(self.__parent) + ' vs ' +
            #      str(nmobject._parent))
            # self._alert(a)
            # return False
        # if nmobject._NMObject__rename_fxnref != self.__rename_fxnref:
        #     different, unless in same container
        #     a = (ue + 'rename() refs: ' + str(self.__rename_fxnref) +
        #     ' vs ' + str(nmobject._NMObject__rename_fxnref))
        #     self._alert(a)
        #     return False
        sp = self.parameters
        spkeys = sp.keys()
        op = other.parameters
        opkeys = op.keys()
        if opkeys != spkeys:
            if alert:
                a = (ue + 'parameter keys: ' + str(opkeys) +
                     ' vs ' + str(spkeys))
                self._alert(a)
            return False
        for k in spkeys:
            if k in self._oktobedifferent:
                continue
            if op[k] != sp[k]:
                if nan_eq_nan:
                    op_nan = isinstance(op[k], float) and math.isnan(op[k])
                    sp_nan = isinstance(sp[k], float) and math.isnan(sp[k])
                    if op_nan and sp_nan:
                        continue  # ok (nan=nan)
                if alert:
                    a = (ue + nmu.quotes(k) + ': ' +
                         nmu.quotes(sp[k]) + ' vs ' + nmu.quotes(op[k]))
                    self._alert(a)
                return False
        return True

    @property
    def _oktobedifferent(self) -> list:
        return ['created', 'modified', 'copy of']

    # TODO: convert to __repr__
    @property
    def parameters(self) -> Dict[str, object]:
        # child class can override and add class parameters
        # used in isequivalent
        p = {'name': self.__name}
        p.update({'created': self.__created})
        p.update({'modified': self.__modified})
        if isinstance(self.__copy_of, NMObject):
            p.update({'copy of': self.__copy_of._tp})
        else:
            p.update({'copy of': None})
        return p

    @property
    def _parent(self) -> object:
        return self.__parent

    # @_parent.setter  # discourage changes to parent
    # def _parent(self, parent: object) -> None:
    #     self.__parent = parent

    @property
    def parameter_list(self) -> List[str]:
        return list(self.parameters.keys())

    @property
    def content(self) -> Dict[str, str]:
        cname = self.__class__.__name__.lower()
        return {cname: self.__name}

    @property
    def content_tree(self) -> Dict[str, str]:
        if self.__parent and isinstance(self.__parent, NMObject):
            k = {}
            k.update(self.__parent.content_tree)  # goes up the ancestry tree
            k.update(self.content)
            return k
        return self.content

    #  ancestry treepath functions
    #  treepath: name0.name1.name2
    #  treepath_list: [object0,object1,object2] or [name0,name1,name2]
    @property
    def _tp(self) -> str:  # shorthand, use with history()
        return self.treepath(for_history=True)

    def _tp_check(  # check treepath string (e.g. fxn arg, see below)
        self,
        tp_str: str,  # tp_str = 'self' to get self.treepath
        for_history: bool = True
    ) -> str:
        if not isinstance(tp_str, str):
            return ''
        if tp_str.lower() == 'self':
            return self.treepath(for_history=for_history)
        return tp_str

    def treepath(
        self,
        for_history: bool = False  # treepath is for NM history
    ) -> str:
        if for_history:  # create treepath for history
            skip = nmp.HISTORY_TREEPATH_SKIP  # NM preferences, names to skip
        else:
            skip = []  # no names to skip
        plist = self.treepath_list(skip=skip)
        if len(plist) > 0:
            tp = '.'.join(plist)
        else:
            tp = self.__name
        return tp  # list of object names seperated by '.'

    def treepath_list(
        self,
        names: bool = True,
        # True: get list of NMObject names
        # False: get list of NMObjects
        skip: List[str] = []  # pass NMObject names to skip/exclude
    ) -> List[str]:
        if not isinstance(skip, list):
            skip = []
        cname = self.__class__.__name__
        if cname in skip:
            return []
        p = self.__parent
        if p and isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            tpl = p.treepath_list(names=names, skip=skip)
            # goes up the ancestry tree
            if names:
                tpl.append(self.__name)
            else:
                tpl.append(self)
            return tpl
        if names:
            return [self.__name]
        return [self]

    def name_ok(
        self,
        name: str,
        ok: List[str] = []
    ) -> bool:
        """ Check name is OK
            See _bad_names
        """
        if not isinstance(name, str):
            return False
        if isinstance(ok, str):
            ok = [ok]
        ok_names = [n.lower() for n in ok]  # lower case
        if name.lower() in ok_names:
            return True
        if not nmu.name_ok(name):  # check if alpha-numeric
            return False
        bad_names = [n.lower() for n in self._bad_names]  # lower case
        return name.lower() not in bad_names

    @property
    def _bad_names(self) -> List[str]:  # names that are not allowed
        return nmp.BAD_NAMES  # default

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, newname: str) -> None:
        # calls _name_set() or NMObjectContainer.rename()
        self.__rename_fxnref(self.__name, newname)

    def _name_set(
        self,
        name_notused: None,
        # name_notused, dummy argument to be consistent with
        # NMObjectContainer.rename(name, newname)
        newname: str,
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(newname, str):
            e = self._type_error('newname', 'string')
            raise TypeError(e)
        if not newname or not self.name_ok(newname):
            e = self._value_error('newname')
            raise ValueError(e)
        tp = self._tp
        oldname = self.__name
        self.__name = newname
        self.note = 'renamed to ' + nmu.quotes(self.__name)
        self._modified()
        h = nmu.history_change('name', oldname, self.__name)
        self._history(h, tp=tp, quiet=quiet)
        return True

    def _rename_fxnref_set(
        self,
        rename_fxnref  # fxn reference
        # rename fxn must have this format: fxn(oldname, newname)
    ) -> bool:
        if not isinstance(rename_fxnref, types.MethodType):
            e = self._type_error('rename_fxnref', 'MethodType')
            raise TypeError(e)
        # TODO: test if function has 2 arguments?
        self.__rename_fxnref = rename_fxnref
        self._modified()
        return True

    @property
    def notes(self) -> List[Dict]:
        return self.__notes

    def notes_print(self) -> None:
        note_seperator = '  '
        if isinstance(self.__notes, list):
            for n in self.__notes:
                keys = n.keys()
                if isinstance(n, dict) and 'date' in keys and 'note' in keys:
                    print(n['date'] + note_seperator + n['note'])

    @property
    def note(self) -> str:
        if isinstance(self.__notes, list) and len(self.__notes) > 0:
            return self.__notes[-1]
        return ''

    @note.setter
    def note(self, thenote: str) -> None:
        self._note_add(thenote)

    def _note_add(self, thenote: str) -> bool:
        if not isinstance(self.__notes, list):
            self.__notes = []
        if not self.__notes_on:
            a = 'notes are off'
            self._alert(a)
            return False
        if thenote is None:
            return False
        if not isinstance(thenote, str):
            thenote = str(thenote)
        n = {'date': str(datetime.datetime.now())}
        n.update({'note': thenote})
        self.__notes.append(n)
        return True

    def notes_clear(self, prompt: bool = True) -> bool:
        if prompt:
            q = ('are you sure you want to clear all notes for ' +
                 nmu.quotes(self.__name) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if yn.lower() != 'y':
                return False
        self.__notes = []
        return True

    @property
    def notes_on(self) -> bool:
        return self.__notes_on

    @notes_on.setter
    def notes_on(self, on: bool) -> None:
        if on is None:
            self.__notes_on = False
        elif isinstance(on, bool):
            self.__notes_on = on

    def notes_ok(notes: List[Dict]) -> bool:  # test notes type format
        if not isinstance(notes, list):
            return False
        for n in notes:
            if not isinstance(n, dict):
                return False
            keys = n.keys()
            if len(keys) == 2 and 'date' in keys and 'note' in keys:
                pass  # ok
            else:
                return False
            for k, v in n.items():
                if not isinstance(v, str):
                    return False
        return True

    @property
    def _manager(self) -> nmu.NMManagerType:  # find NMManager of this NMObject
        return self._find_ancestor('NMManager')

    @property
    def _project(self) -> nmu.NMProjectType:  # find NMProject of this NMObject
        return self._find_ancestor('NMProject')

    @property
    def _folder(self) -> nmu.NMFolderType:  # find NMFolder of this NMObject
        return self._find_ancestor('NMFolder')

    def _find_ancestor(self, classname: str) -> object:
        if self.__parent is None or not isinstance(classname, str):
            return None
        if self.__parent.__class__.__name__ == classname:
            return self.__parent
        if isinstance(self.__parent, NMObject):
            # go up the ancestry tree
            return self.__parent._find_ancestor(classname)
        return None

    def save(
        self,
        path: str = '',
        quiet: bool = nmp.QUIET
    ):
        # TODO
        e = self._error('save under construction')
        raise RuntimeError(e)

    def _modified(self) -> None:
        self.__modified = str(datetime.datetime.now())
        if self.__parent and isinstance(self.__parent, NMObject):
            self.__parent._modified()  # up the ancestry tree

    def _alert(
        self,
        message: str,
        tp: str = 'self',
        quiet: bool = False,
        frame: int = 2
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(message, title='ALERT', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _error(
        self,
        message: str,
        tp: str = 'self',
        quiet: bool = False,
        frame: int = 2
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(message, title='ERROR', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _history(
        self,
        message: str,
        tp: str = 'self',
        quiet: bool = False,
        frame: int = 2
    ) -> str:
        # wrapper, see nmu.history
        return nmu.history(message, tp=self._tp_check(tp), frame=frame,
                           red=False, quiet=self._quiet(quiet))

    def _type_error(
        self,
        obj_name: str,  # name of object that is of the wrong type
        type_expected: str,  # expected type of the object
        # TODO: can pass type
        tp: str = 'self',  # history treepath
        quiet: bool = False,  # history quiet
        frame: int = 2
    ) -> str:
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
        return nmu.history(e, title='ERROR', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _value_error(
        self,
        obj_name: str,  # name of function variable with bad value
        tp: str = 'self',  # history treepath
        quiet: bool = False,  # history quiet
        frame: int = 2
    ) -> str:
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        found_variable = False
        for var_name, var_val in callers_local_vars:  # loop thru dict_items
            if var_name == obj_name:
                obj_val = var_val
                found_variable = True
                break
        if (found_variable):
            if isinstance(obj_val, str):
                v = nmu.quotes(obj_val)
            else:
                v = str(obj_val)
        else:
            v = 'NMObject_TypeError_FailedToFindVariableValue'
        e = 'bad ' + obj_name + ': ' + v
        return nmu.history(e, title='ERROR', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _quiet(
        self,
        quiet: bool
    ) -> bool:
        m = self._manager
        if m and m.__class__.__name__ == 'NMManager':
            return m._quiet(quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return quiet
