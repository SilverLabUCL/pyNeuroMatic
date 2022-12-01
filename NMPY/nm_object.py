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
from typing import List, Dict, NewType

NMobject = NewType('NMObject', object)


class NMObject(object):
    """
    NeuroMatic object to be stored in a Container class (NMObjectContainer).

    Known children:
        Channel, Data, DataSeries, Dimension, DataSeriesSet, Folder, Note,
        Project

    Attributes:
        _parent (NMObject or any object):
        __name (str):
        __rename_fxnref (ref):
        __date (str):
        __modified (str):
        _param_list (list):
    """

    # tp: treepath of the NMObject
    # children of this class should override:
    #   parameters()
    #   _isequivalent()
    #   copy()

    def __init__(
        self,
        parent: object,  # parent of NMObject
        name: str  # name of this NMObject
    ) -> None:
        self._parent = parent
        if not isinstance(name, str):
            e = self._type_error('name', 'string', tp='')  # no tp yet
            raise TypeError(e)
        if not name or not self.name_ok(name):
            e = self._value_error('name', tp='')  # no tp yet
            raise ValueError(e)
        # private attributes
        self.__name = name
        self.__rename_fxnref = self._name_set  # fxn ref for name.setter
        self.__date = str(datetime.datetime.now())  # creation date
        self.__modified = self.__date  # creation date
        # param_list should match dictionary keys in parameters()
        # see param_test()
    #
    #  parameter functions
    #  parameters:
    #    name
    #    date: creation date
    #    modified: last modified
    #
    @property
    def parameters(self) -> Dict[str, str]:  # child class can override
        # and add class parameters
        # used in isequivalent
        p = {'name': self.__name}
        p.update({'date': self.__date})
        p.update({'modified': self.__modified})
        return p

    @property
    def parameter_list(self) -> List[str]:
        return list(self.parameters.keys())

    #
    #  ancestry content functions
    #  content and content_tree are dictionaries {}
    #
    @property
    def _content_name(self) -> str:
        return self.__class__.__name__.lower()  # class name, lower case

    @property
    def content(self) -> Dict[str, str]:
        return {self._content_name: self.__name}

    @property
    def content_tree(self) -> Dict[str, str]:
        if self._parent and isinstance(self._parent, NMObject):
            k = {}
            k.update(self._parent.content_tree)  # goes up the ancestry tree
            k.update(self.content)
            return k
        return self.content
    #
    #  ancestry treepath functions
    #  treepath: name0.name1.name2
    #  treepath_list: [object0,object1,object2] or [name0,name1,name2]
    #
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
        p = self._parent
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

    #
    #  name functions
    #
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
    def name(self, newname: str) -> bool:
        # calls _name_set() or NMObjectContainer.rename()
        return self.__rename_fxnref(self.__name, newname)

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
        self._modified()  # modification date
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
        return True
    #
    #  misc functions
    #
    @property
    def _manager(self) -> object:  # find reference to Manager of this NMObject
        if self._parent is None:
            return None
        if self._parent.__class__.__name__ == 'Manager':
            return self._parent
        if isinstance(self._parent, NMObject):
            return self._parent._manager  # goes up the ancestry tree
        return None

    def _isequivalent(  # compare this NMObject to another NMObject
        self,
        nmobject: NMobject,  # the other NMObject
        alert: bool = False  # write alert to NM history
    ) -> bool:
        self_is_equiv = False  # make this an argument?
        nan_eq_nan = nmp.NAN_EQ_NAN  # make this an argument?
        oktobedifferent = ['date', 'modified']  # make this an argument?
        ue = 'unequivalent '
        if nmobject == self:
            if self_is_equiv:
                return True
            if alert:
                a = 'equivalence with self is False'
                self._alert(a)
            return False
        scn = self.__class__.__name__
        ocn = nmobject.__class__.__name__
        if ocn != scn:
            if alert:
                a = ue + 'NMObject types: ' + scn + ' vs ' + ocn
                self._alert(a)
            return False
        # if nmobject._parent != self._parent:
            # problematic for copying containers
            # compare parent name?
            # a = (ue + 'parents: ' + str(self._parent) + ' vs ' +
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
        op = nmobject.parameters
        if op.keys() != sp.keys():
            if alert:
                a = (ue + 'parameter keys: ' + str(op.keys()) + ' vs ' +
                     str(sp.keys()))
                self._alert(a)
            return False
        for k in sp.keys():
            if k in oktobedifferent:
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

    def copy(self) -> NMobject:
        return NMObject(self._parent, self.__name)

    def save(
        self,
        path: str = '',
        quiet: bool = nmp.QUIET
    ):
        # TODO
        e = self._error('save under construction')
        raise RuntimeError(e)

    def _modified(self) -> str:
        self.__modified = str(datetime.datetime.now())
        if self._parent and isinstance(self._parent, NMObject):
            self._parent._modified()  # goes up the ancestry tree

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
        obj_name: str,  # name of object that has the wrong value
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
        if m.__class__.__name__ == 'Manager':
            return m._quiet(quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return quiet
