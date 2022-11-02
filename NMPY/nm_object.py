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
        parent,  # parent of NMObject
        name  # name of this NMObject
    ):
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
        self._param_list = ['name', 'date', 'modified']
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
    def parameters(self):  # child class should override
        # and add class parameters
        # used in isequivalent
        p = {'name': self.__name}
        p.update({'date': self.__date})
        p.update({'modified': self.__modified})
        return p  # dictionary {}

    def _parameters_key_test(
        self,
        quiet=False
    ):  # used in nm_test.py
        #  verify parameters match self._param_list
        pkeys = self.parameters.keys()
        for k in pkeys:
            if k not in self._param_list:
                e = 'missing param_list item ' + nmu.quotes(k)
                self._error(e, quiet=quiet)
                return False
        for k in self._param_list:
            if k not in pkeys:
                e = 'missing parameters key ' + nmu.quotes(k)
                self._error(e, quiet=quiet)
                return False
        return True
    #
    #  ancestry content functions
    #  content and content_tree are dictionaries {}
    #
    @property
    def _content_name(self):
        return self.__class__.__name__.lower()  # class name, lower case

    @property
    def content(self):
        return {self._content_name: self.__name}

    @property
    def content_tree(self):
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
    def _tp(self):  # shorthand, use with history()
        return self.treepath(for_history=True)

    def _tp_check(  # check treepath string (e.g. fxn arg, see below)
        self,
        tp_str,  # tp_str = 'self' to get self.treepath
        for_history=True
    ):
        if not isinstance(tp_str, str):
            return ''
        if tp_str.lower() == 'self':
            return self.treepath(for_history=for_history)
        return tp_str

    def treepath(
        self,
        for_history=False  # treepath is for NM history
    ):
        for_history = nmu.bool_check(for_history, False)
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
        names=True,
        # True: get list of NMObject names
        # False: get list of NMObjects
        skip=[]  # pass NMObject names to skip/exclude
    ):
        names = nmu.bool_check(names, True)
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
    def name_ok(  # check name is OK, see _bad_names
        self,
        name,  # name to test
        ok=[]  # list of OK names
    ):
        if not nmu.name_ok(name):
            return False
        if not isinstance(ok, list):
            ok = [ok]  # make sure ok is a list
        bad = [n.lower() for n in self._bad_names]  # bad names in lower case
        for ok_item in ok:
            if not isinstance(ok_item, str):
                e = self._type_error('ok_item', 'string')
                raise TypeError(e)
            if ok_item.lower() in bad:
                bad.remove(ok_item.lower())
        return name.lower() not in bad

    @property
    def _bad_names(self):  # names not allowed
        return ['select', 'default', 'all']  # use lower case

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, newname):
        # calls _name_set() or NMObjectContainer.rename()
        return self.__rename_fxnref(self.__name, newname)

    def _name_set(
        self,
        name_notused,
        # name_notused, dummy argument to be consistent with
        # NMObjectContainer.rename(name, newname)
        newname,
        quiet=nmp.QUIET
    ):
        if not isinstance(newname, str):
            e = self._type_error('newname', 'string')
            raise TypeError(e)
        if not newname or not self.name_ok(newname):
            e = self._value_error('newname')
            raise ValueError(e)
        oldname = self.__name
        self.__name = newname
        self._modified()  # modification date
        h = nmu.history_change('name', oldname, self.__name)
        self._history(h, quiet=quiet)
        return True

    def _rename_fxnref_set(
        self,
        rename_fxnref  # fxn reference
        # rename fxn must have this format: fxn(name, newname)
    ):
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
    def _manager(self):  # find reference to Manager of this NMObject
        if self._parent is None:
            return None
        if self._parent.__class__.__name__ == 'Manager':
            return self._parent
        if isinstance(self._parent, NMObject):
            return self._parent._manager  # goes up the ancestry tree
        return None

    def _isequivalent(  # compare this NMObject to another NMObject
        self,
        nmobject,  # the other NMObject
        alert=False  # write alert to NM history
    ):
        self_is_equiv = False  # make this an argument?
        nan_eq_nan = nmp.NAN_EQ_NAN  # make this an argument?
        oktobedifferent = ['date', 'modified']  # make this an argument?
        alert = nmu.bool_check(alert, False)
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

    def copy(self):
        return NMObject(self._parent, self.__name)

    def save(
        self,
        path='',
        quiet=nmp.QUIET
    ):
        # TODO
        e = self._error('save under construction')
        raise RuntimeError(e)

    def _modified(self):
        self.__modified = str(datetime.datetime.now())
        if self._parent and isinstance(self._parent, NMObject):
            self._parent._modified()  # goes up the ancestry tree

    def _alert(self, message, tp='self', quiet=False, frame=2):
        # wrapper, see nmu.history
        return nmu.history(message, title='ALERT', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _error(self, message, tp='self', quiet=False, frame=2):
        # wrapper, see nmu.history
        return nmu.history(message, title='ERROR', tp=self._tp_check(tp),
                           frame=frame, red=True, quiet=self._quiet(quiet))

    def _history(self, message, tp='self', quiet=False, frame=2):
        # wrapper, see nmu.history
        return nmu.history(message, tp=self._tp_check(tp), frame=frame,
                           red=False, quiet=self._quiet(quiet))

    def _type_error(
        self,
        obj_name,  # name of object that is of the wrong type
        type_expected,  # expected type of the object
        tp='self',  # history treepath
        quiet=False,  # history quiet
        frame=2
    ):
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
        obj_name,  # name of object that has the wrong value
        tp='self',  # history treepath
        quiet=False,  # history quiet
        frame=2
    ):
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
        quiet
    ):
        m = self._manager
        if m.__class__.__name__ == 'Manager':
            return m._quiet(quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return nmu.bool_check(quiet, False)


class NMObjectTest(NMObject):

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.myvalue = 1
        self._param_list += ['myvalue']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'myvalue': self.myvalue})
        return k
