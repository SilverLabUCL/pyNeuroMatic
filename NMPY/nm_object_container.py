#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:51:13 2022

@author: jason
"""

import nm_preferences as nmp
import nm_utilities as nmu

from nm_object import NMObject


class NMObjectContainer(NMObject):
    """
    class collections.abc.MutableSequence
    class collections.abc.MutableMapping (implement this?)

    A list container for NMObject items (above), one of which is assigned to
    'select'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_next() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        FolderContainer, DataContainer, NoteContainer, DataSeriesContainer,
        ChannelContainer, DataSeriesSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    def __init__(self, parent, name, type_='NMObject',
                 prefix='NMObject', rename=True, **copy):
        super().__init__(parent, name)
        if not isinstance(type_, str):
            e = self._type_error('type_', 'string')
            raise TypeError(e)
        if not type_ or not nmu.name_ok(type_):
            e = self._value_error('type_')
            raise ValueError(e)
        self.__type = type_
        if prefix is None:
            prefix = ''
        elif not isinstance(prefix, str):
            e = self._type_error('prefix', 'string')
            raise TypeError(e)
        elif not self.name_ok(prefix):
            e = self._value_error('prefix')
            raise ValueError(e)
        self.__prefix = prefix
        self.__rename = nmu.bool_check(rename, True)
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject
        for k, v in copy.items():  # see copy() and thecontainer_copy()
            if k.lower() == 'c_type' and isinstance(v, str):
                self.__type = v
            if k.lower() == 'c_prefix' and isinstance(v, str):
                self.__prefix = v
            if k.lower() == 'c_rename' and isinstance(v, bool):
                self.__rename = v
            if k.lower() == 'c_thecontainer' and isinstance(v, dict):
                if 'thecontainer' in v.keys():
                    if isinstance(v['thecontainer'], list):
                        self.__thecontainer = v['thecontainer']
                        self._thecontainer_rename_fxnref_set()
                        if 'select' in v.keys():
                            if isinstance(v['select'], NMObject):
                                self.__select = v['select']
        self._param_list += ['type', 'prefix', 'rename', 'select']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'type': self.__type})
        k.update({'prefix': self.__prefix})
        k.update({'rename': self.__rename})
        if self.__select:
            k.update({'select': self.__select.name})
            # need name for isequivalent() to work
        else:
            k.update({'select': 'None'})
        return k

    # override, no super
    @property
    def _content_name(self):
        n = self.__class__.__name__.lower()  # e.g. 'foldercontainer'
        n = n.replace('container', 's')  # e.g. 'folders'
        n = n.replace('ss', 's')
        return n

    # override, no super
    @property
    def content(self):
        return {self._content_name: self.names}

    # override
    def _isequivalent(self, container, alert=False):
        if not super()._isequivalent(container, alert=alert):
            return False
        if container.count != self.count:
            if alert:
                a = ('unequivalent container count: ' + str(self.count) +
                     ' vs ' + str(container.count))
                self._alert(a)
            return False
        for i, s in enumerate(self.__thecontainer):
            o = container.getitem(index=i, quiet=True)
            if not s._isequivalent(o, alert=alert):
                return False
        return True

    # override, no super
    def copy(self):
        return NMObjectContainer(self._parent, self.name, type_=self.__type,
                                 prefix=self.prefix, rename=self.__rename,
                                 c_thecontainer=self._thecontainer_copy())

    def _thecontainer_copy(self):
        c = []
        if self.__select and self.__select.name:
            select_name = self.__select.name
        else:
            select_name = ''
        select = None
        for o in self.__thecontainer:
            if o and isinstance(o, NMObject):
                oc = o.copy()
                c.append(oc)
                if oc.name.lower() == select_name.lower():
                    select = oc
        return {'thecontainer': c, 'select': select}

    def _thecontainer_rename_fxnref_set(self):
        for o in self.__thecontainer:
            o._rename_fxnref_set(self.rename)

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        return self._prefix_set(prefix)

    def _prefix_set(self, prefix, quiet=nmp.QUIET):
        if not self.__rename:
            e = self._error(self.__type + ' items cannot be renamed')
            raise RuntimeError(e)
        if not isinstance(prefix, str):
            e = self._type_error('prefix', 'string')
            raise TypeError(e)
        if not self.name_ok(prefix):
            e = self._value_error('prefix')
            raise ValueError(e)
        old = self.__prefix
        self.__prefix = prefix
        self._modified()
        h = nmu.history_change('prefix', old, self.__prefix)
        self._history(h, quiet=quiet)
        return True

    @property
    def names(self):
        """Get list of names of NMObject items in container"""
        return [o.name for o in self.__thecontainer]

    @property
    def count(self):
        """Number of NMObject items stored in container"""
        return len(self.__thecontainer)

    def index(self, name):
        """Find item # of NMObject in container"""
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not name or not nmu.name_ok(name):
            return -1
        if name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return -1
        for i, o in enumerate(self.__thecontainer):
            if name.lower() == o.name.lower():
                return i
        return -1

    def exists(self, name):
        """Check if NMObject exists within container"""
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        return self.index(name) >= 0

    def getitem(self, name='', index=None, quiet=nmp.QUIET):
        """Get NMObject from Container"""
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name, ok='select'):
            e = self._value_error('name')
            raise ValueError(e)
        if index is None:
            pass  # ok
        elif not isinstance(index, int):
            e = self._type_error('index', 'integer')
            raise TypeError(e)
        elif index < 0 and index >= -1 * len(self.__thecontainer):
            return self.__thecontainer[index]
        elif index >= 0 and index < len(self.__thecontainer):
            return self.__thecontainer[index]
        else:
            raise IndexError('bad index: ' + str(index))
        if not name:
            return None
        if name.lower() == 'select':
            return self.__select
        for o in self.__thecontainer:
            if name.lower() == o.name.lower():
                return o
        e = self._exists_error(name)
        self._error(e, quiet=quiet)
        return None

    def getitems(self, names=[], indexes=[]):
        """Get a list of NMObjects from Container"""
        if isinstance(names, list) or isinstance(names, tuple):
            pass  # ok
        elif isinstance(names, str):
            names = [names]
        else:
            e = self._type_error('names', 'string')
            raise TypeError(e)
        if isinstance(indexes, list) or isinstance(indexes, tuple):
            pass  # ok
        elif isinstance(indexes, int):
            indexes = [indexes]
        else:
            e = self._type_error('indexes', 'integer')
            raise TypeError(e)
        olist = []
        all_ = False
        for name in names:
            if not isinstance(name, str):
                e = self._type_error('name', 'string')
                raise TypeError(e)
            if name.lower() == 'all':
                all_ = True
            elif not self.name_ok(name, ok='select'):
                e = self._value_error('name')
                raise ValueError(e)
            if name.lower() == 'select':
                if self.__select not in olist:
                    olist.append(self.__select)
            else:
                for o in self.__thecontainer:
                    if all_ and o not in olist:
                        olist.append(o)
                    elif name.lower() == o.name.lower() and o not in olist:
                        olist.append(o)
        if all_:
            return olist
        for index in indexes:
            if not isinstance(index, int):
                e = self._type_error('index', 'integer')
                raise TypeError(e)
            o = None
            if index < 0 and index >= -1 * len(self.__thecontainer):
                o = self.__thecontainer[index]
            elif index >= 0 and index < len(self.__thecontainer):
                o = self.__thecontainer[index]
            else:
                raise IndexError('index out of range:  ' + str(index))
            if o and o not in olist:
                olist.append(o)
        return olist

    def _exists_error(self, name):
        e = 'failed to find ' + nmu.quotes(name)
        e += '\n' + 'acceptable names: ' + str(self.names)
        return e

    @property
    def select(self):
        return self.__select

    @select.setter
    def select(self, name):
        return self._select_set(name)

    def _select_set(self, name, failure_alert=True, quiet=nmp.QUIET):
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name):
            e = self._value_error('name')
            raise ValueError(e)
        if not name:
            return None
        # if self.__select and name.lower() == self.__select.name.lower():
            # return self.__select  # already selected
        if self.exists(name):
            old = self.__select
            o = self.getitem(name)
            self.__select = o
            self._modified()
            h = NMObjectContainer.__select_history(old, self.__select)
            self._history(h, quiet=quiet)
            return o
        if failure_alert:
            q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                 'do you want to create a new ' + self.__type +
                 ' named ' + nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if yn.lower() == 'y':
                return self.new(name=name, select=True)
            self._history('cancel', quiet=quiet)
            return None
        e = self._exists_error(name)
        self._error(e, quiet=quiet)
        return None

    @staticmethod
    def __select_history(old, new):
        if new:
            n = new.name
        else:
            n = 'None'
        if old:
            return nmu.history_change('select', old.name, n)
        return 'selected ' + nmu.quotes(n)

    def new(self, name='default', nmobject=None, select=True, quiet=nmp.QUIET):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        select = nmu.bool_check(select, True)
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not name or not self.name_ok(name, ok='default'):
            e = self._value_error('name')
            raise ValueError(e)
        if name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        if self.exists(name):
            e = self._error(self.__type + ' ' + nmu.quotes(name) +
                            ' already exists')
            raise RuntimeError(e)
        if nmobject is None:
            o = NMObject(self._parent, name)
        elif isinstance(nmobject, NMObject):
            # child 'new' should pass nmobject
            if nmobject.__class__.__name__ == self.__type:
                o = nmobject
                o._NMObject__name = name  # rename nmobject
                o._parent = self._parent  # reset parent reference
            else:
                e = self._type_error('nmobject', self.__type)
                raise TypeError(e)
        else:
            e = self._type_error('nmobject', 'NMObject')
            raise TypeError(e)
        if not o:
            return None
        self.__thecontainer.append(o)
        self._thecontainer_rename_fxnref_set()
        if select or not self.__select:
            old = self.__select
            self.__select = o
            if old:
                h = 'created ' + nmu.quotes(name)
                self._history(h, quiet=quiet)
                h = NMObjectContainer.__select_history(old, self.__select)
                self._history(h, quiet=quiet)
            else:
                h = 'created/selected ' + nmu.quotes(name)
                self._history(h, quiet=quiet)
        else:
            h = 'created ' + nmu.quotes(name)
            self._history(h, quiet=quiet)
        self._modified()
        return o

    def rename(self, name, newname, quiet=nmp.QUIET):
        if not self.__rename:
            e = self._error(self.__type + ' items cannot be renamed')
            raise RuntimeError(e)
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name, ok='select'):
            e = self._value_error('name')
            raise ValueError(e)
        if not name:
            return ''
        if not self.exists(name):
            e = self._exists_error(name)
            self._error(e, quiet=quiet)
            return ''
        o = self.getitem(name, quiet=quiet)
        if not o:
            return ''
        if not isinstance(newname, str):
            e = self._type_error('newname', 'string')
            raise TypeError(e)
        if not newname or not self.name_ok(newname, ok='default'):
            e = self._value_error('newname')
            raise ValueError(e)
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = self._error(self.__type + ' ' + nmu.quotes(newname) +
                            ' already exists')
            raise RuntimeError(e)
        old = nmu.quotes(o.name)
        # o.name = newname  # double history
        o._NMObject__name = newname  # mangled
        self._modified()
        new = nmu.quotes(o.name)
        h = 'renamed ' + old + ' to ' + new
        self._history(h, quiet=quiet)
        return newname

    def duplicate(self, name, newname, select=True, quiet=nmp.QUIET):
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        select = nmu.bool_check(select, True)
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name, ok='select'):
            e = self._value_error('name')
            raise ValueError(e)
        if not name:
            return None
        if not self.exists(name):
            e = self._exists_error(name)
            self._error(e, quiet=quiet)
            return None
        o = self.getitem(name, quiet=quiet)
        if not o:
            return None
        if not isinstance(newname, str):
            e = self._type_error('newname', 'string')
            raise TypeError(e)
        if not newname or not self.name_ok(newname, ok='default'):
            e = self._value_error('newname')
            raise ValueError(e)
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = self._error(self.__type + ' ' + nmu.quotes(newname) +
                            ' already exists')
            raise RuntimeError(e)
        c = o.copy()
        if not c:
            return None
        c._NMObject__name = newname
        # c._parent = self._parent  # reset parent reference
        self.__thecontainer.append(c)
        self._thecontainer_rename_fxnref_set()
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self._history(h, quiet=quiet)
        if select or not self.__select:
            old = self.__select
            self.__select = c
            h = NMObjectContainer.__select_history(old, self.__select)
            self._history(h, quiet=quiet)
        self._modified()
        return c

    def kill(self, names=[], indexes=[], confirm=True, quiet=nmp.QUIET):
        olist = self.getitems(names=names, indexes=indexes)
        if len(olist) == 0:
            return []
        if nmu.bool_check(confirm, True):
            nlist = []
            for o in olist:
                nlist.append(o.name)
            q = ('are you sure you want to kill the following items?' + '\n' +
                 ', '.join(nlist))
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                self._history('cancel', quiet=quiet)
                return []
        klist = []
        nlist = []
        for o in olist:
            klist.append(o)
            self.__thecontainer.remove(o)
            nlist.append(o.name)
            if self.__select is o:
                select_old = self.__select
                self.__select = None
        h = 'killed ' + ', '.join(nlist)
        self._history(h, quiet=quiet)
        if self.__select is None and len(self.__thecontainer) > 0:
            self.__select = self.__thecontainer[0]
            h = NMObjectContainer.__select_history(select_old, self.__select)
            self._history(h, quiet=quiet)
        self._modified()
        return klist

    def name_next(self, first=0, quiet=nmp.QUIET):
        """Get next default NMObject name based on prefix and sequence #."""
        if not isinstance(first, int):
            e = self._type_error('first', 'integer')
            raise TypeError(e)
        quiet = nmu.bool_check(quiet, nmp.QUIET)
        if not self.__prefix or first < 0:
            e = self._error('cannot generate default names')
            raise RuntimeError(e)
        i = self.name_next_seq(prefix=self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=nmp.QUIET):
        """Get next seq num of default NMObject name based on prefix."""
        if not isinstance(first, int):
            e = self._type_error('first', 'integer')
            raise TypeError(e)
        quiet = nmu.bool_check(quiet, nmp.QUIET)
        if not isinstance(prefix, str):
            e = self._type_error('prefix', 'string')
            raise TypeError(e)
        if not self.name_ok(prefix, ok='default'):
            e = self._value_error('prefix')
            raise ValueError(e)
        if prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            e = self._error('cannot generate default names')
            raise RuntimeError(e)
        elist = []
        for o in self.__thecontainer:
            name = o.name.lower()
            istr = name.replace(prefix.lower(), '')
            if str.isdigit(istr):
                elist.append(int(istr))
        if len(elist) == 0:
            return 0
        i = max(elist)
        return i + 1
