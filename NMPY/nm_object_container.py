#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:51:13 2022

@author: jason
"""
from nm_object import NMObject
import nm_preferences as nmp
import nm_utilities as nmu
from nm_utilities import NMObjectType, NMObjectContainerType
from typing import Dict, List, Optional


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
        NMFolderContainer, NMDataContainer, NMDataSeriesContainer,
        NMChannelContainer, NMDataSeriesSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMObjectContainer',
        nmobject: NMObjectType,  # for typing
        prefix: str = 'NMObject',
        # auto-generation of NMObject names, see next_name()
        rename: bool = True,
        copy: NMObjectContainerType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        if isinstance(copy, NMObjectContainer):
            if isinstance(copy._nmobject, NMObject):
                self.__nmobject = copy._nmobject
            else:
                e = self._type_error('nmobject', 'NMObject', tp='')
                raise TypeError(e)
            self.__prefix = copy._NMObjectContainer__prefix
            self.__rename = copy._NMObjectContainer__rename
            self.__container = copy._container_copy()
            self.__container_update_references()
            if isinstance(copy._NMObjectContainer__select, NMObject):
                select_name = copy._NMObjectContainer__select.name
                self.__select = self.getitem(select_name)
            else:
                self.__select = None
        else:
            self.__nmobject = nmobject
            self.__prefix = prefix
            self.__rename = rename
            self.__select = None  # selected NMObject
            self.__container = []  # list of NMObjects

        if not isinstance(self.__nmobject, NMObject):
            e = self._type_error('nmobject', 'NMObject', tp='')
            raise TypeError(e)

        self.__nmobject._NMObject__parent = parent
        self.__nmobject._NMObject__name = 'ContainerUtility'

        if self.__prefix is None:
            self.__prefix = ''
        elif not isinstance(self.__prefix, str):
            e = self._type_error('prefix', 'string', tp='')
            raise TypeError(e)
        elif not self.name_ok(self.__prefix, ok=['', 'default']):
            e = self._value_error('prefix', tp='')
            raise ValueError(e)
        if self.__prefix.lower() == 'default':
            self.__prefix = 'NMObject'

    # override, no super
    def copy(self) -> NMObjectContainerType:
        return NMObjectContainer(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({'type': self.content_type})
        k.update({'prefix': self.__prefix})
        k.update({'rename': self.__rename})
        if self.__select:
            k.update({'select': self.__select.name})
            # need name for isequivalent() to work
        else:
            k.update({'select': 'None'})
        return k

    @property
    def content_parameters(self) -> List[Dict]:
        plist = []
        for o in self.__container:
            plist.append(o.parameters)
        return plist

    def print_content_parameters(self) -> None:
        for p in self.content_parameters:
            print(p)

    @property
    def content_type(self) -> str:
        return self.__nmobject.__class__.__name__

    # override, no super
    @property
    def content(self) -> Dict[str, str]:
        return {self.content_type + 's': self.names}

    # override
    def _isequivalent(
        self,
        container: NMObjectContainerType,
        alert: bool = False
    ) -> bool:
        if not super()._isequivalent(container, alert=alert):
            return False
        if container.count != self.count:
            if alert:
                a = ('unequivalent container count: ' + str(self.count) +
                     ' vs ' + str(container.count))
                self._alert(a)
            return False
        for i, s in enumerate(self.__container):
            o = container.getitem(index=i, quiet=True)
            if not s._isequivalent(o, alert=alert):
                return False
        return True

    @property
    def _nmobject(self) -> NMObjectType:
        return self.__nmobject

    def _container_copy(self) -> List:
        c = []
        for o in self.__container:
            if o and isinstance(o, NMObject):
                oc = o.copy()
                c.append(oc)
        return c

    def __container_append(self, nmobject):
        if not isinstance(nmobject, NMObject):
            e = self._type_error('nmobject', 'NMObject')
            raise TypeError(e)
        containertype = self.__nmobject.__class__.__name__
        if nmobject.__class__.__name__ != containertype:
            e = self._type_error('nmobject', containertype)
            raise TypeError(e)
        self.__container.append(nmobject)
        self.__container_update_references()
        return True

    def __container_update_references(self):
        for o in self.__container:
            o._rename_fxnref_set(self.rename)  # reference of 'rename' fxn
            o._NMObject__parent = self._parent

    @property
    def prefix(self) -> str:  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, newprefix: str) -> None:
        self._prefix_set(newprefix)

    def _prefix_set(
        self,
        newprefix: Optional[str] = None,
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not self.__rename:
            e = self._error(self.content_type + ' items cannot be renamed')
            raise RuntimeError(e)
        if newprefix is None:
            pass  # ok
        elif not isinstance(newprefix, str):
            e = self._type_error('newprefix', 'string')
            raise TypeError(e)
        elif not self.name_ok(newprefix, ok=''):
            e = self._value_error('newprefix')
            raise ValueError(e)
        elif newprefix == '':
            newprefix = None
        if newprefix == self.__prefix:
            return True  # nothing to do
        oldprefix = self.__prefix
        self.__prefix = newprefix
        if not self.__prefix:
            h = 'prefix = None'
        else:
            h = 'prefix = ' + nmu.quotes(self.__prefix)
        self.note = h
        self._modified()
        h = nmu.history_change('prefix', oldprefix, self.__prefix)
        self._history(h, quiet=quiet)
        return True

    @property
    def names(self) -> List[str]:
        """Get list of names of NMObject items in container"""
        return [o.name for o in self.__container]

    @property
    def count(self) -> int:
        """Number of NMObject items stored in container"""
        return len(self.__container)

    def index(
        self,
        name: str  # special: 'select'
    ) -> int:
        """Find item # of NMObject in container"""
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not name or not self.name_ok(name, ok='select'):
            return -1
        if name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return -1
        for i, o in enumerate(self.__container):
            if name.lower() == o.name.lower():
                return i
        return -1

    def exists(self, name: str) -> bool:
        """Check if NMObject exists within container"""
        return self.index(name) >= 0

    def getitem(
        self,
        name: Optional[str] = None,  # special: 'select'
        index: Optional[int] = None,
        quiet: bool = nmp.QUIET
    ) -> NMObjectType:
        """Get NMObject from Container"""
        if isinstance(index, int):
            if index < 0 and index >= -1 * len(self.__container):
                return self.__container[index]
            elif index >= 0 and index < len(self.__container):
                return self.__container[index]
            else:
                raise IndexError('bad index: ' + str(index))
        if isinstance(name, str):
            if name.lower() == 'select':
                return self.__select
            for o in self.__container:
                if name.lower() == o.name.lower():
                    return o
            e = self._exists_error(name)
            self._error(e, quiet=quiet)
        return None

    def getitems(
        self,
        names: Optional[List[str]] = [],  # special: 'select' or 'all'
        indexes: Optional[List[int]] = []
    ) -> List[NMObjectType]:
        """Get a list of NMObjects from Container"""
        if isinstance(names, list) or isinstance(names, tuple):
            pass  # ok
        elif isinstance(names, str):
            names = [names]
        else:
            e = self._type_error('names', 'List[str]')
            raise TypeError(e)
        if isinstance(indexes, list) or isinstance(indexes, tuple):
            pass  # ok
        elif isinstance(indexes, int):
            indexes = [indexes]
        else:
            e = self._type_error('indexes', 'List[int]')
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
                for o in self.__container:
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
            if index < 0 and index >= -1 * len(self.__container):
                o = self.__container[index]
            elif index >= 0 and index < len(self.__container):
                o = self.__container[index]
            else:
                raise IndexError('index out of range:  ' + str(index))
            if o and o not in olist:
                olist.append(o)
        return olist

    def _exists_error(self, name: str) -> str:
        e = 'failed to find ' + nmu.quotes(name)
        e += '\n' + 'acceptable names: ' + str(self.names)
        return e

    @property
    def select(self) -> NMObjectType:
        return self.__select

    @select.setter
    def select(self, name: str) -> None:
        self._select_set(name)

    def _select_set(
        self,
        name: Optional[str] = None,  # special: 'none' or ''
        index: Optional[int] = None,
        failure_alert: bool = True,
        notes: bool = False,    # too many notes for select
        quiet: bool = nmp.QUIET
    ) -> NMObjectType:
        old = self.__select
        if name is None:
            pass
        elif isinstance(name, str):
            if not name or name.lower() == 'none':
                self.__select = None
                if notes:
                    self.note = 'select = None'
                return None
            elif name.lower() == 'select':
                return self.__select  # nothing to do
            elif self.__select and name.lower() == self.__select.name.lower():
                return self.__select  # already selected
            elif self.exists(name):
                o = self.getitem(name)
                self.__select = o
                if notes:
                    self.note = 'select = ' + nmu.quotes(self.__select.name)
                self._modified()
                h = NMObjectContainer.__select_history(old, self.__select)
                self._history(h, quiet=quiet)
                return o
            elif not self.name_ok(name):
                e = self._value_error('name')
                raise ValueError(e)
            elif failure_alert:
                q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                     'do you want to create a new ' + self.content_type +
                     ' named ' + nmu.quotes(name) + '?')
                yn = nmu.input_yesno(q, tp=self._tp)
                if yn.lower() == 'y':
                    return self.new(name=name, select=True)
                self._history('cancel', quiet=quiet)
            else:
                e = self._exists_error(name)
                self._error(e, quiet=quiet)
                return None
        else:
            e = self._type_error('name', 'string')
            raise TypeError(e)

        # to reach here, name = None
        if index is None:
            self.__select = None
            return None
        elif isinstance(index, int):
            if index < 0 and index >= -1 * len(self.__container):
                o = self.__container[index]
            elif index >= 0 and index < len(self.__container):
                o = self.__container[index]
            else:
                raise IndexError('bad index: ' + str(index))
            self.__select = o
            if notes:
                self.note = 'select = ' + nmu.quotes(self.__select.name)
            self._modified()
            h = NMObjectContainer.__select_history(old, self.__select)
            self._history(h, quiet=quiet)
            return o
        else:
            e = self._type_error('index', 'int')
            raise TypeError(e)

    @staticmethod
    def __select_history(old: NMObjectType, new: NMObjectType) -> str:
        if new:
            n = new.name
        else:
            n = 'None'
        if old:
            return nmu.history_change('select', old.name, n)
        return 'selected ' + nmu.quotes(n)

    def new(
        self,
        name: str = 'default',
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> NMObjectType:
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default.
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not name or not self.name_ok(name, ok='default'):
            e = self._value_error('name')
            raise ValueError(e)
        if name.lower() == 'default':
            name = self.name_next()
        if self.exists(name):
            e = self._error(self.content_type + ' ' + nmu.quotes(name) +
                            ' already exists')
            raise RuntimeError(e)
        nmobject = self.__nmobject.copy()
        nmobject._NMObject__parent = self._parent
        nmobject._NMObject__name = name  # use mangled to prevent notes
        if not self.__container_append(nmobject):
            return None
        if select or not self.__select:
            old = self.__select
            self.__select = nmobject
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
        self.note = h
        self._modified()
        return nmobject

    def append(
        self,
        nmobject: NMObjectType,
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> bool:
        """
        Append a NMObject to container.
        """
        if not self.__container_append(nmobject):
            return False
        if select or not self.__select:
            old = self.__select
            self.__select = nmobject
            if old:
                h = NMObjectContainer.__select_history(old, self.__select)
                self._history(h, quiet=quiet)
            else:
                h = 'added/selected ' + nmu.quotes(nmobject.name)
                self._history(h, quiet=quiet)
        else:
            h = 'added ' + nmu.quotes(nmobject.name)
            self._history(h, quiet=quiet)
        self.note = h
        self._modified()
        return True

    def rename(
        self,
        name: str,
        newname: str,
        quiet: bool = nmp.QUIET
    ) -> str:
        if not self.__rename:
            e = self._error(self.content_type + ' items cannot be renamed')
            raise RuntimeError(e)
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name, ok='select'):
            e = self._value_error('name')
            raise ValueError(e)
        if name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return ''
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
            newname = self.name_next()
        if self.exists(newname):
            e = self._error(self.content_type + ' ' + nmu.quotes(newname) +
                            ' already exists')
            raise RuntimeError(e)
        old = nmu.quotes(o.name)
        # o.name = newname  # double history
        o._NMObject__name = newname  # mangled
        o.note = 'renamed to ' + nmu.quotes(o._NMObject__name)
        self._modified()
        new = nmu.quotes(o.name)
        h = 'renamed ' + old + ' to ' + new
        self.note = h
        self._history(h, quiet=quiet)
        return newname

    def duplicate(
        self,
        name: str,
        newname: str,
        select: bool = True,
        quiet: bool = nmp.QUIET
    ) -> NMObjectType:
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not isinstance(name, str):
            e = self._type_error('name', 'string')
            raise TypeError(e)
        if not self.name_ok(name, ok='select'):
            e = self._value_error('name')
            raise ValueError(e)
        if name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return None
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
            newname = self.name_next()
        if self.exists(newname):
            e = self._error(self.content_type + ' ' + nmu.quotes(newname) +
                            ' already exists')
            raise RuntimeError(e)
        c = o.copy()
        if not c:
            return None
        c._NMObject__name = newname
        self.__container_append(c)
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self.note = h
        self._history(h, quiet=quiet)
        if select or not self.__select:
            old = self.__select
            self.__select = c
            h = NMObjectContainer.__select_history(old, self.__select)
            self.note = 'select = ' + nmu.quotes(self.__select.name)
            self._history(h, quiet=quiet)
        self._modified()
        return c

    def remove(
        self,
        names: Optional[str] = [],
        indexes: Optional[int] = [],
        confirm: bool = True,
        quiet: bool = nmp.QUIET
    ) -> List[NMObjectType]:
        olist = self.getitems(names=names, indexes=indexes)
        if len(olist) == 0:
            return []
        if confirm:
            nlist = []
            for o in olist:
                nlist.append(o.name)
            q = ('are you sure you want to remove the following items?' +
                 '\n' + ', '.join(nlist))
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn.lower() == 'y':
                self._history('cancel', quiet=quiet)
                return []
        klist = []
        nlist = []
        for o in olist:
            klist.append(o)
            self.__container.remove(o)
            nlist.append(o.name)
            if self.__select is o:
                select_old = self.__select
                self.__select = None
        h = 'removed ' + ', '.join(nlist)
        self.note = h
        self._history(h, quiet=quiet)
        if self.__select is None and len(self.__container) > 0:
            self.__select = self.__container[0]
            h = NMObjectContainer.__select_history(select_old, self.__select)
            self._history(h, quiet=quiet)
        self._modified()
        return klist

    def name_next(
        self,
        first: int = 0
    ) -> str:
        """Get next default NMObject name based on prefix and sequence #."""
        if not isinstance(first, int):
            e = self._type_error('first', 'integer')
            raise TypeError(e)
        if not self.__prefix or first < 0:
            e = self._error('cannot generate default names')
            raise RuntimeError(e)
        i = self.name_next_seq(prefix=self.__prefix, first=first)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(
        self,
        prefix: str = 'default',
        first: int = 0
    ) -> int:
        """Get next seq num of default NMObject name based on prefix."""
        if not isinstance(first, int):
            e = self._type_error('first', 'integer')
            raise TypeError(e)
        if not isinstance(prefix, str):
            e = self._type_error('prefix', 'string')
            raise TypeError(e)
        if prefix.lower() == 'default':
            prefix = self.__prefix
        elif not self.name_ok(prefix):
            e = self._value_error('prefix')
            raise ValueError(e)
        if not prefix or first < 0:
            e = self._error('cannot generate default name')
            raise RuntimeError(e)
        elist = []
        for o in self.__container:
            name = o.name.lower()
            istr = name.replace(prefix.lower(), '')
            if str.isdigit(istr):
                elist.append(int(istr))
        if len(elist) == 0:
            return 0
        i = max(elist)
        return i + 1
