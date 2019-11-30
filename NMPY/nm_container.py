#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
import datetime

import nm_configs as nmc
import nm_utilities as nmu


class NMObject(object):
    """
    NM objects to be stored in a 'Container' list (see below).

    Known Children:
        Project, Folder, Data, DataSeries, Channel, EpochSet, Note

    Attributes:
        parent (NMObject):
        name (str):
        date (str):
    """

    __rename = True

    def __init__(self, parent, name):
        self.__parent = parent
        if nmu.name_ok(name):
            self.__name = name
        else:
            self.__name = ''
            nmu.error('bad name ' + nmu.quotes(name))
        self.__date = str(datetime.datetime.now())

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', 'data', etc.
        return {'nmobject': self.name}

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not self.__rename:
            nmu.error(self.__class__.__name__ + ' objects cannot be renamed')
            return False
        if name and nmu.name_ok(name):
            self.__name = name
            return True
        return False

    @property
    def date(self):
        return self.__date

    @property
    def content_tree(self):
        p = self.__parent
        if p and isinstance(p, NMObject):
            k = {}
            k.update(p.content_tree)
            k.update(self.content)
            return k
        return self.content

    def tree_path(self, history=False):
        if history:  # create tree path for history
            skip = nmc.HISTORY_TREE_PATH_SKIP
        else:
            skip = []
        plist = self.tree_path_list(skip=skip)
        if len(plist) > 0:
            return '.'.join(plist)
        return self.name

    def tree_path_list(self, names=True, skip=[]):
        if self.__class__.__name__ in skip:
            return []
        p = self.__parent
        if p and isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            t = p.tree_path_list(names=names, skip=skip)
            if names:
                t.append(self.name)
            else:
                t.append(self)
            return t
        if names:
            return [self.name]
        return [self]

    @property
    def __manager(self):
        p = self
        for i in range(0, 20):  # loop thru parent ancestry
            p = p.parent
            if not p:
                return None
            if p.__class__.__name__ == 'Manager':  # cannot use isinstance
                return p
        return None


class Container(NMObject):
    """
    A list container for NMObject items (see above),
    one of which is 'select'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_default() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        ExperimentContainer, FolderContainer, DataContainer,
        DataSeriesContainer, ChannelContainer, EpochSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_default(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    __thecontainer = []  # container of NMObject items
    __select = None  # selected NMObject
    __nmobj = None
    __classname = 'NoneType'
    __prefix = ''  # default name prefix, see name_default()
    __rename = True
    __duplicate = True
    __kill = True

    def __init__(self, parent, name='NMContainer', nmobj=None, prefix='NMObj'):
        super().__init__(parent, name)
        self.__parent = parent
        if nmobj is None:
            nmobj = NMObject(parent, 'temp')
        elif not isinstance(nmobj, NMObject):
            nmobj = None
            nmu.error('argument ' + nmu.quotes(nmobj) + ' must be a NMObject')
        self.__nmobj = nmobj
        self.__classname = nmobj.__class__.__name__
        if not prefix:
            pass  # no default names
        elif not nmu.name_ok(prefix):
            nmu.error('bad default name prefix ' + nmu.quotes(prefix))
        else:
            self.__prefix = prefix

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', etc
        # and change 'select' to 'folder_select', etc
        k = {'nmobject', self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'select': s})
        return k

    @property
    def prefix(self):  # see name_default()
        return self.__prefix

    @property
    def count(self):
        """Number of NMObject items stored in Container"""
        return len(self.__thecontainer)

    @property
    def names(self):
        """Get list of names of NMObject items in Container"""
        nlist = []
        if self.__thecontainer:
            for o in self.__thecontainer:
                nlist.append(o.name)
        return nlist

    def get(self, name='select', quiet=False):
        """Get NMObject from Container"""
        if not self.__thecontainer:
            tp = self.tree_path(history=True)
            nmu.alert('container ' + tp + ' is empty',
                      quiet=quiet)
            return None
        if not name or name.lower() == 'select':
            return self.__select
        for o in self.__thecontainer:
            if name.casefold() == o.name.casefold():
                return o
        tp = self.tree_path(history=True)
        nmu.error('failed to find ' + nmu.quotes(name) + ' in ' + tp,
                  quiet=quiet)
        nmu.error('acceptable names: ' + str(self.names), quiet=quiet)
        return None

    def get_all(self):
        """Get the container (list) of all NMObject items"""
        return self.__thecontainer

    @property
    def select(self):
        return self.__select

    @select.setter
    def select(self, name):
        return self.select_set(name)

    def select_set(self, name, call_new=False, quiet=False):
        """Select NMObject in Container"""
        if self.exists(name):
            o = self.get(name=name, quiet=True)
            self.__select = o
            tp = o.tree_path(history=True)
            nmu.history('selected' + nmc.S0 + tp, quiet=quiet)
            return o
        if call_new:
            o = self.new(name=name, select=True, quiet=quiet)
            return o is not None
        o = self.get(name, quiet=quiet)  # invokes error message
        return False

    def item_num(self, name):
        """Find item # of NMObject in container"""
        if not self.__thecontainer:
            return -1
        for i in range(0, len(self.__thecontainer)):
            if name.casefold() == self.__thecontainer[i].name.casefold():
                return i
        return -1

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.item_num(name) >= 0

    def new(self, name='default', nmobj=None, select=True, quiet=False):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not name or name.lower() == 'default':
            name = self.name_default(quiet=quiet)
            if not name:
                return None
        elif not nmu.name_ok(name):
            nmu.error('bad name ' + nmu.quotes(name), quiet=quiet)
            return None
        elif self.exists(name):
            nmu.error(self.__classname + ' ' + nmu.quotes(name) +
                      ' already exists', quiet=quiet)
            return None
        if nmobj is None:
            o = NMObject(self.__parent, name)
        elif isinstance(nmobj, NMObject):  # child class will pass nmobj
            if nmobj.__class__.__name__ == self.__classname:
                o = nmobj
                o._NMObject__name = name  # in case name='default'
            else:
                nmu.error('argument ' + nmu.quotes(nmobj) + ' must be a ' +
                          self.__classname + ' object', quiet=quiet)
                return None
        else:
            nmu.error('argument ' + nmu.quotes(nmobj) + ' must be a NMObject',
                      quiet=quiet)
            return None
        self.__thecontainer.append(o)
        h = 'created'
        if select or not self.__select:
            self.__select = o
            h += '/selected'
        tp = o.tree_path(history=True)
        nmu.history(h + nmc.S0 + tp, quiet=quiet)
        return o

    def rename(self, name, newname, quiet=False):
        if not self.__rename:
            nmu.error(self.__classname + ' objects cannot be renamed')
            return False
        o = self.get(name, quiet=quiet)
        if not o:
            return False
        if not nmu.name_ok(newname):
            nmu.error('bad newname ' + nmu.quotes(newname), quiet=quiet)
            return False
        if self.exists(newname):
            e = 'name ' + nmu.quotes(newname) + ' is already in use'
            nmu.error(e, quiet=quiet)
            return False
        old_tp = o.tree_path(history=True)
        o.name = newname
        new_tp = o.tree_path(history=True)
        h = 'renamed' + nmc.S0 + old_tp + ' to ' + new_tp
        nmu.history(h, quiet=quiet)
        return True

    def duplicate(self, name, newname, select=False, quiet=False):
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not self.__duplicate:
            nmu.error(self.__classname + ' objects cannot be duplicated')
            return False
        o = self.get(name=name, quiet=quiet)
        if not o:
            return None
        if not newname or newname.lower() == 'default':
            newname = self.name_default(quiet=quiet)
            if not newname:
                return None
        elif not nmu.name_ok(newname):
            e = 'bad ' + self.__classname + ' name ' + nmu.quotes(newname)
            nmu.error(e, quiet=quiet)
            return None
        oo = self.get(name=newname, quiet=True)
        if oo:
            tp = oo.tree_path(history=True)
            nmu.error(tp + ' already exists', quiet=quiet)
            return None
        c = copy.deepcopy(o)
        if c:
            c.name = newname
            c._NMObject__parent = self.__parent  # reset parent reference
            self.__thecontainer.append(c)
            otp = o.tree_path(history=True)
            ctp = c.tree_path(history=True)
            h = 'copied ' + otp + ' to ' + ctp
            nmu.history(h, quiet=quiet)
        return c

    def kill(self, name, quiet=False):
        """
        Kill NMObject.

        Args:
            name: name of NMObject to kill

        Returns:
            True for success, False otherwise
        """
        if not self.__kill:
            nmu.error(self.__classname + ' objects cannot be killed')
            return False
        o = self.get(name, quiet=quiet)
        if not o:
            return False
        cname = o.__class__.__name__
        if not quiet:
            q = ('are you sure you want to kill ' + cname + ' ' +
                 nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q)
            if not yn == 'y':
                nmu.history('abort')
                return False
        selected = o is self.__select
        if selected:
            i = self.item_num(name)
        self.__thecontainer.remove(o)
        tp = o.tree_path(history=True)
        nmu.history('killed' + nmc.S0 + tp, quiet=quiet)
        items = len(self.__thecontainer)
        if selected and items > 0:
            i = max(i, 0)
            i = min(i, items-1)
            self.select_set(self.__thecontainer[i].name, quiet=quiet)
        return True

    def name_default(self, first=0, quiet=False):
        """Get next default NMObject name based on prefix and sequence #."""
        if not self.__prefix or first < 0:
            e = self.__classname + ' objects do not have default names'
            nmu.error(e, quiet=quiet)
            return ''
        i = self.name_next_seq(self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=False):
        """Get next seq num of default NMObject name based on prefix."""
        if not prefix or prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            e = self.__classname + ' objects do not have default names'
            nmu.error(e, quiet=quiet)
            return -1
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix), quiet=quiet)
            return -1
        n = 10 + len(self.__thecontainer)
        for i in range(first, n):
            name = prefix + str(i)
            if not self.exists(name):
                return i
        return -1
