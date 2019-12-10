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

    def __init__(self, manager, parent, name, fxns, rename=True):
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__rename = rename
        if nmu.name_ok(name):
            self.__name = name
        else:
            self.__name = ''
            self.__error('bad name ' + nmu.quotes(name))
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
            tp = self.tree_path(history=True, quotes=True)
            e = self.__class__.__name__ + ' ' + tp + ' cannot be renamed'
            self.__error(e)
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

    def tree_path(self, history=False, quotes=False):
        if history:  # create tree path for history
            skip = nmc.HISTORY_TREE_PATH_SKIP
        else:
            skip = []
        plist = self.tree_path_list(skip=skip)
        if len(plist) > 0:
            tp = '.'.join(plist)
        else:
            tp = self.name
        if quotes:
            return nmu.quotes(tp)
        return tp

    def tree_path_list(self, names=True, skip=[]):
        if self.__class__.__name__ in skip:
            return []
        p = self.__parent
        if isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            t = p.tree_path_list(names=names, skip=skip)
            if names:
                t.append(self.name)
            else:
                t.append(self)
            return t
        if names:
            return [self.name]
        return [self]

    def save(self, path='', quiet=nmc.QUIET):
        self.__alert('under construction')
        return False


class Container(NMObject):
    """
    A list container for NMObject items (see above),
    one of which is 'select'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_next() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        ExperimentContainer, FolderContainer, DataContainer,
        DataSeriesContainer, ChannelContainer, EpochSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    def __init__(self, manager, parent, name, fxns, nmobj=None,
                 prefix='NMObj', rename=True, duplicate=True, kill=True):
        super().__init__(manager, parent, name, fxns, rename=rename)
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__rename = rename
        self.__duplicate = duplicate
        self.__kill = kill
        if nmobj is None:
            nmobj = NMObject(manager, parent, 'temp', fxns)
        elif not isinstance(nmobj, NMObject):
            nmobj = None
            e = 'argument ' + nmu.quotes('nmobj') + ' must be a NMObject'
            self.__error(e)
        self.__nmobj = nmobj
        self.__classname = nmobj.__class__.__name__
        if not prefix:
            self.__prefix = ''  # no default names
        elif not nmu.name_ok(prefix):
            self.__prefix = ''
            self.__error('bad prefix ' + nmu.quotes(prefix))
        else:
            self.__prefix = prefix
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', etc
        # and change 'select' to 'folder_select', etc
        k = {'nmobjects': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'select': s})
        return k

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if not self.__rename:
            tp = self.tree_path(history=True, quotes=True)
            self.__error('prefix for ' + tp + ' cannot be changed')
            return False
        self.__prefix = prefix
        self.__history(self.__classname + ' prefix = ' + prefix)
        return True

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

    def get(self, name='select', quiet=nmc.QUIET):
        """Get NMObject from Container"""
        if not self.__thecontainer:
            tp = self.tree_path(history=True, quotes=True)
            self.__alert('container ' + tp + ' is empty', quiet=quiet)
            return None
        if not name or name.lower() == 'select':
            return self.__select
        for o in self.__thecontainer:
            if name.casefold() == o.name.casefold():
                return o
        tp = self.tree_path(history=True, quotes=True)
        e = 'failed to find ' + nmu.quotes(name) + ' in ' + tp
        self.__error(e, quiet=quiet)
        self.__error('acceptable names: ' + str(self.names), quiet=quiet)
        return None

    # REMOVED. Do not allow easy access to container.
    # def get_all(self):
    #    """Get the container (list) of all NMObject items"""
    #    return self.__thecontainer

    @property
    def select(self):
        return self.__select

    @select.setter
    def select(self, name):
        return self.__select_set(name=name)

    def __select_set(self, name, call_new=True, ask_new=True, quiet=nmc.QUIET):
        if self.exists(name):
            o = self.get(name, quiet=quiet)
            self.__select = o
            tp = o.tree_path(history=True)
            self.__history('selected' + nmc.S0 + tp, quiet=quiet)
            return o
        if call_new:
            if ask_new and not self.__quiet(quiet):
                tp = self.tree_path(history=True, quotes=True)
                q = ('failed to find ' + nmu.quotes(name) + ' in ' + tp +
                     '. do you want to create a new ' + self.__classname +
                     ' named ' + nmu.quotes(name) + '?')
                yn = nmu.input_yesno(q)
                if not yn == 'y':
                    self.__history('abort')
                    return None
            return self.new(name=name, select=True, quiet=quiet)
        o = self.get(name, quiet=quiet)  # invokes alert
        return None

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

    def new(self, name='default', nmobj=None, select=True, quiet=nmc.QUIET):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
            if not name:
                return None
        elif not nmu.name_ok(name):
            self.__error('bad name ' + nmu.quotes(name), quiet=quiet)
            return None
        elif self.exists(name):
            e = self.__classname + ' ' + nmu.quotes(name) + ' already exists'
            self.__error(e, quiet=quiet)
            return None
        if nmobj is None:
            o = NMObject(self.__manager, self.__parent, name, self.__fxns,
                         rename=self.__rename)
        elif isinstance(nmobj, NMObject):  # child class will pass nmobj
            if nmobj.__class__.__name__ == self.__classname:
                o = nmobj
                o._NMObject__name = name  # in case name='default'
                o._NMObject__parent = self.__parent  # reset parent reference
                o._NMObject__rename = self.__rename
            else:
                e = ('argument ' + nmu.quotes(nmobj) + ' must be a ' +
                     self.__classname + ' object')
                self.__error(e, quiet=quiet)
                return None
        else:
            e = 'argument ' + nmu.quotes(nmobj) + ' must be a NMObject'
            self.__error(e, quiet=quiet)
            return None
        self.__thecontainer.append(o)
        h = 'created'
        if select or not self.__select:
            self.__select = o
            h += '/selected'
        tp = o.tree_path(history=True)
        self.__history(h + nmc.S0 + tp, quiet=quiet)
        return o

    def rename(self, name, newname, quiet=nmc.QUIET):
        if not self.__rename:
            tp = self.tree_path(history=True, quotes=True)
            self.__error('items in ' + tp + ' cannot be renamed', quiet=quiet)
            return False
        o = self.get(name, quiet=quiet)
        if not o:
            return False
        if not nmu.name_ok(newname):
            self.__error('bad newname ' + nmu.quotes(newname), quiet=quiet)
            return False
        if self.exists(newname):
            e = 'name ' + nmu.quotes(newname) + ' is already in use'
            self.__error(e, quiet=quiet)
            return False
        otp = o.tree_path(history=True, quotes=True)
        o.name = newname
        ntp = o.tree_path(history=True, quotes=True)
        h = 'renamed' + nmc.S0 + otp + ' to ' + ntp
        self.__history(h, quiet=quiet)
        return True

    def duplicate(self, name, newname, select=True, quiet=nmc.QUIET):
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not self.__duplicate:
            e = self.__classname + ' objects cannot be duplicated'
            self.__error(e, quiet=quiet)
            return False
        o = self.get(name=name, quiet=quiet)
        if not o:
            return None
        if not newname or newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
            if not newname:
                return None
        elif not nmu.name_ok(newname):
            e = 'bad ' + self.__classname + ' name ' + nmu.quotes(newname)
            self.__error(e, quiet=quiet)
            return None
        oo = self.get(name=newname, quiet=True)
        if oo:
            tp = oo.tree_path(history=True)
            self.__error(tp + ' already exists', quiet=quiet)
            return None
        c = copy.deepcopy(o)
        if c:
            c.name = newname
            c._NMObject__parent = self.__parent  # reset parent reference
            self.__thecontainer.append(c)
            otp = o.tree_path(history=True, quotes=True)
            ctp = c.tree_path(history=True, quotes=True)
            h = 'copied ' + otp + ' to ' + ctp
            self.__history(h, quiet=quiet)
            if select or not self.__select:
                self.__select = c
                tp = c.tree_path(history=True)
                self.__history('selected' + nmc.S0 + tp, quiet=quiet)
        return c

    def kill(self, name, quiet=nmc.QUIET):
        """
        Kill NMObject.

        Args:
            name: name of NMObject to kill

        Returns:
            True for success, False otherwise
        """
        if not self.__kill:
            e = self.__classname + ' objects cannot be killed'
            self.__error(e, quiet=quiet)
            return False
        o = self.get(name, quiet=quiet)
        if not o:
            return False
        name = o.name  # in case name = 'select'
        cname = o.__class__.__name__
        if not self.__quiet(quiet):
            q = ('are you sure you want to kill ' + cname + ' ' +
                 nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q)
            if not yn == 'y':
                self.__history('abort')
                return False
        select_next = o is self.__select  # killing select, so need new select
        if select_next:
            i = self.item_num(name)
        self.__thecontainer.remove(o)
        tp = o.tree_path(history=True)
        self.__history('killed' + nmc.S0 + tp, quiet=quiet)
        items = len(self.__thecontainer)
        if select_next and items > 0:
            i = max(i, 0)
            i = min(i, items-1)
            o = self.__thecontainer[i]
            self.__select = o
            tp = o.tree_path(history=True)
            self.__history('selected' + nmc.S0 + tp, quiet=quiet)
        return True

    def name_next(self, first=0, quiet=nmc.QUIET):
        """Get next default NMObject name based on prefix and sequence #."""
        if not self.__prefix or first < 0:
            e = self.__classname + ' objects do not have default names'
            self.__error(e, quiet=quiet)
            return ''
        i = self.name_next_seq(self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=nmc.QUIET):
        """Get next seq num of default NMObject name based on prefix."""
        if not prefix or prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            e = self.__classname + ' objects do not have default names'
            self.__error(e, quiet=quiet)
            return -1
        if not nmu.name_ok(prefix):
            self.__error('bad prefix ' + nmu.quotes(prefix), quiet=quiet)
            return -1
        n = 10 + len(self.__thecontainer)
        for i in range(first, n):
            name = prefix + str(i)
            if not self.exists(name):
                return i
        return -1
