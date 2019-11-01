#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py
import json
from colorama import Fore
import copy
import datetime
import inspect

import nm_configs as nmc
import nm_utilities as nmu


class NMObject(object):
    """
    NM objects to be stored in a 'Container' list (see below).

    Known Children:
        Experiment, Folder, Data, DataPrefix, Channel, EpochSet

    Attributes:
        parent (NMObject):
        name (str):
        date (str):
    """

    def __init__(self, parent, name):
        self.__parent = parent
        self.__name = name
        self.__date = str(datetime.datetime.now())

    @property
    def parent(self):
        return self.__parent

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name and nmu.name_ok(name):
            self.__name = name

    @property
    def date(self):
        return self.__date

    @property
    def tree_path(self):
        if nmc.TREE_PATH_LONG:
            skip = nmc.TREE_PATH_SKIP_LIST
            plist = self.tree_path_list(skipList=skip)
            return '.'.join(plist)
        return self.name

    def tree_path_list(self, skipList=[]):
        if not self.name:
            return []
        thepath = [self.name]
        p = self
        for i in range(0, 20):  # loop thru parent ancestry
            if not p.parent:
                break  # no more parents
            p = p.parent
            if p.__class__.__name__ == 'Manager':
                break
            elif p.__class__.__name__ in skipList:
                pass
            else:
                thepath.insert(0, p.name)
        return thepath

    @property
    def manager(self):
        p = self
        for i in range(0, 20):  # loop thru parent ancestry
            if not p.parent:
                return None  # no more parents
            p = p.parent
            if p.__class__.__name__ == 'Manager':
                return p


class Container(NMObject):
    """
    A list container for NMObject items (see above),
    one of which is 'selected'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_next() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        ExperimentContainer, FolderContainer, DataContainer,
        DataPrefixContainer, ChannelContainer, EpochSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(), name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject    
    """

    def __init__(self, parent, name, prefix='NMObj', count_from=0):
        super().__init__(parent, name)
        self.__prefix = prefix  # used in name_next()
        self.__count_from = count_from  # used in name_next()
        self.__objects = []  # container of NMObject items
        self.__object_select = None  # selected NMObject

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if not prefix:
            prefix = ''
        elif not nmu.name_ok(prefix):
            return nmu.error('bad prefix ' + nmu.quotes(prefix))
        self.__prefix = prefix
        return True

    @property
    def count(self):
        """Number of NMObject items stored in Container"""
        return len(self.__objects)

    @property
    def name_list(self):
        """Get list of names of NMObject items in Container"""
        nlist = []
        if self.__objects:
            for o in self.__objects:
                nlist.append(o.name)
        return nlist

    def object_new(self, name):  # child class should override this function
        return NMObject  # change NMObject to Experiment, Folder, Data, etc.

    def instance_ok(self, obj):  # child class should override this function
        return isinstance(obj, NMObject)  # change NMObject to Experiment, etc.

    def __type(self):
        if not self.__objects:
            return 'None'
        return self.__objects[0].__class__.__name__

    def __tname(self, name):
        t = self.__type()
        if t == 'None':
            return t
        # return t + " " + nmu.quotes(name)  # object type + name
        return nmu.quotes(name)

    def get(self, name='selected', quiet=False):
        """Get NMObject from Container"""
        if not self.__objects:
            if not quiet:
                nmu.alert('nothing to select in ' + self.parent.tree_path)
            return None
        if not name or name.lower() == 'selected':
            return self.__object_select
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return o
        if not quiet:
            nmu.alert('failed to find ' + nmu.quotes(name) + ' in ' +
                       self.parent.tree_path)
            print('acceptable names: ' + str(self.name_list))
        return None

    def getAll(self):
        """Get the container (list) of all NMObject items"""
        return self.__objects

    @property
    def select(self):
        return self.__object_select

    @select.setter
    def select(self, name):
        return self.select_set(name)

    def select_set(self, name, quiet=False, new=False):
        """Select NMObject in Container"""
        o = self.get(name=name, quiet=quiet)
        if o:
            self.__object_select = o
            if not quiet:
                nmu.history('selected' + nmc.S0 + o.tree_path)
            return True
        if new:
            o = self.new(name=name, quiet=quiet)
            return o is not None
        return False

    def new(self, name='default', select=True, quiet=False):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not name or name.lower() == 'default':
            name = self.name_next()
        elif not nmu.name_ok(name):
            nmu.error('bad name ' + nmu.quotes(name))
            return None
        elif self.exists(name):
            nmu.error(self.__tname(name) + ' already exists')
            return None
        o = self.object_new(name)
        self.__objects.append(o)
        if select or not self.__object_select:
            self.__object_select = o
            if not quiet:
                nmu.history('created/selected' + nmc.S0 + o.tree_path)
            return o
        if not quiet:
            nmu.history('created' + nmc.S0 + o.tree_path)
        return o

    def add(self, nmobj, select=True, quiet=False):
        """Add NMObject to Container."""
        if not self.instance_ok(nmobj):
            if not quiet:
                nmu.error('encountered object not of type ' + self.__type())
            return False
        if not nmu.name_ok(nmobj.name):
            if not quiet:
                nmu.error('bad object name ' + nmu.quotes(nmobj.name))
            return False
        if self.exists(nmobj.name):
            pass  # nothing to do
        else:
            self.__objects.append(nmobj)
        if select or not self.__object_select:
            self.__object_select = nmobj
            if not quiet:
                nmu.history('added/selected' + nmc.S0 + nmobj.tree_path)
            return True
        if not quiet:
            nmu.history('added' + nmc.S0 + nmobj.tree_path)
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
        o = self.get(name=name, quiet=quiet)
        if not o:
            return None
        if not nmu.name_ok(newname):
            if not quiet:
                nmu.error('bad object name ' + nmu.quotes(newname))
            return None
        oo = self.get(name=newname, quiet=True)
        if oo:
            if not quiet:
                nmu.error(oo.tree_path + ' already exists')
            return None
        c = copy.deepcopy(o)
        if c is not None:
            c.name = newname
            self.__objects.append(c)
            if not quiet:
                nmu.history('copied ' + o.tree_path +
                             ' to ' + c.tree_path)
        return c

    def which_item(self, name):
        """Find item # of NMObject in container"""
        if not self.__objects:
            return -1
        for i in range(0, len(self.__objects)):
            if name.casefold() == self.__objects[i].name.casefold():
                return i
        return -1

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.which_item(name) != -1

    def kill(self, name, quiet=False):
        """
        Kill NMObject.

        Args:
            name: name of NMObject to kill

        Returns:
            True for success, False otherwise
        """
        o = self.get(name)
        if not o:
            return False
        cname = o.__class__.__name__
        yn = nmu.input_yesno('are you sure you want to kill ' + cname +
                                    ' ' + nmu.quotes(name) + '?')
        if not yn == 'y':
            nmu.history('abort')
            return False
        selected = o is self.__object_select
        if selected:
            i = self.which_item(name)
        path = o.tree_path
        self.__objects.remove(o)
        if not quiet:
            nmu.history('killed' + nmc.S0 + path)
        items = len(self.__objects)
        if selected and items > 0:
            i = max(i, 0)
            i = min(i, items-1)
            self.select_set(self.__objects[i].name, quiet=quiet)
        return True

    def name_next(self, prefix='selected'):
        """Get next default NMObject name based on prefix."""
        if not prefix or prefix.lower() == 'selected':
            if self.__prefix:
                prefix = self.__prefix
            else:
                return ''
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix))
            return ''
        return prefix + str(self.name_next_seq(prefix))

    def name_next_seq(self, prefix='selected'):
        """Get next seq num of default NMObject name based on prefix."""
        if not prefix or prefix.lower() == 'selected':
            if self.__prefix:
                prefix = self.__prefix
            else:
                return 0
        if not nmu.name_ok(prefix):
            nmu.error('bad prefix ' + nmu.quotes(prefix))
            return 0
        n = 10 + len(self.__objects)
        for i in range(self.__count_from, n):
            name = prefix + str(i)
            if not self.exists(name):
                return i
        return 0

    def rename(self, name, newname, quiet=False):
        o = self.get(name)
        if not o:
            return False
        if not nmu.name_ok(newname):
            return nmu.error('bad newname ' + nmu.quotes(newname))
        if self.exists(newname):
            nmu.error('name ' + nmu.quotes(newname) + ' is already used')
            return False
        old_path = o.tree_path
        o.name = newname
        if not quiet:
            nmu.history('renamed' + nmc.S0 + old_path + ' to ' + o.tree_path)
        return True
