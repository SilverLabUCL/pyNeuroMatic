#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
import datetime

from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import history

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

    @property
    def date(self):
        return self.__date

    @property
    def path(self):
        if not self.name:
            return ""
        thepath = self.name
        p = self
        for i in range(0, 20):  # loop thru parent ancestry
            if not p.parent:
                return thepath  # no more parents
            p = p.parent
            thepath = p.name + "." + thepath


class Container(object):
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

    def __init__(self, parent, prefix="NMObj"):
        self.__parent = parent
        self.__prefix = prefix  # used in name_next()
        self.__count_from = 0  # used in name_next()
        self.__date = str(datetime.datetime.now())
        self.__objects = []  # container of NMObject items
        self.__object_select = None  # selected NMObject

    @property
    def parent(self):
        return self.__parent

    @property
    def date(self):
        return self.__date

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if not prefix:
            prefix = ""
        elif not name_ok(prefix):
            return error("bad prefix " + quotes(prefix))
        self.__prefix = prefix
        return True

    @property
    def count_from(self):  # see name_next()
        return self.__count_from

    @count_from.setter
    def count_from(self, count_from):
        count_from = round(count_from)
        if count_from >= 0:
            self.__count_from = count_from

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
        return NMObject  # change NMObject to Experiment, Folder, TimeSeries, etc.

    def instance_ok(self, obj):  # child class should override this function
        return isinstance(obj, NMObject)  # change NMObject to Experiment, etc.

    def __type(self):
        if not self.__objects:
            return "None"
        return self.__objects[0].__class__.__name__

    def __tname(self, name):
        t = self.__type()
        if t == "None":
            return t
        #return t + " " + quotes(name)  # object type + name
        return quotes(name)

    def get(self, name=""):
        """Get NMObject from Container"""
        if not name or name.casefold() == 'selected':
            return self.__object_select
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return o
        error("failed to find " + self.__tname(name))
        print("acceptable names: " + str(self.name_list))
        return None

    def getAll(self):
        """Get the container (list) of all NMObject items"""
        return self.__objects

    @property
    def select(self):
        if self.__object_select:
            return self.__object_select
        return None

    @select.setter
    def select(self, name):
        """Select NMObject in Container"""
        quiet = False
        if not name_ok(name):
            return error("bad name " + quotes(name))
        if not self.__objects:
            return False
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                self.__object_select = o
                if not quiet:
                    history("selected " + self.__tname(name))
                return True
        error("failed to find " + self.__tname(name))
        print("acceptable names: " + str(self.name_list))
        return False

    def add(self, obj, select=True, quiet=False):
        """Add NMObject to Container."""
        if not self.instance_ok(obj):
            return error("encountered bad Container object")
        if not name_ok(obj.name):
            return error("bad name " + quotes(obj.name))
        if self.exists(obj.name):
            pass  # nothing to do
        else:
            self.__objects.append(obj)
        if select or not self.__object_select:
            self.__object_select = obj
            if not quiet:
                history("added/selected " + self.__tname(obj.name))
            return True
        if not quiet:
            history("added " + self.__tname(obj.name))
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
        o = self.get(name)
        if not o:
            return False
        if not name_ok(newname):
            return error("bad newname " + quotes(newname))
        if self.exists(newname):
            error(self.__tname(newname) + " already exists")
            return False
        c = copy.deepcopy(o)
        if c is not None:
            c.name = newname
            self.__objects.append(c)
            if not quiet:
                history("copied " + self.__tname(name) +
                        " to " + quotes(newname))
        return c

    def exists(self, name):
        """Check if NMObject exists within container"""
        if not self.__objects:
            return False
        if not name_ok(name):
            return error("bad name " + quotes(name))
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return True
        return False

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
        selected = o is self.__object_select
        self.__objects.remove(o)
        if not quiet:
            history("killed " + self.__tname(name))
        if selected and len(self.__objects) > 0:
            self.__object_select = self.__objects[0]
            if not quiet:
                history("selected " + self.__tname(self.__object_select.name))
        return True

    def new(self, name="", select=True, quiet=False):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass "" for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not name:
            name = self.name_next()
        elif not name_ok(name):
            error("bad name " + quotes(name))
            return None
        elif self.exists(name):
            error(self.__tname(name) + " already exists")
            return None
        o = self.object_new(name)
        self.__objects.append(o)
        if select or not self.__object_select:
            self.__object_select = o
            if not quiet:
                history("created/selected " + self.__tname(name))
                print(o.path)
            return o
        if not quiet:
            history("created " + self.__tname(name))
            print(o.path)
        return o

    def name_next(self):
        """Get next default NMObject name based on prefix."""
        if self.__prefix:
            prefix = self.__prefix
        else:
            prefix = "None"
        n = 10 + len(self.__objects)
        for i in range(self.__count_from, n):
            name = prefix + str(i)
            if not self.exists(name):
                return name
        return prefix + "99999"

    def rename(self, name, newname, quiet=False):
        o = self.get(name)
        if not o:
            return False
        if not name_ok(newname):
            return error("bad newname " + quotes(newname))
        if self.exists(newname):
            error("name " + quotes(newname) + " is already used")
            return False
        o.name = newname
        if not quiet:
            history("renamed " + self.__tname(name) +
                    " to " + quotes(newname))
        return True

    def open_(self, path):
        pass

    def save(self, name):
        pass
