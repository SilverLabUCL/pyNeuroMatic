#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import history


class Container(object):
    """
    NM Container class
    Container (i.e. list) for objects, e.g. Experiment, Folder,
    WavePrefix, Channel...
    Each object stored in container must have a unique name.
    The name can start with the same prefix, but this is optional.
    
    Can __objects be a dictionary with names as keys???
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self.__objects = []
        self.__object_select = None

    def object_new(self, name):  # child class should override
        return object  # change object to Experiment, Folder...

    def instance_ok(self, obj):  # child class should override
        return isinstance(obj, object)  # change object to Folder...

    def __type(self):
        if not self.__objects:
            return "None"
        return self.__objects[0].__class__.__name__

    def __tname(self, name):
        return self.__type() + " " + quotes(name)  # object type + name

    @property
    def prefix(self):
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if name_ok(prefix):
            self.__prefix = prefix

    @property
    def name(self):
        if self.__object_select:
            return self.__object_select.name  # name of selected object
        return "None"

    # @name.setter
    # def name(self, name):
    #     if self.__object_select and name_ok(name) and not self.exists(name):
    #         self.__object_select.name = name  # set name of selected object

    @property
    def items(self):
        """Number of objects stored in Container"""
        return len(self.__objects)

    @property
    def names(self):
        """Get list of names of objects stored in Container"""
        nlist = []
        if self.__objects:
            for o in self.__objects:
                nlist.append(o.name)
        return nlist

    def get(self, name):
        """Get object from container"""
        if not name or name.casefold() == "SELECTED".casefold():
            return self.__object_select
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return o
        error("failed to find " + self.__tname(name))
        self.name_list(history=True)
        return None

    def getAll(self):
        """Get the container (list) of all objects"""
        return self.__objects

    def select(self, name):
        """Select object in container"""
        if not name_ok(name):
            return False
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                self.__object_select = o
                history("selected " + self.__tname(name))
                return True
        error("failed to find " + self.__tname(name))
        self.name_list(history=True)
        return False

    def exists(self, name):
        """Check if object exists within container"""
        if self.__objects and name_ok(name):
            for o in self.__objects:
                if name.casefold() == o.name.casefold():
                    return True
        return False

    def add(self, obj, select=True):
        """Add object to container."""
        if not self.instance_ok(obj):
            error("encountered bad Container")
            return False
        if not name_ok(obj.name):
            return False
        if self.exists(obj.name):
            pass  # nothing to do
        else:
            self.__objects.append(obj)
            history("added " + self.__tname(obj.name))
        if select or not self.__object_select:
            self.__object_select = obj
            history("selected " + self.__tname(obj.name))
        return True

    def new(self, name, select=True):
        """
        Create a new object and add to container.

        Args:
            name: unique name of new object, pass "" for default
            select: select this object

        Returns:
            new object if successful, None otherwise
        """
        if not name:
            name = self.name_next()
        elif not name_ok(name):
            return None
        elif self.exists(name):
            error(self.__tname(name) + " already exists")
            return None
        o = self.object_new(name)
        self.__objects.append(o)
        history("created " + self.__tname(name))
        if select or not self.__object_select:
            self.__object_select = o
            history("selected " + self.__tname(name))
        return o

    def name_next(self):
        """Get next default object name based on prefix string."""
        if not self.__prefix:
            prefix = "None"
        else:
            prefix = self.__prefix
        n = 10 + len(self.__objects)
        for i in range(0, n):
            name = prefix + str(i)
            if not self.exists(name):
                return name
        return prefix + "99999"

    def rename(self, name, newname):
        o = self.get(name)
        if not o:
            return False
        if not name_ok(newname):
            return False
        if self.exists(newname):
            error(self.__tname(newname) + " already exists")
            return False
        o.name = newname
        history("renamed " + self.__tname(name) +
                " to " + quotes(newname))
        return True

    def duplicate(self, name, newname, select=False):
        """
        Copy an object.

        Args:
            name: name of object to copy
            newname: name of new object

        Returns:
            new object if successful, None otherwise
        """
        o = self.get(name)
        if not o:
            return False
        if not name_ok(newname):
            return False
        if self.exists(newname):
            error(self.__tname(newname) + " already exists")
            return False
        c = copy.deepcopy(o)
        if c is not None:
            c.name = newname
            self.__objects.append(c)
            history("copied " + self.__tname(name) +
                    " to " + quotes(newname))
        return c

    def kill(self, name):
        """
        Kill an object.

        Args:
            name: name of object to kill

        Returns:
            True for success, False otherwise
        """
        o = self.get(name)
        if not o:
            return False
        selected = o is self.__object_select
        self.__objects.remove(o)
        history("killed " + self.__tname(name))
        if selected and len(self.__objects) > 0:
            self.__object_select = self.__objects[0]
            history("selected " +
                    self.__tname(self.__object_select.name))
        return True

    def open_(self, path):
        pass

    def save(self, name):
        pass
