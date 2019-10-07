#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019
@author: Jason Rothman
Created on Fri Oct  4 12:11:14 2019
"""
import copy
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import history


class Container(object):
    """
    NM Container class
    Container (i.e. list) for objects, where objects are Experiments,
    Folders, WavePrefixes, Channels, etc.
    Each object stored in container must have a unique name.
    """

    OBJECT_NAME_PREFIX = "NMObj"

    def __init__(self, name):
        self.name = name
        self.__objects = []
        self.__object_select = None
        # history("created Container " + quotes(name))
        
    def object_new(self, name):
        return Container(name)
    
    def instance_ok(self, obj):
        return isinstance(obj, Container)

    def object_type(self):
        if not self.__objects:
            return "Object"
        return self.__objects[0].__class__.__name__

    def object_name(self, name):
        return self.object_type() + " " + quotes(name)

    @property
    def name(self):
        """Return name of container"""
        return self.__name

    @name.setter
    def name(self, name):
        """Set name of container"""
        if not name_ok(name):
            return False
        self.__name = name
        return True

    def get(self, name):
        """Get object from container"""
        if not name:
            return self.__object_select
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return o
        error("failed to find " + self.object_name(name))
        self.name_list(history=True)
        return None

    def select(self, name):
        """Select object in container"""
        if not name_ok(name):
            return False
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                self.__object_select = o
                history("selected " + self.object_name(name))
                return True
        error("failed to find " + self.object_name(name))
        self.name_list(history=True)
        return False

    def name_list(self, history=False):
        """Return list of object names"""
        nlist = []
        if self.__objects:
            for o in self.__objects:
                nlist.append(o.name)
        if history:
            print(nlist)
        return nlist

    def exists(self, name):
        """Check if object exists."""
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
            history("added " + self.object_name(obj.name))
        if select or not self.__object_select:
            self.__object_select = obj
            history("selected " + self.object_name(obj.name))
        return True

    def new(self, name, select=True):
        """
        Create a new object and add it to container.

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
            error(self.object_name(name) + " already exists")
            return None
        o = self.object_new(name)
        self.__objects.append(o)
        history("created " + self.object_name(name))
        if select or not self.__object_select:
            self.__object_select = o
            history("selected " + self.object_name(name))
        return o

    def name_next(self):
        """Create next default object name."""
        if not self.OBJECT_NAME_PREFIX:
            prefix = "None"
        else:
            prefix = self.OBJECT_NAME_PREFIX
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
            error(self.object_name(newname) + " already exists")
            return False
        o.name = newname
        history("renamed " + self.object_name(name) + " to " + quotes(newname))
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
            error(self.object_name(newname) + " already exists")
            return False
        c = copy.deepcopy(o)
        if c is not None:
            c.name = newname
            self.__objects.append(c)
            history("copied " + self.object_name(name) + " to " + quotes(newname))
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
        history("killed " + self.object_name(name))
        if selected and len(self.__objects) > 0:
            self.__object_select = self.__objects[0]
            history("selected " + self.object_name(self.__object_select.name))
        return True
