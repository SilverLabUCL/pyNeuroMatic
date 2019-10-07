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
    Container (i.e. list) for objects, e.g. Experiment, Folder,
    WavePrefix, Channel...
    Each object stored in container must have a unique name.
    The name can start with the same prefix, but this is optional.
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self.__objects = []
        self.__object_select = None
        
    def object_new(self, name):
        return Container(name)
    
    def instance_ok(self, obj):
        return isinstance(obj, Container)

    def __object_type(self):
        if not self.__objects:
            return "Object"
        return self.__objects[0].__class__.__name__

    def __object_name(self, name):
        return self.__object_type() + " " + quotes(name)

    @property
    def prefix(self):
        """Get object prefix name"""
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        """Set object prefix name"""
        if not name_ok(prefix):
            return False
        self.__prefix = prefix
        return True
    
    @property
    def name(self):
        if self.__object_select:
            return self.__object_select.name  # name of selected object
        return "None"

    @name.setter
    def name(self, name):
        if self.__object_select and name_ok(name):
            self.__object_select.name = name  # set name of selected object
            return True
        return False

    def get(self, name):
        """Get object from container"""
        if not name or name.casefold() == "SELECTED".casefold():
            return self.__object_select
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                return o
        error("failed to find " + self.__object_name(name))
        self.name_list(history=True)
        return None

    def select(self, name):
        """Select object in container"""
        if not name_ok(name):
            return False
        for o in self.__objects:
            if name.casefold() == o.name.casefold():
                self.__object_select = o
                history("selected " + self.__object_name(name))
                return True
        error("failed to find " + self.__object_name(name))
        self.name_list(history=True)
        return False

    def names(self, history=False):
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
            history("added " + self.__object_name(obj.name))
        if select or not self.__object_select:
            self.__object_select = obj
            history("selected " + self.__object_name(obj.name))
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
            error(self.__object_name(name) + " already exists")
            return None
        o = self.object_new(name)
        self.__objects.append(o)
        history("created " + self.__object_name(name))
        if select or not self.__object_select:
            self.__object_select = o
            history("selected " + self.__object_name(name))
        return o

    def name_next(self):
        """Create next default object name."""
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
            error(self.__object_name(newname) + " already exists")
            return False
        o.name = newname
        history("renamed " + self.__object_name(name) + " to " + quotes(newname))
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
            error(self.__object_name(newname) + " already exists")
            return False
        c = copy.deepcopy(o)
        if c is not None:
            c.name = newname
            self.__objects.append(c)
            history("copied " + self.__object_name(name) + " to " + quotes(newname))
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
        history("killed " + self.__object_name(name))
        if selected and len(self.__objects) > 0:
            self.__object_select = self.__objects[0]
            history("selected " + self.__object_name(self.__object_select.name))
        return True


class ContainerChannel(Container):
    """
    Container for NM Channels
    """

    def object_new(self, name):
        return Channel(name)

    def instance_ok(self, obj):
        return isinstance(obj, Channel)


class ContainerWavePrefix(Container):
    """
    Container for NM WavePrefixes
    """

    def object_new(self, name):
        return WavePrefix(name)

    def instance_ok(self, obj):
        return isinstance(obj, WavePrefix)


class ContainerFolder(Container):
    """
    Container for NM Folders
    """

    def object_new(self, name):
        return Folder(name)

    def instance_ok(self, obj):
        return isinstance(obj, Folder)


class ContainerExperiment(Container):
    """
    Container for NM Experimnents
    """
    
    def object_new(self, name):
        return Experiment(name)

    def instance_ok(self, obj):
        return isinstance(obj, Experiment)


class Channel(object):
    """
    NM Channel class
    """
    
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name


class WavePrefix(object):
    """
    NM WavePrefix class
    """
    
    def __init__(self, name):
        self.__name = name
        self.channel = ContainerChannel("Chan")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name


class Folder(object):
    """
    NM Folder class
    Holds wave prefixes
    """
    
    def __init__(self, name):
        self.__name = name
        self.waveprefix = ContainerWavePrefix("")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name

    
class Experiment(object):
    """
    NM Experiment class
    """
    
    def __init__(self, name):
        self.__name = name
        self.folder = ContainerFolder("NMFolder")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name


class Project(object):
    """
    NM Project class
    """
    
    def __init__(self, name):
        self.__name = name
        self.experiment = ContainerExperiment("NMExp")

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name
