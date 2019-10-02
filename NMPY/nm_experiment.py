# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
import h5py
from nm_folder import Folder
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import history


class Experiment(object):
    """
    NM Experiment class

    Class information here...

    Attributes:
        folders: list of Folder objects
        folder: selected Folder
    """

    def __init__(self,
                 name="Untitled",
                 default_folder = True):
        self.name = name
        self.__folders = []
        self.__folder = None
        if default_folder:
            self.folder_new()  # create empty default Folder

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name
            return True
        error("bad experiment name")
        return False

    @property
    def folder(self):
        return self.__folder

    @folder.setter
    def folder(self, folder):
        if folder is None:
            return False
        self.__folder = folder
        history("selected folder " + quotes(self.__folder.name))
        if not self.folder_exists(folder=folder):
            self.__folders.append(folder)  # save
        return True
    
    def folder_get(self, name: str) -> Folder:
        if not name_ok(name):
            error("bad folder name")
            return None
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                return f
        error("failed to find folder " + quotes(name))
        return None
    
    def folder_set(self, name: str) -> bool:
        if not name_ok(name):
            error("bad folder name")
            return False
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                self.__folder = f
                history("selected folder " + quotes(self.__folder.name))
                return True
        error("failed to find folder " + quotes(name))
        return False

    def folder_name_next(self) -> str:
        """
        Create next default folder name.

        Returns:
            folder name
        """
        n = 10 + len(self.__folders)
        for i in range(0, n):
            name = "NMFolder" + str(i)
            if not self.folder_exists(name=name):
                return name
        return "NMFolder99999"

    def folder_exists(self, 
                      folder: Folder = None, 
                      name: str = "") -> bool:
        """
        Check if folder resides in Folder list.
        Pass a folder object (priority) or folder name

        Args:
            folder: a Folder object
            name: name of folder

        Returns:
            True if folder exists, False otherwise
        """
        if folder is not None:
            for f in self.__folders:
                if f == folder:
                    return True
        elif name_ok(name):
            for f in self.__folders:
                if name.casefold() == f.name.casefold():
                    return True
        return False

    def folder_add(self,
                   folder: Folder,
                   select: bool = True) -> bool:
        """
        Add a folder to Folders list.

        Args:
            folder: the folder to add
            select: select this folder

        Returns:
            True for success, False otherwise
        """
        if folder is None:
            error("encountered null folder")
            return False
        if not name_ok(folder.name):
            error("bad folder name")
            return False
        if self.folder_exists(folder=folder):
            pass  # nothing to do
        else:
            self.__folders.append(folder)
            history("added folder " + quotes(folder.name))
        if select or self.__folder is None:
            self.__folder = folder
            history("selected folder " + quotes(folder.name))
        return True

    def folder_new(self,
                   name: str = "",
                   select: bool = True) -> Folder:
        """
        Create a new folder and add to Folders list.

        Args:
            name: name of new folder
            select: select this folder

        Returns:
            new folder if successful, None otherwise
        """
        if name is None or not name:
            name = self.folder_name_next()
        elif not name_ok(name):
            error("bad folder name")
            return None
        elif self.folder_exists(name=name):
            error("folder " + quotes(name) + " already exists")
            return None
        f = Folder(name=name)
        self.__folders.append(f)
        history("created folder " + quotes(name))
        if select or self.__folder is None:
            self.__folder = f
            history("selected folder " + quotes(name))
        return f

    def folder_copy(self,
                    name: str,
                    newname: str,
                    select: bool = False) -> Folder:
        """
        Copy an existing folder and add copy to Folders list.

        Args:
            name: name of folder to copy
            newname: name of new folder

        Returns:
            new folder if successful, None otherwise
        """
        f = self.folder_get(name=name)
        if f is None:
            error("failed to find folder " + quotes(name))
            return None
        c = copy.deepcopy(f)
        if c is not None:
            self.__folders.append(c)
            history("copied folder " + quotes(name) + " to " + quotes(newname))
        return c

    def folder_kill(self,
                    name: str) -> bool:
        """
        Kill a folder (i.e. remove from Folders list).

        Args:
            name: name of folder

        Returns:
            True for success, False otherwise
        """
        f = self.folder_get(name=name)
        if f is None:
            error("failed to find folder " + quotes(name))
            return False
        selected = f is self.__folder
        self.__folders.remove(f)
        history("killed folder " + quotes(name))
        if selected and len(self.__folders) > 0:
            self.__folder = self.__folders[0]
            history("selected folder " + quotes(self.__folder.name))
        return True
    
    def folder_open(self, path: str) -> Folder:
        pass

    def folder_open_hdf5(self):
        wave_prefix = "Record"
        with h5py.File('nmFolder0.hdf5', 'r') as f:
            #print(f.keys())
            data = []
            for k in f.keys():
                if k[0:len(wave_prefix)] == wave_prefix:
                    print(k)
            # for name in f:
                # print(name)
            d = f['RecordA0']

            for i in d.attrs.keys():
                print(i)
            # cannot get access to attribute values for keys:
            # probably need to update h5py to v 2.10
            #IGORWaveNote
            #IGORWaveType
            #print(d.attrs.__getitem__('IGORWaveNote'))
            #for a in d.attrs:
                #print(item + ":", d.attrs[item])
                #print(item + ":", d.attrs.get(item))
                #print(a.shape)
            #for k in a.keys():
                #print(k)
            #print(a)
            #pf = f['NMPrefix_Record']
            #print(pf)

