# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
import copy
import h5py
from nm_folder import Folder
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import name_list
from nm_utilities import exists
from nm_utilities import error
from nm_utilities import history

NM_FOLDER_PREFIX = "NMFolder"


class Experiment(Container):
    """
    NM Experiment class
    Container for NM Folders
    """
    def __init__(self, name):
        super().__init__(name)
        self.OBJECT_NAME_PREFIX = "NMFolder"
        self.new("")
        # self.new("")

    def object_new(self, name):
        return Folder(name)

    def instance_ok(self, obj):
        return isinstance(obj, Folder)


class ExperimentOLD(object):
    """
    NM Experiment class

    Attributes:
        folders (list): list of Folder objects
        folder (Folder): selected folder
    """

    def __init__(self, name="Untitled", default_folder=True):
        self.name = name
        self.__folders = []  # where folders are saved
        self.__folder = None  # selected folder
        if default_folder:
            self.folder_new()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not name_ok(name):
            error("bad experiment name")
            return False
        self.__name = name
        return True

    @property
    def folder(self):
        return self.__folder

    @folder.setter
    def folder(self, folder):
        if folder is not Folder:
            error("bad Folder object")
            return False
        self.__folder = folder
        history("selected folder " + quotes(folder.name))
        if self.__folders.count(folder) == 0:
            self.__folders.append(folder)  # save
        return True

    def folder_get(self, name):
        if not name_ok(name):
            error("bad folder name")
            return None
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                return f
        error("failed to find folder " + quotes(name))
        return None

    def folder_set(self, name):
        if not name_ok(name):
            error("bad folder name")
            return False
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                self.__folder = f
                history("selected folder " + quotes(f.name))
                return True
        error("failed to find folder " + quotes(name))
        return False

    def folder_list(self):
        """Get list of folder names"""
        return name_list(self.__folders)

    def folder_exists(self, name):
        """
        Check if folder exists.

        Args:
            name (str): folder name

        Returns:
            True if folder exists, False otherwise
        """
        return exists(self.__folders, name)

    def folder_add(self, folder, select=True):
        """
        Add a folder to Folders list.

        Args:
            folder: the folder to add
            select: select this folder

        Returns:
            True for success, False otherwise
        """
        if not folder:
            error("encountered null folder")
            return False
        if not name_ok(folder.name):
            error("bad folder name")
            return False
        if self.folder_exists(folder.name):
            pass  # nothing to do
        else:
            self.__folders.append(folder)
            history("added folder " + quotes(folder.name))
        if select or not self.__folder:
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
        if not name:
            name = self.folder_name_next()
        elif not name_ok(name):
            error("bad folder name")
            return None
        elif self.folder_exists(name):
            error("folder " + quotes(name) + " already exists")
            return None
        f = Folder(name=name)
        self.__folders.append(f)
        history("created folder " + quotes(name))
        if select or not self.__folder:
            self.__folder = f
            history("selected folder " + quotes(name))
        return f
    
    def folder_name_next(self):
        """
        Create next default folder name.

        Returns:
            folder name
        """
        n = 10 + len(self.__folders)
        for i in range(0, n):
            name = NM_FOLDER_PREFIX + str(i)
            if not self.folder_exists(name):
                return name
        return NM_FOLDER_PREFIX + "99999"
    
    def folder_rename(self, name, newname):
        if not name:
            f = self.__folder
        else:
            f = self.folder_get(name=name)
        if not f:
            return False
        if not name_ok(name=newname):
            error("bad folder name")
            return False
        if self.folder_exists(newname):
            error("folder " + quotes(newname) + " already exists")
            return False
        oldname = f.name
        f.name = newname
        history("renamed folder " + quotes(oldname) + " to " + quotes(newname))
        return True

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
        if not f:
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
        if not f:
            error("failed to find folder " + quotes(name))
            return False
        selected = f is self.__folder
        self.__folders.remove(f)
        history("killed folder " + quotes(name))
        if selected and len(self.__folders) > 0:
            self.__folder = self.__folders[0]
            history("selected folder " + quotes(self.__folder.name))
        return True
    
    def folder_open(self, path=""):
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

