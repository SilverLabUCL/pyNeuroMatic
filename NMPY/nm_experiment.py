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


class Experiment(object):
    """
    NM Experiment class

    Class information here...

    Attributes:
        folders: list of Folder objects
        folder: selected Folder
    """

    def __init__(self,
                 name="Untitled"):
        self.name = name
        self.__folders = []
        self.__folder_select = None
        self.__folder_name_select = None
        self.folder_new()  # create empty default Folder

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name
            return True
        print("bad experiment name")
        return False

    @property
    def folder_select(self):
        return self.__folder_select

    @folder_select.setter
    def folder_select(self, folder):
        if folder is None:
            return False
        self.__folder_select = folder
        self.__folder_name_select = folder.name
        if not self.folder_exists(folder):
            self.__folders.append(folder)  # save to folder array
        return True

    @property
    def folder_name_select(self):
        return self.__folder_name_select

    @folder_name_select.setter
    def folder_name_select(self, folder_name):
        folder_name = self.folder_name_check(name=folder_name, notexists=True)
        if folder_name is None or not folder_name:
            return False
        for f in self.__folders:
            if folder_name.casefold() == f.name.casefold():
                self.__folder_select = f
                return True
        return False

    def folder_name_next(self) -> str:
        """
        Create next default name for a Folder.

        Returns:
            string name if successful, None otherwise
        """
        n = 10 + len(self.__folders)
        for i in range(0, n):
            name = "NMFolder" + str(i)
            if not self.folder_name_exists(name):
                return name
        return "NMFolder99999"

    def folder_name_check(self,
                        name: str,
                        exists: bool = False,
                        notexists: bool = False) -> str:
        """
        Check if str name is OK to use for a Folder. It must be unique.
        This function removes all special characters except '_'.

        Args:
            name: name to check
            exists: check if name already exists as a Folder
            notexists: check if name does not exist as a Folder

        Returns:
            string name if successful, None otherwise
        """
        if name is None or not name:
            print("bad Folder name")
            return None
        # name = removeSpecialChars(name)
        # if not name:
            # print("bad Folder name: 0 length")
            # return None
        if exists:
            for f in self.__folders:
                if name.casefold() == f.name.casefold():
                    print("folder " + quotes(name) + " already exists")
                    return None
        if notexists:
            found = False
            for f in self.__folders:
                if name.casefold() == f.name.casefold():
                    found = True
                    break
            if not found:
                print("folder " + quotes(name) + " does not exist")
                return None
        return name  # name is OK

    def folder_exists(self, folder):
        for f in self.__folders:
            if f == folder:
                return True
        return False

    def folder_name_exists(self, name):
        """
        Check if name already exists as a folder name.

        Args:
            name: name to check

        Returns:
            True if name already exists, False otherwise
        """
        if name is None or not name:
            return False
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                return True
        return False

    def folder_add(self,
                   folder: Folder,
                   select: bool = True) -> bool:
        """
        Add a Folder to Folders list.

        Args:
            folder: the Folder to add
            select: select this Folder

        Returns:
            True for success, False otherwise
        """
        if folder is None:
            return False
        self.__folders.append(folder)
        print("added folder " + quotes(folder.name))
        if select or self.folder_select is None:
            self.folder_select = folder
            print("selected folder " + quotes(folder.name))
        return True

    def folder_new(self,
                   name: str = None,
                   select: bool = True) -> Folder:
        """
        Create a new Folder and add to Folders list.

        Args:
            name: name of new Folder
            select: select this Folder

        Returns:
            new Folder if successful, None otherwise
        """
        if name is None or not name:
            name = self.folder_name_next()
        elif self.folder_name_exists(name=name):
            print("folder name " + quotes(name) + " already exists")
            return None
        f = Folder(name=name)
        self.__folders.append(f)
        print("created folder " + quotes(name))
        if select or self.folder_select is None:
            self.folder_select = f
            print("selected folder " + quotes(name))
        return f

    def folder_copy(self,
                    name: str,
                    newname: str,
                    select: bool = False) -> Folder:
        """
        Copy an existing Folder and add to Folders list.

        Args:
            name: name of Folder to copy
            newname: name of new Folder

        Returns:
            new Folder if successful, None otherwise
        """
        name = self.folder_name_check(name=name, notexists=True)
        if name is None or not name:
            return False
        toCopy = None
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                toCopy = f
                break
        if toCopy is None:
            return False
        f = copy.deepcopy(toCopy)
        self.__folders.append(f)
        print("copied Folder " + quotes(name) + " to " + quotes(newname))
        return True

    def folder_kill(self,
                    name: str) -> bool:
        """
        Kill a Folder (i.e. remove from Folders list).

        Args:
            name: name of Folder

        Returns:
            True for success, False otherwise
        """
        if name is None or not name:
            return False
        kill = None
        for f in self.__folders:
            if name.casefold() == f.name.casefold():
                kill = f
                break
        if kill is None:
            return False
        selected = kill is self.folder_select
        self.__folders.remove(kill)
        if selected:
            if not self.__folders:
                self.folder_select = None
            else:
                self.folder_select = self.__folders[0]
        print("killed Folder " + quotes(name))
        return True

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

