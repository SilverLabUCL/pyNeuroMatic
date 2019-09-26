# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
import h5py
from nm_file import File
from nm_utilities import quotes
from nm_utilities import removeSpecialChars


class Experiment(object):
    """
    NM Experiment class
    
    Class information here...
    
    Attributes:
        files: list of File objects
        file: selected File
    """

    def __init__(self,
                 name=None):
        if name is None or len(name) == 0:
            name = "NMExp0"
        self.name = removeSpecialChars(name)
        self.files = []
        self.file = None
        self.file_new(name="", select=True)  # create empty default File

    def file_name_next(self) -> str:
        """
        Create next default name for a File.

        Returns:
            string name if successful, None otherwise
        """
        n = 10 + len(self.files)
        for i in range(0, n):
            name = "NMFile" + str(i)
            found = False
            for f in self.files:
                if name.casefold() == f.name.casefold():
                    found = True
                    break
            if not found:
                return name
        return None

    def file_name_check(self, 
                        name: str, 
                        exists: bool = False, 
                        notexists: bool = False) -> str:
        """
        Check if str name is OK to use for a File. It must be unique.
        This function removes all special characters except '_'.

        Args:
            name: name to check
            exists: check if name already exists as a File
            notexists: check if name does not exist as a File

        Returns:
            string name if successful, None otherwise
        """
        if name is None:
            print("bad File name: None")
            return None
        if len(name) == 0:
            print("bad File name: 0 length")
            return None
        name = removeSpecialChars(name)
        if len(name) == 0:
            print("bad File name: 0 length")
            return None
        if exists:
            for f in self.files:
                if name.casefold() == f.name.casefold():
                    print("File " + quotes(name) + " already exists")
                    return None
        if notexists:
            found = False
            for f in self.files:
                if name.casefold() == f.name.casefold():
                    found = True
                    break
            if not found:
                print("File " + quotes(name) + " does not exist")
                return None
        return name  # name is OK

    def file_add(self, 
                 file: File, 
                 select: bool = True) -> bool:
        """
        Add a File to Files list.

        Args:
            file: the File to add
            select: select this File

        Returns:
            True for success, False otherwise
        """
        if file is None:
            return False
        self.files.append(file)
        print("added File " + quotes(file.name) + " to current experiment")
        if self.file is None or select:
            self.file = file
            print("selected File " + quotes(file.name))
        return True
    
    def file_new(self, 
                 name: str, 
                 select: bool = True) -> File:
        """
        Create a new File and add to Files list.

        Args:
            name: name of new File
            select: select this File

        Returns:
            new File if successful, None otherwise
        """
        if name is None or len(name) == 0:
            name = self.file_name_next()
        name = self.file_name_check(name=name, exists=True)
        if name is None:
            return None
        f = File(name=name)
        self.files.append(f)
        print("created File " + quotes(name))
        if self.file is None or select:
            self.file = f
            print("selected File " + quotes(name))
        return f
        
    def file_copy(self,
                  name: str, 
                  newname: str,
                  select: bool = False) -> File:
        """
        Copy an existing File and add to Files list.

        Args:
            name: name of File to copy
            newname: name of new File

        Returns:
            new File if successful, None otherwise
        """
        name = self.file_name_check(name=name, notexists=True)
        if name is None or len(name) == 0:
            return False
        toCopy = None
        for f in self.files:
            if name.casefold() == f.name.casefold():
                toCopy = f
                break
        if toCopy is None:
            return False
        f = copy.deepcopy(toCopy)
        self.files.append(f)
        print("copied File " + quotes(name) + " to " + quotes(newname))
        return True
        
    def file_kill(self,
                  name: str) -> bool:
        """
        Kill a File (i.e. remove from Files list).

        Args:
            name: name of File

        Returns:
            True for success, False otherwise
        """
        name = self.file_name_check(name=name, notexists=True)
        if name is None or len(name) == 0:
            return False
        kill = None
        for f in self.files:
            if name.casefold() == f.name.casefold():
                kill = f
                break
        if kill is None:
            return False
        selected = kill is self.file
        self.files.remove(kill)
        if selected:
            if len(self.files) == 0:
                self.file = None
            else:
                self.file = self.files[0]
        print("killed File " + quotes(name))
        return True

    def file_select(self,
                    name: str) -> bool:
        """
        Select File.

        Args:
            name: name of File

        Returns:
            True for success, False otherwise
        """
        name = self.file_name_check(name=name, notexists=True)
        if name is None:
            return False
        for f in self.files:
            if name.casefold() == f.name.casefold():
                self.file = f
                return True
        return False
    
    def file_open_hdf5(self):
        f = h5py.File('nmFolder0.hdf5', 'r')
        print(f.keys())

