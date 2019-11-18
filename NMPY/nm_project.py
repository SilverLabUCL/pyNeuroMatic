# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_folder import FolderContainer
import nm_utilities as nmu


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name, {'project': name})
        self.__folder_container = FolderContainer(self, "NMFolders")

    def rename(self, name, quiet=False):
        if not nmu.name_ok(name):
            return nmu.error('bad name ' + nmu.quotes(name), quiet=quiet)
        self.__name = name
        return True

    @property
    def folder_container(self):
        return self.__folder_container

    @property
    def folder_names(self):
        if self.__folder_container:
            return self.__folder_container.names
        return []

    @property
    def content(self):
        c = self.key_tree
        c.update(self.__folder_container.key)
        return c
