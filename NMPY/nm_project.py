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

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        f = FolderContainer(manager, self, 'NMFolderContainer', fxns)
        self.__folder_container = f

    @property  # override, no super
    def content(self):
        k = {'project': self.name}
        k.update(self.__folder_container.content)
        return k

    @property
    def folder(self):
        return self.__folder_container
