# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_folder import FolderContainer


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__folder_container = FolderContainer(self)

    @property
    def key(self):
        return {'project': self.name}

    @property
    def folder(self):
        return self.__folder_container

    @property
    def content(self):
        c = self.key_tree
        c.update(self.__folder_container.key)
        return c
