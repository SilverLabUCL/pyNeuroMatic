# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_folder import FolderContainer
import nm_preferences as nmp
import nm_utilities as nmu


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name, fxns={}, **copy):
        super().__init__(parent, name, fxns=fxns)
        self._content_name = 'project'
        self.__folder_container = None
        for k, v in copy.items():
            if k.lower() == 'folders' and isinstance(v, FolderContainer):
                self.__folder_container = v
        if not isinstance(self.__folder_container, FolderContainer):
            self.__folder_container = FolderContainer(self, 'Folders',
                                                      fxns=fxns)

    # override
    @property
    def content(self):
        k = super().content
        k.update(self.__folder_container.content)
        return k

    # override, no super
    def copy(self):
        return Project(self._parent, self.name, fxns=self._fxns,
                       folders=self.__folder_container.copy())

    # override
    def _equal(self, project, alert=False):
        if not super()._equal(project, alert=alert):
            return False
        return self.__folder_container._equal(project.folder, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
