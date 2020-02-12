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

    def __init__(self, parent, name, fxns={}, rename=True, **copy):
        super().__init__(parent, name, fxns=fxns, rename=rename)
        self.__folder_container = None
        for k, v in copy.items():
            if k.lower() == 'folders' and isinstance(v, FolderContainer):
                self.__folder_container = v
        if not isinstance(self.__folder_container, FolderContainer):
            self.__folder_container = FolderContainer(self, 'Folders',
                                                      fxns=fxns)

    # override, no super
    @property
    def content(self):
        k = {'project': self.name}
        k.update(self.__folder_container.content)
        return k

    # override, no super
    def copy(self):
        c = Project(self._parent, self.name, fxns=self._fxns,
                    rename=self._rename,
                    folders=self.__folder_container.copy())
        return c

    # override
    def _equal(self, project, alert=False):
        if not super()._equal(project, alert=alert):
            return False
        return self.__folder_container._equal(project.folder, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
