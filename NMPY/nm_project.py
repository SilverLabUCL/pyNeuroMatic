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

    def __init__(self, parent, name, fxns={}):
        super().__init__(parent, name, fxns=fxns)
        self.__folder_container = FolderContainer(self, 'Folders', fxns=fxns)

    # override, no super
    @property
    def content(self):
        k = {'project': self.name}
        k.update(self.__folder_container.content)
        return k

    # override, no super
    def copy(self):
        p = Project(self._parent, self.name, fxns=self._fxns)
        p.folder = self.__folder_container.copy()
        return p

    # override
    def _equal(self, project, ignore_name=False, alert=False):
        if not super()._equal(project, ignore_name=ignore_name, alert=alert):
            return False
        return self.__folder_container._equal(project.folder, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
