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

    def __init__(self, parent, name, fxns={}, folder_container=None):
        super().__init__(parent, name, fxns=fxns)
        if folder_container is None:
            self.__folder_container = FolderContainer(self, 'Folders',
                                                      fxns=fxns)
        elif isinstance(folder_container, FolderContainer):
            self.__folder_container = folder_container
        else:
            e = nmu.type_error(folder_container, 'FolderContainer')
            raise TypeError(e)

    # override, no super
    @property
    def content(self):
        k = {'project': self.name}
        k.update(self.__folder_container.content)
        return k

    # override, no super
    def copy(self):
        return Project(self._parent, self.name, fxns=self._fxns,
                       folder_container=self.__folder_container.copy())

    # override
    def _equal(self, project, ignore_name=False, alert=False):
        if not super()._equal(project, ignore_name=ignore_name, alert=alert):
            return False
        return self.__folder_container._equal(project.folder, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
