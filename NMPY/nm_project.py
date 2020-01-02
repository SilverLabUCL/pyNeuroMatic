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

    # override
    def _copy(self, project, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not super()._copy(project, copy_name=copy_name, quiet=True):
            return False
        c = project._Project__folder_container
        if not self.__folder_container._copy(c, quiet=True):
            return False
        h = ('copied Project ' + nmu.quotes(project.name) + ' to ' +
             nmu.quotes(name))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    # override
    def _equal(self, project, ignore_name=False, alert=False):
        if not super()._equal(project, ignore_name=ignore_name, alert=alert):
            return False
        c = project._Project__folder_container
        return self.__folder_container._equal(c, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
