# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_folder import FolderContainer
import nm_utilities as nmu


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name, fxns):
        super().__init__(parent, name, fxns)
        f = FolderContainer(self, 'Folders', fxns)
        self.__folder_container = f

    @property
    def __history(self):
        return self._NMObject__history

    @property
    def __tp(self):
        return self.tree_path(history=True)

    # override, no super
    @property
    def content(self):
        k = {'project': self.name}
        k.update(self.__folder_container.content)
        return k

    # override
    def copy(self, project, copy_name=True, quiet=nmc.QUIET):
        name = self.name
        if not super().copy(project, copy_name=copy_name, quiet=True):
            return False
        c = project._Project__folder_container
        if not self.__folder_container.copy(c, quiet=quiet):
            return False
        h = ('copied Project ' + nmu.quotes(project.name) + ' to ' +
             nmu.quotes(name))
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

    # override
    def equal(self, project, ignore_name=False, alert=False):
        if not super().equal(project, ignore_name=ignore_name, alert=alert):
            return False
        c = project._Project__folder_container
        return self.__folder_container.equal(c, alert=alert)

    @property
    def folder(self):
        return self.__folder_container
