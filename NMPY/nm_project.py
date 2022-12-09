# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject, NMobject
from nm_folder import NMFolderContainer, NMfolderContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, NewType

NMproject = NewType('NMProject', NMobject)


class NMProject(NMObject):
    """
    NM Project class
    TODO: history functions
    TODO: project container?
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMProject',
        copy: NMproject = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__folder_container = None
        # TODO: update copy
        for k, v in copy.items():
            if k.lower() == 'c_folders' and isinstance(v, NMFolderContainer):
                self.__folder_container = v
        if not isinstance(self.__folder_container, NMFolderContainer):
            self.__folder_container = NMFolderContainer(self, 'Folders')

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__folder_container.content)
        return k

    # override, no super
    def copy(self) -> NMproject:
        c = NMProject(copy=self)
        c.note = 'this is a copy of ' + str(self)
        return c

    # override
    def _isequivalent(
        self,
        project: NMproject,
        alert: bool = False
    ) -> bool:
        if not super()._isequivalent(project, alert=alert):
            return False
        c = self.__folder_container
        if c and not c._isequivalent(project.folder, alert=alert):
            return False
        return True

    @property
    def folder(self) -> NMfolderContainer:
        return self.__folder_container
