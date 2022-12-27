# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_object import NMObject
from nm_folder import NMFolderContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List


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
        copy: nmu.NMProjectType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__folder_container = None

        if isinstance(copy, NMProject):
            self.__folder_container = copy.folder._container_copy()

        if not isinstance(self.__folder_container, NMFolderContainer):
            self.__folder_container = NMFolderContainer(parent=self,
                                                        name='Folders')

    # override
    def __eq__(
        self,
        other: nmu.NMProjectType
    ) -> bool:
        if not super().__eq__(other):
            return False
        if self.__folder_container != other._NMProject__folder_container:
            return False
        return True

    # override, no super
    def copy(self) -> nmu.NMProjectType:
        return NMProject(copy=self)

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__folder_container.content)
        return k

    # override
    def _isequivalent(
        self,
        project: nmu.NMProjectType,
        alert: bool = False
    ) -> bool:
        if not super()._isequivalent(project, alert=alert):
            return False
        c = self.__folder_container
        if c and not c._isequivalent(project.folder, alert=alert):
            return False
        return True

    @property
    def folder(self) -> nmu.NMFolderContainerType:
        return self.__folder_container
