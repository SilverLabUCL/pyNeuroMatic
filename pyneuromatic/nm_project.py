# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from typing import Dict, Union

from pyneuromatic.nm_folder import NMFolderContainer
from pyneuromatic.nm_object import NMObject
from pyneuromatic.nm_object_container import NMObjectContainer
import pyneuromatic.nm_utilities as nmu

"""
NM class tree:

NMManager
    NMProjectContainer
        NMProject (project0, project1...)
            NMFolderContainer
                NMFolder (folder0, folder1...)
"""


class NMProject(NMObject):
    """
    NM Project class
    TODO: history functions
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: str = 'NMProject',
        copy: Union[nmu.NMProjectType, None] = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__folder_container = None

        if copy is None:
            pass
        elif isinstance(copy, NMProject):
            self.__folder_container = copy.folders.copy()
            self.__folder_container._parent = self
        else:
            e = nmu.typeerror(copy, 'copy', 'NMProject')
            raise TypeError(e)

        if not isinstance(self.__folder_container, NMFolderContainer):
            self.__folder_container = NMFolderContainer(parent=self)

        return None

    # override
    def __eq__(
        self,
        other: nmu.NMProjectType
    ) -> bool:
        if not super().__eq__(other):
            return False
        return self.folders == other.folders

    # override, no super
    def copy(self) -> nmu.NMProjectType:
        return NMProject(copy=self)

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__folder_container.content)
        return k

    @property
    def folders(self) -> nmu.NMFolderContainerType:
        return self.__folder_container


class NMProjectContainer(NMObjectContainer):
    """
    Container of NMProjects
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: str = 'NMProjectContainer',
        rename_on: bool = True,
        name_prefix: str = 'project',
        name_seq_format: str = '0',
        copy: Union[nmu.NMProjectContainerType, None] = None  # see copy()
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy
        )  # NMObjectContainer

    # override, no super
    def copy(self) -> nmu.NMProjectContainerType:
        return NMProjectContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMProject.__name__

    # override
    def new(
        self,
        name: str = 'default',
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMProjectType:
        name = self._newkey(name)
        p = NMProject(parent=self, name=name)
        super().new(p, select=select)
        return p
