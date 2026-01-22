# -*- coding: utf-8 -*-
"""
[Module description].

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source 
Software Toolkit for Acquisition, Analysis and Simulation of 
Electrophysiological Data. Front. Neuroinform. 12:14. 
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

from pyneuromatic.core.nm_folder import NMFolderContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_utilities as nmu

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
        parent: object | None = None,
        name: str = "NMProject0",
        copy: NMProject | None = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__folder_container = None

        if copy is None:
            pass
        elif isinstance(copy, NMProject):
            if isinstance(copy.folders, NMFolderContainer):
                self.__folder_container = copy.folders.copy()
                self.__folder_container._parent = self
        else:
            e = nmu.typeerror(copy, "copy", NMProject)
            raise TypeError(e)

        if not isinstance(self.__folder_container, NMFolderContainer):
            self.__folder_container = NMFolderContainer(parent=self)

        return None

    # override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMProject):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return self.folders == other.folders

    # override, no super
    def copy(self) -> NMProject:
        return NMProject(copy=self)

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        if self.__folder_container is not None:
            k.update(self.__folder_container.content)
        return k

    @property
    def folders(self) -> NMFolderContainer | None:
        return self.__folder_container


class NMProjectContainer(NMObjectContainer):
    """
    Container of NMProjects
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMProjectContainer0",
        rename_on: bool = True,
        name_prefix: str = "project",
        name_seq_format: str = "0",
        copy: NMProjectContainer | None = None,  # see copy()
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )  # NMObjectContainer

    # override, no super
    def copy(self) -> NMProjectContainer:
        return NMProjectContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMProject.__name__

    # override
    def new(
        self,
        name: str = "default",
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMProject:
        name = self._newkey(name)
        p = NMProject(parent=self, name=name)
        super()._new(p, select=select)
        return p
