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

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_utilities as nmu


class NMToolFolder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMToolFolder0",
        copy: NMToolFolder | None = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__data_container = None  # save results to NumPy arrays

        if copy is None:
            pass
        elif isinstance(copy, NMToolFolder):
            self.__data_container = copy.data.copy()
            self.__dataseries_container = copy.dataseries.copy()
        else:
            e = nmu.typeerror(copy, "copy", NMToolFolder)
            raise TypeError(e)

        if not isinstance(self.__data_container, NMDataContainer):
            self.__data_container = NMDataContainer(parent=self)

    # override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if self.__data_container != other._NMToolFolder__data_container:
            return False

    # override, no super
    def copy(self) -> NMToolFolder:
        return NMToolFolder(copy=self)

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        k.update(self.__data_container.content)
        return k

    @property
    def data(self):
        return self.__data_container


class NMToolFolderContainer(NMObjectContainer):
    """
    Container of NMToolFolders
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMToolFolderContainer0",
        rename_on: bool = True,
        name_prefix: str = "toolfolder",
        name_seq_format: str = "0",
        copy: NMToolFolderContainer = None,  # see copy()
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )

    # override, no super
    def copy(self) -> NMToolFolderContainer:
        return NMToolFolderContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMToolFolder.__name__

    # override
    def new(
        self,
        name: str = "default",
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMToolFolder:
        name = self._newkey(name)
        f = NMToolFolder(parent=self, name=name)
        super()._new(f, select=select)
        return f
