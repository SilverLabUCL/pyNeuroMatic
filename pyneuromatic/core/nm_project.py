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
import copy

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

    # Extend NMObject's special attrs with NMProject's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMProject__folder_container",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMProject0",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__folder_container: NMFolderContainer = NMFolderContainer(parent=self)

    # override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMProject):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return self.folders == other.folders

    def __deepcopy__(self, memo: dict) -> NMProject:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMProject by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMProject
        """
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._NMObject__notes_on = self._NMObject__notes_on
        result._NMObject__notes = copy.deepcopy(self._NMObject__notes, memo)
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMProject's special attributes

        # __folder_container: deep copy and update parent
        if self._NMProject__folder_container is not None:
            result._NMProject__folder_container = copy.deepcopy(
                self._NMProject__folder_container, memo
            )
            result._NMProject__folder_container._parent = result
        else:
            result._NMProject__folder_container = NMFolderContainer(parent=result)

        return result

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
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            auto_name_prefix=name_prefix,
            auto_name_seq_format=name_seq_format,
        )  # NMObjectContainer

    # override, no super
    def content_type(self) -> str:
        return NMProject.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMProject | None:
        actual_name = self._newkey(name)
        p = NMProject(parent=self, name=actual_name)
        if super()._new(p, select=select):
            return p
        return None