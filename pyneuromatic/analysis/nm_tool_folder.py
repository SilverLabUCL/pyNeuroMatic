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

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_utilities as nmu


class NMToolFolder(NMObject):
    """
    NM Data Folder class
    """

    # Extend NMObject's special attrs with NMToolFolder's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMToolFolder__data_container",
        "_NMToolFolder__dataseries_container",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMToolFolder0",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__data_container: NMDataContainer = NMDataContainer(parent=self)
        self.__dataseries_container: NMDataContainer | None = None

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMToolFolder):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__data_container != other.data:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMToolFolder:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMToolFolder by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMToolFolder
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

        # Now handle NMToolFolder's special attributes

        # __data_container: deep copy and update parent
        if self._NMToolFolder__data_container is not None:
            result._NMToolFolder__data_container = copy.deepcopy(
                self._NMToolFolder__data_container, memo
            )
            result._NMToolFolder__data_container._parent = result
        else:
            result._NMToolFolder__data_container = NMDataContainer(parent=result)

        # __dataseries_container: deep copy and update parent
        if self._NMToolFolder__dataseries_container is not None:
            result._NMToolFolder__dataseries_container = copy.deepcopy(
                self._NMToolFolder__dataseries_container, memo
            )
            result._NMToolFolder__dataseries_container._parent = result
        else:
            result._NMToolFolder__dataseries_container = None

        return result

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        if self.__data_container is not None:
            k.update(self.__data_container.content)
        return k

    @property
    def data(self) -> NMDataContainer | None:
        return self.__data_container

    @property
    def dataseries(self) -> NMDataContainer | None:
        return self.__dataseries_container


class NMToolFolderContainer(NMObjectContainer):
    """
    Container of NMToolFolders
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMToolFolderContainer0",
        rename_on: bool = True,
        name_prefix: str = "toolfolder",
        name_seq_format: str = "0",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            auto_name_prefix=name_prefix,
            auto_name_seq_format=name_seq_format,
        )

    # override, no super
    def content_type(self) -> str:
        return NMToolFolder.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMToolFolder | None:
        name = self._newkey(name)
        f = NMToolFolder(parent=self, name=name)
        if super()._new(f, select=select):
            return f
        return None