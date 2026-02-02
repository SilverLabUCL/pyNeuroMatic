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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_data import NMData
    from pyneuromatic.core.nm_folder import NMFolder

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProject (project0)
        NMFolderContainer
            NMFolder (folder0, folder1...)
                NMDataContainer
                    NMData (recordA0, recordA1... avgA0, avgB0)
                NMDataSeriesContainer
                    NMDataSeries (record, avg...)
                        NMChannelContainer
                            NMChannel (A, B, C...)
                        NMEpochContainer
                            NMEpoch (E0, E1, E2...)
DataSeries:
      E0  E1  E2... (epochs)
Ch A  A0  A1  A2...
Ch B  B0  B1  B2...
.
.
.
"""


class NMEpoch(NMObject):
    """
    NM Epoch class
    """

    # Extend NMObject's special attrs with NMEpoch's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMEpoch__thedata",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMEpoch0",
        number: int = -1,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__thedata: list[NMData] = []  # list of NMData references

        if not isinstance(number, int):
            e = nmu.type_error_str(number, "number", "int")
            raise TypeError(e)

        self.__number: int = number

    # override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMEpoch):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__number != other.number:
            return False
        if len(self.__thedata) != len(other.data):
            return False
        if not all(a == b for a, b in zip(self.__thedata, other.data)):
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMEpoch:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMEpoch by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMEpoch
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
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMEpoch's special attributes

        # __thedata: copy list of references
        # If copying within a folder context, try to resolve to copied NMData
        if result._folder is not None:
            from pyneuromatic.core.nm_data import NMData

            result._NMEpoch__thedata = []
            data_container = result._folder.data
            for d in self._NMEpoch__thedata:
                # Check if this NMData was already copied (in memo)
                if id(d) in memo:
                    result._NMEpoch__thedata.append(memo[id(d)])
                else:
                    # Try to find by name in the folder's data container
                    o = data_container.get(d.name)
                    if isinstance(o, NMData):
                        result._NMEpoch__thedata.append(o)
        else:
            # Direct copy: just copy the list of references
            result._NMEpoch__thedata = list(self._NMEpoch__thedata)

        return result

    @property
    def number(self) -> int:
        return self.__number

    @number.setter
    def number(self, integer: int) -> None:
        if isinstance(integer, int):
            self.__number = integer
        return None

    # override
    # @property
    # def parameters(self) -> Dict[str, object]:
    #     k = super().parameters
    #     return k

    @property
    def data(self) -> list[NMData]:
        return self.__thedata


class NMEpochContainer(NMObjectContainer):
    """
    Container of NMEpochs
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMEpochContainer0",
        rename_on: bool = False,
        name_prefix: str = "E",
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
        return NMEpoch.__name__

    # override
    def new(
        self,
        name: str | None = None,  # not used, instead name = name_next()
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMEpoch | None:
        actual_name = self.auto_name_next()
        istr = actual_name.replace(self.auto_name_prefix, "")
        if str.isdigit(istr):
            iseq = int(istr)
        else:
            iseq = -1
        c = NMEpoch(parent=self._parent, name=actual_name, number=iseq)
        if super()._new(c, select=select):
            return c
        return None