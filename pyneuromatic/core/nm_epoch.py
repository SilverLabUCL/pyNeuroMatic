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

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
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

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMEpoch0",
        number: int = -1,
        copy: NMEpoch | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__thedata: list[NMData] = []  # list of NMData references
        self.__number: int = -1

        if copy is None:
            pass
        elif isinstance(copy, NMEpoch):
            if isinstance(self._folder, NMFolder):
                # grab NMData refs from copied NMDataContainer
                # NMDataContainer should be copied before this copy
                data = self._folder.data
                for d in copy.data:
                    if isinstance(d, NMData):
                        o = data.get(d.name)
                        if isinstance(o, NMData):
                            self.__thedata.append(o)
                        else:
                            e = nmu.valueerror(
                                d.name,
                                "data item name not found in copied NMDataContainer",
                            )
                            raise ValueError(e)
                    else:
                        e = nmu.typeerror(d, "copy.data.item", "NMData")
                        raise TypeError(e)
            else:
                # direct copy
                for d in copy.data:
                    if isinstance(d, NMData):
                        self.__thedata.append(d)
                    else:
                        e = nmu.typeerror(d, "copy.data.item", "NMData")
                        raise TypeError(e)
            number = copy.number
        else:
            e = nmu.typeerror(copy, "copy", NMEpoch)
            raise TypeError(e)

        if not isinstance(number, int):
            e = nmu.typeerror(number, "number", "int")
            raise TypeError(e)

        self.__number = number

        return None

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

    # override, no super
    def copy(self) -> NMEpoch:
        return NMEpoch(copy=self)

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
        copy: NMEpochContainer | None = None,
    ) -> None:
        return super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            name_prefix=name_prefix,
            name_seq_format=name_seq_format,
            copy=copy,
        )

    # override, no super
    def copy(self) -> NMEpochContainer:
        return NMEpochContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMEpoch.__name__

    # override
    def new(
        self,
        name: str = "default",  # not used, instead name = name_next()
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMEpoch | None:
        name = self.name_next()
        istr = name.replace(self.name_prefix, "")
        if str.isdigit(istr):
            iseq = int(istr)
        else:
            iseq = -1
        c = NMEpoch(parent=self._parent, name=name, number=iseq)
        if super()._new(c, select=select):
            return c
        return None