#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:42:59 2023

@author: jason
"""
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_utilities as nmu
from typing import List, Union

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
        parent: Union[object, None] = None,
        name: str = "NMEpoch",
        number: int = -1,
        copy: Union[nmu.NMEpochType, None] = None,
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__thedata = []  # list of NMData references
        # self.__number = -1

        if copy is None:
            pass
        elif isinstance(copy, NMEpoch):
            if self._folder is None:
                # direct copy
                self.__thedata = list(copy.data)
                number = copy.number
            else:
                # grab NMData refs from copied NMDataContainer
                # NMDataContainer should be copied before this copy
                data = self._folder.data
                for d in copy.data:
                    o = data.get(d.name)
                    self.__thedata.append(o)
        else:
            e = nmu.typeerror(copy, "copy", "NMEpoch")
            raise TypeError(e)

        if not isinstance(number, int):
            e = nmu.typeerror(number, "number", "int")
            raise TypeError(e)

        self.__number = number

        return None

    # override
    def __eq__(self, other: nmu.NMEpochType) -> bool:
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
    def copy(self) -> nmu.NMEpochType:
        return NMEpoch(copy=self)

    @property
    def number(self) -> int:
        return self.__number

    @number.setter
    def number(self, integer: int) -> None:
        if isinstance(integer, int):
            self.__number = integer

    # override
    # @property
    # def parameters(self) -> Dict[str, object]:
    #     k = super().parameters
    #     return k

    @property
    def data(self) -> List[nmu.NMDataType]:
        return self.__thedata


class NMEpochContainer(NMObjectContainer):
    """
    Container of NMEpochs
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: str = "NMEpochContainer",
        rename_on: bool = False,
        name_prefix: str = "E",
        name_seq_format: str = "0",
        copy: Union[nmu.NMEpochContainerType, None] = None,
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
    def copy(self) -> nmu.NMEpochContainerType:
        return NMEpochContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMEpoch.__name__

    # override
    def new(
        self,
        # name: str = 'A',  use name_next()
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMEpochType:
        name = self.name_next()
        istr = name.replace(self.name_prefix, "")
        if str.isdigit(istr):
            iseq = int(istr)
        else:
            iseq = -1
        c = NMEpoch(parent=self._parent, name=name, number=iseq)
        super().new(c, select=select)
        return c
