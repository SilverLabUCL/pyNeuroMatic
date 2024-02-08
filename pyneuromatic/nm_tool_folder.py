# -*- coding: utf-8 -*-
from typing import Dict, Union

from pyneuromatic.nm_data import NMDataContainer
from pyneuromatic.nm_object import NMObject
from pyneuromatic.nm_object_container import NMObjectContainer
import pyneuromatic.nm_utilities as nmu


class NMToolFolder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: str = "NMToolFolder",
        copy: Union[nmu.NMToolFolderType, None] = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__data_container = None  # save results to NumPy arrays

        if copy is None:
            pass
        elif isinstance(copy, NMToolFolder):
            self.__data_container = copy.data.copy()
            self.__dataseries_container = copy.dataseries.copy()
        else:
            e = nmu.typeerror(copy, "copy", "NMToolFolder")
            raise TypeError(e)

        if not isinstance(self.__data_container, NMDataContainer):
            self.__data_container = NMDataContainer(parent=self)

    # override
    def __eq__(self, other: nmu.NMToolFolderType) -> bool:
        if not super().__eq__(other):
            return False
        if self.__data_container != other._NMToolFolder__data_container:
            return False

    # override, no super
    def copy(self) -> nmu.NMToolFolderType:
        return NMToolFolder(copy=self)

    # override
    @property
    def content(self) -> Dict[str, str]:
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
        name: str = "NMToolFolderContainer",
        rename_on: bool = True,
        name_prefix: str = "toolfolder",
        name_seq_format: str = "0",
        copy: nmu.NMToolFolderContainerType = None,  # see copy()
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
    def copy(self) -> nmu.NMToolFolderContainerType:
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
    ) -> nmu.NMToolFolderType:
        name = self._newkey(name)
        f = NMToolFolder(parent=self, name=name)
        super().new(f, select=select)
        return f
