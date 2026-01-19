# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime
import h5py
from typing import Dict, Union

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeriesContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.analysis.nm_tool_folder import NMToolFolderContainer
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProjectContainer
        NMProject (project0, project1...)
            NMFolderContainer
                NMFolder (folder0, folder1...)
                    NMDataContainer
                        NMData (record0, record1... avg0, avg1)
                    NMDataSeriesContainer
                        NMDataSeries (record, avg...)
"""


class NMFolder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(
        self,
        parent: Union[object, None] = None,
        name: str = "NMFolder",
        copy: Union[nmu.NMFolderType, None] = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__data_container = None
        self.__dataseries_container = None
        self.__toolfolder_container = None  # tool results saved to NumPy arrays
        self.__toolresults = {}  # tool results saved to dict

        if copy is None:
            pass
        elif isinstance(copy, NMFolder):
            self.__data_container = copy.data.copy()
            self.__dataseries_container = copy.dataseries.copy()
            self.__toolfolder_container = copy.toolfolder.copy()
            self.__toolresults = copy.toolresults.copy()
        else:
            e = nmu.typeerror(copy, "copy", "NMFolder")
            raise TypeError(e)

        if not isinstance(self.__data_container, NMDataContainer):
            self.__data_container = NMDataContainer(parent=self)
        if not isinstance(self.__dataseries_container, NMDataSeriesContainer):
            self.__dataseries_container = NMDataSeriesContainer(parent=self)
        if not isinstance(self.__toolfolder_container, NMToolFolderContainer):
            self.__toolfolder_container = NMToolFolderContainer(parent=self)

    # override
    def __eq__(self, other: nmu.NMFolderType) -> bool:
        if not super().__eq__(other):
            return False
        if self.__data_container != other._NMFolder__data_container:
            return False
        s = self.__dataseries_container
        o = other._NMFolder__dataseries_container
        return s == o

    # override, no super
    def copy(self) -> nmu.NMFolderType:
        return NMFolder(copy=self)

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        k.update(self.__toolfolder_container.content)
        return k

    @property
    def data(self) -> nmu.NMDataContainerType:
        return self.__data_container

    @property
    def dataseries(self) -> nmu.NMDataSeriesContainerType:
        return self.__dataseries_container

    @property
    def toolfolder(self) -> nmu.NMToolFolderContainerType:
        return self.__toolfolder_container

    @property
    def toolresults(self) -> Dict[str, object]:
        return self.__toolresults

    def toolresults_save(self, tool: str, results) -> str:
        imax_keys = 99
        if not isinstance(tool, str):
            e = nmu.typeerror(tool, "tool", "string")
            raise TypeError(e)

        tp = self.treepath()
        foundkey = False
        for i in range(imax_keys):
            newkey = tool + str(i)
            if newkey not in self.__toolresults:
                foundkey = True
                break
        if not foundkey:
            e = "failed to find unused key for %s results in %s" % (tool, tp)
            raise KeyError(e)

        t = str(datetime.datetime.now())
        r = {}
        r["tool"] = tool
        r["date"] = t
        r["results"] = results
        self.__toolresults[newkey] = r
        print("saved %s results to %s via key '%s' (%s)" %
              (tool, tp, newkey, t))
        return newkey


class NMFolderContainer(NMObjectContainer):
    """
    Container of NMFolders
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMFolderContainer",
        rename_on: bool = True,
        name_prefix: str = "folder",
        name_seq_format: str = "0",
        copy: nmu.NMFolderContainerType = None,  # see copy()
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
    def copy(self) -> nmu.NMFolderContainerType:
        return NMFolderContainer(copy=self)

    # override, no super
    def content_type(self) -> str:
        return NMFolder.__name__

    # override
    def new(
        self,
        name: str = "default",
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> nmu.NMFolderType:
        name = self._newkey(name)
        f = NMFolder(parent=self, name=name)
        super().new(f, select=select)
        return f

    def open_hdf5(self):
        dataseries = "Record"
        with h5py.File("nmFolder0.hdf5", "r") as f:
            # print(f.keys())
            data = []
            for k in f.keys():
                if k[0 : len(dataseries)] == dataseries:
                    print(k)
            # for name in f:
            # print(name)
            d = f["RecordA0"]

            for i in d.attrs.keys():
                print(i)
            # cannot get access to attribute values for keys:
            # probably need to update h5py to v 2.10
            # IGORWaveNote
            # IGORWaveType
            # print(d.attrs.__getitem__('IGORWaveNote'))
            # for a in d.attrs:
            # print(item + ":", d.attrs[item])
            # print(item + ":", d.attrs.get(item))
            # print(a.shape)
            # for k in a.keys():
            # print(k)
            # print(a)
            # pf = f['NMPrefix_Record']
            # print(pf)
