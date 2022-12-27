# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py

from nm_object import NMObject
from nm_object_container import NMObjectContainer
from nm_data import NMDataContainer
from nm_dataseries import NMDataSeriesContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List


class NMFolder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMFolder',
        copy: nmu.NMFolderType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__data_container = None
        self.__dataseries_container = None

        if isinstance(copy, NMFolder):
            self.__data_container = copy.data._container_copy()
            self.__dataseries_container = copy.dataseries._container_copy()

        if not isinstance(self.__data_container, NMDataContainer):
            self.__data_container = NMDataContainer(parent=self, name='Data')
        if not isinstance(self.__dataseries_container, NMDataSeriesContainer):
            self.__dataseries_container = NMDataSeriesContainer(
                parent=self,
                name='Dataseries')

    # override
    def __eq__(
        self,
        other: nmu.NMObjectType
    ) -> bool:
        if not super().__eq__(other):
            return False
        if self.__data_container != other._NMFolder__data_container:
            return False
        if (self.__dataseries_container !=
                other._NMFolder__dataseries_container):
            return False
        return True

    # override, no super
    def copy(self) -> nmu.NMFolderType:
        return NMFolder(copy=self)

    # override
    @property
    def content(self) -> Dict[str, str]:
        k = super().content
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        return k

    # override
    def _isequivalent(self, folder, alert=False):
        if not super()._isequivalent(folder, alert=alert):
            return False
        c = self.__data_container
        c2 = folder._NMFolder__data_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        c = self.__dataseries_container
        c2 = folder._NMFolder__dataseries_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        return True

    @property
    def data(self):
        return self.__data_container

    @property
    def dataseries(self):
        return self.__dataseries_container


class NMFolderContainer(NMObjectContainer):
    """
    Container for NM Folders
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMFolderContainer',
        prefix: str = 'NMFolder',  # for generating names of NMFolders
        copy: nmu.NMFolderContainerType = None  # see copy()
    ) -> None:
        f = NMFolder(parent=parent, name='ContainerUtility')
        super().__init__(parent=parent, name=name, nmobject=f, prefix=prefix,
                         rename=True, copy=copy)

    # override, no super
    def copy(self) -> nmu.NMFolderContainerType:
        return NMFolderContainer(copy=self)

    def open_hdf5(self):
        dataseries = 'Record'
        with h5py.File('nmFolder0.hdf5', 'r') as f:
            #print(f.keys())
            data = []
            for k in f.keys():
                if k[0:len(dataseries)] == dataseries:
                    print(k)
            # for name in f:
                # print(name)
            d = f['RecordA0']

            for i in d.attrs.keys():
                print(i)
            # cannot get access to attribute values for keys:
            # probably need to update h5py to v 2.10
            #IGORWaveNote
            #IGORWaveType
            #print(d.attrs.__getitem__('IGORWaveNote'))
            #for a in d.attrs:
                #print(item + ":", d.attrs[item])
                #print(item + ":", d.attrs.get(item))
                #print(a.shape)
            #for k in a.keys():
                #print(k)
            #print(a)
            #pf = f['NMPrefix_Record']
            #print(pf)
