# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py

from nm_object import NMObject, NMobject
from nm_object_container import NMObjectContainer, NMobjectContainer
from nm_data import NMDataContainer, NMdataContainer
from nm_dataseries import NMDataSeriesContainer, NMdataSeriesContainer
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, NewType

NMfolder = NewType('NMFolder', NMobject)
NMfolderContainer = NewType('NMFolderContainer', NMobjectContainer)


class NMFolder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMFolder',
        copy: NMfolder = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__data_container = None
        self.__dataseries_container = None
        #TODO: update copy
        for k, v in copy.items():
            if k.lower() == 'c_data' and isinstance(v, DataContainer):
                self.__data_container = v
            if k.lower() == 'c_dataseries' and isinstance(v,
                                                          DataSeriesContainer):
                self.__dataseries_container = v

        if not isinstance(self.__data_container, DataContainer):
            self.__data_container = NMDataContainer(self, 'Data')
        if not isinstance(self.__dataseries_container, DataSeriesContainer):
            self.__dataseries_container = NMDataSeriesContainer(self,
                                                              'DataSeries')

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
        c2 = folder._Folder__data_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        c = self.__dataseries_container
        c2 = folder._Folder__dataseries_container
        if c and not c._isequivalent(c2, alert=alert):
            return False
        return True

    # override, no super
    def copy(self) -> NMfolder:
        c = NMFolder(copy=self)
        c.note = 'this is a copy of ' + str(self)
        return c

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
        copy: NMfolderContainer = None  # see copy()
    ) -> None:
        f = NMFolder(None, 'empty')
        super().__init__(parent=parent, name=name, nmobject=f, prefix=prefix,
                         rename=True, copy=copy)
        # TODO: copy

    # override, no super
    def copy(self) -> NMfolderContainer:
        c = NMFolderContainer(copy=self)
        c.note = 'this is a copy of ' + str(self)
        return c

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = NMFolder(None, 'iwillberenamed')
        return super().new(name=name, nmobject=o, select=select, quiet=quiet)

    # wrapper
    def add(self, folder, select=True, quiet=nmp.QUIET):
        if not isinstance(folder, NMFolder):
            e = self._type_error('folder', 'Folder')
            raise TypeError(e)
        return self.new(name=folder.name, nmobject=folder, select=select,
                        quiet=quiet)

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
