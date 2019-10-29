# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py

import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_data import DataContainer
from nm_dataprefix import DataPrefixContainer


class Folder(NMObject):
    """
    NM Folder class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__data_container = DataContainer(self, 'NMData')
        self.__dataprefix_container = DataPrefixContainer(self,'NMDataPrefix',
                                                          self.__data_container)

    @property
    def data_container(self):
        return self.__data_container

    @property
    def data_list(self):
        return self.__data_container.name_list

    @property
    def dataprefix_container(self):
        return self.__dataprefix_container


class FolderContainer(Container):
    """
    Container for NM Folders
    """

    def __init__(self, parent, name):
        super().__init__(parent, name, nmc.FOLDER_PREFIX)

    def object_new(self, name):  # override, do not call super
        return Folder(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Folder)

    def open_hdf5(self):
        dataprefix = "Record"
        with h5py.File('nmFolder0.hdf5', 'r') as f:
            #print(f.keys())
            data = []
            for k in f.keys():
                if k[0:len(dataprefix)] == dataprefix:
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
