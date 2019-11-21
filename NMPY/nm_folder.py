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
import nm_utilities as nmu


class Folder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__data_container = DataContainer(self)
        self.__dataprefix_container = DataPrefixContainer(self,
                                                          self.__data_container)

    @property
    def content(self):  # override, no super
        k = {'folder': self.name}
        k.update(self.__data_container.content)
        k.update(self.__dataprefix_container.content)
        return k

    @property
    def data(self):
        return self.__data_container

    @property
    def dataprefix(self):
        return self.__dataprefix_container


class FolderContainer(Container):
    """
    Container for NM Folders
    """

    def __init__(self, parent, name='NMFolderContainer'):
        super().__init__(parent, name, nmc.FOLDER_PREFIX)

    @property
    def content(self):  # override, no super
        k = {'folder': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'folder_select': s})
        print(self.name + ', ' + self.select.name)
        return k

    def object_new(self, name):  # override, no super
        return Folder(self._NMObject__parent, name)

    def open_hdf5(self):
        dataprefix = 'Record'
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
