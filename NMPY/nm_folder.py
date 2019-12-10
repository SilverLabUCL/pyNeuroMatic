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
from nm_dataseries import DataSeriesContainer
import nm_utilities as nmu


class Folder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        d = DataContainer(manager, self, 'NMDataContainer', fxns)
        self.__data_container = d
        s = DataSeriesContainer(manager, self, 'NMDataSeriesContainer', fxns)
        self.__dataseries_container = s

    @property
    def content(self):  # override, no super
        k = {'folder': self.name}
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        return k

    @property
    def data(self):
        return self.__data_container

    @property
    def dataseries(self):
        return self.__dataseries_container


class FolderContainer(Container):
    """
    Container for NM Folders
    """

    def __init__(self, manager, parent, name, fxns):
        o = Folder(manager, parent, 'temp', fxns)
        super().__init__(manager, parent, name, fxns, nmobj=o,
                         prefix=nmc.FOLDER_PREFIX)
        self.__manager = manager
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        k = {'folders': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'folder_select': s})
        return k

    # override
    def new(self, name='default', select=True, quiet=nmc.QUIET):
        o = Folder(self.__manager, self.__parent, name, self.__fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    def add(self, folder, select=True, quiet=nmc.QUIET):
        if not isinstance(folder, Folder):
            self.__error('argument ' + nmu.quotes(folder) + ' is not a Folder',
                      quiet=quiet)
            return False
        name = folder.name
        if self.exists(name):
            self.__error('Folder ' + nmu.quotes(name) + ' already exists',
                      quiet=quiet)
            return False
        f = super().new(name=name, nmobj=folder, select=select, quiet=quiet)
        return f is not None

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
