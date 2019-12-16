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
        d = DataContainer(manager, self, 'Data', fxns)
        self.__data_container = d
        s = DataSeriesContainer(manager, self, 'DataSeries', fxns)
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
        super().__init__(manager, parent, name, fxns, type_='Folder',
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
        tp = self.tree_path(history=True)
        if not isinstance(folder, Folder):
            e = 'folder arg: expected type Folder'
            self.__error(e, tp=tp, quiet=quiet)
            return False
        if not isinstance(select, bool):
            select = True
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        name = folder.name
        if not name or not nmu.name_ok(name):
            e = 'bad folder name: ' + nmu.quotes(name)
            self.__error(e, tp=tp, quiet=quiet)
            return False
        if self.exists(name):
            e = nmu.quotes(name) + ' already exists'
            self.__error(e, tp=tp, quiet=quiet)
            return False
        super().new(name=name, nmobj=folder, select=select, quiet=quiet)
        return self.exists(name)

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
