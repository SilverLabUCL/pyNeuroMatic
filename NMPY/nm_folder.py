# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py

from nm_container import NMObject
from nm_container import Container
from nm_data import DataContainer
from nm_dataseries import DataSeriesContainer
import nm_preferences as nmp
import nm_utilities as nmu


class Folder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(self, parent, name, fxns={}, **copy):
        super().__init__(parent, name, fxns=fxns)
        self.__data_container = None
        self.__dataseries_container = None
        for k, v in copy.items():
            if k.lower() == 'data' and isinstance(v, DataContainer):
                self.__data_container = v
            if k.lower() == 'dataseries' and isinstance(v, DataSeriesContainer):
                self.__dataseries_container = v
        if not isinstance(self.__data_container, DataContainer):
            self.__data_container = DataContainer(self, 'Data', fxns=fxns)
        if not isinstance(self.__dataseries_container, DataSeriesContainer):
            self.__dataseries_container = DataSeriesContainer(self,
                                                              'DataSeries',
                                                              fxns=fxns)

    # override, no super
    @property
    def content(self):
        k = {'folder': self.name}
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        return k

    # override
    def _equal(self, folder, alert=False):
        if not super()._equal(folder, alert=alert):
            return False
        c = folder._Folder__data_container
        if not self.__data_container._equal(c, alert=alert):
            return False
        c = folder._Folder__dataseries_container
        return self.__dataseries_container._equal(c, alert=alert)

    # override, no super
    def copy(self):
        c = Folder(self._parent, self.name, fxns=self._fxns,
                   data=self.__data_container.copy(),
                   dataseries=self.__dataseries_container.copy())
        return c

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

    def __init__(self, parent, name, fxns={}, **copy):
        t = Folder(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t,
                         prefix=nmp.FOLDER_PREFIX, **copy)

    # override, no super
    @property
    def content(self):
        return {'folders': self.names}

    # override, no super
    def copy(self):
        return FolderContainer(self._parent, self.name, fxns=self._fxns,
                               thecontainer=self._thecontainer_copy())

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = Folder(self._parent, 'temp', self._fxns)
        return super().new(name=name, nmobj=o, select=select, quiet=quiet)

    def add(self, folder, select=True, quiet=nmp.QUIET):
        if not isinstance(folder, Folder):
            raise TypeError(nmu.type_error(folder, 'Folder'))
        select = nmu.check_bool(select, True)
        name = folder.name
        if not name or not nmu.name_ok(name):
            e = 'bad folder name: ' + nmu.quotes(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        if self.exists(name):
            e = nmu.quotes(name) + ' already exists'
            self._error(e, tp=self._tp, quiet=quiet)
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
