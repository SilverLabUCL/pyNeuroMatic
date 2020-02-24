# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py

from nm_object import NMObject
from nm_object import NMObjectContainer
from nm_data import DataContainer
from nm_dataseries import DataSeriesContainer
import nm_preferences as nmp
import nm_utilities as nmu


class Folder(NMObject):
    """
    NM Data Folder class
    """

    def __init__(self, parent, name, **copy):
        super().__init__(parent, name)
        self.__data_container = None
        self.__dataseries_container = None
        for k, v in copy.items():
            if k.lower() == 'c_data' and isinstance(v, DataContainer):
                self.__data_container = v
            if k.lower() == 'c_dataseries' and isinstance(v,
                                                          DataSeriesContainer):
                self.__dataseries_container = v
        if not isinstance(self.__data_container, DataContainer):
            self.__data_container = DataContainer(self, 'Data')
        if not isinstance(self.__dataseries_container, DataSeriesContainer):
            self.__dataseries_container = DataSeriesContainer(self,
                                                              'DataSeries')

    # override
    @property
    def content(self):
        k = super().content
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
        return Folder(self._parent, self.name,
                      c_data=self.__data_container.copy(),
                      c_dataseries=self.__dataseries_container.copy())

    @property
    def data(self):
        return self.__data_container

    @property
    def dataseries(self):
        return self.__dataseries_container


class FolderContainer(NMObjectContainer):
    """
    Container for NM Folders
    """

    def __init__(self, parent, name, **copy):
        t = Folder(None, 'empty').__class__.__name__
        super().__init__(parent, name, type_=t, prefix=nmp.FOLDER_PREFIX,
                         rename=True, **copy)

    # override, no super
    def copy(self):
        return FolderContainer(self._parent, self.name, c_prefix=self.prefix,
                               c_rename=self.parameters['rename'],
                               c_thecontainer=self._thecontainer_copy())

    # override
    def new(self, name='default', select=True, quiet=nmp.QUIET):
        o = Folder(None, 'iwillberenamed')
        return super().new(name=name, nmobject=o, select=select, quiet=quiet)

    def add(self, folder, select=True, quiet=nmp.QUIET):
        if not isinstance(folder, Folder):
            raise TypeError(nmu.type_error(folder, 'Folder'))
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
