# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import h5py
import nm_configs as nmconfig
from nm_container import Container
from nm_wave import WaveContainer
from nm_waveprefix import WavePrefixContainer
from nm_utilities import name_ok


class Folder(object):
    """
    NM Folder class
    """

    def __init__(self, name):
        self.__name = name
        self.__wave = WaveContainer()
        self.__waveprefix = WavePrefixContainer(self.__wave)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name

    @property
    def wave(self):
        return self.__wave

    @property
    def waveprefix(self):
        return self.__waveprefix


class FolderContainer(Container):
    """
    Container for NM Folders
    """

    def __init__(self):
        super().__init__()
        self.prefix = nmconfig.FOLDER_PREFIX

    def object_new(self, name):  # override, do not call super
        return Folder(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Folder)

    def open_hdf5(self):
        wave_prefix = "Record"
        with h5py.File('nmFolder0.hdf5', 'r') as f:
            #print(f.keys())
            data = []
            for k in f.keys():
                if k[0:len(wave_prefix)] == wave_prefix:
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
