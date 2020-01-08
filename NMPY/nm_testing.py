# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
from nm_folder import Folder
import nm_preferences as nmp
import nm_utilities as nmu


class Test(object):
    """
    NM Test class
    """

    def __init__(self, manager, fxns):
        self._manager = manager
        self._fxns = fxns
        self._history = fxns['history']

    def folder(self):
        nm = self._manager
        nm.configs.quiet = False
        self._history('start...')
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': {'A': 'Vmem', 'B': 'Icmd'},
                'yunits': {'A': 'mV', 'B': 'pA'}}
        nm.folder.new('FolderTest')
        ds = nm.dataseries.new(name='Data')
        if ds:
            ds.make(channels=3, epochs=3, shape=5, dims=dims)
        print(nm.folder.select.content_tree)
        nm.eset.add_epoch('Set1', [0, 1, 2])
        dname = 'DataB0'
        b0 = nm.data.get(dname)
        print(b0.np_array)
        s1 = nm.eset.get('Set1')
        nm.data.kill(dname, confirm=False)
        print(nm.folder.select.content_tree)
        for i in range(0, nm.dataseries.count):
            ds = nm.dataseries.get(item_num=i)
            for c, cdata in ds.thedata.items():
                for d in cdata:
                    if d.name.lower() == dname.lower():
                        print('found in DataSeries: ' + d.name)
        for d in s1.theset:
            if d.name.lower() == dname.lower():
                print('found in Set1: ' + d.name)
        f = Folder(nm.project, 'FolderTest', self._fxns)
        nm.folder.add(f)
        nm.folder.new()
        nm.folder.rename('select', 'FolderNew1')
        nm.folder.duplicate('select', 'FolderNew2')
        #nm.folder.kill('select')
        return True

    def data(self):
        nm = self._manager
        nm.configs.quiet = False
        self._history('start...')
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': {'A': 'Vmem', 'B': 'Icmd'},
                'yunits': {'A': 'mV', 'B': 'pA'}}
        if not nm.folder.select or not nm.dataseries:
            return False
        ds = nm.dataseries.new('Data')
        if ds:
            ds.make(channels=2, epochs=3, shape=5, dims=dims)
            ds.make(channels=2, epochs=3, shape=5, dims=dims)
            x = ds.xdata_make(name='x_Wave', shape=5, dims=dims)
        for i in range(0, len(x.np_array)):
            x.np_array[i] = i * 0.01
        nm.data.select = 'DataA0'
        nm.data.select.yunits = 'test'
        # self.dataseries.new('Test')
        # self.dataseries.kill('Test')
        # self.folder.new()
        # self.folder.duplicate('Folder0', 'Folder1')
        # self.folder.duplicate('Folder0', 'Folder2')
        # self.folder.select = 'Folder2'
        # self.eset.add_epoch(['Set1', 'Set2', 'Set3'], [0,3,4,9,11])
        # self.eset.add_epoch('Set1', [0])
        # self.eset.remove_epoch('Set1', [3,4])
        # self.eset.remove_epoch('Set1', [2])
        # s1 = self.eset.get('Set1')
        # s2 = self.eset.get('Set2')
        # self.eset.add_epoch('Set1', [0, 1, 2])
        # self.eset.add_epoch('Set2', [3])
        # print(s1.names)
        # print(s2.names)
        # s1.union(s2)
        # print(s1.names)
        # self.eset.equation('Set3', ['Set1', '|', 'Set2'])
        # self.eset.select = 'Set1'
        # self.eset.add('Set1', range(0, 8, 2))
        # rdic = self.eset.add('SetX', [4])
        # print(rdic)
        # self.eset.select="Set1"
        # clist = self.dataseries.select.data_names
        # print(clist)
        return True
