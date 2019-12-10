# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
from nm_folder import Folder
import nm_utilities as nmu


class Test(object):
    """
    NM Test class
    """

    def __init__(self, manager, fxns):
        self.__manager = manager
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    def container(self):
        self.__history('start...')
        nm = self.__manager
        nm.configs.quiet = False
        c = Container(nm, nm.project, 'NMContainerTest', self.__fxns,
                      prefix='NMTest', rename=True, duplicate=True, kill=False)
        o = c.get('NMTest0')
        c.new()
        c.new()
        c.new()
        c.select = 'NMTest'
        c.select = 'NMTest1'
        c.prefix = 'NMTesting'
        c.new()
        c.rename('select', 'NMTestX')
        c.rename('NMTestX', 'NMTestXX')
        o = c.get('NMTest2')
        o.name = 'NMTest22'
        print('count=' + str(c.count))
        print(c.item_num('NMTESTXX'))
        n = 'NMTest22'
        if c.exists(n):
            print(n + ' exists')
        else:
            print(n + ' does not exist')
        c.duplicate('NMTest22', 'NMTest2')
        c.duplicate('NMTest22', 'NMTest2')
        c.kill('NMTest22')
        print(c.content)
        return True

    def project(self):
        self.__history('start...')
        nm = self.__manager
        print(nm.project.content)
        nm.project_new('NMProjectNew')
        return True

    def folder(self):
        self.__history('start...')
        nm = self.__manager
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': ['Vmem', 'Icmd'], 'yunits': ['mV', 'pA']}
        nm.folder.new(name='NMFolderTest')
        nm.dataseries.make(name='Data', channels=3, epochs=3, samples=5,
                           noise=noise, dims=dims)
        print(nm.folder.select.content_tree)
        f = Folder(nm.project, 'NMFolderTest')
        nm.folder.add(f)
        nm.folder.new()
        nm.folder.rename('select', 'FolderNew1')
        nm.folder.duplicate('select', 'FolderNew2')
        nm.folder.kill('select')
        return True

    def data(self):
        self.__history('start...')
        nm = self.__manager
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': ['Vmem', 'Icmd'], 'yunits': ['mV', 'pA']}
        nm.dataseries.make(name='Data', channels=2, epochs=3, samples=5,
                           noise=noise, dims=dims)
        nm.dataseries.make(name='Data', channels=2, epochs=3, samples=5,
                           noise=noise, dims=dims)
        # self.dataseries.make(name='Wave', channels=3, epochs=8, samples=5,
        #                     noise=noise)
        x = nm.dataseries.select.xdata_make(name='x_Wave', samples=5,
                                            dims=dims)
        for i in range(0, len(x.thedata)):
            x.thedata[i] = i * 0.01
        nm.data.select = 'DataA0'
        nm.data.select.yunits = 'test'
        # self.dataseries.new('Test')
        # self.dataseries.kill('Test')
        # self.folder.new()
        # self.folder.duplicate('NMFolder0', 'NMFolder1')
        # self.folder.duplicate('NMFolder0', 'NMFolder2')
        # self.folder.select = 'NMFolder2'
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
