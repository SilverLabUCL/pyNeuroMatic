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
        nm = self.__manager
        nm.configs.quiet = False
        self.__history('start...')
        c = Container(nm.project, 'ContainerTest', self.__fxns,
                      prefix='Test', rename=True, duplicate=True)
        o = c.new()
        newname = o.name
        c.new(newname)
        newname = 'Test1'
        c.new(newname)
        c.new(select=False)
        print('select=' + c.select.name)
        c.select = 'Test'
        c.select = 'Test1'
        c.prefix = 'Testing'
        c.new()
        c.rename('select', 'TestX')
        c.rename('TestX', 'TestXX')
        o = c.get('Test2')
        if o:
            o.name = 'Test22'
        print('count=' + str(c.count))
        print(c.item_num('TESTXX'))
        n = 'Test22'
        if c.exists(n):
            print(n + ' exists')
        else:
            print(n + ' does not exist')
        c.duplicate('Test22', 'Test2')
        c.duplicate('Test22', 'Test2')
        c.kill('Test22')
        print(c.content)
        c.kill(all_=True)
        return True

    def project(self):
        nm = self.__manager
        nm.configs.quiet = False
        self.__history('start...')
        print(nm.project.content)
        nm.project_new('ProjectNew$')
        nm.project_new('ProjectNew')
        return True

    def folder(self):
        nm = self.__manager
        nm.configs.quiet = False
        self.__history('start...')
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': ['Vmem', 'Icmd'], 'yunits': ['mV', 'pA']}
        nm.folder.new('FolderTest')
        nm.dataseries.make(name='Data', channels=3, epochs=3, samples=5,
                           noise=noise, dims=dims)
        print(nm.folder.select.content_tree)
        nm.eset.add_epoch('Set1', [0, 1, 2])
        dname = 'DataB0'
        b0 = nm.data.get(dname)
        print(b0.thedata)
        s1 = nm.eset.get('Set1')
        nm.data.kill(dname, ask=False)
        print(nm.folder.select.content_tree)
        for i in range(0, nm.dataseries.count):
            ds = nm.dataseries.get(item_num=i)
            for cdata in ds.thedata:
                for d in cdata:
                    if d.name.lower() == dname.lower():
                        print('found in DataSeries: ' + d.name)
        for d in s1.theset:
            if d.name.lower() == dname.lower():
                print('found in Set1: ' + d.name)
        f = Folder(nm.project, 'FolderTest', self.__fxns)
        nm.folder.add(f)
        nm.folder.new()
        nm.folder.rename('select', 'FolderNew1')
        nm.folder.duplicate('select', 'FolderNew2')
        #nm.folder.kill('select')
        return True

    def data(self):
        nm = self.__manager
        nm.configs.quiet = False
        self.__history('start...')
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': ['Vmem', 'Icmd'], 'yunits': ['mV', 'pA']}
        if not nm.folder.select or not nm.dataseries:
            return False
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
