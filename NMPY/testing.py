#!/usr/bin/env python[3]
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import inspect
import unittest
import numpy as np

from nm_channel import Channel
from nm_channel import ChannelContainer
from nm_container import NMObject
from nm_container import Container
from nm_data import Data
from nm_data import DataContainer
from nm_dataseries import DataSeries
from nm_dataseries import DataSeriesContainer
from nm_manager import Manager
from nm_project import Project
import nm_utilities as nmu

nm = Manager(quiet=True)
BADNAME = 'b&dn@me!'


def test_type_error(a, b, c, got):
    return nmu.type_error(c, got)


class Test(unittest.TestCase):

    def test_nmobject(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'object0'
        name1 = 'object1'
        badtypes = [True, 1, float('nan'), [], {}, set(), None, self]
        badnames = ['select', 'default', 'all']  # may need updating
        param_list = ['name', 'rename', 'date', 'modified', 'source']
        # param_list may need updating
        for b in badtypes:
            with self.assertRaises(TypeError):
                o0 = NMObject(parent, b, fxns=nm._fxns, rename=True)
        for b in badnames + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0 = NMObject(parent, b, fxns=nm._fxns, rename=True)
        o0 = NMObject(parent, name0, fxns=nm._fxns, rename=True)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertEqual(o0.name, name0)
        self.assertEqual(o1.name, name1)
        # name_ok(()
        for b in badtypes:
            self.assertFalse(o1.name_ok(b))
        for b in badnames + [BADNAME]:
            self.assertFalse(o1.name_ok(b))
        for b in badnames:
            self.assertTrue(o1.name_ok(b, ok=badnames))
        good = [name0, name1, '']
        for g in good:
            self.assertTrue(o1.name_ok(g))
        # _bad_names()
        self.assertEqual(badnames, o1._bad_names)  # check if list changes
        # parameters()
        plist = list(o0.parameters.keys())
        self.assertEqual(plist, param_list)  # check if list changes
        # _name_set()
        for b in badtypes:
            with self.assertRaises(TypeError):
                o0._name_set(b)
        for b in badnames + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0._name_set(b)
        good = ['test', name0]
        for g in good:
            self.assertTrue(o0._name_set(g))
            self.assertEqual(g, o0.name)
        with self.assertRaises(RuntimeError):
            o1._name_set(name0)  # rename = False
        # content()
        self.assertEqual(o0.content, {'nmobject': o0.name})
        self.assertEqual(o0.content_tree, {'nmobject': o0.name})
        # tree_path()
        self.assertEqual(o0.tree_path(), o0.name)
        self.assertEqual(o0.tree_path_list(), [o0.name])
        # _equal()
        self.assertFalse(o1._equal(None, alert=True))
        self.assertFalse(o1._equal(self, alert=True))
        self.assertFalse(o1._equal(o0, alert=True))
        self.assertFalse(o1._equal(o0, ignore_name=True, alert=True))
        # _copy()
        o0.name = name0
        for b in badtypes:
            with self.assertRaises(TypeError):
                o1._copy(b)
        self.assertTrue(o1._copy(o0, copy_name=False))
        self.assertFalse(o1._equal(o0, ignore_name=False, alert=True))
        self.assertTrue(o1._equal(o0, ignore_name=True, alert=True))
        self.assertTrue(o1._copy(o0, copy_name=True))
        self.assertTrue(o1._equal(o0, alert=True))
        

    def test_container(self):
        nm.configs.quiet = False
        parent = self
        name0 = 'container0'
        name1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        badtypes = [True, 1, float('nan'), [], {}, set(), None, self]
        badnames = ['select', 'default', 'all']  # may need updating
        param_list = ['name', 'rename', 'date', 'modified', 'source']
        param_list += ['type', 'prefix', 'duplicate']
        # param_list may need updating
        n = [p0 + str(i) for i in range(0, 6)]
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b,
                               prefix=p0, rename=True, duplicate=True)
        for b in [BADNAME, '']:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b,
                               prefix=p0, rename=True, duplicate=True)
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b, rename=True, duplicate=True)
        for b in badnames + [BADNAME]:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b, rename=True, duplicate=True)
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix='',
                       rename=True, duplicate=True)
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix=p0,
                       rename=True, duplicate=True)
        c1 = Container(parent, name1, fxns=nm._fxns, type_=type_, prefix=p1,
                       rename=False, duplicate=False)
        self.assertEqual(c0.name, name0)
        self.assertEqual(c1.name, name1)
        self.assertEqual(c0.prefix, p0)
        self.assertEqual(c1.prefix, p1)
        # parameters()
        plist = list(c0.parameters.keys())
        self.assertEqual(plist, param_list)
        # prefix()
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0._prefix_set(b)
        for b in badnames + [BADNAME]:
            with self.assertRaises(ValueError):
                c0._prefix_set(b)
        self.assertTrue(c0._prefix_set(p1))
        self.assertEqual(c0.prefix, p1)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c1.prefix, p1)
        c0.prefix = p0  # reset
        # name_next()
        self.assertEqual(c0.name_next(), n[0])
        self.assertEqual(c0.name_next_seq(), 0)
        # new()
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.new(b)
        for b in badnames + [BADNAME]:
            if b.lower() == 'default':
                continue  # 'default' is ok
            with self.assertRaises(ValueError):
                c0.new(b)
        o = c0.new('default')  # ok
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, n[0])
        self.assertEqual(c0.select, o)
        self.assertEqual(c0.select.name, n[0])
        with self.assertRaises(RuntimeError):
            c0.new(n[0])  # already exists
        self.assertEqual(c0.name_next(), n[1])
        self.assertEqual(c0.name_next_seq(), 1)
        o = c0.new(select=False)
        self.assertEqual(o.name, n[1])
        self.assertEqual(c0.select.name, n[0])
        self.assertEqual(c0.name_next(), n[2])
        self.assertEqual(c0.name_next_seq(), 2)
        o = c0.new(n[2])
        self.assertEqual(o.name, n[2])
        # skip n[3]
        o = NMObject(parent, n[4], fxns=nm._fxns, rename=True)
        o = c0.new(name=n[4], nmobj=o)
        self.assertEqual(o.name, n[4])
        o = c0.new()
        self.assertEqual(o.name, n[5])
        self.assertEqual(c0.count, 5)
        c1.new()
        c1.new()
        c1.new()
        # names()
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[4], n[5]])
        # content()
        c = c0.content
        self.assertEqual(list(c.keys()), ['nmobjects', 'select'])
        self.assertEqual(c['nmobjects'], c0.names)
        self.assertEqual(c['select'], c0.get('select').name)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), ['nmobjects', 'select'])
        self.assertEqual(c['nmobjects'], c0.names)
        self.assertEqual(c['select'], c0.get('select').name)
        # tree_path()
        self.assertEqual(c0.tree_path(), name0)
        self.assertEqual(c0.tree_path_list(), [name0])
        # item_num()
        for b in badtypes:
            self.assertEqual(c0.item_num(b), -1)
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # 'select' is ok
            self.assertEqual(c0.item_num(b), -1)
        self.assertEqual(c0.item_num(n[0]), 0)
        self.assertEqual(c0.item_num(n[1]), 1)
        self.assertEqual(c0.item_num(n[2]), 2)
        self.assertEqual(c0.item_num(n[3]), -1)  # skipped
        self.assertEqual(c0.item_num(n[4]), 3)
        self.assertEqual(c0.item_num(n[5]), 4)
        self.assertEqual(c0.item_num('select'), 4)
        # exists()
        for b in badtypes:
            self.assertFalse(c0.exists(b))
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # 'select' is ok
            self.assertFalse(c0.exists(b))
        self.assertTrue(c0.exists(n[0]))
        self.assertTrue(c0.exists(n[1]))
        self.assertTrue(c0.exists(n[2]))
        self.assertFalse(c0.exists(n[3]))  # skipped
        self.assertTrue(c0.exists(n[4]))
        self.assertTrue(c0.exists(n[5]))
        self.assertTrue(c0.exists('select'))
        # get()
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.get(b)
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # 'select' is ok
            with self.assertRaises(ValueError):
                c0.get(b)
        self.assertIsNone(c0.get(''))
        self.assertEqual(c0.get('select'), c0.select)
        self.assertIsInstance(c0.get(n[0]), NMObject)
        self.assertIsInstance(c0.get(n[1]), NMObject)
        self.assertIsInstance(c0.get(n[2]), NMObject)
        self.assertIsNone(c0.get(n[3]))
        self.assertIsInstance(c0.get(n[4]), NMObject)
        self.assertIsInstance(c0.get(n[5]), NMObject)
        self.assertIsInstance(c0.get(item_num=0), NMObject)
        self.assertIsInstance(c0.get(item_num=1), NMObject)
        self.assertIsInstance(c0.get(item_num=2), NMObject)
        self.assertIsInstance(c0.get(item_num=3), NMObject)
        self.assertIsInstance(c0.get(item_num=4), NMObject)
        self.assertIsNone(c0.get(item_num=-1))
        for b in [3.14, float('inf'), float('nan'), [1, 2, 3], {}]:
            with self.assertRaises(TypeError):
                c0.get(item_num=b)
        for b in [5, 10, 100]:
            with self.assertRaises(IndexError):
                c0.get(item_num=b)
        # get_items()
        with self.assertRaises(TypeError):
            olist0 = c0.get_items(names=[n[0], n[1], n[2], None])
        with self.assertRaises(ValueError):
            olist0 = c0.get_items(names=[n[0], n[1], n[2], BADNAME])
        with self.assertRaises(ValueError):
            olist0 = c0.get_items(names=[n[0], n[1], n[2], 'default'])
        with self.assertRaises(IndexError):
            olist1 = c0.get_items(item_nums=[0, 1, 2, 3, 50])
        olist0 = c0.get_items(names=[n[0], n[1], n[2], n[4]])
        olist1 = c0.get_items(item_nums=[0, 1, 2, 3])
        self.assertEqual(olist0, olist1)
        olist0 = c0.get_items(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()
        self.assertEqual(len(olist0), c0.count-1)
        # select()
        sname = c0.select.name
        with self.assertRaises(TypeError):
            c0.select = None
        with self.assertRaises(ValueError):
            c0.select = BADNAME
        with self.assertRaises(ValueError):
            c0.select = 'select'
        with self.assertRaises(ValueError):
            c0.select = 'default'
        c0.select = n[3]  # does not exist
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        c0.select = n[0]  # ok
        self.assertEqual(c0.select.name, n[0])
        # rename()
        with self.assertRaises(TypeError):
            c0.rename(None, n[3])
        with self.assertRaises(ValueError):
            c0.rename(BADNAME, n[3])
        with self.assertRaises(ValueError):
            c0.rename('default', n[3])
        with self.assertRaises(TypeError):
            c0.rename(n[4], None)
        with self.assertRaises(ValueError):
            c0.rename(n[4], BADNAME)
        with self.assertRaises(ValueError):
            c0.rename(n[4], 'select')
        with self.assertRaises(ValueError):
            c0.rename(n[4], '')
        with self.assertRaises(RuntimeError):
            c0.rename('select', n[0])  # already exists
        with self.assertRaises(RuntimeError):
            c1.rename('select', 'test')  # rename = False
        self.assertEqual(c0.rename('', n[3]), '')
        i = c0.item_num(n[4])
        nnext = c0.name_next()
        newname = c0.rename(n[4], 'default')
        self.assertEqual(newname, nnext)
        self.assertTrue(c0.rename(newname, n[3]))
        o = c0.get(item_num=i)
        self.assertEqual(o.name, n[3])
        self.assertTrue(c0.rename(n[5], n[4]))
        # duplicate()
        with self.assertRaises(TypeError):
            c0.duplicate(None, 'default')
        with self.assertRaises(ValueError):
            c0.duplicate(BADNAME, 'default')
        with self.assertRaises(ValueError):
            c0.duplicate('default', 'default')
        with self.assertRaises(TypeError):
            c0.duplicate(n[0], None)
        with self.assertRaises(ValueError):
            c0.duplicate(n[0], BADNAME)
        with self.assertRaises(ValueError):
            c0.duplicate(n[0], 'select')
        with self.assertRaises(ValueError):
            c0.duplicate(n[0], '')
        with self.assertRaises(RuntimeError):
            c0.duplicate(n[0], n[1])  # already exists
        with self.assertRaises(RuntimeError):
            c1.duplicate('select', 'default')  # duplicate = False
        self.assertIsNone(c0.duplicate('', 'default'))
        o = c0.get(n[0])
        nnext = c0.name_next()
        oc = c0.duplicate(n[0], 'default')
        self.assertEqual(oc.name, nnext)
        self.assertFalse(o._equal(oc, alert=True))  # names are different
        self.assertTrue(o._equal(oc, ignore_name=True, alert=True))
        # _equal()
        self.assertFalse(c0._equal(c1, alert=True))
        self.assertFalse(c0._equal(c1, ignore_name=True, alert=True))
        # _copy()
        with self.assertRaises(TypeError):
            c1._copy(self)
        c1._copy(c0, copy_name=False)
        self.assertFalse(c1._equal(c0, alert=True))
        self.assertFalse(c1._equal(c0, ignore_name=True, alert=True))
        c1._copy(c0, copy_name=False, clear_before_copy=True)
        self.assertTrue(c1._equal(c0, ignore_name=True, alert=True))
        # Kill()
        with self.assertRaises(TypeError):
            c0.kill(name=None, confirm=False)
        with self.assertRaises(ValueError):
            c0.kill(name=BADNAME, confirm=False)
        with self.assertRaises(ValueError):
            c0.kill(name='default', confirm=False)
        self.assertEqual(c0.kill(name='', confirm=False), [])
        s = c0.select
        klist = c0.kill(name='select', confirm=False)
        self.assertEqual(klist, [s])
        self.assertIsNone(c0.get(s.name))
        o = c0.get(n[0])
        klist = c0.kill(name=n[0], confirm=False)
        self.assertEqual(klist, [o])
        self.assertIsNone(c0.get(n[0]))
        names = c1.names
        klist = c1.kill(all_=True, confirm=False)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(c1.count, 0)
        self.assertIsNone(c1.select)

    def test_data(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'RecordA0'
        name1 = 'RecordA1'
        shape = [5]  # test different shapes
        dims0 = {'xstart': 10, 'xdelta': 0.01,
                 'xlabel': 'time', 'xunits': 'ms',
                 'ylabel': 'Vmem', 'yunits': 'mV'}
        dims1 = {'xstart': -10, 'xdelta': 0.05,
                 'xlabel': 'time', 'xunits': 's',
                 'ylabel': 'Imem', 'yunits': 'pA'}
        xdims = {'xstart': 0, 'xdelta': 1,
                 'xlabel': 'sample', 'xunits': '',
                 'ylabel': 'time', 'yunits': 's'}
        d0 = Data(parent, name0, fxns=nm._fxns, shape=shape,
                  fill_value=np.nan, dims=dims0)
        d1 = Data(parent, name1, fxns=nm._fxns, shape=shape,
                  fill_value=0, dims=dims1)
        xdata = Data(parent, 'xdata', fxns=nm._fxns, shape=shape,
                     fill_value=0, dims=xdims)
        self.assertEqual(d0.dims, dims0)
        self.assertEqual(d1.dims, dims1)
        # dims()
        dimlist = ['xdata', 'xstart', 'xdelta', 'xlabel', 'xunits', 'ylabel',
                   'yunits']
        self.assertEqual(dimlist, nmu.DIM_LIST)
        bad = [None, [], 'test', 1.0, self]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._dims_set(b)
        bad = {'xstart': 10, 'xd': 0.01}
        with self.assertRaises(KeyError):
            d0.dims = bad
        self.assertTrue(d0._dims_set(dims1))
        self.assertEqual(dims1, d0.dims)
        # _xdata_set()
        bad = [[], {}, 'test', 1.0, self]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._xdata_set(b)
        good = [None, xdata]
        for g in good:
            self.assertTrue(d0._xdata_set(g))
            self.assertEqual(g, d0.xdata)
        # _xstart_set()
        bad = [self, 'start', [], {}]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._xstart_set(b)
        bad = [float('inf'), float('nan')]
        for b in bad:
            with self.assertRaises(ValueError):
                d0._xstart_set(b)
        good = [-100, 0, 100]
        for g in good:
            self.assertTrue(d0._xstart_set(g))
            self.assertEqual(g, d0.xstart)
        # _xdelta_set()
        bad = [self, 'start', [], {}]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._xdelta_set(b)
        bad = [float('inf'), float('nan'), 0]
        for b in bad:
            with self.assertRaises(ValueError):
                d0._xdelta_set(b)
        good = [-100, 1, 100]
        for g in good:
            self.assertTrue(d0._xdelta_set(g))
            self.assertEqual(g, d0.xdelta)
        # _xlabel_set()
        # _xunits_set()
        # _ylabel_set()
        # _yunits_set()
        bad = [self, float('inf'), 1, [], {}]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._xlabel_set(b)
            with self.assertRaises(TypeError):
                d0._xunits_set(b)
            with self.assertRaises(TypeError):
                d0._ylabel_set(b)
            with self.assertRaises(TypeError):
                d0._yunits_set(b)
        good_list = ['test', '', BADNAME]
        for s in good_list:
            self.assertTrue(d0._xlabel_set(s))
            self.assertTrue(d0._xunits_set(s))
            self.assertTrue(d0._ylabel_set(s))
            self.assertTrue(d0._yunits_set(s))
            self.assertEqual(s, d0.xlabel)
            self.assertEqual(s, d0.xunits)
            self.assertEqual(s, d0.ylabel)
            self.assertEqual(s, d0.yunits)
        # _np_array_set()
        bad = [self, 'test', float('inf'), 1, [], {}]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        nparray = np.full(shape, 0)
        self.assertTrue(d0._np_array_set(nparray))
        # self.assertEqual(nparray, d0.np_array)
        # print(type(d0.np_array))
        # print(isinstance(d0.np_array, np.ndarray))
        # print(d0.np_array)
        # d0.np_array.clip(-1, 1, out=d0.np_array)
        # print(d0.np_array.flags)
        # print(d0.parameters)

    def test_project(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'Project0'
        name1 = 'Project1'
        noise = [0, 0.1]
        dims = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': {'A': 'Vmem', 'B': 'Icmd'},
                'yunits': {'A': 'mV', 'B': 'pA'}}
        """
        p0 = Project(parent, name0, fxns=nm._fxns)
        p1 = Project(parent, name1, fxns=nm._fxns)
        f = p0.folder.new()
        ds = f.dataseries.new('Record')
        ds.make(channels=1, epochs=3, shape=5, dims=dims)
        f = p0.folder.new()
        ds = f.dataseries.new('Wave')
        ds.make(channels=2, epochs=3, shape=5, dims=dims)
        # p1._copy(p0)
        # self.assertTrue(p1._equal(p0, alert=True))
        """

    def test_channel(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'A'
        name1 = 'B'
        """
        o0 = Channel(parent, name0, fxns=nm._fxns)
        o0.name = name1  # rename = False
        self.assertNotEqual(o0.name, name1)
        o1 = Channel(parent, name1, fxns=nm._fxns)
        o1._copy(o0)
        self.assertTrue(o1._equal(o0, alert=True))
        """

    def test_channel_container(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'ChannelContainer0'
        name1 = 'ChannelContainer1'
        """
        c0 = ChannelContainer(parent, name0, fxns=nm._fxns)
        c0.new()
        c0.new()
        c0.new()
        c1 = ChannelContainer(parent, name1, fxns=nm._fxns)
        c1._copy(c0)
        self.assertTrue(c1._equal(c0, alert=True))
        """

    def test_name_ok(self):
        self.assertTrue(nmu.name_ok('test'))
        self.assertTrue(nmu.name_ok('test1234567890'))
        self.assertTrue(nmu.name_ok('test_'))
        self.assertTrue(nmu.name_ok('test_OK'))
        self.assertTrue(nmu.name_ok('test_OK_OK'))
        self.assertTrue(nmu.name_ok('0'))
        self.assertFalse(nmu.name_ok('test*'))
        self.assertFalse(nmu.name_ok('_'))
        self.assertFalse(nmu.name_ok('*'))
        self.assertFalse(nmu.name_ok('@test'))
        self.assertFalse(nmu.name_ok('te.st'))
        self.assertFalse(nmu.name_ok('test?'))
        self.assertFalse(nmu.name_ok(None))
        self.assertFalse(nmu.name_ok(0))
        self.assertTrue(nmu.name_ok(['test']))
        self.assertTrue(nmu.name_ok(['test0', 'test1', 'test2']))
        self.assertFalse(nmu.name_ok(['test0', 'test1', 'test2?']))
        self.assertFalse(nmu.name_ok(['test0', 1, 'test2']))

    def test_number_ok(self):
        self.assertFalse(nmu.number_ok(None))
        self.assertFalse(nmu.number_ok(False))
        self.assertFalse(nmu.number_ok(complex(1, -1)))
        self.assertTrue(nmu.number_ok(0))
        self.assertTrue(nmu.number_ok(-5))
        self.assertTrue(nmu.number_ok(1.34))
        self.assertFalse(nmu.number_ok(1.34, only_integer=True))
        self.assertFalse(nmu.number_ok(float('inf')))
        self.assertFalse(nmu.number_ok(float('-inf')))
        self.assertFalse(nmu.number_ok(float('nan')))
        self.assertTrue(nmu.number_ok(float('inf'), no_inf=False))
        self.assertTrue(nmu.number_ok(float('-inf'), no_inf=False))
        self.assertTrue(nmu.number_ok(float('nan'), no_nan=False))
        self.assertTrue(nmu.number_ok(0, no_neg=True))
        self.assertTrue(nmu.number_ok(1.34, no_neg=True))
        self.assertFalse(nmu.number_ok(-1.34, no_neg=True))
        self.assertTrue(nmu.number_ok(0, no_pos=True))
        self.assertFalse(nmu.number_ok(1.34, no_pos=True))
        self.assertTrue(nmu.number_ok(-1.34, no_pos=True))
        self.assertFalse(nmu.number_ok(0, no_zero=True))
        self.assertTrue(nmu.number_ok(1.34, no_zero=True))
        self.assertTrue(nmu.number_ok(-1.34, no_zero=True))
        self.assertTrue(nmu.number_ok([0]))
        self.assertTrue(nmu.number_ok([0, -5, 1.34]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34], only_integer=True))
        self.assertTrue(nmu.number_ok([0, 3, 4], no_neg=True))
        self.assertFalse(nmu.number_ok([-1, 3, 4], no_neg=True))
        self.assertFalse(nmu.number_ok([0, 3, 4], no_pos=True))
        self.assertTrue(nmu.number_ok([0, -3, -4], no_pos=True))
        self.assertFalse(nmu.number_ok([0, 3, 4], no_zero=True))
        self.assertTrue(nmu.number_ok([-4, 4], no_zero=True))
        self.assertTrue(nmu.number_ok([0, -5, 1.34]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, 'test']))

    def test_remove_special_chars(self):
        self.assertEqual(nmu.remove_special_chars('test*'), 'test')
        self.assertEqual(nmu.remove_special_chars('test'), 'test')
        self.assertEqual(nmu.remove_special_chars('t@st*'), 'tst')
        self.assertEqual(nmu.remove_special_chars(''), '')
        self.assertEqual(nmu.remove_special_chars(None), '')
        self.assertEqual(nmu.remove_special_chars(['test*']), '')

    def test_int_list_to_seq_str(self):
        self.assertEqual(nmu.int_list_to_seq_str([1, 2, 3, 4, 6]), '1-4, 6')
        i = [0, 1, 2, 16, 145]
        r = '0-2, 16, 145'
        self.assertEqual(nmu.int_list_to_seq_str(i), r)
        self.assertEqual(nmu.int_list_to_seq_str([0, 2, 4, 6]), '0, 2, 4, 6')
        i = [0, 1, 5, 6, 7, 12, 19, 20, 21, 22, 124]
        r = '0,1,5-7,12,19-22,124'
        self.assertEqual(nmu.int_list_to_seq_str(i, space=False), r)

    def test_channel_char(self):
        self.assertEqual(nmu.channel_char(0), 'A')
        self.assertEqual(nmu.channel_char(1), 'B')
        self.assertEqual(nmu.channel_char(2), 'C')
        self.assertEqual(nmu.channel_char(10), 'K')
        self.assertEqual(nmu.channel_char(26), '')
        self.assertEqual(nmu.channel_char(-2), '')
        self.assertEqual(nmu.channel_char(float('inf')), '')
        self.assertEqual(nmu.channel_char(float('nan')), '')
        self.assertEqual(nmu.channel_char(None), '')
        self.assertEqual(nmu.channel_char([0, 1, 2]), '')

    def test_channel_num(self):
        self.assertEqual(nmu.channel_num('A'), 0)
        self.assertEqual(nmu.channel_num('a'), 0)
        self.assertEqual(nmu.channel_num('b'), 1)
        self.assertEqual(nmu.channel_num('K'), 10)
        self.assertEqual(nmu.channel_num(''), -1)
        self.assertEqual(nmu.channel_num('AA'), -1)
        self.assertEqual(nmu.channel_num(None), -1)
        self.assertEqual(nmu.channel_num(['A', 'B', 'C']), -1)

    def test_channel_char_exists(self):
        self.assertTrue(nmu.channel_char_exists('testA1', 'A'))
        self.assertTrue(nmu.channel_char_exists('testa111', 'A'))
        self.assertTrue(nmu.channel_char_exists('testA', 'A'))
        self.assertTrue(nmu.channel_char_exists('A', 'A'))
        self.assertFalse(nmu.channel_char_exists('testA111', 'B'))
        self.assertFalse(nmu.channel_char_exists('A', 'B'))
        self.assertFalse(nmu.channel_char_exists('taste', 'A'))

    def test_type_error(self):
        e = test_type_error(self, self, self, 'NMObject')
        e2 = "bad a: expected NMObject, but got __main__.Test"
        self.assertEqual(e, e2)
        # print(e)
        a = True
        e = nmu.type_error(a, 'NMObject')
        e2 = "bad a: expected NMObject, but got bool"
        self.assertEqual(e, e2)
        # print(e)

    def test_history(self):
        quiet = True
        r = 'nm.Test.test_history: test'
        self.assertEqual(nmu.history('test', quiet=quiet), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.test_history: test'
        self.assertEqual(nmu.history('test', tp=tp, quiet=quiet), r)

    def test_get_tree_path(self):
        stack = inspect.stack()
        r = 'nm.Test.run'
        self.assertEqual(nmu.get_tree_path(stack), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.run'
        self.assertEqual(nmu.get_tree_path(stack, tp=tp), r)

    def test_get_class(self):
        stack = inspect.stack()
        self.assertEqual(nmu.get_class(stack), 'Test')
        self.assertEqual(nmu.get_class(stack, module=True), '__main__.Test')

    def test_get_method(self):
        stack = inspect.stack()
        self.assertEqual(nmu.get_method(stack), 'run')


if __name__ == '__main__':
    unittest.main()
