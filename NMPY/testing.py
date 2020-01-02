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


class Test(unittest.TestCase):

    def test_nmobject(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'object0'
        name1 = 'object1'
        with self.assertRaises(ValueError):
            o0 = NMObject(parent, None, fxns=nm._fxns, rename=True)
        with self.assertRaises(ValueError):
            o0 = NMObject(parent, BADNAME, fxns=nm._fxns, rename=True)
        # with self.assertRaises(ValueError):
        #    o0 = NMObject(parent, name0, badfxns, rename=True)
        o0 = NMObject(parent, name0, fxns=nm._fxns, rename=True)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertEqual(o0.name, name0)
        self.assertEqual(o1.name, name1)
        # parameters()
        k = list(o0.parameters.keys())
        self.assertEqual(k, ['name', 'rename', 'date', 'modified', 'source'])
        # name()
        o0.name = 'select'  # bad
        self.assertEqual(o0.name, name0)
        o0.name = 'default'  # bad
        self.assertEqual(o0.name, name0)
        o0.name = BADNAME
        self.assertEqual(o0.name, name0)
        o0.name = name1  # ok
        self.assertEqual(o0.name, name1)
        o1.name = name0  # cannot rename
        self.assertEqual(o1.name, name1)
        # content()
        self.assertEqual(o0.content, {'nmobject': name1})
        self.assertEqual(o0.content_tree, {'nmobject': name1})
        # tree_path()
        self.assertEqual(o0.tree_path(), name1)
        self.assertEqual(o0.tree_path_list(), [name1])
        # _equal()
        self.assertFalse(o1._equal(None, alert=True))
        self.assertFalse(o1._equal(self, alert=True))
        self.assertFalse(o1._equal(o0, alert=True))
        self.assertFalse(o1._equal(o0, ignore_name=True, alert=True))
        # _copy()
        o0.name = name0
        self.assertFalse(o1._copy(None))
        self.assertFalse(o1._copy(self))
        self.assertTrue(o1._copy(o0, copy_name=False))
        self.assertFalse(o1._equal(o0, ignore_name=False, alert=True))
        self.assertTrue(o1._equal(o0, ignore_name=True, alert=True))
        self.assertTrue(o1._copy(o0, copy_name=True))
        self.assertTrue(o1._equal(o0, alert=True))

    def test_container(self):
        nm.configs.quiet = True
        parent = self
        name0 = 'container0'
        name1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        n = []
        for i in range(0, 6):
            n.append(p0 + str(i))
        with self.assertRaises(ValueError):
            c0 = Container(parent, name0, fxns=nm._fxns, type_=None, prefix=p0,
                           rename=True, duplicate=True)
        with self.assertRaises(ValueError):
            c0 = Container(parent, name0, fxns=nm._fxns, type_=BADNAME,
                           prefix=p0, rename=True, duplicate=True)
        with self.assertRaises(ValueError):
            c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                           prefix=None, rename=True, duplicate=True)
        with self.assertRaises(ValueError):
            c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                           prefix=BADNAME, rename=True, duplicate=True)
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix=p0,
                       rename=True, duplicate=True)
        c1 = Container(parent, name1, fxns=nm._fxns, type_=type_, prefix=p1,
                       rename=False, duplicate=False)
        self.assertEqual(c0.name, name0)
        self.assertEqual(c1.name, name1)
        self.assertEqual(c0.prefix, p0)
        self.assertEqual(c1.prefix, p1)
        # parameters()
        k = list(c0.parameters.keys())
        self.assertEqual(k, ['name', 'rename', 'date', 'modified', 'source',
                             'type', 'prefix', 'duplicate'])
        # prefix()
        c0.prefix = 'select'  # bad
        self.assertEqual(c0.prefix, p0)
        c0.prefix = 'default'  # bad
        self.assertEqual(c0.prefix, p0)
        c0.prefix = BADNAME
        self.assertEqual(c0.prefix, p0)
        c0.prefix = p1  # ok
        self.assertEqual(c0.prefix, p1)
        c1.prefix = p0  # cannot rename
        self.assertEqual(c1.prefix, p1)
        c0.prefix = p0  # reset to original
        # name_next()
        self.assertEqual(c0.name_next(), n[0])
        self.assertEqual(c0.name_next_seq(), 0)
        # new()
        self.assertIsNone(c0.new(None))
        self.assertIsNone(c0.new('select'))
        self.assertIsNone(c0.new(BADNAME))
        self.assertIsNone(c0.new(nmobj=self))  # not a NMObject
        o = c0.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, n[0])
        self.assertEqual(c0.select.name, n[0])
        self.assertIsNone(c0.new(n[0]))  # already exists
        self.assertEqual(c0.name_next(), n[1])
        self.assertEqual(c0.name_next_seq(), 1)
        o = c0.new('default', select=False)
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
        self.assertEqual(c['select'], c0.get().name)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), ['nmobjects', 'select'])
        self.assertEqual(c['nmobjects'], c0.names)
        self.assertEqual(c['select'], c0.get().name)
        # tree_path()
        self.assertEqual(c0.tree_path(), name0)
        self.assertEqual(c0.tree_path_list(), [name0])
        # item_num()
        self.assertEqual(c0.item_num(n[0]), 0)
        self.assertEqual(c0.item_num(n[1]), 1)
        self.assertEqual(c0.item_num(n[2]), 2)
        self.assertEqual(c0.item_num(n[3]), -1)  # skipped
        self.assertEqual(c0.item_num(n[4]), 3)
        self.assertEqual(c0.item_num(n[5]), 4)
        # exists()
        self.assertTrue(c0.exists(n[0]))
        self.assertTrue(c0.exists(n[1]))
        self.assertTrue(c0.exists(n[2]))
        self.assertFalse(c0.exists(n[3]))  # skipped
        self.assertTrue(c0.exists(n[4]))
        self.assertTrue(c0.exists(n[5]))
        # get()
        self.assertIsNone(c0.get(None))
        self.assertIsNone(c0.get('default'))
        self.assertIsNone(c0.get(BADNAME))
        o = c0.get()
        self.assertEqual(o, c0.select)
        o = c0.get('')
        self.assertEqual(o, c0.select)
        o = c0.get('select')
        self.assertEqual(o, c0.select)
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
        self.assertIsNone(c0.get(item_num=5))
        # get_items()
        olist0 = [c0.get(n[0]), c0.get(n[1]), c0.get(n[2]), c0.get(n[4])]
        olist1 = c0.get_items(names=[n[0], n[1], n[2], n[4], BADNAME])
        self.assertEqual(olist0, olist1)
        olist1 = c0.get_items(item_nums=[0, 1, 2, 3, 50, -50])
        self.assertEqual(olist0, olist1)
        # select()
        s = c0.select.name
        c0.select = 'select'  # bad
        self.assertEqual(c0.select.name, s)
        c0.select = 'default'  # bad
        self.assertEqual(c0.select.name, s)
        c0.select = BADNAME
        self.assertEqual(c0.select.name, s)
        c0.select = n[3]  # does not exist
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, s)
        c0.select = n[0]  # ok
        self.assertEqual(c0.select.name, n[0])
        # rename()
        self.assertFalse(c0.rename(None, n[3]))
        self.assertFalse(c0.rename('default', n[3]))
        self.assertFalse(c0.rename(BADNAME, n[3]))
        self.assertFalse(c0.rename('', n[0]))  # already exists
        self.assertFalse(c0.rename('', n[4]))  # already exists
        i = c0.item_num(n[4])
        self.assertFalse(c0.rename(n[4], None))
        self.assertFalse(c0.rename(n[4], ''))
        self.assertFalse(c0.rename(n[4], BADNAME))
        self.assertTrue(c0.rename(n[4], n[3]))
        o = c0.get(item_num=i)
        self.assertEqual(o.name, n[3])
        self.assertTrue(c0.rename(n[5], n[4]))
        self.assertFalse(c1.rename('', 'test'))  # cannot rename
        # duplicate()
        self.assertIsNone(c0.duplicate(None, 'default'))
        self.assertIsNone(c0.duplicate('default', 'default'))
        self.assertIsNone(c0.duplicate(BADNAME, 'default'))
        o = c0.get(n[0])
        oc = c0.duplicate(n[0], 'default')
        self.assertFalse(o._equal(oc, alert=True))
        self.assertTrue(o._equal(oc, ignore_name=True, alert=True))
        self.assertIsNone(c1.duplicate('select', 'default'))  # cannot dup.
        # _equal()
        self.assertFalse(c0._equal(c1, alert=True))
        self.assertFalse(c0._equal(c1, ignore_name=True, alert=True))
        # _copy()
        c1._copy(c0, copy_name=False)
        self.assertFalse(c1._equal(c0, alert=True))
        self.assertFalse(c1._equal(c0, ignore_name=True, alert=True))
        c1._copy(c0, copy_name=False, clear_before_copy=True)
        self.assertTrue(c1._equal(c0, ignore_name=True, alert=True))
        # Kill()
        self.assertFalse(c0.kill(name=None, ask=False))  # bad
        self.assertFalse(c0.kill(name='', ask=False))  # bad
        self.assertFalse(c0.kill(name=BADNAME, ask=False))
        s = c0.select.name
        self.assertTrue(c0.kill(name='select', ask=False))  # ok
        self.assertIsNone(c0.get(s))
        self.assertTrue(c0.kill(name=n[0], ask=False))  # ok
        self.assertIsNone(c0.get(n[0]))
        self.assertTrue(c1.kill(all_=True, ask=False))
        self.assertEqual(c1.count, 0)
        self.assertIsNone(c1.select)

    def test_data(self):
        nm.configs.quiet = False
        parent = self
        name0 = 'RecordA0'
        name1 = 'RecordA1'
        shape = [2, 2]
        fill_value = np.nan
        dtype = np.float64
        noise = [0, 1]
        dims0 = {'xstart': -10, 'xdelta': 0.01,
                 'xlabel': 'time', 'xunits': 'ms',
                 'ylabel': 'Vmem', 'yunits': 'mV'}
        dims1 = {'xstart': 10, 'xdelta': 0.05,
                 'xlabel': 'time', 'xunits': 's',
                 'ylabel': 'Imem', 'yunits': 'pA'}
        d0 = Data(parent, name0, fxns=nm._fxns, shape=shape,
                  fill_value=fill_value, noise=noise, dims=dims0)
        d1 = Data(parent, name1, fxns=nm._fxns, shape=shape,
                  fill_value=fill_value, noise=noise, dims=dims1)
        d0.dims = dims0
        d0.dims = dims1
        self.assertEqual(d0.dims, dims1)
        print(type(d0.np_array))
        print(isinstance(d0.np_array, np.ndarray))
        print(d0.np_array)
        d0.np_array.clip(-1, 1, out=d0.np_array)
        print(d0.np_array.flags)
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
        ds.make(channels=1, epochs=3, shape=5, noise=noise, dims=dims)
        f = p0.folder.new()
        ds = f.dataseries.new('Wave')
        ds.make(channels=2, epochs=3, shape=5, noise=noise, dims=dims)
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

    def test_history(self):
        quiet=True
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
