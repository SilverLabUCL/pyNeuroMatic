#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import inspect
import unittest
from nm_container import NMObject
from nm_container import Container
from nm_manager import Manager
import nm_utilities as nmu


class Test(unittest.TestCase):

    def test_nmobject(self):
        manager = Manager(quiet=True)
        parent = self
        name = 'test'
        name2 = 'testing'
        fxns = manager._Manager__fxns
        o = NMObject(manager, parent, name, fxns, rename=True)
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, name)
        o.name = name2
        self.assertEqual(o.name, name2)
        self.assertEqual(o.content_tree, {'nmobject': name2})
        self.assertEqual(o.tree_path(), name2)
        self.assertEqual(o.tree_path_list(), [name2])
        o = NMObject(manager, parent, name, fxns, rename=False)
        o.name = name2
        self.assertEqual(o.name, name)

    def test_container(self):
        manager = Manager(quiet=True)
        parent = self
        name = 'ContainerTest'
        fxns = manager._Manager__fxns
        prefix = 'test'
        type_ = 'NMObject'
        c = Container(manager, parent, name, fxns, type_, prefix, rename=True,
                      duplicate=True)
        self.assertIsInstance(c, Container)
        name0 = prefix + '0'
        name1 = prefix + '1'
        name2 = prefix + '2'
        name3 = prefix + '3'
        name4 = prefix + '4'
        name5 = prefix + '5'
        o = c.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, name0)
        o = c.new()
        self.assertEqual(o.name, name1)
        o = c.new(name=name0)
        self.assertIsNone(o)  # already exists
        o = c.new(name=name2)
        self.assertEqual(o.name, name2)
        o = c.new(nmobj=self)
        self.assertIsNone(o)  # not a NMObject
        o = NMObject(manager, parent, name4, fxns, rename=True)
        o = c.new(name=name4, nmobj=o)
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, name4)
        o = c.new()
        self.assertEqual(o.name, name5)

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
        self.assertFalse(nmu.name_ok(['test0', 'test1', 'test2']))

    def test_names_ok(self):
        self.assertTrue(nmu.names_ok('test'))
        self.assertFalse(nmu.names_ok('*'))
        self.assertFalse(nmu.names_ok(None))
        self.assertFalse(nmu.names_ok(0))
        self.assertTrue(nmu.names_ok(['test']))
        self.assertTrue(nmu.names_ok(['test0', 'test1', 'test2']))
        self.assertFalse(nmu.names_ok(['test0', 'test1', 'test2?']))
        self.assertFalse(nmu.names_ok(['test0', 1, 'test2']))

    def test_number_ok(self):
        self.assertTrue(nmu.number_ok(0))
        self.assertTrue(nmu.number_ok(-5))
        self.assertTrue(nmu.number_ok(1.34))
        self.assertTrue(nmu.number_ok(float('inf'), no_inf=False))
        self.assertTrue(nmu.number_ok(float('-inf'), no_inf=False))
        self.assertTrue(nmu.number_ok(float('nan'), no_nan=False))
        self.assertFalse(nmu.number_ok(float('inf')))
        self.assertFalse(nmu.number_ok(float('-inf')))
        self.assertFalse(nmu.number_ok(float('nan')))
        self.assertTrue(nmu.number_ok(0, no_neg=True))
        self.assertTrue(nmu.number_ok(1.34, no_neg=True))
        self.assertFalse(nmu.number_ok(-1.34, no_neg=True))
        self.assertTrue(nmu.number_ok(0, no_pos=True))
        self.assertFalse(nmu.number_ok(1.34, no_pos=True))
        self.assertTrue(nmu.number_ok(-1.34, no_pos=True))
        self.assertFalse(nmu.number_ok(0, no_zero=True))
        self.assertTrue(nmu.number_ok(1.34, no_zero=True))
        self.assertTrue(nmu.number_ok(-1.34, no_zero=True))
        self.assertFalse(nmu.number_ok(None))
        self.assertFalse(nmu.number_ok([0]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34]))

    def test_numbers_ok(self):
        self.assertTrue(nmu.numbers_ok(0))
        self.assertTrue(nmu.numbers_ok(-5))
        self.assertTrue(nmu.numbers_ok(1.34))
        self.assertTrue(nmu.numbers_ok(float('inf'), no_inf=False))
        self.assertTrue(nmu.numbers_ok(float('-inf'), no_inf=False))
        self.assertTrue(nmu.numbers_ok(float('nan'), no_nan=False))
        self.assertFalse(nmu.numbers_ok(float('inf')))
        self.assertFalse(nmu.numbers_ok(float('-inf')))
        self.assertFalse(nmu.numbers_ok(float('nan')))
        self.assertTrue(nmu.numbers_ok([0, 3, 4], no_neg=True))
        self.assertFalse(nmu.numbers_ok([-1, 3, 4], no_neg=True))
        self.assertFalse(nmu.numbers_ok([0, 3, 4], no_pos=True))
        self.assertTrue(nmu.numbers_ok([0, -3, -4], no_pos=True))
        self.assertFalse(nmu.numbers_ok([0, 3, 4], no_zero=True))
        self.assertTrue(nmu.numbers_ok([-4, 4], no_zero=True))
        self.assertFalse(nmu.numbers_ok(None))
        self.assertTrue(nmu.numbers_ok([0, -5, 1.34]))
        self.assertFalse(nmu.numbers_ok([0, -5, 1.34, 'test']))

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

    def test_alert(self):
        r = 'ALERT: nm.Test.test_alert: test'
        self.assertEqual(nmu.alert('test'), r)
        tp = 'one.two.three'
        r = 'ALERT: nm.one.two.three.test_alert: test'
        self.assertEqual(nmu.alert('test', tp=tp), r)
        self.assertEqual(nmu.alert('test', quiet=True), '')

    def test_error(self):
        r = 'ERROR: nm.Test.test_error: test'
        self.assertEqual(nmu.error('test'), r)
        tp = 'one.two.three'
        r = 'ERROR: nm.one.two.three.test_error: test'
        self.assertEqual(nmu.error('test', tp=tp), r)
        self.assertEqual(nmu.error('test', quiet=True), '')

    def test_history(self):
        r = 'nm.Test.test_history: test'
        self.assertEqual(nmu.history('test'), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.test_history: test'
        self.assertEqual(nmu.history('test', tp=tp), r)
        self.assertEqual(nmu.history('test', quiet=True), '')

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
