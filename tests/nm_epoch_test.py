#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:35:21 2023

@author: jason
"""
import unittest

from pyneuromatic.nm_data import NMData
from pyneuromatic.nm_epoch import NMEpoch, NMEpochContainer
from pyneuromatic.nm_manager import NMManager
import pyneuromatic.nm_utilities as nmu

NM = NMManager(quiet=True)
ENAME0 = 'E0'
ENAME1 = 'E1'
DNLIST0 = ['dataA0', 'dataB0', 'dataC0']
DNLIST1 = ['dataA1', 'dataB1', 'dataC1']


class NMEpochTest(unittest.TestCase):

    def setUp(self):  # executed before each test

        self.e0 = NMEpoch(parent=NM, name=ENAME0, number=0)

        self.dolist0 = []
        for n in DNLIST0:
            d = NMData(parent=NM, name=n)
            self.e0.data.append(d)
            self.dolist0.append(d)

        self.e1 = NMEpoch(parent=NM, name=ENAME1, number=1)

        self.dolist1 = []
        for n in DNLIST1:
            d = NMData(parent=NM, name=n)
            self.e1.data.append(d)
            self.dolist1.append(d)

        self.e0_copy = NMEpoch(parent=None, name='test', copy=self.e0)

    def test00_init(self):
        # args: parent, name, copy (see NMObject)
        with self.assertRaises(TypeError):
            NMEpoch(copy=NM)

        self.assertEqual(self.e0._parent, NM)
        self.assertEqual(self.e0.name, ENAME0)
        self.assertEqual(self.e0.number, 0)

        for i, o in enumerate(self.e0.data):
            self.assertEqual(o.name, DNLIST0[i])

        self.assertEqual(self.e0_copy._parent, NM)
        self.assertEqual(self.e0_copy.name, ENAME0)
        self.assertEqual(self.e0_copy.number, 0)

        for i, o in enumerate(self.e0_copy.data):
            self.assertEqual(o.name, DNLIST0[i])

    def test02_eq(self):
        # args; other
        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.e0 == b)
        self.assertFalse(self.e0 == self.e1)

        e0 = NMEpoch(parent=NM, name=ENAME0, number=0)

        self.assertTrue(len(e0.data) != len(self.e0.data))
        self.assertFalse(e0 == self.e0)

        for n in DNLIST0:
            d = NMData(parent=NM, name=n)
            e0.data.append(d)

        self.assertTrue(len(e0.data) == len(self.e0.data))
        self.assertTrue(e0 == self.e0)

        e0.number = -1
        self.assertFalse(e0 == self.e0)

    def test03_copy(self):
        # TODO: test copy when copying NMFolder
        pass

    def test04_epoch_container(self):
        # args: parent, name
        # rename_on, name_prefix, name_seq_format, copy
        # see NMObjectContainer
        epochs = NMEpochContainer(parent=NM, name='NMEpochs')

        # parameters
        p = epochs.parameters
        self.assertEqual(p['content_type'], 'nmepoch')
        self.assertFalse(p['rename_on'])
        self.assertEqual(p['name_prefix'], 'E')
        self.assertEqual(p['name_seq_format'], '0')
        self.assertEqual(p['select'], None)

        # content_type
        self.assertEqual(epochs.content_type(), 'NMEpoch')

        # content_type_ok
        self.assertFalse(epochs.content_type_ok(NM))
        self.assertTrue(epochs.content_type_ok(self.e0))

        # name
        self.assertEqual(epochs.name_next(), 'E0')
        epochs.name_prefix = ''
        self.assertEqual(epochs.name_prefix, '')
        self.assertEqual(epochs.name_next(), '0')
        epochs.name_prefix = 'E'  # reset

        # new
        self.assertEqual(epochs.name_next(), 'E0')
        e = epochs.new()
        self.assertIsInstance(e, NMEpoch)
        self.assertEqual(e.name, 'E0')
        self.assertEqual(e.number, 0)
        self.assertEqual(epochs.name_next(), 'E1')
        e = epochs.new()
        self.assertEqual(e.name, 'E1')
        self.assertEqual(e.number, 1)
        self.assertEqual(epochs.name_next(), 'E2')
        self.assertEqual(epochs.name_next(use_counter=True), 'E2')
        self.assertEqual(epochs._name_seq_counter(), '2')
        epochs._name_seq_counter_increment()
        self.assertEqual(epochs._name_seq_counter(), '3')
        self.assertEqual(epochs.name_next(use_counter=True), 'E3')
        self.assertEqual(epochs.name_next(), 'E2')

        # copy
        c = epochs.copy()
        self.assertTrue(epochs == c)
        self.assertFalse(epochs != c)
        self.assertFalse(epochs is c)

        # equal
        c = NMEpochContainer(parent=NM, name='NMEpochs')
        c.new()
        c.new()
        self.assertTrue(epochs == c)
        c.name = 'test'
        self.assertFalse(epochs == c)

        # duplicate
        c = epochs.duplicate(key='E0')
        self.assertEqual(c.name, 'E2')
        e0 = epochs.get('E0')
        self.assertFalse(c == e0)  # name is different
        self.assertFalse(c.name == e0.name)
