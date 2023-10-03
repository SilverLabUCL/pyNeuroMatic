#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:43:19 2022

@author: jason
"""
import unittest

from nm_channel import NMChannel, NMChannelContainer
from nm_data import NMData
from nm_manager import NMManager
from nm_scale import NMScale, NMScaleX
import nm_utilities as nmu

NM = NMManager(quiet=True)

CNAME0 = 'A'
CNAME1 = 'B'
YSCALE0 = {'label': 'Vm', 'units': 'mV'}
XSCALE0 = {'label': 'time', 'units': 'ms', 'start': 10, 'delta': 0.01}
YSCALE1 = {'label': 'Im', 'units': 'pA'}
XSCALE1 = {'label': 'time', 'units': 's', 'start': 0, 'delta': 0.2}

DNLIST0 = ['dataA' + str(i) for i in range(8)]
DNLIST1 = ['recordB' + str(i) for i in range(11)]


class NMChannelTest(unittest.TestCase):

    def setUp(self):  # executed before each test

        self.c0 = NMChannel(parent=NM, name=CNAME0, xscale=XSCALE0,
                            yscale=YSCALE0)

        self.dolist0 = []
        for n in DNLIST0:
            d = NMData(parent=NM, name=n)
            self.c0.data.append(d)
            self.dolist0.append(d)

        self.c1 = NMChannel(parent=NM, name=CNAME1, xscale=XSCALE1,
                            yscale=YSCALE1)

        self.dolist1 = []
        for n in DNLIST1:
            d = NMData(parent=NM, name=n)
            self.c1.data.append(d)
            self.dolist1.append(d)

        self.c0_copy = NMChannel(parent=None, name='test', copy=self.c0)

    def test00_init(self):
        # args: parent, name, copy (see NMObject)
        # args: xscale, yscale (see NMScale, NMScaleX)
        bad = list(nmu.BADTYPES)
        bad.remove(None)
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                NMChannel(xscale=b)
            with self.assertRaises(TypeError):
                NMChannel(yscale=b)

        with self.assertRaises(TypeError):
            NMChannel(copy=NM)

        self.assertEqual(self.c0._parent, NM)
        self.assertEqual(self.c0.name, CNAME0)

        for key in XSCALE0.keys():
            self.assertEqual(self.c0.x.scale[key], XSCALE0[key])
        for key in YSCALE0.keys():
            self.assertEqual(self.c0.y.scale[key], YSCALE0[key])

        for i, o in enumerate(self.c0.data):
            self.assertEqual(o.name, DNLIST0[i])

        self.assertEqual(self.c0_copy._parent, NM)
        self.assertEqual(self.c0_copy.name, CNAME0)
        self.assertEqual(self.c0_copy.x, self.c0.x)
        self.assertEqual(self.c0_copy.y, self.c0.y)
        self.assertTrue(isinstance(self.c0_copy.x, NMScaleX))
        self.assertTrue(isinstance(self.c0_copy.x, NMScale))
        self.assertTrue(isinstance(self.c0_copy.y, NMScale))
        self.assertFalse(isinstance(self.c0_copy.y, NMScaleX))

        for i, o in enumerate(self.c0_copy.data):
            self.assertEqual(o.name, DNLIST0[i])

    def test01_parameters(self):
        klist = ['name', 'created', 'copy of']  # NMObject
        klist += ['xscale', 'yscale']
        plist = self.c0.parameters
        self.assertEqual(klist, list(plist.keys()))
        self.assertEqual(plist['xscale'], XSCALE0)
        self.assertEqual(plist['yscale'], YSCALE0)

    def test02_eq(self):
        # args; other
        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.c0 == b)
        self.assertFalse(self.c0 == self.c1)

        c0 = NMChannel(parent=NM, name=CNAME0, xscale=XSCALE0, yscale=YSCALE0)

        self.assertTrue(len(c0.data) != len(self.c0.data))
        self.assertFalse(c0 == self.c0)
        self.assertTrue(c0.x == self.c0.x)
        self.assertTrue(c0.y == self.c0.y)

        for n in DNLIST0:
            d = NMData(parent=NM, name=n)
            c0.data.append(d)

        self.assertTrue(len(c0.data) == len(self.c0.data))
        self.assertTrue(c0 == self.c0)
        self.assertFalse(c0 is self.c0)

        c0.name = CNAME1
        self.assertFalse(c0 == self.c0)
        self.assertTrue(c0 != self.c0)
        c0.name = CNAME0
        self.assertTrue(c0 == self.c0)
        c0.data.remove(self.dolist0[0])
        self.assertFalse(c0 == self.c0)

    def test03_copy(self):
        # TODO: test copy when copying NMFolder
        pass

    def test04_channel_container(self):
        # args: parent, name (see NMObject)
        # rename_on, name_prefix, name_seq_format, copy
        # see NMObjectContainer
        channels = NMChannelContainer(parent=NM, name='NMChannels')

        # parameters
        p = channels.parameters
        self.assertEqual(p['content_type'], 'nmchannel')
        self.assertFalse(p['rename_on'])
        self.assertEqual(p['name_prefix'], '')
        self.assertEqual(p['name_seq_format'], 'A')
        self.assertEqual(p['select'], None)

        # content_type
        self.assertEqual(channels.content_type(), 'NMChannel')

        # content_type_ok
        self.assertFalse(channels.content_type_ok(NM))
        self.assertTrue(channels.content_type_ok(self.c0))

        # name
        self.assertEqual(channels.name_next(), 'A')
        channels.name_prefix = 'Ch'
        self.assertEqual(channels.name_prefix, 'Ch')
        self.assertEqual(channels.name_next(), 'ChA')
        channels.name_prefix = ''  # reset

        # new
        self.assertEqual(channels.name_next(), 'A')
        c = channels.new(xscale=XSCALE0, yscale=YSCALE0)
        self.assertIsInstance(c, NMChannel)
        self.assertEqual(c.name, 'A')
        self.assertEqual(channels.name_next(), 'B')
        c = channels.new(xscale=XSCALE1, yscale=YSCALE1)
        self.assertEqual(c.name, 'B')
        self.assertEqual(channels.name_next(), 'C')
        self.assertEqual(channels.name_next(use_counter=True), 'C')
        self.assertEqual(channels._name_seq_counter(), 'C')
        channels._name_seq_counter_increment()
        self.assertEqual(channels._name_seq_counter(), 'D')
        self.assertEqual(channels.name_next(use_counter=True), 'D')
        self.assertEqual(channels.name_next(), 'C')

        # copy
        c = channels.copy()
        self.assertTrue(channels == c)
        self.assertFalse(channels != c)
        self.assertFalse(channels is c)
        c.name_seq_format = '0'
        self.assertFalse(channels == c)

        # equal
        c = NMChannelContainer(parent=NM, name='NMChannels')
        c.new(xscale=XSCALE0, yscale=YSCALE0)
        c.new(xscale=XSCALE1, yscale=YSCALE1)
        self.assertTrue(channels == c)
        c = NMChannelContainer(parent=NM, name='NMChannels')
        c.new(xscale=XSCALE0, yscale=YSCALE0)
        c.new(xscale=XSCALE0, yscale=YSCALE0)
        self.assertFalse(channels == c)
        c = NMChannelContainer(parent=NM, name='chans')
        c.new(xscale=XSCALE0, yscale=YSCALE0)
        c.new(xscale=XSCALE1, yscale=YSCALE1)
        self.assertFalse(channels == c)

        # duplicate
        c = channels.duplicate(key='A')
        self.assertEqual(c.name, 'C')
        a = channels.get('A')
        self.assertFalse(c == a)  # name is different
        self.assertFalse(c.name == a.name)
        self.assertTrue(c.x == a.x)
        self.assertTrue(c.y == a.y)
