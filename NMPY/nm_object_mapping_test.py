#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:43:19 2022

@author: jason
"""

import unittest

from nm_manager import NMManager
from nm_object import NMObject
from nm_object_mapping import NMObjectMapping
import nm_preferences as nmp

NM = NMManager(new_project=False, quiet=True)
BADTYPES = [None, True, 1, 3.14, [], (), {}, set(), 'test', NM]
# BADTYPES: all types, use continue to ignore OK types
BADNAME = 'b&dn@me!'
BADNAMES = nmp.BAD_NAMES + [BADNAME]
ALERT = True
CONFIRM = False


class NMObjectContainerTest(unittest.TestCase):

    def setUp(self):
        self.nm = NMManager(new_project=False, quiet=True)
        self.n0 = 'map0'
        self.n1 = 'map1'
        self.p0 = 'TestA'
        self.p1 = 'TestB'
        self.select0 = 0
        self.select1 = 1
        self.nlist0 = [self.p0 + str(i) for i in range(6)]
        self.nlist1 = [self.p1 + str(i) for i in range(3)]
        self.o0 = NMObject(parent=None, name='dummy')
        self.o1 = NMObject(parent=None, name='dummy')
        self.m0 = NMObjectMapping(
            parent=self.nm,
            name=self.n0,
            nmobjects=self.o0,
            allow_renaming=True,
            default_prefix=self.p0,
        )
        self.m1 = NMObjectMapping(
            parent=self.nm,
            name=self.n1,
            nmobjects=self.o1,
            allow_renaming=False,
            default_prefix=self.p1,
        )

    def test00_init(self):
        # arg: parent, name (NMObject)
        # arg: nmobjects, prefix, rename, copy

        bad = list(BADTYPES)
        bad.remove(None)
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectMapping(nmobjects=b)  # move code to test_update()

        bad = list(BADTYPES)
        bad.remove(True)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectMapping(allow_renaming=b)

        bad = list(BADTYPES)
        bad.remove(None)
        bad.remove('test')
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectMapping(default_prefix=b)

        bad = list(BADNAMES)
        bad.remove('')
        bad.remove('default')
        for b in bad:
            with self.assertRaises(ValueError):
                NMObjectMapping(default_prefix=b)

        NMObjectMapping()
