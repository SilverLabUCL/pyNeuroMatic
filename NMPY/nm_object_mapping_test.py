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
        self.nlist0 = [self.p0 + str(i) for i in range(6)]
        self.nlist1 = [self.p1 + str(i) for i in range(3)]
        self.olist0 = []
        self.olist1 = []
        for n in self.nlist0:
            self.olist0.append(NMObject(parent=NM, name=n))
        for n in self.nlist1:
            self.olist1.append(NMObject(parent=NM, name=n))
        '''
        self.olist0 = {}
        self.olist1 = {}
        for n in self.nlist0:
            self.olist0.update({n: NMObject(parent=NM, name=n)})
        for n in self.nlist1:
            self.olist1.update({n: NMObject(parent=NM, name=n)})
        '''
        self.m0 = NMObjectMapping(
            parent=self.nm,
            name=self.n0,
            nmobjects=self.olist0,
            rename_on=True,
            name_prefix=self.p0,
        )
        self.m1 = NMObjectMapping(
            parent=self.nm,
            name=self.n1,
            nmobjects=self.olist1,
            rename_on=False,
            name_prefix=self.p1,
        )

    def xtest00_init(self):
        # arg: parent, name (NMObject)
        # arg: nmobjects, prefix, rename, copy

        # for nmobjects, see test_update()

        bad = list(BADTYPES)
        bad.remove(True)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectMapping(rename_on=b)
        bad = list(BADTYPES)  # move to test_name_prefix_set()
        bad.remove(None)
        bad.remove('test')
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectMapping(name_prefix=b)
        bad = list(BADNAMES)  # move to test_name_prefix_set()
        bad.remove('')
        bad.remove('default')
        for b in bad:
            with self.assertRaises(ValueError):
                NMObjectMapping(name_prefix=b)

        NMObjectMapping()

        self.assertEqual(self.m0._NMObject__parent, self.nm)
        self.assertEqual(self.m1._NMObject__parent, self.nm)
        self.assertEqual(self.m0._NMObject__name, self.n0)
        self.assertEqual(self.m1._NMObject__name, self.n1)
        self.assertTrue(self.m0._NMObjectMapping__rename_on)
        self.assertFalse(self.m1._NMObjectMapping__rename_on)
        self.assertEqual(self.m0._NMObjectMapping__name_prefix, self.p0)
        self.assertEqual(self.m1._NMObjectMapping__name_prefix, self.p1)
        self.assertEqual(self.m0._NMObjectMapping__select, [])
        self.assertEqual(self.m1._NMObjectMapping__select, [])
        self.assertEqual(len(self.m0._NMObjectMapping__map), len(self.nlist0))
        self.assertEqual(len(self.m1._NMObjectMapping__map), len(self.nlist1))

    def test01_setitem(self):
        with self.assertRaises(RuntimeError):
            self.m0['TestA0'] = NMObject(parent=NM, name='test')
            # cannot use '='

    def test02_update(self):
        n1 = 'test1'
        o1 = NMObject(parent=NM, name=n1)
        o2 = NMObject(parent=NM, name=n1.upper())

        bad = list(BADTYPES)
        bad.remove(None)
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                self.m0.update(b)
            with self.assertRaises(TypeError):
                self.m0.update([o1, b])

        rfr_before = o1._NMObject__rename_fxnref
        self.m0.update(o1)
        self.assertEqual(len(self.m0), len(self.nlist0) + 1)
        rfr_after = o1._NMObject__rename_fxnref
        self.assertNotEqual(rfr_before, rfr_after)
        self.assertEqual(rfr_after, self.m0.rename)
        for i, k in enumerate(self.m0.keys()):
            if k.lower() == n1.lower():
                self.assertEqual(i, len(self.m0)-1)
        with self.assertRaises(KeyError):
            self.m0.update(o1)  # name in use
        with self.assertRaises(KeyError):
            self.m0.update(o2)  # name in use

    def test03_getitem(self):
        # get(), items(), values()
        self.assertIsNone(self.m0.get('test'))
        for i, n in enumerate(self.nlist0):
            self.assertEqual(self.m0.get(n), self.olist0[i])
            self.assertEqual(self.m0.get(n.upper()), self.olist0[i])
            # case insensitive
        for i, (k, v) in enumerate(self.m0.items()):
            self.assertEqual(k, self.nlist0[i])
            self.assertEqual(v, self.olist0[i])
        for i, v in enumerate(self.m0.values()):
            self.assertEqual(v, self.olist0[i])

    def test04_setdefault(self):
        # calls __getitem__()
        self.assertIsNone(self.m0.setdefault('test'))
        self.assertEqual(self.m0.get(self.nlist0[0]), self.olist0[0])
        self.assertEqual(self.m0.setdefault('test', default='ok'), 'ok')

    def test05_delitem(self):
        with self.assertRaises(RuntimeError):
            del self.m0[self.nlist0[0]]  # deprecated

    def test06_pop(self):
        with self.assertRaises(KeyError):
            self.m0.pop('test')
        self.assertEqual(self.m0.pop('test', default='ok'), 'ok')
        for i, n in enumerate(self.nlist0):
            self.assertEqual(self.m0.pop(n, confirm=CONFIRM), self.olist0[i])
        self.assertEqual(len(self.m0), 0)

    def test07_popitem(self):
        for i in reversed(range(len(self.nlist0))):
            t = (self.nlist0[i], self.olist0[i])
            self.assertEqual(self.m0.popitem(confirm=CONFIRM), t)
        self.assertEqual(self.m0.popitem(confirm=CONFIRM), ())

    def test08_clear(self):
        self.m0.clear(confirm=CONFIRM)
        self.assertEqual(len(self.m0), 0)

    def test09_iter(self):
        o_iter = iter(self.m0)
        for n in self.nlist0:
            self.assertEqual(next(o_iter), n)

    def test10_len(self):
        self.assertEqual(len(self.m0), len(self.nlist0))

    def test11_contains(self):
        for n in self.nlist0:
            self.assertTrue(n in self.m0)
            self.assertTrue(n.upper() in self.m0)  # case insensitive
            self.assertFalse(n in self.m1)

    def test12_eq_ne(self):
        for b in BADTYPES:
            self.assertFalse(self.m0 == b)
        self.assertTrue(self.m0 is self.m0)
        self.assertFalse(self.m0 is self.m1)
        self.assertTrue(self.m0 == self.m0)
        self.assertFalse(self.m0 == self.m1)
        self.assertFalse(self.m0 != self.m0)
        self.assertTrue(self.m0 != self.m1)
        olist0 = []
        for n in self.nlist0:
            olist0.append(NMObject(parent=NM, name=n))
        m0 = NMObjectMapping(
            parent=self.nm,
            name=self.n0,
            nmobjects=olist0,
            rename_on=True,
            name_prefix=self.p0,
        )
        self.assertTrue(m0 == self.m0)
        self.assertFalse(m0 != self.m0)
        