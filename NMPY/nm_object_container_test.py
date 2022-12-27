#!/usr/bin/env python[3]
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import unittest

from nm_manager import NMManager
from nm_object import NMObject
from nm_object_container import NMObjectContainer
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
        self.n0 = 'container0'
        self.n1 = 'container1'
        self.p0 = 'TestA'
        self.p1 = 'TestB'
        self.select0 = 0
        self.select1 = 1
        self.nlist0 = [self.p0 + str(i) for i in range(6)]
        self.nlist1 = [self.p1 + str(i) for i in range(3)]
        self.o0 = NMObject(parent=None, name='dummy')
        self.o1 = NMObject(parent=None, name='dummy')
        self.c0 = NMObjectContainer(parent=self.nm,
                                    name=self.n0,
                                    nmobject=self.o0,
                                    prefix=self.p0,
                                    rename=True)
        self.c1 = NMObjectContainer(parent=self.nm,
                                    name=self.n1,
                                    nmobject=self.o1,
                                    prefix=self.p1,
                                    rename=False)
        for n in self.nlist0:
            self.c0.new(n)
        for n in self.nlist1:
            self.c1.new(n)
        self.c0.select = self.p0 + str(self.select0)
        self.c1.select = self.p1 + str(self.select1)

    def test00_init(self):
        # arg: parent, name (NMObject)
        # arg: nmobject, prefix, rename, copy
        for b in BADTYPES:  # nmobject
            with self.assertRaises(TypeError):
                NMObjectContainer(parent=self.nm, name=self.n0, nmobject=b)
        for b in BADTYPES:  # prefix
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                NMObjectContainer(parent=self.nm, name=self.n0,
                                  nmobject=self.o0, prefix=b)
        for b in BADNAMES:  # prefix
            if b == '' or b == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                NMObjectContainer(parent=self.nm, name=self.n0,
                                  nmobject=self.o0, prefix=b)
        c0 = NMObjectContainer(parent=self.nm, name=self.n0, nmobject=self.o0,
                               prefix='')
        self.assertEqual(c0.prefix, '')  # '' is ok
        c0 = NMObjectContainer(parent=self.nm, name=self.n0, nmobject=self.o0,
                               prefix=None)
        self.assertEqual(c0.prefix, '')  # None is ok
        c0 = NMObjectContainer(parent=self.nm, name=self.n0, nmobject=self.o0,
                               prefix='default')
        self.assertEqual(c0.prefix, 'NMObject')

        # test self.c0 and self.c1
        self.assertTrue(isinstance(self.c0._nmobject, NMObject))
        self.assertTrue(isinstance(self.c1._nmobject, NMObject))
        self.assertTrue(self.c0._NMObjectContainer__rename)
        self.assertFalse(self.c1._NMObjectContainer__rename)
        self.assertEqual(self.c0.prefix, self.p0)
        self.assertEqual(self.c1.prefix, self.p1)
        self.assertEqual(self.c0.select.name, self.p0 + str(self.select0))
        self.assertEqual(self.c1.select.name, self.p1 + str(self.select1))
        self.assertEqual(len(self.c0._NMObjectContainer__container),
                         len(self.nlist0))
        self.assertEqual(len(self.c1._NMObjectContainer__container),
                         len(self.nlist1))

    def test01_eq_ne(self):
        for b in BADTYPES:
            self.assertFalse(self.c0 == b)
        self.assertTrue(self.c0 is self.c0)
        self.assertFalse(self.c0 is self.c1)
        self.assertTrue(self.c0 == self.c0)
        self.assertFalse(self.c0 == self.c1)
        self.assertFalse(self.c0 != self.c0)
        self.assertTrue(self.c0 != self.c1)

    def test02_copy(self):
        c = self.c0.copy()
        self.assertIsInstance(c, NMObjectContainer)
        self.assertTrue(self.c0 == c)  # __eq__
        self.assertEqual(self.c0._parent, c._parent)
        self.assertEqual(self.c0.name, c.name)
        self.assertTrue(self.c0._nmobject == c._nmobject)  # __eq__
        self.assertEqual(self.c0.prefix, c.prefix)
        self.assertEqual(self.c0._NMObjectContainer__rename,
                         c._NMObjectContainer__rename)
        fr0 = self.c0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        # self.assertNotEqual(fr0, frc)
        self.assertEqual(fr0, frc)  # names are the same, but refs different
        self.assertEqual(self.c0._name_set, c._name_set)
        p0 = self.c0.parameters
        p = c.parameters
        self.assertNotEqual(p0.get('created'), p.get('created'))
        self.assertNotEqual(p0.get('modified'), p.get('modified'))
        self.assertTrue(self.c0.select == c.select)  # __eq__
        self.assertEqual(self.c0.select.name, c.select.name)
        for i in range(self.c0.count):
            o0 = self.c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertTrue(o0 == oc)  # __eq__
            fr0 = o0._NMObject__rename_fxnref
            frc = oc._NMObject__rename_fxnref
            # self.assertNotEqual(fr0, frc)
            self.assertEqual(fr0, frc)
            self.assertEqual(fr0, self.c0.rename)
            self.assertEqual(frc, c.rename)
            self.assertEqual(self.c0.rename, c.rename)

    def test03_parameters(self):
        plist = ['name', 'created', 'modified', 'copy of', 'type', 'prefix',
                 'rename', 'select']
        self.assertEqual(self.c0.parameter_list, plist)

    def test04_content_type(self):
        self.assertEqual(self.c0.content_type, self.o0.__class__.__name__)

    def test05_prefix_set(self):
        # arg: newprefix
        for b in BADTYPES:  # newprefix
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0._prefix_set(newprefix=b)
        for b in BADNAMES:  # newprefix
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0._prefix_set(newprefix=b)
        for p in [None, self.p1, '', self.p0]:
            self.assertTrue(self.c0._prefix_set(newprefix=p))
            if p == '':
                self.assertEqual(self.c0.prefix, None)
            else:
                self.assertEqual(self.c0.prefix, p)
        with self.assertRaises(RuntimeError):
            self.c1.prefix = self.p0  # rename = False

    def test06_name_next(self):
        i = len(self.nlist0)
        self.assertEqual(self.c0.name_next_seq(), i)
        n = self.p0 + str(i)
        self.assertEqual(self.c0.name_next(), n)

    def test07_new(self):
        # args: name, select
        for b in BADTYPES:  # name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.new(name=b)
        for b in BADNAMES:  # name
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.new(name=b)
        self.c0._NMObjectContainer__container.clear()  # empty container
        o = self.c0.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, self.nlist0[0])
        self.assertEqual(self.c0.select, o)
        self.assertEqual(self.c0.select.name, self.nlist0[0])
        self.assertEqual(o._NMObject__rename_fxnref, self.c0.rename)
        with self.assertRaises(RuntimeError):
            self.c0.new(name=self.nlist0[0])  # already exists
        self.assertEqual(self.c0.name_next_seq(), 1)
        self.assertEqual(self.c0.name_next(), self.nlist0[1])
        o = self.c0.new(select=False)
        self.assertEqual(o.name, self.nlist0[1])
        self.assertEqual(self.c0.select.name, self.nlist0[0])
        self.assertEqual(self.c0.name_next_seq(), 2)
        self.assertEqual(self.c0.name_next(), self.nlist0[2])
        o = self.c0.new(name=self.nlist0[2])
        self.assertEqual(o.name, self.nlist0[2])
        # skip nlist0[3]
        o = self.c0.new(name=self.nlist0[4])
        self.assertEqual(o.name, self.nlist0[4])
        o = self.c0.new()
        self.assertEqual(o.name, self.nlist0[5])
        self.assertEqual(self.c0.count, 5)
        with self.assertRaises(RuntimeError):
            o.name = self.nlist0[0]  # name is used

    def test08_append(self):
        # args: nmobject, select
        for b in BADTYPES:  # nmobject
            with self.assertRaises(TypeError):
                self.c1.new(nmobject=b)
        self.c1._NMObjectContainer__container.clear()  # empty container
        o = NMObject(parent=self.nm, name=self.nlist1[0])
        self.assertTrue(self.c1.append(nmobject=o))
        o = self.c1.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, self.nlist1[1])

    def test09_names(self):
        self.assertEqual(self.c0.names, self.nlist0)
        self.assertEqual(self.c1.names, self.nlist1)

    def test10_content(self):
        content_name = 'NMObjects'
        c = self.c0.content
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], self.c0.names)
        c = self.c0.content_tree
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], self.c0.names)

    def test11_index(self):
        # arg: name
        for b in BADTYPES:  # name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.index(b)
        for b in BADNAMES:  # name
            if b.lower() == 'select':
                continue  # ok
            self.assertEqual(self.c0.index(b), -1)
        self.assertEqual(self.c0.index('doesnotexist'), -1)
        for i in range(len(self.nlist0)):
            self.assertEqual(self.c0.index(self.nlist0[i]), i)
        self.assertEqual(self.c0.index('select'), self.select0)
        self.assertEqual(self.c1.index('select'), self.select1)

    def test12_exists(self):
        # arg: name
        for b in BADTYPES:  # name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.exists(b)
        for b in BADNAMES:  # name
            if b.lower() == 'select':
                self.assertTrue(self.c0.exists(b))
            else:
                self.assertFalse(self.c0.exists(b))
        self.assertFalse(self.c0.exists('doesnotexist'))
        for i in range(len(self.nlist0)):
            self.assertTrue(self.c0.exists(self.nlist0[i]))

    def test13_getitem(self):
        # args: name, index
        for b in BADTYPES:  # name
            if isinstance(b, str):
                continue  # ok
            self.assertIsNone(self.c0.getitem(name=b))
        for b in BADNAMES:  # name
            if b.lower() == 'select':
                continue  # ok
            self.assertIsNone(self.c0.getitem(name=b))
        for b in BADTYPES:  # index
            if isinstance(b, int):
                continue  # ok
            self.assertIsNone(self.c0.getitem(index=b))
        self.assertIsNone(self.c0.getitem(name=''))
        self.assertEqual(self.c0.getitem(name='select'), self.c0.select)
        for i in range(len(self.nlist0)):
            o = self.c0.getitem(name=self.nlist0[i])
            self.assertIsInstance(o, NMObject)
            self.assertEqual(o.name, self.nlist0[i])
        self.assertIsNone(self.c0.getitem(index=None))
        for i in range(self.c0.count):
            o = self.c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            self.assertEqual(o.name, self.nlist0[i])
        for i in range(-1, -1 * (self.c0.count + 1), -1):
            o = self.c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            self.assertEqual(o.name, self.nlist0[i])
        i = -1 * (self.c0.count + 1)
        for b in [self.c0.count, 100, i, -100]:
            with self.assertRaises(IndexError):
                self.c0.getitem(index=b)

    def test14_getitems(self):
        # args: names, indexes
        for b in BADTYPES:  # names
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.getitems(names=b)
        for b in BADTYPES:  # names
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.getitems(names=[b])
        for b in BADNAMES:  # names
            if b.lower() == 'select' or b.lower() == 'all' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.getitems(names=[b])
        for b in BADTYPES:  # indexes
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.getitems(indexes=b)
        for b in BADTYPES:  # indexes
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.getitems(indexes=[b])
        self.assertEqual(self.c0.getitems(), [])
        i = -1 * (self.c0.count + 1)
        for b in [self.c0.count, 50, i]:
            with self.assertRaises(IndexError):
                olist1 = self.c0.getitems(indexes=[b])
        olist0 = self.c0.getitems(names=[self.nlist0[0], self.nlist0[1],
                                  self.nlist0[2], self.nlist0[4]])
        olist1 = self.c0.getitems(indexes=[0, 1, 2, 3])
        self.assertEqual(olist0, olist1)
        olist1 = self.c0.getitems(indexes=[-5, -4, -3, -2])
        self.assertEqual(olist0, olist1)
        olist0 = self.c0.getitems(names='all')
        self.assertEqual(len(olist0), self.c0.count)
        olist0.pop()  # pop should NOT change container list
        self.assertEqual(len(olist0), self.c0.count-1)

    def test15_select_set(self):
        # args: name, index, failure_alert
        for b in BADTYPES:  # test name
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0._select_set(name=b)
        for b in BADNAMES:  # test name
            if b == '' or b == 'none' or b == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0._select_set(name=b)
        for b in BADTYPES:  # test index
            if b is None or isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0._select_set(index=b)
        sname = self.c0.select.name
        self.assertIsNone(self.c0._select_set())
        self.assertIsNone(self.c0._select_set(name=None))
        self.assertIsNone(self.c0._select_set(name=''))
        self.assertIsNone(self.c0._select_set(index=None))
        self.assertIsNone(self.c0._select_set(name=self.nlist0[3]))  # 'no'
        if self.nm.configs.quiet:
            self.assertEqual(self.c0.select.name, sname)
        self.assertIsInstance(self.c0._select_set(name=self.nlist0[0]),
                              NMObject)
        self.assertEqual(self.c0.select.name, self.nlist0[0])

    def test16_rename(self):
        # args: name, newname
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.rename(b, self.nlist0[3])
        for b in BADNAMES:  # test name
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.rename(b, self.nlist0[3])
        for b in BADTYPES:  # test newname
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.rename(self.nlist0[4], b)
        for b in BADNAMES:  # test newname
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.rename(self.nlist0[4], b)
        with self.assertRaises(RuntimeError):
            self.c0.rename('select', self.nlist0[0])  # name already used
        with self.assertRaises(RuntimeError):
            self.c1.rename('select', 'test')  # rename = False
        # c0.names = ['TestA0', 'TestA1', 'TestA2', 'TestA4', 'TestA5']
        s = self.c0.rename('select', self.nlist0[3])
        self.assertIsInstance(s, str)
        self.assertEqual(s, self.nlist0[3])
        # c0.names = ['TestA3', 'TestA1', 'TestA2', 'TestA4', 'TestA5']
        i = self.c0.index(self.nlist0[4])  # 'TestA4', i = 3
        nnext = self.c0.name_next()
        self.assertEqual(nnext, 'TestA6')
        s = self.c0.rename(self.nlist0[4], 'default')
        self.assertEqual(s, nnext)
        # c0.names = ['TestA3', 'TestA1', 'TestA2', 'TestA6', 'TestA5']
        for i in range(5):
            o = self.c0.getitem(index=i)
            self.c0.rename(o.name, 'temp' + str(i))
        for i in range(5):
            o = self.c0.getitem(index=i)
            self.c0.rename(o.name, self.nlist0[i])
        # c0.names = ['TestA0', 'TestA1', 'TestA2', 'TestA3', 'TestA4']
        self.assertEqual(self.c0.names, [self.nlist0[0],
                                         self.nlist0[1],
                                         self.nlist0[2],
                                         self.nlist0[3],
                                         self.nlist0[4]])

    def test17_duplicate(self):
        # args: name, newname, select
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.duplicate(b, 'default')
        for b in BADNAMES:  # test name
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.duplicate(b, 'default')
        for b in BADTYPES:  # test newname
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.c0.duplicate(self.nlist0[0], b)
        for b in BADNAMES:  # test newname
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                self.c0.duplicate(self.nlist0[0], b)
        with self.assertRaises(RuntimeError):
            self.c0.duplicate(self.nlist0[0], self.nlist0[1])  # already exists
        # self.assertIsNone(c0.duplicate('', 'default'))
        icount = self.c0.count
        o = self.c0.getitem(name=self.nlist0[0])
        nnext = self.c0.name_next()  # 'TestA5'
        c = self.c0.duplicate(self.nlist0[0], 'default')
        self.assertIsInstance(c, NMObject)
        self.assertEqual(c.name, nnext)
        self.assertEqual(c._NMObject__rename_fxnref, self.c0.rename)
        self.assertFalse(o._isequivalent(c, alert=ALERT))  # different names
        self.assertEqual(self.c0.count, icount + 1)
        with self.assertRaises(RuntimeError):
            c.name = self.nlist0[0]  # name already used
        c._NMObject__name = self.nlist0[0]  # manual name change
        self.assertTrue(o._isequivalent(c, alert=ALERT))
        c._NMObject__name = nnext  # back to default name

        # self.c0.notes_print()
        # c0.getitem(index=0).notes_print()

    def test18_remove(self):
        # args: names, indexes, confirm
        # uses getitems, so no need to test BADTYPES and BADNAMES
        self.assertEqual(self.c0.remove(), [])  # nothing killed
        select = self.c0.select
        klist = self.c0.remove(names='select', confirm=CONFIRM)
        self.assertEqual(klist, [select])
        self.assertIsNone(self.c0.getitem(name=select.name))
        self.assertEqual(self.c0.select.name, self.nlist0[0])
        o = self.c0.getitem(name=self.nlist0[0])
        klist = self.c0.remove(names=self.nlist0[0], confirm=CONFIRM)
        self.assertEqual(klist, [o])
        self.assertIsNone(self.c0.getitem(name=self.nlist0[0]))
        self.assertEqual(self.c0.select.name, self.nlist0[1])
        names = self.c0.names
        klist = self.c0.remove(names='all', confirm=CONFIRM)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(self.c0.count, 0)
        self.assertIsNone(self.c0.select)
        olist = self.c1.getitems(indexes=[0, 1, 2])
        klist = self.c1.remove(indexes=[0, 1, 2], confirm=CONFIRM)
        self.assertEqual(klist, olist)
        self.assertIsNone(self.c1.select)


if __name__ == '__main__':
    unittest.main()
