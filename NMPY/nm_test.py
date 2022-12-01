#!/usr/bin/env python[3]
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import inspect
import unittest
import numpy as np
import time

from nm_channel import Channel
from nm_channel import ChannelContainer
from nm_data import Data
from nm_data import DataContainer
from nm_dataseries import DataSeries
from nm_dataseries import DataSeriesContainer
from nm_dataseries_set import DataSeriesSet
from nm_dataseries_set import DataSeriesSetContainer
from nm_dimension import NMDimension
from nm_dimension import NMDimensionX
from nm_manager import Manager
from nm_object import NMObject
# from nm_object import NMObjectTest
from nm_object_container import NMObjectContainer
from nm_note import NMNote
from nm_note import NMNoteContainer
from nm_project import Project
import nm_preferences as nmp
import nm_utilities as nmu

nm = Manager(new_project=False, quiet=True)
PARENT = nm
BADTYPES = [None, True, 1, 3.14, [], (), {}, set(), 'test', nm]
# BADTYPES: all types, use continue to ignore OK types
BADNAME = 'b&dn@me!'
BADNAMES = nmp.BAD_NAMES + [BADNAME]
PLIST = ['name', 'created', 'modified']  # NMObject parameter list
YDIM0 = {'offset': 0, 'label': 'Vmem', 'units': 'mV'}
XDIM0 = {'offset': 0, 'start': 10, 'delta': 0.01, 'label': 'time',
         'units': 'ms'}
YDIM1 = {'offset': 0, 'label': 'Imem', 'units': 'pA'}
XDIM1 = {'offset': 0, 'start': -10, 'delta': 0.2, 'label': 't',
         'units': 'seconds'}
YDIMx = {'offset': 0, 'label': 'time interval', 'units': 'usec'}
XDIMx = {'offset': 0, 'start': 0, 'delta': 1, 'label': 'sample',
         'units': '#'}
ALERT = True
CONFIRM = False


class Test(unittest.TestCase):

    def test_all(self):
        nm.configs.quiet = False
        # self._test_nmobject()
        # self._test_nmobject_container()
        # self._test_nmnote()
        # self._test_nmnote_container()
        self._test_dimension()
        # self._test_data()
        # self._test_data_container()
        # self._test_channel()
        # self._test_channel_container()
        # self._test_dataseries_set()
        # self._test_dataseries_set_container()
        # self._test_utilities()
        # breakpoint()

    def _test_nmobject(self):

        # __init__()
        # arg: parent, not testing since it can be any object
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                o0 = NMObject(PARENT, b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                o0 = NMObject(PARENT, b)
        n0 = 'object0'
        n1 = 'object1'
        o0 = NMObject(PARENT, n0)
        o1 = NMObject(PARENT, n1)
        self.assertEqual(o0._parent, nm)
        self.assertEqual(o0.name, n0)
        self.assertEqual(o0._NMObject__rename_fxnref, o0._name_set)

        # parameters() and parameter_list()
        self.assertEqual(o0.parameter_list, PLIST)

        # content()
        content_name = 'nmobject'
        self.assertEqual(o0.content, {content_name: o0.name})
        self.assertEqual(o0.content_tree, {content_name: o0.name})

        # treepath()
        self.assertEqual(o0._tp, o0.name)
        self.assertEqual(o0._tp_check(0), '')
        self.assertEqual(o0._tp_check(None), '')
        self.assertEqual(o0._tp_check(''), '')
        self.assertEqual(o0._tp_check('self'), o0.name)
        self.assertEqual(o0._tp_check('nm.test'), 'nm.test')
        self.assertEqual(o0.treepath(), o0.name)
        self.assertEqual(o0.treepath_list(), [o0.name])

        # name_ok()
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            self.assertFalse(o0.name_ok(b))
        for b in BADNAMES:
            self.assertFalse(o0.name_ok(b))
        self.assertEqual(o0._bad_names, nmp.BAD_NAMES)
        for b in o0._bad_names:
            self.assertFalse(o0.name_ok(b))
        for n in [n0, n1]:
            self.assertTrue(o0.name_ok(n))

        # _name_set()
        # arg: name_notused
        # arg: newname
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                o0._name_set('notused', b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                o0._name_set('notused', b)
        for n in ['test', n0]:
            self.assertTrue(o0._name_set('notused', n))
            self.assertEqual(n, o0.name)

        # rename_fxnref_set()
        # arg: rename_fxnref
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                o0._rename_fxnref_set(b)
        o0.name = 'test1'  # calls _name_set()
        self.assertEqual(o0.name, 'test1')
        self.assertTrue(o0._rename_fxnref_set(self.rename_dummy))
        o0.name = 'test2'
        self.assertEqual(o0.name, 'test1')  # name of o0 does not change

        # manager()
        self.assertEqual(o0._manager, nm)

        # isequivalent()
        # arg: nmobject
        for b in BADTYPES:
            self.assertFalse(o0._isequivalent(b, alert=ALERT))
        self.assertFalse(o0._isequivalent(o1, alert=ALERT))
        self.assertFalse(o0._isequivalent(o0, alert=ALERT))
        o0 = NMObject(PARENT, n0)
        o1 = NMObject(PARENT, n0)
        self.assertTrue(o0._isequivalent(o1, alert=ALERT))
        o0 = NMObjectTest(PARENT, n0)
        self.assertFalse(o0._isequivalent(o1, alert=ALERT))
        o1 = NMObjectTest(PARENT, n0)
        self.assertTrue(o0._isequivalent(o1, alert=ALERT))
        o0.myvalue = 1
        o1.myvalue = 2
        self.assertFalse(o0._isequivalent(o1, alert=ALERT))
        o0.myvalue = float('nan')
        o1.myvalue = float('nan')
        self.assertTrue(o0._isequivalent(o1, alert=ALERT))

        # copy()
        time.sleep(2)  # forces date to be different
        o0 = NMObject(PARENT, n0)
        c = o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(o0._isequivalent(c, alert=ALERT))
        self.assertEqual(o0._parent, c._parent)
        self.assertEqual(o0.name, c.name)
        self.assertNotEqual(o0._NMObject__created, c._NMObject__created)
        self.assertNotEqual(o0._NMObject__modified, c._NMObject__modified)
        self.assertEqual(c._NMObject__rename_fxnref, c._name_set)
        fr0 = o0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        self.assertNotEqual(fr0, frc)  # should be different

        # modified()
        m1 = o0._NMObject__modified
        o0._modified()
        self.assertNotEqual(m1, o0._NMObject__modified)

        # alert(), error(), history()
        # wrappers for nmu.history()
        # args: obj, type_expected, tp, quiet, frame
        dum_arg = {}
        e1 = o0._type_error('dum_arg', 'list')
        e2 = ('ERROR: object0: bad dum_arg: expected list but got dict')
        self.assertEqual(e1, e2)
        dum_str = 'test'
        e1 = o0._value_error('dum_str')
        e2 = ("ERROR: object0: bad dum_str: 'test'")
        self.assertEqual(e1, e2)

        # quiet()
        # arg: quiet
        self.assertFalse(o0._quiet(False))
        self.assertTrue(o0._quiet(True))
        q = nm.configs.quiet
        nm.configs.quiet = True  # Manager quiet overrides when True
        self.assertTrue(o0._quiet(False))
        self.assertTrue(o0._quiet(True))
        nm.configs.quiet = q

        # save()  # TODO

    def rename_dummy(self, oldname, newname, quiet=nmp.QUIET):
        # dummy function to test NMObject._rename_fxnref_set()
        print('test rename: ' + oldname + ' -> ' + newname)
        return False

    def _test_nmobject_container(self):

        # __init__()
        # arg: parent, not testing since it can be any object
        # arg: name, see _test_nmobject()
        # arg: type_, prefix, rename, copy
        n0 = 'container0'
        n1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        # type_ = 'NMObject'
        o0 = NMObject(None, 'dummy')
        # print(type(o0.__class__.__bases__[0]))
        nlist0 = [p0 + str(i) for i in range(0, 6)]
        nlist1 = [p1 + str(i) for i in range(0, 3)]
        # name is tested above for NMObject constructor
        for b in BADTYPES:  # test nmobject
            with self.assertRaises(TypeError):
                c0 = NMObjectContainer(PARENT, n0, nmobject=b)
        for b in BADTYPES:  # test prefix
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix=b)
        for b in BADNAMES:  # test prefix
            if b == '' or b == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix=b)
        c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix='')
        self.assertEqual(c0.prefix, '')  # '' is ok
        c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix=None)
        self.assertEqual(c0.prefix, '')  # None is ok
        c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix='default')
        self.assertEqual(c0.prefix, 'NMObject')
        c0 = NMObjectContainer(PARENT, n0, nmobject=o0, prefix=p0, rename=True)
        c1 = NMObjectContainer(PARENT, n1, nmobject=o0, prefix=p1,
                               rename=False)
        self.assertEqual(c0.parameters['type'], 'NMObject')
        self.assertEqual(c1.parameters['type'], 'NMObject')
        self.assertTrue(c0.parameters['rename'])
        self.assertFalse(c1.parameters['rename'])
        self.assertEqual(c0.prefix, p0)
        self.assertEqual(c1.prefix, p1)
        self.assertIsNone(c0.select)
        self.assertIsNone(c1.select)
        self.assertEqual(c0.names, [])
        self.assertEqual(c1.names, [])
        self.assertEqual(c0.count, 0)
        self.assertEqual(c1.count, 0)

        # parameters()
        plist = PLIST + ['type', 'prefix', 'rename', 'select']
        self.assertEqual(c0.parameter_list, plist)
        
        # content_type()
        self.assertEqual(c0.content_type, 'NMObject')
        
        # prefix_set()
        # arg: newprefix
        for b in BADTYPES:
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0._prefix_set(newprefix=b)
        for b in BADNAMES:
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0._prefix_set(newprefix=b)
        self.assertTrue(c0._prefix_set(newprefix=''))
        self.assertEqual(c0.prefix, None)
        for p in [None, p1, p0]:
            self.assertTrue(c0._prefix_set(newprefix=p))
            self.assertEqual(c0.prefix, p)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c0.prefix, p0)
        self.assertEqual(c1.prefix, p1)

        # name_next()
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), nlist0[0])

        # new()
        # args: name, select
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.new(name=b)
        for b in BADNAMES:  # test name
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.new(name=b)
        for b in BADTYPES:  # test nmobject
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0.new(nmobject=b)
        o = c0.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, nlist0[0])
        self.assertEqual(c0.select, o)
        self.assertEqual(c0.select.name, nlist0[0])
        self.assertEqual(o._NMObject__rename_fxnref, c0.rename)
        with self.assertRaises(RuntimeError):
            c0.new(name=nlist0[0])  # already exists
        self.assertEqual(c0.name_next_seq(), 1)
        self.assertEqual(c0.name_next(), nlist0[1])
        o = c0.new(select=False)
        self.assertEqual(o.name, nlist0[1])
        self.assertEqual(c0.select.name, nlist0[0])
        self.assertEqual(c0.name_next_seq(), 2)
        self.assertEqual(c0.name_next(), nlist0[2])
        o = c0.new(name=nlist0[2])
        self.assertEqual(o.name, nlist0[2])
        # skip nlist0[3]
        o = c0.new(name=nlist0[4])
        self.assertEqual(o.name, nlist0[4])
        o = c0.new()
        self.assertEqual(o.name, nlist0[5])
        self.assertEqual(c0.count, 5)
        # self.assertIsInstance(c1.new(), NMObject)
        # self.assertIsInstance(c1.new(), NMObject)
        # self.assertIsInstance(c1.new(), NMObject)
        with self.assertRaises(RuntimeError):
            o.name = nlist0[0]  # already exists

        # add()
        # args: nmobject, select
        for b in BADTYPES:  # test name
            with self.assertRaises(TypeError):
                c1.new(nmobject=b)
        o = NMObject(PARENT, nlist1[0])
        self.assertTrue(c1.add(nmobject=o))
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)

        # names()
        n0 = nlist0.copy()
        n0.remove(nlist0[3])
        self.assertEqual(c0.names, n0)
        self.assertEqual(c1.names, nlist1)

        # content()
        content_name = 'NMObjects'
        c = c0.content
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)

        # index()
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.index(b)
        for b in BADNAMES:
            if b.lower() == 'select':
                continue  # ok
            self.assertEqual(c0.index(b), -1)
        self.assertEqual(c0.index(nlist0[0]), 0)
        self.assertEqual(c0.index(nlist0[1]), 1)
        self.assertEqual(c0.index(nlist0[2]), 2)
        self.assertEqual(c0.index(nlist0[3]), -1)  # does not exist
        self.assertEqual(c0.index(nlist0[4]), 3)
        self.assertEqual(c0.index(nlist0[5]), 4)
        self.assertEqual(c0.index('select'), 4)
        self.assertEqual(c1.index('select'), 2)

        # exists()
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.exists(b)
        for b in BADNAMES:
            if b.lower() == 'select':
                self.assertTrue(c0.exists(b))
            else:
                self.assertFalse(c0.exists(b))
        for i in range(0, 6):
            if i == 3:
                self.assertFalse(c0.exists(nlist0[i]))
            else:
                self.assertTrue(c0.exists(nlist0[i]))

        # getitem()
        # args: name, index
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            self.assertIsNone(c0.getitem(name=b))
        for b in BADNAMES:  # test name
            if b.lower() == 'select':
                continue  # ok
            self.assertIsNone(c0.getitem(name=b))
        for b in BADTYPES:  # test index
            if isinstance(b, int):
                continue  # ok
            self.assertIsNone(c0.getitem(index=b))
        self.assertIsNone(c0.getitem(name=''))
        self.assertEqual(c0.getitem(name='select'), c0.select)
        for i in range(0, len(nlist0)):
            o = c0.getitem(name=nlist0[i])
            if i == 3:
                self.assertIsNone(o)
            else:
                self.assertIsInstance(o, NMObject)
                self.assertEqual(o.name, nlist0[i])
        self.assertIsNone(c0.getitem(index=None))
        for i in range(0, c0.count):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            if i <= 2:
                self.assertEqual(o.name, nlist0[i])
            else:
                self.assertEqual(o.name, nlist0[i+1])
        for i in range(-1, -1 * (c0.count + 1)):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
        i = -1 * (c0.count + 1)
        for b in [c0.count, 100, i, -100]:
            with self.assertRaises(IndexError):
                c0.getitem(index=b)

        # getitems()
        # args: names, indexes
        for b in BADTYPES:  # test names
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(names=b)
        for b in BADTYPES:  # test names
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(names=[b])
        for b in BADNAMES:  # test names
            if b.lower() == 'select' or b.lower() == 'all' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.getitems(names=[b])
        for b in BADTYPES:  # test indexes
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(indexes=b)
        for b in BADTYPES:  # test indexes
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(indexes=[b])
        self.assertEqual(c0.getitems(), [])
        i = -1 * (c0.count + 1)
        for b in [c0.count, 50, i]:
            with self.assertRaises(IndexError):
                olist1 = c0.getitems(indexes=[b])
        olist0 = c0.getitems(names=[nlist0[0], nlist0[1], nlist0[2],
                                    nlist0[4]])
        olist1 = c0.getitems(indexes=[0, 1, 2, 3])
        self.assertEqual(olist0, olist1)
        olist1 = c0.getitems(indexes=[-5, -4, -3, -2])
        self.assertEqual(olist0, olist1)
        olist0 = c0.getitems(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()  # pop should NOT change container list
        self.assertEqual(len(olist0), c0.count-1)

        # select_set()
        # args: name, index, failure_alert
        for b in BADTYPES:  # test name
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0._select_set(name=b)
        for b in BADNAMES:  # test name
            if b == '' or b == 'none' or b == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0._select_set(name=b)
        for b in BADTYPES:  # test index
            if b is None or isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0._select_set(index=b)
        sname = c0.select.name
        self.assertIsNone(c0._select_set())
        self.assertIsNone(c0._select_set(name=None))
        self.assertIsNone(c0._select_set(name=''))
        self.assertIsNone(c0._select_set(index=None))
        self.assertIsNone(c0._select_set(name=nlist0[3]))  # enter 'no'
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        self.assertIsInstance(c0._select_set(name=nlist0[0]), NMObject)
        self.assertEqual(c0.select.name, nlist0[0])

        # rename()
        # args: name, newname
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(b, nlist0[3])
        for b in BADNAMES:  # test name
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(b, nlist0[3])
        for b in BADTYPES:  # test newname
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(nlist0[4], b)
        for b in BADNAMES:  # test newname
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(nlist0[4], b)
        with self.assertRaises(RuntimeError):
            c0.rename('select', nlist0[0])  # name already used
        with self.assertRaises(RuntimeError):
            c1.rename('select', 'test')  # rename = False
        # c0.names = ['TestA0', 'TestA1', 'TestA2', 'TestA4', 'TestA5']
        s = c0.rename('select', nlist0[3])
        self.assertIsInstance(s, str)
        self.assertEqual(s, nlist0[3])
        # c0.names = ['TestA3', 'TestA1', 'TestA2', 'TestA4', 'TestA5']
        i = c0.index(nlist0[4])  # 'TestA4', i = 3
        nnext = c0.name_next()
        self.assertEqual(nnext, 'TestA6')
        s = c0.rename(nlist0[4], 'default')
        self.assertEqual(s, nnext)
        # c0.names = ['TestA3', 'TestA1', 'TestA2', 'TestA6', 'TestA5']
        for i in range(5):
            o = c0.getitem(index=i)
            c0.rename(o.name, 'temp' + str(i))
        for i in range(5):
            o = c0.getitem(index=i)
            c0.rename(o.name, nlist0[i])
        # c0.names = ['TestA0', 'TestA1', 'TestA2', 'TestA3', 'TestA4']
        self.assertEqual(c0.names, [nlist0[0], nlist0[1], nlist0[2], nlist0[3],
                                    nlist0[4]])

        # duplicate()
        # args: name, newname, select
        for b in BADTYPES:  # test name
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.duplicate(b, 'default')
        for b in BADNAMES:  # test name
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(b, 'default')
        for b in BADTYPES:  # test newname
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.duplicate(nlist0[0], b)
        for b in BADNAMES:  # test newname
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(nlist0[0], b)
        with self.assertRaises(RuntimeError):
            c0.duplicate(nlist0[0], nlist0[1])  # already exists
        # self.assertIsNone(c0.duplicate('', 'default'))
        icount = c0.count
        o = c0.getitem(name=nlist0[0])
        nnext = c0.name_next()  # 'TestA5'
        c = c0.duplicate(nlist0[0], 'default')
        self.assertIsInstance(c, NMObject)
        self.assertEqual(c.name, nnext)
        self.assertEqual(c._NMObject__rename_fxnref, c0.rename)
        self.assertFalse(o._isequivalent(c, alert=ALERT))  # different names
        self.assertEqual(c0.count, icount + 1)
        with self.assertRaises(RuntimeError):
            c.name = nlist0[0]  # name already used
        c._NMObject__name = nlist0[0]  # manual name change
        self.assertTrue(o._isequivalent(c, alert=ALERT))
        c._NMObject__name = nnext  # back to default name

        # isequivalent()
        # arg: NMObjectContainer
        for b in BADTYPES:
            self.assertFalse(c0._isequivalent(b, alert=ALERT))
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        self.assertFalse(c0._isequivalent(c0, alert=ALERT))
        # see copy() for test with something that is equivalent

        # copy()
        c = c0.copy()
        self.assertIsInstance(c, NMObjectContainer)
        self.assertTrue(c0._isequivalent(c, alert=ALERT))
        self.assertEqual(c0._parent, c._parent)
        self.assertEqual(c0.name, c.name)
        self.assertEqual(c0.parameters['type'], c.parameters['type'])
        self.assertEqual(c0.prefix, c.prefix)
        self.assertEqual(c0.parameters['rename'], c.parameters['rename'])
        fr0 = c0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        self.assertNotEqual(fr0, frc)
        self.assertNotEqual(c0._NMObject__created, c._NMObject__created)
        self.assertNotEqual(c0._NMObject__modified, c._NMObject__modified)
        self.assertNotEqual(c0.select, c.select)  # refs not equal
        self.assertEqual(c0.select.name, c.select.name)  # but names equal
        for i in range(0, c0.count):
            o0 = c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertTrue(o0._isequivalent(oc, alert=ALERT))
            fr0 = o0._NMObject__rename_fxnref
            frc = oc._NMObject__rename_fxnref
            self.assertNotEqual(fr0, frc)
            self.assertEqual(fr0, c0.rename)
            self.assertEqual(frc, c.rename)

        # c0.print_content()

        # remove()
        # args: names, indexes, confirm
        # uses getitems, so no need to test BADTYPES and BADNAMES
        self.assertEqual(c0.remove(), [])  # nothing killed
        select = c0.select
        klist = c0.remove(names='select', confirm=CONFIRM)
        self.assertEqual(klist, [select])
        self.assertIsNone(c0.getitem(name=select.name))
        self.assertEqual(c0.select.name, nlist0[0])
        o = c0.getitem(name=nlist0[0])
        klist = c0.remove(names=nlist0[0], confirm=CONFIRM)
        self.assertEqual(klist, [o])
        self.assertIsNone(c0.getitem(name=nlist0[0]))
        self.assertEqual(c0.select.name, nlist0[1])
        names = c0.names
        klist = c0.remove(names='all', confirm=CONFIRM)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(c0.count, 0)
        self.assertIsNone(c0.select)
        olist = c1.getitems(indexes=[0, 1, 2])
        klist = c1.remove(indexes=[0, 1, 2], confirm=CONFIRM)
        self.assertEqual(klist, olist)
        self.assertIsNone(c1.select)

    def _test_nmnote(self):

        # __init__()
        # args: parent, name, thenote, copy
        thenote = 'note!'
        n0 = NMNote(PARENT, 'Note0', thenote=thenote)
        self.assertEqual(n0.thenote, thenote)

        # parameters()
        plist = PLIST + ['thenote']
        self.assertEqual(n0.parameter_list, plist)

        # isequivalent()
        # arg: NMNote
        n1 = NMNote(PARENT, 'Note0', thenote=thenote)
        self.assertTrue(n0._isequivalent(n1, alert=ALERT))
        n1 = NMNote(PARENT, 'Note0', thenote='different')
        self.assertFalse(n0._isequivalent(n1, alert=ALERT))

        # copy()
        c = n0.copy()
        self.assertIsInstance(c, NMNote)
        self.assertTrue(n0._isequivalent(c, alert=ALERT))
        self.assertEqual(n0.thenote, c.thenote)

        # thenote_set()
        # arg: thenote
        self.assertTrue(n0._thenote_set(123))
        self.assertEqual(n0.thenote, '123')
        self.assertTrue(n0._thenote_set(None))
        self.assertEqual(n0.thenote, 'None')
        self.assertTrue(n0._thenote_set('test'))
        self.assertEqual(n0.thenote, 'test')
        self.assertTrue(n0._thenote_set([1]))
        self.assertEqual(n0.thenote, '[1]')

    def _test_nmnote_container(self):

        # __init__()
        # args: parent, name, copy
        txt = 'test #'
        prefix = 'Note'
        nlist = [prefix + str(i) for i in range(0, 4)]
        thenotes = [txt + str(i) for i in range(0, 4)]
        notes = NMNoteContainer(PARENT, 'MyNotes')
        self.assertEqual(notes.parameters['type'], 'NMNote')
        self.assertEqual(notes.prefix, prefix)
        self.assertFalse(notes.parameters['rename'])
        self.assertFalse(notes._NMNoteContainer__off)

        # new()
        # args: thenote, select
        self.assertEqual(notes.name_next_seq(), 0)
        self.assertEqual(notes.name_next(), nlist[0])
        for i in range(0, 4):
            thenote = txt + str(i)
            n = notes.new(thenote=thenote)
            self.assertIsInstance(n, NMNote)
            self.assertEqual(n.name, nlist[i])
            self.assertEqual(n.thenote, thenote)
        # notes.print_content_parameters()
        notes.off = True
        self.assertIsNone(notes.new('test'))  # notes are off

        # copy()
        c = notes.copy()
        self.assertIsInstance(c, NMNoteContainer)
        self.assertTrue(notes._isequivalent(c, alert=ALERT))

        # all_()
        self.assertEqual(thenotes, notes.all_)
        self.assertEqual(thenotes, c.all_)
        # notes.print_all()

        # duplicate()
        with self.assertRaises(RuntimeError):  # notes cannot be duplicated
            notes.duplicate()

    def _test_dimension(self):

        # __init__()
        # args: parent, name, dim, notes, copy
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = NMDimension(PARENT, 'ydim0', dim=b)
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = NMDimension(PARENT, 'ydim0', notes=b)
        notes = NMNoteContainer(PARENT, 'Notes')
        y0 = NMDimension(PARENT, 'ydim0', dim=YDIM0, notes=notes)
        x0 = NMDimensionX(PARENT, 'xdim0', dim=XDIM0, notes=notes)
        y1 = NMDimension(PARENT, 'ydim1', dim=YDIM1, notes=notes)
        x1 = NMDimensionX(PARENT, 'xdim1', dim=XDIM1, notes=notes)
        self.assertEqual(y0._NMDimension__notes_container, notes)
        self.assertEqual(y0._offset, YDIM0['offset'])
        self.assertEqual(y0._label, YDIM0['label'])
        self.assertEqual(y0._units, YDIM0['units'])
        self.assertIsNone(y0._master)
        self.assertEqual(x0._NMDimension__notes_container, notes)
        self.assertEqual(x0._offset, XDIM0['offset'])
        self.assertEqual(x0._start, XDIM0['start'])
        self.assertEqual(x0._delta, XDIM0['delta'])
        self.assertEqual(x0._label, XDIM0['label'])
        self.assertEqual(x0._units, XDIM0['units'])
        self.assertIsNone(x0._master)
        self.assertIsNone(x0._xdata)

        # parameters()
        plist = PLIST + ['offset', 'label', 'units', 'master']
        self.assertEqual(y0.parameter_list, plist)
        xplist = plist + ['start', 'delta', 'xdata']
        self.assertEqual(x0.parameter_list, xplist)

        # note_new()
        # arg: thenote
        note = 'test123'
        n = y0._note_new(note)
        self.assertIsInstance(n, NMNote)
        self.assertEqual(n.thenote, note)
        n = x0._note_new(note)
        self.assertEqual(n.thenote, note)

        # master_set()
        # arg: dimension
        for b in BADTYPES + [x0]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y1._master_set(b)
        for b in BADTYPES + [y0]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x1._master_set(b)
        with self.assertRaises(ValueError):
            y1._master_set(y1)  # self cannot be master
        with self.assertRaises(ValueError):
            x1._master_set(x1)  # self cannot be master
        self.assertTrue(y1._master_set(None))  # ok
        self.assertTrue(x1._master_set(None))  # ok
        self.assertIsNone(y1._master)
        self.assertIsNone(x1._master)
        self.assertFalse(y1._master_lock)
        self.assertFalse(x1._master_lock)
        self.assertTrue(y1._master_set(y0))
        self.assertIsInstance(y1._master, NMDimension)
        self.assertEqual(y1._master, y0)
        self.assertTrue(y1._master_lock)
        self.assertTrue(x1._master_set(x0))
        self.assertIsInstance(x1._master, NMDimensionX)
        self.assertEqual(x1._master, x0)
        self.assertTrue(x1._master_lock)
        dim = y1.dim
        for k in YDIM0.keys():  # y0 is master
            self.assertEqual(YDIM0[k], dim[k])
        self.assertEqual(dim['master'], y0)
        dim = x1.dim
        for k in XDIM0.keys():  # x0 is master
            self.assertEqual(XDIM0[k], dim[k])
        self.assertEqual(dim['master'], x0)
        with self.assertRaises(RuntimeError):
            y1._dim_set(YDIM0)  # master on
        self.assertTrue(y1._offset_set(3.14))  # offset free from master
        with self.assertRaises(RuntimeError):
            y1._label_set('test')  # master on
        with self.assertRaises(RuntimeError):
            y1._units_set('test')  # master on
        with self.assertRaises(RuntimeError):
            x1._dim_set(XDIM0)  # master on
        self.assertTrue(x1._offset_set(3.14))  # offset free from master
        with self.assertRaises(RuntimeError):
            x1._label_set('test')  # master on
        with self.assertRaises(RuntimeError):
            x1._units_set('test')  # master on
        with self.assertRaises(RuntimeError):
            x1._start_set(0)  # master on
        with self.assertRaises(RuntimeError):
            x1._delta_set(1)  # master on
        xdata = Data(PARENT, 'xdata', xdim=XDIMx, ydim=YDIMx)
        with self.assertRaises(RuntimeError):
            x1._xdata_set(xdata)  # master on
        with self.assertRaises(RuntimeError):
            y0._master_set(y1)  # y1 has master
        with self.assertRaises(RuntimeError):
            x0._master_set(x1)  # x1 has master

        # dim_set()
        # arg: dim
        for b in BADTYPES + [y1]:
            if isinstance(b, dict):
                continue
            with self.assertRaises(TypeError):
                y0._dim_set(b)
            with self.assertRaises(TypeError):
                x0._dim_set(b)
        bad_dim = {'label': 'Vmem', 'units': 'mV', 'test': 0}
        with self.assertRaises(KeyError):
            y0._dim_set(bad_dim)
        with self.assertRaises(KeyError):
            y0._dim_set(XDIM0)
        with self.assertRaises(KeyError):
            x0._dim_set(bad_dim)
        self.assertTrue(y0._dim_set(YDIM0))
        self.assertTrue(x0._dim_set(YDIM0))  # ok
        self.assertTrue(x0._dim_set(XDIM0))

        # offset_set()
        # arg: offset
        for b in BADTYPES + [y1]:
            if isinstance(b, int) or isinstance(b, float):
                continue  # ok
            with self.assertRaises(TypeError):
                y0._offset_set(b)
            with self.assertRaises(TypeError):
                x0._offset_set(b)
        badvalues = [float('nan'), float('inf')]
        for b in badvalues:
            with self.assertRaises(ValueError):
                y0._offset_set(b)
        for b in badvalues:
            with self.assertRaises(ValueError):
                x0._offset_set(b)
        self.assertTrue(y0._offset_set(3.14))
        self.assertEqual(y0._offset, 3.14)
        self.assertTrue(x0._offset_set(3.14))
        self.assertEqual(x0._offset, 3.14)
        self.assertTrue(y1._master_lock)
        self.assertTrue(y1._offset_set(3.14))  # offset free from master
        self.assertEqual(y1._offset, 3.14)  # offset free from master
        self.assertTrue(x1._master_lock)
        self.assertTrue(x1._offset_set(3.14))  # offset free from master
        self.assertEqual(x1._offset, 3.14)  # offset free from master

        # label_set()
        # arg: label
        for b in BADTYPES + [y1]:
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                y0._label_set(b)
            with self.assertRaises(TypeError):
                x0._label_set(b)
        self.assertTrue(y0._label_set(None))
        self.assertEqual(y0._label, '')
        self.assertTrue(x0._label_set(None))
        self.assertEqual(x0._label, '')
        self.assertTrue(y0._label_set('test'))
        self.assertEqual(y0._label, 'test')
        self.assertTrue(x0._label_set('test'))
        self.assertEqual(x0._label, 'test')
        with self.assertRaises(RuntimeError):
            y1._label_set('test')  # master on
        with self.assertRaises(RuntimeError):
            x1._label_set('test')  # master on

        # units_set()
        # arg: units
        for b in BADTYPES + [y1]:
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                y0._units_set(b)
            with self.assertRaises(TypeError):
                x0._units_set(b)
        self.assertTrue(y0._units_set(None))
        self.assertEqual(y0._units, '')
        self.assertTrue(x0._units_set(None))
        self.assertEqual(x0._units, '')
        self.assertTrue(y0._units_set('test'))
        self.assertEqual(y0._units, 'test')
        self.assertTrue(x0._units_set('test'))
        self.assertEqual(x0._units, 'test')
        with self.assertRaises(RuntimeError):
            y1._units_set('test')  # master on
        with self.assertRaises(RuntimeError):
            x1._units_set('test')  # master on

        # start_set()
        # arg: start
        for b in BADTYPES + [x1]:
            if isinstance(b, int) or isinstance(b, float):
                continue  # continue
            with self.assertRaises(TypeError):
                x0._start_set(b)
        goodvalues = [0, 10, -10.2, float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._start_set(g))
            self.assertEqual(x0._start, g)
        self.assertTrue(x0._start_set(float('nan')))  # nan ok
        with self.assertRaises(RuntimeError):
            x1._start_set(0)  # master on

        # delta_set()
        # arg: delta
        for b in BADTYPES + [x1]:
            if isinstance(b, int) or isinstance(b, float):
                continue  # continue
            with self.assertRaises(TypeError):
                x0._delta_set(b)
        goodvalues = [0, 10, -10.2, float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._delta_set(g))
            self.assertEqual(x0._delta, g)
        self.assertTrue(x0._delta_set(float('nan')))  # nan ok
        with self.assertRaises(RuntimeError):
            x1._delta_set(0)  # master on

        # xdata_set()
        # arg: xdata
        for b in BADTYPES + [x1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x0._xdata_set(b)
        self.assertTrue(x0._xdata_set(xdata))
        self.assertEqual(x0._xdata, xdata)
        self.assertTrue(x0._offset_set(3.14))  # offset free from xdata
        with self.assertRaises(RuntimeError):
            x0._start_set(0)  # xdata on
        with self.assertRaises(RuntimeError):
            x0._delta_set(1)  # xdata on
        with self.assertRaises(RuntimeError):
            x0._label_set('test')  # xdata on
        with self.assertRaises(RuntimeError):
            x0._units_set('test')  # xdata on
        self.assertTrue(x0._xdata_set(None))
        self.assertEqual(x0._xdata, None)
        with self.assertRaises(RuntimeError):
            x1._xdata_set(xdata)  # master on

        # copy()
        c = y0.copy()
        self.assertIsInstance(c, NMDimension)
        self.assertTrue(y0._isequivalent(c, alert=ALERT))
        c = x0.copy()
        self.assertIsInstance(c, NMDimensionX)
        self.assertTrue(x0._isequivalent(c, alert=ALERT))

    def _test_data(self):
        # args: parent, name, np_array, xdim, ydim, dataseries, copy
        n0 = 'RecordA0'
        n1 = 'RecordA1'
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, n0, np_array=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, n0, xdim=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, n0, ydim=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, n0, dataseries=b)
        nparray0 = np.full([4], 3.14, dtype=np.float64, order='C')
        nparray1 = np.full([5], 6.28, dtype=np.float64, order='C')
        nparrayx = np.full([6], 12.56, dtype=np.float64, order='C')
        d0 = Data(PARENT, n0, np_array=nparray0, xdim=XDIM0, ydim=YDIM0)
        d1 = Data(PARENT, n1, np_array=nparray1, xdim=XDIM1, ydim=YDIM1)
        xdata = Data(
                    PARENT, 'xdata', np_array=nparrayx, xdim=XDIMx, ydim=YDIMx
                )
        self.assertTrue(np.array_equal(d0._Data__np_array, nparray0))
        self.assertTrue(np.array_equal(d1._Data__np_array, nparray1))
        # parameters
        plist = PLIST + ['xdim', 'ydim', 'dataseries']
        self.assertEqual(d0.parameter_list, plist)

        # content
        content_name = 'data'
        c = d0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), [content_name, 'notes'])
        self.assertEqual(c[content_name], d0.name)
        self.assertEqual(c['notes'], d0.note.names)
        # isequivalent, args: Data
        self.assertFalse(d0._isequivalent(d1, alert=ALERT))
        d00 = Data(PARENT, n0, np_array=nparray0, xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(d0._isequivalent(d00, alert=ALERT))
        nparray00 = np.full([4], 3.14, dtype=np.float64, order='F')
        d00 = Data(PARENT, n0, np_array=nparray00, xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(d0._isequivalent(d00, alert=ALERT))
        nparray0[2] = 5
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray0[2] = 0
        d00 = Data(PARENT, n0, np_array=None, xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = np.full([5, 2], 3.14, dtype=np.float64, order='C')
        d00 = Data(PARENT, n0, np_array=nparray00, xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = np.full([5], 3.14, dtype=np.int32, order='C')
        d00 = Data(PARENT, n0, np_array=nparray00, xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = np.full([5], 3.14, dtype=np.float64, order='F')
        d00 = Data(PARENT, n0, np_array=nparray00, xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        d00 = Data(PARENT, n0, np_array=nparray0, xdim=XDIM1, ydim=YDIM0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        d00 = Data(PARENT, n0, np_array=nparray0, xdim=XDIM0, ydim=YDIM1)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        # copy
        c = d0.copy()
        self.assertIsInstance(c, Data)
        self.assertTrue(d0._isequivalent(c, alert=ALERT))
        # np_array_set, args: np_array
        for b in BADTYPES:
            if b is None:
                continue
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        self.assertTrue(d0._np_array_set(None))
        self.assertIsNone(d0.np_array)
        self.assertTrue(d0._np_array_set(nparray1))
        self.assertIsInstance(d0.np_array, np.ndarray)
        # np_array_make, args: shape, fill_value, dtype, order
        # wrapper fxn, so basic testing here
        self.assertTrue(d0.np_array_make((10, 2)))
        # np_array_make_random_normal, args: shape, mean, stdv
        # wrapper fxn, so basic testing here
        self.assertTrue(d0.np_array_make_random_normal(10, mean=3, stdv=1))
        # TODO
        # dataseries_str
        # dataseries_add
        # dataseries_remove
        # ds = DataSeries(PARENT, 'Record')

    def _test_data_container(self):
        c0 = DataContainer(PARENT, 'Data')
        c1 = DataContainer(PARENT, 'Data')
        self.assertEqual(c0.parameters['type'], 'Data')
        self.assertEqual(c0.prefix, nmp.DATA_PREFIX)
        self.assertTrue(c0.parameters['rename'])
        # new, args: name, np_array, xdim=, ydim, dataseries, select
        # wrapper for Data.new() and NMObjectContainer.new()
        nlist = ['RecordA0', 'WaveA0', 'Xdata']
        for n in nlist:
            self.assertIsInstance(c0.new(name=n, xdim=XDIM0, ydim=YDIM0), Data)
            self.assertIsInstance(c1.new(name=n, xdim=XDIM0, ydim=YDIM0), Data)
        # add, args: data, select
        # wrapper for NMObjectContainer.new()
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0.add(b)
        nparray0 = np.full([4], 3.14, dtype=np.float64, order='C')
        d0 = Data(PARENT, 'RecordA1', np_array=nparray0, xdim=XDIM0,
                  ydim=YDIM0)
        self.assertTrue(c0.add(d0))
        # copy
        c = c0.copy()
        self.assertIsInstance(c, DataContainer)
        self.assertTrue(c._isequivalent(c0, alert=ALERT))
        # isequivalent, args: DataContainer
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        d1 = Data(PARENT, 'RecordA1', np_array=nparray0, xdim=XDIM0,
                  ydim=YDIM0)
        c1.add(d1)
        self.assertTrue(c0._isequivalent(c1, alert=ALERT))
        self.assertIsInstance(c0._select_set('WaveA0'), Data)
        self.assertIsInstance(c1._select_set('RecordA0'), Data)
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        # remove(), args: names, indexes, confirm
        # wrapper for NMObjectContainer.remove()
        # TODO, test if Data is removed from dataseries and sets

    def _test_channel(self):
        # args: parent, name, xdim, ydim, copy
        # ydim tested by NMDimension
        # xdim tested by NMDimensionX
        c0 = Channel(PARENT, 'A', xdim=XDIM0, ydim=YDIM0)
        for k in XDIM0.keys():
            self.assertEqual(c0.x.dim[k], XDIM0[k])
        for k in YDIM0.keys():
            self.assertEqual(c0.y.dim[k], YDIM0[k])
        # parameters
        plist = PLIST + ['xdim', 'ydim']
        self.assertEqual(c0.parameter_list, plist)

        # copy
        c = c0.copy()
        self.assertTrue(c0._isequivalent(c, alert=ALERT))
        self.assertTrue(c0.x._isequivalent(c.x, alert=ALERT))
        self.assertTrue(c0.y._isequivalent(c.y, alert=ALERT))
        # isequivalent, args: Channel
        c1 = Channel(PARENT, 'A', xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(c0._isequivalent(c1, alert=ALERT))
        self.assertTrue(c0.x._isequivalent(c1.x, alert=ALERT))
        self.assertTrue(c0.y._isequivalent(c1.y, alert=ALERT))
        c1.name = 'B'
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))

    def _test_channel_container(self):
        # args: parent, name, copy
        c0 = ChannelContainer(PARENT, 'channels')
        self.assertEqual(c0.parameters['type'], 'Channel')
        self.assertEqual(c0.prefix, '')
        self.assertFalse(c0.parameters['rename'])
        # name
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), 'A')
        c0._NMObjectContainer__rename = True
        self.assertTrue(c0._prefix_set('Test'))
        self.assertEqual(c0.prefix, 'Test')
        self.assertEqual(c0.name_next(), 'A')  # prefix not used
        self.assertTrue(c0._prefix_set(''))  # reset
        c0._NMObjectContainer__rename = False
        # new
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), 'A')
        c = c0.new(xdim=XDIM0, ydim=YDIM0)
        self.assertIsInstance(c, Channel)
        self.assertEqual(c.name, 'A')
        self.assertEqual(c0.name_next_seq(), 1)
        self.assertEqual(c0.name_next(), 'B')
        c = c0.new(xdim=XDIM0, ydim=YDIM1)
        self.assertEqual(c.name, 'B')
        self.assertEqual(c0.name_next_seq(), 2)
        self.assertEqual(c0.name_next(), 'C')
        # copy
        c = c0.copy()
        self.assertTrue(c0._isequivalent(c, alert=ALERT))
        # isequivalent, args: ChannelContainer
        c00 = ChannelContainer(PARENT, 'channels')
        c00.new(xdim=XDIM0, ydim=YDIM0)
        c00.new(xdim=XDIM0, ydim=YDIM1)
        self.assertTrue(c0._isequivalent(c00, alert=ALERT))
        c00 = ChannelContainer(PARENT, 'channels')
        c00.new(xdim=XDIM0, ydim=YDIM0)
        c00.new(xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(c0._isequivalent(c00, alert=ALERT))
        c00 = ChannelContainer(PARENT, 'chans')
        c00.new(xdim=XDIM0, ydim=YDIM0)
        c00.new(xdim=XDIM0, ydim=YDIM1)
        self.assertFalse(c0._isequivalent(c00, alert=ALERT))
        # duplicate
        with self.assertRaises(RuntimeError):
            c0.duplicate()

    def _test_dataseries_set(self):
        # args: parent, name, copy
        quiet = True
        s_all = DataSeriesSet(PARENT, 'All')
        s1 = DataSeriesSet(PARENT, 'Set1')
        s2 = DataSeriesSet(PARENT, 'Set2')
        s3 = DataSeriesSet(PARENT, 'Set3')
        num_epochs = 10
        data_a = []
        data_a_c = []
        data_b = []
        data_c = []
        epoch = []
        for ep in range(0, num_epochs):
            da = Data(PARENT, 'RecordA' + str(ep))
            db = Data(PARENT, 'RecordB' + str(ep))
            dc = Data(PARENT, 'RecordC' + str(ep))
            da.np_array_make_random_normal(10, mean=0, stdv=1)
            da_c = da.copy()
            db.np_array_make_random_normal(10, mean=0, stdv=1)
            dc.np_array_make_random_normal(10, mean=0, stdv=1)
            data_a.append(da)
            data_a_c.append(da_c)
            data_b.append(db)
            data_c.append(dc)
            epoch.append({'A': da, 'B': db, 'C': dc})
        self.assertEqual(s1.channel_count, 0)
        self.assertEqual(s1.epoch_count, 0)
        self.assertEqual(s1.data_names, {})
        self.assertEqual(s1.equation, [])
        self.assertEqual(s1.equation_lock, [])
        self.assertFalse(s1.locked)
        self.assertTrue(s1.isempty)
        self.assertTrue(s2.isempty)
        self.assertEqual(s_all.data_names, {'ALL'})
        # parameters
        plist = PLIST + ['sort_template', 'eq_lock']
        self.assertEqual(s1.parameter_list, plist)

        # bad_names
        self.assertEqual(s1._bad_names, ['select', 'default'])
        # data_dict_check, args: data_dict, chan_default
        for b in BADTYPES:
            if isinstance(b, list) or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                s1._data_dict_check(b)
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1._data_dict_check([b])
        for b in [None, True, 1, 3.14, 'test', nm, '', '0', '&']:
            with self.assertRaises(ValueError):
                s1._data_dict_check({b: data_a})
        for b in BADTYPES:
            if isinstance(b, list):
                continue  # ok
            with self.assertRaises(TypeError):
                s1._data_dict_check({'A': b})
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                s1._data_dict_check(data_a, chan_default=b)
        for b in ['test', '', '0', '&']:
            with self.assertRaises(ValueError):
                s1._data_dict_check(data_a, chan_default=b)
        self.assertEqual(s1._data_dict_check({}), {})
        self.assertEqual(s1._data_dict_check([]), {})
        dA0 = data_a[0]
        dB0 = data_b[0]
        dd = s1._data_dict_check(dA0)
        self.assertEqual(dd, {'A': [dA0]})
        dd = s1._data_dict_check(dA0, chan_default='c')
        self.assertEqual(dd, {'C': [dA0]})
        dd = s1._data_dict_check(dA0, chan_default='ALL_EXISTING')
        self.assertEqual(dd, {})  # nothing because theset is empty
        dd = s1._data_dict_check([dA0])
        self.assertEqual(dd, {'A': [dA0]})
        dd = s1._data_dict_check([dA0], chan_default='c')
        self.assertEqual(dd, {'C': [dA0]})
        dd = s1._data_dict_check([dA0], chan_default='ALL_EXISTING')
        self.assertEqual(dd, {})  # nothing because theset is empty
        dd = s1._data_dict_check({'A': dA0}, chan_default='c')
        self.assertEqual(dd, {'A': [dA0]})
        dd = s1._data_dict_check({'A': dA0, 'B': dB0}, chan_default='c')
        self.assertEqual(dd, {'A': [dA0], 'B': [dB0]})
        dd = s1._data_dict_check({'A': dA0, 'B': [dB0]}, chan_default='c')
        self.assertEqual(dd, {'A': [dA0], 'B': [dB0]})
        # theset_clear_if_empty
        self.assertTrue(s1.clear(confirm=False, quiet=quiet))
        self.assertTrue(s1.add({'A': [dA0], 'B': [dB0]}, quiet=quiet))
        self.assertFalse(s1._theset_clear_if_empty())
        self.assertEqual(s1._DataSeriesSet__theset, {'A': [dA0], 'B': [dB0]})
        s1._DataSeriesSet__theset = {'A': [], 'B': []}
        self.assertTrue(s1._theset_clear_if_empty())
        self.assertEqual(s1._DataSeriesSet__theset, {})
        # add, args: data_dict
        # discard, args: data_dict
        # see data_dict_check()
        s1.clear(confirm=False, quiet=quiet)
        self.assertTrue(s1.add(data_a, quiet=quiet))  # added to chan 'A'
        self.assertFalse(s1.add(data_a, quiet=quiet))  # already exists
        for ep in range(0, num_epochs):
            self.assertTrue(data_a[ep] in s1._DataSeriesSet__theset['A'])
        nlist_a = [data_a[ep].name for ep in range(0, num_epochs)]
        self.assertEqual(s1.data_names, {'A': nlist_a})
        self.assertTrue(s1.discard(data_a[1], quiet=quiet))
        nlist_a.remove(data_a[1].name)
        self.assertEqual(s1.data_names, {'A': nlist_a})
        self.assertFalse(s1.discard(data_b[1], quiet=quiet))
        self.assertTrue(s1.discard(data_a, quiet=quiet))
        self.assertEqual(s1.data_names, {})
        # test epochs
        s1.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(0, num_epochs):
            self.assertTrue(data_a[ep] in s1._DataSeriesSet__theset['A'])
            self.assertTrue(data_b[ep] in s1._DataSeriesSet__theset['B'])
            self.assertTrue(data_c[ep] in s1._DataSeriesSet__theset['C'])
        nlist_a = [data_a[ep].name for ep in range(0, num_epochs)]
        nlist_b = [data_b[ep].name for ep in range(0, num_epochs)]
        nlist_c = [data_c[ep].name for ep in range(0, num_epochs)]
        self.assertEqual(s1.data_names, {'A': nlist_a, 'B': nlist_b,
                                         'C': nlist_c})
        self.assertEqual(s1.channel_count, 3)
        self.assertEqual(s1.epoch_count, num_epochs)
        self.assertTrue(s1.discard(epoch[1], quiet=quiet))
        self.assertFalse(s1.discard(epoch[1], quiet=quiet))
        nlist_a.remove(data_a[1].name)
        nlist_b.remove(data_b[1].name)
        nlist_c.remove(data_c[1].name)
        self.assertEqual(s1.data_names, {'A': nlist_a, 'B': nlist_b,
                                         'C': nlist_c})
        self.assertEqual(s1.epoch_count, num_epochs-1)
        # data_dict_check (ALL_EXISTING)
        dd = s1._data_dict_check(dA0, chan_default='ALL_EXISTING')
        self.assertEqual(dd, {'A': [dA0], 'B': [dA0], 'C': [dA0]})
        dd = s1._data_dict_check(data_a, chan_default='ALL_EXISTING')
        self.assertEqual(dd, {'A': data_a, 'B': data_a, 'C': data_a})
        # chan_list_check, args: chan_list
        for b in BADTYPES:
            if isinstance(b, list) or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                s1._chan_list_check(chan_list=b)
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(ValueError):
                s1._chan_list_check(chan_list=[b])
        # reverse, args: chan_list
        # see chan_list_check()
        self.assertTrue(s1.reverse(chan_list=['A', 'C'], quiet=quiet))
        nlist_a.reverse()
        nlist_c.reverse()
        self.assertEqual(s1.data_names,
                         {'A': nlist_a, 'B': nlist_b, 'C': nlist_c})
        self.assertTrue(s1.reverse(quiet=quiet))
        nlist_a.reverse()
        nlist_b.reverse()
        nlist_c.reverse()
        self.assertEqual(s1.data_names,
                         {'A': nlist_a, 'B': nlist_b, 'C': nlist_c})
        self.assertTrue(s1.reverse(chan_list=['B'], quiet=quiet))
        nlist_b.reverse()
        self.assertEqual(s1.data_names,
                         {'A': nlist_a, 'B': nlist_b, 'C': nlist_c})
        # clear, args: chan_list, confirm
        # see chan_list_check()
        self.assertTrue(s1.clear(chan_list=['A', 'C'], confirm=CONFIRM,
                                 quiet=quiet))
        self.assertEqual(s1.data_names, {'A': [], 'B': nlist_b, 'C': []})
        self.assertTrue(s1.clear(confirm=CONFIRM, quiet=quiet))
        self.assertEqual(s1.data_names, {})
        # contains, args: data_dict
        # see data_dict_check()
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        self.assertTrue(s1.add(data_a, quiet=quiet))
        self.assertTrue(s1.contains(data_a, alert=ALERT))
        self.assertFalse(s1.contains(data_b, alert=ALERT))
        self.assertFalse(s1.contains({'B': data_a}, alert=ALERT))
        self.assertTrue(s1.discard(data_a[1], quiet=quiet))
        self.assertFalse(s1.contains(data_a[1], alert=ALERT))
        for ep in range(0, num_epochs):
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s2.contains({'A': data_a}, alert=ALERT))
        self.assertTrue(s2.contains({'B': data_b}, alert=ALERT))
        self.assertTrue(s2.contains({'C': data_c}, alert=ALERT))
        self.assertFalse(s2.contains({'D': data_c}, alert=ALERT))
        self.assertTrue(s2.discard(epoch[1], quiet=quiet))
        self.assertFalse(s2.contains(epoch[1], alert=ALERT))
        # get_channel, args: chan_char
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(ValueError):
                s1.get_channel(b)
        s1.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        dlist = s1.get_channel('A')
        self.assertEqual(dlist, data_a)
        dlist = s1.get_channel('B')
        self.assertEqual(dlist, data_b)
        dlist = s1.get_channel('C')
        self.assertEqual(dlist, data_c)
        dlist = s1.get_channel('D')
        self.assertEqual(dlist, [])
        self.assertTrue(s1.discard(epoch[0], quiet=quiet))
        self.assertEqual(s1.epoch_count, num_epochs-1)
        dlist = s1.get_channel('A')
        self.assertNotEqual(dlist, data_a)
        dlist.append(data_a[0])  # should not change theset
        dlist = s1.get_channel('A')
        self.assertFalse(data_a[0].name in s1.data_names['A'])
        # get, args: chan_list
        # see chan_list_check()
        s1.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        s = s1.get('A')
        self.assertEqual(s, {'A': data_a})
        s = s1.get('B')
        self.assertEqual(s, {'B': data_b})
        s = s1.get('C')
        self.assertEqual(s, {'C': data_c})
        s = s1.get('D')
        self.assertEqual(s, {})
        self.assertTrue(s1.discard(epoch[0], quiet=quiet))
        s = s1.get('A')
        dlist = s['A']
        self.assertEqual(len(dlist), num_epochs-1)
        dlist.append(data_a[0])  # should not change theset
        s = s1.get('A')
        dlist = s['A']
        self.assertEqual(len(dlist), num_epochs-1)
        self.assertFalse(data_a[0].name in s1.data_names['A'])
        s.update({'A': data_a[0]})  # should not change theset
        s = s1.get('A')
        dlist = s['A']
        self.assertEqual(len(dlist), num_epochs-1)
        # difference, args: DataSeriesSet, alert
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.difference(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):  # 0-9
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(0, num_epochs-3):  # 0-6
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.difference(s2, quiet=quiet))  # 7-9
        for ep in range(0, num_epochs):
            if ep < num_epochs-3:
                self.assertFalse(s1.contains(epoch[ep], alert=ALERT))
            else:
                self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):  # 0-9
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(0, num_epochs-3):  # 0-6
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s2.difference(s1, quiet=quiet))  # empty
        self.assertEqual(s2.data_names, {})
        # intersection, args: DataSeriesSet, alert
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.intersection(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs-2):  # 0-7
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(num_epochs-5, num_epochs):  # 5-9
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.intersection(s2, quiet=quiet))  # 5-7
        for ep in range(0, num_epochs):
            if ep < num_epochs-5:
                self.assertFalse(s1.contains(epoch[ep], alert=ALERT))
            elif ep < num_epochs-2:
                self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
            else:
                self.assertFalse(s1.contains(epoch[ep], alert=ALERT))
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs-5):  # 0-4
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(num_epochs-5, num_epochs):  # 5-9
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.intersection(s2, quiet=quiet))
        self.assertTrue(s1.isempty)
        self.assertFalse(s2.isempty)
        # symmetric_difference, args: DataSeriesSet, alert
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.symmetric_difference(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs-3):  # 0-6
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(num_epochs-6, num_epochs):  # 4-9
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.symmetric_difference(s2, quiet=quiet))  # 0-3, 7-9
        for ep in range(0, num_epochs):
            if ep < num_epochs-6:
                self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
            elif ep < num_epochs-3:
                self.assertFalse(s1.contains(epoch[ep], alert=ALERT))
            else:
                self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs-3):  # 0-6
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.symmetric_difference(s2, quiet=quiet))
        self.assertTrue(s1.isempty)
        # union, args: DataSeriesSet, alert
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.union(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(1, num_epochs, 2):  # 1, 3, 5...
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.union(s2, quiet=quiet))
        for ep in range(0, num_epochs):
            self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs-3):  # 0-6
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(num_epochs-6, num_epochs):  # 4-9
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.union(s2, quiet=quiet))
        for ep in range(0, num_epochs):
            self.assertTrue(s1.contains(epoch[ep], alert=ALERT))
        # isdisjoint, args: DataSeriesSet
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.isdisjoint(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(1, num_epochs, 2):  # 1, 3, 5...
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.isdisjoint(s2))
        self.assertTrue(s2.isdisjoint(s1))
        self.assertTrue(s1.add(epoch[3], quiet=quiet))
        self.assertFalse(s1.isdisjoint(s2))
        self.assertFalse(s2.isdisjoint(s1))
        # issubset, args: DataSeriesSet
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.issubset(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):  # 0-9
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(1, num_epochs, 2):  # 1, 3, 5...
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s2.issubset(s1))
        self.assertFalse(s1.issubset(s2))
        # issuperset, args: DataSeriesSet
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.issuperset(b)
        self.assertTrue(s1.issuperset(s2))
        self.assertFalse(s2.issuperset(s1))
        # isequal, args: DataSeriesSet
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                s1.isequal(b)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        self.assertTrue(s1.isequal(s2, alert=ALERT))
        s2.discard({'A': data_a[0]}, quiet=quiet)
        self.assertFalse(s1.isequal(s2, alert=ALERT))
        s2.add({'A': data_a[0]}, quiet=quiet)
        self.assertTrue(s1.isequal(s2, alert=ALERT))  # order does not matter
        s2.add({'B': data_a[0]}, quiet=quiet)
        self.assertFalse(s1.isequal(s2, alert=ALERT))
        s2.discard({'B': data_a[0]}, quiet=quiet)
        self.assertTrue(s1.isequal(s2, alert=ALERT))
        s2.add({'D': data_a[0]}, quiet=quiet)
        self.assertFalse(s1.isequal(s2, alert=ALERT))
        s1.clear(confirm=False, quiet=quiet)
        self.assertFalse(s1.isequal(s2, alert=ALERT))
        s2.clear(confirm=False, quiet=quiet)
        self.assertTrue(s1.isequal(s2, alert=ALERT))
        # isequivalent, args: DataSeriesSet
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s1.add(data_a[ep], quiet=quiet))
            self.assertTrue(s2.add(data_a[ep], quiet=quiet))  # not copies
        self.assertFalse(s1._isequivalent(s2, alert=ALERT))  # different names
        s2.name = 'Set1'
        self.assertFalse(s1._isequivalent(s2, alert=ALERT))
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s2.add(data_a_c[ep], quiet=quiet))  # copies
        self.assertTrue(s1._isequivalent(s2, alert=ALERT))
        self.assertFalse(s1.isequal(s2, alert=ALERT))  # different refs
        s2.discard({'A': data_a_c[0]}, quiet=quiet)
        self.assertFalse(s1._isequivalent(s2, alert=ALERT))
        s2.add({'A': data_a_c[0]}, quiet=quiet)
        self.assertFalse(s1._isequivalent(s2, alert=ALERT))  # different order
        s2.name = 'Set2'
        # eq_list_check, args: eq_list, lock
        for b in BADTYPES:
            if b is None or isinstance(b, list):
                continue
            with self.assertRaises(TypeError):
                s1._eq_list_check(b, False)
        self.assertEqual(s1._eq_list_check(None, False), [])
        self.assertEqual(s1._eq_list_check([], False), [])
        self.assertEqual(s1._eq_list_check(s1, False), [s1])
        with self.assertRaises(TypeError):
            s1._eq_list_check([nm], False)
        with self.assertRaises(ValueError):
            s1._eq_list_check([s1, '|'], False)  # missing another set
        with self.assertRaises(TypeError):
            s1._eq_list_check(['|', s1, s2], False)
        with self.assertRaises(TypeError):
            s1._eq_list_check([s1, 0, s2], False)
        with self.assertRaises(ValueError):
            s1._eq_list_check([s1, 'a', s2], False)
        with self.assertRaises(ValueError):
            s1._eq_list_check([s1, '|', s2], True)
        eq_list = [s1, '|', s2]
        self.assertEqual(s1._eq_list_check(eq_list, False), eq_list)
        # equation, args: eq_list, lock
        # see eq_list_check()
        s_all.clear(confirm=False, quiet=quiet)
        s1.clear(confirm=False, quiet=quiet)
        s2.clear(confirm=False, quiet=quiet)
        for ep in range(0, num_epochs):  # 0, 1, 2...
            self.assertTrue(s_all.add(epoch[ep], quiet=quiet))
        for ep in range(0, num_epochs, 2):  # 0, 2, 4...
            self.assertTrue(s1.add(epoch[ep], quiet=quiet))
        for ep in range(1, num_epochs, 2):  # 1, 3, 5...
            self.assertTrue(s2.add(epoch[ep], quiet=quiet))
        quiet = False
        self.assertTrue(s3._equation(eq_list, lock=True, quiet=quiet))
        for ep in range(0, num_epochs):
            self.assertTrue(s3.contains(epoch[ep], alert=ALERT))
        self.assertEqual(s3.equation_lock, eq_list)
        self.assertEqual(s3.equation_lock_str, s1.name + ' | ' + s2.name)
        self.assertTrue(s3.locked)
        with self.assertRaises(RuntimeError):
            s3._equation(eq_list, lock=False)  # locked
        self.assertTrue(s3._equation(None, quiet=quiet))  # unlock
        self.assertEqual(s3.equation_lock, [])
        self.assertTrue(s3._equation(s1, lock=False, quiet=quiet))
        self.assertNotEqual(s1, s3)
        self.assertEqual(s1.data_names, s3.data_names)
        self.assertTrue(s3._equation(s1, lock=True, quiet=quiet))
        self.assertTrue(s3._equation(eq_list, lock=True, quiet=quiet))
        with self.assertRaises(RuntimeError):
            s3.add(epoch[0])
        with self.assertRaises(RuntimeError):
            s3.discard(epoch[0])
        with self.assertRaises(RuntimeError):
            s3.clear()
        with self.assertRaises(RuntimeError):
            s3.difference(s1)
        with self.assertRaises(RuntimeError):
            s3.intersection(s1)
        with self.assertRaises(RuntimeError):
            s3.symmetric_difference(s1)
        with self.assertRaises(RuntimeError):
            s3.union(s1)
        s3.locked = False
        with self.assertRaises(ValueError):
            s3.locked = True  # must use equation()
        self.assertTrue(s3._equation(eq_list, lock=True, quiet=quiet))
        for ep in range(0, num_epochs):
            self.assertTrue(s3.contains(epoch[ep], alert=ALERT))
        self.assertEqual(s3.data_names['A'], s1.data_names['A'] +
                         s2.data_names['A'])
        self.assertEqual(s3.data_names['B'], s1.data_names['B'] +
                         s2.data_names['B'])
        self.assertEqual(s3.data_names['C'], s1.data_names['C'] +
                         s2.data_names['C'])
        self.assertTrue(s1.discard(epoch[2]))
        self.assertFalse(s1.contains(epoch[2]))
        self.assertTrue(s2.discard(epoch[5]))
        self.assertFalse(s2.contains(epoch[5]))
        self.assertTrue(s3._eq_lock_update())
        for ep in range(0, num_epochs):
            if ep == 2 or ep == 5:
                self.assertFalse(s3.contains(epoch[ep], alert=ALERT))
            else:
                self.assertTrue(s3.contains(epoch[ep], alert=ALERT))
        self.assertTrue(s2.add(epoch[5]))
        self.assertTrue(s3._eq_lock_update())
        for ep in range(0, num_epochs):
            if ep == 2:
                self.assertFalse(s3.contains(epoch[ep], alert=ALERT))
            else:
                self.assertTrue(s3.contains(epoch[ep], alert=ALERT))
        self.assertEqual(s3.data_names['A'], s1.data_names['A'] +
                         s2.data_names['A'])
        self.assertEqual(s3.data_names['B'], s1.data_names['B'] +
                         s2.data_names['B'])
        self.assertEqual(s3.data_names['C'], s1.data_names['C'] +
                         s2.data_names['C'])
        # sort_template_set, args: DataSeriesSet
        for b in BADTYPES + [nm]:
            with self.assertRaises(TypeError):
                s3._sort_template_set(b)
        self.assertTrue(s3._sort_template_set(s_all))
        # sort_update
        self.assertTrue(s3._eq_lock_update())  # calls sort_update() sort()
        nlist_a = [data_a[ep].name for ep in range(0, num_epochs)]
        nlist_b = [data_b[ep].name for ep in range(0, num_epochs)]
        nlist_c = [data_c[ep].name for ep in range(0, num_epochs)]
        nlist_a.remove(data_a[2].name)
        nlist_b.remove(data_b[2].name)
        nlist_c.remove(data_c[2].name)
        self.assertEqual(s3.data_names['A'], nlist_a)
        self.assertEqual(s3.data_names['B'], nlist_b)
        self.assertEqual(s3.data_names['C'], nlist_c)
        # sort, args: DataSeriesSet
        for b in BADTYPES + [nm]:
            with self.assertRaises(TypeError):
                s3.sort(b)
        s_all.reverse(chan_list=['B'], quiet=quiet)
        s3.sort(s_all)
        nlist_b.reverse()
        self.assertEqual(s3.data_names['A'], nlist_a)
        self.assertEqual(s3.data_names['B'], nlist_b)
        self.assertEqual(s3.data_names['C'], nlist_c)
        # copy
        """
        c = s3.copy()
        self.assertIsInstance(c, DataSeriesSet)
        self.assertFalse(s3._isequivalent(c, alert=ALERT))
        self.assertTrue(s3.isequal(c))
        """

    def _test_dataseries_set_container(self):
        # args: parent, name, copy
        pass

    def _test_project(self):
        """
        name0 = 'Project0'
        name1 = 'Project1'
        p0 = Project(PARENT, name0)
        p1 = Project(PARENT, name1)
        f = p0.folder.new()
        ds = f.dataseries.new('Record')
        ds.make(channels=1, epochs=3, shape=5, dim=dim)
        f = p0.folder.new()
        ds = f.dataseries.new('Wave')
        ds.make(channels=2, epochs=3, shape=5, dim=dim)
        # p1._copy(p0)
        # self.assertTrue(p1._isequivalent(p0, alert=ALERT))
        """

    def _test_utilities(self):
        # name_ok
        self.assertFalse(nmu.name_ok(None))
        self.assertFalse(nmu.name_ok(0))
        self.assertFalse(nmu.name_ok(''))
        self.assertTrue(nmu.name_ok('test'))
        self.assertFalse(nmu.name_ok('_'))
        self.assertFalse(nmu.name_ok('*'))
        self.assertFalse(nmu.name_ok('0'))
        self.assertTrue(nmu.name_ok('test1234567890'))
        # NM preferences: NAME_SYMBOLS_OK = ['_']
        self.assertTrue(nmu.name_ok('test_'))
        self.assertTrue(nmu.name_ok('test_OK'))
        self.assertTrue(nmu.name_ok('test_OK_OK'))
        self.assertFalse(nmu.name_ok('test*'))
        self.assertFalse(nmu.name_ok('@test'))
        self.assertFalse(nmu.name_ok('te.st'))
        self.assertTrue(nmu.name_ok('te.st', ok_list=['.']))
        self.assertFalse(nmu.name_ok([]))
        self.assertTrue(nmu.name_ok(['test0', 'test1', 'test2']))
        self.assertFalse(nmu.name_ok(['test0', 'test1', 'test2?']))
        self.assertFalse(nmu.name_ok(['test0', 1, 'test2']))
        self.assertFalse(nmu.name_ok(['test0', '', 'test2']))
        # number_ok
        self.assertFalse(nmu.number_ok(None))
        self.assertFalse(nmu.number_ok("one"))
        self.assertFalse(nmu.number_ok(False))
        self.assertTrue(nmu.number_ok(-5))
        self.assertTrue(nmu.number_ok(1.34))
        self.assertFalse(nmu.number_ok(1.34, must_be_integer=True))
        self.assertFalse(nmu.number_ok(float('inf')))
        self.assertTrue(nmu.number_ok(float('inf'), inf_is_ok=True))
        self.assertFalse(nmu.number_ok(float('-inf')))
        self.assertTrue(nmu.number_ok(float('-inf'), inf_is_ok=True))
        self.assertFalse(nmu.number_ok(float('nan')))
        self.assertTrue(nmu.number_ok(float('nan'), nan_is_ok=True))
        self.assertTrue(nmu.number_ok(0, neg_is_ok=False))
        self.assertTrue(nmu.number_ok(1.34, neg_is_ok=False))
        self.assertFalse(nmu.number_ok(-1.34, neg_is_ok=False))
        self.assertTrue(nmu.number_ok(0, pos_is_ok=False))
        self.assertFalse(nmu.number_ok(1.34, pos_is_ok=False))
        self.assertTrue(nmu.number_ok(-1.34, pos_is_ok=False))
        self.assertTrue(nmu.number_ok(0))
        self.assertFalse(nmu.number_ok(0, zero_is_ok=False))
        self.assertTrue(nmu.number_ok(1.34, zero_is_ok=False))
        self.assertTrue(nmu.number_ok(-1.34, zero_is_ok=False))
        self.assertFalse(nmu.number_ok(complex(1, -1)))
        self.assertTrue(nmu.number_ok(complex(1, -1), complex_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, 'one']))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, None]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, False]))
        self.assertTrue(nmu.number_ok([0, -5, 1.34]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34], must_be_integer=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float('inf')]))
        self.assertTrue(nmu.number_ok([0, -5, float('inf')], inf_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float('-inf')]))
        self.assertTrue(nmu.number_ok([0, -5, float('-inf')], inf_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float('nan')]))
        self.assertTrue(nmu.number_ok([0, -5, float('nan')], nan_is_ok=True))
        self.assertTrue(nmu.number_ok([0, 3, 4], neg_is_ok=False))
        self.assertFalse(nmu.number_ok([-1, 3, 4], neg_is_ok=False))
        self.assertFalse(nmu.number_ok([0, 3, 4], pos_is_ok=False))
        self.assertTrue(nmu.number_ok([0, -3, -4], pos_is_ok=False))
        self.assertFalse(nmu.number_ok([0, 3, 4], zero_is_ok=False))
        self.assertTrue(nmu.number_ok([-4, 4], zero_is_ok=False))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, complex(1, -1)]))
        self.assertTrue(nmu.number_ok([0, -5, 1.34, complex(1, -1)],
                        complex_is_ok=True))
        # quotes
        self.assertEqual(nmu.quotes('test'), "'test'")
        self.assertEqual(nmu.quotes('test', single=False), '"test"')
        self.assertEqual(nmu.quotes(None), "'None'")
        self.assertEqual(nmu.quotes(False), "'False'")
        self.assertEqual(nmu.quotes(0), "'0'")
        self.assertEqual(nmu.quotes(nm), "'" + str(nm) + "'")
        # remove_special_char
        self.assertEqual(nmu.remove_special_char('test*'), 'test')
        self.assertEqual(nmu.remove_special_char('test'), 'test')
        self.assertEqual(nmu.remove_special_char('t@st*'), 'tst')
        self.assertEqual(nmu.remove_special_char(''), '')
        self.assertEqual(
            nmu.remove_special_char(['test*', 't@st*']), ['test', 'tst']
        )
        self.assertEqual(
            nmu.remove_special_char(['test', None, False]), ['test', '', '']
        )
        self.assertEqual(nmu.remove_special_char(None), '')
        self.assertEqual(nmu.remove_special_char(False), '')
        self.assertEqual(nmu.remove_special_char('test_*'), 'test')
        self.assertEqual(
            nmu.remove_special_char('test_*', ok_char=['_']), 'test_'
        )
        self.assertEqual(
            nmu.remove_special_char('test_*', ok_char=['_', '*']), 'test_*'
        )
        self.assertEqual(
            nmu.remove_special_char(
                'test_*',
                ok_char=['_', '*'],
                bad_char=['_', '*']
            ),
            'test'
        )
        self.assertEqual(
            nmu.remove_special_char('test0_*', bad_char=['t', '0']), 'es'
        )
        self.assertEqual(
            nmu.remove_special_char(
                'test0_*',
                ok_char=['_', '*'],
                bad_char=['t', '0']
            ),
            'es_*'
        )
        # int_list_to_seq_str
        i = [1, 2, 3, 4, 6]
        s = '1-4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 1, 2, 16, 145]
        s = '0-2, 16, 145'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 2, 4, 6]
        s = '0, 2, 4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 1, 5, 6, 7, 12, 19, 20, 21, 22, 124]
        s = '0, 1, 5-7, 12, 19-22, 124'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 4, 5.0, 6]
        s = '0, 4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, None, 4, 5, 6]
        s = '0, 4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, float('nan'), 4, 5, 6]
        s = '0, 4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 2, 4, 5, 6]
        s = '0,2,4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator=','), s)
        i = [0, 2, 3, 4, 6]
        s = '0$2-4$6'
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator='$'), s)
        i = [0, 2, 3, 4, 6]
        s = '0, 2-4, 6, '
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator_at_end=True), s)
        # channel_char
        self.assertEqual(nmu.channel_char(0), 'A')
        self.assertEqual(nmu.channel_char(1), 'B')
        self.assertEqual(nmu.channel_char(2), 'C')
        self.assertEqual(nmu.channel_char(10), 'K')
        self.assertEqual(nmu.channel_char(26), '')
        self.assertEqual(nmu.channel_char(-2), '')
        self.assertEqual(nmu.channel_char(float('inf')), '')
        self.assertEqual(nmu.channel_char(float('nan')), '')
        self.assertEqual(nmu.channel_char(None), '')
        self.assertEqual(nmu.channel_char([]), [])
        self.assertEqual(nmu.channel_char([0, 1, 3]), ['A', 'B', 'D'])
        self.assertEqual(nmu.channel_char([0, 1, 26]), ['A', 'B', ''])
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_char(0, char_list=clist), 'W')
        self.assertEqual(nmu.channel_char(3, char_list=clist), 'Z')
        self.assertEqual(nmu.channel_char(4, char_list=clist), '')
        self.assertEqual(nmu.channel_char([0, 1, 3], char_list=clist),
                         ['W', 'X', 'Z'])
        clist = ['AA', 'BB', 'CC', 'DD']
        self.assertEqual(nmu.channel_char(3, char_list=clist), 'DD')
        # channel_num
        self.assertEqual(nmu.channel_num(None), -1)
        self.assertEqual(nmu.channel_num('A'), 0)
        self.assertEqual(nmu.channel_num('a'), 0)
        self.assertEqual(nmu.channel_num('b'), 1)
        self.assertEqual(nmu.channel_num('K'), 10)
        self.assertEqual(nmu.channel_num(''), -1)
        self.assertEqual(nmu.channel_num('AA'), -1)
        clist = ['AA', 'BB', 'CC', 'DD']
        self.assertEqual(nmu.channel_num('AA', char_list=clist), 0)
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_num('A', char_list=clist), -1)
        self.assertEqual(nmu.channel_num('Y', char_list=clist), 2)
        self.assertEqual(nmu.channel_num([]), [])
        self.assertEqual(nmu.channel_num(['A', 'B', 'D']), [0, 1, 3])
        self.assertEqual(nmu.channel_num(['A', 'B', '@']), [0, 1, -1])
        self.assertEqual(nmu.channel_num(['c', 'a', 'f']), [2, 0, 5])
        self.assertEqual(nmu.channel_num(['A', 'B', 'C'], char_list=clist),
                         [-1, -1, -1])
        self.assertEqual(nmu.channel_num(['w', 'z', 'x'], char_list=clist),
                         [0, 3, 1])
        # channel_char_check
        self.assertEqual(nmu.channel_char_check(None), '')
        self.assertEqual(nmu.channel_char_check('A'), 'A')
        self.assertEqual(nmu.channel_char_check('Z'), 'Z')
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_char_check('A', char_list=clist), '')
        self.assertEqual(nmu.channel_char_check('Z', char_list=clist), 'Z')
        self.assertEqual(nmu.channel_char_check([]), [])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'D']),
                         ['A', 'B', 'D'])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'DD']),
                         ['A', 'B', ''])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'D'],
                         char_list=clist), ['', '', ''])
        # channel_char_search
        self.assertEqual(nmu.channel_char_search(None, 'A'), -1)
        self.assertEqual(nmu.channel_char_search('testA1', None), -1)
        with self.assertRaises(TypeError):
            self.assertEqual(nmu.channel_char_search('testA1', 'A1'), -1)
        with self.assertRaises(TypeError):
            self.assertEqual(nmu.channel_char_search('testA1', 'A$'), -1)
        self.assertEqual(nmu.channel_char_search('testA1', 'a'), 4)
        self.assertEqual(nmu.channel_char_search('testa111', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('testA', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('A', 'A'), 0)
        self.assertEqual(nmu.channel_char_search('testA111', 'B'), -1)
        self.assertEqual(nmu.channel_char_search('A', 'B'), -1)
        self.assertEqual(nmu.channel_char_search('taste', 'A'), -1)
        self.assertEqual(nmu.channel_char_search('testAA12', 'AA'), 4)
        self.assertEqual(nmu.channel_char_search('testAAA1267', 'AAA'), 4)
        self.assertEqual(nmu.channel_char_search('testAAA@1267', 'AAA'), 4)
        self.assertEqual(nmu.channel_char_search('testA@1267', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('A@1267', 'A'), 0)
        # history_change
        # history
        quiet = False
        fxn = '_test_utilities'
        c = 'Test'  # this class
        h = 'history message'
        r = 'nm.' + c + '.' + fxn + ': ' + h
        self.assertEqual(nmu.history(h, quiet=quiet), r)
        tp = 'nm.one.two.three'
        r = tp + ': ' + h
        self.assertEqual(nmu.history(h, tp=tp, quiet=quiet), r)
        """
        # get_treepath
        stack = inspect.stack()
        fxn = 'test_all'  # calling fxn
        r = 'nm.' + c + '.' + fxn
        self.assertEqual(nmu.get_treepath(stack), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.' + fxn
        self.assertEqual(nmu.get_treepath(stack, tp=tp), r)
        # get_class
        stack = inspect.stack()
        self.assertEqual(nmu.get_class(stack), c)
        self.assertEqual(nmu.get_class(stack, module=True), '__main__.' + c)
        # get_method
        stack = inspect.stack()
        self.assertEqual(nmu.get_method(stack), fxn)
        """


class NMObjectTest(NMObject):

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.myvalue = 1

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'myvalue': self.myvalue})
        return k


if __name__ == '__main__':
    unittest.main()
