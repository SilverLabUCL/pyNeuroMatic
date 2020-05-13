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
from nm_dimension import Dimension
from nm_dimension import XDimension
from nm_manager import Manager
from nm_object import NMObject
from nm_object import NMObjectTest
from nm_object import NMObjectContainer
from nm_note import Note
from nm_note import NoteContainer
from nm_project import Project
import nm_preferences as nmp
import nm_utilities as nmu

nm = Manager(new_project=False, quiet=True)
PARENT = nm
BADTYPES = [None, True, 1, 3.14, [], (), {}, set(), 'test', nm]
# BADTYPES: all types, use continue to ignore OK types
BADNAME = 'b&dn@me!'
BADNAMES = ['select', 'default', 'all'] + [BADNAME, '']
# BADNAMES: special NM argument flags, list may need updating
PLIST = ['name', 'date', 'modified']
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
        # self._test_note()
        # self._test_note_container()
        # self._test_dimension()
        # self._test_data()
        # self._test_data_container()
        # self._test_channel()
        # self._test_channel_container()
        self._test_dataseries_set()
        # self._test_dataseries_set_container()
        # self._test_utilities()

    def _test_nmobject(self):
        # args: parent, name
        # not testing parent as it can be any object
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
        # parameters
        self.assertEqual(o0._param_list, PLIST)
        self.assertTrue(o0._param_test())
        o0._param_list += ['test']
        self.assertFalse(o0._param_test())
        # content
        content_name = 'nmobject'
        self.assertEqual(o0._content_name, content_name)
        self.assertEqual(o0.content, {content_name: o0.name})
        self.assertEqual(o0.content_tree, {content_name: o0.name})
        # treepath
        self.assertEqual(o0._tp, o0.name)
        self.assertEqual(o0._tp_check(0), '')
        self.assertEqual(o0._tp_check(None), '')
        self.assertEqual(o0._tp_check(''), '')
        self.assertEqual(o0._tp_check('self'), o0.name)
        self.assertEqual(o0._tp_check('nm.test'), 'nm.test')
        self.assertEqual(o0.treepath(), o0.name)
        self.assertEqual(o0.treepath_list(), [o0.name])
        # manager
        self.assertEqual(o0._manager, nm)
        # name_ok, args: name, ok
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            self.assertFalse(o0.name_ok(b))
        for b in BADNAMES:
            if b == '':
                continue  # ok
            self.assertFalse(o0.name_ok(b))
        for b in BADTYPES:
            if isinstance(b, list) or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                o0.name_ok('test', ok=b)
        badnames = ['select', 'default', 'all']  # may need updating
        self.assertEqual(o0._bad_names, badnames)  # check if list changes
        for b in badnames:
            self.assertFalse(o0.name_ok(b))
        for b in badnames:
            self.assertTrue(o0.name_ok(b, ok=badnames))
        for n in [n0, n1, '']:
            self.assertTrue(o0.name_ok(n))
        # name_set, args: name_notused, newname
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
        # rename_fxnref_set, args: rename_fxnref
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                o0._rename_fxnref_set(b)
        o0.name = 'test1'
        self.assertEqual(o0.name, 'test1')
        self.assertTrue(o0._rename_fxnref_set(self.rename))
        o0.name = 'test2'
        self.assertEqual(o0.name, 'test1')  # does not change
        # isequivalent, args: nmobject
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
        # copy
        time.sleep(2)  # forces date to be different
        o0 = NMObject(PARENT, n0)
        c = o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(o0._isequivalent(c, alert=ALERT))
        self.assertEqual(o0._parent, c._parent)
        self.assertEqual(o0.name, c.name)
        self.assertNotEqual(o0._NMObject__date, c._NMObject__date)
        self.assertNotEqual(o0._NMObject__modified, c._NMObject__modified)
        self.assertEqual(c._NMObject__rename_fxnref, c._name_set)
        fr0 = o0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        self.assertNotEqual(fr0, frc)  # should be different
        # modified
        m1 = o0._NMObject__modified
        o0._modified()
        self.assertNotEqual(m1, o0._NMObject__modified)
        # alert, error, history
        # wrappers for nmu.history()
        # type_error, args: obj, type_expected, tp, quiet, frame
        dum_arg = {}
        e1 = o0._type_error(dum_arg, 'list')
        e2 = ('nm.object0._test_nmobject: ' +
              'bad dum_arg: expected list but got dict')
        self.assertEqual(e1, e2)
        # value_error, args: obj, tp, quiet, frame
        dum_str = 'test'
        e1 = o0._value_error(dum_str)
        e2 = ("nm.object0._test_nmobject: bad dum_str: 'test'")
        self.assertEqual(e1, e2)
        # quiet, args: quiet
        self.assertFalse(o0._quiet(False))
        self.assertTrue(o0._quiet(True))
        q = nm.configs.quiet
        nm.configs.quiet = True  # Manager quiet overrides when True
        self.assertTrue(o0._quiet(False))
        self.assertTrue(o0._quiet(True))
        nm.configs.quiet = q
        # save  # TODO

    def rename(self, name, newname, quiet=nmp.QUIET):
        # dummy function to test NMObject._rename_fxnref_set
        if not nm.configs.quiet:
            print('test rename: ' + name + ' -> ' + newname)
        return False

    def _test_nmobject_container(self):
        # args: parent, name, type_, prefix, rename, copy
        n0 = 'container0'
        n1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        nlist = [p0 + str(i) for i in range(0, 6)]
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = NMObjectContainer(PARENT, n0, type_=b)
        for b in [BADNAME, '']:
            with self.assertRaises(ValueError):
                c0 = NMObjectContainer(PARENT, n0, type_=b)
        for b in BADTYPES:
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = NMObjectContainer(PARENT, n0, prefix=b)
        for b in BADNAMES:
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0 = NMObjectContainer(PARENT, n0, prefix=b)
        c0 = NMObjectContainer(PARENT, n0, prefix='')
        self.assertEqual(c0.prefix, '')  # '' is ok
        c0 = NMObjectContainer(PARENT, n0, type_=type_, prefix=p0, rename=True)
        c1 = NMObjectContainer(PARENT, n1, type_=type_, prefix=p1,
                               rename=False)
        self.assertEqual(c0.parameters['type'], type_)
        self.assertTrue(c0.parameters['rename'])
        self.assertEqual(c0.prefix, p0)
        self.assertIsNone(c0.select)
        self.assertEqual(c0.names, [])
        self.assertEqual(c0.count, 0)
        self.assertFalse(c1.parameters['rename'])
        # parameters
        plist = PLIST + ['type', 'prefix', 'rename', 'select']
        self.assertEqual(c0._param_list, plist)
        self.assertTrue(c0._param_test())
        # content
        content_name = 'nmobjects'
        self.assertEqual(c0._content_name, content_name)
        # prefix_set, args: prefix
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0._prefix_set(b)
        for b in BADNAMES:
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0._prefix_set(b)
        for p in ['', p1, p0]:
            self.assertTrue(c0._prefix_set(p))
            self.assertEqual(c0.prefix, p)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c1.prefix, p1)
        # name_next
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), nlist[0])
        # new, args: name, nmobject, select
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.new(name=b)
        for b in BADNAMES:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.new(name=b)
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0.new(nmobject=b)
        o = c0.new()
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, nlist[0])
        self.assertEqual(c0.select, o)
        self.assertEqual(c0.select.name, nlist[0])
        self.assertEqual(o._NMObject__rename_fxnref, c0.rename)
        with self.assertRaises(RuntimeError):
            c0.new(name=nlist[0])  # already exists
        self.assertEqual(c0.name_next_seq(), 1)
        self.assertEqual(c0.name_next(), nlist[1])
        o = c0.new(select=False)
        self.assertEqual(o.name, nlist[1])
        self.assertEqual(c0.select.name, nlist[0])
        self.assertEqual(c0.name_next_seq(), 2)
        self.assertEqual(c0.name_next(), nlist[2])
        o = c0.new(nlist[2])
        self.assertEqual(o.name, nlist[2])
        # skip n[3]
        o = NMObject(PARENT, nlist[4])
        o = c0.new(name=nlist[4], nmobject=o)
        self.assertEqual(o.name, nlist[4])
        o = c0.new()
        self.assertEqual(o.name, nlist[5])
        self.assertEqual(c0.count, 5)
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        with self.assertRaises(RuntimeError):
            o.name = nlist[0]  # already exists
        # names
        self.assertEqual(c0.names, [nlist[0], nlist[1], nlist[2], nlist[4],
                                    nlist[5]])
        # content
        c = c0.content
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)
        # index, args: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.index(b)
        for b in BADNAMES:
            if b.lower() == 'select':
                continue  # ok
            self.assertEqual(c0.index(b), -1)
        self.assertEqual(c0.index(nlist[0]), 0)
        self.assertEqual(c0.index(nlist[1]), 1)
        self.assertEqual(c0.index(nlist[2]), 2)
        self.assertEqual(c0.index(nlist[3]), -1)  # does not exist
        self.assertEqual(c0.index(nlist[4]), 3)
        self.assertEqual(c0.index(nlist[5]), 4)
        self.assertEqual(c0.index('select'), 4)
        self.assertEqual(c1.index('select'), 2)
        # exists, args: name
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
                self.assertFalse(c0.exists(nlist[i]))
            else:
                self.assertTrue(c0.exists(nlist[i]))
        # getitem, args: name, index
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitem(name=b)
        for b in BADNAMES:
            if b.lower() == 'select' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.getitem(name=b)
        for b in BADTYPES:
            if b is None or isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitem(index=b)
        self.assertIsNone(c0.getitem(name=''))
        self.assertEqual(c0.getitem(name='select'), c0.select)
        for i in range(0, len(nlist)):
            o = c0.getitem(name=nlist[i])
            if i == 3:
                self.assertIsNone(o)
            else:
                self.assertIsInstance(o, NMObject)
                self.assertEqual(o.name, nlist[i])
        self.assertIsNone(c0.getitem(index=None))
        for i in range(0, c0.count):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            if i <= 2:
                self.assertEqual(o.name, nlist[i])
            else:
                self.assertEqual(o.name, nlist[i+1])
        for i in range(-1, -1 * (c0.count + 1)):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
        i = -1 * (c0.count + 1)
        for b in [c0.count, 100, i, -100]:
            with self.assertRaises(IndexError):
                c0.getitem(index=b)
        # getitems, args: names, indexes
        for b in BADTYPES:
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(names=b)
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(names=[b])
        for b in BADTYPES:
            if isinstance(b, list) or isinstance(b, tuple):
                continue  # ok
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(indexes=b)
        for b in BADTYPES:
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.getitems(indexes=[b])
        for b in BADNAMES:
            if b.lower() == 'select' or b.lower() == 'all' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.getitems(names=[b])
        self.assertEqual(c0.getitems(), [])
        i = -1 * (c0.count + 1)
        for b in [c0.count, 50, i]:
            with self.assertRaises(IndexError):
                olist1 = c0.getitems(indexes=[b])
        olist0 = c0.getitems(names=[nlist[0], nlist[1], nlist[2], nlist[4]])
        olist1 = c0.getitems(indexes=[0, 1, 2, 3])
        self.assertEqual(olist0, olist1)
        olist1 = c0.getitems(indexes=[-5, -4, -3, -2])
        self.assertEqual(olist0, olist1)
        olist0 = c0.getitems(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()  # pop should NOT change container list
        self.assertEqual(len(olist0), c0.count-1)
        # select_set, args: name, failure_alert
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0._select_set(b)
        for b in BADNAMES:
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0._select_set(b)
        sname = c0.select.name
        self.assertIsNone(c0._select_set(''))
        self.assertIsNone(c0._select_set(nlist[3]))
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        self.assertIsInstance(c0._select_set(nlist[0]), NMObject)
        self.assertEqual(c0.select.name, nlist[0])
        # rename, args: name, newname
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(b, nlist[3])
        for b in BADNAMES:
            if b.lower() == 'select' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(b, nlist[3])
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(nlist[4], b)
        for b in BADNAMES:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(nlist[4], b)
        with self.assertRaises(RuntimeError):
            c0.rename('select', nlist[0])  # name already used
        with self.assertRaises(RuntimeError):
            c1.rename('select', 'test')  # rename = False
        s = c0.rename('', nlist[3])
        self.assertIsInstance(s, str)
        self.assertEqual(s, '')
        i = c0.index(nlist[4])
        nnext = c0.name_next()
        s = c0.rename(nlist[4], 'default')
        self.assertEqual(s, nnext)
        s = c0.rename(nnext, nlist[3])
        self.assertEqual(s, nlist[3])
        o = c0.getitem(index=i)
        self.assertEqual(o.name, nlist[3])
        self.assertEqual(c0.rename(nlist[5], nlist[4]), nlist[4])
        self.assertEqual(c0.names, [nlist[0], nlist[1], nlist[2], nlist[3],
                                    nlist[4]])
        # duplicate, args: name, newname, select
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.duplicate(b, 'default')
        for b in BADNAMES:
            if b.lower() == 'select' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(b, 'default')
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.duplicate(nlist[0], b)
        for b in BADNAMES:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(nlist[0], b)
        with self.assertRaises(RuntimeError):
            c0.duplicate(nlist[0], nlist[1])  # already exists
        self.assertIsNone(c0.duplicate('', 'default'))
        icount = c0.count
        o = c0.getitem(name=nlist[0])
        nnext = c0.name_next()
        c = c0.duplicate(nlist[0], 'default')
        self.assertIsInstance(c, NMObject)
        self.assertEqual(c.name, nnext)
        self.assertEqual(c._NMObject__rename_fxnref, c0.rename)
        self.assertFalse(o._isequivalent(c, alert=ALERT))  # names different
        self.assertEqual(c0.count, icount + 1)
        with self.assertRaises(RuntimeError):
            c.name = nlist[0]  # name already used
        c._NMObject__name = nlist[0]
        self.assertTrue(o._isequivalent(c, alert=ALERT))
        c._NMObject__name = nnext
        # isequivalent, args: NMObjectContainer
        for b in BADTYPES:
            self.assertFalse(c0._isequivalent(b, alert=ALERT))
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        self.assertFalse(c0._isequivalent(c0, alert=ALERT))
        # copy
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
        self.assertNotEqual(c0._NMObject__date, c._NMObject__date)
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
        # kill, args: names, indexes, confirm
        # kill uses getitems, so no need to test BADTYPES and BADNAMES
        self.assertEqual(c0.kill(), [])
        s = c0.select
        klist = c0.kill(names='select', confirm=CONFIRM)
        self.assertEqual(klist, [s])
        self.assertIsNone(c0.getitem(name=s.name))
        self.assertEqual(c0.select.name, nlist[0])
        o = c0.getitem(name=nlist[0])
        klist = c0.kill(names=nlist[0], confirm=CONFIRM)
        self.assertEqual(klist, [o])
        self.assertIsNone(c0.getitem(name=nlist[0]))
        self.assertEqual(c0.select.name, nlist[1])
        names = c0.names
        klist = c0.kill(names='all', confirm=CONFIRM)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(c0.count, 0)
        self.assertIsNone(c0.select)
        olist = c1.getitems(indexes=[0, 1, 2])
        klist = c1.kill(indexes=[0, 1, 2], confirm=CONFIRM)
        self.assertEqual(klist, olist)
        self.assertIsNone(c1.select)

    def _test_note(self):
        # args: parent, name, thenote, copy
        thenote = 'note!'
        n0 = Note(PARENT, 'Note0', thenote=thenote)
        self.assertEqual(n0.thenote, thenote)
        # parameters
        plist = PLIST + ['thenote']
        self.assertEqual(n0._param_list, plist)
        self.assertTrue(n0._param_test())
        # content
        self.assertEqual(n0._content_name, 'note')
        # isequivalent, args: Note
        n1 = Note(PARENT, 'Note0', thenote=thenote)
        self.assertTrue(n0._isequivalent(n1, alert=ALERT))
        n1 = Note(PARENT, 'Note0', thenote='different')
        self.assertFalse(n0._isequivalent(n1, alert=ALERT))
        # copy
        c = n0.copy()
        self.assertIsInstance(c, Note)
        self.assertTrue(n0._isequivalent(c, alert=ALERT))
        self.assertEqual(n0.thenote, c.thenote)
        # thenote_set, args: thenote
        self.assertTrue(n0._thenote_set(123))
        self.assertEqual(n0.thenote, '123')
        self.assertTrue(n0._thenote_set(None))
        self.assertEqual(n0.thenote, '')
        self.assertTrue(n0._thenote_set('test'))
        self.assertEqual(n0.thenote, 'test')
        self.assertTrue(n0._thenote_set([1]))
        self.assertEqual(n0.thenote, '[1]')

    def _test_note_container(self):
        # args: parent, name, copy
        txt = 'test #'
        prefix = 'Note'
        nlist = [prefix + str(i) for i in range(0, 4)]
        thenotes = [txt + str(i) for i in range(0, 4)]
        notes = NoteContainer(PARENT, 'Notes')
        self.assertEqual(notes.parameters['type'], 'Note')
        self.assertEqual(notes.prefix, prefix)
        self.assertFalse(notes.parameters['rename'])
        self.assertFalse(notes._NoteContainer__off)
        # content
        self.assertEqual(notes._content_name, 'notes')
        # new, args: thenote, select
        self.assertEqual(notes.name_next_seq(), 0)
        self.assertEqual(notes.name_next(), nlist[0])
        for i in range(0, 4):
            thenote = txt + str(i)
            n = notes.new(thenote=thenote)
            self.assertIsInstance(n, Note)
            self.assertEqual(n.name, nlist[i])
            self.assertEqual(n.thenote, thenote)
        notes.off = True
        self.assertIsNone(notes.new('test'))
        # copy
        c = notes.copy()
        self.assertIsInstance(c, NoteContainer)
        self.assertTrue(notes._isequivalent(c, alert=ALERT))
        # thenotes
        self.assertEqual(thenotes, notes.thenotes())
        self.assertEqual(thenotes, c.thenotes())
        # duplicate
        with self.assertRaises(RuntimeError):
            notes.duplicate()

    def _test_dimension(self):
        # args: parent, name, dim, notes, copy
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = Dimension(PARENT, 'ydim0', dim=b)
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = Dimension(PARENT, 'ydim0', notes=b)
        notes = NoteContainer(PARENT, 'Notes')
        y0 = Dimension(PARENT, 'ydim0', dim=YDIM0, notes=notes)
        x0 = XDimension(PARENT, 'xdim0', dim=XDIM0, notes=notes)
        y1 = Dimension(PARENT, 'ydim1', dim=YDIM1, notes=notes)
        x1 = XDimension(PARENT, 'xdim1', dim=XDIM1, notes=notes)
        self.assertEqual(y0._note_container, notes)
        self.assertEqual(y0._offset, YDIM0['offset'])
        self.assertEqual(y0._label, YDIM0['label'])
        self.assertEqual(y0._units, YDIM0['units'])
        self.assertIsNone(y0._master)
        self.assertEqual(x0._note_container, notes)
        self.assertEqual(x0._offset, XDIM0['offset'])
        self.assertEqual(x0._start, XDIM0['start'])
        self.assertEqual(x0._delta, XDIM0['delta'])
        self.assertEqual(x0._label, XDIM0['label'])
        self.assertEqual(x0._units, XDIM0['units'])
        self.assertIsNone(x0._master)
        self.assertIsNone(x0._xdata)
        # parameters
        plist = PLIST + ['offset', 'label', 'units', 'master']
        self.assertEqual(y0._param_list, plist)
        self.assertTrue(y0._param_test())
        xplist = plist + ['start', 'delta', 'xdata']
        self.assertEqual(x0._param_list, xplist)
        self.assertTrue(x0._param_test())
        # content
        self.assertEqual(y0._content_name, 'dimension')
        self.assertEqual(x0._content_name, 'xdimension')
        # note_new, arg: thenote
        note = 'test123'
        n = y0._note_new(note)
        self.assertIsInstance(n, Note)
        self.assertEqual(n.thenote, note)
        n = x0._note_new(note)
        self.assertEqual(n.thenote, note)
        # master_set, arg: dimension
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
        self.assertIsInstance(y1._master, Dimension)
        self.assertEqual(y1._master, y0)
        self.assertTrue(y1._master_lock)
        self.assertTrue(x1._master_set(x0))
        self.assertIsInstance(x1._master, XDimension)
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
        # dim_set, arg: dim
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
        # offset_set, arg: offset
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
        # label_set, args: label
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
        # units_set, arg: units
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
        # start_set, args: start
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
        # delta_set, args: delta
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
        # xdata_set, args|: xdata
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
        # copy
        c = y0.copy()
        self.assertIsInstance(c, Dimension)
        self.assertTrue(y0._isequivalent(c, alert=ALERT))
        c = x0.copy()
        self.assertIsInstance(c, XDimension)
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
        xdata = Data(PARENT, 'xdata', np_array=nparrayx, xdim=XDIMx,
                     ydim=YDIMx)
        self.assertTrue(np.array_equal(d0._Data__np_array, nparray0))
        self.assertTrue(np.array_equal(d1._Data__np_array, nparray1))
        # parameters
        plist = PLIST + ['xdim', 'ydim', 'dataseries']
        self.assertEqual(d0._param_list, plist)
        self.assertTrue(d0._param_test())
        # content
        content_name = 'data'
        self.assertEqual(d0._content_name, content_name)
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
        # content
        self.assertEqual(c0._content_name, 'data')
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
        # kill, args: names, indexes, confirm
        # wrapper for NMObjectContainer.kill()
        # TODO, test if Data is removed from dataseries and sets

    def _test_channel(self):
        # args: parent, name, xdim, ydim, copy
        # ydim tested by Dimension
        # xdim tested by XDimension
        c0 = Channel(PARENT, 'A', xdim=XDIM0, ydim=YDIM0)
        for k in XDIM0.keys():
            self.assertEqual(c0.x.dim[k], XDIM0[k])
        for k in YDIM0.keys():
            self.assertEqual(c0.y.dim[k], YDIM0[k])
        # parameters
        plist = PLIST + ['xdim', 'ydim']
        self.assertEqual(c0._param_list, plist)
        self.assertTrue(c0._param_test())
        # content
        self.assertEqual(c0._content_name, 'channel')
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
        # content
        self.assertEqual(c0._content_name, 'channels')
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
        self.assertEqual(s1._param_list, plist)
        self.assertTrue(s1._param_test())
        # content
        self.assertEqual(s1._content_name, 'dataseriesset')
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
        self.assertTrue(nmu.name_ok(''))
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
        # number_ok
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
        # remove_special_chars
        self.assertEqual(nmu.remove_special_chars('test*'), 'test')
        self.assertEqual(nmu.remove_special_chars('test'), 'test')
        self.assertEqual(nmu.remove_special_chars('t@st*'), 'tst')
        self.assertEqual(nmu.remove_special_chars(''), '')
        self.assertEqual(nmu.remove_special_chars(None), '')
        self.assertEqual(nmu.remove_special_chars(['test*']), '')
        # int_list_to_seq_str
        self.assertEqual(nmu.int_list_to_seq_str([1, 2, 3, 4, 6]), '1-4, 6')
        i = [0, 1, 2, 16, 145]
        r = '0-2, 16, 145'
        self.assertEqual(nmu.int_list_to_seq_str(i), r)
        self.assertEqual(nmu.int_list_to_seq_str([0, 2, 4, 6]), '0, 2, 4, 6')
        i = [0, 1, 5, 6, 7, 12, 19, 20, 21, 22, 124]
        r = '0,1,5-7,12,19-22,124'
        self.assertEqual(nmu.int_list_to_seq_str(i, space=False), r)
        # chan_char
        self.assertEqual(nmu.chan_char(0), 'A')
        self.assertEqual(nmu.chan_char(1), 'B')
        self.assertEqual(nmu.chan_char(2), 'C')
        self.assertEqual(nmu.chan_char(10), 'K')
        self.assertEqual(nmu.chan_char(26), '')
        self.assertEqual(nmu.chan_char(-2), '')
        self.assertEqual(nmu.chan_char(float('inf')), '')
        self.assertEqual(nmu.chan_char(float('nan')), '')
        self.assertEqual(nmu.chan_char(None), '')
        self.assertEqual(nmu.chan_char([0, 1, 2]), '')
        # chan_num
        self.assertEqual(nmu.chan_num('A'), 0)
        self.assertEqual(nmu.chan_num('a'), 0)
        self.assertEqual(nmu.chan_num('b'), 1)
        self.assertEqual(nmu.chan_num('K'), 10)
        self.assertEqual(nmu.chan_num(''), -1)
        self.assertEqual(nmu.chan_num('AA'), -1)
        self.assertEqual(nmu.chan_num(None), -1)
        self.assertEqual(nmu.chan_num(['A', 'B', 'C']), -1)
        # chan_char_exists
        self.assertTrue(nmu.chan_char_exists('testA1', 'A'))
        self.assertTrue(nmu.chan_char_exists('testa111', 'A'))
        self.assertTrue(nmu.chan_char_exists('testA', 'A'))
        self.assertTrue(nmu.chan_char_exists('A', 'A'))
        self.assertFalse(nmu.chan_char_exists('testA111', 'B'))
        self.assertFalse(nmu.chan_char_exists('A', 'B'))
        self.assertFalse(nmu.chan_char_exists('taste', 'A'))
        # history
        quiet = True
        fxn = '_test_utilities'
        c = 'Test'  # this class
        h = 'testing code'
        r = 'nm.' + c + '.' + fxn + ': ' + h
        self.assertEqual(nmu.history(h, quiet=quiet), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.' + fxn + ': ' + h
        self.assertEqual(nmu.history(h, tp=tp, quiet=quiet), r)
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


if __name__ == '__main__':
    unittest.main()
