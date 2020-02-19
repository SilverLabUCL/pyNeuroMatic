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
from nm_container import NMObject
from nm_container import Container
from nm_data import Data
from nm_data import DataContainer
from nm_dataseries import DataSeries
from nm_dataseries import DataSeriesContainer
from nm_dimension import Dimension
from nm_dimension import XDimension
from nm_epochset import EpochSet
from nm_epochset import EpochSetContainer
from nm_manager import Manager
from nm_note import Note
from nm_note import NoteContainer
from nm_project import Project
import nm_preferences as nmp
import nm_utilities as nmu

nm = Manager(new_project=False, quiet=True)
PARENT = nm
BADTYPES = [None, True, 1, 3.14, [], (), {}, set(), 'test', nm]
BADNAME = 'b&dn@me!'
BADNAMES = ['select', 'default', 'all'] + [BADNAME, '']  # may need updating
YDIM0 = {'offset': 0, 'label': 'Vmem', 'units': 'mV'}
XDIM0 = {'offset': 0, 'start': 10, 'delta': 0.01, 'label': 'time',
         'units': 'ms'}
YDIM1 = {'offset': 0, 'label': 'Imem', 'units': 'pA'}
XDIM1 = {'offset': 0, 'start': -10, 'delta': 0.2, 'label': 't',
         'units': 'seconds'}
YDIMx = {'offset': 0, 'label': 'time interval', 'units': 'usec'}
XDIMx = {'offset': 0, 'start': 0, 'delta': 1, 'label': 'sample',
         'units': '#'}


def test_type_error(a, b, c, got):
    return nmu.type_error(c, got)


class Test(unittest.TestCase):

    def test_all(self):
        nm.configs.quiet = False
        # self._test_nmobject()
        self._test_container()
        # self._test_note()
        # self._test_note_container()
        # self._test_dimension()
        # self._test_data()
        # self._test_data_container()
        # self._test_channel()
        # self._test_channel_container()
        # self._test_epochset()
        # self._test_utilities()

    def _test_nmobject(self):
        name0 = 'object0'
        name1 = 'object1'
        content_name = 'nmobject'
        plist = ['name', 'date', 'modified']
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                o0 = NMObject(PARENT, b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                o0 = NMObject(PARENT, b)
        for b in BADTYPES:
            if isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                o0 = NMObject(PARENT, name0, fxns=b)
        o0 = NMObject(PARENT, name0, fxns=nm._fxns)
        o1 = NMObject(PARENT, name1, fxns=nm._fxns)
        self.assertEqual(o0.name, name0)
        self.assertEqual(o0._quiet, nm._fxns['quiet'])
        self.assertEqual(o0._rename, o0._NMObject__rename)
        self.assertEqual(o0._content_name, content_name)
        self.assertEqual(o0._param_list, plist)
        # parameters
        self.assertTrue(o0._param_test())
        # content
        self.assertEqual(o0.content, {content_name: o0.name})
        self.assertEqual(o0.content_tree, {content_name: o0.name})
        # treepath
        self.assertEqual(o0.treepath(), o0.name)
        self.assertEqual(o0.treepath_list(), [o0.name])
        # name_ok
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            self.assertFalse(o0.name_ok(b))
        for b in BADNAMES:
            if b == '':
                continue  # ok
            self.assertFalse(o0.name_ok(b))
        badnames = ['select', 'default', 'all']  # may need updating
        self.assertEqual(o0._bad_names, badnames)  # check if list changes
        for b in badnames:
            self.assertFalse(o0.name_ok(b))
        for b in badnames:
            self.assertTrue(o0.name_ok(b, ok=badnames))
        for n in [name0, name1, '']:
            self.assertTrue(o0.name_ok(n))
        # name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                o0._rename('', b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                o0._rename('', b)
        for n in ['test', name0]:
            self.assertTrue(o0._rename('', n))
            self.assertEqual(n, o0.name)
        # equal
        for b in BADTYPES:
            self.assertFalse(o0._equal(b, alert=True))
        self.assertFalse(o0._equal(o1, alert=True))
        self.assertTrue(o0._equal(o0, alert=True))
        o0 = NMObject(PARENT, name0, fxns=nm._fxns)
        o1 = NMObject(PARENT, name0, fxns=nm._fxns)
        self.assertTrue(o0._equal(o1, alert=True))
        o1 = NMObject(PARENT, name0, fxns={})  # different quiet()
        self.assertFalse(o0._equal(o1, alert=True))
        # copy
        time.sleep(2)  # forces date to be different
        c = o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(o0._equal(c, alert=True))
        self.assertEqual(o0._parent, c._parent)
        self.assertEqual(o0._fxns, c._fxns)
        self.assertEqual(o0.name, c.name)
        self.assertNotEqual(o0._NMObject__date, c._NMObject__date)
        self.assertNotEqual(o0._NMObject__modified, c._NMObject__modified)
        self.assertEqual(c._quiet, nm._fxns['quiet'])
        self.assertEqual(c._rename, c._NMObject__rename)
        self.assertEqual(o0._quiet, c._quiet)
        self.assertNotEqual(o0._rename, c._rename)
        # save TODO

    def _test_container(self):
        name0 = 'container0'
        name1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        content_name = 'nmobjects'
        plist = ['name', 'date', 'modified', 'type', 'prefix', 'rename',
                 'select']
        n = [p0 + str(i) for i in range(0, 6)]
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = Container(PARENT, name0, type_=b)
        for b in [BADNAME, '']:
            with self.assertRaises(ValueError):
                c0 = Container(PARENT, name0, type_=b)
        for b in BADTYPES:
            if b is None or isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = Container(PARENT, name0, prefix=b)
        for b in BADNAMES:
            if b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0 = Container(PARENT, name0, prefix=b)
        c0 = Container(PARENT, name0, prefix='')
        self.assertEqual(c0.prefix, '')  # '' is ok
        c0 = Container(PARENT, name0, fxns=nm._fxns, type_=type_, prefix=p0,
                       rename=True)
        c1 = Container(PARENT, name1, fxns=nm._fxns, type_=type_, prefix=p1,
                       rename=False)
        self.assertEqual(c0._rename, c0.rename)  # fxn refs
        self.assertEqual(c0.name, name0)
        self.assertEqual(c0._type, type_)
        self.assertTrue(c0._rename_)
        self.assertEqual(c0.prefix, p0)
        self.assertIsNone(c0.select)
        self.assertEqual(c0._content_name, content_name)
        self.assertEqual(c0.names, [])
        self.assertEqual(c0.count, 0)
        self.assertEqual(c0._param_list, plist)
        self.assertFalse(c1._rename_)
        # parameters
        self.assertTrue(c0._param_test())
        # prefix
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
        for p in ['', p1]:
            self.assertTrue(c0._prefix_set(p))
            self.assertEqual(c0.prefix, p)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c1.prefix, p1)
        c0._prefix_set(p0)  # reset
        # name_next
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), n[0])
        # new
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
        self.assertEqual(o.name, n[0])
        self.assertEqual(c0.select, o)
        self.assertEqual(c0.select.name, n[0])
        with self.assertRaises(RuntimeError):
            c0.new(name=n[0])  # already exists
        self.assertEqual(c0.name_next_seq(), 1)
        self.assertEqual(c0.name_next(), n[1])
        o = c0.new(select=False)
        self.assertEqual(o.name, n[1])
        self.assertEqual(c0.select.name, n[0])
        self.assertEqual(c0.name_next_seq(), 2)
        self.assertEqual(c0.name_next(), n[2])
        o = c0.new(n[2])
        self.assertEqual(o.name, n[2])
        # skip n[3]
        o = NMObject(PARENT, n[4], fxns=nm._fxns)
        o = c0.new(name=n[4], nmobject=o)
        self.assertEqual(o.name, n[4])
        o = c0.new()
        self.assertEqual(o.name, n[5])
        self.assertEqual(c0.count, 5)
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        # names
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[4], n[5]])
        # content
        c = c0.content
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], c0.names)
        # index
        for b in BADTYPES:
            self.assertEqual(c0.index(b), -1)
        for b in BADNAMES:
            if b.lower() == 'select':
                continue  # ok
            self.assertEqual(c0.index(b), -1)
        self.assertEqual(c0.index(n[0]), 0)
        self.assertEqual(c0.index(n[1]), 1)
        self.assertEqual(c0.index(n[2]), 2)
        self.assertEqual(c0.index(n[3]), -1)  # does not exist
        self.assertEqual(c0.index(n[4]), 3)
        self.assertEqual(c0.index(n[5]), 4)
        self.assertEqual(c0.index('select'), 4)
        self.assertEqual(c1.index('select'), 2)
        # exists
        for b in BADTYPES:
            self.assertFalse(c0.exists(b))
        for b in BADNAMES:
            if b.lower() == 'select':
                self.assertTrue(c0.exists(b))
            else:
                self.assertFalse(c0.exists(b))
        for i in range(0, 6):
            if i == 3:
                self.assertFalse(c0.exists(n[i]))
            else:
                self.assertTrue(c0.exists(n[i]))
        # getitem
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
        for i in range(0, len(n)):
            o = c0.getitem(name=n[i])
            if i == 3:
                self.assertIsNone(o)
            else:
                self.assertIsInstance(o, NMObject)
                self.assertEqual(o.name, n[i])
        self.assertIsNone(c0.getitem(index=None))
        for i in range(0, c0.count):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            if i <= 2:
                self.assertEqual(o.name, n[i])
            else:
                self.assertEqual(o.name, n[i+1])
        for i in range(-1, -1 * (c0.count + 1)):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
        i = -1 * (c0.count + 1)
        for b in [c0.count, 100, i, -100]:
            with self.assertRaises(IndexError):
                c0.getitem(index=b)
        # getitems
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
            if b == 'select' or b == 'all' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.getitems(names=[b])
        self.assertEqual(c0.getitems(), [])
        i = -1 * (c0.count + 1)
        for b in [c0.count, 50, i]:
            with self.assertRaises(IndexError):
                olist1 = c0.getitems(indexes=[b])
        olist0 = c0.getitems(names=[n[0], n[1], n[2], n[4]])
        olist1 = c0.getitems(indexes=[0, 1, 2, 3])
        self.assertEqual(olist0, olist1)
        olist1 = c0.getitems(indexes=[-5, -4, -3, -2])
        self.assertEqual(olist0, olist1)
        olist0 = c0.getitems(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()  # pop should NOT change container list
        self.assertEqual(len(olist0), c0.count-1)
        # select
        sname = c0.select.name
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
        self.assertIsNone(c0._select_set(''))
        self.assertIsNone(c0._select_set(n[3]))
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        self.assertIsInstance(c0._select_set(n[0]), NMObject)
        self.assertEqual(c0.select.name, n[0])
        # rename
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(b, n[3])
        for b in BADNAMES:
            if b.lower() == 'select' or b == '':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(b, n[3])
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.rename(n[4], b)
        for b in BADNAMES:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(n[4], b)
        with self.assertRaises(RuntimeError):
            c0.rename('select', n[0])  # already exists
        with self.assertRaises(RuntimeError):
            c1.rename('select', 'test')  # rename = False
        s = c0.rename('', n[3])
        self.assertIsInstance(s, str)
        self.assertEqual(s, '')
        i = c0.index(n[4])
        nnext = c0.name_next()
        s = c0.rename(n[4], 'default')
        self.assertEqual(s, nnext)
        s = c0.rename(nnext, n[3])
        self.assertEqual(s, n[3])
        o = c0.getitem(index=i)
        self.assertEqual(o.name, n[3])
        self.assertEqual(c0.rename(n[5], n[4]), n[4])
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[3], n[4]])
        # duplicate
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
                c0.duplicate(n[0], b)
        for b in BADNAMES:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(n[0], b)
        with self.assertRaises(RuntimeError):
            c0.duplicate(n[0], n[1])  # already exists
        self.assertIsNone(c0.duplicate('', 'default'))
        icount = c0.count
        o = c0.getitem(name=n[0])
        nnext = c0.name_next()
        c = c0.duplicate(n[0], 'default')
        self.assertIsInstance(c, NMObject)
        self.assertEqual(c.name, nnext)
        self.assertFalse(o._equal(c, alert=True))  # names are different
        self.assertEqual(c0.count, icount + 1)
        with self.assertRaises(RuntimeError):
            c.name = n[0]  # must use Container.rename()
        c._NMObject__name = n[0]
        self.assertTrue(o._equal(c, alert=True))
        c._NMObject__name = nnext
        # equal
        for b in BADTYPES:
            self.assertFalse(c0._equal(b, alert=True))
        self.assertFalse(c0._equal(c1, alert=True))
        # copy
        c = c0.copy()
        self.assertIsInstance(c, Container)
        self.assertTrue(c0._equal(c, alert=True))
        self.assertEqual(c0._parent, c._parent)
        self.assertEqual(c0.name, c.name)
        self.assertEqual(c0._type, c._type)
        self.assertEqual(c0.prefix, c.prefix)
        self.assertEqual(c0._rename_, c._rename_)
        self.assertEqual(c0._quiet, c._quiet)
        self.assertNotEqual(c0._rename, c._rename)
        self.assertNotEqual(c0._NMObject__date, c._NMObject__date)
        self.assertNotEqual(c0._NMObject__modified, c._NMObject__modified)
        self.assertNotEqual(c0.select, c.select)  # refs not equal
        self.assertEqual(c0.select.name, c.select.name)  # but names equal
        for i in range(0, c0.count):
            o0 = c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertTrue(o0._equal(oc, alert=True))
        # kill
        # kill uses getitems, so no need to test BADTYPES and BADNAMES
        self.assertEqual(c0.kill(), [])
        s = c0.select
        klist = c0.kill(names='select', confirm=False)
        self.assertEqual(klist, [s])
        self.assertIsNone(c0.getitem(name=s.name))
        self.assertEqual(c0.select.name, n[0])
        o = c0.getitem(name=n[0])
        klist = c0.kill(names=n[0], confirm=False)
        self.assertEqual(klist, [o])
        self.assertIsNone(c0.getitem(name=n[0]))
        self.assertEqual(c0.select.name, n[1])
        names = c0.names
        klist = c0.kill(names='all', confirm=False)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(c0.count, 0)
        self.assertIsNone(c0.select)
        olist = c1.getitems(indexes=[0, 1, 2])
        klist = c1.kill(indexes=[0, 1, 2], confirm=False)
        self.assertEqual(klist, olist)
        self.assertIsNone(c1.select)

    def _test_note(self):
        thenote = 'note!'
        plist = ['name', 'rename', 'date', 'modified', 'thenote']
        note0 = Note(PARENT, 'Note0', fxns=nm._fxns, thenote=thenote)
        self.assertEqual(note0._content_name, 'note')
        self.assertEqual(note0.thenote, thenote)
        self.assertEqual(note0._param_list, plist)
        # parameters
        self.assertTrue(note0._param_test())
        # equal
        note1 = Note(PARENT, 'Note0', fxns=nm._fxns, thenote=thenote)
        self.assertTrue(note0._equal(note1, alert=True))
        note1 = Note(PARENT, 'Note0', fxns=nm._fxns, thenote='different')
        self.assertFalse(note0._equal(note1, alert=True))
        # copy
        c = note0.copy()
        self.assertIsInstance(c, Note)
        self.assertTrue(note0._equal(c, alert=True))
        self.assertEqual(note0._Note__thenote, c._Note__thenote)
        # thenote
        self.assertTrue(note0._thenote_set(123))
        self.assertEqual(note0.thenote, '123')
        self.assertTrue(note0._thenote_set(None))
        self.assertEqual(note0.thenote, '')
        self.assertTrue(note0._thenote_set('test'))
        self.assertEqual(note0.thenote, 'test')

    def _test_note_container(self):
        notes = NoteContainer(PARENT, 'Notes', fxns=nm._fxns)
        self.assertEqual(notes._type, 'Note')
        self.assertEqual(notes.prefix, 'Note')
        self.assertFalse(notes._rename_)
        self.assertEqual(notes._content_name, 'notes')
        self.assertFalse(notes._NoteContainer__off)
        # new
        self.assertEqual(notes.name_next_seq(), 0)
        self.assertEqual(notes.name_next(), 'Note0')
        nlist = []
        for i in range(0, 4):
            thenote = 'note test #' + str(i)
            n = notes.new(thenote)
            self.assertIsInstance(n, Note)
            self.assertEqual(n.thenote, thenote)
            nlist.append(n.thenote)
        notes.off = True
        self.assertIsNone(notes.new('test'))
        # copy
        c = notes.copy()
        self.assertIsInstance(c, NoteContainer)
        self.assertTrue(notes._equal(c, alert=True))
        notes._rename_ = True  # test copy_extra()
        notes.prefix = 'test'  # test copy_extra()
        c = notes.copy()
        self.assertTrue(notes._equal(c, alert=True))
        # thenotes
        self.assertEqual(nlist, notes.thenotes())
        self.assertEqual(nlist, c.thenotes())
        # duplicate
        with self.assertRaises(RuntimeError):
            notes.duplicate()

    def _test_dimension(self):
        notes = NoteContainer(PARENT, 'Notes', fxns=nm._fxns)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = Dimension(PARENT, 'ydim0', fxns=nm._fxns, dim=b)
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = Dimension(PARENT, 'ydim0', fxns=nm._fxns, notes=b)
        y0 = Dimension(PARENT, 'ydim0', fxns=nm._fxns, dim=YDIM0, notes=notes)
        x0 = XDimension(PARENT, 'xdim0', fxns=nm._fxns, dim=XDIM0, notes=notes)
        y1 = Dimension(PARENT, 'ydim1', fxns=nm._fxns, dim=YDIM1, notes=notes)
        x1 = XDimension(PARENT, 'xdim1', fxns=nm._fxns, dim=XDIM1, notes=notes)
        self.assertTrue(y0._rename_)
        self.assertEqual(y0._note_container, notes)
        self.assertEqual(y0._offset, YDIM0['offset'])
        self.assertEqual(y0._label, YDIM0['label'])
        self.assertEqual(y0._units, YDIM0['units'])
        self.assertIsNone(y0._master)
        self.assertEqual(y0._content_name, 'dimension')
        self.assertTrue(x0._rename_)
        self.assertEqual(x0._note_container, notes)
        self.assertEqual(x0._offset, XDIM0['offset'])
        self.assertEqual(x0._start, XDIM0['start'])
        self.assertEqual(x0._delta, XDIM0['delta'])
        self.assertEqual(x0._label, XDIM0['label'])
        self.assertEqual(x0._units, XDIM0['units'])
        self.assertIsNone(x0._master)
        self.assertIsNone(x0._xdata)
        self.assertEqual(x0._content_name, 'xdimension')
        # parameters
        p = ['name', 'rename', 'date', 'modified', 'offset', 'label', 'units',
             'master']
        self.assertEqual(y0._param_list, p)
        self.assertTrue(y0._param_test())
        p = ['name', 'rename', 'date', 'modified', 'offset', 'label', 'units',
             'master', 'start', 'delta', 'xdata']
        self.assertEqual(x0._param_list, p)
        self.assertTrue(x0._param_test())
        # note_new
        note = 'test123'
        n = y0._note_new(note)
        self.assertIsInstance(n, Note)
        self.assertEqual(n.thenote, note)
        n = x0._note_new(note)
        self.assertEqual(n.thenote, note)
        # master
        self.assertIsNone(y1._master)
        self.assertIsNone(x1._master)
        self.assertTrue(y1._master_set(None))  # ok
        self.assertTrue(x1._master_set(None))  # ok
        self.assertFalse(y1._master_lock)
        self.assertFalse(x1._master_lock)
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
            self.assertTrue(y1._master_set(y1))  # self cannot be master
        with self.assertRaises(ValueError):
            self.assertTrue(x1._master_set(x1))  # self cannot be master
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
        self.assertFalse(y1._dim_set(YDIM0))  # master on
        self.assertTrue(y1._offset_set(3.14))  # offset free from master
        self.assertFalse(y1._label_set('test'))
        self.assertFalse(y1._units_set('test'))
        self.assertFalse(x1._dim_set(XDIM0))  # master on
        self.assertTrue(x1._offset_set(3.14))  # offset free from master
        self.assertFalse(x1._label_set('test'))
        self.assertFalse(x1._units_set('test'))
        self.assertFalse(x1._start_set(0))
        self.assertFalse(x1._delta_set(1))
        xdata = Data(PARENT, 'xdata', fxns=nm._fxns, xdim=XDIMx, ydim=YDIMx)
        self.assertFalse(x1._xdata_set(xdata))  # master on
        with self.assertRaises(RuntimeError):
            y0._master_set(y1)  # y1 has master
        with self.assertRaises(RuntimeError):
            x0._master_set(x1)  # x1 has master
        # dim
        for b in BADTYPES + [y1]:
            if isinstance(b, dict):
                continue
            with self.assertRaises(TypeError):
                y0._dim_set(b)
        for b in BADTYPES + [x1]:
            if isinstance(b, dict):
                continue
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
        # offset
        for b in BADTYPES + [y1]:
            if isinstance(b, int) or isinstance(b, float):
                continue  # ok
            with self.assertRaises(TypeError):
                y0._offset_set(b)
        for b in BADTYPES + [y1]:
            if isinstance(b, int) or isinstance(b, float):
                continue  # ok
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
        self.assertTrue(y1._offset_set(3.14))  # offset free from master
        self.assertEqual(y1._offset, 3.14)  # offset free from master
        self.assertTrue(x1._offset_set(3.14))  # offset free from master
        self.assertEqual(x1._offset, 3.14)  # offset free from master
        # label
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
        self.assertFalse(y1._label_set('test'))  # master is on
        self.assertFalse(x1._label_set('test'))  # master is on
        # units
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
        self.assertFalse(y1._units_set('test'))  # master is on
        self.assertFalse(x1._units_set('test'))  # master is on
        # start
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
        self.assertFalse(x1._start_set(0))  # master is on
        # delta
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
        self.assertFalse(x1._delta_set(0))  # master is on
        # xdata
        for b in BADTYPES + [x1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x0._xdata_set(b)
        self.assertTrue(x0._xdata_set(xdata))
        self.assertEqual(x0._xdata, xdata)
        self.assertTrue(x0._offset_set(3.14))  # offset free from xdata
        self.assertFalse(x0._start_set(0))  # xdata is on
        self.assertFalse(x0._delta_set(1))  # xdata is on
        self.assertFalse(x0._label_set('test'))  # xdata is on
        self.assertFalse(x0._units_set('test'))  # xdata is on
        self.assertTrue(x0._xdata_set(None))
        self.assertEqual(x0._xdata, None)
        self.assertFalse(x1._xdata_set(xdata))  # master is on
        # copy
        c = y0.copy()
        self.assertIsInstance(c, Dimension)
        self.assertTrue(y0._equal(c, alert=True))
        c = x0.copy()
        self.assertIsInstance(c, XDimension)
        self.assertTrue(x0._equal(c, alert=True))
        y0._rename_ = False  # test copy_extra()
        c = y0.copy()
        self.assertTrue(y0._equal(c, alert=True))
        x0._rename_ = False  # test copy_extra()
        c = x0.copy()
        self.assertTrue(x0._equal(c, alert=True))

    def _test_data(self):
        name0 = 'RecordA0'
        name1 = 'RecordA1'
        nparray0 = np.full([4], 3.14, dtype=np.float64, order='C')
        nparray1 = np.full([5], 6.28, dtype=np.float64, order='C')
        nparrayx = np.full([6], 12.56, dtype=np.float64, order='C')
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, name0, fxns=nm._fxns, np_array=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, name0, fxns=nm._fxns, xdim=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, name0, fxns=nm._fxns, ydim=b)
        for b in BADTYPES:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(PARENT, name0, fxns=nm._fxns, dataseries=b)
        d0 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray0, xdim=XDIM0,
                  ydim=YDIM0)
        d1 = Data(PARENT, name1, fxns=nm._fxns, np_array=nparray1, xdim=XDIM1,
                  ydim=YDIM1)
        xdata = Data(PARENT, 'xdata', fxns=nm._fxns, np_array=nparrayx,
                     xdim=XDIMx, ydim=YDIMx)
        self.assertTrue(np.array_equal(d0._Data__np_array, nparray0))
        self.assertTrue(d0._rename_)
        self.assertEqual(d0._content_name, 'data')
        # parameters
        p = ['name', 'rename', 'date', 'modified', 'xdim', 'ydim',
             'dataseries']
        self.assertEqual(d0._param_list, p)
        self.assertTrue(d0._param_test())
        # content
        c = d0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['data', 'notes'])
        self.assertEqual(c['data'], d0.name)
        self.assertEqual(c['notes'], d0.note.names)
        # equal
        for b in BADTYPES:
            self.assertFalse(d0._equal(b, alert=True))
        self.assertFalse(d0._equal(d1, alert=True))
        self.assertTrue(d0._equal(d0, alert=True))
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray0,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(d0._equal(d00, alert=True))
        nparray00 = np.full([4], 3.14, dtype=np.float64, order='F')
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray00,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(d0._equal(d00, alert=True))
        nparray0[2] = 5
        self.assertFalse(d0._equal(d00, alert=True))
        nparray0[2] = 0
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=None,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5, 2], 3.14, dtype=np.float64, order='C')
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray00,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5], 3.14, dtype=np.int32, order='C')
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray00,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5], 3.14, dtype=np.float64, order='F')
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray00,
                   xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(d0._equal(d00, alert=True))
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray0,
                   xdim=XDIM1, ydim=YDIM0)
        self.assertFalse(d0._equal(d00, alert=True))
        d00 = Data(PARENT, name0, fxns=nm._fxns, np_array=nparray0,
                   xdim=XDIM0, ydim=YDIM1)
        self.assertFalse(d0._equal(d00, alert=True))
        # copy
        c = d0.copy()
        self.assertIsInstance(c, Data)
        self.assertTrue(d0._equal(c, alert=True))
        d0._rename_ = False  # test copy_extra()
        c = d0.copy()
        self.assertTrue(d0._equal(c, alert=True))
        # np_array
        for b in BADTYPES:
            if b is None:
                continue
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        self.assertTrue(d0._np_array_set(None))
        self.assertIsNone(d0.np_array)
        self.assertTrue(d0._np_array_set(nparray1))
        self.assertIsInstance(d0.np_array, np.ndarray)
        # np_array_make
        self.assertTrue(d0.np_array_make((10, 2)))
        self.assertTrue(d0.np_array_make_random_normal(10, mean=3, stdv=1))
        # dataseries
        # ds = DataSeries(PARENT, 'Record', fxns=nm._fxns)
        # add_dataseries()
        # remove_dataseries()

    def _test_data_container(self):
        c0 = DataContainer(PARENT, 'Data', fxns=nm._fxns)
        c1 = DataContainer(PARENT, 'Data', fxns=nm._fxns)
        self.assertEqual(c0._type, 'Data')
        self.assertEqual(c0.prefix, nmp.DATA_PREFIX)
        self.assertTrue(c0._rename_)
        self.assertEqual(c0._content_name, 'data')
        # new
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                c0.new(name=b)
        nlist = ['RecordA0', 'WaveA0', 'Xdata']
        for n in nlist:
            self.assertIsInstance(c0.new(name=n, xdim=XDIM0), Data)
            self.assertIsInstance(c1.new(name=n, xdim=XDIM0), Data)
        o = c0.getitem(index=0)
        o.name = 'WaveA0'
        return
        # copy
        c = c0.copy()
        self.assertIsInstance(c, DataContainer)
        self.assertTrue(c._equal(c0, alert=True))
        for i in range(0, c.count):
            o0 = c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertTrue(o0._equal(oc, alert=True))
        c0.prefix = 'Test'  # test copy_extra()
        c0._rename_ = False  # test copy_extra()
        c = c0.copy()
        self.assertTrue(c._equal(c0, alert=True))
        c0._rename_ = True
        c0.prefix = nmp.DATA_PREFIX
        # equal
        self.assertTrue(c0._equal(c0, alert=True))
        self.assertTrue(c0._equal(c1, alert=True))
        self.assertIsInstance(c1._select_set('RecordA0'), Data)
        self.assertFalse(c0._equal(c1, alert=True))
        # kill TODO, test if Data is removed from dataseries and esets

    def _test_channel(self):
        c0 = Channel(PARENT, 'A', fxns=nm._fxns, xdim=XDIM0, ydim=YDIM0)
        # ydim argument tested by Dimension
        # xdim argument tested by XDimension
        self.assertFalse(c0._rename_)
        self.assertEqual(c0._content_name, 'channel')
        for k in XDIM0.keys():
            self.assertEqual(c0.x.dim[k], XDIM0[k])
        for k in YDIM0.keys():
            self.assertEqual(c0.y.dim[k], YDIM0[k])
        # parameters
        p = ['name', 'rename', 'date', 'modified', 'xdim', 'ydim']
        self.assertEqual(c0._param_list, p)
        self.assertTrue(c0._param_test())
        # copy
        c = c0.copy()
        self.assertTrue(c0._equal(c, alert=True))
        self.assertTrue(c0.x._equal(c.x, alert=True))
        self.assertTrue(c0.y._equal(c.y, alert=True))
        c0._rename_ = True  # test copy_extra()
        c = c0.copy()
        self.assertTrue(c0._equal(c, alert=True))
        c0._rename_ = False
        # equal
        c00 = Channel(PARENT, 'A', fxns=nm._fxns, xdim=XDIM0, ydim=YDIM0)
        self.assertTrue(c0._equal(c00, alert=True))
        self.assertTrue(c0.x._equal(c00.x, alert=True))
        self.assertTrue(c0.y._equal(c00.y, alert=True))

    def _test_channel_container(self):
        c0 = ChannelContainer(PARENT, 'channels', fxns=nm._fxns)
        self.assertEqual(c0._type, 'Channel')
        self.assertEqual(c0.prefix, '')
        self.assertFalse(c0._rename_)
        self.assertEqual(c0._content_name, 'channels')
        # name
        self.assertEqual(c0.name_next_seq(), 0)
        self.assertEqual(c0.name_next(), 'A')
        c0._rename_ = True
        c0.prefix = 'Test'
        self.assertEqual(c0.name_next(), 'A')  # prefix not used
        c0.prefix = ''
        c0._rename_ = False
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
        self.assertTrue(c0._equal(c, alert=True))
        c0._rename_ = True # test copy_extra()
        c0.prefix = 'Test' # test copy_extra()
        c = c0.copy()
        self.assertTrue(c0._equal(c, alert=True))
        c0.prefix = ''
        c0._rename_ = False
        # equal
        c00 = ChannelContainer(PARENT, 'channels', fxns=nm._fxns)
        c00.new(xdim=XDIM0, ydim=YDIM0)
        c00.new(xdim=XDIM0, ydim=YDIM1)
        self.assertTrue(c0._equal(c00, alert=True))
        c00 = ChannelContainer(PARENT, 'channels', fxns=nm._fxns)
        c00.new(xdim=XDIM0, ydim=YDIM0)
        c00.new(xdim=XDIM0, ydim=YDIM0)
        self.assertFalse(c0._equal(c00, alert=True))
        # duplicate
        with self.assertRaises(RuntimeError):
            c0.duplicate()

    def _test_epochset(self):
        s0 = EpochSet(PARENT, 'All', fxns=nm._fxns)
        s1 = EpochSet(PARENT, 'Set1', fxns=nm._fxns)
        s2 = EpochSet(PARENT, 'Set2', fxns=nm._fxns)
        sx = EpochSet(PARENT, 'SetX', fxns=nm._fxns)
        self.assertTrue(s0._rename_)
        self.assertEqual(s0.eq_list, [])
        self.assertTrue(s0.eq_lock)
        self.assertIsInstance(s0.theset, set)
        # bad_names
        self.assertEqual(s0._bad_names, ['select', 'default'])
        # parameters
        p = ['name', 'rename', 'date', 'modified', 'eq_list', 'eq_lock']
        self.assertEqual(s0._param_list, p)
        self.assertTrue(s0._param_test())
        # copy
        
        # equal

    def _test_project(self):
        name0 = 'Project0'
        name1 = 'Project1'
        """
        p0 = Project(PARENT, name0, fxns=nm._fxns)
        p1 = Project(PARENT, name1, fxns=nm._fxns)
        f = p0.folder.new()
        ds = f.dataseries.new('Record')
        ds.make(channels=1, epochs=3, shape=5, dim=dim)
        f = p0.folder.new()
        ds = f.dataseries.new('Wave')
        ds.make(channels=2, epochs=3, shape=5, dim=dim)
        # p1._copy(p0)
        # self.assertTrue(p1._equal(p0, alert=True))
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
        self.assertEqual(nmu.channel_char([0, 1, 2]), '')
        # channel_num
        self.assertEqual(nmu.channel_num('A'), 0)
        self.assertEqual(nmu.channel_num('a'), 0)
        self.assertEqual(nmu.channel_num('b'), 1)
        self.assertEqual(nmu.channel_num('K'), 10)
        self.assertEqual(nmu.channel_num(''), -1)
        self.assertEqual(nmu.channel_num('AA'), -1)
        self.assertEqual(nmu.channel_num(None), -1)
        self.assertEqual(nmu.channel_num(['A', 'B', 'C']), -1)
        # channel_char_exists
        self.assertTrue(nmu.channel_char_exists('testA1', 'A'))
        self.assertTrue(nmu.channel_char_exists('testa111', 'A'))
        self.assertTrue(nmu.channel_char_exists('testA', 'A'))
        self.assertTrue(nmu.channel_char_exists('A', 'A'))
        self.assertFalse(nmu.channel_char_exists('testA111', 'B'))
        self.assertFalse(nmu.channel_char_exists('A', 'B'))
        self.assertFalse(nmu.channel_char_exists('taste', 'A'))
        # type_error
        e = test_type_error(self, self, self, 'NMObject')
        e2 = "bad a: expected NMObject, but got __main__.Test"
        self.assertEqual(e, e2)
        # print(e)
        a = True
        e = nmu.type_error(a, 'NMObject')
        e2 = "bad a: expected NMObject, but got bool"
        self.assertEqual(e, e2)
        # print(e)
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
