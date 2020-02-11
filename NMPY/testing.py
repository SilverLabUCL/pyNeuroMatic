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
import nm_dimension as nmd
from nm_manager import Manager
from nm_note import Note
from nm_note import NoteContainer
from nm_project import Project
import nm_utilities as nmu

nm = Manager(new_project=False, quiet=True)
BADTYPES = [True, 1, 3.14, float('nan'), float('inf'), [1, 2], (1, 2),
            {'one': 1, 'two': 2}, set([1, 2]), None, nm]
BADNAMES = ['select', 'default', 'all']  # may need updating
BADNAME = 'b&dn@me!'


def test_type_error(a, b, c, got):
    return nmu.type_error(c, got)


class Test(unittest.TestCase):

    def test_nmobject(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        name0 = 'object0'
        name1 = 'object1'
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                o0 = NMObject(parent, b)
        for b in BADNAMES + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0 = NMObject(parent, b)
        o0 = NMObject(parent, name0, fxns=nm._fxns, rename=True)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertEqual(o0.name, name0)
        self.assertTrue(o0._rename)
        self.assertEqual(o1.name, name1)
        self.assertFalse(o1._rename)
        # name_ok
        for b in BADTYPES:
            self.assertFalse(o1.name_ok(b))
        for b in BADNAMES + [BADNAME]:
            self.assertFalse(o1.name_ok(b))
        for b in BADNAMES:
            self.assertTrue(o1.name_ok(b, ok=BADNAMES))
        self.assertIsInstance(o1.name_ok(''), bool)
        good = [name0, name1, '']
        for g in good:
            self.assertTrue(o1.name_ok(g))
        # bad_names
        self.assertEqual(BADNAMES, o1._bad_names)  # check if list changes
        # parameters
        self.assertTrue(o0._param_test())
        self.assertTrue(o1._param_test())
        # name_set
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                o0._name_set(b)
        for b in BADNAMES + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0._name_set(b)
        self.assertIsInstance(o0._name_set('test'), bool)
        good = ['test', name0]
        for g in good:
            self.assertTrue(o0._name_set(g))
            self.assertEqual(g, o0.name)
        with self.assertRaises(RuntimeError):
            o1._name_set(name0)  # rename = False
        # content
        self.assertIsInstance(o0.content, dict)
        self.assertEqual(o0.content, {'nmobject': o0.name})
        self.assertEqual(o0.content_tree, {'nmobject': o0.name})
        # treepath
        self.assertIsInstance(o0.treepath(), str)
        self.assertIsInstance(o0.treepath_list(), list)
        self.assertEqual(o0.treepath(), o0.name)
        self.assertEqual(o0.treepath_list(), [o0.name])
        # equal
        for b in BADTYPES + ['test']:
            self.assertFalse(o1._equal(b, alert=True))
        self.assertIsInstance(o1._equal(o1), bool)
        self.assertFalse(o1._equal(o0, alert=True))
        self.assertTrue(o1._equal(o1, alert=True))
        o0 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertTrue(o1._equal(o0, alert=True))
        # copy
        c = o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(o0._equal(c, alert=True))
        self.assertEqual(o0._parent, c._parent)
        self.assertEqual(o0._fxns, c._fxns)
        self.assertEqual(o0.name, c.name)
        self.assertEqual(o0._rename, c._rename)
        self.assertNotEqual(o0._NMObject__date, c._NMObject__date)
        self.assertNotEqual(o0._NMObject__modified, c._NMObject__modified)
        # save TODO

    def test_container(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        name0 = 'container0'
        name1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        n = [p0 + str(i) for i in range(0, 6)]
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b)
        for b in [BADNAME, '']:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b)
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b)
        for b in BADNAMES + [BADNAME]:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b)
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix='',
                       rename=True, duplicate=True)
        self.assertEqual(c0.prefix, '')
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix=p0,
                       rename=True, duplicate=True)
        c1 = Container(parent, name1, fxns=nm._fxns, type_=type_, prefix=p1,
                       rename=False, duplicate=False)
        self.assertEqual(c0.name, name0)
        self.assertEqual(c0._type, type_)
        self.assertTrue(c0._rename)
        self.assertTrue(c0._duplicate)
        self.assertEqual(c0.prefix, p0)
        self.assertIsNone(c0.select)
        self.assertEqual(c1.name, name1)
        self.assertEqual(c1._type, type_)
        self.assertFalse(c1._rename)
        self.assertFalse(c1._duplicate)
        self.assertEqual(c1.prefix, p1)
        self.assertIsNone(c1.select)
        # parameters
        self.assertTrue(c0._param_test())
        self.assertTrue(c1._param_test())
        # prefix
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0._prefix_set(b)
        for b in BADNAMES + [BADNAME]:
            with self.assertRaises(ValueError):
                c0._prefix_set(b)
        self.assertIsInstance(c0._prefix_set(''), bool)
        self.assertTrue(c0._prefix_set(p1))
        self.assertEqual(c0.prefix, p1)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c1.prefix, p1)
        c0.prefix = p0  # reset
        # name_next
        self.assertIsInstance(c0.name_next(), str)
        self.assertIsInstance(c0.name_next_seq(), int)
        self.assertEqual(c0.name_next(), n[0])
        self.assertEqual(c0.name_next_seq(), 0)
        # new
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.new(b)
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'default':
                continue  # 'default' is ok
            with self.assertRaises(ValueError):
                c0.new(b)
        o = c0.new('default')  # ok
        self.assertIsInstance(o, NMObject)
        self.assertEqual(o.name, n[0])
        self.assertEqual(c0.select, o)
        self.assertEqual(c0.select.name, n[0])
        with self.assertRaises(RuntimeError):
            c0.new(n[0])  # already exists
        self.assertEqual(c0.name_next(), n[1])
        self.assertEqual(c0.name_next_seq(), 1)
        o = c0.new(select=False)
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
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        self.assertIsInstance(c1.new(), NMObject)
        # names
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[4], n[5]])
        # content
        c = c0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['nmobjects'])
        self.assertEqual(c['nmobjects'], c0.names)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), ['nmobjects'])
        self.assertEqual(c['nmobjects'], c0.names)
        # index
        for b in BADTYPES:
            self.assertEqual(c0.index(b), -1)
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            self.assertEqual(c0.index(b), -1)
        self.assertIsInstance(c0.index(n[0]), int)
        self.assertEqual(c0.index(n[0]), 0)
        self.assertEqual(c0.index(n[1]), 1)
        self.assertEqual(c0.index(n[2]), 2)
        self.assertEqual(c0.index(n[3]), -1)  # does not exist
        self.assertEqual(c0.index(n[4]), 3)
        self.assertEqual(c0.index(n[5]), 4)
        self.assertEqual(c0.index('select'), 4)
        # exists
        for b in BADTYPES:
            self.assertFalse(c0.exists(b))
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            self.assertFalse(c0.exists(b))
        self.assertIsInstance(c0.exists(''), bool)
        for i in range(0, 6):
            if i == 3:
                self.assertFalse(c0.exists(n[i]))
            else:
                self.assertTrue(c0.exists(n[i]))
        self.assertTrue(c0.exists('select'))
        # getitem
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.getitem(b)
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.getitem(b)
        self.assertIsNone(c0.getitem(''))
        self.assertEqual(c0.getitem('select'), c0.select)
        for i in range(0, 6):
            o = c0.getitem(n[i])
            if i == 3:
                self.assertIsNone(o)
            else:
                self.assertIsInstance(o, NMObject)
                self.assertEqual(o.name, n[i])
        for i in range(0, 5):
            o = c0.getitem(index=i)
            self.assertIsInstance(o, NMObject)
            if i <= 2:
                self.assertEqual(o.name, n[i])
            else:
                self.assertEqual(o.name, n[i+1])
        self.assertIsNone(c0.getitem(index=-1))
        for b in [3.14, float('inf'), float('nan'), [1, 2, 3], {}]:
            with self.assertRaises(TypeError):
                c0.getitem(index=b)
        for b in [5, 10, 100]:
            with self.assertRaises(IndexError):
                c0.getitem(index=b)
        # getitems
        with self.assertRaises(TypeError):
            olist0 = c0.getitems(names=[n[0], n[1], n[2], None])
        with self.assertRaises(ValueError):
            olist0 = c0.getitems(names=[n[0], n[1], n[2], BADNAME])
        with self.assertRaises(ValueError):
            olist0 = c0.getitems(names=[n[0], n[1], n[2], 'default'])
        with self.assertRaises(IndexError):
            olist1 = c0.getitems(indexes=[0, 1, 2, 3, 50])
        olist0 = c0.getitems(names=[n[0], n[1], n[2], n[4]])
        olist1 = c0.getitems(indexes=[0, 1, 2, 3])
        self.assertIsInstance(olist0, list)
        self.assertIsInstance(olist1, list)
        self.assertEqual(olist0, olist1)
        olist0 = c0.getitems(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()
        self.assertEqual(len(olist0), c0.count-1)
        # select
        sname = c0.select.name
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0._select_set(b)
        for b in BADNAMES + [BADNAME]:
            with self.assertRaises(ValueError):
                c0._select_set(b)
        self.assertIsNone(c0._select_set(n[3]))
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        self.assertIsInstance(c0._select_set(n[0]), NMObject)
        self.assertEqual(c0.select.name, n[0])
        # rename
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.rename(b, n[3])
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(b, n[3])
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.rename(n[4], b)
        for b in BADNAMES + [BADNAME, '']:
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
        self.assertEqual(c0.rename(n[4], 'default'), nnext)
        self.assertEqual(c0.rename(nnext, n[3]), n[3])
        o = c0.getitem(index=i)
        self.assertEqual(o.name, n[3])
        self.assertEqual(c0.rename(n[5], n[4]), n[4])
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[3], n[4]])
        # duplicate
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.duplicate(b, 'default')
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(b, 'default')
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.duplicate(n[0], b)
        for b in BADNAMES + [BADNAME, '']:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(n[0], b)
        with self.assertRaises(RuntimeError):
            c0.duplicate(n[0], n[1])  # already exists
        with self.assertRaises(RuntimeError):
            c1.duplicate('select', 'default')  # duplicate = False
        self.assertIsNone(c0.duplicate('', 'default'))
        o = c0.getitem(n[0])
        nnext = c0.name_next()
        oc = c0.duplicate(n[0], 'default')
        self.assertIsInstance(oc, NMObject)
        self.assertEqual(oc.name, nnext)
        self.assertFalse(o._equal(oc, alert=True))  # names are different
        with self.assertRaises(RuntimeError):
            oc.name = n[0]  # must use Container.rename()
        oc._NMObject__name = n[0]
        self.assertTrue(o._equal(oc, alert=True))
        oc._NMObject__name = nnext
        # equal
        for b in BADTYPES + ['test']:
            self.assertFalse(c0._equal(b, alert=True))
        self.assertIsInstance(c0._equal(c0), bool)
        self.assertFalse(c0._equal(c1, alert=True))
        self.assertFalse(c0._equal(c1, alert=True))
        # copy
        c = c0.copy()
        self.assertIsInstance(c, Container)
        self.assertTrue(c0._equal(c, alert=True))
        self.assertEqual(c0._parent, c._parent)
        self.assertEqual(c0._fxns, c._fxns)
        self.assertEqual(c0.name, c.name)
        self.assertEqual(c0._rename, c._rename)
        self.assertNotEqual(c0._NMObject__date, c._NMObject__date)
        self.assertNotEqual(c0._NMObject__modified, c._NMObject__modified)
        self.assertEqual(c0._type, c._type)
        self.assertEqual(c0.prefix, c.prefix)
        self.assertEqual(c0._duplicate, c._duplicate)
        self.assertNotEqual(c0.select, c.select)  # refs not equal
        self.assertEqual(c0.select.name, c.select.name)  # but names equal
        for i in range(0, c0.count):
            o0 = c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertIsInstance(oc, NMObject)
            self.assertTrue(o0._equal(oc, alert=True))
        # Kill
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                c0.kill(name=b, confirm=False)
        for b in BADNAMES + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.kill(name=b, confirm=False)
        klist = c0.kill(name='', confirm=False)
        self.assertIsInstance(klist, list)
        self.assertEqual(klist, [])
        s = c0.select
        klist = c0.kill(name='select', confirm=False)
        self.assertEqual(klist, [s])
        self.assertIsNone(c0.getitem(s.name))
        o = c0.getitem(n[0])
        klist = c0.kill(name=n[0], confirm=False)
        self.assertEqual(klist, [o])
        self.assertIsNone(c0.getitem(n[0]))
        names = c1.names
        klist = c1.kill(all_=True, confirm=False)
        self.assertEqual(len(klist), len(names))
        self.assertEqual(c1.count, 0)
        self.assertIsNone(c1.select)

    def test_note(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        thenote = 'note!'
        note = Note(parent, 'Note0', fxns=nm._fxns, thenote=thenote)
        self.assertFalse(note._rename)
        self.assertEqual(note.thenote, thenote)
        # parameters
        self.assertTrue(note._param_test())
        # content
        c = note.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['note'])
        self.assertEqual(c['note'], note.name)
        # equal
        note1 = Note(parent, 'Note0', fxns=nm._fxns, thenote=thenote)
        self.assertTrue(note._equal(note1, alert=True))
        note1 = Note(parent, 'Note0', fxns=nm._fxns, thenote='test')
        self.assertFalse(note._equal(note1, alert=True))
        # copy
        note1 = note.copy()
        self.assertIsInstance(note1, Note)
        self.assertTrue(note._equal(note1, alert=True))
        self.assertEqual(note._Note__thenote, note1._Note__thenote)
        # thenote
        r = note._thenote_set(123)
        self.assertIsInstance(r, bool)
        self.assertEqual(note.thenote, '123')
        note._thenote_set(None)
        self.assertEqual(note.thenote, '')
        note._thenote_set(thenote + thenote)
        self.assertEqual(note.thenote, thenote + thenote)

    def test_note_container(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        notes = NoteContainer(parent, 'Notes', fxns=nm._fxns)
        self.assertEqual(notes._type, 'Note')
        self.assertEqual(notes.prefix, 'Note')
        self.assertFalse(notes._rename)
        self.assertFalse(notes._duplicate)
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
        # content
        c = notes.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['notes'])
        self.assertEqual(c['notes'], ['Note' + str(i) for i in range(0, 4)])
        # copy
        c = notes.copy()
        self.assertIsInstance(c, NoteContainer)
        self.assertTrue(notes._equal(c, alert=True))
        # thenotes
        self.assertEqual(nlist, notes.thenotes())
        self.assertEqual(nlist, c.thenotes())

    def test_dimension(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        notes = NoteContainer(parent, 'Notes', fxns=nm._fxns)
        offset = 5.5
        label = 'testlabel'
        units = 'testunits'
        ydim0 = {'label': 'Vmem', 'units': 'mV', 'offset': 0}
        xdim0 = {'offset': 0, 'start': 10, 'delta': 0.01, 'label': 'time',
                 'units': 'ms'}
        ydim1 = {'label': 'Imem', 'units': 'pA', 'offset': 0}
        xdim1 = {'offset': 0, 'start': -10, 'delta': 0.2, 'label': 't',
                 'units': 'seconds'}
        xy = {'label': 'time interval', 'units': 'usec', 'offset': 0}
        xx = {'offset': 0, 'start': 0, 'delta': 1, 'label': 'sample',
              'units': '#'}
        for b in BADTYPES + ['test']:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = nmd.Dimension(parent, 'ydim0', fxns=nm._fxns, dim=b)
        for b in BADTYPES + ['test']:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0 = nmd.Dimension(parent, 'ydim0', fxns=nm._fxns, notes=b)
        y0 = nmd.Dimension(parent, 'ydim0', fxns=nm._fxns, dim=ydim0,
                           notes=notes)
        x0 = nmd.XDimension(parent, 'xdim0', fxns=nm._fxns, dim=xdim0,
                            notes=notes)
        y1 = nmd.Dimension(parent, 'ydim1', fxns=nm._fxns, dim=ydim1,
                           notes=notes)
        x1 = nmd.XDimension(parent, 'xdim1', fxns=nm._fxns, dim=xdim1,
                            notes=notes)
        self.assertEqual(y0._note_container, notes)
        self.assertEqual(y0._offset, ydim0['offset'])
        self.assertEqual(y0._label, ydim0['label'])
        self.assertEqual(y0._units, ydim0['units'])
        self.assertIsNone(y0._master)
        self.assertEqual(x0._note_container, notes)
        self.assertEqual(x0._offset, xdim0['offset'])
        self.assertEqual(x0._start, xdim0['start'])
        self.assertEqual(x0._delta, xdim0['delta'])
        self.assertEqual(x0._label, xdim0['label'])
        self.assertEqual(x0._units, xdim0['units'])
        self.assertIsNone(x0._master)
        self.assertIsNone(x0._xdata)
        # parameters
        self.assertTrue(y0._param_test())
        self.assertTrue(x0._param_test())
        # content
        c = y0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['dimension'])
        self.assertEqual(c['dimension'], y0.name)
        c = x0.content
        self.assertEqual(list(c.keys()), ['xdimension'])
        self.assertEqual(c['xdimension'], x0.name)
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
        for b in BADTYPES + ['test', x0]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y1._master_set(b)
        for b in BADTYPES + ['test', y0]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x1._master_set(b)
        with self.assertRaises(ValueError):
            self.assertTrue(y1._master_set(y1))  # no self master
        with self.assertRaises(ValueError):
            self.assertTrue(x1._master_set(x1))  # no self master
        r = y1._master_set(y0)
        self.assertIsInstance(r, bool)
        self.assertIsInstance(y1._master, nmd.Dimension)
        self.assertTrue(r)
        self.assertEqual(y1._master, y0)
        self.assertTrue(y1._master_lock)
        r = x1._master_set(x0)
        self.assertIsInstance(r, bool)
        self.assertIsInstance(x1._master, nmd.XDimension)
        self.assertTrue(r)
        self.assertEqual(x1._master, x0)
        self.assertTrue(x1._master_lock)
        dim = y1.dim
        self.assertIsInstance(dim, dict)
        for k in ydim0.keys():  # y0 is master
            self.assertEqual(ydim0[k], dim[k])
        self.assertEqual(dim['master'], y0)
        dim = x1.dim
        self.assertIsInstance(dim, dict)
        for k in xdim0.keys():  # x0 is master
            self.assertEqual(xdim0[k], dim[k])
        self.assertEqual(dim['master'], x0)
        self.assertFalse(y1._dim_set(ydim0))  # master on
        self.assertTrue(y1._offset_set(offset))  # offset free from master
        self.assertFalse(y1._label_set(label))
        self.assertFalse(y1._units_set(units))
        self.assertFalse(x1._dim_set(xdim0))  # master on
        self.assertTrue(x1._offset_set(offset))  # offset free from master
        self.assertFalse(x1._label_set(label))
        self.assertFalse(x1._units_set(units))
        self.assertFalse(x1._start_set(0))
        self.assertFalse(x1._delta_set(1))
        xdata = Data(parent, 'xdata', fxns=nm._fxns, xdim=xx, ydim=xy)
        self.assertFalse(x1._xdata_set(xdata))
        with self.assertRaises(RuntimeError):
            y0._master_set(y1)  # y1 has master
        with self.assertRaises(RuntimeError):
            x0._master_set(x1)  # x1 has master
        # dim
        for b in BADTYPES + ['test', y1]:
            if isinstance(b, dict):
                continue
            with self.assertRaises(TypeError):
                y0._dim_set(b)
        for b in BADTYPES + ['test', x1]:
            if isinstance(b, dict):
                continue
            with self.assertRaises(TypeError):
                x0._dim_set(b)
        bad_dim = {'label': 'Vmem', 'units': 'mV', 'test': 0}
        with self.assertRaises(KeyError):
            y0._dim_set(bad_dim)
        with self.assertRaises(KeyError):
            y0._dim_set(xdim0)
        with self.assertRaises(KeyError):
            x0._dim_set(bad_dim)
        r = y0._dim_set(ydim0)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertTrue(x0._dim_set(ydim0))  # ok
        self.assertTrue(x0._dim_set(xdim0))
        # offset
        for b in BADTYPES + ['test', y1]:
            if nmu.number_ok(b, no_inf=False, no_nan=False):
                continue
            with self.assertRaises(TypeError):
                y0._offset_set(b)
        for b in BADTYPES + ['test', y1]:
            if nmu.number_ok(b, no_inf=False, no_nan=False):
                continue
            with self.assertRaises(TypeError):
                x0._offset_set(b)
        badvalues = [float('nan'), float('inf')]
        for b in badvalues:
            with self.assertRaises(ValueError):
                y0._offset_set(b)
        for b in badvalues:
            with self.assertRaises(ValueError):
                x0._offset_set(b)
        r = y0._offset_set(offset)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertEqual(y0._offset, offset)
        self.assertTrue(x0._offset_set(offset))
        self.assertEqual(x0._offset, offset)
        self.assertTrue(y1._offset_set(offset))  # offset free from master
        self.assertEqual(y1._offset, offset)  # offset free from master
        self.assertTrue(x1._offset_set(offset))  # offset free from master
        self.assertEqual(x1._offset, offset)  # offset free from master
        # label
        for b in BADTYPES + [y1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0._label_set(b)
        for b in BADTYPES + [x1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x0._label_set(b)
        r = y0._label_set(None)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertEqual(y0._label, '')
        self.assertTrue(x0._label_set(None))
        self.assertEqual(x0._label, '')
        self.assertTrue(y0._label_set(label))
        self.assertEqual(y0._label, label)
        self.assertTrue(x0._label_set(label))
        self.assertEqual(x0._label, label)
        self.assertFalse(y1._label_set(label))  # master is on
        self.assertFalse(x1._label_set(label))  # master is on
        # units
        for b in BADTYPES + [y1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                y0._units_set(b)
        for b in BADTYPES + [x1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x0._units_set(b)
        r = y0._units_set(None)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertEqual(y0._units, '')
        self.assertTrue(x0._units_set(None))
        self.assertEqual(x0._units, '')
        self.assertTrue(y0._units_set(units))
        self.assertEqual(y0._units, units)
        self.assertTrue(x0._units_set(units))
        self.assertEqual(x0._units, units)
        self.assertFalse(y1._units_set(units))  # master is on
        self.assertFalse(x1._units_set(units))  # master is on
        # start
        for b in BADTYPES + ['test', x1]:
            if nmu.number_ok(b, no_inf=False, no_nan=False):
                continue
            with self.assertRaises(TypeError):
                x0._start_set(b)
        r = x0._start_set(0)
        self.assertIsInstance(r, bool)
        goodvalues = [0, 10, -10.2, float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._start_set(g))
            self.assertEqual(x0._start, g)
        self.assertTrue(x0._start_set(float('nan')))  # nan ok
        self.assertFalse(x1._start_set(0))  # master is on
        # delta
        for b in BADTYPES + ['test', x1]:
            if nmu.number_ok(b, no_inf=False, no_nan=False):
                continue
            with self.assertRaises(TypeError):
                x0._delta_set(b)
        r = x0._delta_set(0)
        self.assertIsInstance(r, bool)
        goodvalues = [0, 10, -10.2, float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._delta_set(g))
            self.assertEqual(x0._delta, g)
        self.assertTrue(x0._delta_set(float('nan')))  # nan ok
        self.assertFalse(x1._delta_set(0))  # master is on
        # xdata
        for b in BADTYPES + ['test', x1]:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                x0._xdata_set(b)
        r = x0._xdata_set(xdata)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertEqual(x0._xdata, xdata)
        self.assertTrue(x0._offset_set(offset))  # offset free from xdata
        self.assertFalse(x0._start_set(0))  # xdata is on
        self.assertFalse(x0._delta_set(1))  # xdata is on
        self.assertFalse(x0._label_set(label))  # xdata is on
        self.assertFalse(x0._units_set(units))  # xdata is on
        self.assertTrue(x0._xdata_set(None))
        self.assertEqual(x0._xdata, None)
        self.assertFalse(x1._xdata_set(xdata))  # master is on
        # copy
        yc = y0.copy()
        self.assertIsInstance(yc, nmd.Dimension)
        self.assertTrue(y0._equal(yc, alert=True))
        xc = x0.copy()
        self.assertIsInstance(xc, nmd.XDimension)
        self.assertTrue(x0._equal(xc, alert=True))

    def test_data(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        name0 = 'RecordA0'
        name1 = 'RecordA1'
        ydim0 = {'units': 'mV', 'offset': 0, 'label': 'Vmem'}
        xdim0 = {'offset': 0, 'start': 10, 'delta': 0.01, 'label': 'time',
                 'units': 'ms'}
        ydim1 = {'label': 'Imem', 'units': 'pA', 'offset': 0}
        xdim1 = {'offset': 0, 'start': -10, 'delta': 0.2, 'label': 't',
                 'units': 'seconds'}
        xy = {'label': 'time interval', 'units': 'usec', 'offset': 0}
        xx = {'offset': 0, 'start': 0, 'delta': 1, 'label': 'sample',
              'units': '#'}
        nparray0 = np.full([5], 3.14, dtype=np.float64, order='C')
        nparray1 = np.full([5], 6.28, dtype=np.float64, order='C')
        nparrayx = np.full([6], 3.14, dtype=np.float64, order='C')
        for b in BADTYPES + ['test']:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, np_array=b)
        for b in BADTYPES + ['test']:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, notes=b)
        for b in BADTYPES + ['test']:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, xdim=b)
        for b in BADTYPES + ['test']:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, ydim=b)
        for b in BADTYPES + ['test']:
            if b is None or isinstance(b, dict):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, dataseries=b)
        d0 = Data(parent, name0, fxns=nm._fxns, np_array=nparray0, xdim=xdim0,
                  ydim=ydim0)
        d1 = Data(parent, name1, fxns=nm._fxns, np_array=nparray1, xdim=xdim1,
                  ydim=ydim1)
        xdata = Data(parent, 'xdata', fxns=nm._fxns, np_array=nparrayx,
                     xdim=xx, ydim=xy)
        self.assertTrue(np.array_equal(d0._Data__np_array, nparray0))
        # parameters
        self.assertTrue(d0._param_test())
        # content
        c = d0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['data', 'notes'])
        self.assertEqual(c['data'], d0.name)
        self.assertEqual(c['notes'], d0.notes.names)
        # equal
        for b in BADTYPES + ['test']:
            self.assertFalse(d0._equal(b, alert=True))
        r = d0._equal(d1, alert=True)
        self.assertIsInstance(r, bool)
        self.assertFalse(r)
        self.assertTrue(d0._equal(d0, alert=True))
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=nparray0)
        self.assertTrue(d0._equal(d00, alert=True))
        nparray00 = np.full([5], 3.14, dtype=np.float64, order='F')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=nparray00)
        self.assertTrue(d0._equal(d00, alert=True))
        nparray0[2] = 5
        self.assertFalse(d0._equal(d00, alert=True))
        nparray0[2] = 0
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=None)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5, 2], 3.14, dtype=np.float64, order='C')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=nparray00)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5], 3.14, dtype=np.int32, order='C')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=nparray00)
        self.assertFalse(d0._equal(d00, alert=True))
        nparray00 = np.full([5], 3.14, dtype=np.float64, order='F')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim0,
                   np_array=nparray00)
        self.assertFalse(d0._equal(d00, alert=True))
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim1, ydim=ydim0,
                   np_array=nparray0)
        self.assertFalse(d0._equal(d00, alert=True))
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, ydim=ydim1,
                   np_array=nparray0)
        self.assertFalse(d0._equal(d00, alert=True))
        # copy
        c = d0.copy()
        self.assertIsInstance(c, Data)
        self.assertTrue(d0._equal(c, alert=True))
        # np_array
        for b in BADTYPES + ['test']:
            if b is None:
                continue
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        r = d0._np_array_set(None)
        self.assertIsInstance(r, bool)
        self.assertTrue(r)
        self.assertIsNone(d0.np_array)
        self.assertTrue(d0._np_array_set(nparray1))
        self.assertIsInstance(d0.np_array, np.ndarray)
        # np_array_make
        self.assertTrue(d0.np_array_make((10, 2)))
        self.assertTrue(d0.np_array_make_random_normal(10, mean=3, stdv=1))
        # dataseries
        # ds = DataSeries(parent, 'Record', fxns=nm._fxns)
        # add_dataseries()
        # remove_dataseries()

    def test_data_container(self):
        on = True
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        for b in BADTYPES + ['test']:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                data = DataContainer(parent, 'Data', fxns=nm._fxns,
                                     dataseries_container=b)
        data = DataContainer(parent, 'Data', fxns=nm._fxns,
                                     dataseries_container=None)
        # new
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                data.new(name=b)
        nlist = []
        # content
        c = data.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), ['data'])

    def test_project(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        name0 = 'Project0'
        name1 = 'Project1'
        noise = [0, 0.1]
        dim = {'xstart': -10, 'xdelta': 0.01,
                'xlabel': 'time', 'xunits': 'ms',
                'ylabel': {'A': 'Vmem', 'B': 'Icmd'},
                'yunits': {'A': 'mV', 'B': 'pA'}}
        """
        p0 = Project(parent, name0, fxns=nm._fxns)
        p1 = Project(parent, name1, fxns=nm._fxns)
        f = p0.folder.new()
        ds = f.dataseries.new('Record')
        ds.make(channels=1, epochs=3, shape=5, dim=dim)
        f = p0.folder.new()
        ds = f.dataseries.new('Wave')
        ds.make(channels=2, epochs=3, shape=5, dim=dim)
        # p1._copy(p0)
        # self.assertTrue(p1._equal(p0, alert=True))
        """

    def test_channel(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
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
        on = False
        if not on:
            return
        nm.configs.quiet = False
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

    def test_type_error(self):
        e = test_type_error(self, self, self, 'NMObject')
        e2 = "bad a: expected NMObject, but got __main__.Test"
        self.assertEqual(e, e2)
        # print(e)
        a = True
        e = nmu.type_error(a, 'NMObject')
        e2 = "bad a: expected NMObject, but got bool"
        self.assertEqual(e, e2)
        # print(e)

    def test_history(self):
        quiet = True
        r = 'nm.Test.test_history: test'
        self.assertEqual(nmu.history('test', quiet=quiet), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.test_history: test'
        self.assertEqual(nmu.history('test', tp=tp, quiet=quiet), r)

    def test_get_treepath(self):
        stack = inspect.stack()
        r = 'nm.Test.run'
        self.assertEqual(nmu.get_treepath(stack), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.run'
        self.assertEqual(nmu.get_treepath(stack, tp=tp), r)

    def test_get_class(self):
        stack = inspect.stack()
        self.assertEqual(nmu.get_class(stack), 'Test')
        self.assertEqual(nmu.get_class(stack, module=True), '__main__.Test')

    def test_get_method(self):
        stack = inspect.stack()
        self.assertEqual(nmu.get_method(stack), 'run')


if __name__ == '__main__':
    unittest.main()
