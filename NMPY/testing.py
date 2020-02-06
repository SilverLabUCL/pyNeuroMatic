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
        badtypes = [True, 1, float('nan'), [], {}, set(), None, self]
        badnames = ['select', 'default', 'all']  # may need updating
        for b in badtypes:
            with self.assertRaises(TypeError):
                o0 = NMObject(parent, b, fxns=nm._fxns, rename=True)
        for b in badnames + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0 = NMObject(parent, b, fxns=nm._fxns, rename=True)
        o0 = NMObject(parent, name0, fxns=nm._fxns, rename=True)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertEqual(o0.name, name0)
        self.assertEqual(o1.name, name1)
        # name_ok
        for b in badtypes:
            self.assertFalse(o1.name_ok(b))
        for b in badnames + [BADNAME]:
            self.assertFalse(o1.name_ok(b))
        for b in badnames:
            self.assertTrue(o1.name_ok(b, ok=badnames))
        good = [name0, name1, '']
        for g in good:
            self.assertTrue(o1.name_ok(g))
        # bad_names
        self.assertEqual(badnames, o1._bad_names)  # check if list changes
        # parameters
        self.assertTrue(o0._param_test())
        self.assertTrue(o1._param_test())
        # name_set
        for b in badtypes:
            with self.assertRaises(TypeError):
                o0._name_set(b)
        for b in badnames + ['', BADNAME]:
            with self.assertRaises(ValueError):
                o0._name_set(b)
        good = ['test', name0]
        for g in good:
            self.assertTrue(o0._name_set(g))
            self.assertEqual(g, o0.name)
        with self.assertRaises(RuntimeError):
            o1._name_set(name0)  # rename = False
        # content
        self.assertEqual(o0.content, {'nmobject': o0.name})
        self.assertEqual(o0.content_tree, {'nmobject': o0.name})
        # treepath
        self.assertEqual(o0.treepath(), o0.name)
        self.assertEqual(o0.treepath_list(), [o0.name])
        # equal
        for b in badtypes + ['test']:
            self.assertFalse(o1._equal(b, alert=True))
        self.assertFalse(o1._equal(o0, alert=True))
        self.assertFalse(o1._equal(o0, ignore_name=True, alert=True))
        o0 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        o1 = NMObject(parent, name1, fxns=nm._fxns, rename=False)
        self.assertTrue(o1._equal(o0, alert=True))
        # copy
        c = o0.copy()
        p0 = o0.parameters
        pc = c.parameters
        self.assertTrue(o0._equal(c, alert=True))
        self.assertEqual(o0._parent, c._parent)
        self.assertEqual(o0._fxns, c._fxns)
        self.assertEqual(p0['name'], pc['name'])
        self.assertEqual(p0['rename'], pc['rename'])
        self.assertNotEqual(p0['date'], pc['date'])  # different
        self.assertNotEqual(p0['modified'], pc['modified'])  # different

    def test_container(self):
        on = True
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        name0 = 'container0'
        name1 = 'container1'
        p0 = 'TestA'
        p1 = 'TestB'
        type_ = 'NMObject'
        badtypes = [True, 1, float('nan'), [], {}, set(), None, self]
        badnames = ['select', 'default', 'all']  # may need updating
        n = [p0 + str(i) for i in range(0, 6)]
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b,
                               prefix=p0, rename=True, duplicate=True)
        for b in [BADNAME, '']:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=b,
                               prefix=p0, rename=True, duplicate=True)
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b, rename=True, duplicate=True)
        for b in badnames + [BADNAME]:
            with self.assertRaises(ValueError):
                c0 = Container(parent, name0, fxns=nm._fxns, type_=type_,
                               prefix=b, rename=True, duplicate=True)
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix='',
                       rename=True, duplicate=True)
        self.assertEqual(c0.prefix, '')
        c0 = Container(parent, name0, fxns=nm._fxns, type_=type_, prefix=p0,
                       rename=True, duplicate=True)
        c1 = Container(parent, name1, fxns=nm._fxns, type_=type_, prefix=p1,
                       rename=False, duplicate=False)
        self.assertEqual(c0.name, name0)
        self.assertEqual(c1.name, name1)
        self.assertEqual(c0.prefix, p0)
        self.assertEqual(c1.prefix, p1)
        # parameters
        self.assertTrue(c0._param_test())
        self.assertTrue(c1._param_test())
        # prefix
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0._prefix_set(b)
        for b in badnames + [BADNAME]:
            with self.assertRaises(ValueError):
                c0._prefix_set(b)
        self.assertTrue(c0._prefix_set(p1))
        self.assertEqual(c0.prefix, p1)
        with self.assertRaises(RuntimeError):
            c1.prefix = p0  # rename = False
        self.assertEqual(c1.prefix, p1)
        c0.prefix = p0  # reset
        # name_next
        self.assertEqual(c0.name_next(), n[0])
        self.assertEqual(c0.name_next_seq(), 0)
        # new
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.new(b)
        for b in badnames + [BADNAME]:
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
        self.assertEqual(list(c.keys()), ['nmobjects'])
        self.assertEqual(c['nmobjects'], c0.names)
        c = c0.content_tree
        self.assertEqual(list(c.keys()), ['nmobjects'])
        self.assertEqual(c['nmobjects'], c0.names)
        # treepath
        self.assertEqual(c0.treepath(), name0)
        self.assertEqual(c0.treepath_list(), [name0])
        # index
        for b in badtypes:
            self.assertEqual(c0.index(b), -1)
        for b in badnames + [BADNAME]:
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
        # exists
        for b in badtypes:
            self.assertFalse(c0.exists(b))
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            self.assertFalse(c0.exists(b))
        for i in range(0, 6):
            if i == 3:
                self.assertFalse(c0.exists(n[i]))
            else:
                self.assertTrue(c0.exists(n[i]))
        self.assertTrue(c0.exists('select'))
        # getitem
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.getitem(b)
        for b in badnames + [BADNAME]:
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
        self.assertEqual(olist0, olist1)
        olist0 = c0.getitems(names='all')
        self.assertEqual(len(olist0), c0.count)
        olist0.pop()
        self.assertEqual(len(olist0), c0.count-1)
        # select
        sname = c0.select.name
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0._select_set(b)
        for b in badnames + [BADNAME]:
            with self.assertRaises(ValueError):
                c0._select_set(b)
        self.assertIsNone(c0._select_set(n[3]))
        if nm.configs.quiet:
            self.assertEqual(c0.select.name, sname)
        self.assertIsInstance(c0._select_set(n[0]), NMObject)
        self.assertEqual(c0.select.name, n[0])
        # rename
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.rename(b, n[3])
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(b, n[3])
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.rename(n[4], b)
        for b in badnames + [BADNAME, '']:
            if b.lower() == 'default':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.rename(n[4], b)
        with self.assertRaises(RuntimeError):
            c0.rename('select', n[0])  # already exists
        with self.assertRaises(RuntimeError):
            c1.rename('select', 'test')  # rename = False
        self.assertEqual(c0.rename('', n[3]), '')
        i = c0.index(n[4])
        nnext = c0.name_next()
        self.assertEqual(c0.rename(n[4], 'default'), nnext)
        self.assertEqual(c0.rename(nnext, n[3]), n[3])
        o = c0.getitem(index=i)
        self.assertEqual(o.name, n[3])
        self.assertEqual(c0.rename(n[5], n[4]), n[4])
        self.assertEqual(c0.names, [n[0], n[1], n[2], n[3], n[4]])
        # duplicate
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.duplicate(b, 'default')
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.duplicate(b, 'default')
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.duplicate(n[0], b)
        for b in badnames + [BADNAME, '']:
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
        self.assertEqual(oc.name, nnext)
        self.assertFalse(o._equal(oc, alert=True))  # names are different
        self.assertTrue(o._equal(oc, ignore_name=True, alert=True))
        # equal
        for b in badtypes + ['test']:
            self.assertFalse(c0._equal(b, alert=True))
        self.assertFalse(c0._equal(c1, alert=True))
        self.assertFalse(c0._equal(c1, ignore_name=True, alert=True))
        # copy
        c = c0.copy()
        self.assertTrue(c0._equal(c, alert=True))
        self.assertEqual(c0._parent, c._parent)
        self.assertEqual(c0._fxns, c._fxns)
        p0 = c0.parameters
        pc = c.parameters
        self.assertEqual(p0['name'], pc['name'])
        self.assertEqual(p0['rename'], pc['rename'])
        self.assertNotEqual(p0['date'], pc['date'])  # different
        self.assertNotEqual(p0['modified'], pc['modified'])  # different
        self.assertEqual(p0['type'], pc['type'])
        self.assertEqual(p0['prefix'], pc['prefix'])
        self.assertEqual(p0['duplicate'], pc['duplicate'])
        self.assertEqual(p0['select'], pc['select'])
        for i in range(0, c0.count):
            o0 = c0.getitem(index=i)
            oc = c.getitem(index=i)
            self.assertTrue(o0._equal(oc, alert=True))
            self.assertEqual(o0._parent, oc._parent)
            self.assertEqual(o0._fxns, oc._fxns)
            p0 = o0.parameters
            pc = oc.parameters
            self.assertEqual(p0['name'], pc['name'])
            self.assertEqual(p0['rename'], pc['rename'])
            self.assertNotEqual(p0['date'], pc['date'])  # different
            self.assertNotEqual(p0['modified'], pc['modified'])  # different
        # Kill
        for b in badtypes:
            with self.assertRaises(TypeError):
                c0.kill(name=b, confirm=False)
        for b in badnames + [BADNAME]:
            if b.lower() == 'select':
                continue  # ok
            with self.assertRaises(ValueError):
                c0.kill(name=b, confirm=False)
        self.assertEqual(c0.kill(name='', confirm=False), [])
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
        thenote = 'note!!'
        note = Note(parent, 'Note0', fxns=nm._fxns, thenote=thenote)
        self.assertEqual(note.thenote, thenote)
        # parameters
        self.assertTrue(note._param_test())
        # content
        self.assertEqual(list(note.content.keys()), ['note'])
        # copy
        c = note.copy()
        self.assertTrue(note._equal(c, alert=True))
        self.assertEqual(c.thenote, thenote)
        # thenote
        note._thenote_set(123)
        self.assertEqual(note.thenote, '123')
        note._thenote_set(None)
        self.assertEqual(note.thenote, '')

    def test_note_container(self):
        on = False
        if not on:
            return
        nm.configs.quiet = False
        parent = self
        notes = NoteContainer(parent, 'Notes', fxns=nm._fxns)
        # new
        self.assertEqual(notes.name_next(), 'Note0')
        nlist = []
        for i in range(0, 4):
            thenote = 'note test #' + str(i)
            n = notes.new(thenote)
            self.assertIsInstance(n, Note)
            self.assertEqual(n.thenote, thenote)
            nlist.append(n.thenote)
        # content
        self.assertEqual(list(notes.content.keys()), ['notes'])
        # copy
        c = notes.copy()
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
        badtypes = [True, 1, 3.1, float('nan'), [], {}, set(), self]
        for b in badtypes + [label]:
            with self.assertRaises(TypeError):
                y0 = nmd.Dimension(parent, 'ydim0', fxns=nm._fxns, dim=ydim0,
                                   notes=b)
        y0 = nmd.Dimension(parent, 'ydim0', fxns=nm._fxns, dim=ydim0,
                           notes=notes)
        x0 = nmd.XDimension(parent, 'xdim0', fxns=nm._fxns, dim=xdim0,
                            notes=notes)
        y1 = nmd.Dimension(parent, 'ydim1', fxns=nm._fxns, dim=ydim1,
                           notes=notes)
        x1 = nmd.XDimension(parent, 'xdim1', fxns=nm._fxns, dim=xdim1,
                            notes=notes)
        xdata = Data(parent, 'xdata', fxns=nm._fxns, xdim=xx, ydim=xy)
        dim = y0.dim
        for k in ydim0.keys():
            self.assertEqual(ydim0[k], dim[k])
        self.assertIsNone(dim['master'])
        dim = x0.dim
        for k in xdim0.keys():
            self.assertEqual(xdim0[k], dim[k])
        self.assertIsNone(dim['xdata'])
        self.assertIsNone(dim['master'])
        # parameters
        self.assertTrue(y0._param_test())
        self.assertTrue(x0._param_test())
        # note_new
        note = 'test123'
        n = y0._note_new(note)
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
        for b in badtypes + [label, x0]:
            with self.assertRaises(TypeError):
                y1._master_set(b)
        for b in badtypes + [label, y0]:
            with self.assertRaises(TypeError):
                x1._master_set(b)
        with self.assertRaises(ValueError):
            self.assertTrue(y1._master_set(y1))
        with self.assertRaises(ValueError):
            self.assertTrue(x1._master_set(x1))
        self.assertTrue(y1._master_set(y0))
        self.assertEqual(y1._master, y0)
        self.assertTrue(x1._master_set(x0))
        self.assertEqual(x1._master, x0)
        dim = y1.dim
        for k in ydim0.keys():  # y0 is master
            self.assertEqual(ydim0[k], dim[k])
        self.assertEqual(dim['master'], y0)
        self.assertEqual(y1.master, y0)
        self.assertTrue(y1._master_lock)
        dim = x1.dim
        for k in xdim0.keys():  # x0 is master
            self.assertEqual(xdim0[k], dim[k])
        self.assertEqual(dim['master'], x0)
        self.assertEqual(x1.master, x0)
        self.assertTrue(x1._master_lock)
        self.assertTrue(y1._dim_set(ydim0))  # ok, but nothing changes
        self.assertTrue(y1._offset_set(offset))  # offset free from master
        self.assertFalse(y1._label_set(label))
        self.assertFalse(y1._units_set(units))
        self.assertTrue(x1._dim_set(xdim0))  # ok, but nothing changes
        self.assertTrue(x1._offset_set(offset))  # offset free from master
        self.assertFalse(x1._label_set(label))
        self.assertFalse(x1._units_set(units))
        self.assertFalse(x1._start_set(0))
        self.assertFalse(x1._delta_set(1))
        self.assertFalse(x1._xdata_set(xdata))
        with self.assertRaises(RuntimeError):
            y0._master_set(y1)
        with self.assertRaises(RuntimeError):
            x0._master_set(x1)
        # dim
        for b in badtypes + [None, label, y1]:
            if isinstance(b, dict):
                continue
            with self.assertRaises(TypeError):
                y0._dim_set(b)
        for b in badtypes + [None, label, x1]:
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
        self.assertTrue(y0._dim_set(ydim0))
        self.assertTrue(x0._dim_set(ydim0))  # ok
        self.assertTrue(x0._dim_set(xdim0))
        # offset
        badtypes2 = [True, [], {}, set(), self, None, label, y1]
        for b in badtypes2:
            with self.assertRaises(TypeError):
                y0._offset_set(b)
        for b in badtypes2:
            with self.assertRaises(TypeError):
                x0._offset_set(b)
        badvalues = [float('nan'), float('inf')]
        for b in badvalues:
            with self.assertRaises(ValueError):
                y0._offset_set(b)
        for b in badvalues:
            with self.assertRaises(ValueError):
                x0._offset_set(b)
        self.assertTrue(y0._offset_set(offset))
        self.assertEqual(y0._offset, offset)
        self.assertTrue(x0._offset_set(offset))
        self.assertEqual(x0._offset, offset)
        self.assertTrue(y1._offset_set(offset))  # offset free from master
        self.assertEqual(y1._offset, offset)  # offset free from master
        self.assertTrue(x1._offset_set(offset))  # offset free from master
        self.assertEqual(x1._offset, offset)  # offset free from master
        # label
        for b in badtypes + [y1]:
            with self.assertRaises(TypeError):
                y0._label_set(b)
        for b in badtypes + [x1]:
            with self.assertRaises(TypeError):
                x0._label_set(b)
        self.assertTrue(y0._label_set(None))
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
        for b in badtypes + [y1]:
            with self.assertRaises(TypeError):
                y0._units_set(b)
        for b in badtypes + [x1]:
            with self.assertRaises(TypeError):
                x0._units_set(b)
        self.assertTrue(y0._units_set(None))
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
        badtypes2 = [True, label, [], {}, set(), self, y0]
        for b in badtypes2:
            with self.assertRaises(TypeError):
                x0._start_set(b)
        goodvalues = [0, 10, -10.2]
        for g in goodvalues:
            self.assertTrue(x0._start_set(g))
            self.assertEqual(x0._start, g)
        goodvalues = [float('nan'), float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._start_set(g))
        self.assertFalse(x1._start_set(0))  # master is on
        # delta
        for b in badtypes2:
            with self.assertRaises(TypeError):
                x0._delta_set(b)
        goodvalues = [0, 10, -10.2]
        for g in goodvalues:
            self.assertTrue(x0._delta_set(g))
            self.assertEqual(x0._delta, g)
        goodvalues = [float('nan'), float('inf')]
        for g in goodvalues:
            self.assertTrue(x0._delta_set(g))
        self.assertFalse(x1._delta_set(0))  # master is on
        # xdata
        for b in badtypes + [label, y0]:
            with self.assertRaises(TypeError):
                x0._xdata_set(b)
        self.assertTrue(x0._xdata_set(xdata))
        self.assertEqual(x0._xdata, xdata)
        self.assertTrue(x0._offset_set(offset))  # offset free from xdata
        self.assertFalse(x0._start_set(0))  # xdata is on
        self.assertFalse(x0._delta_set(1))  # xdata is on
        self.assertFalse(x0._label_set(label))  # xdata is on
        self.assertFalse(x0._units_set(units))  # xdata is on
        self.assertTrue(x0._xdata_set(None))
        self.assertEqual(x0._xdata, None)
        self.assertFalse(x1._xdata_set(xdata))  # master is on
        # equal
        for b in badtypes + ['test', x0, y1]:
            self.assertFalse(y0._equal(b, ignore_name=True, alert=True))
        self.assertTrue(y0._equal(y0, ignore_name=False, alert=True))
        self.assertTrue(y1._master_set(None))
        self.assertEqual(y1._master, None)
        self.assertTrue(y0._dim_set(ydim0))
        self.assertTrue(y1._dim_set(ydim0))
        self.assertTrue(y0._equal(y1, ignore_name=True, alert=True))
        for b in badtypes + ['test', y0, x1]:
            self.assertFalse(x0._equal(b, ignore_name=True, alert=True))
        self.assertTrue(x0._equal(x0, ignore_name=False, alert=True))
        self.assertTrue(x1._master_set(None))
        self.assertEqual(x1._master, None)
        self.assertTrue(x0._dim_set(xdim0))
        self.assertTrue(x1._dim_set(xdim0))
        self.assertTrue(x0._equal(x1, ignore_name=True, alert=True))
        # copy
        yc = y0.copy()
        self.assertTrue(y0._equal(yc))
        xc = x0.copy()
        self.assertTrue(x0._equal(xc))

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
        nparray0 = np.full([5], np.nan, dtype=np.float64, order='C')
        nparray1 = np.full([5], np.nan, dtype=np.float64, order='C')
        nparrayx = np.full([6], np.nan, dtype=np.float64, order='C')
        badtypes = [True, 1, 3.1, float('nan'), 'test', [], {}, set(), self]
        ds = DataSeries(parent, 'Record', fxns=nm._fxns)
        for b in badtypes:
            with self.assertRaises(TypeError):
                d0 = Data(parent, name0, fxns=nm._fxns, np_array=b, xdim=xdim0,
                          ydim=ydim0, dataseries=ds)
        d0 = Data(parent, name0, fxns=nm._fxns, np_array=nparray0, xdim=xdim0,
                  ydim=ydim0, dataseries=ds)
        d1 = Data(parent, name1, fxns=nm._fxns, np_array=nparray1, xdim=xdim1,
                  ydim=ydim1, dataseries=ds)
        xdata = Data(parent, 'xdata', fxns=nm._fxns, np_array=nparrayx,
                     xdim=xx, ydim=xy)
        # parameters
        self.assertTrue(d0._param_test())
        # content
        self.assertEqual(list(d0.content.keys()), ['data', 'notes'])
        # equal
        for b in badtypes:
            self.assertFalse(d0._equal(b, ignore_name=True, alert=True))
        self.assertFalse(d0._equal(d1, ignore_name=True, alert=True))
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, np_array=nparray0,
                   ydim=ydim0, dataseries=ds)
        self.assertTrue(d0._equal(d00, ignore_name=False, alert=True))
        self.assertTrue(d0._equal(d00, ignore_name=False, alert=True))
        nparray00 = np.full([5], np.nan, dtype=np.float64, order='F')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0,
                   np_array=nparray00, ydim=ydim0, dataseries=ds)
        self.assertTrue(d0._equal(d00, ignore_name=False, alert=True))
        nparray0[2] = 5
        self.assertFalse(d0._equal(d00, ignore_name=False, alert=True))
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0, np_array=None,
                   ydim=ydim0, dataseries=ds)
        self.assertFalse(d0._equal(d00, ignore_name=False, alert=True))
        self.assertFalse(d00._equal(d0, ignore_name=False, alert=True))
        nparray00 = np.full([5, 2], np.nan, dtype=np.float64, order='C')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0,
                   np_array=nparray00, ydim=ydim0, dataseries=ds)
        self.assertFalse(d0._equal(d00, ignore_name=False, alert=True))
        nparray00 = np.full([5], 0, dtype=np.int32, order='C')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0,
                   np_array=nparray00, ydim=ydim0, dataseries=ds)
        self.assertFalse(d0._equal(d00, ignore_name=False, alert=True))
        nparray00 = np.full([5], 0, dtype=np.float64, order='F')
        d00 = Data(parent, name0, fxns=nm._fxns, xdim=xdim0,
                   np_array=nparray00, ydim=ydim0, dataseries=ds)
        self.assertFalse(d0._equal(d00, ignore_name=False, alert=True))
        # copy
        d0.notes.new('test')
        cd0 = d0.copy()
        self.assertTrue(d0._equal(cd0, alert=True))
        
        # add_dataseries()
        # remove_dataseries()
        # np_array()
        
        """
        # np_array_set()
        bad = [self, 'test', float('inf'), 1, [], {}]
        for b in bad:
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        nparray = np.full(shape, 0)
        self.assertTrue(d0._np_array_set(nparray))
        # self.assertEqual(nparray, d0.np_array)
        # print(type(d0.np_arrayn))
        # print(isinstance(d0.np_array, np.ndarray))
        # print(d0.np_array)
        # d0.np_array.clip(-1, 1, out=d0.np_array)
        # print(d0.np_array.flags)
        # print(d0.parameters)
        """

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
