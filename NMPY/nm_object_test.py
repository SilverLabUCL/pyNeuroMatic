#!/usr/bin/env python[3]
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import unittest
import time
import math

from nm_manager import NMManager
from nm_object import NMObject
import nm_preferences as nmp

NM = NMManager(new_project=False, quiet=True)
BADTYPES = [None, True, 1, 3.14, [], (), {}, set(), 'test', NM]
# BADTYPES: all types, use continue to ignore OK types
BADNAME = 'b&dn@me!'
BADNAMES = nmp.BAD_NAMES + [BADNAME]
ALERT = True


class NMObjectTest(unittest.TestCase):

    def setUp(self):
        self.nm = NMManager(new_project=False, quiet=True)
        self.n0 = 'object0'
        self.n1 = 'object1'
        self.o0 = NMObject(parent=self.nm, name=self.n0)
        self.o1 = NMObject(parent=self.nm, name=self.n1)

    # def tearDown(self):
    #    pass

    def test00_init(self):
        # arg: parent, not testing since it can be any object
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                NMObject(parent=self.nm, name=b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                NMObject(parent=self.nm, name=b)
        self.assertEqual(self.o0._parent, self.nm)
        self.assertEqual(self.o0.name, self.n0)
        self.assertEqual(self.o0._NMObject__rename_fxnref, self.o0._name_set)

    def test01_eq_ne(self):
        for b in BADTYPES:
            self.assertFalse(self.o0 == b)
        self.assertTrue(self.o0 is self.o0)
        self.assertFalse(self.o0 is self.o1)
        self.assertTrue(self.o0 == self.o0)
        self.assertFalse(self.o0 == self.o1)
        self.assertFalse(self.o0 != self.o0)
        self.assertTrue(self.o0 != self.o1)
        o0 = NMObject(parent=self.nm, name=self.n0)
        o1 = NMObject(parent=self.nm, name=self.n0)
        self.assertTrue(o0 == o1)
        self.assertFalse(o0 != o1)
        o0 = NMObject2(parent=self.nm, name=self.n0)
        self.assertFalse(o0 == o1)
        o1 = NMObject2(parent=self.nm, name=self.n0)
        self.assertTrue(o0 == o1)
        o0.myvalue = 1
        o1.myvalue = 2
        self.assertFalse(o0 == o1)
        o0.myvalue = float('nan')
        o1.myvalue = float('nan')
        self.assertFalse(o0 == o1)  # NAN != NAN
        o0.myvalue = 2
        o1.myvalue = 2
        o0.note = 'my note'
        o1.note = 'my note'
        self.assertFalse(self.o0 == self.o1)  # different time stamps

    def test02_copy(self):
        c = self.o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(self.o0 == c)
        self.assertEqual(self.o0._parent, c._parent)
        self.assertEqual(self.o0.name, c.name)
        p0 = self.o0.parameters
        p = c.parameters
        self.assertNotEqual(p0.get('created'), p.get('created'))
        self.assertEqual(p0.get('modified'), p.get('modified'))
        self.assertEqual(c._NMObject__rename_fxnref, c._name_set)
        fr0 = self.o0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        self.assertEqual(fr0, frc)  # both refs point to _name_set()

    def test03_parameters(self):
        plist = ['name', 'created', 'modified', 'copy of']
        self.assertEqual(self.o0.parameter_list, plist)

    def test04_content(self):
        self.assertEqual(self.o0.content, {'nmobject': self.o0.name})
        self.assertEqual(self.o0.content_tree, {'nmobject': self.o0.name})

    def test05_treepath(self):
        self.assertEqual(self.o0._tp, self.o0.name)
        self.assertEqual(self.o0._tp_check(0), '')
        self.assertEqual(self.o0._tp_check(None), '')
        self.assertEqual(self.o0._tp_check(''), '')
        self.assertEqual(self.o0._tp_check('self'), self.o0.name)
        self.assertEqual(self.o0._tp_check('nm.test'), 'nm.test')
        self.assertEqual(self.o0.treepath(), self.o0.name)
        self.assertEqual(self.o0.treepath_list(), [self.o0.name])

    def test06_notes(self):
        self.assertTrue(self.o0.notes_on)
        self.o0.note = 'added TTX'
        self.assertTrue(self.o0._note_add('added AP5'))
        self.o0.notes_on = None
        self.assertFalse(self.o0.notes_on)
        self.o0.notes_on = True
        self.assertTrue(self.o0.notes_on)
        self.o0.notes_on = False
        self.assertFalse(self.o0.notes_on)
        self.assertFalse(self.o0._note_add('added NBQX'))
        self.assertTrue(isinstance(self.o0.notes, list))
        self.assertEqual(len(self.o0.notes), 2)
        self.assertEqual(self.o0.notes[0].get('note'), 'added TTX')
        self.assertEqual(self.o0.notes[1].get('note'), 'added AP5')
        if self.o0.notes_clear():
            self.assertEqual(len(self.o0.notes), 0)
        else:
            self.assertEqual(len(self.o0.notes), 2)
        self.assertTrue(NMObject.notes_ok([{'note': 'hey', 'date': '111'}]))
        self.assertTrue(NMObject.notes_ok([{'date': '111', 'note': 'hey'}]))
        self.assertFalse(NMObject.notes_ok([{'n': 'hey', 'date': '111'}]))
        self.assertFalse(NMObject.notes_ok([{'note': 'hey', 'd': '111'}]))
        self.assertFalse(NMObject.notes_ok([{'note': 'hey', 'date': None}]))
        self.assertTrue(NMObject.notes_ok([{'note': 'hey', 'date': 'None'}]))
        self.assertFalse(NMObject.notes_ok([{'note': 'hey'}]))
        self.assertFalse(NMObject.notes_ok([{'date': '111'}]))
        self.assertFalse(NMObject.notes_ok([{'note': 'hey', 'date': '111',
                                             'more': '1'}]))

    def test07_name_ok(self):
        # arg: name
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            self.assertFalse(self.o0.name_ok(b))
        for b in BADNAMES:
            self.assertFalse(self.o0.name_ok(b))
        self.assertEqual(self.o0._bad_names, nmp.BAD_NAMES)
        for b in self.o0._bad_names:
            self.assertFalse(self.o0.name_ok(b))
        for n in [self.n0, self.n1]:
            self.assertTrue(self.o0.name_ok(n))

    def test08_name_set(self):
        # arg: name_notused
        # arg: newname
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                self.o0._name_set('notused', b)
        for b in BADNAMES:
            with self.assertRaises(ValueError):
                self.o0._name_set('notused', b)
        for n in ['test', self.n0]:
            self.assertTrue(self.o0._name_set('notused', n))
            self.assertEqual(n, self.o0.name)

    def test09_rename_fxnref_set(self):
        # arg: rename_fxnref
        for b in BADTYPES:
            with self.assertRaises(TypeError):
                self.o0._rename_fxnref_set(b)
        self.o0.name = 'test1'  # calls _name_set()
        self.assertEqual(self.o0.name, 'test1')
        self.assertTrue(self.o0._rename_fxnref_set(self.rename_dummy))
        self.o0.name = 'test2'
        self.assertEqual(self.o0.name, 'test1')  # name of o0 does not change

    def rename_dummy(self, oldname, newname, quiet=nmp.QUIET):
        # dummy function to test NMObject._rename_fxnref_set()
        print('test rename: ' + oldname + ' -> ' + newname)
        return False

    def test10_manager(self):
        self.assertEqual(self.o0._manager, self.nm)

    def test11_modified(self):
        m1 = self.o0.parameters.get('modified')
        self.o0._modified()
        m2 = self.o0.parameters.get('modified')
        self.assertNotEqual(m1, m2)

    def test12_error(self):
        # alert(), error(), history()
        # wrappers for nmu.history()
        # args: obj, type_expected, tp, quiet, frame
        dum_arg = {}
        e1 = self.o0._type_error('dum_arg', 'list')
        e2 = ('ERROR: object0: bad dum_arg: expected list but got dict')
        self.assertEqual(e1, e2)
        dum_str = 'test'
        e1 = self.o0._value_error('dum_str')
        e2 = ("ERROR: object0: bad dum_str: 'test'")
        self.assertEqual(e1, e2)

    def test13_quiet(self):
        # arg: quiet
        self.nm.configs.quiet = False
        self.assertFalse(self.o0._quiet(False))
        self.assertTrue(self.o0._quiet(True))
        self.nm.configs.quiet = True  # Manager quiet overrides when True
        self.assertTrue(self.o0._quiet(False))
        self.assertTrue(self.o0._quiet(True))
        self.nm.configs.quiet = False

    def test_save(self):
        # TODO
        pass


class NMObject2(NMObject):

    def __init__(self, parent, name):
        super().__init__(parent=parent, name=name)
        self.myvalue = 1

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if self.myvalue != other.myvalue:
            # if math.isnan(self.myvalue) and math.isnan(other.myvalue):
            #    return True
            return False
        return True


if __name__ == '__main__':
    unittest.main()
