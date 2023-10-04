#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 07:56:04 2023

@author: jason
"""

import unittest

from nm_folder import NMFolder, NMFolderContainer
from nm_manager import NMManager
from nm_project import NMProject, NMProjectContainer
import nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)
PNAME0 = 'project0'
PNAME1 = 'project1'
PSETS_NLIST = ['set' + str(i) for i in range(3)]
FNLIST0 = ['folder' + str(i) for i in range(6)]
FNLIST1 = ['efolder' + str(i) for i in range(7)]
FSETS_NLIST0 = ['set' + str(i) for i in range(3)]
FSETS_NLIST1 = ['S' + str(i) for i in range(3)]


class NMProjectTest(unittest.TestCase):

    def setUp(self):  # executed before each test

        self.project0 = NMProject(parent=NM, name=PNAME0)

        self.folist0 = []
        for n in FNLIST0:
            f = self.project0.folders.new(n)
            self.folist0.append(f)
        self.project0.folders.select_key = FNLIST0[-1]

        n = len(FNLIST0)
        slist = [FNLIST0[i] for i in range(0, n, 2)]
        self.project0.folders.sets.add(FSETS_NLIST0[0], slist)
        slist = [FNLIST0[i] for i in range(1, n, 2)]
        self.project0.folders.sets.add(FSETS_NLIST0[1], slist)
        slist = [FSETS_NLIST0[0], '|', FSETS_NLIST0[1]]
        self.project0.folders.sets.add(FSETS_NLIST0[2], slist)

        self.project1 = NMProject(parent=NM, name=PNAME1)

        self.folist1 = []
        for n in FNLIST1:
            f = self.project1.folders.new(n)
            self.folist1.append(f)
        self.project1.folders.select_key = FNLIST1[-1]

        n = len(FNLIST1)
        slist = [FNLIST1[i] for i in range(0, n-3, 1)]
        self.project1.folders.sets.add(FSETS_NLIST1[0], slist)
        slist = [FNLIST1[i] for i in range(3, n, 1)]
        self.project1.folders.sets.add(FSETS_NLIST1[1], slist)
        slist = [FSETS_NLIST1[0], '|', FSETS_NLIST1[1]]
        self.project1.folders.sets.add(FSETS_NLIST1[2], slist)

        self.projectcontainer = NMProjectContainer(parent=NM)
        self.projectcontainer.update([self.project0, self.project1])
        self.projectcontainer.select_key = PNAME0

        self.projectcontainer.sets.add(PSETS_NLIST[0], PNAME0)
        self.projectcontainer.sets.add(PSETS_NLIST[1], PNAME1)
        self.projectcontainer.sets.add(PSETS_NLIST[2],
                                       [PSETS_NLIST[0], '|', PSETS_NLIST[1]])

    def test00_init(self):
        # args: parent, name, copy (see NMObject)

        f = NMFolder()
        with self.assertRaises(TypeError):
            NMProject(copy=f)

        folders = self.project0.folders
        self.assertTrue(isinstance(folders, NMFolderContainer))
        self.assertEqual(len(folders), len(FNLIST0))
        self.assertEqual(list(folders.keys()), FNLIST0)
        self.assertEqual(folders.select_key, FNLIST0[-1])

        self.assertEqual(len(folders.sets), len(FSETS_NLIST0))
        self.assertEqual(folders.sets.get(FSETS_NLIST0[2]), self.folist0)
        self.assertEqual(folders.sets.get(FSETS_NLIST0[2], get_keys=True),
                         FNLIST0)

        folders = self.project1.folders
        self.assertTrue(isinstance(folders, NMFolderContainer))
        self.assertEqual(len(folders), len(FNLIST1))
        self.assertEqual(list(folders.keys()), FNLIST1)
        self.assertEqual(folders.select_key, FNLIST1[-1])

        self.assertEqual(len(folders.sets), len(FSETS_NLIST1))
        self.assertEqual(folders.sets.get(FSETS_NLIST1[2]), self.folist1)
        self.assertEqual(folders.sets.get(FSETS_NLIST1[2], get_keys=True),
                         FNLIST1)

    def test01_eq(self):
        # args: other

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.project0 == b)

        self.assertFalse(self.project0 is self.project1)
        self.assertFalse(self.project0 == self.project1)
        self.assertTrue(self.project0 != self.project1)

        c = self.project0.copy()
        self.assertFalse(c is self.project0)
        self.assertTrue(c == self.project0)
        c.folders.sets.remove(FSETS_NLIST0[0], FNLIST0[0])
        self.assertFalse(c == self.project0)
        c.folders.sets.add(FSETS_NLIST0[0], FNLIST0[0])
        self.assertTrue(c == self.project0)

        # recreate project0
        project0 = NMProject(parent=NM, name=PNAME0)
        self.assertFalse(self.project0 == project0)

        folist0 = []
        for n in FNLIST0:
            f = project0.folders.new(n)
            folist0.append(f)
        project0.folders.select_key = FNLIST0[-1]

        self.assertFalse(self.project0 == project0)

        self.project0.folders._eq_list.remove('sets')
        self.assertTrue(self.project0 == project0)
        self.project0.folders._eq_list.append('sets')
        self.assertFalse(self.project0 == project0)

        for s in self.project0.folders.sets.keys():
            olist = self.project0.folders.sets.get(s, get_equation=True,
                                                   get_keys=True)
            project0.folders.sets.add(s, olist)

        self.assertFalse(self.project0 == project0)

        project0.folders.popitem(confirm_answer='y')
        self.assertFalse(self.project0 == project0)

    def test02_copy(self):
        c = self.project0.copy()
        self.assertFalse(c is self.project0)
        self.assertTrue(type(c) == type(self.project0))
        self.assertTrue(c == self.project0)  # see test01_eq()

    def test03_content(self):
        c = {'nmproject': PNAME0, 'NMFolderContainer': FNLIST0}
        self.assertEqual(self.project0.content, c)

    def test04_folders(self):
        self.assertTrue(isinstance(self.project0.folders, NMFolderContainer))
        with self.assertRaises(AttributeError):
            self.project0.folders = None

    def test05_project_container(self):
        self.assertEqual(len(self.projectcontainer), 2)
        p = self.projectcontainer.new('project2')
        self.assertEqual(len(self.projectcontainer), 3)
        self.assertTrue(isinstance(p, NMProject))
        self.assertEqual(p.name, 'project2')
        self.assertTrue(p in self.projectcontainer)
        self.assertTrue(self.project0 in self.projectcontainer)
        self.assertTrue(self.project1 in self.projectcontainer)
        self.assertTrue(PNAME0.upper() in self.projectcontainer)
        self.assertTrue(PNAME1.upper() in self.projectcontainer)
        for p in self.projectcontainer.values():
            self.assertEqual(p._parent, NM)
        self.assertEqual(self.projectcontainer.content_type(),
                         NMProject.__name__)

        self.projectcontainer.sets.add('set0', 'project2')
        self.projectcontainer.sets.add('set1', 'project2')

        c = self.projectcontainer.copy()
        self.assertTrue(c == self.projectcontainer)
        self.assertEqual(len(c), 3)
        self.assertFalse(p in c)
        self.assertFalse(self.project0 in c)
        self.assertFalse(self.project1 in c)
        self.assertTrue(p.name.upper() in c)
        self.assertTrue(self.project0.name.upper() in c)
        self.assertTrue(self.project1.name.upper() in c)
        self.assertEqual(c.select_key, PNAME0)
        p = c.get(PNAME0)
        self.assertFalse(p is self.project0)
        self.assertTrue(p == self.project0)
        p = c.get(PNAME1)
        self.assertFalse(p is self.project1)
        self.assertTrue(p == self.project1)

        self.assertEqual(len(c.sets), 3)
        self.assertTrue('set0' in c.sets)
        self.assertTrue('set1' in c.sets)
        self.assertTrue('set2' in c.sets)
        self.assertFalse('set3' in c.sets)
        self.assertTrue(c.sets.contains('set0', PNAME0))
        self.assertFalse(c.sets.contains('set0', PNAME1))
        self.assertTrue(c.sets.contains('set2', PNAME1))
        self.assertTrue(c.sets.isequation('set2'))

        self.assertEqual(len(p.folders), len(FNLIST1))
        for f in FNLIST1:
            self.assertTrue(f in p.folders)
        for s in FSETS_NLIST1:
            self.assertTrue(s in p.folders.sets)
        self.assertEqual(len(p.folders.sets), len(FSETS_NLIST1))
        for s in FSETS_NLIST1:
            self.assertTrue(s in p.folders.sets)
