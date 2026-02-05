#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 07:56:04 2023

@author: jason
"""

import unittest

from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_project import NMProject
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)
PNAME0 = "project0"
PNAME1 = "project1"
PSETS_NLIST = ["set" + str(i) for i in range(3)]
FNLIST0 = ["folder" + str(i) for i in range(6)]
FNLIST1 = ["efolder" + str(i) for i in range(7)]
FSETS_NLIST0 = ["set" + str(i) for i in range(3)]
FSETS_NLIST1 = ["S" + str(i) for i in range(3)]


class NMProjectTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.project0 = NMProject(parent=NM, name=PNAME0)

        self.folist0 = []
        for n in FNLIST0:
            f = self.project0.folders.new(n)
            self.folist0.append(f)
        self.project0.folders.selected_name = FNLIST0[-1]

        n = len(FNLIST0)
        slist = [FNLIST0[i] for i in range(0, n, 2)]
        self.project0.folders.sets.add(FSETS_NLIST0[0], slist)
        slist = [FNLIST0[i] for i in range(1, n, 2)]
        self.project0.folders.sets.add(FSETS_NLIST0[1], slist)
        self.project0.folders.sets.define_or(FSETS_NLIST0[2], FSETS_NLIST0[0], FSETS_NLIST0[1])

        self.project1 = NMProject(parent=NM, name=PNAME1)

        self.folist1 = []
        for n in FNLIST1:
            f = self.project1.folders.new(n)
            self.folist1.append(f)
        self.project1.folders.selected_name = FNLIST1[-1]

        n = len(FNLIST1)
        slist = [FNLIST1[i] for i in range(0, n - 3, 1)]
        self.project1.folders.sets.add(FSETS_NLIST1[0], slist)
        slist = [FNLIST1[i] for i in range(3, n, 1)]
        self.project1.folders.sets.add(FSETS_NLIST1[1], slist)
        self.project1.folders.sets.define_or(FSETS_NLIST1[2], FSETS_NLIST1[0], FSETS_NLIST1[1])

    def test00_init(self):
        # args: parent, name, copy (see NMObject)

        f = NMFolder()
        with self.assertRaises(TypeError):
            NMProject(copy=f)

        folders = self.project0.folders
        self.assertTrue(isinstance(folders, NMFolderContainer))
        self.assertEqual(len(folders), len(FNLIST0))
        self.assertEqual(list(folders.keys()), FNLIST0)
        self.assertEqual(folders.selected_name, FNLIST0[-1])

        self.assertEqual(len(folders.sets), len(FSETS_NLIST0))
        self.assertEqual(folders.sets.get(FSETS_NLIST0[2]), self.folist0)
        self.assertEqual(folders.sets.get(FSETS_NLIST0[2], get_keys=True), FNLIST0)

        folders = self.project1.folders
        self.assertTrue(isinstance(folders, NMFolderContainer))
        self.assertEqual(len(folders), len(FNLIST1))
        self.assertEqual(list(folders.keys()), FNLIST1)
        self.assertEqual(folders.selected_name, FNLIST1[-1])

        self.assertEqual(len(folders.sets), len(FSETS_NLIST1))
        self.assertEqual(folders.sets.get(FSETS_NLIST1[2]), self.folist1)
        self.assertEqual(folders.sets.get(FSETS_NLIST1[2], get_keys=True), FNLIST1)

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
        project0.folders.selected_name = FNLIST0[-1]

        self.assertFalse(self.project0 == project0)

        # self.project0.folders._eq_list.remove("sets")
        # self.assertTrue(self.project0 == project0)
        # self.project0.folders._eq_list.append("sets")
        # self.assertFalse(self.project0 == project0)

        for s in self.project0.folders.sets.keys():
            olist = self.project0.folders.sets.get(s, get_equation=True, get_keys=True)
            project0.folders.sets.add(s, olist)

        self.assertTrue(self.project0 == project0)

        project0.folders.popitem()
        self.assertFalse(self.project0 == project0)

    def test02_copy(self):
        c = self.project0.copy()
        self.assertFalse(c is self.project0)
        self.assertTrue(type(c) == type(self.project0))
        self.assertTrue(c == self.project0)  # see test01_eq()

    def test03_content(self):
        c = {"nmproject": PNAME0, "NMFolderContainer": FNLIST0}
        # print(c)
        # print(self.project0.content)
        # self.assertEqual(self.project0.content, c)

    def test04_folders(self):
        self.assertTrue(isinstance(self.project0.folders, NMFolderContainer))
        with self.assertRaises(AttributeError):
            self.project0.folders = None
