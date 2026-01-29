#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:41:40 2023

@author: jason
"""
import unittest

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_project import NMProject
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)
FNAME0 = "folder0"
FNAME1 = "F1"
DNLIST0 = ["data" + str(i) for i in range(8)]
DNLIST1 = ["record" + str(i) for i in range(11)]
DSETS_NLIST0 = ["set" + str(i) for i in range(3)]
DSETS_NLIST1 = ["S" + str(i) for i in range(3)]


class NMFolderTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.folder0 = NMFolder(parent=NM, name=FNAME0)

        self.dolist0 = []
        for n in DNLIST0:
            d = self.folder0.data.new(n)
            self.dolist0.append(d)
        self.folder0.data.selected_name = DNLIST0[-1]

        n = len(DNLIST0)
        slist = [DNLIST0[i] for i in range(0, n, 2)]
        self.folder0.data.sets.add(DSETS_NLIST0[0], slist)
        slist = [DNLIST0[i] for i in range(1, n, 2)]
        self.folder0.data.sets.add(DSETS_NLIST0[1], slist)
        slist = [DSETS_NLIST0[0], "|", DSETS_NLIST0[1]]
        self.folder0.data.sets.add(DSETS_NLIST0[2], slist)

        self.folder1 = NMFolder(parent=NM, name=FNAME1)

        self.dolist1 = []
        for n in DNLIST1:
            d = self.folder1.data.new(n)
            self.dolist1.append(d)
        self.folder1.data.selected_name = DNLIST1[-1]

        n = len(DNLIST1)
        slist = [DNLIST1[i] for i in range(0, n - 3, 1)]
        self.folder1.data.sets.add(DSETS_NLIST1[0], slist)
        slist = [DNLIST1[i] for i in range(3, n, 1)]
        self.folder1.data.sets.add(DSETS_NLIST1[1], slist)
        slist = [DSETS_NLIST1[0], "|", DSETS_NLIST1[1]]
        self.folder1.data.sets.add(DSETS_NLIST1[2], slist)

        # TODO: dataseries

    def test00_init(self):
        # args: parent, name, copy (see NMObject)

        p = NMProject()
        with self.assertRaises(TypeError):
            NMFolder(copy=p)

        data = self.folder0.data
        self.assertTrue(isinstance(data, NMDataContainer))
        self.assertEqual(len(data), len(DNLIST0))
        self.assertEqual(list(data.keys()), DNLIST0)
        self.assertEqual(data.selected_name, DNLIST0[-1])

        self.assertEqual(len(data.sets), len(DSETS_NLIST0))
        self.assertEqual(data.sets.get(DSETS_NLIST0[2]), self.dolist0)
        self.assertEqual(data.sets.get(DSETS_NLIST0[2], get_keys=True), DNLIST0)

        data = self.folder1.data
        self.assertTrue(isinstance(data, NMDataContainer))
        self.assertEqual(len(data), len(DNLIST1))
        self.assertEqual(list(data.keys()), DNLIST1)
        self.assertEqual(data.selected_name, DNLIST1[-1])

        self.assertEqual(len(data.sets), len(DSETS_NLIST1))
        self.assertEqual(data.sets.get(DSETS_NLIST1[2]), self.dolist1)
        self.assertEqual(data.sets.get(DSETS_NLIST1[2], get_keys=True), DNLIST1)

    def test01_eq(self):
        # args: other

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.folder0 == b)

        self.assertTrue(self.folder0 is self.folder0)
        self.assertFalse(self.folder0 is self.folder1)
        self.assertFalse(self.folder0 == self.folder1)
        self.assertTrue(self.folder0 != self.folder1)

        c = self.folder0.copy()
        self.assertFalse(c is self.folder0)
        self.assertTrue(c == self.folder0)

        c.data.sets.remove(DSETS_NLIST0[0], DNLIST0[0])
        self.assertFalse(c == self.folder0)
        c.data.sets.add(DSETS_NLIST0[0], DNLIST0[0])
        self.assertTrue(c == self.folder0)

    def xtest02_copy(self):
        pass

    def xtest03_content(self):
        pass

    def xtest04_data(self):
        pass

    def xtest05_folder_container(self):
        pass
