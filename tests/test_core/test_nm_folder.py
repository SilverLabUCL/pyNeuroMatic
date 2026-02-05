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
        self.folder0.data.sets.define_or(DSETS_NLIST0[2], DSETS_NLIST0[0], DSETS_NLIST0[1])

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
        self.folder1.data.sets.define_or(DSETS_NLIST1[2], DSETS_NLIST1[0], DSETS_NLIST1[1])

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

    def test02_notes(self):
        # Test notes property
        self.assertTrue(isinstance(self.folder0.notes, list))
        self.assertEqual(len(self.folder0.notes), 0)

        # Test note setter and note_add
        self.folder0.note = "added TTX"
        self.folder0.note_add("added AP5")
        self.assertEqual(len(self.folder0.notes), 2)
        self.assertEqual(self.folder0.notes[0].get("note"), "added TTX")
        self.assertEqual(self.folder0.notes[1].get("note"), "added AP5")

        # Test note getter (returns last note text)
        self.assertEqual(self.folder0.note, "added AP5")

        # Test notes have timestamps
        self.assertIn("date", self.folder0.notes[0])
        self.assertIn("date", self.folder0.notes[1])

        # Test notes_clear
        self.folder0.notes_clear()
        self.assertEqual(len(self.folder0.notes), 0)
        self.assertEqual(self.folder0.note, "")

        # Test notes_ok validator
        self.assertTrue(NMFolder.notes_ok([{"note": "hey", "date": "111"}]))
        self.assertTrue(NMFolder.notes_ok([{"date": "111", "note": "hey"}]))
        self.assertFalse(NMFolder.notes_ok([{"n": "hey", "date": "111"}]))
        self.assertFalse(NMFolder.notes_ok([{"note": "hey", "d": "111"}]))
        self.assertFalse(NMFolder.notes_ok([{"note": "hey", "date": None}]))
        self.assertTrue(NMFolder.notes_ok([{"note": "hey", "date": "None"}]))
        self.assertFalse(NMFolder.notes_ok([{"note": "hey"}]))
        self.assertFalse(NMFolder.notes_ok([{"date": "111"}]))
        self.assertFalse(
            NMFolder.notes_ok([{"note": "hey", "date": "111", "more": "1"}])
        )

        # Test notes equality
        self.folder0.note_add("test note")
        c = self.folder0.copy()
        self.assertTrue(c == self.folder0)
        c.note_add("another note")
        self.assertFalse(c == self.folder0)

    def xtest03_copy(self):
        pass

    def xtest04_content(self):
        pass

    def xtest05_data(self):
        pass

    def xtest06_folder_container(self):
        pass
