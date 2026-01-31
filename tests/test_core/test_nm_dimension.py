#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:43:19 2022

@author: jason
"""
import copy
import unittest

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager

NM = NMManager(quiet=True)

YSNAME0 = "yscale0"
XSNAME0 = "xscale0"
YSNAME1 = "yscale1"
XSNAME1 = "xscale1"

YSCALE0 = {"label": "Vm", "units": "mV"}
XSCALE0 = {"label": "time", "units": "ms", "start": 10, "delta": 0.01}
YSCALE1 = {"label": "Im", "units": "pA"}
XSCALE1 = {"label": "time", "units": "s", "start": -10, "delta": 0.2}


class NMDimensionTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.y0 = NMDimension(parent=NM, name=YSNAME0, scale=YSCALE0)
        self.x0 = NMDimensionX(parent=NM, name=XSNAME0, scale=XSCALE0)
        self.y1 = NMDimension(parent=NM, name=YSNAME1, scale=YSCALE1)
        self.x1 = NMDimensionX(parent=NM, name=XSNAME1, scale=XSCALE1)

        self.y0_copy = copy.deepcopy(self.y0)
        self.x0_copy = copy.deepcopy(self.x0)
        self.y1_copy = copy.deepcopy(self.y1)
        self.x1_copy = copy.deepcopy(self.x1)

    def test00_init(self):
        # args: parent, name, copy (see NMObject)
        # scale

        bad = list(nmu.BADTYPES)
        bad.remove(None)  # ok
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                NMDimension(scale=b)

        self.assertEqual(self.y0.name, YSNAME0)
        self.assertEqual(self.y0.label, YSCALE0["label"])
        self.assertEqual(self.y0.units, YSCALE0["units"])

        self.assertEqual(self.x0.name, XSNAME0)
        self.assertEqual(self.x0.label, XSCALE0["label"])
        self.assertEqual(self.x0.units, XSCALE0["units"])
        self.assertEqual(self.x0.start, XSCALE0["start"])
        self.assertEqual(self.x0.delta, XSCALE0["delta"])

        self.assertEqual(self.y0_copy.name, YSNAME0)
        self.assertEqual(self.y0_copy.label, YSCALE0["label"])
        self.assertEqual(self.y0_copy.units, YSCALE0["units"])

        self.assertEqual(self.x0_copy.name, XSNAME0)
        self.assertEqual(self.x0_copy.label, XSCALE0["label"])
        self.assertEqual(self.x0_copy.units, XSCALE0["units"])
        self.assertEqual(self.x0_copy.start, XSCALE0["start"])
        self.assertEqual(self.x0_copy.delta, XSCALE0["delta"])

    def test01_eq(self):
        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.y0 == b)

        self.assertFalse(self.y0 == self.x0)
        self.assertFalse(self.y0 == self.y1)
        self.assertTrue(self.y0 == self.y0_copy)

        self.assertFalse(self.x0 == self.y0)
        self.assertFalse(self.x0 == self.x1)
        self.assertTrue(self.x0 == self.x0_copy)

        self.x0_copy.name = "test"
        self.assertFalse(self.x0 == self.x0_copy)
        self.x0_copy.name = XSNAME0
        self.assertTrue(self.x0 == self.x0_copy)

        self.x0_copy.label = "test"
        self.assertFalse(self.x0 == self.x0_copy)
        self.x0_copy.label = XSCALE0["label"]
        self.assertTrue(self.x0 == self.x0_copy)

        self.x0_copy.units = "test"
        self.assertFalse(self.x0 == self.x0_copy)
        self.x0_copy.units = XSCALE0["units"]
        self.assertTrue(self.x0 == self.x0_copy)

        self.x0_copy.start = -100
        self.assertFalse(self.x0 == self.x0_copy)
        self.x0_copy.start = XSCALE0["start"]
        self.assertTrue(self.x0 == self.x0_copy)

        self.x0_copy.delta = 100
        self.assertFalse(self.x0 == self.x0_copy)
        self.x0_copy.delta = XSCALE0["delta"]
        self.assertTrue(self.x0 == self.x0_copy)

    def test02_copy(self):
        # Test deepcopy creates an equal but separate object
        c = copy.deepcopy(self.y1)
        self.assertTrue(c == self.y1)
        self.assertFalse(c is self.y1)  # different object
        self.assertFalse(c == self.y0)

        c = NMDimension(parent=NM, name=YSNAME0, scale=YSCALE0)
        self.assertFalse(c == self.y1)
        self.assertTrue(c == self.y0)

    def test03_parameters(self):
        klist = ["name", "created", "copy of"]  # NMObject
        klist += ["label", "units"]
        plist = self.y0.parameters
        self.assertEqual(klist, list(plist.keys()))
        self.assertEqual(plist["name"], YSNAME0)
        self.assertIsNone(plist["copy of"])
        self.assertEqual(plist["label"], YSCALE0["label"])
        self.assertEqual(plist["units"], YSCALE0["units"])

        plist = self.y0_copy.parameters
        self.assertEqual(klist, list(plist.keys()))
        self.assertEqual(plist["name"], YSNAME0)
        self.assertEqual(plist["copy of"], YSNAME0)
        self.assertEqual(plist["label"], YSCALE0["label"])
        self.assertEqual(plist["units"], YSCALE0["units"])

        klist = ["name", "created", "copy of"]  # NMObject
        klist += ["label", "units", "start", "delta"]
        plist = self.x0.parameters
        self.assertEqual(klist, list(plist.keys()))
        self.assertEqual(plist["name"], XSNAME0)
        self.assertIsNone(plist["copy of"])
        self.assertEqual(plist["label"], XSCALE0["label"])
        self.assertEqual(plist["units"], XSCALE0["units"])
        self.assertEqual(plist["start"], XSCALE0["start"])
        self.assertEqual(plist["delta"], XSCALE0["delta"])

    def test04_scale(self):
        # args: scale

        bad = list(nmu.BADTYPES)
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                self.y0._scale_set(b)
            with self.assertRaises(TypeError):
                self.x0._scale_set(b)

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.y0._scale_set({"label": b})
            with self.assertRaises(TypeError):
                self.y0._scale_set({"units": b})

        bad = list(nmu.BADTYPES)
        bad.remove(3)
        bad.remove(3.14)
        for b in bad:
            with self.assertRaises(TypeError):
                self.x0._scale_set({"start": b})
            with self.assertRaises(TypeError):
                self.x0._scale_set({"delta": b})

        scale = {"label": "Vmem", "units": "mV", "test": 0}
        with self.assertRaises(KeyError):
            self.y0._scale_set(scale)
        with self.assertRaises(KeyError):
            self.y0.scale = self.x0.scale

        scale = {"LABEL": "test", "UNITS": "test"}
        self.y0._scale_set(scale)
        self.assertEqual(self.y0.label, "test")
        self.assertEqual(self.y0.units, "test")

        scale = {"LABEL": "test", "UNITS": "test", "start": -10, "delta": 10}
        self.x0._scale_set(scale)
        self.assertEqual(self.x0.label, "test")
        self.assertEqual(self.x0.units, "test")
        self.assertEqual(self.x0.start, -10)
        self.assertEqual(self.x0.delta, 10)

        self.assertTrue(self.y0.scale != self.y1.scale)
        self.y0.scale = self.y1.scale
        self.assertTrue(self.y0.scale == self.y1.scale)
        self.assertTrue(self.x0.scale != self.x1.scale)
        self.x0.scale = self.x1.scale
        self.assertTrue(self.x0.scale == self.x1.scale)

    def test05_label(self):
        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.y0.label = b
        self.y0.label = None
        self.assertEqual(self.y0.label, "")
        self.y0.label = ""
        self.assertEqual(self.y0.label, "")
        self.y0.label = "TEST"
        self.assertEqual(self.y0.label, "TEST")

    def test06_units(self):
        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.y0.units = b
        self.y0.units = None
        self.assertEqual(self.y0.units, "")
        self.y0.units = ""
        self.assertEqual(self.y0.units, "")
        self.y0.units = "TEST"
        self.assertEqual(self.y0.units, "TEST")

    def test07_start(self):
        bad = list(nmu.BADTYPES)
        bad.remove(3)
        bad.remove(3.14)
        for b in bad:
            with self.assertRaises(TypeError):
                self.x0.start = b

        with self.assertRaises(ValueError):
            self.x0.start = float("inf")
        with self.assertRaises(ValueError):
            self.x0.start = float("-inf")
        with self.assertRaises(ValueError):
            self.x0.start = float("nan")
        self.x0.start = -199
        self.assertEqual(self.x0.start, -199)

    def test08_delta(self):
        bad = list(nmu.BADTYPES)
        bad.remove(3)
        bad.remove(3.14)
        for b in bad:
            with self.assertRaises(TypeError):
                self.x0.delta = b

        with self.assertRaises(ValueError):
            self.x0.delta = float("inf")
        with self.assertRaises(ValueError):
            self.x0.delta = float("-inf")
        with self.assertRaises(ValueError):
            self.x0.delta = float("nan")
        self.x0.delta = -199
        self.assertEqual(self.x0.delta, -199)
