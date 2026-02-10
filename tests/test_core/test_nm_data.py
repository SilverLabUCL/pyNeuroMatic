#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:00:26 2023

@author: jason
"""
import copy
import numpy
import unittest

from pyneuromatic.core.nm_data import NMData, NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)
DNAME0 = "data0"
DNAME1 = "record1"
NPARRAY0 = numpy.full([4], 3.14, dtype=numpy.float64, order="C")
NPARRAY1 = numpy.full([5], 6.28, dtype=numpy.float64, order="C")
YSCALE0 = {"label": "Vm", "units": "mV"}
XSCALE0 = {"label": "time", "units": "ms", "start": 10, "delta": 0.01}
YSCALE1 = {"label": "Im", "units": "pA"}
XSCALE1 = {"label": "time", "units": "s", "start": -10, "delta": 0.2}


class NMDataTest(unittest.TestCase):

    def setUp(self):
        self.d0 = NMData(
            parent=NM, name=DNAME0, nparray=NPARRAY0,
            xscale=XSCALE0, yscale=YSCALE0
        )
        self.d1 = NMData(
            parent=NM, name=DNAME1, nparray=NPARRAY1,
            xscale=XSCALE1, yscale=YSCALE1
        )
        self.d0_copy = copy.deepcopy(self.d0)
        self.d1_copy = copy.deepcopy(self.d1)

        self.ds0 = NMDataSeries(parent=NM, name="dataseries0")

    # def tearDown(self):
    #    pass

    def test00_init(self):
        # args: parent, name (see NMObject)
        # args: nparray, xscale, yscale

        self.assertEqual(self.d0._parent, NM)
        self.assertEqual(self.d0.name, DNAME0)
        self.assertEqual(self.d0.yscale, YSCALE0)
        self.assertEqual(self.d0.xscale, XSCALE0)
        self.assertTrue(isinstance(self.d0.nparray, numpy.ndarray))

        # deepcopy preserves attributes
        self.assertEqual(self.d0_copy._parent, NM)
        self.assertEqual(self.d0_copy.name, DNAME0)
        self.assertEqual(self.d0_copy.yscale, YSCALE0)
        self.assertEqual(self.d0_copy.xscale, XSCALE0)
        self.assertTrue(isinstance(self.d0_copy.nparray, numpy.ndarray))
        # deepcopy creates a new array
        self.assertFalse(self.d0_copy.nparray is self.d0.nparray)

    def xtest01_eq(self):
        self.assertTrue(self.d0 == self.d0)
        self.assertFalse(self.d0 == self.d1)
        self.assertTrue(self.d0_copy == self.d0)

        x = XSCALE0.copy()
        d0 = NMData(
            parent=None, name=DNAME0, nparray=NPARRAY0, xscale=x, yscale=YSCALE0
        )
        self.assertTrue(d0 == self.d0)
        d0.xscale["delta"] = 0.05
        self.assertFalse(d0 == self.d0)
        d0.xscale["delta"] = XSCALE0["delta"]
        self.assertTrue(d0 == self.d0)
        d0.yscale["units"] = "test"
        self.assertFalse(d0 == self.d0)
        d0.yscale["units"] = YSCALE0["units"]
        self.assertTrue(d0 == self.d0)

        d0.nparray = numpy.full([4], 3.14, dtype=numpy.float64, order="C")
        self.assertTrue(d0 == self.d0)
        d0.nparray = numpy.full([4], 3.14, dtype=numpy.float64, order="C")
        d0.nparray[-1] = 0
        self.assertFalse(d0 == self.d0)
        d0.nparray = numpy.full([5], 3.14, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        d0.nparray = numpy.full([4], 3.143, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        d0.nparray = numpy.full([4], 3.14, dtype=numpy.float16, order="C")
        self.assertFalse(d0 == self.d0)
        d0.nparray = numpy.full([4], numpy.nan, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        self.d0.nparray.fill(numpy.nan)
        if nmp.NAN_EQ_NAN:
            self.assertTrue(d0 == self.d0)
        else:
            self.assertFalse(d0 == self.d0)

        # TODO: dataseries

    def xtest02_copy(self):
        c = self.d0.copy()
        self.assertIsInstance(c, NMData)
        self.assertTrue(self.d0 == c)
        self.assertEqual(self.d0._parent, c._parent)
        self.assertEqual(self.d0.name, c.name)
        p0 = self.d0.parameters
        p = c.parameters
        self.assertNotEqual(p0.get("created"), p.get("created"))

    def xtest03_parameters(self):
        plist = ["name", "created", "copy of"]
        plist += ["xscale", "yscale", "nparray", "dataseries"]
        klist = list(self.d0.parameters.keys())
        self.assertEqual(klist, plist)

    def xtest03_x(self):
        with self.assertRaises(AttributeError):
            self.d0.xscale = None
        start = 100
        self.d0.xscale["start"] = start
        self.assertEqual(self.d0.xscale["start"], start)

    def xtest04_y(self):
        with self.assertRaises(AttributeError):
            self.d0.yscale = None
        label = "test"
        self.d0.yscale["label"] = label
        self.assertEqual(self.d0.yscale["label"], label)

    def xtest05_dataseries(self):
        """
        bad = list(nmu.BADTYPES)
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.d0.dataseries = b
        """

    def xtest06_nparray(self):
        bad = list(nmu.BADTYPES)
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.d0.nparray = b

    def xtest07_datacontainer(self):
        c0 = NMDataContainer(parent=NM, name="DataContainer0")
        dnlist0 = []
        dolist0 = []
        ndata = 10
        for i in range(ndata):
            n = "data" + str(i)
            d = NMData(
                parent=NM, name=n, nparray=NPARRAY0, xscale=XSCALE0, yscale=YSCALE0
            )
            c0.update(d)
            dnlist0.append(n)
            dolist0.append(d)
        nlist = [dnlist0[i] for i in range(0, ndata, 2)]
        c0.sets.add("set0", nlist)
        nlist = [dnlist0[i] for i in range(1, ndata, 2)]
        c0.sets.add("set1", nlist)
        c0.sets.define_or("set2", "set0", "set1")
        klist = list(c0.sets.keys())
        self.assertEqual(klist, ["set0", "set1", "set2"])

        c1 = NMDataContainer(parent=NM, name="DataContainer1")
        dnlist1 = []
        dolist1 = []
        ndata = 13
        for i in range(ndata):
            n = "record" + str(i)
            d = NMData(
                parent=NM, name=n, nparray=NPARRAY1, xscale=XSCALE1, yscale=YSCALE1
            )
            c1.update(d)
            dnlist1.append(n)
            dolist1.append(d)
        nlist = [dnlist1[i] for i in range(0, ndata, 2)]
        c1.sets.add("set0", nlist)
        nlist = [dnlist1[i] for i in range(1, ndata, 2)]
        c1.sets.add("set1", nlist)
        c1.sets.define_or("set2", "set0", "set1")

        # eq
        self.assertFalse(c0 == c1)

        # copy
        c0_copy = NMDataContainer(copy=c0)
        self.assertTrue(c0_copy == c0)

        # content_type
        self.assertEqual(c0.content_type(), "NMData")

        # new
        dnew = c0.new("test")
        self.assertTrue(isinstance(dnew, NMData))
        self.assertFalse(c0_copy == c0)
        with self.assertRaises(KeyError):
            dnew = c0.new("test")
        c0.pop("test")
        self.assertTrue(c0_copy == c0)
        c0.sets.remove("set0", dnlist0[0])
        self.assertFalse(c0_copy == c0)
        c0.sets.add("set0", dnlist0[0])
        self.assertTrue(c0_copy == c0)

    """
    def _test_data(self):

        # __init__()
        # args: parent, name, nparray, xscale, yscale, copy
        # dataseries, dataseries_channel, dataseries_epoch
        n0 = 'RecordA0'
        n1 = 'RecordA1'
        ds = NMDataSeries(parent=PARENT, name='Record')

        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = NMData(parent=PARENT, name=n0, dataseries=b)
        for b in BADTYPES:
            if isinstance(b, str):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = NMData(parent=PARENT, name=n0, dataseries=ds,
                            dataseries_channel=b)
        for b in BADTYPES:
            if isinstance(b, int):
                continue  # ok
            with self.assertRaises(TypeError):
                d0 = NMData(parent=PARENT, name=n0, dataseries=ds,
                            dataseries_channel='A', dataseries_epoch=b)
        nparray0 = numpy.full([4], 3.14, dtype=numpy.float64, order='C')
        nparray1 = numpy.full([5], 6.28, dtype=numpy.float64, order='C')
        d0 = NMData(parent=PARENT, name=n0, nparray=nparray0, xscale=XSCALE0,
                    yscale=YSCALE0)
        d1 = NMData(parent=PARENT, name=n1, nparray=nparray1, xscale=XSCALE1,
                    yscale=YSCALE1)
        self.assertTrue(numpy.array_equal(d0._NMData__nparray, nparray0))
        self.assertTrue(numpy.array_equal(d1._NMData__nparray, nparray1))

        # parameters()
        plist = PLIST + ['xscale', 'yscale', 'dataseries']
        self.assertEqual(d0.parameter_list, plist)

        # content()
        content_name = 'nmdata'
        c = d0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], d0.name)

        # copy()
        c = d0.copy()
        self.assertIsInstance(c, NMData)
        self.assertTrue(d0._isequivalent(c, alert=ALERT))

        # nparray setter
        for b in BADTYPES:
            if b is None:
                continue
            with self.assertRaises(TypeError):
                d0.nparray = b
        d0.nparray = None
        self.assertIsNone(d0.nparray)
        d0.nparray = nparray1
        self.assertIsInstance(d0.nparray, numpy.ndarray)
        # TODO
        # dataseries
        # dataseries_set
        # TODO x- and y-scale

    def _test_data_container(self):
        c0 = NMDataContainer(parent=PARENT, name='Data')
        c1 = NMDataContainer(parent=PARENT, name='Data')
        self.assertEqual(c0.parameters['type'], 'NMData')
        self.assertEqual(c0.prefix, nmp.DATA_PREFIX)
        self.assertTrue(c0.parameters['rename'])
        # new, args: name, nparray, xscale=, yscale, dataseries, select
        nlist = ['RecordA0', 'WaveA0', 'Xdata']
        for n in nlist:
            self.assertIsInstance(c0.new(name=n, xscale=XSCALE0,
                                         yscale=YSCALE0), NMData)
            self.assertIsInstance(c1.new(name=n, xscale=XSCALE0,
                                         yscale=YSCALE0), NMData)
        # append(), args: data, select
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0.append(b)
        nparray0 = numpy.full([4], 3.14, dtype=numpy.float64, order='C')
        d0 = NMData(parent=PARENT, name='RecordA1', nparray=nparray0,
                    xscale=XSCALE0, yscale=YSCALE0)
        self.assertTrue(c0.append(d0))

        # copy()
        c = c0.copy()
        self.assertIsInstance(c, NMDataContainer)
        self.assertTrue(c._isequivalent(c0, alert=ALERT))
        # isequivalent, args: DataContainer
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        d1 = NMData(parent=PARENT, name='RecordA1', nparray=nparray0,
                    xscale=XSCALE0, yscale=YSCALE0)
        c1.append(d1)
        self.assertTrue(c0._isequivalent(c1, alert=ALERT))
        self.assertIsInstance(c0._select_set('WaveA0'), NMData)
        self.assertIsInstance(c1._select_set('RecordA0'), NMData)
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
    """
