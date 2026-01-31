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
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
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
        # Create dimensions with nparray
        xdim0 = NMDimensionX(parent=None, name="xscale", scale=XSCALE0)
        ydim0 = NMDimension(parent=None, name="yscale", nparray=NPARRAY0, scale=YSCALE0)
        xdim1 = NMDimensionX(parent=None, name="xscale", scale=XSCALE1)
        ydim1 = NMDimension(parent=None, name="yscale", nparray=NPARRAY1, scale=YSCALE1)

        self.d0 = NMData(parent=NM, name=DNAME0, xdim=xdim0, ydim=ydim0)
        self.d1 = NMData(parent=NM, name=DNAME1, xdim=xdim1, ydim=ydim1)
        self.d0_copy = copy.deepcopy(self.d0)
        self.d1_copy = copy.deepcopy(self.d1)

        self.ds0 = NMDataSeries(parent=NM, name="dataseries0")

    # def tearDown(self):
    #    pass

    def test00_init(self):
        # args: parent, name (see NMObject)
        # args: xdim, ydim, dataseries_channel, dataseries_epoch

        self.assertEqual(self.d0._parent, NM)
        self.assertEqual(self.d0.name, DNAME0)
        self.assertEqual(self.d0.y.scale, YSCALE0)
        self.assertEqual(self.d0.x.scale, XSCALE0)
        self.assertTrue(isinstance(self.d0.y.nparray, numpy.ndarray))

        # deepcopy preserves attributes
        self.assertEqual(self.d0_copy._parent, NM)
        self.assertEqual(self.d0_copy.name, DNAME0)
        self.assertEqual(self.d0_copy.y.scale, YSCALE0)
        self.assertEqual(self.d0_copy.x.scale, XSCALE0)
        self.assertTrue(isinstance(self.d0_copy.y.nparray, numpy.ndarray))
        # deepcopy creates a new array
        self.assertFalse(self.d0_copy.y.nparray is self.d0.y.nparray)

    def xtest01_eq(self):
        self.assertTrue(self.d0 == self.d0)
        self.assertFalse(self.d0 == self.d1)
        self.assertTrue(self.d0_copy == self.d0)

        x = XSCALE0.copy()
        d0 = NMData(
            parent=None, name=DNAME0, np_array=NPARRAY0, xscale=x, yscale=YSCALE0
        )
        self.assertTrue(d0 == self.d0)
        d0.x.delta = 0.05
        self.assertFalse(d0 == self.d0)
        d0.x.delta = XSCALE0["delta"]
        self.assertTrue(d0 == self.d0)
        d0.y.units = "test"
        self.assertFalse(d0 == self.d0)
        d0.y.units = YSCALE0["units"]
        self.assertTrue(d0 == self.d0)

        d0.np_array = numpy.full([4], 3.14, dtype=numpy.float64, order="C")
        self.assertTrue(d0 == self.d0)
        d0.np_array = numpy.full([4], 3.14, dtype=numpy.float64, order="C")
        d0.np_array[-1] = 0
        self.assertFalse(d0 == self.d0)
        d0.np_array = numpy.full([5], 3.14, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        d0.np_array = numpy.full([4], 3.143, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        d0.np_array = numpy.full([4], 3.14, dtype=numpy.float16, order="C")
        self.assertFalse(d0 == self.d0)
        d0.np_array = numpy.full([4], numpy.nan, dtype=numpy.float64, order="C")
        self.assertFalse(d0 == self.d0)
        self.d0.np_array.fill(numpy.nan)
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
        plist += ["xscale", "yscale", "np_array", "dataseries"]
        klist = list(self.d0.parameters.keys())
        self.assertEqual(klist, plist)

    def xtest03_x(self):
        with self.assertRaises(AttributeError):
            self.d0.x = None
        self.assertIsInstance(self.d0.x, NMDimensionX)
        start = 100
        self.d0.x.scale["start"] = start  # no change
        self.assertEqual(self.d0.x.start, XSCALE0["start"])
        self.d0.x.start = start
        self.assertEqual(self.d0.x.start, start)

    def xtest04_y(self):
        with self.assertRaises(AttributeError):
            self.d0.y = None
        self.assertIsInstance(self.d0.y, NMDimension)
        label = "test"
        self.d0.y.scale["label"] = label  # no change
        self.assertEqual(self.d0.y.label, YSCALE0["label"])
        self.d0.y.label = label
        self.assertEqual(self.d0.y.label, label)

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
                self.d0.np_array = b

    def xtest07_datacontainer(self):
        c0 = NMDataContainer(parent=NM, name="DataContainer0")
        dnlist0 = []
        dolist0 = []
        ndata = 10
        for i in range(ndata):
            n = "data" + str(i)
            d = NMData(
                parent=NM, name=n, np_array=NPARRAY0, xscale=XSCALE0, yscale=YSCALE0
            )
            c0.update(d)
            dnlist0.append(n)
            dolist0.append(d)
        nlist = [dnlist0[i] for i in range(0, ndata, 2)]
        c0.sets.add("set0", nlist)
        nlist = [dnlist0[i] for i in range(1, ndata, 2)]
        c0.sets.add("set1", nlist)
        nlist = ["set0", "|", "set1"]
        c0.sets.add("set2", nlist)
        klist = list(c0.sets.keys())
        self.assertEqual(klist, ["set0", "set1", "set2"])

        c1 = NMDataContainer(parent=NM, name="DataContainer1")
        dnlist1 = []
        dolist1 = []
        ndata = 13
        for i in range(ndata):
            n = "record" + str(i)
            d = NMData(
                parent=NM, name=n, np_array=NPARRAY1, xscale=XSCALE1, yscale=YSCALE1
            )
            c1.update(d)
            dnlist1.append(n)
            dolist1.append(d)
        nlist = [dnlist1[i] for i in range(0, ndata, 2)]
        c1.sets.add("set0", nlist)
        nlist = [dnlist1[i] for i in range(1, ndata, 2)]
        c1.sets.add("set1", nlist)
        nlist = ["set0", "|", "set1"]
        c1.sets.add("set2", nlist)

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
        c0.pop("test", auto_confirm="y")
        self.assertTrue(c0_copy == c0)
        c0.sets.remove("set0", dnlist0[0])
        self.assertFalse(c0_copy == c0)
        c0.sets.add("set0", dnlist0[0])
        self.assertTrue(c0_copy == c0)

    """
    def _test_data(self):

        # __init__()
        # args: parent, name, np_array, xscale, yscale, copy
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
        # nparrayx = numpy.full([6], 12.56, dtype=numpy.float64, order='C')
        d0 = NMData(parent=PARENT, name=n0, np_array=nparray0, xscale=XSCALE0,
                    yscale=YSCALE0)
        d1 = NMData(parent=PARENT, name=n1, np_array=nparray1, xscale=XSCALE1,
                    yscale=YSCALE1)
        # xdata = NMData(parent=PARENT, name='xdata', np_array=nparrayx,
        #               xscale=XSCALEx, yscale=YSCALEx)
        self.assertTrue(numpy.array_equal(d0._NMData__np_array, nparray0))
        self.assertTrue(numpy.array_equal(d1._NMData__np_array, nparray1))

        # parameters()
        plist = PLIST + ['xscale', 'yscale', 'dataseries']
        self.assertEqual(d0.parameter_list, plist)

        # content()
        content_name = 'nmdata'
        c = d0.content
        self.assertIsInstance(c, dict)
        self.assertEqual(list(c.keys()), [content_name])
        self.assertEqual(c[content_name], d0.name)
        # self.assertEqual(c['notes'], d0.note.names)

        # isequivalent()
        # args: NMData
        self.assertFalse(d0._isequivalent(d1, alert=ALERT))
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray0,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertTrue(d0._isequivalent(d00, alert=ALERT))
        nparray00 = numpy.full([4], 3.14, dtype=numpy.float64, order='F')
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray00,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertTrue(d0._isequivalent(d00, alert=ALERT))
        nparray0[2] = 5
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray0[2] = 0
        d00 = NMData(parent=PARENT, name=n0, np_array=None,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = numpy.full([5, 2], 3.14, dtype=numpy.float64, order='C')
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray00,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = numpy.full([5], 3.14, dtype=numpy.int32, order='C')
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray00,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        nparray00 = numpy.full([5], 3.14, dtype=numpy.float64, order='F')
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray00,
                     xscale=XSCALE0, yscale=YSCALE0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray0,
                     xscale=XSCALE1, yscale=YSCALE0)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))
        d00 = NMData(parent=PARENT, name=n0, np_array=nparray0,
                     xscale=XSCALE0, yscale=YSCALE1)
        self.assertFalse(d0._isequivalent(d00, alert=ALERT))

        # copy()
        c = d0.copy()
        self.assertIsInstance(c, NMData)
        self.assertTrue(d0._isequivalent(c, alert=ALERT))

        # np_array_set()
        # args: np_array
        for b in BADTYPES:
            if b is None:
                continue
            with self.assertRaises(TypeError):
                d0._np_array_set(b)
        self.assertTrue(d0._np_array_set(None))
        self.assertIsNone(d0.np_array)
        self.assertTrue(d0._np_array_set(nparray1))
        self.assertIsInstance(d0.np_array, numpy.ndarray)
        # np_array_make, args: shape, fill_value, dtype, order
        # wrapper fxn, so basic testing here
        self.assertTrue(d0.np_array_make((10, 2)))
        # np_array_make_random_normal, args: shape, mean, stdv
        # wrapper fxn, so basic testing here
        self.assertTrue(d0.np_array_make_random_normal(10, mean=3, stdv=1))
        # TODO
        # dataseries
        # dataseries_set
        # ds = NMDataSeries(parent=PARENT, name='Record')
        # TODO x- and y-scale

    def _test_data_container(self):
        c0 = NMDataContainer(parent=PARENT, name='Data')
        c1 = NMDataContainer(parent=PARENT, name='Data')
        self.assertEqual(c0.parameters['type'], 'NMData')
        self.assertEqual(c0.prefix, nmp.DATA_PREFIX)
        self.assertTrue(c0.parameters['rename'])
        # new, args: name, np_array, xscale=, yscale, dataseries, select
        # wrapper for Data.new() and NMObjectContainer.new()
        nlist = ['RecordA0', 'WaveA0', 'Xdata']
        for n in nlist:
            self.assertIsInstance(c0.new(name=n, xscale=XSCALE0,
                                         yscale=YSCALE0), NMData)
            self.assertIsInstance(c1.new(name=n, xscale=XSCALE0,
                                         yscale=YSCALE0), NMData)
        # append(), args: data, select
        # wrapper for NMObjectContainer.new()
        for b in BADTYPES:
            if b is None:
                continue  # ok
            with self.assertRaises(TypeError):
                c0.append(b)
        nparray0 = numpy.full([4], 3.14, dtype=numpy.float64, order='C')
        d0 = NMData(parent=PARENT, name='RecordA1', np_array=nparray0,
                    xscale=XSCALE0, yscale=YSCALE0)
        self.assertTrue(c0.append(d0))

        # copy()
        c = c0.copy()
        self.assertIsInstance(c, NMDataContainer)
        self.assertTrue(c._isequivalent(c0, alert=ALERT))
        # isequivalent, args: DataContainer
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        d1 = NMData(parent=PARENT, name='RecordA1', np_array=nparray0,
                    xscale=XSCALE0, yscale=YSCALE0)
        c1.append(d1)
        self.assertTrue(c0._isequivalent(c1, alert=ALERT))
        self.assertIsInstance(c0._select_set('WaveA0'), NMData)
        self.assertIsInstance(c1._select_set('RecordA0'), NMData)
        self.assertFalse(c0._isequivalent(c1, alert=ALERT))
        # remove(), args: names, indexes, confirm
        # wrapper for NMObjectContainer.remove()
        # TODO, test if Data is removed from dataseries and sets
    """
