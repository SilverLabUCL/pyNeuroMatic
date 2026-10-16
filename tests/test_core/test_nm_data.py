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
from pyneuromatic.core.nm_notes import NMNotes
import pyneuromatic.core.nm_configurations as nmc
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
        d0.xscale.delta = 0.05
        self.assertFalse(d0 == self.d0)
        d0.xscale.delta = XSCALE0["delta"]
        self.assertTrue(d0 == self.d0)
        d0.yscale.units = "test"
        self.assertFalse(d0 == self.d0)
        d0.yscale.units = YSCALE0["units"]
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
        if nmc.NAN_EQ_NAN:
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
        self.d0.xscale.start = start
        self.assertEqual(self.d0.xscale.start, start)

    def xtest04_y(self):
        with self.assertRaises(AttributeError):
            self.d0.yscale = None
        label = "test"
        self.d0.yscale.label = label
        self.assertEqual(self.d0.yscale.label, label)

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

class TestNMDataContainerNew(unittest.TestCase):
    """Tests for NMDataContainer.new()."""

    def setUp(self):
        from pyneuromatic.core.nm_folder import NMFolder
        self.folder = NMFolder(parent=NM, name="TestFolder")

    def test_new_returns_nmdata(self):
        d = self.folder.data.new("wave0", nparray=NPARRAY0)
        self.assertIsInstance(d, NMData)

    def test_new_accepts_xarray(self):
        xa = numpy.linspace(0.0, 0.003, len(NPARRAY0))
        d = self.folder.data.new("wave0", nparray=NPARRAY0, xarray=xa)
        self.assertIsInstance(d, NMData)
        numpy.testing.assert_array_equal(d.xarray, xa)

    def test_new_xarray_none_by_default(self):
        d = self.folder.data.new("wave0", nparray=NPARRAY0)
        self.assertIsNone(d.xarray)

    def test_new_xarray_and_xscale_coexist(self):
        xa = numpy.linspace(0.0, 0.003, len(NPARRAY0))
        d = self.folder.data.new("wave0", nparray=NPARRAY0,
                                 xarray=xa, xscale=XSCALE0)
        numpy.testing.assert_array_equal(d.xarray, xa)
        self.assertEqual(d.xscale.units, XSCALE0["units"])


class TestNMDataNotes(unittest.TestCase):
    """Tests for NMData notes integration with NMNotes."""

    def setUp(self):
        self.d0 = NMData(
            parent=NM, name=DNAME0, nparray=NPARRAY0,
            xscale=XSCALE0, yscale=YSCALE0
        )

    def test_notes_returns_nmnotes(self):
        self.assertIsInstance(self.d0.notes, NMNotes)

    def test_notes_initially_empty(self):
        self.assertEqual(len(self.d0.notes), 0)

    def test_notes_add(self):
        self.d0.notes.add("filtered with 1kHz lowpass")
        self.d0.notes.add("baseline subtracted")
        self.assertEqual(len(self.d0.notes), 2)
        self.assertEqual(self.d0.notes[0].get("note"), "filtered with 1kHz lowpass")

    def test_notes_affect_equality(self):
        c = copy.deepcopy(self.d0)
        self.assertTrue(c == self.d0)
        c.notes.add("transformed")
        self.assertFalse(c == self.d0)

    def test_deepcopy_preserves_notes(self):
        self.d0.notes.add("test note")
        c = copy.deepcopy(self.d0)
        self.assertEqual(len(c.notes), 1)
        self.assertEqual(c.notes[0].get("note"), "test note")
        # Verify independence
        c.notes.add("new note")
        self.assertEqual(len(self.d0.notes), 1)
        self.assertEqual(len(c.notes), 2)


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
        self.assertEqual(c0.prefix, nmc.DATA_PREFIX)
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


class TestNMDataContainerDataSeriesMembership(unittest.TestCase):
    """Deleting, renaming and copying data that belongs to a data series."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.folder = self.nm.folders.new("Demo")
        self.other = self.nm.folders.new("Other")
        for name in ("RecordA0", "RecordA1", "RecordA2", "RecordB0", "Loose0"):
            self.folder.data.new(name, nparray=numpy.zeros(3))
        self.series = self.folder.sync_dataseries("Record")
        self.data = self.folder.data

    def test_dataseries_of(self):
        self.assertEqual([s.name for s in self.data.dataseries_of("RecordA1")], ["Record"])
        self.assertEqual(self.data.dataseries_of("Loose0"), [])
        with self.assertRaises(KeyError):
            self.data.dataseries_of("Missing")

    def test_pop_unlinks_data_from_series(self):
        removed = self.data.pop("RecordA1")
        self.assertIsNone(self.series.get_data("A", "E1"))
        for item in list(self.series.channels.values()) + list(self.series.epochs.values()):
            self.assertNotIn(removed, item.data)
        self.assertIs(self.series.get_data("A", "E0"), self.data["RecordA0"])

    def test_rename_leaves_series_unless_name_still_matches(self):
        self.data.rename("RecordA2", "Kept")
        self.assertIsNone(self.series.get_data("A", "E2"))
        self.assertEqual(self.data.dataseries_of("Kept"), [])
        self.data.rename("Kept", "RecordA5")
        self.assertEqual([s.name for s in self.data.dataseries_of("RecordA5")], ["Record"])

    def test_locked_series_blocks_pop_and_rename(self):
        self.series.locked = True
        with self.assertRaisesRegex(RuntimeError, "locked data series 'Record'"):
            self.data.pop("RecordA0")
        with self.assertRaisesRegex(RuntimeError, "locked data series 'Record'"):
            self.data.rename("RecordA0", "Other0")
        self.assertIn("RecordA0", self.data)
        self.assertIs(self.series.get_data("A", "E0"), self.data["RecordA0"])
        # Data outside the series is unaffected
        self.data.rename("Loose0", "Free0")
        self.data.pop("Free0")

    def test_copy_to_another_folder(self):
        copy = self.data.copy_to("RecordA0", self.other.data)
        self.assertIs(self.other.data["RecordA0"], copy)
        self.assertIsNot(copy, self.data["RecordA0"])
        self.assertEqual(self.other.data.dataseries_of("RecordA0"), [])
        renamed = self.data.copy_to("RecordA0", self.other.data, "Copy0")
        self.assertEqual(renamed.name, "Copy0")
        with self.assertRaises(KeyError):
            self.data.copy_to("RecordA0", self.other.data)  # name taken
        with self.assertRaises(TypeError):
            self.data.copy_to("RecordA0", self.other)

    def test_lock_flag(self):
        self.assertFalse(self.series.locked)
        self.series.locked = True
        self.assertTrue(self.series.parameters["locked"])
        self.assertTrue(copy.deepcopy(self.series).locked)
        with self.assertRaises(TypeError):
            self.series.locked = "yes"


class TestLockedDataSeriesMembership(unittest.TestCase):
    """A locked data series neither loses nor gains members."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        self.folder = self.nm.folders.new("Demo")
        for name in ("RecordA0", "RecordA1", "Loose0"):
            self.folder.data.new(name, nparray=numpy.zeros(3))
        self.series = self.folder.sync_dataseries("Record")
        self.series.locked = True

    def test_sync_does_not_add_to_locked_series(self):
        self.folder.data.new("RecordA2", nparray=numpy.zeros(3))
        self.assertIs(self.folder.sync_dataseries("Record"), self.series)
        self.assertEqual(list(self.series.epochs.keys()), ["E0", "E1"])
        self.assertEqual(self.folder.data.dataseries_of("RecordA2"), [])
        self.series.locked = False
        self.folder.sync_dataseries("Record")
        self.assertEqual(list(self.series.epochs.keys()), ["E0", "E1", "E2"])

    def test_rename_into_locked_series_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "fits locked data series 'Record'"):
            self.folder.data.rename("Loose0", "RecordA7")
        self.assertIn("Loose0", self.folder.data)
        self.assertIsNone(self.folder.data.locked_dataseries_for_name("C_RecordA7"))

    def test_copy_under_new_prefix_is_unlocked(self):
        copy = self.folder.copy_dataseries("Record", "Work")
        self.assertFalse(copy.locked)
        self.folder.data.rename("WorkA1", "Edited")
        self.assertEqual(self.folder.data.dataseries_of("Edited"), [])
        # The original is untouched
        self.assertIs(self.series.get_data("A", "E1"), self.folder.data["RecordA1"])
