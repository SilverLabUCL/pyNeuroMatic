#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:56:59 2023

@author: jason
"""
import unittest
import random

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_sets import NMSets
import pyneuromatic.core.nm_utilities as nmu
from tests.test_core.nm_object_test import NMObject2

QUIET = True
NM = NMManager(quiet=QUIET)
NM0 = NMManager(quiet=QUIET)
NM1 = NMManager(quiet=QUIET)
SETS_PREFIX0 = "set"
SETS_PREFIX1 = "S"
SETS_NAME0 = "sets0"
SETS_NAME1 = "sets1"
NUM_NMOBJECTS0 = 16
NUM_NMOBJECTS1 = 12
ONLIST0 = ["TestA" + str(i) for i in range(NUM_NMOBJECTS0)]
ONLIST1 = ["TestB" + str(i) for i in range(NUM_NMOBJECTS1)]
ONLIST1.reverse()  # REVERSE names


class NMSetsTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.olist0 = []
        self.olist1 = []
        self.odict0 = {}  # NM object dict
        self.odict1 = {}
        self.olist0_copy = []
        self.olist1_copy = []
        self.odict0_copy = {}
        self.odict1_copy = {}

        # create NMObjects and put in a dictionary

        for n in ONLIST0:
            o = NMObject(parent=NM0, name=n)
            self.olist0.append(o)
            self.odict0.update({n: o})
            c = o.copy()
            self.olist0_copy.append(c)
            self.odict0_copy.update({n: c})

        for n in ONLIST1:
            o = NMObject(parent=NM1, name=n)
            self.olist1.append(o)
            self.odict1.update({n: o})
            c = o.copy()
            self.olist1_copy.append(c)
            self.odict1_copy.update({n: c})

        # create sets0

        self.sets0 = NMSets(
            parent=NM0,
            name=SETS_NAME0,
            nmobjects=self.odict0
            # nmobjects_fxnref=self._nmobjects_dict0  # function reference
        )

        self.sets0_init = []
        num_sets = 2

        for i in range(num_sets):
            sname = SETS_PREFIX0 + str(i)

            ilist = NMSetsTest.__ilist_random(NUM_NMOBJECTS0)
            for i in ilist:
                self.sets0.add(sname, ONLIST0[i])  # add in random order

            ilist.sort()  # sort according to nmobjects dict
            nlist = []
            olist = []
            eqlist = []

            for i in ilist:
                nlist.append(ONLIST0[i])
                olist.append(self.olist0[i])

            sdict = {}
            sdict["name"] = sname
            sdict["ilist"] = ilist
            sdict["nlist"] = nlist
            sdict["olist"] = olist
            sdict["eqlist"] = eqlist

            self.sets0_init.append(sdict)

        # create sets1

        self.sets1 = NMSets(
            parent=NM1,
            name=SETS_NAME1,
            # nmobjects=self.odict1
            nmobjects_fxnref=self._nmobjects_dict1,  # function reference
        )

        self.sets1_init = []
        num_sets = 3  # 3 or more
        oplist = ["&", "|", "-", "^"]

        for i in range(num_sets):
            sname = SETS_PREFIX1 + str(i + 1)

            if i == num_sets - 1:  # last set is an equation
                sname0 = self.sets1_init[0]["name"]
                sname1 = self.sets1_init[1]["name"]
                s0 = set(self.sets1_init[0]["ilist"])
                s1 = set(self.sets1_init[1]["ilist"])
                j = random.randint(0, 3)
                op = oplist[j]
                if op == "&":
                    ilist = list(s0 & s1)
                elif op == "|":
                    ilist = list(s0 | s1)
                elif op == "-":
                    ilist = list(s0 - s1)
                elif op == "^":
                    ilist = list(s0 ^ s1)
                else:
                    raise RuntimeError("no op")
                # ilist.sort()
                eqlist = [sname0, op, sname1]
                # print('\nset3 = ' + str(eqlist) + ', ilist = ' + str(ilist))
            else:
                ilist = NMSetsTest.__ilist_random(NUM_NMOBJECTS1)
                eqlist = []

            if eqlist:
                self.sets1[sname] = eqlist
            else:
                for i in ilist:
                    self.sets1.add(sname, ONLIST1[i])

            ilist.sort()
            nlist = []
            olist = []

            for i in ilist:
                nlist.append(ONLIST1[i])
                olist.append(self.olist1[i])

            sdict = {}
            sdict["name"] = sname
            sdict["ilist"] = ilist
            sdict["nlist"] = nlist
            sdict["olist"] = olist
            sdict["eqlist"] = eqlist

            self.sets1_init.append(sdict)

        # copy sets0 and sets1

        self.sets0_copy = NMSets(
            parent=None, name=None, nmobjects=self.odict0_copy, copy=self.sets0
        )

        nm_other = NMManager(quiet=QUIET)

        self.sets1_copy = NMSets(
            parent=nm_other, name="test", nmobjects=self.odict1_copy, copy=self.sets1
        )

    def __ilist_random(irange):
        icount = irange * 0.5
        ilist = []
        for i in range(irange * 2):
            j = random.randint(0, irange - 1)
            if j not in ilist:
                ilist.append(j)
            if len(ilist) >= icount:
                break
        # ilist.sort()
        return ilist

    def _nmobjects_dict0(self):  # for testing nmobjects_fxnref
        return self.odict0

    def _nmobjects_dict1(self):  # for testing nmobjects_fxnref
        return self.odict1

    def test00_init(self):
        # args: parent, name, copy (see NMObject)
        # args: nmobjects_fxnref, nmobjects

        bad = list(nmu.BADTYPES)
        for b in bad:
            with self.assertRaises(ValueError):
                NMSets(nmobjects_fxnref=b)

        bad = list(nmu.BADTYPES)
        bad.remove({})
        for b in bad:
            with self.assertRaises(ValueError):
                NMSets(nmobjects=b)

        o2 = NMObject2(parent=NM0, name="test")
        with self.assertRaises(TypeError):
            NMSets(nmobjects=self.odict0, copy=o2)

        with self.assertRaises(ValueError):
            NMSets(name="test")  # no args for nmobjects_fxnref or nmobjects

        klist = []
        for d in self.sets0_init:
            klist.append(d["name"])

        self.assertEqual(self.sets0._parent, NM0)
        self.assertEqual(self.sets0.name, SETS_NAME0)
        self.assertEqual(list(self.sets0.keys()), klist)
        self.assertEqual(list(self.sets0._nmobjects_dict.keys()), ONLIST0)

        self.assertEqual(self.sets0_copy._parent, NM0)
        self.assertEqual(self.sets0_copy.name, SETS_NAME0)
        self.assertEqual(list(self.sets0_copy.keys()), klist)
        self.assertEqual(list(self.sets0_copy._nmobjects_dict.keys()), ONLIST0)

        for i, o in enumerate(self.olist0):
            self.assertTrue(o == self.olist0_copy[i])
            self.assertFalse(o is self.olist0_copy[i])

        klist = []
        for d in self.sets1_init:
            klist.append(d["name"])

        self.assertEqual(self.sets1._parent, NM1)
        self.assertEqual(self.sets1.name, SETS_NAME1)
        self.assertEqual(list(self.sets1.keys()), klist)
        self.assertEqual(list(self.sets1._nmobjects_dict.keys()), ONLIST1)

        self.assertEqual(self.sets1_copy._parent, NM1)
        self.assertNotEqual(self.sets1_copy.name, "test")
        self.assertEqual(list(self.sets1_copy.keys()), klist)
        self.assertEqual(list(self.sets1_copy._nmobjects_dict.keys()), ONLIST1)

        for i, o in enumerate(self.olist1):
            self.assertTrue(o == self.olist1_copy[i])
            self.assertFalse(o is self.olist1_copy[i])

    def test01_copy(self):
        odict0 = {}
        for n in ONLIST0:
            o = NMObject2(parent=NM0, name=n)
            odict0.update({n: o})

        with self.assertRaises(TypeError):  # NMObject2 instead of NMObject
            s0_copy = self.sets0.copy(nmobjects=odict0)

        odict0 = {}
        for n in ONLIST0:
            o = NMObject(parent=NM0, name=n)
            odict0.update({n: o})
        odict0.update({4: o})  # bad key

        with self.assertRaises(TypeError):
            s0_copy = self.sets0.copy(nmobjects=odict0)

        odict0 = {}
        for n in ONLIST0:
            o = NMObject(parent=NM0, name=n)
            odict0.update({n: o})

        s0_copy = self.sets0.copy(nmobjects=odict0)

        self.assertIsInstance(s0_copy, NMSets)
        self.assertTrue(self.sets0 == s0_copy)
        self.assertEqual(self.sets0._parent, s0_copy._parent)
        self.assertEqual(self.sets0.name, s0_copy.name)
        p0 = self.sets0.parameters
        p = s0_copy.parameters
        self.assertNotEqual(p0.get("created"), p.get("created"))
        self.assertEqual(self.sets0.keys(), s0_copy.keys())

        # add new NMObject

        o = NMObject(parent=NM0, name="test")
        odict0.update({"test": o})
        self.assertTrue(self.sets0 == s0_copy)
        s0_copy.add("set1", "test")
        self.assertFalse(self.sets0 == s0_copy)

    def test02_parameters(self):
        klist = ["name", "created", "copy of"]  # # NMObject
        klist += ["sets"]

        klist2 = []
        for d in self.sets0_init:
            klist2.append(d["name"])

        plist = self.sets0.parameters
        self.assertEqual(list(plist.keys()), klist)
        self.assertEqual(plist["sets"], klist2)

        plist = self.sets0_copy.parameters
        self.assertEqual(list(plist.keys()), klist)
        self.assertEqual(plist["sets"], klist2)

        klist2 = []
        for d in self.sets1_init:
            klist2.append(d["name"])

        plist = self.sets1.parameters
        self.assertEqual(plist["sets"], klist2)

        plist = self.sets1_copy.parameters
        self.assertEqual(plist["sets"], klist2)

    def test03_getitem(self):
        # args: key
        # get(), items(), values() and '='

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            olist = self.sets0.get(b)
            self.assertIsNone(olist)
            olist = self.sets0[b]
            self.assertIsNone(olist)

        olist = self.sets0.get("test")
        self.assertIsNone(olist)
        olist = self.sets0.get("test", default="default")
        self.assertEqual(olist, "default")
        olist = self.sets0["test"]
        self.assertIsNone(olist)

        for d in self.sets0_init:
            sname = d["name"].upper()  # should be case-insensitive
            nlist = d["nlist"]
            olist = d["olist"]
            eqlist = d["eqlist"]
            olist_g = self.sets0.get(sname)
            nlist_g = self.sets0.get(sname, get_keys=True)
            self.assertEqual(olist, olist_g)
            self.assertEqual(nlist, nlist_g)
            olist_g = self.sets0[sname]  # '=' format
            self.assertEqual(olist, olist_g)
            if eqlist:
                eqlist_g = self.sets0.get(sname, get_equation=True)
                self.assertEqual(eqlist, eqlist_g)
            else:
                eqlist_g = self.sets0.get(sname, get_equation=True)
                self.assertEqual(olist, eqlist_g)
                eqlist_g = self.sets0.get(sname, default="test", get_equation=True)
                self.assertEqual(olist, eqlist_g)

        for d in self.sets1_init:
            sname = d["name"].upper()  # should be case-insensitive
            nlist = d["nlist"]
            olist = d["olist"]
            eqlist = d["eqlist"]
            olist_g = self.sets1.get(sname)
            nlist_g = self.sets1.get(sname, get_keys=True)
            self.assertEqual(olist, olist_g)
            self.assertEqual(nlist, nlist_g)
            olist_g = self.sets1[sname]  # '=' format
            self.assertEqual(olist, olist_g)
            if eqlist:
                eqlist_g = self.sets1.get(sname, get_equation=True)
                self.assertEqual(eqlist, eqlist_g)
            else:
                eqlist_g = self.sets1.get(sname, get_equation=True)
                self.assertEqual(olist, eqlist_g)

        # order of NMObjects matches order of self.odict0
        self.sets0.add("set99", ONLIST0[3])  # add backwards
        self.sets0.add("set99", ONLIST0[1])
        self.sets0.add("set99", ONLIST0[5])
        nlist = [ONLIST0[1], ONLIST0[3], ONLIST0[5]]
        olist = [self.olist0[1], self.olist0[3], self.olist0[5]]
        nlist2 = self.sets0.get("set99", get_keys=True)
        olist2 = self.sets0.get("set99")
        self.assertEqual(nlist, nlist2)
        self.assertEqual(olist, olist2)
        self.sets0.pop("set99", confirm_answer="y")

        for i, (k, v) in enumerate(self.sets0.items()):
            d = self.sets0_init[i]
            sname = d["name"]
            olist = d["olist"]
            self.assertEqual(k, sname)
            self.assertEqual(v, olist)

        for i, (k, v) in enumerate(self.sets1.items()):
            d = self.sets1_init[i]
            sname = d["name"]
            olist = d["olist"]
            self.assertEqual(k, sname)
            self.assertEqual(v, olist)

        # keys() and values() should be same as items()
        # no way to obtain set equation via items() or values()

    def xtest04_setitem(self):
        # args: key, olist, add
        # update(), add() and '='
        # see test36_equation()

        d = self.sets0_init[0]
        sname0 = d["name"].upper()  # should be case-insensitive
        d = self.sets0_init[1]
        sname1 = d["name"].upper()  # should be case-insensitive

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.add(b, "")
            with self.assertRaises(TypeError):
                self.sets0.update({b: ""})
            with self.assertRaises(TypeError):
                self.sets0[b] = ""

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        bad.remove("select")
        for b in bad:
            with self.assertRaises(ValueError):
                self.sets0.add(b, "")
            with self.assertRaises(ValueError):
                self.sets0.update({b: ""})
            with self.assertRaises(ValueError):
                self.sets0[b] = ""

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        bad.remove([])  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.add(sname0, b)
            with self.assertRaises(TypeError):
                self.sets0.update(sname0, b)
            with self.assertRaises(TypeError):
                self.sets0[sname0] = b

        with self.assertRaises(KeyError):
            self.sets0.add(sname0, "test")  # test does not exist
        with self.assertRaises(KeyError):
            self.sets0.update({sname0: "test"})
        with self.assertRaises(KeyError):
            self.sets0[sname0] = "test"

        for o in self.olist0:  # ok to add nmobject multiple times
            if o in self.sets0[sname0]:
                self.sets0.add(sname0, o.name.upper())
                break

        for o in self.olist0:  # name case-insensitive
            if o not in self.sets0[sname0]:
                self.sets0.add(sname0, o.name.upper())
                self.assertTrue(o in self.sets0[sname0])
                break

        for o in self.olist0:  # ok to add nmobject
            if o not in self.sets0[sname0]:
                self.sets0.add(sname0, o)
                self.assertTrue(o in self.sets0[sname0])
                break

        nlist = [ONLIST0[0], ONLIST0[1], ONLIST0[2]]
        self.sets0[sname0] = nlist  # replace with 3 items
        self.assertEqual(len(self.sets0[sname0]), len(nlist))
        self.sets0.add(sname0, ONLIST0[2])
        self.assertEqual(len(self.sets0[sname0]), len(nlist))
        self.sets0.add(sname0, ONLIST0[3])
        self.assertEqual(len(self.sets0[sname0]), len(nlist) + 1)

        self.sets0.update({sname0: ONLIST0[0], sname0: ONLIST0[1]})
        # repeated key
        self.assertEqual(len(self.sets0[sname0]), 1)
        self.assertEqual(
            self.sets0.get(sname0, get_keys=True), [ONLIST0[1]]
        )  # last item overwrites

        self.sets0.update({sname0: ONLIST0[0], sname1: ONLIST0[1]})

        # change NMObjects and dictionary

        odict0 = {}  # NM object dict
        for n in ONLIST0:
            o = NMObject(parent=NM0, name=n)
            odict0.update({n: o})  # different NMObject with same name
            with self.assertRaises(ValueError):
                self.sets0.add(sname0, o)

        # now change reference of NMObject dict
        self.sets0._NMSets__nmobjects = odict0
        self.sets0[sname0] = []  # empty

        for o in odict0.values():
            self.sets0.add(sname0, o)  # ok now
            self.assertTrue(o in self.sets0[sname0])

        self.sets0[sname0] = []  # empty

        for o in odict0.values():
            self.sets0.add(sname0, [o, o])  # double objects ok
            self.sets0.add(sname0, o)

        d = self.sets1_init[-1]
        sname1 = d["name"].upper()  # should be case-insensitive
        eqlist1 = d["eqlist"]

        self.assertTrue(NMSets.listisequation(eqlist1))

        with self.assertRaises(ValueError):
            self.sets1.add(sname1, "test")  # cannot add to an equation

    def xtest05_delitem(self):
        # args: key
        # see pop()
        for d in self.sets0_init:
            sname = d["name"].upper()
            print("\nanswer YES")
            del self.sets0[sname]
        self.assertEqual(len(self.sets0), 0)

    def xtest06_iter(self):
        s_iter = iter(self.sets0)
        for d in self.sets0_init:
            sname = d["name"]
            s = next(s_iter)
            self.assertEqual(s, sname)

    def xtest07_len(self):
        self.assertEqual(len(self.sets0), len(self.sets0_init))
        self.assertEqual(len(self.sets1), len(self.sets1_init))

    def xtest08_contains(self):
        for d in self.sets0_init:
            sname = d["name"]
            self.assertTrue(sname in self.sets0)
        self.assertFalse("test" in self.sets0)
        for d in self.sets1_init:
            sname = d["name"]
            self.assertTrue(sname in self.sets1)
        self.assertFalse("test" in self.sets1)

    def xtest09_eq(self):
        # args: other

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.sets0 == b)

        self.assertTrue(self.sets0 is self.sets0)
        self.assertFalse(self.sets0 is self.sets1)
        self.assertTrue(self.sets0 == self.sets0)
        self.assertFalse(self.sets0 == self.sets1)
        self.assertFalse(self.sets0 != self.sets0)
        self.assertTrue(self.sets0 != self.sets1)
        self.assertTrue(self.sets0 == self.sets0_copy)
        self.assertTrue(self.sets1 == self.sets1_copy)

        # recreate self.sets0

        odict0 = {}
        for n in ONLIST0:
            o = NMObject(parent=NM0, name=n)
            odict0.update({n: o})

        s0 = NMSets(parent=NM0, name=SETS_NAME0, nmobjects=odict0)

        for d in self.sets0_init:
            sname = d["name"]
            olist = d["olist"]
            for o in olist:
                with self.assertRaises(ValueError):
                    s0.add(sname, o)  # different objects
                s0.add(sname, o.name)  # but name is ok

        self.assertTrue(s0 == self.sets0)

        # name
        s0.name = SETS_NAME1
        self.assertFalse(s0 == self.sets0)
        s0.name = SETS_NAME0
        self.assertTrue(s0 == self.sets0)

        # NMObjects {}
        s0._NMSets__nmobjects = self.odict1
        self.assertTrue(s0 == self.sets0)  # nmobjects {} not compared
        s0._eq_list.append("nmobjects")
        self.assertFalse(s0 == self.sets0)  # nmobjects {} compared
        s0._NMSets__nmobjects = self.odict0
        self.assertTrue(s0 == self.sets0)

        save_fxnref = s0._NMSets__nmobjects_fxnref
        s0._NMSets__nmobjects = {}
        s0._NMSets__nmobjects_fxnref = self._nmobjects_dict1
        self.assertFalse(s0 == self.sets0)
        s0._NMSets__nmobjects_fxnref = self._nmobjects_dict0
        self.assertTrue(s0 == self.sets0)
        s0._NMSets__nmobjects_fxnref = save_fxnref
        s0._NMSets__nmobjects = odict0
        self.assertTrue(s0 == self.sets0)

        # add another NMObject to odict0
        o = NMObject(parent=NM0, name="test")
        odict0.update({"test": o})

        self.assertFalse(s0 == self.sets0)
        s0._eq_list.remove("nmobjects")
        self.assertTrue(s0 == self.sets0)

        # different object types but with same names

        odict0 = {}
        for n in ONLIST0:
            o = NMObject2(parent=NM0, name=n)  # NMObject2
            odict0.update({n: o})

        s0 = NMSets(  # recreate self.sets0
            parent=NM0, name=SETS_NAME0, nmobjects=odict0
        )

        for d in self.sets0_init:
            sname = d["name"]
            olist = d["olist"]
            for o in olist:
                s0.add(sname, o.name)

        self.assertFalse(self.sets0 == s0)

    def xtest10_keys(self):
        klist1 = []
        for d in self.sets0_init:
            klist1.append(d["name"])
        klist2 = list(self.sets0.keys())
        self.assertEqual(klist1, klist2)
        klist2 = list(self.sets0_copy.keys())
        self.assertEqual(klist1, klist2)
        klist1 = []
        for d in self.sets1_init:
            klist1.append(d["name"])
        klist2 = list(self.sets1.keys())
        self.assertEqual(klist1, klist2)
        klist2 = list(self.sets1_copy.keys())
        self.assertEqual(klist1, klist2)

    def xtest11_items(self):
        # see test03_getitem()
        pass

    def xtest12_values(self):
        # see test03_getitem()
        pass

    def xtest13_get(self):
        # see test03_getitem()
        pass

    def xtest14_pop(self):
        # args: key

        d = self.sets0_init[0]
        sname0 = d["name"].upper()  # should be case-insensitive
        olist0 = d["olist"]
        d = self.sets0_init[1]
        sname1 = d["name"].upper()  # should be case-insensitive
        olist1 = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.pop(b)

        with self.assertRaises(KeyError):
            self.sets0.pop("test")  # does not exist

        self.assertTrue(sname0 in self.sets0)
        olist = self.sets0.pop(sname0, confirm_answer="n")
        self.assertTrue(sname0 in self.sets0)
        self.assertIsNone(olist)

        olist = self.sets0.pop(sname0, confirm_answer="y")
        self.assertFalse(sname0 in self.sets0)
        self.assertEqual(olist, olist0)
        olist = self.sets0.get(sname0)
        self.assertIsNone(olist)

        olist = self.sets0.pop(sname1, confirm_answer="y")
        self.assertFalse(sname1 in self.sets0)
        self.assertEqual(olist, olist1)
        olist = self.sets0.get(sname1)
        self.assertIsNone(olist)

    def xtest15_popitem(self):
        # pop last set

        n_sets = len(self.sets0)
        first = True
        for i in range(n_sets):
            j = -1 * (i + 1)

            d = self.sets0_init[j]  # last item
            sname = d["name"]
            olist = d["olist"]

            self.assertTrue(sname in self.sets0)

            if first:
                olist2 = self.sets0.popitem(confirm_answer="n")
                self.assertTrue(sname in self.sets0)
                self.assertEqual(olist2, ())  # empty tuple
                first = False

            olist2 = self.sets0.popitem(confirm_answer="y")
            self.assertFalse(sname in self.sets0)
            self.assertEqual(olist2, (sname, olist))  # tuple

        self.assertEqual(len(self.sets0), 0)

        n_sets = len(self.sets1)
        first = True
        for i in range(n_sets):
            j = -1 * (i + 1)

            d = self.sets1_init[j]  # last item
            sname = d["name"]
            olist = d["olist"]
            eqlist = d["eqlist"]

            self.assertTrue(sname in self.sets1)

            if first:
                olist2 = self.sets1.popitem(confirm_answer="n")
                self.assertTrue(sname in self.sets1)
                self.assertEqual(olist2, ())  # empty tuple
                first = False

            olist2 = self.sets1.popitem(confirm_answer="y")
            self.assertFalse(sname in self.sets1)
            if eqlist:
                self.assertEqual(olist2, (sname, eqlist))  # tuple
            else:
                self.assertEqual(olist2, (sname, olist))  # tuple

        self.assertEqual(len(self.sets1), 0)

    def xtest16_clear(self):
        self.sets0.clear(confirm_answer="n")
        for d in self.sets0_init:
            sname = d["name"].upper()
            self.assertTrue(sname in self.sets0)

        self.sets0.clear(confirm_answer="y")
        for d in self.sets0_init:
            sname = d["name"].upper()
            self.assertFalse(sname in self.sets0)

        self.sets1.clear(confirm_answer="y")
        for d in self.sets1_init:
            sname = d["name"].upper()
            self.assertFalse(sname in self.sets1)

    def xtest17_update(self):
        # see test04_setitem()
        pass

    def xtest18_setdefault(self):
        # args: key, default
        # see get() and __setitem__()
        """
        test = {'one': 1, 'two': 2, 'three': 3}
        print(test)
        print(test.get('one', 'missing'))
        print(test.get('onex', 'missing'))  # does not add new element
        print(test)
        print(test.setdefault('one', 'missing'))
        print(test.setdefault('onex', 'missing'))  # adds new element
        print(test)
        """
        d = self.sets0_init[0]
        sname = d["name"].upper()
        olist = d["olist"]

        olist2 = self.sets0.setdefault(sname)  # like get()
        self.assertEqual(olist, olist2)

        olist2 = self.sets0.setdefault("test")  # like get()
        self.assertIsNone(olist2)
        self.assertFalse("test" in self.sets0)

        olist2 = self.sets0.setdefault("test", [])
        self.assertEqual(olist2, [])
        self.assertTrue("test" in self.sets0)
        olist2 = self.sets0.get("test")
        self.assertEqual(olist2, [])

    def xtest19_getkey(self):
        # args: key, error1, error2

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                key = self.sets0._getkey(b)
            key = self.sets0._getkey(b, error1=False)
            self.assertIsNone(key)

        bad = list(nmu.BADNAMES)
        bad.remove("select")
        for b in bad:
            with self.assertRaises(KeyError):
                key = self.sets0._getkey(b)
            key = self.sets0._getkey(b, error2=False)
            self.assertIsNone(key)

        # test keys are case insensitive
        for d in self.sets0_init:
            sname = d["name"]
            key = self.sets0._getkey(sname.upper())
            self.assertEqual(key, sname)

        sname1 = self.sets0._getkey("select")
        sname2 = self.sets0._NMSets__select_key
        self.assertEqual(sname1, sname2)

    def xtest20_newkey(self):
        # args: newkey

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                key = self.sets0._newkey(b)

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        for b in bad:
            with self.assertRaises(ValueError):
                key = self.sets0._newkey(b)

        for d in self.sets0_init:
            sname = d["name"].upper()
            with self.assertRaises(KeyError):
                key = self.sets0._newkey(sname)
            with self.assertRaises(KeyError):
                key = self.sets0._newkey(sname.upper())
            key = self.sets0._newkey(sname + "x")
            self.assertEqual(key, sname + "x")

        # test key = 'default'
        nn = self.sets0.name_next()
        key = self.sets0._newkey("default")
        self.assertEqual(key, nn)

    def xtest21_namenext(self):
        nn = self.sets0.name_next()
        n = len(self.sets0)
        self.assertEqual(nn, SETS_PREFIX0 + str(n))
        nn = self.sets0.add("default", [])
        nn = self.sets0.name_next()
        self.assertEqual(nn, SETS_PREFIX0 + str(len(self.sets0)))
        nn = self.sets0.name_next(prefix="test")
        self.assertEqual(nn, "test0")

    def xtest22__getnmobjectkey(self):
        # args: nmobject_key, error1, error2

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                key = self.sets0._get_nmobject_key(b)
            key = self.sets0._get_nmobject_key(b, error1=False)
            self.assertIsNone(key)

        bad = list(nmu.BADNAMES)
        for b in bad:
            with self.assertRaises(KeyError):
                key = self.sets0._get_nmobject_key(b)
            key = self.sets0._get_nmobject_key(b, error2=False)
            self.assertIsNone(key)

        # test keys are case insensitive
        for o in self.olist0:
            key = self.sets0._get_nmobject_key(o.name.upper())
            self.assertEqual(key, o.name)

    def xtest23_contains(self):
        # args: key, olist

        d = self.sets0_init[0]
        sname = d["name"].upper()
        nlist = d["nlist"]
        olist = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            self.assertFalse(self.sets0.contains(b, "test"))

        self.assertFalse(self.sets0.contains(sname, "test"))

        for n in nlist:
            self.assertTrue(self.sets0.contains(sname, n.upper()))

        for o in olist:
            self.assertTrue(self.sets0.contains(sname, o))

        d = self.sets1_init[0]
        sname = d["name"].upper()
        nlist = d["nlist"]
        olist = d["olist"]

        for n in nlist:
            self.assertTrue(self.sets1.contains(sname, n.upper()))

        for o in olist:
            self.assertTrue(self.sets1.contains(sname, o))

        d = self.sets1_init[-1]
        sname = d["name"].upper()
        nlist = d["nlist"]
        olist = d["olist"]
        eqlist = d["eqlist"]

        self.assertTrue(NMSets.listisequation(eqlist))

        for n in nlist:
            self.assertTrue(self.sets1.contains(sname, n.upper()))

        for o in olist:
            self.assertTrue(self.sets1.contains(sname, o))

    def xtest24_nmobject_keys_match_order(self):
        # args: nmobject_keys

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0._NMSets__match_to_nmobject_keys(b)

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0._NMSets__match_to_nmobject_keys([b])

        with self.assertRaises(KeyError):
            self.sets0._NMSets__match_to_nmobject_keys(["test"])

        klist = list(self.sets0._nmobjects_dict.keys())

        nlist0 = ONLIST0.copy()
        nlist0.reverse()

        with self.assertRaises(KeyError):  # wrong {}
            nlist1 = self.sets1._NMSets__match_to_nmobject_keys(nlist0)

        nlist1 = self.sets0._NMSets__match_to_nmobject_keys(nlist0)
        self.assertEqual(nlist1, klist)

    def xtest25_isequation(self):
        # args: key

        bad = list(nmu.BADTYPES)
        bad.remove([])
        for b in bad:
            self.assertFalse(self.sets0.isequation(b))

        self.assertFalse(self.sets0.isequation("test"))

        for d in self.sets0_init:
            sname = d["name"]
            self.assertFalse(self.sets0.isequation(sname))

        for d in self.sets1_init:
            sname = d["name"]
            eqlist = d["eqlist"]
            if eqlist:
                self.assertTrue(self.sets1.isequation(sname))
            else:
                self.assertFalse(self.sets1.isequation(sname))

    def xtest26_listisequation(self):
        # args: equation
        # ['&', '|', '-', '^']
        # tests only string format, not whether sets exist

        bad = list(nmu.BADTYPES)
        bad.remove([])
        for b in bad:
            self.assertFalse(NMSets.listisequation(b))

        self.assertFalse(NMSets.listisequation([]))
        self.assertFalse(NMSets.listisequation(["set1", "+", "SET2"]))
        self.assertFalse(NMSets.listisequation(["set1", "|", 222]))

        self.assertFalse(NMSets.listisequation(["set1", "SET2", "|"]))
        self.assertTrue(NMSets.listisequation(["set1", "|", "SET2"]))

        self.assertFalse(NMSets.listisequation(["set1", "|", "SET2", "+", "set3"]))
        self.assertTrue(NMSets.listisequation(["set1", "|", "SET2", "^", "set3"]))
        self.assertFalse(
            NMSets.listisequation(["set1", "|", "SET2", "^", "set3", "set4"])
        )
        self.assertTrue(NMSets.listisequation(["set1", "|", "SET2", "^", "S3", "&"]))

    def xtest27_new(self):
        # args: key

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.new(b)

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        for b in bad:
            with self.assertRaises(ValueError):
                self.sets0.new(b)

        for d in self.sets0_init:
            with self.assertRaises(KeyError):
                self.sets0.new(d["name"])

        for d in self.sets1_init:
            with self.assertRaises(KeyError):
                self.sets1.new(d["name"])

        klist0 = list(self.sets0.keys())
        nn = self.sets0.name_next()
        rtuple = self.sets0.new()
        newkey = rtuple[0]
        new_olist = rtuple[1]
        self.assertEqual(newkey, nn)
        self.assertTrue(newkey in self.sets0)
        klist0.append(newkey)
        self.assertEqual(klist0, list(self.sets0.keys()))
        olist = self.sets0.get(newkey)
        self.assertEqual(olist, [])
        self.assertEqual(new_olist, [])

        self.sets0.add(newkey, [ONLIST0[0], ONLIST0[1]])
        self.assertEqual(new_olist, [self.olist0[0], self.olist0[1]])
        olist = self.sets0.get(newkey)
        self.assertEqual(olist, [self.olist0[0], self.olist0[1]])

        nn = self.sets1.name_next()
        rtuple = self.sets1.new()
        newkey = rtuple[0]
        self.assertEqual(newkey, nn)

    def xtest28_duplicate(self):
        # args: key, newkey

        d = self.sets0_init[0]
        sname0 = d["name"]
        olist0 = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.duplicate(b, "test")

        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.duplicate(sname0, b)

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        for b in bad:
            with self.assertRaises(ValueError):
                self.sets0.duplicate(sname0, b)

        with self.assertRaises(KeyError):
            self.sets0.duplicate(sname0, sname0)

        klist0 = list(self.sets0.keys())
        newkey = sname0 + "_c"
        rtuple = self.sets0.duplicate(sname0, newkey)
        newkey = rtuple[0]
        new_olist = rtuple[1]
        self.assertTrue(newkey in self.sets0)
        self.assertEqual(self.sets0.get(sname0), olist0)
        self.assertEqual(self.sets0.get(newkey), olist0)
        self.assertEqual(new_olist, olist0)
        klist0.append(newkey)
        self.assertEqual(klist0, list(self.sets0.keys()))

        # remove item from new set
        # should not effect original set
        new_olist.pop()
        self.assertNotEqual(self.sets0.get(newkey), olist0)
        self.assertEqual(self.sets0.get(sname0), olist0)

        nn = self.sets0.name_next()
        rtuple = self.sets0.duplicate(sname0)  # default newkey
        newkey = rtuple[0]
        new_olist = rtuple[1]
        self.assertEqual(newkey, nn)
        self.assertTrue(nn in self.sets0)
        self.assertEqual(self.sets0.get(nn), olist0)
        self.assertEqual(new_olist, olist0)

    def xtest29_rename(self):
        # args: key, newkey

        d = self.sets0_init[0]
        sname0 = d["name"]
        olist0 = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.rename(b, "test")

        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.rename(sname0, b)

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        for b in bad:
            with self.assertRaises(ValueError):
                self.sets0.rename(sname0, b)

        with self.assertRaises(KeyError):
            self.sets0.rename(sname0, sname0)

        newkey = sname0 + "_c"
        rname = self.sets0.rename(sname0, newkey)
        self.assertEqual(rname, newkey)
        self.assertFalse(sname0 in self.sets0)
        self.assertTrue(newkey in self.sets0)
        self.assertEqual(self.sets0.get(newkey), olist0)

        nn = self.sets1.name_next()
        rname = self.sets0.rename(newkey)  # default newkey
        self.assertEqual(rname, nn)
        self.assertFalse(newkey in self.sets0)
        self.assertTrue(nn in self.sets0)
        self.assertEqual(self.sets0.get(nn), olist0)

    def xtest30_reorder(self):
        # args: newkeyorder

        bad = list(nmu.BADTYPES)
        bad.remove([])  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.reorder(b)

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.reorder([b])

        with self.assertRaises(KeyError):
            self.sets0.reorder(["test1", "test2"])

        klist0 = list(self.sets0.keys())
        newkeyorder = klist0.copy()
        newkeyorder.reverse()
        self.sets0.reorder(newkeyorder)
        klist0_new = list(self.sets0.keys())
        self.assertEqual(klist0_new, newkeyorder)

        # remove one key
        klist1 = list(self.sets1.keys())
        newkeyorder = klist1.copy()
        newkeyorder.reverse()
        newkeyorder.pop()
        with self.assertRaises(KeyError):
            self.sets1.reorder(newkeyorder)

        # add extra key
        nn = self.sets1.name_next()
        klist1 = list(self.sets1.keys())
        newkeyorder = klist1.copy()
        newkeyorder.append(nn)
        with self.assertRaises(KeyError):
            self.sets1.reorder(newkeyorder)

    def xtest31_empty(self):
        # args: key, confirm

        d = self.sets0_init[0]
        sname0 = d["name"]
        olist0 = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.empty(b)

        with self.assertRaises(KeyError):
            self.sets0.empty("test")

        self.sets0.empty(sname0, confirm_answer="n")
        self.assertTrue(sname0 in self.sets0)
        self.assertEqual(self.sets0.get(sname0), olist0)

        self.sets0.empty(sname0, confirm_answer="y")
        self.assertEqual(self.sets0.get(sname0), [])

        d = self.sets1_init[-1]
        sname1 = d["name"]

        self.sets1.empty(sname1, confirm_answer="y")
        self.assertEqual(self.sets1.get(sname1), [])

    def xtest32_emptyall(self):
        # args: confirm

        self.sets0.empty_all(confirm_answer="n")

        for d in self.sets0_init:
            sname = d["name"]
            olist = d["olist"]
            self.assertEqual(self.sets0.get(sname), olist)

        self.sets0.empty_all(confirm_answer="y")

        for d in self.sets0_init:
            sname = d["name"]
            olist = d["olist"]
            self.assertEqual(self.sets0.get(sname), [])

    def xtest33_add(self):
        # args: key, olist
        # see __setitem__()
        pass

    def xtest34_remove(self):
        # args: key, nmobject_keys

        d = self.sets0_init[0]
        sname0 = d["name"]
        nlist0 = d["nlist"]
        olist0 = d["olist"]

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.remove(b, "")

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.remove(sname0, b)

        with self.assertRaises(KeyError):
            self.sets0.remove("test", "")

        with self.assertRaises(ValueError):
            self.sets0.remove(sname0, "test")  # default error=True

        olist = self.sets0.remove(sname0, "test", error=False)
        self.assertEqual(olist, [])

        self.assertTrue(self.sets0.contains(sname0, nlist0[0].lower()))
        olist = self.sets0.remove(sname0, nlist0[0])
        self.assertEqual(olist, [olist0[0]])
        self.assertFalse(self.sets0.contains(sname0, nlist0[0].lower()))

        self.assertTrue(self.sets0.contains(sname0, olist0[1]))
        olist = self.sets0.remove(sname0, olist0[1])
        self.assertEqual(olist, [olist0[1]])
        self.assertFalse(self.sets0.contains(sname0, olist0[1]))

        self.assertTrue(self.sets0.contains(sname0, olist0[2]))
        self.assertTrue(self.sets0.contains(sname0, olist0[3]))
        olist = self.sets0.remove(sname0, [nlist0[2], nlist0[3]])
        self.assertEqual(olist, [olist0[2], olist0[3]])
        self.assertFalse(self.sets0.contains(sname0, olist0[2]))
        self.assertFalse(self.sets0.contains(sname0, olist0[3]))

        d = self.sets1_init[-1]
        sname1 = d["name"]
        nlist1 = d["nlist"]
        olist1 = d["olist"]
        eqlist1 = d["eqlist"]

        self.assertTrue(NMSets.listisequation(eqlist1))
        self.assertTrue(self.sets1.contains(sname1, nlist1[0]))

        with self.assertRaises(ValueError):
            olist = self.sets1.remove(sname1, nlist1[0])

        with self.assertRaises(ValueError):
            olist = self.sets1.remove(sname1, olist1[0])

    def xtest35_removefromall(self):
        # args: nmobject_keys

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        bad.remove([])  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0.remove_from_all(b)

        with self.assertRaises(ValueError):
            self.sets0.remove_from_all("test", error=True)

        self.sets0.remove_from_all("test")  # default error=False

        nlist = [ONLIST0[0], ONLIST0[1], ONLIST0[2], ONLIST0[3]]

        for d in self.sets0_init:
            sname = d["name"]
            self.sets0.add(sname, nlist)
            self.assertTrue(self.sets0.contains(sname, nlist))

        self.sets0.remove_from_all(nlist)

        for d in self.sets0_init:
            sname = d["name"]
            self.assertFalse(self.sets0.contains(sname, nlist))

        nlist = [ONLIST1[0], ONLIST1[1], ONLIST1[2], ONLIST1[3]]

        for d in self.sets1_init:
            sname = d["name"]
            eqlist = d["eqlist"]
            if not eqlist:
                self.sets1.add(sname, nlist)
                self.assertTrue(self.sets1.contains(sname, nlist))

        self.sets1.remove_from_all(nlist)

        for d in self.sets1_init:
            sname = d["name"]
            self.assertFalse(self.sets1.contains(sname, nlist))

    def xtest36_select(self):
        # args: key

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        bad.remove(None)  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.sets0._select_key_set(b)
            self.assertFalse(self.sets0.is_select_key(b))

        with self.assertRaises(KeyError):
            self.sets0._select_key_set("SELECT")
        self.assertFalse(self.sets0.is_select_key("SELECT"))

        with self.assertRaises(KeyError):
            self.sets0._select_key_set("test")
        self.assertFalse(self.sets0.is_select_key("test"))

        d = self.sets0_init[-1]
        sname = d["name"]
        nlist = d["nlist"]
        olist = d["olist"]
        eqlist = d["eqlist"]
        self.assertEqual(sname, self.sets0.select_key)
        self.assertTrue(self.sets0.is_select_key(sname))

        olist2 = self.sets0.get("select")
        self.assertEqual(olist, olist2)
        nlist2 = self.sets0.get("select", get_keys=True)
        self.assertEqual(nlist, nlist2)
        eqlist2 = self.sets0.get("select", get_equation=True)
        self.assertEqual(olist, eqlist2)

        self.sets0.select_key = None
        self.assertIsNone(self.sets0.select_key)
        olist2 = self.sets0.get("select")
        self.assertIsNone(olist2)
        nlist2 = self.sets0.get("select", get_keys=True)
        self.assertIsNone(nlist2)
        eqlist2 = self.sets0.get("select", get_equation=True)
        self.assertIsNone(eqlist2)
        self.assertTrue(self.sets0.is_select_key(None))

        self.sets0.select_key = sname
        self.assertEqual(self.sets0.select_key, sname)
        self.assertTrue(self.sets0.is_select_key(sname))

        for i, n in enumerate(ONLIST0):
            if n not in nlist:
                o = self.olist0[i]
                self.assertFalse(n in self.sets0.get(sname, get_keys=True))
                self.assertFalse(n in self.sets0.get("select", get_keys=True))
                self.assertFalse(o in self.sets0.get(sname))
                self.assertFalse(o in self.sets0.get("select"))
                self.sets0.add(sname, n)
                self.assertFalse(n in nlist)
                self.assertTrue(n in self.sets0.get(sname, get_keys=True))
                self.assertTrue(n in self.sets0.get("select", get_keys=True))
                self.assertTrue(o in self.sets0.get(sname))
                self.assertTrue(o in self.sets0.get("select"))

        d = self.sets1_init[-1]
        sname = d["name"]
        nlist = d["nlist"]
        olist = d["olist"]
        eqlist = d["eqlist"]
        self.assertEqual(sname, self.sets1.select_key)
        self.assertTrue(self.sets1.is_select_key(sname))

        olist2 = self.sets1.get("select")
        self.assertEqual(olist, olist2)
        nlist2 = self.sets1.get("select", get_keys=True)
        self.assertEqual(nlist, nlist2)
        eqlist2 = self.sets1.get("select", get_equation=True)
        self.assertEqual(eqlist, eqlist2)

        self.sets1.pop(sname, confirm_answer="y")
        self.assertIsNone(self.sets1.select_key)

    def xtest37_equation(self):
        # args: key, eq_list
        # see __setitem__()
        # ['&', '|', '-', '^']

        d = self.sets0_init[0]
        sname0 = d["name"]

        with self.assertRaises(TypeError):
            self.sets0.add(sname0, [1, 2, "set1"])

        s0 = set(self.sets0.get(sname0, get_keys=True))
        d = self.sets0_init[1]
        sname1 = d["name"]
        s1 = set(self.sets0.get(sname1, get_keys=True))

        self.assertFalse(self.sets0.isequation(sname0.upper()))
        self.assertFalse(self.sets0.isequation(sname1.upper()))

        nlist = [ONLIST0[0], ONLIST0[4], ONLIST0[7], ONLIST0[11]]
        s2 = set(nlist)
        sname2 = self.sets0.name_next()
        self.sets0.add(sname2, nlist)

        sname3 = self.sets0.name_next()
        eqlist = [sname0, "%", sname1]
        with self.assertRaises(KeyError):
            self.sets0.update({sname3: eqlist})  # bad equation

        eqlist = [sname0, "|", sname1]
        with self.assertRaises(ValueError):
            self.sets0.update({sname0: eqlist})  # already exists

        self.assertFalse(sname3 in self.sets0)
        self.sets0.update({sname3: eqlist})
        self.assertTrue(sname3.upper() in self.sets0)
        self.assertTrue(self.sets0.isequation(sname3.upper()))
        eqlist2 = self.sets0.get(sname3.upper(), get_equation=True)
        self.assertEqual(eqlist, eqlist2)
        s3 = set(self.sets0.get(sname3, get_keys=True))
        self.assertEqual(s3, s0 | s1)
        olist = self.sets0.get(sname3)
        for o in olist:
            self.assertTrue(o.name in s3)

        self.sets0.update({sname3: eqlist})
        self.sets0.empty(sname3, confirm_answer="y")
        self.assertEqual(self.sets0.get(sname3), [])

        eqlist = [sname0, "&", sname1]
        self.sets0.update({sname3: eqlist})
        eqlist2 = self.sets0.get(sname3.upper(), get_equation=True)
        self.assertEqual(eqlist, eqlist2)
        s3 = set(self.sets0.get(sname3, get_keys=True))
        self.assertEqual(s3, s0 & s1)
        olist = self.sets0.get(sname3)
        for o in olist:
            self.assertTrue(o.name in s3)

        eqlist = [sname0, "-", sname1]
        self.sets0.update({sname3: eqlist})
        eqlist2 = self.sets0.get(sname3.upper(), get_equation=True)
        self.assertEqual(eqlist, eqlist2)
        s3 = set(self.sets0.get(sname3, get_keys=True))
        self.assertEqual(s3, s0 - s1)
        olist = self.sets0.get(sname3)
        for o in olist:
            self.assertTrue(o.name in s3)

        eqlist = [sname0, "^", sname1]
        self.sets0.update({sname3: eqlist})
        eqlist2 = self.sets0.get(sname3.upper(), get_equation=True)
        self.assertEqual(eqlist, eqlist2)
        s3 = set(self.sets0.get(sname3, get_keys=True))
        self.assertEqual(s3, s0 ^ s1)
        olist = self.sets0.get(sname3)
        for o in olist:
            self.assertTrue(o.name in s3)

        eqlist = [sname0, "|", sname1, "^", sname2]
        self.sets0.update({sname3: eqlist})
        eqlist2 = self.sets0.get(sname3.upper(), get_equation=True)
        self.assertEqual(eqlist, eqlist2)
        s3 = set(self.sets0.get(sname3, get_keys=True))
        self.assertEqual(s3, (s0 | s1) ^ s2)
        olist = self.sets0.get(sname3)
        for o in olist:
            self.assertTrue(o.name in s3)

        rtuple = self.sets0.duplicate(sname3)
        sname4 = rtuple[0]
        s4 = set(self.sets0.get(sname4, get_keys=True))
        self.assertEqual(s3, s4)
        eqlist2 = self.sets0.get(sname4, get_equation=True)
        self.assertEqual(eqlist2, eqlist)
        self.assertEqual(self.sets0.get(sname3), self.sets0.get(sname4))
