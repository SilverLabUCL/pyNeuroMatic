#!/usr/bin/env python[3]
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:23:07 2019

@author: jason
"""
import unittest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)

NM0 = NMManager(quiet=QUIET)
NM1 = NMManager(quiet=QUIET)
ONAME0 = "object0"
ONAME1 = "object1"


class NMObjectTest(unittest.TestCase):
    def setUp(self):
        self.o0 = NMObject(parent=NM0, name=ONAME0)
        self.o1 = NMObject(parent=NM1, name=ONAME1)
        self.o0_copy = NMObject(copy=self.o0)
        self.o1_copy = NMObject(parent=NM, name="object1_copy", copy=self.o1)
        # print(self.o0.__dict__)

    # def tearDown(self):
    #    pass

    def test00_init(self):
        # args: parent, not testing since it can be any object
        # args: name

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObject(name=b)

        bad = list(nmu.BADNAMES)
        for b in bad:
            with self.assertRaises(ValueError):
                NMObject(name=b)

        bad = list(nmu.BADTYPES)
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObject(copy=b)

        self.assertEqual(self.o0._parent, NM0)
        self.assertEqual(self.o0.name, ONAME0)
        self.assertEqual(self.o0._NMObject__rename_fxnref, self.o0._name_set)

        self.assertEqual(self.o1._parent, NM1)
        self.assertEqual(self.o1.name, ONAME1)
        self.assertEqual(self.o1._NMObject__rename_fxnref, self.o1._name_set)

        self.assertEqual(self.o0_copy._parent, NM0)
        self.assertEqual(self.o0_copy.name, ONAME0)
        self.assertEqual(self.o0_copy._NMObject__rename_fxnref, self.o0_copy._name_set)

        self.assertEqual(self.o1_copy._parent, NM1)  # copy overrides
        self.assertEqual(self.o1_copy.name, ONAME1)  # copy overrides
        self.assertEqual(self.o1_copy._NMObject__rename_fxnref, self.o1_copy._name_set)

    def test01_eq(self):
        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.o0 == b)

        self.assertTrue(self.o0 is self.o0)
        self.assertFalse(self.o0 is self.o1)
        self.assertTrue(self.o0 == self.o0)
        self.assertFalse(self.o0 != self.o0)
        self.assertFalse(self.o0 == self.o1)
        self.assertTrue(self.o0 != self.o1)
        self.assertTrue(self.o0 == self.o0_copy)
        self.assertTrue(self.o1 == self.o1_copy)

        nmo0 = NMObject(parent=NM0, name=ONAME0)
        nmo1 = NMObject(parent=NM0, name=ONAME0)
        self.assertTrue(nmo0 == nmo1)
        o0 = NMObject2(parent=NM0, name=ONAME0)  # NMObject2
        self.assertFalse(o0 == nmo1)
        o1 = NMObject2(parent=None, name=ONAME0)

        self.assertTrue(o0 == o1)  # parent not tested
        o0._eq_list.append("parent")
        self.assertFalse(o0 == o1)  # parent tested
        o1._parent = NM1
        self.assertTrue(o0 == o1)  # parents are same type

        o0.myvalue = 1
        o1.myvalue = 2
        self.assertFalse(o0 == o1)
        o0.myvalue = 2
        self.assertTrue(o0 == o1)
        o0.myvalue = float("nan")
        o1.myvalue = float("nan")
        self.assertFalse(o0 == o1)  # NAN != NAN
        o0.myvalue = 2
        o1.myvalue = 2
        self.assertTrue(o0 == o1)

        o0.note = "my note"
        o1.note = "my note"
        self.assertTrue(o0 == o1)  # notes not tested
        o0._eq_list.append("notes")
        self.assertFalse(o0 == o1)  # notes tested, different time stamps
        o0._notes_delete(confirm_answer="y")
        o1._notes_delete(confirm_answer="y")
        self.assertTrue(o0 == o1)
        o0.note = "my note 0"
        o0.note = "my note 1"
        for n in o0.notes:
            o1._NMObject__notes.append(dict(n))  # notes have same time stamp
        self.assertTrue(o0 == o1)

    def test02_lists_are_equal(self):
        olist0 = []
        for i in range(10):
            o = NMObject(parent=NM0, name="test" + str(i))
            olist0.append(o)
        olist1 = []

        for i in range(10):
            o = NMObject(parent=NM0, name="test" + str(i))
            olist1.append(o)
        self.assertTrue(NMObject.lists_are_equal(olist0, olist1))

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(NMObject.lists_are_equal(b, olist1))

        for i in range(10):
            o = NMObject2(parent=NM0, name="test" + str(i))  # NMObject2
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

        for i in range(9):
            o = NMObject(parent=NM0, name="test" + str(i))
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

        for i in range(11):
            o = NMObject(parent=NM0, name="test" + str(i))
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

        for i in range(10):
            if i == 7:
                o = NMObject2(parent=NM0, name="test" + str(i))
            else:
                o = NMObject(parent=NM0, name="test" + str(i))
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

        self.assertTrue(NMObject.lists_are_equal(None, None))

        olist0 = []
        for i in range(10):
            o = NMObject2(parent=NM0, name="test" + str(i))
            olist0.append(o)
        olist1 = []

        for i in range(10):
            o = NMObject2(parent=NM0, name="test" + str(i))
            olist1.append(o)
        self.assertTrue(NMObject.lists_are_equal(olist0, olist1))

        for i in range(10):
            o = NMObject2(parent=NM0, name="test" + str(i))
            o.myvalue = i
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

    def test03_copy(self):
        c = self.o0.copy()
        self.assertIsInstance(c, NMObject)
        self.assertTrue(self.o0 == c)
        self.assertEqual(self.o0._parent, c._parent)
        self.assertEqual(self.o0.name, c.name)
        p0 = self.o0.parameters
        p = c.parameters
        self.assertNotEqual(p0.get("created"), p.get("created"))
        self.assertEqual(c._NMObject__rename_fxnref, c._name_set)
        fr0 = self.o0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        # Compare underlying functions, not bound methods
        # (bound methods from different instances are never equal)
        self.assertEqual(fr0.__func__, frc.__func__)

    def test04_parameters(self):
        plist = self.o0.parameters
        plist2 = ["name", "created", "copy of"]
        self.assertEqual(list(plist.keys()), plist2)
        self.assertEqual(plist["name"], ONAME0)
        self.assertIsNone(plist["copy of"])

    def test05_content(self):
        self.assertEqual(self.o0.content, {"nmobject": self.o0.name})
        ct = {"nmmanager": "nm", "nmobject": self.o0.name}
        self.assertEqual(self.o0.content_tree, ct)

    def test06_treepath(self):
        tp = NM0.name + "." + self.o0.name
        self.assertEqual(self.o0._treepath_str(), tp)
        tp = [NM0.name, self.o0.name]
        self.assertEqual(self.o0.treepath(), tp)

        o2 = NMObject2(parent=NM0, name=ONAME0)
        tp = NM0.name + "." + ONAME0 + ".myobject"
        self.assertEqual(o2.myobject._treepath_str(), tp)
        tp = [NM0.name, ONAME0, "myobject"]
        self.assertEqual(o2.myobject.treepath(), tp)

    def test07_notes(self):
        self.assertTrue(self.o0.notes_on)
        self.o0.note = "added TTX"
        self.assertTrue(self.o0._notes_append("added AP5"))
        self.o0.notes_on = None
        self.assertTrue(self.o0.notes_on)
        self.o0.notes_on = True
        self.assertTrue(self.o0.notes_on)
        self.o0.notes_on = False
        self.assertFalse(self.o0.notes_on)
        self.assertFalse(self.o0._notes_append("added NBQX"))
        self.assertTrue(isinstance(self.o0.notes, list))
        self.assertEqual(len(self.o0.notes), 2)
        self.assertEqual(self.o0.notes[0].get("note"), "added TTX")
        self.assertEqual(self.o0.notes[1].get("note"), "added AP5")
        self.o0._notes_delete(confirm_answer="y")
        self.assertEqual(len(self.o0.notes), 0)
        self.assertTrue(NMObject.notes_ok([{"note": "hey", "date": "111"}]))
        self.assertTrue(NMObject.notes_ok([{"date": "111", "note": "hey"}]))
        self.assertFalse(NMObject.notes_ok([{"n": "hey", "date": "111"}]))
        self.assertFalse(NMObject.notes_ok([{"note": "hey", "d": "111"}]))
        self.assertFalse(NMObject.notes_ok([{"note": "hey", "date": None}]))
        self.assertTrue(NMObject.notes_ok([{"note": "hey", "date": "None"}]))
        self.assertFalse(NMObject.notes_ok([{"note": "hey"}]))
        self.assertFalse(NMObject.notes_ok([{"date": "111"}]))
        self.assertFalse(
            NMObject.notes_ok([{"note": "hey", "date": "111", "more": "1"}])
        )

    def test08_name_set(self):
        # args: name_notused
        # args: newname

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.o0._name_set("notused", b)

        bad = list(nmu.BADNAMES)
        for b in bad:
            with self.assertRaises(ValueError):
                self.o0._name_set("notused", b)

        for n in ["test", ONAME0]:
            self.o0._name_set("notused", n)
            self.assertEqual(n, self.o0.name)

    def test09_rename_fxnref_set(self):
        # args: rename_fxnref

        bad = list(nmu.BADTYPES)
        for b in bad:
            with self.assertRaises(TypeError):
                self.o0._rename_fxnref_set(b)

        self.o0.name = "test1"  # calls _name_set()
        self.assertEqual(self.o0.name, "test1")
        self.o0._rename_fxnref_set(self.rename_dummy)
        self.o0.name = "test2"
        self.assertEqual(self.o0.name, "test1")  # name of o0 does not change

    def rename_dummy(self, oldname, newname, quiet=nmp.QUIET):
        # dummy function to test NMObject._rename_fxnref_set()
        print("test rename: " + oldname + " -> " + newname)
        return False

    def test10_manager(self):
        self.assertEqual(self.o0._manager, NM0)

    def test11_error(self):
        pass
        # alert(), error(), history()
        # wrappers for nmu.history()
        # args: obj, type_expected, tp, quiet, frame
        # dum_arg = {}
        # e1 = self.o0._type_error('dum_arg', 'list')
        # tp = 'object0'
        # tp = 'nm.NMObjectTest.test11_error'
        # e2 = ('ERROR: ' + tp + ': bad dum_args: expected list but got dict')
        # self.assertEqual(e1, e2)
        # dum_str = 'test'
        # e1 = self.o0._value_error('dum_str')
        # e2 = ("ERROR: " + tp + ": bad dum_str: 'test'")
        # self.assertEqual(e1, e2)

    def test12_quiet(self):
        # args: quiet
        # TODO
        """
        NM0.configs.quiet = False
        self.assertFalse(self.o0._quiet(False))
        self.assertTrue(self.o0._quiet(True))
        NM0.configs.quiet = True  # Manager quiet overrides when True
        self.assertTrue(self.o0._quiet(False))
        self.assertTrue(self.o0._quiet(True))
        NM0.configs.quiet = False
        """

    def test_save(self):
        # TODO
        pass


class NMObject2(NMObject):
    def __init__(self, parent, name):
        super().__init__(parent=parent, name=name)
        self.myvalue = 1
        self.myobject = NMObject(parent=self, name="myobject")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMObject2):
            return False
        if not super().__eq__(other):
            return False
        if self.myvalue != other.myvalue:
            # if math.isnan(self.myvalue) and math.isnan(other.myvalue):
            #    return True
            return False
        return True


if __name__ == '__main__':
    unittest.main()
