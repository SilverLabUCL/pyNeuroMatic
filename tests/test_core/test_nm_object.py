#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMObject.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject

QUIET = True
NM0 = NMManager(quiet=QUIET)
NM1 = NMManager(quiet=QUIET)
ONAME0 = "object0"
ONAME1 = "object1"

# Common bad types for testing type validation
BAD_TYPES = (None, 3, 3.14, True, [], (), {}, set())
BAD_NAMES = ("", " ", "  ", "\t", "\n")


class NMObject2(NMObject):
    """Test subclass of NMObject with additional attributes."""

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
            return False
        return True


class NMObjectTestBase(unittest.TestCase):
    """Base class with shared setUp for NMObject tests."""

    def setUp(self):
        self.o0 = NMObject(parent=NM0, name=ONAME0)
        self.o1 = NMObject(parent=NM1, name=ONAME1)
        self.o0_copy = copy.deepcopy(self.o0)
        self.o1_copy = copy.deepcopy(self.o1)


class TestNMObjectInit(NMObjectTestBase):
    """Tests for NMObject initialization."""

    def test_rejects_non_string_name(self):
        bad = list(BAD_TYPES)
        bad.remove(None)  # None is allowed
        for b in bad:
            with self.assertRaises(TypeError):
                NMObject(name=b)

    def test_rejects_invalid_name(self):
        for b in BAD_NAMES:
            with self.assertRaises(ValueError):
                NMObject(name=b)

    def test_stores_parent(self):
        self.assertEqual(self.o0._parent, NM0)
        self.assertEqual(self.o1._parent, NM1)

    def test_stores_name(self):
        self.assertEqual(self.o0.name, ONAME0)
        self.assertEqual(self.o1.name, ONAME1)

    def test_sets_rename_fxnref(self):
        self.assertEqual(self.o0._NMObject__rename_fxnref, self.o0._name_set)
        self.assertEqual(self.o1._NMObject__rename_fxnref, self.o1._name_set)

    def test_deepcopy_preserves_parent(self):
        self.assertEqual(self.o0_copy._parent, NM0)
        self.assertEqual(self.o1_copy._parent, NM1)

    def test_deepcopy_preserves_name(self):
        self.assertEqual(self.o0_copy.name, ONAME0)
        self.assertEqual(self.o1_copy.name, ONAME1)

    def test_deepcopy_preserves_rename_fxnref(self):
        self.assertEqual(self.o0_copy._NMObject__rename_fxnref, self.o0_copy._name_set)
        self.assertEqual(self.o1_copy._NMObject__rename_fxnref, self.o1_copy._name_set)


class TestNMObjectEquality(NMObjectTestBase):
    """Tests for NMObject equality comparison."""

    def test_not_equal_to_bad_types(self):
        for b in BAD_TYPES:
            self.assertFalse(self.o0 == b)

    def test_is_same_object(self):
        self.assertTrue(self.o0 is self.o0)
        self.assertFalse(self.o0 is self.o1)

    def test_equal_to_self(self):
        self.assertTrue(self.o0 == self.o0)
        self.assertFalse(self.o0 != self.o0)

    def test_not_equal_to_different_object(self):
        self.assertFalse(self.o0 == self.o1)
        self.assertTrue(self.o0 != self.o1)

    def test_equal_to_deepcopy(self):
        self.assertTrue(self.o0 == self.o0_copy)
        self.assertTrue(self.o1 == self.o1_copy)

    def test_equal_same_name_and_parent(self):
        nmo0 = NMObject(parent=NM0, name=ONAME0)
        nmo1 = NMObject(parent=NM0, name=ONAME0)
        self.assertTrue(nmo0 == nmo1)

    def test_not_equal_different_type(self):
        nmo = NMObject(parent=NM0, name=ONAME0)
        o2 = NMObject2(parent=NM0, name=ONAME0)
        self.assertFalse(o2 == nmo)

    def test_subclass_equality_ignores_parent(self):
        o0 = NMObject2(parent=NM0, name=ONAME0)
        o1 = NMObject2(parent=None, name=ONAME0)
        self.assertTrue(o0 == o1)

    def test_subclass_equality_checks_attributes(self):
        o0 = NMObject2(parent=NM0, name=ONAME0)
        o1 = NMObject2(parent=NM0, name=ONAME0)
        o0.myvalue = 1
        o1.myvalue = 2
        self.assertFalse(o0 == o1)
        o0.myvalue = 2
        self.assertTrue(o0 == o1)

    def test_nan_not_equal_to_nan(self):
        o0 = NMObject2(parent=NM0, name=ONAME0)
        o1 = NMObject2(parent=NM0, name=ONAME0)
        o0.myvalue = float("nan")
        o1.myvalue = float("nan")
        self.assertFalse(o0 == o1)


class TestNMObjectListsAreEqual(NMObjectTestBase):
    """Tests for NMObject.lists_are_equal static method."""

    def test_equal_lists(self):
        olist0 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(10)]
        olist1 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(10)]
        self.assertTrue(NMObject.lists_are_equal(olist0, olist1))

    def test_rejects_bad_types(self):
        olist1 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(10)]
        for b in BAD_TYPES:
            self.assertFalse(NMObject.lists_are_equal(b, olist1))

    def test_different_lengths(self):
        olist0 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(10)]
        olist1 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(9)]
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

    def test_different_types_in_list(self):
        olist0 = [NMObject(parent=NM0, name="test" + str(i)) for i in range(10)]
        olist1 = []
        for i in range(10):
            if i == 7:
                olist1.append(NMObject2(parent=NM0, name="test" + str(i)))
            else:
                olist1.append(NMObject(parent=NM0, name="test" + str(i)))
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))

    def test_both_none(self):
        self.assertTrue(NMObject.lists_are_equal(None, None))

    def test_subclass_lists_equal(self):
        olist0 = [NMObject2(parent=NM0, name="test" + str(i)) for i in range(10)]
        olist1 = [NMObject2(parent=NM0, name="test" + str(i)) for i in range(10)]
        self.assertTrue(NMObject.lists_are_equal(olist0, olist1))

    def test_subclass_lists_different_values(self):
        olist0 = [NMObject2(parent=NM0, name="test" + str(i)) for i in range(10)]
        olist1 = []
        for i in range(10):
            o = NMObject2(parent=NM0, name="test" + str(i))
            o.myvalue = i
            olist1.append(o)
        self.assertFalse(NMObject.lists_are_equal(olist0, olist1))


class TestNMObjectCopy(NMObjectTestBase):
    """Tests for NMObject.copy method."""

    def test_returns_nmobject(self):
        c = self.o0.copy()
        self.assertIsInstance(c, NMObject)

    def test_copy_equals_original(self):
        c = self.o0.copy()
        self.assertTrue(self.o0 == c)

    def test_copy_has_same_parent(self):
        c = self.o0.copy()
        self.assertEqual(self.o0._parent, c._parent)

    def test_copy_has_same_name(self):
        c = self.o0.copy()
        self.assertEqual(self.o0.name, c.name)

    def test_copy_has_created_timestamp(self):
        c = self.o0.copy()
        self.assertIsNotNone(c.parameters.get("created"))

    def test_copy_has_rename_fxnref(self):
        c = self.o0.copy()
        self.assertEqual(c._NMObject__rename_fxnref, c._name_set)

    def test_copy_rename_fxnref_same_function(self):
        c = self.o0.copy()
        fr0 = self.o0._NMObject__rename_fxnref
        frc = c._NMObject__rename_fxnref
        self.assertEqual(fr0.__func__, frc.__func__)


class TestNMObjectParameters(NMObjectTestBase):
    """Tests for NMObject.parameters property."""

    def test_has_expected_keys(self):
        plist = self.o0.parameters
        expected_keys = ["name", "created", "copy of"]
        self.assertEqual(list(plist.keys()), expected_keys)

    def test_name_matches(self):
        plist = self.o0.parameters
        self.assertEqual(plist["name"], ONAME0)

    def test_copy_of_is_none(self):
        plist = self.o0.parameters
        self.assertIsNone(plist["copy of"])


class TestNMObjectContent(NMObjectTestBase):
    """Tests for NMObject.content and content_tree properties."""

    def test_content(self):
        self.assertEqual(self.o0.content, {"nmobject": self.o0.name})

    def test_content_tree(self):
        ct = {"nmobject": self.o0.name}
        self.assertEqual(self.o0.content_tree, ct)


class TestNMObjectPath(NMObjectTestBase):
    """Tests for NMObject path properties."""

    def test_path_list(self):
        expected_path = [self.o0.name]
        self.assertEqual(self.o0.path, expected_path)

    def test_path_str(self):
        expected_str = self.o0.name
        self.assertEqual(self.o0.path_str, expected_str)

    def test_path_objects(self):
        path_objs = self.o0.path_objects
        self.assertEqual(len(path_objs), 1)
        self.assertIs(path_objs[0], self.o0)

    def test_nested_object_path(self):
        o2 = NMObject2(parent=NM0, name=ONAME0)
        expected_path = [ONAME0, "myobject"]
        self.assertEqual(o2.myobject.path, expected_path)

    def test_nested_object_path_str(self):
        o2 = NMObject2(parent=NM0, name=ONAME0)
        expected_str = ONAME0 + ".myobject"
        self.assertEqual(o2.myobject.path_str, expected_str)


class TestNMObjectNameSet(NMObjectTestBase):
    """Tests for NMObject._name_set method."""

    def test_rejects_non_string(self):
        bad = list(BAD_TYPES)
        bad.remove(None)  # None might be handled differently
        for b in bad:
            with self.assertRaises(TypeError):
                self.o0._name_set("notused", b)

    def test_rejects_invalid_name(self):
        for b in BAD_NAMES:
            with self.assertRaises(ValueError):
                self.o0._name_set("notused", b)

    def test_sets_name(self):
        for n in ["test", ONAME0]:
            self.o0._name_set("notused", n)
            self.assertEqual(n, self.o0.name)


class TestNMObjectRenameFxnref(NMObjectTestBase):
    """Tests for NMObject._rename_fxnref_set method."""

    def rename_dummy(self, oldname, newname, quiet=False):
        """Dummy function that rejects all renames."""
        return False

    def test_rejects_bad_types(self):
        for b in BAD_TYPES:
            with self.assertRaises(TypeError):
                self.o0._rename_fxnref_set(b)

    def test_default_rename_works(self):
        self.o0.name = "test1"
        self.assertEqual(self.o0.name, "test1")

    def test_custom_rename_fxnref(self):
        self.o0.name = "test1"
        self.assertEqual(self.o0.name, "test1")
        self.o0._rename_fxnref_set(self.rename_dummy)
        self.o0.name = "test2"
        self.assertEqual(self.o0.name, "test1")  # Name unchanged due to dummy


class TestNMObjectManager(NMObjectTestBase):
    """Tests for NMObject._manager property."""

    def test_returns_manager(self):
        self.assertEqual(self.o0._manager, NM0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
