#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMSets.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_sets import NMSets, EQUATION_OPERATORS
import pyneuromatic.core.nm_utilities as nmu

QUIET = True


class NMSetsTestBase(unittest.TestCase):
    """Base class with common setup for NMSets tests."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        # Create NMObjects for testing
        self.obj_names = [f"obj{i}" for i in range(10)]
        self.objects = {}
        for name in self.obj_names:
            obj = NMObject(parent=self.nm, name=name)
            self.objects[name] = obj

        self.sets = NMSets(
            name="TestSets",
            nmobjects=self.objects,
        )


class TestNMSetsInit(NMSetsTestBase):
    """Tests for NMSets initialization."""

    def test_init_with_nmobjects(self):
        sets = NMSets(name="test", nmobjects=self.objects)
        self.assertEqual(sets.name, "test")

    def _get_objects_method(self):
        """Method reference for testing nmobjects_fxnref."""
        return self.objects

    def test_init_with_nmobjects_fxnref(self):
        sets = NMSets(name="test", nmobjects_fxnref=self._get_objects_method)
        self.assertEqual(sets.name, "test")

    def test_init_requires_nmobjects_arg(self):
        with self.assertRaises(ValueError):
            NMSets(name="test")

    def test_init_rejects_both_args(self):
        with self.assertRaises(ValueError):
            NMSets(nmobjects_fxnref=self._get_objects_method, nmobjects=self.objects)

    def test_init_rejects_invalid_nmobjects(self):
        bad_types = [None, 3, 3.14, True, [], (), "string"]
        for b in bad_types:
            with self.assertRaises(ValueError):
                NMSets(nmobjects=b)


class TestNMSetsBasicOperations(NMSetsTestBase):
    """Tests for basic set operations."""

    def test_add_single_item(self):
        self.sets.add("set0", "obj0")
        self.assertIn("obj0", self.sets.get("set0", get_keys=True))

    def test_add_multiple_items(self):
        self.sets.add("set0", ["obj0", "obj1", "obj2"])
        keys = self.sets.get("set0", get_keys=True)
        self.assertIn("obj0", keys)
        self.assertIn("obj1", keys)
        self.assertIn("obj2", keys)

    def test_add_nmobject_directly(self):
        self.sets.add("set0", self.objects["obj0"])
        self.assertIn(self.objects["obj0"], self.sets.get("set0"))

    def test_get_returns_nmobjects(self):
        self.sets.add("set0", ["obj0", "obj1"])
        result = self.sets.get("set0")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(o, NMObject) for o in result))

    def test_get_keys_returns_strings(self):
        self.sets.add("set0", ["obj0", "obj1"])
        result = self.sets.get("set0", get_keys=True)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(k, str) for k in result))

    def test_get_nonexistent_returns_default(self):
        result = self.sets.get("nonexistent")
        self.assertIsNone(result)
        result = self.sets.get("nonexistent", default="default")
        self.assertEqual(result, "default")

    def test_contains(self):
        self.sets.add("set0", ["obj0", "obj1"])
        self.assertTrue("set0" in self.sets)
        self.assertFalse("nonexistent" in self.sets)

    def test_len(self):
        self.assertEqual(len(self.sets), 0)
        self.sets.add("set0", [])
        self.assertEqual(len(self.sets), 1)
        self.sets.add("set1", [])
        self.assertEqual(len(self.sets), 2)

    def test_keys(self):
        self.sets.add("set0", [])
        self.sets.add("set1", [])
        keys = list(self.sets.keys())
        self.assertEqual(keys, ["set0", "set1"])

    def test_pop(self):
        self.sets.add("set0", ["obj0"])
        result = self.sets.pop("set0")
        self.assertFalse("set0" in self.sets)
        self.assertEqual(len(result), 1)

    def test_pop_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        result = self.sets.pop("setC")
        self.assertFalse("setC" in self.sets)
        # Should return the equation tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "and")

    def test_popitem(self):
        self.sets.add("set0", ["obj0"])
        self.sets.add("set1", ["obj1"])
        key, value = self.sets.popitem()
        self.assertEqual(key, "set1")  # Last item
        self.assertFalse("set1" in self.sets)

    def test_popitem_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        key, value = self.sets.popitem()
        self.assertEqual(key, "setC")  # Last item
        # Should return the equation tuple
        self.assertIsInstance(value, tuple)
        self.assertEqual(value[0], "and")

    def test_clear(self):
        self.sets.add("set0", [])
        self.sets.add("set1", [])
        self.sets.clear()
        self.assertEqual(len(self.sets), 0)

    def test_case_insensitive_keys(self):
        self.sets.add("Set0", ["obj0"])
        self.assertTrue("set0" in self.sets)
        self.assertTrue("SET0" in self.sets)
        result = self.sets.get("SET0", get_keys=True)
        self.assertIn("obj0", result)


class TestNMSetsEquations(NMSetsTestBase):
    """Tests for equation functionality."""

    def setUp(self):
        super().setUp()
        # Create two source sets
        self.sets.add("setA", ["obj0", "obj1", "obj2", "obj3"])
        self.sets.add("setB", ["obj2", "obj3", "obj4", "obj5"])

    def test_define_and(self):
        self.sets.define_and("setC", "setA", "setB")
        result = self.sets.get("setC", get_keys=True)
        # AND should give intersection: obj2, obj3
        self.assertEqual(set(result), {"obj2", "obj3"})

    def test_define_or(self):
        self.sets.define_or("setC", "setA", "setB")
        result = self.sets.get("setC", get_keys=True)
        # OR should give union: obj0, obj1, obj2, obj3, obj4, obj5
        self.assertEqual(set(result), {"obj0", "obj1", "obj2", "obj3", "obj4", "obj5"})

    def test_equation_via_setitem(self):
        self.sets["setC"] = ("and", "setA", "setB")
        result = self.sets.get("setC", get_keys=True)
        self.assertEqual(set(result), {"obj2", "obj3"})

    def test_equation_get_equation(self):
        self.sets.define_and("setC", "setA", "setB")
        eq = self.sets.get("setC", get_equation=True)
        self.assertIsInstance(eq, tuple)
        self.assertEqual(eq[0], "and")

    def test_equation_dynamic_evaluation(self):
        # Define equation first
        self.sets.define_and("setC", "setA", "setB")
        result1 = set(self.sets.get("setC", get_keys=True))
        self.assertEqual(result1, {"obj2", "obj3"})

        # Modify source set
        self.sets.add("setA", "obj4")
        result2 = set(self.sets.get("setC", get_keys=True))
        # Now intersection should include obj4
        self.assertEqual(result2, {"obj2", "obj3", "obj4"})

    def test_is_equation(self):
        self.sets.define_and("setC", "setA", "setB")
        self.assertTrue(self.sets.is_equation("setC"))
        self.assertFalse(self.sets.is_equation("setA"))
        self.assertFalse(self.sets.is_equation("nonexistent"))

    def test_equation_rejects_nonexistent_set(self):
        with self.assertRaises(KeyError):
            self.sets.define_and("setC", "nonexistent", "setB")
        with self.assertRaises(KeyError):
            self.sets.define_and("setC", "setA", "nonexistent")

    def test_cannot_overwrite_non_equation_with_equation(self):
        with self.assertRaises(ValueError):
            self.sets["setA"] = ("and", "setA", "setB")

    def test_cannot_add_to_equation(self):
        self.sets.define_and("setC", "setA", "setB")
        with self.assertRaises(ValueError):
            self.sets.add("setC", "obj0")

    def test_cannot_remove_from_equation(self):
        self.sets.define_and("setC", "setA", "setB")
        with self.assertRaises(ValueError):
            self.sets.remove("setC", "obj2")


class TestNMSetsTupleIsEquation(unittest.TestCase):
    """Tests for tuple_is_equation static method."""

    def test_valid_and_equation(self):
        self.assertTrue(NMSets.tuple_is_equation(("and", "set1", "set2")))

    def test_valid_or_equation(self):
        self.assertTrue(NMSets.tuple_is_equation(("or", "set1", "set2")))

    def test_rejects_list(self):
        self.assertFalse(NMSets.tuple_is_equation(["and", "set1", "set2"]))

    def test_rejects_wrong_length(self):
        self.assertFalse(NMSets.tuple_is_equation(("and", "set1")))
        self.assertFalse(NMSets.tuple_is_equation(("and", "set1", "set2", "extra")))

    def test_rejects_invalid_operator(self):
        self.assertFalse(NMSets.tuple_is_equation(("invalid", "set1", "set2")))
        self.assertFalse(NMSets.tuple_is_equation(("&", "set1", "set2")))
        self.assertFalse(NMSets.tuple_is_equation(("|", "set1", "set2")))

    def test_rejects_non_string_sets(self):
        self.assertFalse(NMSets.tuple_is_equation(("and", 123, "set2")))
        self.assertFalse(NMSets.tuple_is_equation(("and", "set1", 456)))

    def test_rejects_non_tuple_types(self):
        self.assertFalse(NMSets.tuple_is_equation(None))
        self.assertFalse(NMSets.tuple_is_equation(123))
        self.assertFalse(NMSets.tuple_is_equation("string"))
        self.assertFalse(NMSets.tuple_is_equation({}))


class TestNMSetsEquationOperators(unittest.TestCase):
    """Tests for EQUATION_OPERATORS constant."""

    def test_contains_and(self):
        self.assertIn("and", EQUATION_OPERATORS)

    def test_contains_or(self):
        self.assertIn("or", EQUATION_OPERATORS)

    def test_only_two_operators(self):
        self.assertEqual(len(EQUATION_OPERATORS), 2)


class TestNMSetsCopy(NMSetsTestBase):
    """Tests for copy functionality."""

    def test_deepcopy(self):
        self.sets.add("set0", ["obj0", "obj1"])
        sets_copy = copy.deepcopy(self.sets)
        self.assertEqual(self.sets.name, sets_copy.name)
        self.assertEqual(list(self.sets.keys()), list(sets_copy.keys()))

    def test_copy_method(self):
        self.sets.add("set0", ["obj0", "obj1"])
        sets_copy = self.sets.copy()
        self.assertEqual(self.sets.name, sets_copy.name)
        self.assertEqual(list(self.sets.keys()), list(sets_copy.keys()))

    def test_deepcopy_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")

        sets_copy = copy.deepcopy(self.sets)

        # Check equation is preserved
        self.assertTrue(sets_copy.is_equation("setC"))
        eq = sets_copy.get("setC", get_equation=True)
        self.assertEqual(eq, ("and", "setA", "setB"))


class TestNMSetsRemove(NMSetsTestBase):
    """Tests for remove operations."""

    def test_remove_by_name(self):
        self.sets.add("set0", ["obj0", "obj1", "obj2"])
        removed = self.sets.remove("set0", "obj1")
        self.assertEqual(len(removed), 1)
        self.assertNotIn("obj1", self.sets.get("set0", get_keys=True))

    def test_remove_by_object(self):
        self.sets.add("set0", ["obj0", "obj1", "obj2"])
        removed = self.sets.remove("set0", self.objects["obj1"])
        self.assertEqual(len(removed), 1)

    def test_remove_nonexistent_raises(self):
        self.sets.add("set0", ["obj0"])
        with self.assertRaises(ValueError):
            self.sets.remove("set0", "obj9")

    def test_remove_nonexistent_error_false(self):
        self.sets.add("set0", ["obj0"])
        result = self.sets.remove("set0", "obj9", error=False)
        self.assertEqual(result, [])

    def test_remove_from_all(self):
        self.sets.add("set0", ["obj0", "obj1"])
        self.sets.add("set1", ["obj0", "obj2"])
        self.sets.remove_from_all("obj0")
        self.assertNotIn("obj0", self.sets.get("set0", get_keys=True))
        self.assertNotIn("obj0", self.sets.get("set1", get_keys=True))


class TestNMSetsRenameItem(NMSetsTestBase):
    """Tests for rename_item (used when container renames an object)."""

    def test_rename_item_updates_set(self):
        self.sets.add("set0", ["obj0", "obj1", "obj2"])
        self.objects["obj_new"] = self.objects.pop("obj0")
        self.objects["obj_new"]._name_set(newname="obj_new", quiet=True)
        self.sets.rename_item("obj0", "obj_new")
        keys = self.sets.get("set0", get_keys=True)
        self.assertNotIn("obj0", keys)
        self.assertIn("obj_new", keys)

    def test_rename_item_in_multiple_sets(self):
        self.sets.add("set0", ["obj0", "obj1"])
        self.sets.add("set1", ["obj0", "obj2"])
        self.objects["obj_new"] = self.objects.pop("obj0")
        self.objects["obj_new"]._name_set(newname="obj_new", quiet=True)
        self.sets.rename_item("obj0", "obj_new")
        self.assertIn("obj_new", self.sets.get("set0", get_keys=True))
        self.assertIn("obj_new", self.sets.get("set1", get_keys=True))

    def test_rename_item_not_in_set(self):
        self.sets.add("set0", ["obj1", "obj2"])
        # Renaming obj0 shouldn't affect set0
        self.sets.rename_item("obj0", "obj_new")
        keys = self.sets.get("set0", get_keys=True)
        self.assertEqual(keys, ["obj1", "obj2"])


class TestNMSetsNew(NMSetsTestBase):
    """Tests for new and duplicate operations."""

    def test_new_creates_empty_set(self):
        key, olist = self.sets.new("newset")
        self.assertEqual(key, "newset")
        self.assertEqual(olist, [])
        self.assertTrue("newset" in self.sets)

    def test_new_auto_name(self):
        key1, _ = self.sets.new()
        key2, _ = self.sets.new()
        self.assertEqual(key1, "set0")
        self.assertEqual(key2, "set1")

    def test_duplicate(self):
        self.sets.add("set0", ["obj0", "obj1"])
        key, olist = self.sets.duplicate("set0", "set0_copy")
        self.assertEqual(key, "set0_copy")
        self.assertEqual(
            self.sets.get("set0", get_keys=True),
            self.sets.get("set0_copy", get_keys=True)
        )

    def test_duplicate_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        key, value = self.sets.duplicate("setC", "setC_copy")
        self.assertEqual(key, "setC_copy")
        # Should duplicate the equation, not the evaluated result
        self.assertTrue(self.sets.is_equation("setC_copy"))
        self.assertEqual(
            self.sets.get("setC", get_equation=True),
            self.sets.get("setC_copy", get_equation=True)
        )


class TestNMSetsRename(NMSetsTestBase):
    """Tests for rename and reorder operations."""

    def test_rename(self):
        self.sets.add("set0", ["obj0"])
        self.sets.rename("set0", "renamed")
        self.assertFalse("set0" in self.sets)
        self.assertTrue("renamed" in self.sets)

    def test_rename_updates_equations(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        self.sets.rename("setA", "setX")
        # Equation should now reference "setX" instead of "setA"
        eq = self.sets.get("setC", get_equation=True)
        self.assertEqual(eq, ("and", "setX", "setB"))
        # Equation should still evaluate correctly
        result = set(self.sets.get("setC", get_keys=True))
        self.assertEqual(result, {"obj1"})

    def test_rename_updates_equations_second_operand(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_or("setC", "setA", "setB")
        self.sets.rename("setB", "setY")
        eq = self.sets.get("setC", get_equation=True)
        self.assertEqual(eq, ("or", "setA", "setY"))

    def test_reorder(self):
        self.sets.add("set0", [])
        self.sets.add("set1", [])
        self.sets.add("set2", [])
        self.sets.reorder(["set2", "set1", "set0"])
        keys = list(self.sets.keys())
        self.assertEqual(keys, ["set2", "set1", "set0"])


class TestNMSetsEmpty(NMSetsTestBase):
    """Tests for empty operations."""

    def test_empty(self):
        self.sets.add("set0", ["obj0", "obj1"])
        self.sets.empty("set0")
        self.assertEqual(self.sets.get("set0"), [])

    def test_empty_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        self.assertTrue(self.sets.is_equation("setC"))
        self.sets.empty("setC")
        # After emptying, it should no longer be an equation
        self.assertFalse(self.sets.is_equation("setC"))
        self.assertEqual(self.sets.get("setC"), [])

    def test_empty_all(self):
        self.sets.add("set0", ["obj0"])
        self.sets.add("set1", ["obj1"])
        self.sets.empty_all()
        self.assertEqual(self.sets.get("set0"), [])
        self.assertEqual(self.sets.get("set1"), [])

    def test_empty_all_with_equation(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.sets.add("setB", ["obj1", "obj2"])
        self.sets.define_and("setC", "setA", "setB")
        self.sets.empty_all()
        self.assertEqual(self.sets.get("setA"), [])
        self.assertEqual(self.sets.get("setB"), [])
        self.assertEqual(self.sets.get("setC"), [])
        self.assertFalse(self.sets.is_equation("setC"))


class TestNMSetsEquality(NMSetsTestBase):
    """Tests for equality comparison."""

    def test_equal_empty_sets(self):
        sets2 = NMSets(name="TestSets", nmobjects=self.objects)
        self.assertEqual(self.sets, sets2)

    def test_equal_with_items(self):
        sets2 = NMSets(name="TestSets", nmobjects=self.objects)
        self.sets.add("set0", ["obj0", "obj1"])
        sets2.add("set0", ["obj0", "obj1"])
        self.assertEqual(self.sets, sets2)

    def test_not_equal_different_items(self):
        sets2 = NMSets(name="TestSets", nmobjects=self.objects)
        self.sets.add("set0", ["obj0"])
        sets2.add("set0", ["obj1"])
        self.assertNotEqual(self.sets, sets2)

    def test_equal_with_equations(self):
        sets2 = NMSets(name="TestSets", nmobjects=self.objects)
        for s in [self.sets, sets2]:
            s.add("setA", ["obj0", "obj1"])
            s.add("setB", ["obj1", "obj2"])
            s.define_and("setC", "setA", "setB")
        self.assertEqual(self.sets, sets2)


class TestNMSetsPathStr(unittest.TestCase):
    """Tests for NMSets.path_str property."""

    def test_path_str_with_parent(self):
        nm = NMManager(quiet=True)
        from pyneuromatic.core.nm_object_container import NMObjectContainer
        container = NMObjectContainer(parent=nm, name="Data")
        self.assertEqual(container.sets.path_str, container.path_str + ".sets")

    def test_path_str_without_parent(self):
        sets = NMSets(name="MySets", nmobjects={})
        self.assertEqual(sets.path_str, "MySets")

    def test_path_str_deepcopy_has_no_parent(self):
        nm = NMManager(quiet=True)
        from pyneuromatic.core.nm_object_container import NMObjectContainer
        container = NMObjectContainer(parent=nm, name="Data")
        sets_copy = copy.deepcopy(container.sets)
        self.assertEqual(sets_copy.path_str, "NMObjectContainerSets")


class TestNMSetsHistory(unittest.TestCase):
    """Tests for history logging in NMSets mutation methods."""

    def setUp(self):
        self.nm = NMManager(quiet=True)
        from pyneuromatic.core.nm_object_container import NMObjectContainer
        self.container = NMObjectContainer(
            parent=self.nm,
            name="Data",
            rename_on=True,
            auto_name_prefix="obj",
            auto_name_seq_format="0",
        )
        for i in range(5):
            self.container.new("obj%d" % i)
        self.nm.history.clear()
        self.sets = self.container.sets

    def _last_message(self):
        buf = self.nm.history.buffer
        if buf:
            return buf[-1]["message"]
        return None

    def _messages(self):
        return [e["message"] for e in self.nm.history.buffer]

    def test_new_logs(self):
        self.sets.new("mySet", quiet=False)
        self.assertIn("new set 'mySet'", self._last_message())

    def test_add_logs(self):
        self.sets.add("setA", ["obj0", "obj1"], quiet=False)
        msg = self._last_message()
        self.assertIn("setA", msg)
        self.assertIn("obj0", msg)
        self.assertIn("obj1", msg)

    def test_define_and_logs(self):
        self.sets.add("s1", ["obj0", "obj1"])
        self.sets.add("s2", ["obj1", "obj2"])
        self.nm.history.clear()
        self.sets.define_and("s3", "s1", "s2", quiet=False)
        msg = self._last_message()
        self.assertIn("s3", msg)
        self.assertIn("s1", msg)
        self.assertIn("AND", msg)
        self.assertIn("s2", msg)

    def test_define_or_logs(self):
        self.sets.add("s1", ["obj0", "obj1"])
        self.sets.add("s2", ["obj1", "obj2"])
        self.nm.history.clear()
        self.sets.define_or("s3", "s1", "s2", quiet=False)
        msg = self._last_message()
        self.assertIn("s3", msg)
        self.assertIn("s1", msg)
        self.assertIn("OR", msg)
        self.assertIn("s2", msg)

    def test_remove_logs(self):
        self.sets.add("setA", ["obj0", "obj1", "obj2"])
        self.nm.history.clear()
        self.sets.remove("setA", "obj1", quiet=False)
        msg = self._last_message()
        self.assertIn("setA", msg)
        self.assertIn("obj1", msg)

    def test_pop_logs(self):
        self.sets.add("setA", ["obj0"])
        self.nm.history.clear()
        self.sets.pop("setA", quiet=False)
        self.assertIn("removed set 'setA'", self._last_message())

    def test_clear_logs(self):
        self.sets.add("s1", ["obj0"])
        self.sets.add("s2", ["obj1"])
        self.nm.history.clear()
        self.sets.clear(quiet=False)
        msg = self._last_message()
        self.assertIn("cleared all sets", msg)
        self.assertIn("s1", msg)
        self.assertIn("s2", msg)

    def test_clear_empty_no_log(self):
        self.sets.clear(quiet=False)
        self.assertEqual(len(self.nm.history.buffer), 0)

    def test_rename_logs(self):
        self.sets.add("setA", ["obj0"])
        self.nm.history.clear()
        self.sets.rename("setA", "setB", quiet=False)
        msg = self._last_message()
        self.assertIn("setA", msg)
        self.assertIn("setB", msg)

    def test_duplicate_logs(self):
        self.sets.add("setA", ["obj0"])
        self.nm.history.clear()
        self.sets.duplicate("setA", "setA_copy", quiet=False)
        msg = self._last_message()
        self.assertIn("setA", msg)
        self.assertIn("setA_copy", msg)

    def test_reorder_logs(self):
        self.sets.add("s1", ["obj0"])
        self.sets.add("s2", ["obj1"])
        self.nm.history.clear()
        self.sets.reorder(["s2", "s1"], quiet=False)
        self.assertIn("reordered", self._last_message())

    def test_reorder_same_order_no_log(self):
        self.sets.add("s1", ["obj0"])
        self.sets.add("s2", ["obj1"])
        self.nm.history.clear()
        self.sets.reorder(["s1", "s2"], quiet=False)
        self.assertEqual(len(self.nm.history.buffer), 0)

    def test_empty_logs(self):
        self.sets.add("setA", ["obj0", "obj1"])
        self.nm.history.clear()
        self.sets.empty("setA", quiet=False)
        self.assertIn("emptied set 'setA'", self._last_message())


if __name__ == "__main__":
    unittest.main(verbosity=2)
