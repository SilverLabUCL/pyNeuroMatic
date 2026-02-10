#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMObjectContainer.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer, ExecuteMode
from tests.test_core.test_nm_object import NMObject2

QUIET = True
NM = NMManager(quiet=QUIET)
NM0 = NMManager(quiet=QUIET)
NM1 = NMManager(quiet=QUIET)
CNAME0 = "map0"
CNAME1 = "map1"
OPREFIX0 = "object"
OPREFIX1 = "obj"
OSEQFORMAT0 = "0"
OSEQFORMAT1 = "A"
ONLIST0 = [OPREFIX0 + str(i) for i in range(6)]
ONLIST1 = [OPREFIX1 + nmu.CHANNEL_CHARS[i] for i in range(6)]
SETS_NLIST0 = ["set" + str(i) for i in range(3)]
SETS_NLIST1 = ["s" + str(i) for i in range(3)]

# Common bad types for testing type validation
BAD_TYPES = list(nmu.BADTYPES)
BAD_NAMES = list(nmu.BADNAMES)


class NMObjectContainerTestBase(unittest.TestCase):
    """Base class with shared setUp for NMObjectContainer tests."""

    def setUp(self):
        self.olist0 = []
        self.olist1 = []
        for n in ONLIST0:
            self.olist0.append(NMObject(parent=NM0, name=n))
        for n in ONLIST1:
            self.olist1.append(NMObject(parent=NM1, name=n))

        self.map0 = NMObjectContainer(
            parent=NM0,
            name=CNAME0,
            rename_on=True,
            auto_name_prefix=OPREFIX0,
            auto_name_seq_format=OSEQFORMAT0,
        )
        self.map0.update(self.olist0)

        self.sets0 = {SETS_NLIST0[0]: [ONLIST0[0], ONLIST0[2], ONLIST0[3]]}
        self.map0.sets.update(self.sets0)

        self.map1 = NMObjectContainer(
            parent=NM1,
            name=CNAME1,
            rename_on=False,
            auto_name_prefix=OPREFIX1,
            auto_name_seq_format=OSEQFORMAT1,
        )
        self.map1.update(self.olist1)
        self.map1.selected_name = ONLIST1[2]

        self.sets1 = {
            SETS_NLIST1[0]: [ONLIST1[0], ONLIST1[2], ONLIST1[4]],
            SETS_NLIST1[1]: [ONLIST1[1], ONLIST1[3], ONLIST1[5]],
        }
        self.map1.sets.update(self.sets1)

        self.map1_copy = copy.deepcopy(self.map1)


class TestNMObjectContainerInit(NMObjectContainerTestBase):
    """Tests for NMObjectContainer initialization."""

    def test_rejects_non_bool_rename_on(self):
        bad = list(BAD_TYPES)
        bad.remove(None)
        bad.remove(True)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectContainer(rename_on=b)

    def test_allows_no_arguments(self):
        c = NMObjectContainer()
        self.assertIsNotNone(c)

    def test_stores_parent_map0(self):
        self.assertEqual(self.map0._parent, NM0)

    def test_stores_name_map0(self):
        self.assertEqual(self.map0.name, CNAME0)

    def test_stores_rename_on_true(self):
        self.assertTrue(self.map0._NMObjectContainer__rename_on)

    def test_stores_auto_name_prefix(self):
        self.assertEqual(self.map0._NMObjectContainer__auto_name_prefix, OPREFIX0)

    def test_stores_auto_name_seq_format(self):
        self.assertEqual(self.map0._NMObjectContainer__auto_name_seq_format, OSEQFORMAT0)

    def test_default_selected_is_none(self):
        self.assertIsNone(self.map0.selected_name)
        self.assertIsNone(self.map0.selected_value)

    def test_default_execute_mode_is_selected(self):
        self.assertEqual(self.map0.execute_mode, ExecuteMode.SELECTED)

    def test_default_execute_target_name_is_none(self):
        self.assertEqual(self.map0.execute_target_name, None)

    def test_execute_targets_empty_when_no_selection(self):
        self.assertEqual(self.map0.execute_targets, [])

    def test_stores_sets(self):
        self.assertEqual(list(self.map0.sets.keys()), [SETS_NLIST0[0]])

    def test_stores_all_objects(self):
        self.assertEqual(len(self.map0._NMObjectContainer__map), len(ONLIST0))

    def test_map1_stores_parent(self):
        self.assertEqual(self.map1._parent, NM1)

    def test_map1_stores_name(self):
        self.assertEqual(self.map1.name, CNAME1)

    def test_map1_rename_on_false(self):
        self.assertFalse(self.map1._NMObjectContainer__rename_on)

    def test_map1_stores_prefix(self):
        self.assertEqual(self.map1._NMObjectContainer__auto_name_prefix, OPREFIX1)

    def test_map1_stores_seq_format(self):
        self.assertEqual(self.map1._NMObjectContainer__auto_name_seq_format, OSEQFORMAT1)

    def test_map1_selected_name(self):
        self.assertEqual(self.map1.selected_name, ONLIST1[2])

    def test_map1_selected_value(self):
        self.assertEqual(self.map1.selected_value, self.olist1[2])

    def test_map1_execute_targets(self):
        self.assertEqual(self.map1.execute_targets, [self.olist1[2]])

    def test_map1_stores_multiple_sets(self):
        self.assertEqual(list(self.map1.sets.keys()), [SETS_NLIST1[0], SETS_NLIST1[1]])


class TestNMObjectContainerCopy(NMObjectContainerTestBase):
    """Tests for NMObjectContainer deepcopy."""

    def test_copy_preserves_parent(self):
        self.assertEqual(self.map1_copy._parent, NM1)

    def test_copy_preserves_name(self):
        self.assertEqual(self.map1_copy.name, CNAME1)

    def test_copy_preserves_rename_on(self):
        self.assertFalse(self.map1_copy._NMObjectContainer__rename_on)

    def test_copy_preserves_prefix(self):
        self.assertEqual(self.map1_copy._NMObjectContainer__auto_name_prefix, OPREFIX1)

    def test_copy_preserves_seq_format(self):
        self.assertEqual(self.map1_copy._NMObjectContainer__auto_name_seq_format, OSEQFORMAT1)

    def test_copy_preserves_selected_name(self):
        self.assertEqual(self.map1_copy.selected_name, ONLIST1[2])

    def test_copy_preserves_execute_mode(self):
        self.assertEqual(self.map1_copy.execute_mode, ExecuteMode.SELECTED)

    def test_copy_preserves_sets(self):
        self.assertEqual(list(self.map1_copy.sets.keys()), [SETS_NLIST1[0], SETS_NLIST1[1]])

    def test_copy_preserves_object_count(self):
        self.assertEqual(len(self.map1_copy._NMObjectContainer__map), len(ONLIST1))

    def test_copied_objects_are_equal(self):
        for n in ONLIST1:
            original = self.map1.get(n)
            copied = self.map1_copy.get(n)
            self.assertTrue(original == copied)

    def test_copied_objects_are_not_same_instance(self):
        for n in ONLIST1:
            original = self.map1.get(n)
            copied = self.map1_copy.get(n)
            self.assertFalse(original is copied)

    def test_copied_selected_values_equal(self):
        self.assertTrue(self.map1.selected_value == self.map1_copy.selected_value)

    def test_copied_selected_values_not_same_instance(self):
        self.assertFalse(self.map1.selected_value is self.map1_copy.selected_value)


class TestNMObjectContainerParameters(NMObjectContainerTestBase):
    """Tests for NMObjectContainer parameters property."""

    def test_has_expected_keys(self):
        expected_keys = [
            "name", "created", "copy of", "content_type", "rename_on",
            "auto_name_prefix", "auto_name_seq_format", "selected_name",
            "execute_mode", "execute_target_name", "sets"
        ]
        plist = self.map0.parameters
        self.assertEqual(list(plist.keys()), expected_keys)

    def test_map0_name_matches(self):
        plist = self.map0.parameters
        self.assertEqual(plist["name"], CNAME0)

    def test_map0_copy_of_is_none(self):
        plist = self.map0.parameters
        self.assertIsNone(plist["copy of"])

    def test_map0_content_type(self):
        plist = self.map0.parameters
        self.assertEqual(plist["content_type"], "nmobject")

    def test_map0_rename_on_true(self):
        plist = self.map0.parameters
        self.assertTrue(plist["rename_on"])

    def test_map0_auto_name_prefix(self):
        plist = self.map0.parameters
        self.assertEqual(plist["auto_name_prefix"], OPREFIX0)

    def test_map0_auto_name_seq_format(self):
        plist = self.map0.parameters
        self.assertEqual(plist["auto_name_seq_format"], OSEQFORMAT0)

    def test_map0_selected_name_none(self):
        plist = self.map0.parameters
        self.assertIsNone(plist["selected_name"])

    def test_map0_execute_mode(self):
        plist = self.map0.parameters
        self.assertEqual(plist["execute_mode"], ExecuteMode.SELECTED.name)

    def test_map0_execute_target_name(self):
        plist = self.map0.parameters
        self.assertEqual(plist["execute_target_name"], None)

    def test_map0_sets(self):
        plist = self.map0.parameters
        self.assertEqual(plist["sets"], [SETS_NLIST0[0]])

    def test_map1_rename_on_false(self):
        plist = self.map1.parameters
        self.assertFalse(plist["rename_on"])

    def test_map1_selected_name(self):
        plist = self.map1.parameters
        self.assertEqual(plist["selected_name"], ONLIST1[2])

    def test_map1_copy_has_copy_of(self):
        plist = self.map1_copy.parameters
        # NMManager is not an NMObject, so path_str is just the container name
        self.assertEqual(plist["copy of"], CNAME1)


class TestNMObjectContainerContentType(NMObjectContainerTestBase):
    """Tests for content_type and content_type_ok methods."""

    def test_content_type_returns_nmobject(self):
        self.assertEqual(self.map0.content_type(), "NMObject")

    def test_rejects_manager_as_content(self):
        self.assertFalse(self.map0.content_type_ok(NM0))

    def test_rejects_container_as_content(self):
        self.assertFalse(self.map0.content_type_ok(self.map1))

    def test_accepts_nmobject_as_content(self):
        self.assertTrue(self.map0.content_type_ok(self.olist0[0]))

    def test_content_parameters_is_list(self):
        plist = self.map0.content_parameters
        self.assertTrue(isinstance(plist, list))

    def test_content_parameters_length_matches(self):
        plist = self.map0.content_parameters
        self.assertEqual(len(plist), len(self.map0))

    def test_content_parameters_are_dicts(self):
        plist = self.map0.content_parameters
        for p in plist:
            self.assertTrue(isinstance(p, dict))


class TestNMObjectContainerGetItem(NMObjectContainerTestBase):
    """Tests for __getitem__, get(), items(), values() methods."""

    def test_get_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove(None)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.get(b)

    def test_get_returns_correct_objects(self):
        for i, n in enumerate(ONLIST0):
            o = self.map0.get(n)
            self.assertEqual(o, self.olist0[i])

    def test_getitem_returns_correct_objects(self):
        for i, n in enumerate(ONLIST0):
            o = self.map0[n]
            self.assertEqual(o, self.olist0[i])

    def test_getitem_raises_keyerror_for_missing(self):
        with self.assertRaises(KeyError):
            _ = self.map0["test"]

    def test_get_returns_none_for_missing(self):
        o = self.map0.get("test")
        self.assertIsNone(o)

    def test_get_is_case_insensitive(self):
        for i, k in enumerate(ONLIST0):
            self.assertEqual(self.map0.get(k), self.olist0[i])
            self.assertEqual(self.map0.get(k.upper()), self.olist0[i])

    def test_items_returns_correct_pairs(self):
        for i, (k, v) in enumerate(self.map0.items()):
            self.assertEqual(k, ONLIST0[i])
            self.assertEqual(v, self.olist0[i])

    def test_values_returns_correct_objects(self):
        for i, v in enumerate(self.map0.values()):
            self.assertEqual(v, self.olist0[i])


class TestNMObjectContainerSetItem(NMObjectContainerTestBase):
    """Tests for __setitem__ method."""

    def test_rejects_bad_key_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0[b] = NMObject(parent=NM0, name="test")

    def test_rejects_bad_value_types(self):
        n = ONLIST0[1]
        for b in BAD_TYPES:
            with self.assertRaises(TypeError):
                self.map0[n] = b

    def test_rejects_mismatched_name_and_key(self):
        n = ONLIST0[1]
        badname = n + "x"
        with self.assertRaises(KeyError):
            self.map0[n] = NMObject(parent=NM0, name=badname)

    def test_replaces_existing_object(self):
        n = ONLIST0[1]
        o1 = self.map0.get(n)
        self.assertTrue(o1 is self.olist0[1])
        o2 = NMObject(parent=NM0, name=n)
        self.map0[n.upper()] = o2
        o1 = self.map0.get(n)
        self.assertFalse(o1 is self.olist0[1])
        self.assertTrue(o1 is o2)

    def test_updates_rename_fxnref(self):
        n = ONLIST0[1]
        o2 = NMObject(parent=NM0, name=n)
        rfr_before = o2._NMObject__rename_fxnref
        self.map0[n.upper()] = o2
        o1 = self.map0.get(n)
        rfr_after = o1._NMObject__rename_fxnref
        self.assertNotEqual(rfr_before, rfr_after)
        self.assertEqual(rfr_after, self.map0.rename)

    def test_length_unchanged_when_replacing(self):
        n = ONLIST0[1]
        o2 = NMObject(parent=NM0, name=n)
        self.map0[n.upper()] = o2
        self.assertEqual(len(self.map0), len(ONLIST0))

    def test_adds_new_object(self):
        n = "test"
        o3 = NMObject(parent=NM0, name=n)
        self.assertFalse(n in self.map0)
        self.map0[n.upper()] = o3
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)


class TestNMObjectContainerDelItem(NMObjectContainerTestBase):
    """Tests for __delitem__ and pop methods."""

    def test_delitem_raises_keyerror_for_missing(self):
        with self.assertRaises(KeyError):
            del self.map0["test"]

    def test_delitem_removes_object(self):
        del self.map0[ONLIST0[1]]
        self.assertFalse(ONLIST0[1] in self.map0)

    def test_delitem_raises_keyerror_after_removal(self):
        del self.map0[ONLIST0[1]]
        with self.assertRaises(KeyError):
            del self.map0[ONLIST0[1]]

    def test_pop_raises_keyerror_for_missing(self):
        with self.assertRaises(KeyError):
            self.map0.pop("test")

    def test_pop_returns_object(self):
        o = self.map0.pop(ONLIST0[1])
        self.assertEqual(o, self.olist0[1])

    def test_pop_removes_object(self):
        self.map0.pop(ONLIST0[1])
        self.assertFalse(ONLIST0[1] in self.map0)

    def test_pop_all_objects(self):
        for i, n in enumerate(ONLIST1):
            o = self.map1.pop(n)
            self.assertEqual(o, self.olist1[i])
            self.assertFalse(n in self.map1)
        self.assertEqual(len(self.map1), 0)

    def test_popitem_removes_last_object(self):
        o = self.map0.popitem()
        self.assertFalse(ONLIST0[-1] in self.map0)
        t = (ONLIST0[-1], self.olist0[-1])
        self.assertEqual(o, t)

    def test_popitem_all_objects(self):
        for i, n in reversed(list(enumerate(ONLIST1))):
            o = self.map1.popitem()
            self.assertFalse(n in self.map1)
        self.assertEqual(len(self.map1), 0)

    def test_clear_empties_container(self):
        o = self.map0.clear()
        self.assertIsNone(o)
        self.assertEqual(len(self.map0), 0)


class TestNMObjectContainerIterLen(NMObjectContainerTestBase):
    """Tests for __iter__ and __len__ methods."""

    def test_iter_yields_all_keys(self):
        o_iter = iter(self.map0)
        for n in ONLIST0:
            self.assertEqual(next(o_iter), n)

    def test_len_returns_object_count(self):
        self.assertEqual(len(self.map0), len(ONLIST0))


class TestNMObjectContainerContains(NMObjectContainerTestBase):
    """Tests for __contains__ and contains_value methods."""

    def test_empty_string_not_in_container(self):
        self.assertFalse("" in self.map0)

    def test_missing_key_not_in_container(self):
        self.assertFalse("test" in self.map0)

    def test_keys_in_container(self):
        for n in ONLIST0:
            self.assertTrue(n in self.map0)

    def test_keys_case_insensitive(self):
        for n in ONLIST0:
            self.assertTrue(n.upper() in self.map0)

    def test_keys_not_in_other_container(self):
        for n in ONLIST0:
            self.assertFalse(n in self.map1)

    def test_map1_keys_in_container(self):
        for n in ONLIST1:
            self.assertTrue(n in self.map1)

    def test_contains_value_true_for_own_objects(self):
        for o in self.olist0:
            self.assertTrue(self.map0.contains_value(o))

    def test_contains_value_false_for_other_objects(self):
        for o in self.olist0:
            self.assertFalse(self.map1.contains_value(o))


class TestNMObjectContainerEquality(NMObjectContainerTestBase):
    """Tests for __eq__ and __ne__ methods."""

    def test_not_equal_to_bad_types(self):
        for b in BAD_TYPES:
            self.assertFalse(self.map0 == b)

    def test_is_same_instance(self):
        self.assertTrue(self.map0 is self.map0)

    def test_not_same_instance_as_other(self):
        self.assertFalse(self.map0 is self.map1)

    def test_equals_self(self):
        self.assertTrue(self.map0 == self.map0)

    def test_not_equals_other(self):
        self.assertFalse(self.map0 == self.map1)

    def test_ne_self_is_false(self):
        self.assertFalse(self.map0 != self.map0)

    def test_ne_other_is_true(self):
        self.assertTrue(self.map0 != self.map1)

    def test_equals_recreated_container(self):
        map0 = NMObjectContainer(
            parent=NM0,
            name=CNAME0,
            rename_on=True,
            auto_name_prefix=OPREFIX0,
        )
        olist0 = []
        for n in ONLIST0:
            olist0.append(NMObject(parent=NM0, name=n))
        map0.update(olist0)
        self.assertFalse(map0 == self.map0)  # sets are not equal
        map0.sets.update(self.sets0)
        self.assertTrue(map0 == self.map0)

    def test_not_equals_with_extra_set(self):
        map0 = NMObjectContainer(
            parent=NM0,
            name=CNAME0,
            rename_on=True,
            auto_name_prefix=OPREFIX0,
        )
        olist0 = []
        for n in ONLIST0:
            olist0.append(NMObject(parent=NM0, name=n))
        map0.update(olist0)
        map0.sets.update(self.sets0)
        map0.sets.new(SETS_NLIST0[1])
        self.assertFalse(map0 == self.map0)


class TestNMObjectContainerKeys(NMObjectContainerTestBase):
    """Tests for keys(), _getkey(), and _newkey() methods."""

    def test_keys_returns_all_keys(self):
        klist = list(self.map0.keys())
        self.assertEqual(klist, ONLIST0)

    def test_map1_keys_returns_all_keys(self):
        klist = list(self.map1.keys())
        self.assertEqual(klist, ONLIST1)

    def test_getkey_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0._getkey(b)

    def test_getkey_returns_none_for_bad_names(self):
        for b in BAD_NAMES:
            key = self.map0._getkey(b)
            self.assertIsNone(key)

    def test_getkey_returns_none_for_missing(self):
        for n in ONLIST0:
            key = self.map0._getkey(n + "x")
            self.assertIsNone(key)

    def test_getkey_is_case_insensitive(self):
        key = self.map0._getkey(ONLIST0[1].upper())
        self.assertEqual(key, ONLIST0[1])

    def test_newkey_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0._newkey(b)

    def test_newkey_rejects_bad_names(self):
        for b in BAD_NAMES:
            with self.assertRaises(ValueError):
                self.map0._newkey(b)

    def test_newkey_rejects_existing_keys(self):
        for n in ONLIST0:
            with self.assertRaises(KeyError):
                self.map0._newkey(n)
            with self.assertRaises(KeyError):
                self.map0._newkey(n.upper())

    def test_newkey_accepts_nonexistent_keys(self):
        for n in ONLIST0:
            key = self.map0._newkey(n + "x")
            self.assertEqual(key, n + "x")

    def test_newkey_none_returns_next_auto_name(self):
        key = self.map0._newkey(None)
        n = len(ONLIST0)
        self.assertEqual(key, OPREFIX0 + str(n))


class TestNMObjectContainerUpdate(NMObjectContainerTestBase):
    """Tests for update() method."""

    def test_update_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove(None)
        bad.remove([])
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.update(b)

    def test_update_rejects_bad_items_in_list(self):
        o1 = NMObject(parent=NM0, name="test")
        for b in BAD_TYPES:
            with self.assertRaises(TypeError):
                self.map0.update([b, o1])

    def test_update_adds_object(self):
        n1 = "test"
        o1 = NMObject(parent=NM0, name=n1)
        self.map0.update(o1)
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)

    def test_update_sets_rename_fxnref(self):
        n1 = "test_new"
        o1 = NMObject(parent=NM0, name=n1)
        rfr_before = o1._NMObject__rename_fxnref
        self.map0.update(o1)
        rfr_after = o1._NMObject__rename_fxnref
        self.assertNotEqual(rfr_before, rfr_after)
        self.assertEqual(rfr_after, self.map0.rename)

    def test_update_replaces_same_name(self):
        n1 = "test"
        o1 = NMObject(parent=NM0, name=n1)
        o2 = NMObject(parent=NM0, name=n1.upper())
        self.map0.update(o1)
        self.map0.update(o2)
        self.assertTrue(self.map0.get(n1) is o2)

    def test_update_dict_ignores_key(self):
        n1 = "test"
        o2 = NMObject(parent=NM0, name=n1.upper())
        self.map0.update({"test": o2})
        self.assertTrue(self.map0.get(n1) is o2)

    def test_update_dict_rejects_mismatched_keys(self):
        n1 = "test"
        o1 = NMObject(parent=NM0, name=n1)
        o2 = NMObject(parent=NM0, name=n1.upper())
        with self.assertRaises(KeyError):
            self.map0.update({"test1": o1, "test2": o2})

    def test_update_rejects_wrong_type(self):
        o3 = NMObject2(parent=NM0, name="test3")
        with self.assertRaises(TypeError):
            self.map0.update(o3)

    def test_update_from_another_container(self):
        new_len = len(self.map0) + len(self.map1)
        self.map0.update(self.map1)
        self.assertEqual(len(self.map0), new_len)
        for n in ONLIST1:
            self.assertTrue(n in self.map0)


class TestNMObjectContainerSetDefault(NMObjectContainerTestBase):
    """Tests for setdefault() method."""

    def test_raises_keyerror_for_missing_without_default(self):
        with self.assertRaises(KeyError):
            self.map0.setdefault("test")

    def test_returns_default_for_missing(self):
        o = self.map0.setdefault("test", default="ok")
        self.assertEqual(o, "ok")

    def test_returns_object_for_existing(self):
        for i, n in enumerate(ONLIST0):
            self.assertEqual(self.map0.setdefault(n), self.olist0[i])

    def test_does_not_modify_length(self):
        for n in ONLIST0:
            self.map0.setdefault(n)
        self.assertEqual(len(self.map0), len(ONLIST0))


class TestNMObjectContainerRename(NMObjectContainerTestBase):
    """Tests for rename() method."""

    def test_rename_rejects_bad_key_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.rename(b, ONLIST0[3])

    def test_rename_rejects_bad_newkey_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.rename(ONLIST0[4], b)

    def test_rename_rejects_bad_key_names(self):
        for b in BAD_NAMES:
            with self.assertRaises(KeyError):
                self.map0.rename(b, ONLIST0[3])

    def test_rename_rejects_bad_newkey_names(self):
        for b in BAD_NAMES:
            with self.assertRaises(ValueError):
                self.map0.rename(ONLIST0[4], b)

    def test_rename_rejects_same_name_different_case(self):
        with self.assertRaises(KeyError):
            self.map0.rename(ONLIST0[0], ONLIST0[0].upper())

    def test_rename_raises_when_rename_off(self):
        with self.assertRaises(RuntimeError):
            self.map1.rename(ONLIST1[0], 'test')

    def test_rename_succeeds(self):
        self.map0.pop(ONLIST0[3])
        s = self.map0.rename(ONLIST0[0], ONLIST0[3])
        self.assertTrue(s)
        klist = [OPREFIX0 + str(i) for i in [3, 1, 2, 4, 5]]
        self.assertEqual(list(self.map0.keys()), klist)

    def test_rename_with_none_uses_next_auto_name(self):
        self.map0.pop(ONLIST0[3])
        self.map0.rename(ONLIST0[0], ONLIST0[3])
        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "0")
        s = self.map0.rename(ONLIST0[4], None)
        self.assertTrue(s)
        klist = [OPREFIX0 + str(i) for i in [3, 1, 2, 0, 5]]
        self.assertEqual(list(self.map0.keys()), klist)

    def test_rename_via_property_assignment(self):
        self.map0.pop(ONLIST0[3])
        self.map0.rename(ONLIST0[0], ONLIST0[3])
        self.map0.rename(ONLIST0[4], None)
        for i, v in enumerate(self.map0.values()):
            o = self.map0.get(v.name)
            o.name = "test" + str(i)
        klist = ["test0", "test1", "test2", "test3", "test4"]
        self.assertEqual(list(self.map0.keys()), klist)


class TestNMObjectContainerRenameUpdatesSets(NMObjectContainerTestBase):
    """Tests that rename updates string keys in sets."""

    def test_rename_updates_sets(self):
        self.map0.sets.add("myset", [ONLIST0[0], ONLIST0[1]])
        self.map0.pop(ONLIST0[3])
        self.map0.rename(ONLIST0[0], ONLIST0[3])
        keys = self.map0.sets.get("myset", get_keys=True)
        self.assertIn(ONLIST0[3], keys)
        self.assertNotIn(ONLIST0[0], keys)
        self.assertIn(ONLIST0[1], keys)


class TestNMObjectContainerReorder(NMObjectContainerTestBase):
    """Tests for reorder() method."""

    def test_reorder_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.reorder(b)

    def test_reorder_rejects_bad_key_types_in_list(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.reorder([b])

    def test_reorder_rejects_wrong_count(self):
        klist = [OPREFIX0 + str(i) for i in [0, 1, 2, 3, 4]]
        with self.assertRaises(KeyError):
            self.map0.reorder(klist)

    def test_reorder_rejects_invalid_key(self):
        klist = [OPREFIX0 + str(i) for i in [0, 1, 2, 3, 4, 6]]
        with self.assertRaises(KeyError):
            self.map0.reorder(klist)

    def test_reorder_same_order_succeeds(self):
        self.map0.reorder(ONLIST0)  # No change

    def test_reorder_reverses_order(self):
        klist = ONLIST0.copy()
        klist.reverse()
        self.map0.reorder(klist)
        self.assertEqual(list(self.map0.keys()), klist)


class TestNMObjectContainerDuplicate(NMObjectContainerTestBase):
    """Tests for duplicate() method."""

    def test_duplicate_rejects_bad_key_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.duplicate(b, None)

    def test_duplicate_rejects_bad_newkey_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.duplicate(OPREFIX0 + "0", b)

    def test_duplicate_rejects_existing_name(self):
        with self.assertRaises(KeyError):
            self.map0.duplicate(ONLIST0[0], ONLIST0[3])

    def test_duplicate_with_none_uses_next_name(self):
        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "6")
        c = self.map0.duplicate(ONLIST0[1], None)
        self.assertEqual(c.name, nnext)
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)

    def test_duplicate_sets_copy_of(self):
        c = self.map0.duplicate(ONLIST0[1], None)
        o = self.map0.get(ONLIST0[1])
        self.assertFalse(c == o)
        pc = c.parameters
        self.assertEqual(pc["copy of"], ONLIST0[1])

    def test_duplicate_with_custom_name(self):
        c = self.map0.duplicate(ONLIST0[0], "test")
        self.assertEqual(c.name, "test")
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)
        pc = c.parameters
        self.assertEqual(pc["copy of"], ONLIST0[0])


class TestNMObjectContainerNew(NMObjectContainerTestBase):
    """Tests for new() and _new() methods."""

    def test_new_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.new(b)

    def test_new_rejects_existing_name(self):
        with self.assertRaises(KeyError):
            self.map0.new(ONLIST0[3])

    def test_new_adds_object(self):
        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "6")
        o = NMObject(parent=NM0, name=nnext)
        self.assertTrue(self.map0._new(o))
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)


class TestNMObjectContainerAutoNamePrefix(NMObjectContainerTestBase):
    """Tests for auto_name_prefix property."""

    def test_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove(None)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.auto_name_prefix = b

    def test_rejects_bad_names(self):
        bad = list(BAD_NAMES)
        bad.remove("")
        for b in bad:
            with self.assertRaises(ValueError):
                self.map0.auto_name_prefix = b

    def test_returns_current_prefix(self):
        self.assertEqual(self.map0.auto_name_prefix, OPREFIX0)

    def test_sets_new_prefix(self):
        self.map0.auto_name_prefix = "Test"
        self.assertEqual(self.map0.auto_name_prefix, "Test")


class TestNMObjectContainerAutoNameSeqFormat(NMObjectContainerTestBase):
    """Tests for auto_name_seq_format property."""

    def test_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.auto_name_seq_format = b

    def test_rejects_invalid_string(self):
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "string"

    def test_rejects_special_chars(self):
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "*"

    def test_rejects_non_zero_single_digit(self):
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "5"

    def test_returns_current_format(self):
        self.assertEqual(self.map0.auto_name_seq_format, OSEQFORMAT0)
        self.assertEqual(len(self.map0.auto_name_seq_format), 1)

    def test_rejects_mixed_formats(self):
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "01"
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "A0"

    def test_accepts_integer_zero(self):
        self.map0.auto_name_seq_format = 0
        self.assertEqual(self.map0.auto_name_seq_format, "0")

    def test_accepts_triple_zero(self):
        self.map0.auto_name_seq_format = "000"
        self.assertEqual(self.map0.auto_name_seq_format, "000")


class TestNMObjectContainerAutoNameSeq(NMObjectContainerTestBase):
    """Tests for auto name sequence methods."""

    def test_seq_next_str_numeric(self):
        seq_str = self.map0._auto_name_seq_next_str()
        seq_next_str = str(len(ONLIST0))
        self.assertEqual(seq_str, seq_next_str)

    def test_seq_counter_starts_at_zero(self):
        seq_str = self.map0._auto_name_seq_counter()
        self.assertEqual(seq_str, "0")

    def test_seq_next_str_with_padding(self):
        self.map0.auto_name_seq_format = "000"
        seq_str = self.map0._auto_name_seq_next_str()
        seq_next_str = "00" + str(len(ONLIST0))
        self.assertEqual(seq_str, seq_next_str)

    def test_seq_counter_with_padding(self):
        self.map0.auto_name_seq_format = "000"
        seq_str = self.map0._auto_name_seq_counter()
        self.assertEqual(seq_str, "000")

    def test_seq_next_str_alpha(self):
        seq_str = self.map1._auto_name_seq_next_str()
        seq_next = len(ONLIST1)
        seq_next_str = nmu.CHANNEL_CHARS[seq_next]
        self.assertEqual(seq_str, seq_next_str)

    def test_seq_counter_alpha_starts_at_a(self):
        seq_str = self.map1._auto_name_seq_counter()
        self.assertEqual(seq_str, "A")

    def test_seq_next_str_alpha_with_padding(self):
        self.map1.auto_name_seq_format = "AAA"
        seq_next = len(ONLIST1)
        self.assertEqual(self.map1._auto_name_seq_next_str(), "AA" + nmu.CHANNEL_CHARS[seq_next])

    def test_seq_counter_alpha_with_padding(self):
        self.map1.auto_name_seq_format = "AAA"
        seq_str = self.map1._auto_name_seq_counter()
        self.assertEqual(seq_str, "AAA")


class TestNMObjectContainerAutoNameSeqIncrement(NMObjectContainerTestBase):
    """Tests for auto_name_seq_counter_increment."""

    def test_numeric_counter_increments(self):
        for i in range(10):
            self.assertEqual(self.map0._auto_name_seq_counter(), str(i))
            if i == 9:
                with self.assertRaises(RuntimeError):
                    self.map0._auto_name_seq_counter_increment()
            else:
                self.map0._auto_name_seq_counter_increment()

    def test_numeric_counter_with_padding(self):
        self.map0.auto_name_seq_format = "000"
        for i in range(1000):
            if i < 10:
                self.assertEqual(self.map0._auto_name_seq_counter(), "00" + str(i))
            elif i < 100:
                self.assertEqual(self.map0._auto_name_seq_counter(), "0" + str(i))
            elif i < 1000:
                self.assertEqual(self.map0._auto_name_seq_counter(), str(i))
            if i == 999:
                with self.assertRaises(RuntimeError):
                    self.map0._auto_name_seq_counter_increment()
            else:
                self.map0._auto_name_seq_counter_increment()
        self.assertEqual(self.map0._auto_name_seq_counter(), "999")

    def test_alpha_counter_increments(self):
        for s in nmu.CHANNEL_CHARS:
            self.assertEqual(self.map1._auto_name_seq_counter(), s)
            if s == "Z":
                with self.assertRaises(RuntimeError):
                    self.map1._auto_name_seq_counter_increment()
            else:
                self.map1._auto_name_seq_counter_increment()

    def test_alpha_counter_double_increments(self):
        self.map1.auto_name_seq_format = "AA"
        for s1 in nmu.CHANNEL_CHARS:
            for s0 in nmu.CHANNEL_CHARS:
                self.assertEqual(self.map1._auto_name_seq_counter(), s1 + s0)
                if s1 == "Z" and s0 == "Z":
                    with self.assertRaises(RuntimeError):
                        self.map1._auto_name_seq_counter_increment()
                else:
                    self.map1._auto_name_seq_counter_increment()
        self.assertEqual(self.map1._auto_name_seq_counter(), "ZZ")


class TestNMObjectContainerAutoNameNext(NMObjectContainerTestBase):
    """Tests for auto_name_next() method."""

    def test_returns_next_name_numeric(self):
        name = self.map0.auto_name_next()
        seq_str = str(len(ONLIST0))
        self.assertEqual(name, OPREFIX0 + seq_str)

    def test_use_counter_returns_same_initially(self):
        name = self.map0.auto_name_next(use_counter=True)
        seq_str = str(len(ONLIST0))
        self.assertEqual(name, OPREFIX0 + seq_str)

    def test_use_counter_after_increment(self):
        # First call syncs counter with container length
        name = self.map0.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX0 + str(len(ONLIST0)))
        # Now increment and verify counter advances
        self.map0._auto_name_seq_counter_increment()
        self.map0._auto_name_seq_counter_increment()
        name = self.map0.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX0 + "8")

    def test_returns_next_name_alpha(self):
        name = self.map1.auto_name_next()
        i = len(ONLIST1)
        seq_str = nmu.CHANNEL_CHARS[i]
        self.assertEqual(name, OPREFIX1 + seq_str)

    def test_use_counter_alpha(self):
        name = self.map1.auto_name_next(use_counter=True)
        n = 6
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[n])

    def test_use_counter_alpha_after_increment(self):
        # First call syncs counter with container length
        name = self.map1.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[len(ONLIST1)])
        # Now increment and verify counter advances
        self.map1._auto_name_seq_counter_increment()
        name = self.map1.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[7])


class TestNMObjectContainerSelection(NMObjectContainerTestBase):
    """Tests for selected_name, selected_value, and is_selected."""

    def test_selected_name_rejects_bad_types(self):
        bad = list(BAD_TYPES)
        bad.remove(None)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.selected_name = b

    def test_selected_name_sets_correctly(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertEqual(self.map0.selected_name, ONLIST0[3])

    def test_selected_value_matches(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertEqual(self.map0.selected_value, self.olist0[3])

    def test_is_selected_true(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertTrue(self.map0.is_selected(ONLIST0[3]))

    def test_is_selected_false_for_other(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertFalse(self.map0.is_selected(ONLIST0[0]))

    def test_is_selected_false_for_bad_type(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertFalse(self.map0.is_selected(1))

    def test_selected_name_rejects_invalid_key(self):
        with self.assertRaises(KeyError):
            self.map0.selected_name = "test"

    def test_selected_name_accepts_none(self):
        self.map0.selected_name = ONLIST0[3]
        self.map0.selected_name = None
        self.assertIsNone(self.map0.selected_name)

    def test_selected_cleared_on_pop(self):
        self.map0.selected_name = ONLIST0[3]
        self.map0.pop(ONLIST0[3])
        self.assertIsNone(self.map0.selected_name)


class TestNMObjectContainerExecute(NMObjectContainerTestBase):
    """Tests for execute_mode, execute_target_name, execute_targets, and is_execute_target."""

    def test_execute_mode_rejects_bad_types(self):
        for b in BAD_TYPES:
            with self.assertRaises(TypeError):
                self.map0.execute_mode = b

    def test_is_execute_target_false_for_bad_types(self):
        for b in BAD_TYPES:
            self.assertFalse(self.map0.is_execute_target(b))

    def test_execute_mode_name_with_no_target(self):
        self.map0.execute_mode = ExecuteMode.NAME
        self.assertIsNone(self.map0.execute_target_name)

    def test_execute_target_name_rejects_invalid(self):
        self.map0.execute_mode = ExecuteMode.NAME
        with self.assertRaises(KeyError):
            self.map0.execute_target_name = "test"

    def test_is_execute_target_false_for_invalid(self):
        self.map0.execute_mode = ExecuteMode.NAME
        self.assertFalse(self.map0.is_execute_target("test"))

    def test_execute_mode_name_sets_target(self):
        self.map0.execute_mode = ExecuteMode.NAME
        self.map0.execute_target_name = ONLIST0[3]
        self.assertTrue(self.map0.is_execute_target(ONLIST0[3]))

    def test_execute_mode_selected_clears_target_name(self):
        self.map0.execute_mode = ExecuteMode.NAME
        self.map0.execute_target_name = ONLIST0[3]
        self.map0.execute_mode = ExecuteMode.SELECTED
        self.assertIsNone(self.map0.execute_target_name)

    def test_execute_targets_selected_mode(self):
        self.map0.selected_name = ONLIST0[3]
        self.assertEqual(self.map0.selected_value, self.olist0[3])
        self.assertTrue(self.map0.is_execute_target(ONLIST0[3]))
        self.assertEqual(self.map0.execute_targets, [self.olist0[3]])

    def test_execute_targets_name_mode(self):
        self.map0.execute_mode = ExecuteMode.NAME
        self.map0.execute_target_name = ONLIST0[4]
        self.assertEqual(self.map0.execute_targets, [self.olist0[4]])
        self.assertFalse(self.map0.is_execute_target(ONLIST0[3]))
        self.assertTrue(self.map0.is_execute_target(ONLIST0[4]))

    def test_execute_targets_set_mode(self):
        self.map0.execute_mode = ExecuteMode.SET
        self.map0.execute_target_name = SETS_NLIST0[0]
        self.assertEqual(self.map0.execute_targets, [self.olist0[0], self.olist0[2], self.olist0[3]])

    def test_execute_targets_all_mode(self):
        self.map0.execute_mode = ExecuteMode.ALL
        self.assertIsNone(self.map0.execute_target_name)
        self.assertEqual(self.map0.execute_targets, self.olist0)

    def test_is_execute_target_all_mode(self):
        self.map0.execute_mode = ExecuteMode.ALL
        for n in ONLIST0:
            self.assertTrue(self.map0.is_execute_target(n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
