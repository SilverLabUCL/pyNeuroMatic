#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMGroups and NMEpochContainer.groups.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import copy
import unittest

from pyneuromatic.core.nm_group import NMGroups
from pyneuromatic.core.nm_manager import NMManager

QUIET = True


# ---------------------------------------------------------------------------
# Shared setup helper
# ---------------------------------------------------------------------------

def _make_groups(n=6):
    """Return an NMGroups with n item names pre-assigned cyclically to 3 groups."""
    g = NMGroups(name="TestGroups", parent=None)
    names = ["E%d" % i for i in range(n)]
    g.assign_cyclic(names, n_groups=3, quiet=QUIET)
    return g, names


# =========================================================================
# NMGroups — init
# =========================================================================

class TestNMGroupsInit(unittest.TestCase):

    def test_empty_on_construction(self):
        g = NMGroups()
        self.assertEqual(len(g), 0)

    def test_name_default(self):
        g = NMGroups()
        self.assertEqual(g.name, "NMGroups0")

    def test_name_custom(self):
        g = NMGroups(name="MyGroups")
        self.assertEqual(g.name, "MyGroups")

    def test_path_str_no_parent(self):
        g = NMGroups(name="G")
        self.assertEqual(g.path_str, "G")

    def test_n_groups_empty(self):
        g = NMGroups()
        self.assertEqual(g.n_groups, 0)


# =========================================================================
# NMGroups — assign
# =========================================================================

class TestNMGroupsAssign(unittest.TestCase):

    def setUp(self):
        self.g = NMGroups(parent=None)

    def test_assign_single(self):
        self.g.assign("E0", 0, quiet=QUIET)
        self.assertEqual(self.g.get_group("E0"), 0)

    def test_assign_replaces_previous(self):
        self.g.assign("E0", 0, quiet=QUIET)
        self.g.assign("E0", 2, quiet=QUIET)
        self.assertEqual(self.g.get_group("E0"), 2)

    def test_assign_multiple(self):
        for i in range(5):
            self.g.assign("E%d" % i, i, quiet=QUIET)
        self.assertEqual(len(self.g), 5)

    def test_assign_group_zero(self):
        self.g.assign("E0", 0, quiet=QUIET)
        self.assertEqual(self.g.get_group("E0"), 0)

    def test_assign_bad_name_type_raises(self):
        with self.assertRaises(TypeError):
            self.g.assign(0, 1, quiet=QUIET)

    def test_assign_bool_group_raises(self):
        with self.assertRaises(TypeError):
            self.g.assign("E0", True, quiet=QUIET)

    def test_assign_float_group_raises(self):
        with self.assertRaises(TypeError):
            self.g.assign("E0", 1.0, quiet=QUIET)

    def test_assign_negative_group_raises(self):
        with self.assertRaises(ValueError):
            self.g.assign("E0", -1, quiet=QUIET)

    def test_assign_none_group_raises(self):
        with self.assertRaises(TypeError):
            self.g.assign("E0", None, quiet=QUIET)


# =========================================================================
# NMGroups — assign_cyclic
# =========================================================================

class TestNMGroupsAssignCyclic(unittest.TestCase):

    def test_cyclic_3_groups_6_names(self):
        g = NMGroups()
        names = ["E%d" % i for i in range(6)]
        g.assign_cyclic(names, n_groups=3, quiet=QUIET)
        self.assertEqual(g.get_group("E0"), 0)
        self.assertEqual(g.get_group("E1"), 1)
        self.assertEqual(g.get_group("E2"), 2)
        self.assertEqual(g.get_group("E3"), 0)
        self.assertEqual(g.get_group("E4"), 1)
        self.assertEqual(g.get_group("E5"), 2)

    def test_cyclic_5_groups_iv_relation(self):
        g = NMGroups()
        names = ["E%d" % i for i in range(15)]
        g.assign_cyclic(names, n_groups=5, quiet=QUIET)
        for i in range(15):
            self.assertEqual(g.get_group("E%d" % i), i % 5)

    def test_cyclic_1_group(self):
        g = NMGroups()
        names = ["E0", "E1", "E2"]
        g.assign_cyclic(names, n_groups=1, quiet=QUIET)
        for name in names:
            self.assertEqual(g.get_group(name), 0)

    def test_cyclic_empty_list(self):
        g = NMGroups()
        g.assign_cyclic([], n_groups=3, quiet=QUIET)
        self.assertEqual(len(g), 0)

    def test_cyclic_bad_names_type_raises(self):
        g = NMGroups()
        with self.assertRaises(TypeError):
            g.assign_cyclic("E0", n_groups=3, quiet=QUIET)

    def test_cyclic_bad_n_groups_type_raises(self):
        g = NMGroups()
        with self.assertRaises(TypeError):
            g.assign_cyclic(["E0"], n_groups=1.0, quiet=QUIET)

    def test_cyclic_bool_n_groups_raises(self):
        g = NMGroups()
        with self.assertRaises(TypeError):
            g.assign_cyclic(["E0"], n_groups=True, quiet=QUIET)

    def test_cyclic_zero_n_groups_raises(self):
        g = NMGroups()
        with self.assertRaises(ValueError):
            g.assign_cyclic(["E0"], n_groups=0, quiet=QUIET)

    def test_cyclic_negative_n_groups_raises(self):
        g = NMGroups()
        with self.assertRaises(ValueError):
            g.assign_cyclic(["E0"], n_groups=-1, quiet=QUIET)


# =========================================================================
# NMGroups — get_items
# =========================================================================

class TestNMGroupsGetItems(unittest.TestCase):

    def setUp(self):
        self.g, self.names = _make_groups(n=6)

    def test_group_0_members(self):
        self.assertEqual(self.g.get_items(0), ["E0", "E3"])

    def test_group_1_members(self):
        self.assertEqual(self.g.get_items(1), ["E1", "E4"])

    def test_group_2_members(self):
        self.assertEqual(self.g.get_items(2), ["E2", "E5"])

    def test_nonexistent_group_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.g.get_items(99)

    def test_key_error_message_lists_existing_groups(self):
        try:
            self.g.get_items(99)
        except KeyError as e:
            self.assertIn("does not exist", str(e))
            self.assertIn("[0, 1, 2]", str(e))

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            self.g.get_items("0")

    def test_bool_raises(self):
        with self.assertRaises(TypeError):
            self.g.get_items(True)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            self.g.get_items(-1)


# =========================================================================
# NMGroups — get_group
# =========================================================================

class TestNMGroupsGetGroup(unittest.TestCase):

    def setUp(self):
        self.g, _ = _make_groups(n=3)

    def test_returns_correct_group(self):
        self.assertEqual(self.g.get_group("E0"), 0)
        self.assertEqual(self.g.get_group("E1"), 1)
        self.assertEqual(self.g.get_group("E2"), 2)

    def test_unassigned_returns_none(self):
        self.assertIsNone(self.g.get_group("E99"))

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            self.g.get_group(0)


# =========================================================================
# NMGroups — unassign
# =========================================================================

class TestNMGroupsUnassign(unittest.TestCase):

    def setUp(self):
        self.g, _ = _make_groups(n=3)

    def test_unassign_removes_entry(self):
        self.g.unassign("E0", quiet=QUIET)
        self.assertIsNone(self.g.get_group("E0"))

    def test_unassign_reduces_len(self):
        before = len(self.g)
        self.g.unassign("E0", quiet=QUIET)
        self.assertEqual(len(self.g), before - 1)

    def test_unassign_unknown_error_true_raises(self):
        with self.assertRaises(KeyError):
            self.g.unassign("E99", error=True, quiet=QUIET)

    def test_unassign_unknown_error_false_silent(self):
        self.g.unassign("E99", error=False, quiet=QUIET)  # no exception

    def test_unassign_bad_type_raises(self):
        with self.assertRaises(TypeError):
            self.g.unassign(0, quiet=QUIET)


# =========================================================================
# NMGroups — clear
# =========================================================================

class TestNMGroupsClear(unittest.TestCase):

    def test_clear_empties_all(self):
        g, _ = _make_groups(n=6)
        g.clear(quiet=QUIET)
        self.assertEqual(len(g), 0)

    def test_clear_already_empty_no_error(self):
        g = NMGroups()
        g.clear(quiet=QUIET)  # no exception

    def test_get_group_after_clear_returns_none(self):
        g, _ = _make_groups(n=3)
        g.clear(quiet=QUIET)
        self.assertIsNone(g.get_group("E0"))


# =========================================================================
# NMGroups — rename_item
# =========================================================================

class TestNMGroupsRenameItem(unittest.TestCase):

    def test_rename_updates_key(self):
        g, _ = _make_groups(n=3)
        g.rename_item("E0", "E0_new")
        self.assertEqual(g.get_group("E0_new"), 0)
        self.assertIsNone(g.get_group("E0"))

    def test_rename_preserves_group_number(self):
        g, _ = _make_groups(n=3)
        g.rename_item("E2", "E2_renamed")
        self.assertEqual(g.get_group("E2_renamed"), 2)

    def test_rename_unknown_is_noop(self):
        g, _ = _make_groups(n=3)
        g.rename_item("E99", "E99_new")  # no exception, no change
        self.assertEqual(len(g), 3)


# =========================================================================
# NMGroups — n_groups
# =========================================================================

class TestNMGroupsGroupNumbers(unittest.TestCase):

    def test_group_numbers_cyclic(self):
        g, _ = _make_groups(n=6)
        self.assertEqual(g.group_numbers, [0, 1, 2])

    def test_group_numbers_sorted(self):
        g = NMGroups()
        g.assign("E0", 2, quiet=QUIET)
        g.assign("E1", 0, quiet=QUIET)
        g.assign("E2", 1, quiet=QUIET)
        self.assertEqual(g.group_numbers, [0, 1, 2])

    def test_group_numbers_empty(self):
        g = NMGroups()
        self.assertEqual(g.group_numbers, [])

    def test_group_numbers_after_unassign(self):
        g, _ = _make_groups(n=3)   # groups 0, 1, 2
        g.unassign("E2", quiet=QUIET)   # group 2 now empty
        self.assertEqual(g.group_numbers, [0, 1])


class TestNMGroupsNGroups(unittest.TestCase):

    def test_n_groups_cyclic(self):
        g, _ = _make_groups(n=6)
        self.assertEqual(g.n_groups, 3)

    def test_n_groups_after_unassign(self):
        g, _ = _make_groups(n=3)   # groups 0, 1, 2 each have 1 member
        g.unassign("E2", quiet=QUIET)   # group 2 now empty
        self.assertEqual(g.n_groups, 2)

    def test_n_groups_one_group(self):
        g = NMGroups()
        g.assign_cyclic(["E0", "E1", "E2"], n_groups=1, quiet=QUIET)
        self.assertEqual(g.n_groups, 1)


# =========================================================================
# NMGroups — contains / iter / len
# =========================================================================

class TestNMGroupsCollectionProtocol(unittest.TestCase):

    def setUp(self):
        self.g, self.names = _make_groups(n=3)

    def test_contains_assigned(self):
        self.assertIn("E0", self.g)

    def test_not_contains_unassigned(self):
        self.assertNotIn("E99", self.g)

    def test_contains_non_string_false(self):
        self.assertNotIn(0, self.g)

    def test_len(self):
        self.assertEqual(len(self.g), 3)

    def test_iter(self):
        self.assertEqual(set(self.g), {"E0", "E1", "E2"})


# =========================================================================
# NMGroups — dict-like dunder access
# =========================================================================

class TestNMGroupsDunderAccess(unittest.TestCase):

    def setUp(self):
        self.g, self.names = _make_groups(n=3)  # E0→0, E1→1, E2→2

    def test_getitem_returns_group(self):
        self.assertEqual(self.g["E0"], 0)
        self.assertEqual(self.g["E1"], 1)

    def test_getitem_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            _ = self.g["E99"]

    def test_setitem_assigns(self):
        self.g["E0"] = 2
        self.assertEqual(self.g.get_group("E0"), 2)

    def test_setitem_bad_group_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            self.g["E0"] = -1

    def test_delitem_unassigns(self):
        del self.g["E0"]
        self.assertNotIn("E0", self.g)

    def test_delitem_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            del self.g["E99"]


# =========================================================================
# NMGroups — equality
# =========================================================================

class TestNMGroupsEquality(unittest.TestCase):

    def test_equal_empty(self):
        self.assertEqual(NMGroups(), NMGroups())

    def test_equal_same_assignments(self):
        g1, _ = _make_groups(n=3)
        g2, _ = _make_groups(n=3)
        self.assertEqual(g1, g2)

    def test_unequal_different_assignments(self):
        g1, _ = _make_groups(n=3)
        g2, _ = _make_groups(n=3)
        g2.assign("E0", 2, quiet=QUIET)
        self.assertNotEqual(g1, g2)

    def test_not_equal_to_non_nmgroups(self):
        g, _ = _make_groups(n=3)
        self.assertNotEqual(g, "not a group")


# =========================================================================
# NMGroups — deepcopy
# =========================================================================

class TestNMGroupsDeepCopy(unittest.TestCase):

    def test_deepcopy_preserves_assignments(self):
        g, _ = _make_groups(n=6)
        g2 = copy.deepcopy(g)
        self.assertEqual(g, g2)

    def test_deepcopy_is_independent(self):
        g, _ = _make_groups(n=3)
        g2 = copy.deepcopy(g)
        g2.assign("E0", 2, quiet=QUIET)
        self.assertEqual(g.get_group("E0"), 0)  # original unchanged

    def test_deepcopy_parent_cleared(self):
        class _FakeParent:
            path_str = "root"
        g = NMGroups(parent=_FakeParent())
        g2 = copy.deepcopy(g)
        self.assertIsNone(g2._parent)

    def test_copy_method(self):
        g, _ = _make_groups(n=3)
        g2 = g.copy()
        self.assertEqual(g, g2)


# =========================================================================
# NMEpochContainer.groups integration
# =========================================================================

class TestNMEpochContainerGroups(unittest.TestCase):
    """groups property lives on NMEpochContainer and syncs with container ops."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)
        folder = self.nm.project.folders.new("f0")
        self.ds = folder.dataseries.new("Record")
        # Create 6 epochs
        for i in range(6):
            self.ds.epochs.new("E%d" % i)
        self.epoch_names = ["E%d" % i for i in range(6)]
        self.ds.epochs.groups.assign_cyclic(
            self.epoch_names, n_groups=3, quiet=QUIET
        )

    def test_groups_property_exists(self):
        self.assertIsInstance(self.ds.epochs.groups, NMGroups)

    def test_groups_cyclic_assignment(self):
        grp = self.ds.epochs.groups
        for i, name in enumerate(self.epoch_names):
            self.assertEqual(grp.get_group(name), i % 3)

    def test_get_items_returns_correct_epochs(self):
        grp = self.ds.epochs.groups
        members = grp.get_items(0)
        self.assertEqual(members, ["E0", "E3"])

    def test_pop_removes_from_groups(self):
        self.ds.epochs.pop("E0", quiet=QUIET)
        self.assertIsNone(self.ds.epochs.groups.get_group("E0"))

    def test_pop_unassigned_epoch_no_error(self):
        # Create epoch not in any group
        self.ds.epochs.new(quiet=QUIET)   # E6
        self.ds.epochs.pop("E6", quiet=QUIET)   # no exception

    def test_clear_removes_all_group_assignments(self):
        self.ds.epochs.clear(quiet=QUIET)
        self.assertEqual(len(self.ds.epochs.groups), 0)

    def test_deepcopy_includes_groups(self):
        ds2 = copy.deepcopy(self.ds)
        grp2 = ds2.epochs.groups
        for i, name in enumerate(self.epoch_names):
            self.assertEqual(grp2.get_group(name), i % 3)

    def test_deepcopy_groups_independent(self):
        ds2 = copy.deepcopy(self.ds)
        ds2.epochs.groups.assign("E0", 2, quiet=QUIET)
        self.assertEqual(self.ds.epochs.groups.get_group("E0"), 0)

    def test_rename_epoch_raises_runtime_error(self):
        # Epochs are not renameable (rename_on=False); groups are therefore
        # never left with stale keys via the standard API
        with self.assertRaises(RuntimeError):
            self.ds.epochs.rename("E0", "E0_new")


if __name__ == "__main__":
    unittest.main(verbosity=2)
