# -*- coding: utf-8 -*-
"""
NMGroups - Mutually exclusive integer group assignments for NMObject names.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiological Data. Front. Neuroinform. 12:14.
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

import copy

import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


class NMGroups:
    """Mutually exclusive integer group assignments for container item names.

    Stores a mapping ``{item_name: group_number}`` where each name belongs
    to at most one group.  Group numbers are non-negative integers.

    Typical use case — a repeating I-V relation of *n* voltage steps::

        groups.assign_cyclic(epoch_names, n_groups=5)
        # assigns 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...

    Items of a single group can be retrieved for averaging or selection::

        groups.get_items(0)   # → ["RecordA0", "RecordA5", ...]

    Combined set+group selection (AND) is a plain Python set intersection::

        set_items   = dataseries.epochs.sets.get("set1", get_keys=True)
        group_items = dataseries.epochs.groups.get_items(0)
        selected    = set(set_items) & set(group_items)
    """

    def __init__(
        self,
        name: str = "NMGroups0",
        parent: object | None = None,
    ) -> None:
        self._name = name
        self._parent = parent
        self._map: dict[str, int] = {}  # item_name → group_number

    # ------------------------------------------------------------------
    # Identity / dunder

    @property
    def name(self) -> str:
        return self._name

    @property
    def path_str(self) -> str:
        if self._parent is not None and hasattr(self._parent, "path_str"):
            return self._parent.path_str + ".groups"
        return self._name

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def __contains__(self, item_name: object) -> bool:
        if not isinstance(item_name, str):
            return False
        return item_name in self._map

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMGroups):
            return NotImplemented
        return self._map == other._map

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, dict(self._map))

    def __deepcopy__(self, memo: dict) -> "NMGroups":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._name = self._name
        result._parent = None   # parent reference not copied (same pattern as NMSets)
        result._map = dict(self._map)   # str→int: shallow copy is sufficient
        return result

    def copy(self) -> "NMGroups":
        """Return a deep copy (parent reference cleared)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Validation helpers

    @staticmethod
    def _check_group(group: object) -> None:
        """Raise TypeError/ValueError for invalid group values."""
        if isinstance(group, bool) or not isinstance(group, int):
            raise TypeError(
                "group: expected non-negative int, got %s"
                % type(group).__name__
            )
        if group < 0:
            raise ValueError("group must be >= 0, got %d" % group)

    # ------------------------------------------------------------------
    # Core operations

    def assign(
        self,
        item_name: str,
        group: int,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Assign *item_name* to *group* (replaces any previous assignment).

        Args:
            item_name: Name of the item (e.g. epoch name).
            group: Non-negative integer group number.
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_name* is not a string or *group* is not an int.
            ValueError: If *group* is negative.
        """
        if not isinstance(item_name, str):
            raise TypeError(nmu.type_error_str(item_name, "item_name", "string"))
        self._check_group(group)
        self._map[item_name] = group
        nmh.history(
            "groups: '%s' → %d" % (item_name, group),
            path=self.path_str,
            quiet=quiet,
        )

    def assign_cyclic(
        self,
        item_names: list[str],
        n_groups: int,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Assign *item_names* cyclically to groups ``0 … n_groups-1``.

        Example::

            groups.assign_cyclic(["E0","E1","E2","E3","E4","E5"], n_groups=3)
            # E0→0, E1→1, E2→2, E3→0, E4→1, E5→2

        Args:
            item_names: Ordered list of item names.
            n_groups: Number of groups (must be >= 1).
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_names* is not a list or *n_groups* is not an int.
            ValueError: If *n_groups* < 1.
        """
        if not isinstance(item_names, list):
            raise TypeError(nmu.type_error_str(item_names, "item_names", "list"))
        if isinstance(n_groups, bool) or not isinstance(n_groups, int):
            raise TypeError(nmu.type_error_str(n_groups, "n_groups", "int"))
        if n_groups < 1:
            raise ValueError("n_groups must be >= 1, got %d" % n_groups)
        for i, name in enumerate(item_names):
            self.assign(name, i % n_groups, quiet=True)
        nmh.history(
            "groups: cyclic assignment, %d names, %d groups"
            % (len(item_names), n_groups),
            path=self.path_str,
            quiet=quiet,
        )

    def get_items(self, group: int) -> list[str]:
        """Return item names assigned to *group*, in insertion order.

        Args:
            group: Non-negative integer group number.

        Returns:
            List of item names assigned to *group*.

        Raises:
            TypeError/ValueError: If *group* is an invalid type or negative.
            KeyError: If *group* does not exist, with a message listing the
                existing group numbers.
        """
        self._check_group(group)
        if group not in self.group_numbers:
            raise KeyError(
                "group %d does not exist; existing groups: %s"
                % (group, self.group_numbers)
            )
        return [name for name, g in self._map.items() if g == group]

    def get_group(self, item_name: str) -> int | None:
        """Return the group number for *item_name*, or ``None`` if unassigned.

        Args:
            item_name: Name of the item.

        Raises:
            TypeError: If *item_name* is not a string.
        """
        if not isinstance(item_name, str):
            raise TypeError(nmu.type_error_str(item_name, "item_name", "string"))
        return self._map.get(item_name)

    def unassign(
        self,
        item_name: str,
        error: bool = True,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Remove *item_name* from its group.

        Args:
            item_name: Name of the item to unassign.
            error: If True, raise KeyError when *item_name* is not assigned.
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_name* is not a string.
            KeyError: If *item_name* is not assigned and *error* is True.
        """
        if not isinstance(item_name, str):
            raise TypeError(nmu.type_error_str(item_name, "item_name", "string"))
        if item_name not in self._map:
            if error:
                raise KeyError(
                    "'%s' is not assigned to any group" % item_name
                )
            return
        group = self._map.pop(item_name)
        nmh.history(
            "groups: unassigned '%s' from group %d" % (item_name, group),
            path=self.path_str,
            quiet=quiet,
        )

    def clear(self, quiet: bool = nmc.QUIET) -> None:
        """Remove all group assignments.

        Args:
            quiet: Suppress history output.
        """
        if not self._map:
            return
        self._map.clear()
        nmh.history(
            "groups: cleared all assignments",
            path=self.path_str,
            quiet=quiet,
        )

    def rename_item(self, old_name: str, new_name: str) -> None:
        """Rename an item across all group assignments.

        Called automatically by the container when an item is renamed.
        If *old_name* has no assignment, this is a no-op.

        Args:
            old_name: Current item name.
            new_name: New item name.
        """
        if old_name in self._map:
            self._map[new_name] = self._map.pop(old_name)

    # ------------------------------------------------------------------
    # Properties

    @property
    def group_numbers(self) -> list[int]:
        """Sorted list of distinct group numbers currently assigned."""
        return sorted(set(self._map.values()))

    @property
    def n_groups(self) -> int:
        """Number of distinct group numbers currently in use."""
        return len(self.group_numbers)
