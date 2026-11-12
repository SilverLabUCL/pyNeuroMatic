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
import re

import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_history as nmh
from pyneuromatic.core.nm_notes import NMNotes
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


GROUP_PREFIX = "Group"
_GROUP_NAME = re.compile(r"(?:group|g)\s*(\d+)", re.IGNORECASE)  # Group0, G0


def group_name(number: int) -> str:
    """A group's name, as sets are named Set0, Set1...: e.g. "Group0" (#509)."""
    return "%s%d" % (GROUP_PREFIX, number)


def group_number(key) -> int | None:
    """The group number of *key*: a number (0), or a name ("Group0"; any case,
    "Group 0" too; or the shorthand "G0", for scripts); None if neither."""
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key if key >= 0 else None
    if isinstance(key, str):
        match = _GROUP_NAME.fullmatch(key.strip())
        if match:
            return int(match.group(1))
    return None


def group_names(numbers) -> str:
    """Groups as text: "Group0", "Group0, Group2" or a run "Group0–Group3"."""
    numbers = sorted(numbers)
    if len(numbers) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return "%s–%s" % (group_name(numbers[0]), group_name(numbers[-1]))
    return ", ".join(group_name(n) for n in numbers)


class NMGroup:
    """One group of an NMGroups, as ``groups['Group0']`` (or ``groups[0]``)
    gives it (#509), like ``epochs['E0']``: its name, its items (the epochs)
    and its notes. A read-only list of its items: iterate, ``len``, ``in`` (by
    name or object). Its items change through the groups' methods (``assign``,
    ``unassign``...), which are recorded.
    """

    def __init__(self, groups: NMGroups, number: int) -> None:
        self._groups = groups
        self._number = number

    @property
    def name(self) -> str:
        return group_name(self._number)

    @property
    def number(self) -> int:
        return self._number

    @property
    def names(self) -> list[str]:
        """Its items' names (e.g. epochs "E0", "E2")."""
        return self._groups.get_items(self._number)

    @property
    def items(self) -> list:
        """Its items: the epochs (or names, outside a container)."""
        parent = self._groups._parent
        names = self.names
        try:
            return [parent[name] for name in names] if parent is not None else names
        except (KeyError, TypeError):
            return names

    @property
    def notes(self) -> NMNotes:
        return self._groups.notes(self._number)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, item: object) -> bool:
        name = item if isinstance(item, str) else getattr(item, "name", None)
        return name in self.names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMGroup):
            return NotImplemented
        return other._groups is self._groups and other._number == self._number

    def __hash__(self) -> int:
        return hash((id(self._groups), self._number))

    def __repr__(self) -> str:
        return "NMGroup(%r, n=%d)" % (self.name, len(self))


class NMGroups:
    """Mutually exclusive integer group assignments for container item names.

    By group (#509), as NMSets is by set: ``groups['Group0']`` (or
    ``groups[0]``) is a group (:class:`NMGroup`), ``len(groups)`` the number
    of groups, iterating gives their names (Group0, Group1...) and
    ``'Group0' in groups`` asks if a group exists. An item's group:
    ``get_group('E0')``.

    Stores a mapping ``{item_name: group_number}`` where each name belongs
    to at most one group, along with explicitly created group numbers that
    may not have any members yet. Group numbers are non-negative integers.

    Typical use case — a repeating I-V relation of *n* voltage steps::

        groups.assign_cyclic(epoch_names, n_groups=5)
        # assigns 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...

    Items of a single group can be retrieved for averaging or selection::

        groups.get_items(0)   # → ["RecordA0", "RecordA5", ...]

    Combined set+group selection (AND) is a plain Python set intersection::

        set_items   = dataseries.epochs.sets.get_items("set1", get_keys=True)
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
        self._declared_groups: set[int] = set()
        # Notes describing each group (e.g. "-80 mV steps"), by group number
        self._notes: dict[int, NMNotes] = {}

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

    @property
    def history_path(self) -> str:
        parent = getattr(self._parent, "history_path", None)
        return parent + ".groups" if parent else self._name

    # By group (#509): groups['Group0'] / groups[0], len, iteration, in

    def __len__(self) -> int:
        return len(self.group_numbers)

    def __iter__(self):
        return iter([group_name(n) for n in self.group_numbers])

    def __contains__(self, key: object) -> bool:
        number = group_number(key)
        return number is not None and number in self.group_numbers

    def __getitem__(self, key) -> NMGroup:
        number = group_number(key)
        if number is None or number not in self.group_numbers:
            raise KeyError("group %r does not exist; existing groups: %s"
                           % (key, ", ".join(self) or "none"))
        return NMGroup(self, number)

    def __setitem__(self, key, item_names) -> None:
        """``groups['Group0'] = ['E0', 'E3']``: the group's items are these
        (others of it unassigned), as ``sets['Set0'] = [...]`` (#509)."""
        number = group_number(key)
        if number is None:
            raise KeyError("not a group: %r (e.g. 'Group0' or 0)" % (key,))
        names = [item_names] if isinstance(item_names, str) else list(item_names)
        if number in self.group_numbers:
            others = [n for n in self.get_items(number) if n not in names]
            if others:
                self.unassign(others)
        if names:
            self.assign(names, number)
        elif number not in self.group_numbers:
            self.add_group(number)

    def __delitem__(self, key) -> None:
        """``del groups['Group0']``: delete the group (its items are kept)."""
        number = group_number(key)
        if number is None or number not in self.group_numbers:
            raise KeyError("group %r does not exist" % (key,))
        self.delete_group(number)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMGroups):
            return NotImplemented
        return (
            self._map == other._map
            and self._declared_groups == other._declared_groups
            and self._noted() == other._noted()
        )

    def _noted(self) -> dict[int, list]:
        """Notes of each group that has any (empty notes don't count)."""
        return {group: list(notes) for group, notes in self._notes.items() if len(notes)}

    def __repr__(self) -> str:
        return "%s(%r, groups=%r)" % (
            self.__class__.__name__,
            dict(self._map),
            self.group_numbers,
        )

    def __deepcopy__(self, memo: dict) -> "NMGroups":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._name = self._name
        result._parent = None   # parent reference not copied (same pattern as NMSets)
        result._map = dict(self._map)   # str→int: shallow copy is sufficient
        result._declared_groups = set(self._declared_groups)
        result._notes = copy.deepcopy(self._notes, memo)
        return result

    @property
    def _nm_cmd_path(self) -> str | None:
        """E.g. "folders['Demo'].dataseries['Rec'].epochs.groups"; None when not in
        the manager (not recorded)."""
        path = getattr(self._parent, "_epochs_cmd_path", None)
        return None if path is None else path + ".groups"

    def _echo(self, message: str, quiet: bool = nmc.QUIET) -> None:
        """A history message that says only what its command does: written
        only when no command is recorded (#441)."""
        if self._nm_cmd_path is None:
            nmh.history(message, path=self.history_path, quiet=quiet)

    def _record(self, call: str) -> None:
        """Record a command on these groups, e.g. "assign(['E0', 'E1'], 1)"."""
        path = self._nm_cmd_path
        if path is not None:
            nmch.add_nm_command("%s.%s" % (path, call))

    @staticmethod
    def _as_list(value) -> tuple[list, bool]:
        """(*value* as a list, whether it was one) for an item or a list."""
        if isinstance(value, (list, tuple)):
            return list(value), True
        return [value], False

    @staticmethod
    def _names_text(names: list) -> str:
        """e.g. "'E0', 'E1', 'E2'" or "'E0', 'E1', ... 'E99' (100)"."""
        if len(names) <= 6:
            return ", ".join("'%s'" % n for n in names)
        return "'%s', '%s', ... '%s' (%d)" % (names[0], names[1], names[-1], len(names))

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

    @nmch.records_command
    def add_group(
        self,
        group: int | list[int],
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Create an empty group with the given non-negative number (or
        several, e.g. [0, 1, 2])."""
        numbers, many = self._as_list(group)
        for number in numbers:
            self._check_group(number)
            if number in self.group_numbers:
                raise ValueError("group %d already exists" % number)
        self._declared_groups.update(numbers)
        self._echo("created %s" % group_names(numbers), quiet)
        self._record("add_group(%r)" % (numbers if many else numbers[0],))

    @nmch.records_command
    def assign(
        self,
        item_name: str | list[str],
        group: int,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Assign *item_name* to *group* (replaces any previous assignment).

        Args:
            item_name: Name of the item (e.g. epoch name), or a list of names
                (one command for them all).
            group: Non-negative integer group number.
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_name* is not a string or *group* is not an int.
            ValueError: If *group* is negative.
        """
        names, many = self._as_list(item_name)
        for name in names:
            if not isinstance(name, str):
                raise TypeError(nmu.type_error_str(name, "item_name", "string"))
        self._check_group(group)
        # A group left empty by moving its items is kept (until deleted)
        self._declared_groups.update(self._map[n] for n in names if n in self._map)
        for name in names:
            self._map[name] = group
        self._echo("%s → %s" % (self._names_text(names), group_name(group)), quiet)
        self._record("assign(%s, %r)" % (nmu.names_repr(names) if many else repr(names[0]), group))

    @nmch.records_command
    def assign_cyclic(
        self,
        item_names: list[str],
        n_groups: int,
        block: int = 1,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Assign *item_names* cyclically to groups ``0 … n_groups-1``.

        Each group number is repeated *block* times before the next one.

        Example::

            groups.assign_cyclic(["E0","E1","E2","E3","E4","E5"], n_groups=3)
            # E0→0, E1→1, E2→2, E3→0, E4→1, E5→2
            groups.assign_cyclic(names_of_12_epochs, n_groups=2, block=3)
            # 0 0 0 1 1 1 0 0 0 1 1 1

        Args:
            item_names: Ordered list of item names.
            n_groups: Number of groups (must be >= 1).
            block: Items per group before the next group (must be >= 1).
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_names* is not a list, or *n_groups* or *block*
                is not an int.
            ValueError: If *n_groups* or *block* < 1.
        """
        if not isinstance(item_names, list):
            raise TypeError(nmu.type_error_str(item_names, "item_names", "list"))
        if isinstance(n_groups, bool) or not isinstance(n_groups, int):
            raise TypeError(nmu.type_error_str(n_groups, "n_groups", "int"))
        if n_groups < 1:
            raise ValueError("n_groups must be >= 1, got %d" % n_groups)
        if isinstance(block, bool) or not isinstance(block, int):
            raise TypeError(nmu.type_error_str(block, "block", "int"))
        if block < 1:
            raise ValueError("block must be >= 1, got %d" % block)
        self._declared_groups.update(  # groups left empty are kept
            self._map[n] for n in item_names if n in self._map)
        for i, name in enumerate(item_names):
            self.assign(name, (i // block) % n_groups, quiet=True)
        # What the command doesn't say: how many each group now has
        sizes = [0] * n_groups
        for i in range(len(item_names)):
            sizes[(i // block) % n_groups] += 1
        if item_names:
            nmh.history(
                "%d groups of %d" % (n_groups, sizes[0]) if len(set(sizes)) == 1
                else "group sizes %s" % ", ".join(map(str, sizes)),
                path=self.history_path,
                quiet=quiet,
            )
        self._record("assign_cyclic(%s, %r%s)" % (
            nmu.names_repr(list(item_names)), n_groups,
            nmu.kwargs_repr(self.assign_cyclic, {"block": block})))

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

    @nmch.records_command
    def unassign(
        self,
        item_name: str | list[str],
        error: bool = True,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Remove *item_name* from its group; the group is kept, even if it
        empties (delete_group removes a group).

        Args:
            item_name: Name of the item to unassign, or a list of names (one
                command for them all).
            error: If True, raise KeyError when *item_name* is not assigned.
            quiet: Suppress history output.

        Raises:
            TypeError: If *item_name* is not a string.
            KeyError: If *item_name* is not assigned and *error* is True.
        """
        names, many = self._as_list(item_name)
        for name in names:
            if not isinstance(name, str):
                raise TypeError(nmu.type_error_str(name, "item_name", "string"))
            if name not in self._map and error:
                raise KeyError("'%s' is not assigned to any group" % name)
        names = [name for name in names if name in self._map]
        if not names:
            return
        groups = sorted({self._map.pop(name) for name in names})
        self._declared_groups.update(groups)  # kept, even if empty (until deleted)
        self._echo("unassigned %s from %s"
                   % (self._names_text(names), group_names(groups)), quiet)
        self._record("unassign(%s)" % (nmu.names_repr(names) if many else repr(names[0])))

    @nmch.records_command
    def clear(self, quiet: bool = nmc.QUIET) -> None:
        """Remove all group assignments.

        Args:
            quiet: Suppress history output.
        """
        if not self._map and not self._declared_groups and not self._noted():
            return
        self._map.clear()
        self._declared_groups.clear()
        self._notes.clear()
        self._echo("cleared all assignments", quiet)
        self._record("clear()")

    @nmch.records_command
    def clear_group(self, group: int | list[int], quiet: bool = nmc.QUIET) -> None:
        """Take every item out of *group* (or several groups); the groups are
        kept, empty.

        Args:
            group: Group number, or a list of numbers.
            quiet: Suppress history output.

        Raises:
            TypeError: If *group* is not an int.
            KeyError: If *group* does not exist.
        """
        numbers, many = self._as_list(group)
        for number in numbers:
            self._check_group(number)
            if number not in self.group_numbers:
                raise KeyError(
                    "group %d does not exist; existing groups: %s"
                    % (number, self.group_numbers)
                )
        for number in numbers:
            for name in [n for n, g in self._map.items() if g == number]:
                del self._map[name]
        self._declared_groups.update(numbers)
        self._echo("cleared %s" % group_names(numbers), quiet)
        self._record("clear_group(%r)" % (numbers if many else numbers[0],))

    @nmch.records_command
    def delete_group(self, group: int | list[int], quiet: bool = nmc.QUIET) -> None:
        """Delete *group*: its items are unassigned (the items are kept).

        Args:
            group: Group number to delete, or a list of numbers.
            quiet: Suppress history output.

        Raises:
            TypeError: If *group* is not an int.
            KeyError: If *group* does not exist.
        """
        numbers, many = self._as_list(group)
        for number in numbers:
            self._check_group(number)
            if number not in self.group_numbers:
                raise KeyError(
                    "group %d does not exist; existing groups: %s"
                    % (number, self.group_numbers)
                )
        for number in numbers:
            for name in [n for n, g in self._map.items() if g == number]:
                del self._map[name]
            self._declared_groups.discard(number)
            self._notes.pop(number, None)
        self._echo("deleted %s" % group_names(numbers), quiet)
        self._record("delete_group(%r)" % (numbers if many else numbers[0],))

    def notes(self, group: int) -> NMNotes:
        """Notes for *group* (e.g. what its epochs are). A group with notes
        is kept even when it has no items; deleting it drops them.

        Raises:
            TypeError: If *group* is not an int.
            KeyError: If *group* does not exist.
        """
        self._check_group(group)
        if group not in self.group_numbers:
            raise KeyError(
                "group %d does not exist; existing groups: %s"
                % (group, self.group_numbers)
            )
        if group not in self._notes:
            self._notes[group] = NMNotes()
        return self._notes[group].set_owner(self, group)  # a command's path (#510)

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
        """Sorted group numbers, including empty groups created explicitly
        or described with notes."""
        return sorted(set(self._map.values()).union(self._declared_groups, self._noted()))

    @property
    def n_groups(self) -> int:
        """Number of distinct group numbers currently in use."""
        return len(self.group_numbers)
