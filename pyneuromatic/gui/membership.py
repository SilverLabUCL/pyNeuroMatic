"""Checkbox list for editing a set's or group's members in the browser.

Every item of a container (e.g. every epoch of a data series, or every data
array of a folder) is listed with a checkbox; ticking adds the item to the
set (or group), unticking removes it.
"""
from __future__ import annotations

from typing import Any

from PyQt6 import QtCore

from pyneuromatic.core.nm_object_container import NMObjectContainer


def unassign_from_group(groups, names, group: int) -> None:
    """Take *names* out of *group*, keeping the group even if it empties.

    A group made by assigning epochs (rather than created empty) would
    otherwise disappear with its last epoch, and with it the browser row.
    """
    for name in names:
        if groups.get_group(name) == group:
            groups.unassign(name, error=False, quiet=True)
    if group not in groups.group_numbers:
        groups.add_group(group, quiet=True)


def set_item_group(groups, name: str, group: int | None) -> None:
    """Put *name* in *group* (None: no group), keeping a group it empties."""
    old = groups.get_group(name)
    if old == group:
        return
    if group is None:
        unassign_from_group(groups, [name], old)
        return
    groups.assign(name, group, quiet=True)
    if old is not None and old not in groups.group_numbers:
        groups.add_group(old, quiet=True)


class MembershipModel(QtCore.QAbstractTableModel):
    """Every item of *container*, with a checkbox for membership of a target.

    The target is a set (``set_name``) or, for epochs, a group (``group``).
    An item belongs to at most one group, so ticking an item that is in
    another group moves it; a second column shows each item's group.
    """

    # The set or group changed (number of items changed)
    membership_changed = QtCore.pyqtSignal(int)
    # A row's checkbox was clicked: row, now checked
    toggled = QtCore.pyqtSignal(int, bool)

    def __init__(
        self,
        container: NMObjectContainer,
        set_name: str | None = None,
        group: int | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if (set_name is None) == (group is None):
            raise ValueError("give a set name or a group number")
        self.container = container
        self.set_name = set_name
        self.group = group
        self._names = list(container.keys())
        self._members: set[str] = set()
        self._reload_members()

    # --- target -------------------------------------------------------------

    @property
    def is_group(self) -> bool:
        return self.group is not None

    @property
    def target_label(self) -> str:
        return f"Group {self.group}" if self.is_group else str(self.set_name)

    def same_target(self, container, set_name=None, group=None) -> bool:
        return (container is self.container and set_name == self.set_name
                and group == self.group)

    def _reload_members(self) -> None:
        if self.is_group:
            self._members = set(self.container.groups.get_items(self.group))
        else:
            self._members = set(
                self.container.sets.get_items(self.set_name, get_keys=True) or []
            )

    def reload(self) -> None:
        """Re-read the items and members, after a change made elsewhere."""
        names = list(self.container.keys())
        if names != self._names:
            self.beginResetModel()
            self._names = names
            self._reload_members()
            self.endResetModel()
            return
        self._reload_members()
        if self._names:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1,
                                                               self.columnCount() - 1))

    def members(self) -> list[str]:
        """Member names, in container order."""
        return [name for name in self._names if name in self._members]

    # --- Qt model -----------------------------------------------------------

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._names)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else (2 if self.is_group else 1)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return ["Name", "Group"][section]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._names[index.row()]
        if index.column() == 0:
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return name
            if role == QtCore.Qt.ItemDataRole.CheckStateRole:
                return (QtCore.Qt.CheckState.Checked if name in self._members
                        else QtCore.Qt.CheckState.Unchecked)
        elif role == QtCore.Qt.ItemDataRole.DisplayRole:
            group = self.container.groups.get_group(name)
            return "" if group is None else str(group)
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index: QtCore.QModelIndex, value: Any,
                role: int = QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if role != QtCore.Qt.ItemDataRole.CheckStateRole or index.column() != 0:
            return False
        checked = QtCore.Qt.CheckState(value) == QtCore.Qt.CheckState.Checked
        self.set_checked([index.row()], checked)
        self.toggled.emit(index.row(), checked)
        return True

    # --- editing ------------------------------------------------------------

    def set_checked(self, rows, checked: bool) -> int:
        """Add (or remove) the items in *rows*; returns how many changed."""
        names = [self._names[row] for row in rows if 0 <= row < len(self._names)]
        change = [n for n in names if (n in self._members) != checked]
        if not change:
            return 0
        if self.is_group:
            groups = self.container.groups
            if checked:
                before = set(groups.group_numbers)
                for name in change:
                    groups.assign(name, self.group, quiet=True)
                # Groups that moved their last epoch here stay (empty)
                for group in before.difference(groups.group_numbers):
                    groups.add_group(group, quiet=True)
            else:
                unassign_from_group(groups, change, self.group)
        else:
            items = [self.container[name] for name in change]
            sets = self.container.sets
            (sets.add if checked else sets.remove)(self.set_name, items)
        self._reload_members()
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1,
                                                           self.columnCount() - 1))
        self.membership_changed.emit(len(change))
        return len(change)

    def check_all(self) -> int:
        return self.set_checked(range(len(self._names)), True)

    def check_ungrouped(self) -> int:
        """Tick every item that is in no group (other groups are kept)."""
        groups = self.container.groups
        return self.set_checked(
            [row for row, name in enumerate(self._names)
             if groups.get_group(name) is None],
            True,
        )

    def check_none(self) -> int:
        return self.set_checked(range(len(self._names)), False)

    def invert(self) -> int:
        """Tick every unticked item and untick every ticked one."""
        ticked = [row for row, name in enumerate(self._names) if name in self._members]
        unticked = [row for row, name in enumerate(self._names) if name not in self._members]
        # Remove first: for groups, adding never moves items out of this group
        changed = self.set_checked(ticked, False)
        return changed + self.set_checked(unticked, True)


__all__ = ["MembershipModel", "set_item_group", "unassign_from_group"]
