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


_HEADERS = {"name": "Name", "data": "Data", "group": "Group"}


def _is_epochs(container) -> bool:
    """True for a data series' epochs (they get a Data column)."""
    return hasattr(container, "groups")


def _arrays_text(container, name: str) -> str:
    """An epoch's data arrays, every channel: "RecordA0, RecordB0"."""
    epoch = container.get(name)
    return ", ".join(data.name for data in getattr(epoch, "data", []))


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

    def name_at(self, row: int) -> str:
        return self._names[row]

    def members(self) -> list[str]:
        """Member names, in container order."""
        return [name for name in self._names if name in self._members]

    # --- Qt model -----------------------------------------------------------

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._names)

    @property
    def _column_keys(self) -> list[str]:
        """Name, then Data for epochs (their arrays), then Group for groups."""
        keys = ["name"]
        if _is_epochs(self.container):
            keys.append("data")
        if self.is_group:
            keys.append("group")
        return keys

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._column_keys)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return _HEADERS[self._column_keys[section]]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._names[index.row()]
        key = self._column_keys[index.column()]
        if key == "name":
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return name
            if role == QtCore.Qt.ItemDataRole.CheckStateRole:
                return (QtCore.Qt.CheckState.Checked if name in self._members
                        else QtCore.Qt.CheckState.Unchecked)
        elif role == QtCore.Qt.ItemDataRole.DisplayRole:
            if key == "data":
                return _arrays_text(self.container, name)
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

    # The bulk ticks work on *rows* (e.g. those a filter shows), or every row

    def _rows(self, rows) -> list[int]:
        return list(range(len(self._names))) if rows is None else list(rows)

    def check_all(self, rows=None) -> int:
        return self.set_checked(self._rows(rows), True)

    def check_ungrouped(self, rows=None) -> int:
        """Tick every item that is in no group (other groups are kept)."""
        groups = self.container.groups
        return self.set_checked(
            [row for row in self._rows(rows) if groups.get_group(self._names[row]) is None],
            True,
        )

    def check_none(self, rows=None) -> int:
        return self.set_checked(self._rows(rows), False)

    def invert(self, rows=None) -> int:
        """Tick every unticked item and untick every ticked one."""
        rows = self._rows(rows)
        ticked = [row for row in rows if self._names[row] in self._members]
        unticked = [row for row in rows if self._names[row] not in self._members]
        # Remove first: for groups, adding never moves items out of this group
        changed = self.set_checked(ticked, False)
        return changed + self.set_checked(unticked, True)


class ChoiceModel(QtCore.QAbstractTableModel):
    """Every item of *container* with a checkbox, to choose several at once
    (e.g. to delete them). Unlike :class:`MembershipModel`, a tick changes
    nothing in the container; :meth:`chosen` lists the ticked items.
    """

    # The ticked items changed (how many changed)
    membership_changed = QtCore.pyqtSignal(int)
    # A row's checkbox was clicked: row, now checked
    toggled = QtCore.pyqtSignal(int, bool)

    is_group = False

    def __init__(self, container, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.container = container
        self._names = list(container.keys())
        self._chosen: set[str] = set()

    @property
    def target_label(self) -> str:
        return ""

    def same_target(self, container, set_name=None, group=None) -> bool:
        return container is self.container and set_name is None and group is None

    def chosen(self) -> list:
        """The ticked items, in container order."""
        return [self.container[name] for name in self._names if name in self._chosen]

    def name_at(self, row: int) -> str:
        return self._names[row]

    def reload(self) -> None:
        """Re-read the items (e.g. after a delete); ticks of gone items go."""
        self.beginResetModel()
        self._names = list(self.container.keys())
        self._chosen &= set(self._names)
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._names)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        # Name, then Data for epochs (their arrays)
        return 0 if parent.isValid() else (2 if _is_epochs(self.container) else 1)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and \
                orientation == QtCore.Qt.Orientation.Horizontal:
            return ["Name", "Data"][section]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        name = self._names[index.row()]
        if index.column() == 1:
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return _arrays_text(self.container, name)
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return name
        if role == QtCore.Qt.ItemDataRole.CheckStateRole:
            return (QtCore.Qt.CheckState.Checked if name in self._chosen
                    else QtCore.Qt.CheckState.Unchecked)
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:  # not the Data column
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

    def set_checked(self, rows, checked: bool) -> int:
        names = [self._names[row] for row in rows if 0 <= row < len(self._names)]
        change = [n for n in names if (n in self._chosen) != checked]
        if not change:
            return 0
        if checked:
            self._chosen.update(change)
        else:
            self._chosen.difference_update(change)
        self.dataChanged.emit(self.index(0, 0),
                              self.index(self.rowCount() - 1, self.columnCount() - 1))
        self.membership_changed.emit(len(change))
        return len(change)

    def _rows(self, rows) -> list[int]:
        return list(range(len(self._names))) if rows is None else list(rows)

    def check_all(self, rows=None) -> int:
        return self.set_checked(self._rows(rows), True)

    def check_none(self, rows=None) -> int:
        return self.set_checked(self._rows(rows), False)

    def invert(self, rows=None) -> int:
        rows = self._rows(rows)
        ticked = [row for row in rows if self._names[row] in self._chosen]
        changed = self.set_checked(ticked, False)
        return changed + self.set_checked(
            [row for row in rows if row not in ticked], True
        )


__all__ = ["ChoiceModel", "MembershipModel", "set_item_group", "unassign_from_group"]
