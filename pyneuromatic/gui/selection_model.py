"""Observable GUI selection state backed by an NMManager."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6 import QtCore

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS, NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


class SelectionModel(QtCore.QObject):
    """Expose shared selection state and notify GUI subscribers on changes.

    The hierarchy tiers are read from and written to ``NMManager``. The
    Set and group selections are GUI context; the manager's epoch groups and
    container sets are queried by the selector widget. The operator combines
    those two selections when a tool consumes the snapshot.
    """

    selection_changed = QtCore.pyqtSignal(dict)

    _GUI_KEYS = ("set", "group", "group_operator")
    _KEYS = HIERARCHY_SELECT_KEYS + _GUI_KEYS

    def __init__(
        self,
        manager: NMManager,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(manager, NMManager):
            raise TypeError("manager must be an NMManager")
        self._manager = manager
        initial = manager.select_values
        if initial.get("data") is not None and initial.get("dataseries") is not None:
            manager.select_keys = {"data": None}
        self._set: str | None = None
        self._group: int | None = None
        self._group_operator: str | None = None
        self._selector_memory: dict[
            tuple[str | None, ...], tuple[str | None, int | None, str | None]
        ] = {}
        self._emitted_selection = self.selection.copy()

    @property
    def manager(self) -> NMManager:
        return self._manager

    @property
    def selection(self) -> dict[str, Any]:
        """Current selection snapshot, including GUI group/set context."""
        values = self._manager.select_values
        values.update(
            {
                "set": self._set,
                "group": self._group,
                "group_operator": self._group_operator,
            }
        )
        return values

    @staticmethod
    def tier_for_object(obj: NMObject) -> str:
        """Return the manager selection tier represented by *obj*."""
        for tier, object_type in (
            ("folder", NMFolder),
            ("toolfolder", NMToolFolder),
            ("data", NMData),
            ("dataseries", NMDataSeries),
            ("channel", NMChannel),
            ("epoch", NMEpoch),
        ):
            if isinstance(obj, object_type):
                return tier
        raise TypeError(f"unsupported selection object type: {type(obj).__name__}")

    def update(
        self,
        changes: Mapping[str, Any] | None = None,
        **updates: Any,
    ) -> None:
        """Partially update selection from names or ``NMObject`` references."""
        if changes is not None:
            if not isinstance(changes, Mapping):
                raise TypeError("changes must be a mapping")
            updates = {**changes, **updates}
        if not updates:
            return

        unknown = set(updates).difference(self._KEYS)
        if unknown:
            raise KeyError(f"unknown selection tier(s): {sorted(unknown)}")
        self._validate(updates)
        previous = self.selection
        previous_scope = self._scope_key()
        self._remember_selectors(previous_scope)

        object_updates = {
            tier: value
            for tier, value in updates.items()
            if tier in HIERARCHY_SELECT_KEYS and isinstance(value, NMObject)
        }
        manager_updates = {
            tier: value
            for tier, value in updates.items()
            if tier in HIERARCHY_SELECT_KEYS and not isinstance(value, NMObject)
        }
        requested_data = object_updates.get("data", manager_updates.get("data"))
        requested_dataseries = object_updates.get(
            "dataseries", manager_updates.get("dataseries")
        )
        if requested_data is not None:
            manager_updates["dataseries"] = None
        elif requested_dataseries is not None:
            manager_updates["data"] = None

        for tier in HIERARCHY_SELECT_KEYS:
            value = object_updates.get(tier)
            if value is not None:
                self._manager.select_value_set(value)

        if manager_updates:
            ordered_updates = {
                tier: manager_updates[tier]
                for tier in HIERARCHY_SELECT_KEYS
                if tier in manager_updates
            }
            self._manager.select_keys = ordered_updates

        self._normalize_data_mode(
            "data" if requested_data is not None else
            "dataseries" if requested_dataseries is not None else None
        )

        current_scope = self._scope_key()
        if current_scope != previous_scope:
            self._restore_selectors(current_scope)

        for key, attribute in (
            ("set", "_set"),
            ("group", "_group"),
            ("group_operator", "_group_operator"),
        ):
            if key in updates:
                setattr(self, attribute, updates[key])

        if (
            ("set" in updates and updates["set"] is None)
            or ("group" in updates and updates["group"] is None)
        ) and "group_operator" not in updates:
            self._group_operator = None

        self._remember_selectors(current_scope)

        self._emit_if_changed(previous)

    def _normalize_data_mode(self, preferred: str | None = None) -> None:
        current = self._manager.select_values
        if current["data"] is None or current["dataseries"] is None:
            return
        inactive_tier = "dataseries" if preferred == "data" else "data"
        self._manager.select_keys = {inactive_tier: None}

    def clear(self, *tiers: str) -> None:
        """Clear all selection, or the requested tiers and their descendants."""
        if not tiers:
            tiers = self._KEYS
        unknown = set(tiers).difference(self._KEYS)
        if unknown:
            raise KeyError(f"unknown selection tier(s): {sorted(unknown)}")

        previous = self.selection
        previous_scope = self._scope_key()
        clear_selector_keys = set(tiers).intersection(self._GUI_KEYS)
        if "set" in clear_selector_keys:
            self._set = None
        if "group" in clear_selector_keys:
            self._group = None
        if "group_operator" in clear_selector_keys or clear_selector_keys.intersection(
            ("set", "group")
        ):
            self._group_operator = None
        if clear_selector_keys:
            self._selector_memory.pop(previous_scope, None)

        if "folder" in tiers:
            self._clear_all_folder_selections()
            self._selector_memory.clear()
        else:
            descendants = set()
            for tier in tiers:
                descendants.update(self._descendants_for(tier))
            clear_keys = descendants.union(
                tier for tier in tiers if tier in HIERARCHY_SELECT_KEYS
            )
            if clear_keys:
                self._manager.select_keys = {
                    tier: None for tier in HIERARCHY_SELECT_KEYS if tier in clear_keys
                }

        current_scope = self._scope_key()
        if not clear_selector_keys and current_scope != previous_scope:
            self._restore_selectors(current_scope)

        self._emit_if_changed(previous)

    def _scope_key(self) -> tuple[str | None, ...]:
        keys = self._manager.select_keys
        return tuple(
            keys.get(tier)
            for tier in ("folder", "toolfolder", "data", "dataseries")
        )

    def _remember_selectors(self, scope: tuple[str | None, ...]) -> None:
        state = (self._set, self._group, self._group_operator)
        if any(value is not None for value in state):
            self._selector_memory[scope] = state

    def _restore_selectors(self, scope: tuple[str | None, ...]) -> None:
        self._set, self._group, self._group_operator = self._selector_memory.get(
            scope, (None, None, None)
        )

    def _clear_all_folder_selections(self) -> None:
        for folder in self._manager.folders.values():
            self._manager.select_keys = {
                "folder": folder.name,
                "toolfolder": None,
                "data": None,
                "dataseries": None,
                "channel": None,
                "epoch": None,
            }
        self._manager.select_keys = {"folder": None}

    def refresh(self) -> None:
        """Notify subscribers if selection was changed directly on the manager."""
        previous = self._last_emitted_selection()
        previous_scope = tuple(
            value.name if isinstance(value, NMObject) else value
            for value in (
                previous.get(tier)
                for tier in ("folder", "toolfolder", "data", "dataseries")
            )
        )
        self._remember_selectors(previous_scope)
        current = self.selection
        current_scope = self._scope_key()
        if current_scope != previous_scope:
            self._restore_selectors(current_scope)
        self._emit_if_changed(previous)

    @staticmethod
    def _descendants_for(tier: str) -> tuple[str, ...]:
        descendants = {
            "folder": ("epoch", "channel", "dataseries", "data", "toolfolder"),
            "toolfolder": ("epoch", "channel", "dataseries", "data"),
            "dataseries": ("epoch", "channel"),
        }
        return descendants.get(tier, ())

    def _validate(self, updates: Mapping[str, Any]) -> None:
        for tier, value in updates.items():
            if value is None:
                continue
            if tier == "set":
                if not isinstance(value, str):
                    raise TypeError("set must be a string or None")
            elif tier == "group":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("group must be an integer or None")
            elif tier == "group_operator":
                if value not in ("AND", "OR"):
                    raise ValueError("group_operator must be 'AND', 'OR', or None")
            elif not isinstance(value, (str, NMObject)):
                raise TypeError(f"{tier} must be an object, string, or None")
            elif isinstance(value, NMObject):
                if value._manager is not self._manager:
                    raise ValueError("selection object belongs to a different manager")
                if self.tier_for_object(value) != tier:
                    raise TypeError(f"{tier} selection must be a {tier} object")

    def _last_emitted_selection(self) -> dict[str, Any]:
        return self._emitted_selection

    def _emit_if_changed(self, previous: Mapping[str, Any]) -> None:
        current = self.selection
        if all(self._same_value(previous.get(key), current.get(key)) for key in self._KEYS):
            return
        self._emitted_selection = current.copy()
        self.selection_changed.emit(current)

    @staticmethod
    def _same_value(first: Any, second: Any) -> bool:
        if isinstance(first, NMObject) or isinstance(second, NMObject):
            return first is second
        return first == second


def combine_scope_names(
    all_names: list[str],
    set_names: list[str] | None,
    group_names: list[str] | None,
    operator: str | None,
) -> list[str]:
    """Names from *all_names* allowed by a Set and/or Group selection.

    ``None`` means that tier is not selected. When both are selected,
    *operator* (``"AND"`` or ``"OR"``) combines them.
    """
    if set_names is not None and group_names is not None:
        if operator not in ("AND", "OR"):
            raise RuntimeError("Choose AND or OR to combine the selected Set and Group")
        allowed = (
            set(set_names).intersection(group_names)
            if operator == "AND"
            else set(set_names).union(group_names)
        )
    elif set_names is not None:
        allowed = set(set_names)
    elif group_names is not None:
        allowed = set(group_names)
    else:
        return all_names
    return [name for name in all_names if name in allowed]


def scoped_epoch_names(
    dataseries: NMDataSeries, selection: Mapping[str, Any]
) -> list[str] | None:
    """Epoch names of *dataseries* in the selected Set/Group, in epoch order.

    Returns None when neither a Set nor a Group is selected, so callers can
    choose their own default (the current epoch, or all epochs).
    """
    selected_set = selection.get("set")
    selected_group = selection.get("group")
    if selected_set is None and selected_group is None:
        return None
    set_names = None
    group_names = None
    if selected_set is not None:
        set_names = dataseries.epochs.sets.get_items(selected_set, get_keys=True) or []
    if selected_group is not None:
        group_names = dataseries.epochs.groups.get_items(selected_group)
    return combine_scope_names(
        list(dataseries.epochs.keys()),
        set_names,
        group_names,
        selection.get("group_operator"),
    )


__all__ = ["SelectionModel", "combine_scope_names", "scoped_epoch_names"]