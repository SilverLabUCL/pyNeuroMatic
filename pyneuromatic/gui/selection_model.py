"""Observable GUI selection state backed by an NMManager."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6 import QtCore

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import EXCLUDE_SET, NMEpoch
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS, NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


# Built-in Epoch Set choices (not stored sets): every epoch, or only the
# current one (Igor NeuroMatic's "All" and "This Wave")
ALL_EPOCHS = "All"
CURRENT_EPOCH = "Current Epoch"
BUILTIN_EPOCH_SETS = (ALL_EPOCHS, CURRENT_EPOCH)


class SelectionModel(QtCore.QObject):
    """Expose shared selection state and notify GUI subscribers on changes.

    The hierarchy tiers are read from and written to ``NMManager``. A data
    array and a data series can both be selected; ``source`` says which one
    tools analyse (``"data"`` or ``"dataseries"``). Choosing an item in one
    of them makes it the source.

    The remaining keys are GUI context: ``data_set`` is a set of the folder's
    (or subfolder's) data, while ``epoch_set``, ``group`` and
    ``group_operator`` scope the data series' epochs. Each is remembered per
    folder/subfolder (data set) or per data series (epoch scope).
    """

    selection_changed = QtCore.pyqtSignal(dict)

    SOURCES = ("data", "dataseries")
    _EPOCH_KEYS = ("epoch_set", "group", "group_operator")
    _GUI_KEYS = ("source", "data_set") + _EPOCH_KEYS
    _KEYS = HIERARCHY_SELECT_KEYS + _GUI_KEYS
    # Tiers whose selection makes that section the source
    _SOURCE_FOR_KEY = {
        "data": "data",
        "data_set": "data",
        "dataseries": "dataseries",
        "channel": "dataseries",
        "epoch": "dataseries",
        "epoch_set": "dataseries",
        "group": "dataseries",
    }

    def __init__(
        self,
        manager: NMManager,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(manager, NMManager):
            raise TypeError("manager must be an NMManager")
        self._manager = manager
        self._source: str | None = None
        self._data_set: str | None = None
        self._epoch_set: str | None = None
        self._group: int | None = None
        self._group_operator: str | None = None
        # Remembered GUI context, restored when returning to a scope
        self._data_set_memory: dict[tuple[str | None, ...], str] = {}
        self._epoch_memory: dict[
            tuple[str | None, ...], tuple[str | None, int | None, str | None]
        ] = {}
        self._normalize_source()
        if self._manager.select_values.get("dataseries") is not None:
            self._epoch_set = ALL_EPOCHS
        self._emitted_selection = self.selection.copy()

    @property
    def manager(self) -> NMManager:
        return self._manager

    @property
    def selection(self) -> dict[str, Any]:
        """Current selection snapshot, including the GUI context keys."""
        values = self._manager.select_values
        values.update(
            {
                "source": self._source,
                "data_set": self._data_set,
                "epoch_set": self._epoch_set,
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
        self._remember_context()
        previous_scopes = self._data_scope(), self._epoch_scope()

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
        for tier in HIERARCHY_SELECT_KEYS:
            value = object_updates.get(tier)
            if value is not None:
                self._manager.select_value_set(value)
        if manager_updates:
            self._manager.select_keys = {
                tier: manager_updates[tier]
                for tier in HIERARCHY_SELECT_KEYS
                if tier in manager_updates
            }

        self._restore_context(previous_scopes)
        for key in ("data_set",) + self._EPOCH_KEYS:
            if key in updates:
                setattr(self, f"_{key}", updates[key])
        if (
            ("epoch_set" in updates and updates["epoch_set"] is None)
            or ("group" in updates and updates["group"] is None)
        ) and "group_operator" not in updates:
            self._group_operator = None

        if "source" in updates:
            self._source = updates["source"]
        else:
            # The last section the user chose an item in becomes the source
            for key, source in self._SOURCE_FOR_KEY.items():
                if key in updates and updates[key] is not None:
                    self._source = source
        self._normalize_source()
        self._remember_context()
        self._emit_if_changed(previous)

    def _normalize_source(self) -> None:
        """Keep ``source`` pointing at a selected data array or data series."""
        values = self._manager.select_values
        available = [tier for tier in self.SOURCES if values.get(tier) is not None]
        if self._source not in available:
            # Prefer the data series, as the analysis default
            self._source = available[-1] if available else None

    def clear(self, *tiers: str) -> None:
        """Clear all selection, or the requested tiers and their descendants."""
        if not tiers:
            tiers = self._KEYS
        unknown = set(tiers).difference(self._KEYS)
        if unknown:
            raise KeyError(f"unknown selection tier(s): {sorted(unknown)}")

        previous = self.selection
        previous_scopes = self._data_scope(), self._epoch_scope()
        if "data_set" in tiers:
            self._data_set = None
            self._data_set_memory.pop(previous_scopes[0], None)
        cleared_epoch_keys = set(tiers).intersection(self._EPOCH_KEYS)
        if "epoch_set" in cleared_epoch_keys:
            self._epoch_set = None
        if "group" in cleared_epoch_keys:
            self._group = None
        if cleared_epoch_keys:
            self._group_operator = None
            self._epoch_memory.pop(previous_scopes[1], None)

        if "folder" in tiers:
            self._clear_all_folder_selections()
            self._data_set_memory.clear()
            self._epoch_memory.clear()
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

        self._restore_context(previous_scopes)
        self._normalize_source()
        self._emit_if_changed(previous)

    # --- remembered context -------------------------------------------------

    def _data_scope(self) -> tuple[str | None, ...]:
        keys = self._manager.select_keys
        return keys.get("folder"), keys.get("toolfolder")

    def _epoch_scope(self) -> tuple[str | None, ...]:
        keys = self._manager.select_keys
        return keys.get("folder"), keys.get("toolfolder"), keys.get("dataseries")

    def _remember_context(self) -> None:
        if self._data_set is not None:
            self._data_set_memory[self._data_scope()] = self._data_set
        epoch_state = (self._epoch_set, self._group, self._group_operator)
        # A data series remembers an empty Epoch Set too (not back to All)
        if any(value is not None for value in epoch_state) or self._epoch_scope()[2] is not None:
            self._epoch_memory[self._epoch_scope()] = epoch_state

    def _restore_context(self, previous_scopes: tuple[tuple, tuple]) -> None:
        """Restore remembered context for any scope that changed."""
        data_scope, epoch_scope = self._data_scope(), self._epoch_scope()
        if data_scope != previous_scopes[0]:
            self._data_set = self._data_set_memory.get(data_scope)
        if epoch_scope != previous_scopes[1]:
            # A data series starts with All epochs
            default_set = ALL_EPOCHS if epoch_scope[2] is not None else None
            self._epoch_set, self._group, self._group_operator = (
                self._epoch_memory.get(epoch_scope, (default_set, None, None))
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

        def name(tier):
            value = previous.get(tier)
            return value.name if isinstance(value, NMObject) else value

        previous_scopes = (
            (name("folder"), name("toolfolder")),
            (name("folder"), name("toolfolder"), name("dataseries")),
        )
        self._restore_context(previous_scopes)
        self._normalize_source()
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
            if tier in ("data_set", "epoch_set"):
                if not isinstance(value, str):
                    raise TypeError(f"{tier} must be a string or None")
            elif tier == "source":
                if value not in self.SOURCES:
                    raise ValueError("source must be 'data', 'dataseries', or None")
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
    """Names from *all_names* allowed by a set and/or group selection.

    ``None`` means that tier is not selected. When both are selected,
    *operator* (``"AND"`` or ``"OR"``) combines them.
    """
    if set_names is not None and group_names is not None:
        if operator not in ("AND", "OR"):
            raise RuntimeError("Choose AND or OR to combine the selected Epoch Set and Group")
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


def is_stored_epoch_set(name: str | None) -> bool:
    """True for a real epoch set (Set1, SetX...), not All or Current Epoch."""
    return name is not None and name not in BUILTIN_EPOCH_SETS


def scoped_epoch_names(
    dataseries: NMDataSeries,
    selection: Mapping[str, Any],
    exclude: bool = True,
) -> list[str] | None:
    """Epoch names of *dataseries* in the selected Epoch Set/Group, in order.

    "All" means every epoch. Returns None for "Current Epoch" (or no Epoch
    Set and no Group), so callers choose their default, usually the current
    epoch. With *exclude*, epochs in the exclusion set (SetX) are left out,
    unless SetX itself is the Epoch Set.
    """
    selected_set = selection.get("epoch_set")
    selected_group = selection.get("group")
    if selected_set == CURRENT_EPOCH or (selected_set is None and selected_group is None):
        return None
    set_names = None
    group_names = None
    if is_stored_epoch_set(selected_set):
        set_names = dataseries.epochs.sets.get_items(selected_set, get_keys=True) or []
    if selected_group is not None:
        group_names = dataseries.epochs.groups.get_items(selected_group)
    names = combine_scope_names(
        list(dataseries.epochs.keys()),
        set_names,
        group_names,
        selection.get("group_operator"),
    )
    if exclude and (selected_set or "").lower() != EXCLUDE_SET.lower():
        excluded = dataseries.epochs.excluded_names()
        names = [name for name in names if name not in excluded]
    return names


def source_object(selection: Mapping[str, Any], source: str) -> Any:
    """The selected data array or data series if it is the active *source*."""
    return selection.get(source) if selection.get("source") == source else None


def selection_summary(selection: Mapping[str, Any]) -> list[str]:
    """Names describing the selection: folder, subfolder, then the source's.

    For a data source: the data array and Data Set. For a data series: the
    prefix, channel, epoch and epoch scope.
    """

    def name(key: str) -> str | None:
        value = selection.get(key)
        return value.name if isinstance(value, NMObject) else value

    parts = [name("folder"), name("toolfolder")]
    source = selection.get("source")
    if source == "data":
        parts.append(name("data"))
        if selection.get("data_set") is not None:
            parts.append(f"Data Set: {selection['data_set']}")
    elif source == "dataseries":
        parts += [name("dataseries"), name("channel"), _epochs_summary(selection, name)]
    return [part for part in parts if part is not None]


def _epochs_summary(selection: Mapping[str, Any], name) -> str | None:
    """The epochs part of a summary: "Epoch Set1", "All Epochs", "E1"...

    A stored Epoch Set reads "Epoch Set1" (with any Group, e.g. "Epoch Set1
    AND Group 0"); All reads "All Epochs"; Current Epoch (or no Epoch Set or
    Group) reads the current epoch's name.
    """
    epoch_set = selection.get("epoch_set")
    group = selection.get("group")
    group_text = f"Group {group}" if group is not None else None
    if is_stored_epoch_set(epoch_set):
        set_text = f"Epoch {epoch_set}"
        if group_text is None:
            return set_text
        operator = selection.get("group_operator") or "+"
        return f"{set_text} {operator} {group_text}"
    if epoch_set == ALL_EPOCHS:
        return group_text or "All Epochs"
    if epoch_set is None and group_text is not None:
        return group_text
    return name("epoch")


def epoch_scope_label(selection: Mapping[str, Any]) -> str:
    """Short description of the epoch scope, e.g. "Epoch Set: Set1 AND Group 0"."""
    parts = []
    epoch_set = selection.get("epoch_set")
    if is_stored_epoch_set(epoch_set):
        parts.append(f"Epoch Set: {epoch_set}")
    elif epoch_set == CURRENT_EPOCH:
        parts.append(CURRENT_EPOCH)
    if selection.get("group") is not None:
        parts.append(f"Group {selection['group']}")
    operator = selection.get("group_operator") or "+"
    return f" {operator} ".join(parts)


__all__ = [
    "ALL_EPOCHS",
    "BUILTIN_EPOCH_SETS",
    "CURRENT_EPOCH",
    "SelectionModel",
    "is_stored_epoch_set",
    "combine_scope_names",
    "epoch_scope_label",
    "scoped_epoch_names",
    "selection_summary",
    "source_object",
]
