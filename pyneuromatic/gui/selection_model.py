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
from pyneuromatic.core.nm_utilities import combine_scope_names
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_group import group_name


# Built-in Epoch Set choice (not a stored set): every epoch (Igor
# NeuroMatic's "All"). The blank choice (None) is the current epoch, as
# blank Data Set is the current array (#444).
ALL_EPOCHS = "All"
BUILTIN_EPOCH_SETS = (ALL_EPOCHS,)
# Once a choice for the current epoch; now read as blank. Not allowed as a
# set name, so old selections can't mean a set.
CURRENT_EPOCH = "Current Epoch"
RESERVED_EPOCH_SET_NAMES = (ALL_EPOCHS, CURRENT_EPOCH)

# Run once per set or per group (Igor NeuroMatic's "All Sets" / "All Groups").
# Elsewhere (Table, stepping, counts) they mean the union of the sets
# (SetX left out) or of the groups.
ALL_SETS = "All Sets"
ALL_GROUPS = "All Groups"
# Run once per channel. The current channel stays selected (Plot and Table
# show it); only runs go over every channel.
ALL_CHANNELS = "All Channels"


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
    _GUI_KEYS = ("source", "data_set", "all_channels") + _EPOCH_KEYS
    _KEYS = HIERARCHY_SELECT_KEYS + _GUI_KEYS
    # Tiers whose selection makes that section the source
    _SOURCE_FOR_KEY = {
        "data": "data",
        "data_set": "data",
        "dataseries": "dataseries",
        "channel": "dataseries",
        "all_channels": "dataseries",
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
        self._all_channels = False  # runs go over every channel
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
                "all_channels": True if self._all_channels else None,  # None: off
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
        if updates.get("epoch_set") == CURRENT_EPOCH:
            # No longer a choice: blank, and (as it ignored the Group) no Group
            updates = {"group": None, **updates, "epoch_set": None}

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
        # One select_keys call (one command in the history), objects first:
        # each brings the path above it
        select: dict[str, Any] = {}
        for tier in HIERARCHY_SELECT_KEYS:
            value = object_updates.get(tier)
            if value is not None:
                select.update(self._manager._select_value_keys(value))
        select.update(manager_updates)
        if select:
            self._manager.select_keys = {
                tier: select[tier] for tier in HIERARCHY_SELECT_KEYS if tier in select
            }

        self._restore_context(previous_scopes)
        for key in ("data_set",) + self._EPOCH_KEYS:
            if key in updates:
                setattr(self, f"_{key}", updates[key])
        self._either_set_or_group(updates)
        if "all_channels" in updates:
            self._all_channels = bool(updates["all_channels"])  # None or False: off
        elif updates.get("channel") is not None:
            self._all_channels = False  # one channel chosen
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

    def _either_set_or_group(self, updates: Mapping[str, Any]) -> None:
        """Epoch Set and Group are either/or unless AND/OR combines them
        (#444): choosing one blanks the other; blanking AND/OR keeps the Set."""
        if self._group_operator is not None:
            return
        if updates.get("group") is not None and "epoch_set" not in updates:
            self._epoch_set = None
        elif updates.get("epoch_set") is not None and "group" not in updates:
            self._group = None
        elif ("group_operator" in updates and "epoch_set" not in updates
                and "group" not in updates
                and self._epoch_set is not None and self._group is not None):
            self._group = None

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
        if {"all_channels", "channel", "dataseries"} & set(tiers):
            self._all_channels = False
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
            self._all_channels = False  # another data series: its own choice
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
                if value != ALL_GROUPS and (isinstance(value, bool) or not isinstance(value, int)):
                    raise TypeError("group must be an integer, ALL_GROUPS or None")
            elif tier == "all_channels":
                if not isinstance(value, bool):  # (None, like False, turns it off)
                    raise TypeError("all_channels must be True or False")
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


def is_stored_epoch_set(name: str | None) -> bool:
    """True for a real epoch set (Set1, SetX...), not All or All Sets."""
    return name is not None and name not in RESERVED_EPOCH_SET_NAMES and name != ALL_SETS


def all_set_names(dataseries: NMDataSeries) -> list[str]:
    """The sets "All Sets" runs over: every stored set except SetX."""
    return [name for name in dataseries.epochs.sets.keys()
            if name.lower() != EXCLUDE_SET.lower()]


def scope_runs(dataseries: NMDataSeries, selection: Mapping[str, Any]) -> list[tuple[str, dict]] | None:
    """For "All Sets" / "All Groups": one (label, selection) per set or group,
    each selection naming that one set or group; None otherwise.

    Raises:
        RuntimeError: For All Sets with All Groups (not supported yet).
    """
    all_sets = selection.get("epoch_set") == ALL_SETS
    all_groups = selection.get("group") == ALL_GROUPS
    if all_sets and all_groups:
        raise RuntimeError("All Sets with All Groups isn't supported yet: "
                           "choose one set or one group")
    if all_sets:
        return [(name, {**selection, "epoch_set": name}) for name in all_set_names(dataseries)]
    if all_groups:
        return [(group_name(n), {**selection, "group": n})
                for n in dataseries.epochs.groups.group_numbers]
    return None


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
    if selected_set is None and selected_group is None:
        return None
    set_names = None
    group_names = None
    sets, groups = dataseries.epochs.sets, dataseries.epochs.groups
    if selected_set == ALL_SETS:  # the union of the sets (SetX left out)
        set_names = list({name: None for set_name in all_set_names(dataseries)
                          for name in (sets.get_items(set_name, get_keys=True) or [])})
    elif is_stored_epoch_set(selected_set):
        set_names = sets.get_items(selected_set, get_keys=True) or []
    if selected_group == ALL_GROUPS:  # the union of the groups
        group_names = [name for n in groups.group_numbers for name in groups.get_items(n)]
    elif selected_group is not None:
        group_names = groups.get_items(selected_group)
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
        # A Data Set is what a run uses (the array is only the one shown),
        # as an Epoch Set is for a data series
        data_set = selection.get("data_set")
        parts.append(name("data") if data_set is None else f"Data {data_set}")
    elif source == "dataseries":
        channel = ALL_CHANNELS if selection.get("all_channels") else name("channel")
        parts += [name("dataseries"), channel, _epochs_summary(selection, name)]
    return [part for part in parts if part is not None]


def _epochs_summary(selection: Mapping[str, Any], name) -> str | None:
    """The epochs part of a summary: "Epoch Set1", "All Epochs", "E1"...

    A stored Epoch Set reads "Epoch Set1" (with any Group, e.g. "Epoch Set1
    AND Group 0"); All reads "All Epochs"; Current Epoch (or no Epoch Set or
    Group) reads the current epoch's name.
    """
    epoch_set = selection.get("epoch_set")
    group = selection.get("group")
    group_text = (ALL_GROUPS if group == ALL_GROUPS
                  else group_name(group) if group is not None else None)
    if is_stored_epoch_set(epoch_set) or epoch_set == ALL_SETS:
        set_text = ALL_SETS if epoch_set == ALL_SETS else f"Epoch {epoch_set}"
        if group_text is None:
            return set_text
        operator = selection.get("group_operator")
        # "?": AND or OR not chosen yet (a "+" looked like a choice)
        return f"{set_text} {operator or '?'} {group_text}"
    if epoch_set == ALL_EPOCHS:
        return group_text or "All Epochs"
    if epoch_set is None and group_text is not None:
        return group_text
    return name("epoch")


def needs_group_operator(selection: Mapping[str, Any]) -> bool:
    """True when an Epoch Set and a Group are both chosen but not AND or OR."""
    epoch_set = selection.get("epoch_set")
    return ((is_stored_epoch_set(epoch_set) or epoch_set == ALL_SETS)
            and selection.get("group") is not None
            and selection.get("group_operator") is None)


def epoch_scope_label(selection: Mapping[str, Any]) -> str:
    """Short description of the epoch scope, e.g. "Epoch Set: Set1 AND Group 0"."""
    parts = []
    epoch_set = selection.get("epoch_set")
    if is_stored_epoch_set(epoch_set) or epoch_set == ALL_SETS:
        parts.append(f"Epoch Set: {epoch_set}")
    group = selection.get("group")
    if group == ALL_GROUPS:
        parts.append(ALL_GROUPS)
    elif group is not None:
        parts.append(group_name(group))
    operator = selection.get("group_operator") or "?"  # "?": AND or OR not chosen
    return f" {operator} ".join(parts)


__all__ = [
    "ALL_CHANNELS",
    "ALL_EPOCHS",
    "ALL_GROUPS",
    "ALL_SETS",
    "all_set_names",
    "scope_runs",
    "BUILTIN_EPOCH_SETS",
    "CURRENT_EPOCH",
    "RESERVED_EPOCH_SET_NAMES",
    "SelectionModel",
    "is_stored_epoch_set",
    "combine_scope_names",
    "epoch_scope_label",
    "needs_group_operator",
    "scoped_epoch_names",
    "selection_summary",
    "source_object",
]
